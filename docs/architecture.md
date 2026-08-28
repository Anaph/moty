# Moty architecture

Pure-C LLM inference engine. Zero runtime dependencies: no BLAS, no Python,
no GPU required. This document describes the code layout (module folders),
the layering contract between them, and how a token flows through the
system.

```
c/
├── engines/    one .c per model family + moty.c (multi-model dispatcher)
├── nn/         shared compute kernels (paste-in headers, L0..L1)
├── hw/         hardware abstraction: SIMD/GPU backends behind one contract
├── runtime/    engine scaffolding (hooks) + MoE expert cache
├── io/         model file formats: safetensors, GGUF, io_uring, tiering
├── tok/        tokenizers (BPE byte-level, SentencePiece)
└── util/       json, compat shims, grammar, profiling, misc
```

## The three layering rules

1. **Engines are thin.** An engine defines the model's *shape*: config
   fields, layer fields, which shared kernel runs where. No SIMD, no file
   parsing, no generation loop — those live below.
2. **Compute is shared, pasted in.** Kernels are `static` headers included
   by `nn/nn.h`. Each engine compiles its own instance; the contract is the
   function signature, documented at the top of every header.
3. **Hardware is abstracted.** Engines and kernels never write intrinsics.
   They call `dot_i8i8`, `dot_i4i8p`, `dot_i4g8p`, `qrow_i8`, `px_permute`,
   `px_sum` (contract in `hw/hw.h`); `hw/hw_avx512.h`, `hw_avx2.h`,
   `hw_neon.h`, `hw_sve2.h`, `hw_vsx.h`, `hw_scalar.h` implement them per
   target, selected at compile time by `-march`. `hw/hw_quant.h` holds the
   portable fallbacks for the newer grouped/permuted kernels.

### Kernel contract highlights

- `dot_i8i8(w, x, I)` — int8×int8 dot. **x must be in [-127, 127]** (the
  AVX512-VNNI sign trick breaks on -128). `qrow_i8` output satisfies this.
- `dot_i4i8p / dot_i4g8p` — int4-weight variants; the `p` suffix expects
  the activation vector pre-permuted by `px_permute` (nibble-offset trick:
  Σ(n−8)x = Σn·x − 8Σx, with `px_sum` supplying Σx).
- `dot_i4g8p` — grouped int4 (one scale per `gs` elements, `gs` a power of
  two ≥16; VNNI path tuned for gs=32).

## Module reference

| Folder | Key files | Role |
|---|---|---|
| `engines/` | `moty.c`, `lfm2.c`, `qwenmoe.c`, `qwen.c`, `gemma.c`, `olmoe.c`, `glm.c` | `moty` dispatches by GGUF arch string to per-family engines; each can also build standalone |
| `nn/` | `nn.h` (umbrella), `nn_mat.h` (`Mat`, `MotyCommon`, `mat_apply`), `nn_matmul.h`, `nn_attn_kernels.h`, `attn.h` (`MotyAttnView`), `conv.h`, `ffn.h`, `moe.h` (views + variants), `nn_deltanet.h`, `nn_rope.h`, `nn_norm.h`, `nn_sample.h`, `nn_quant.h`, `nn_alloc.h`, `mla.h` | All shared math — **one compiled copy** in `libmoty-nn` (M3). Layers take view structs (weights + config + `MotyCommon` storage); gate/shared-expert variants are named functions, not macros |
| `hw/` | `hw.h` (single include point), `hw_avx512.h`, `hw_avx2.h`, `hw_neon.h`, `hw_sve2.h`, `hw_vsx.h`, `hw_scalar.h`, `hw_quant.h` (portable fallbacks), `hw_backend.h` (runtime dispatch), stubs for CUDA/Metal/OpenCL/Vulkan/WASM | One compile-time ladder, no runtime CPUID dispatch |
| `runtime/` | `runtime.h` (hook contract + umbrella), `rt_model_load.h`, `rt_kv_cache.h`, `rt_gen_loop.h`, `rt_env_cfg.h`, `moe.h` (expert cache, LRU/pin), `decode_batch.h` | The engine scaffolding: load a model from GGUF/safetensors, budget RAM, run prefill/decode, env config. Engines implement ~10 hooks |
| `io/` | `st.h` (safetensors, multi-shard), `gguf.h` (GGUF v2/v3 incl. K-quants), `uring.h` (async expert loads), `tier.h` (LRU scoring) | Everything that reads bytes from disk |
| `tok/` | `tok.h`, `tok_unicode.h` | Byte-level BPE and SentencePiece (metaspace + byte-fallback) |
| `util/` | `json.h`, `compat.h`, `grammar.h` (GBNF), `schema_gbnf.h`, `prof.h`, `stw.h` (sliding-window KV), `simd.h` (compat wrapper → `hw/hw.h`) | Support code with no engine knowledge |

## Memory: the per-Model Scratch arena (P5)

All kernel scratch lives in `Model.scr` / `Model.bscr` (fields injected by
`MODEL_COMMON_FIELDS` in `nn/nn_mat.h`; API in `nn/nn_alloc.h`):

```
scr_reset(&m->scr);                // entering a kernel
scr_reserve(&m->scr, total_bytes); // the ONLY realloc point
float *a = scr_take(&m->scr, n);   // 64B-aligned, never moves the base
```

Contract: **reserve before take** — after a reserve, no take reallocates, so
every pointer handed out stays valid. Calling context is serial (never
inside an OpenMP region). This replaced ~15 function-local `grow()` statics
and makes two Models safe in one process. `tests/scratch_tests.c` locks the
contract down.

## Profiling

`util/prof.h` gates all instrumentation behind `MOTY_PROF` at compile time
(zero cost in production builds). Build an engine with
`-DMOTY_PROF` and it prints windowed per-phase ms/token (`[prof] win=...`)
every ~2 s of generation. Use runs of ≥256 tokens: short runs are dominated
by expert-cache warmup.

## Token flow (decode, LFM2-style)

```
moty: GGUF arch → engine_main
 └─ runtime gen loop (rt_gen_loop.h): pick_tok (nn_sample.h) → step()
     └─ engine step(): for each layer
         ├── conv layers:  nn/conv.h      (fused VNNI region, causal state)
         ├── attn layers:  nn/attn.h      (fused q/k/v VNNI, QK-norm, RoPE,
         │                                  KV store f32/int8, scores+accum)
         └── MoE:          nn/moe.h       (sigmoid/topk variants, expert cache)
                           ├── router gate (sigmoid+bias or softmax top-k)
                           ├── 2 fork/join regions: all gate/up rows, then
                           │   all down rows (x quantized once)
                           └── expert bytes from runtime/moe.h cache
```

## Testing

`make test` — GoogleTest via a separate CMake build (test logic is plain C;
C++ confined to thin glue). 118 tests: kernel exactness (grouped/permuted
int4 vs dequantized references, bit-exact invariants), attention
(batch-vs-incremental, independent serial reference), conv causality,
scratch-arena contract, sampler branches, MoE cache, plus engine-level
greedy-vs-oracle suites. `make test-native` reruns everything with
`-march=native` so reference tests exercise your CPU's actual kernels.
`make check` = clean + portable build + full suite.
