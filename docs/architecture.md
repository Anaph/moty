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
- `moty_hw_q4r4_gemm` + `moty_hw_quant_g32` — the Q4R4 int4 format
  (below); `moty_hw_rmsnorm`, `_silu_mul`, `_softmax`,
  `_axpy`, `_add`, `_shortconv_step` — row ops (`hw/hw_ops.h`). Every one
  has a `*_ref` scalar reference that is also the portable kernel.

### Parallel loops (`nn/par.h`)

The dense kernels parallelise through one call, `moty_par_for(n, chunk,
fn, ctx)`: `fn(ctx, i0, i1, tid)` covers `[i0, i1)` on thread `tid`;
`chunk == 0` gives each thread one contiguous range split like
`schedule(static)` (`moty_par_range`), `chunk > 0` hands out chunks in
order like `schedule(dynamic, chunk)` (the Q4R4/Q8R4 block loops use 8
blocks). The end of a call is the barrier; a call from inside a region
runs serially on the calling thread. Two backends, chosen at build time:
OpenMP (default: thin wrappers over `#pragma omp`) and `MOTY_THREADPOOL`
(`make THREADPOOL=1`, CMake `-DMOTY_THREADPOOL=ON`): persistent pthread
workers pinned one per CPU of the process's mask, a generation counter
that publishes each region, workers polling it (`MOTY_POOL_SPIN`) and then
sleeping on a per-worker condition variable, the caller doing its own
share and waiting on a done counter. `moty_par_set_threads` is what
`THREADS` and `THREADS_DECODE` call; workers outside a smaller team go to
sleep instead of polling, which leaves their core to other processes.
Per-thread scratch (attention score rows) is indexed by `tid` and sized
by `moty_par_threads()`. Sites not converted (MoE, DeltaNet, the
glm/olmoe/qwenmoe/gemma engines) keep their pragmas and run serially in
a pool build.

## Module reference

| Folder | Key files | Role |
|---|---|---|
| `engines/` | `moty.c`, `lfm2.c`, `qwenmoe.c`, `qwen.c`, `gemma.c`, `olmoe.c` | `moty` dispatches by GGUF arch string to per-family engines; each can also build standalone |

> `engines/glm.c` (GLM-5.2, `glm_moe_dsa`) is **compile-only and not
> documented as supported**: it has never been validated against a real
> model in this tree. It still builds and registers, but no claims are
> made about it. It is excluded from the user-facing surface on purpose.
| `nn/` | `nn.h` (umbrella), `nn_mat.h` (`Mat`, `MotyCommon`, `mat_apply`), `nn_matmul.h`, `nn_attn_kernels.h`, `attn.h` (`MotyAttnView`), `conv.h`, `ffn.h`, `moe.h` (views + variants), `nn_deltanet.h`, `nn_rope.h`, `nn_norm.h`, `nn_sample.h`, `nn_quant.h`, `nn_alloc.h`, `mla.h` | All shared math — **one compiled copy** in `libmoty-nn` (M3). Layers take view structs (weights + config + `MotyCommon` storage); gate/shared-expert variants are named functions, not macros |
| `hw/` | `hw.h` (single include point), `hw_avx512.h`, `hw_avx2.h`, `hw_neon.h`, `hw_sve2.h`, `hw_vsx.h`, `hw_scalar.h`, `hw_quant.h` (portable fallbacks), `hw_q4r4.h` (Q4R4 int4 GEMV/GEMM), `hw_ops.h` (row ops), `hw_backend.h` (runtime dispatch), stubs for CUDA/Metal/OpenCL/Vulkan/WASM | One compile-time ladder, no runtime CPUID dispatch |
| `runtime/` | `runtime.h` (hook contract + umbrella), `rt_model_load.h`, `rt_kv_cache.h`, `rt_gen_loop.h`, `rt_env_cfg.h`, `moe.h` (expert cache, LRU/pin), `decode_batch.h` | The engine scaffolding: load a model from GGUF/safetensors, budget RAM, run prefill/decode, env config. Engines implement ~10 hooks |
| `io/` | `st.h` (safetensors, multi-shard), `gguf.h` (GGUF v2/v3 incl. K-quants), `uring.h` (async expert loads), `tier.h` (LRU scoring) | Everything that reads bytes from disk |
| `tok/` | `tok.h`, `tok_unicode.h` | Byte-level BPE and SentencePiece (metaspace + byte-fallback) |
| `util/` | `json.h`, `compat.h`, `grammar.h` (GBNF), `schema_gbnf.h`, `prof.h`, `stw.h` (sliding-window KV), `simd.h` (compat wrapper → `hw/hw.h`) | Support code with no engine knowledge |

## int4 Q4R4 and the ARMv8.0 NEON kernels

`QBITS=4` on aarch64 stores layer matrices (and a tied lm_head) as
`WF_Q4R4` (`Q4FMT=r4`; the default there, selectable everywhere). It
targets in-order ARMv8.0 cores without `SDOT` or fp16 arithmetic
(Cortex-A53): the inner loop is `SMULL`/`SMLAL(2)` of int8 lanes into
int16 accumulators.

**Layout** (`hw/hw_q4r4.h`, packed by `moty_pack_q4r4_block` in
`nn/quant.c`). Rows are grouped in blocks of 4 (the last block of a
matrix with `O % 4 != 0` is zero-padded), the input dimension in groups of
32 (`I % 32 == 0`, otherwise the matrix falls back to the grouped int4).
Per (block, group) the weight stream holds 64 bytes: row `r` byte `j` =
`q[j] | q[j+16] << 4`, `q` in [0,15], value `(q-8)·d`, so unpacking is one
`AND 0x0F` and one `USHR #4` with no zip. The four f16 scales of the
block live in a separate stream `[block][group][4]`. 4.5 bits/weight.
Activations are quantized to int8 per group of 32 (`moty_hw_quant_g32`:
f32 scale `amax/127` + the integer group sum), so the weight offset folds
out as `Σ(q−8)x = Σq·x − 8·Σx` and `q` stays unsigned.

**Scale rule ("no-clip Q4_0").** GGUF Q4_0 maps the largest-magnitude
weight `mx` of a group to −8 (using the extra negative code) and clips
opposite-sign weights beyond `7·d`. Q4R4 takes the finest step that
clips neither sign: `d = −sign(mx)·max(|mx|/8, opp/7)` with `opp` the
largest magnitude of the other sign. Measured with HF fake-quant on 2048
tokens (`tools/ref/hf_qsim.py`), KL(f32‖int4) on LFM2.5-350M: amax/7
1.056, Q4_0 0.844, no-clip 0.747; on MiniCPM5-1B the rule ties with Q4_0
(0.271 vs 0.267). Group size 64 would halve the per-group overhead but
costs quality (LFM2.5 KL 0.747 → 1.004), so it stays 32.

**Kernels.**

- *Decode GEMV* (one token, 4 rows): per group 16 `SMULL/SMLAL(2)` (4
  rows × 32 products into int16 lanes, at most 16 products per lane, so
  no overflow), `ADDP` tree + `SADDLP` to int32, `FCVTL` of the 4 f16
  scales, one `FMLA`. Two groups per loop iteration with two f32
  accumulators: the in-order A53 only overlaps one group's reduction
  chain with the next group's multiplies inside one loop body (68 → 56
  cycles per group). `PRFM PLDL1KEEP` 1 KB ahead on the weight stream
  **and** on the scale stream: on the A53, prefetching 256–512 bytes
  ahead hurt and 1–2 KB helped (`tests/bench_a53.c bw`); the scale-stream
  prefetch alone gave +9–23 % weight bandwidth.
- *Prefill GEMM* (`moty_hw_q4r4_gemm4t`): register tile of 4 rows × 4
  tokens, so each unpacked group is reused for 4 tokens. The driver lays
  out the 4 tokens' per-group scales as vectors once per tile
  (`moty_hw_q4r4_tile_scales`: `xs` and the offset term `8·xsum·xs`), so
  the kernel applies them with FMUL/FMLS by lane instead of per-token
  scalar loads, broadcasts and an int32 offset subtraction (156 → ~124
  instructions per group, 64 of them multiplies; ~3 MAC/cycle per core).
  Tokens beyond a multiple of 4 go through the GEMV;
  `moty_hw_q4r4_gemm(…, ns, …)` remains the generic entry point.
- *Driver* (`moty_matmul_q4r4_s`, `nn/matmul.c`): quantizes the S
  activation rows once, walks tokens in tiles of 32 (their int8
  activations stay in L2 while the weights stream), splits 4-row blocks
  across threads with `schedule(dynamic, 8)` (a core busy with another
  process must not stall a static partition), handles the ragged last
  block.
- *Fused projections*: when every matrix of a resident layer is Q4R4,
  `q/k/v` and `gate/up` are concatenated row-wise at load
  (`moty_mat_fuse_rows`, views `.qkv` / `.gate_up` in the layer) so one
  driver call and one parallel region replace three / two. Bit-identical
  to the unfused path (tests `Lfm2.FusedProjectionsBitExact`,
  `QwenLlama.FusedProjectionsBitExact`).

The integer group sums are exact, so the NEON kernels and the scalar
references agree bit for bit on them; only the f32 accumulation may differ
by FMA contraction. `tests/q4r4_tests.c` checks this (and the no-clip
invariants) on x86 through the references and, as a standalone runner,
on aarch64 — CI runs it under `qemu-aarch64 -cpu cortex-a53`.

**Row ops** (`hw/hw_ops.h`): RMSNorm, SiLU·up with a vector `exp`
(degree-6 polynomial after range reduction, rel. error < 2e-7), softmax,
axpy/add, and the LFM2 short-conv step (`vld2`/`vld3` de-interleave for
K = 3). RoPE uses a per-position cos/sin table instead of
`powf`/`cosf`/`sinf` per head and layer (bit-identical, same expression).
On non-NEON targets the public functions are the scalar references, which
are the loops they replaced, so x86 numerics did not change.

**Pre-packed containers** (`SAVE_PACKED=<dir>`, `runtime/rt_model_load.h`):
a safetensors file where each Q4R4 matrix `name` is stored as U8 `name`
(the 4-row blocks) plus F16 `name.s16` (the scale stream); all other
tensors are copied unchanged, the tied head is written as
`lm_head.weight`, and `config.json` / tokenizer files are copied next to
it. `load_mat_q4r4` reads such pairs raw (size-checked) instead of
reading bf16 and packing. Output is bit-identical to loading the original
snapshot.

**Q8R4 and mixed precision.** `WF_Q8R4` is the int8 sibling of Q4R4 for
the tensors int4 damages most: the same 4-row blocks, groups of 32, f16
scale stream and group-32 int8 activations, 128 bytes per block-group (32
signed codes per row, the no-clip rule with −128/127). The GEMV keeps two
products per int16 lane and folds them with SADDLP/SADALP; the prefill
tile (`moty_hw_q8r4_gemm4t`) reuses a group's weight registers for 4
tokens. `Q8_TENSORS=<glob>,…` (engine-side names, `*` wildcard; the tied
head is `lm_head.weight`) picks Q8R4 per tensor under `QBITS=4 Q4FMT=r4`;
`SAVE_PACKED` stores Q8R4 as I8 + F16 `.s16`, and the loader tells the
formats apart by size, so a mixed container needs no policy at load time.
`tools/ref/pack_r4.py` writes the same containers with GPTQ-chosen codes.

**Attention rows** (`moty_hw_attn_scores` / `moty_hw_attn_accum`): for
head_dim 64 the query (scores) or the 64 output accumulators (value
pass) stay in 16 NEON registers across all cached positions, two rows per
iteration; the previous per-row `dot_f32` / `axpy` loops are the
references (and the kernels for other head sizes and non-NEON builds).

**Two-stage lm_head** (`HEAD_TOPK=K`, `nn/head.c`, opt-in): at load a
1-bit copy of a Q4R4 head is built from the packed codes (sign bits,
per-row mean |w| and popcount; V·D/8 bytes). Per decode step the
activation is quantized to 3 bits, stage 1 scores every row with
bit-plane popcounts (`moty_hw_popc4x3`: AND/CNT/ADD into 12 register
accumulators), the K best rows are selected (a strided-sample threshold,
then an exact select among the survivors), and stage 2 computes exact
Q4R4 logits for their 4-row blocks; all other logits are -1e30. Greedy
decoding and sampling therefore see a top-K-truncated distribution;
REF and PPL always use the full head.

**Embedding injection** (`EMBEDS=<file>`, `runtime/rt_model_load.h`
`embed_row`): raw f32 rows `[N][hidden]` replace the token embedding at
every `EMBEDS_TOKEN` position (default: the snapshot's `image_token_id`),
in prompt order — how HF inserts projected image features, so any
external vision encoder can drive a text backbone. Vision-language
checkpoints load through the `model.language_model.` name fallback in
`io/st.h` and `text_config`.

**EMBED=disk** reuses the micro-RSS row gather: the embedding table is
not resident, each input token's row is read from the snapshot. A tied
lm_head then needs its own copy, which Q4R4 provides.

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

The per-operation table (`OP_T` / `OP_ACC`, `moty_prof_op[]`) is filled by
the lfm2 and qwen engines and by the shared layers in `libmoty-nn`
(attention, conv, FFN), so the libraries must be built with `-DMOTY_PROF`
too; the gen loop prints `[prof-op]` rows (ms, ms/token, share of wall,
plus the unattributed remainder) after each prefill and decode turn.

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
C++ confined to thin glue). 139 tests: kernel exactness (grouped/permuted
int4 vs dequantized references, Q4R4 packing/GEMM/driver and the row ops
vs their scalar references, bit-exact invariants), attention
(batch-vs-incremental, independent serial reference), conv causality,
scratch-arena contract, sampler branches, MoE cache, plus engine-level
greedy-vs-oracle suites (LFM2 dense and Llama-family configs, f32/int8
references, fused-projection and packed-container bit-exactness). `make test-native` reruns everything with
`-march=native` so reference tests exercise your CPU's actual kernels.
`make check` = clean + portable build + full suite.
