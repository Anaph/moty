# Adding a new model to Moty

How to support a new architecture end-to-end: GGUF in, tokens out. Budget
1–3 days for a transformer variant that reuses existing layers; the shared
kernels and runtime do the heavy lifting.

Worked reference: `engines/lfm2.c` (~380 lines — the smallest complete
engine, dense and MoE, GGUF and HF names) and `engines/qwenmoe.c` (~750
lines — MoE with shared expert).
Read this alongside `docs/architecture.md` for the layering.

## 0. What you need before starting

- A GGUF of the model (llama.cpp's `gguf-py` converts most HF checkpoints).
  Know the **arch string** in its metadata (`general.architecture`).
- A greedy reference for validation: prompt ids + full ids from the
  original model (see "Validate" below).
- The modeling file from transformers (`modeling_<arch>.py`) — the forward
  pass ground truth.

## 1. Choose the integration level

First check whether an existing engine already covers the model with a
config switch — that is often a few dozen lines, not an engine:

- **LFM2.5 dense (LFM2.5-350M)** went into `lfm2`: a table maps the HF
  transformers tensor names (`model.layers.N.conv.in_proj.weight`, ...) to
  the GGUF-style names the engine used (the source is detected once from
  the final-norm name); the dense config keys (`norm_eps`,
  `rope_parameters`, `tie_embedding`, `block_auto_adjust_ff_dim`) are read
  next to the MoE ones.
- **Llama-family dense (MiniCPM5-1B, arch `llama`)** went into `qwen`:
  the only structural difference from Qwen3 is the absence of per-head
  QK-norm, so `q_norm`/`k_norm` became optional (NULL → skipped in
  `nn/attn.c`) and `moty.c` maps `llama` to `qwen_main`. Check for hidden
  multipliers (μP-style embedding/residual/logit scales) in the modeling
  file and in `config.json` — a logit-level comparison (below) exposes
  them immediately.
- **Tokenizer quirks** belong in `tok/tok.h`, detected from
  `tokenizer.json`, never keyed on a model name. Example: MiniCPM5 splits
  digit runs of ≤3 first (`Split` pre-tokenizer with `\p{N}{1,3}`,
  `Isolated`), which changes how whitespace next to digits groups; the
  tokenizer enables `digit_presplit` when it sees that exact sequence. A
  post-processor that prepends BOS is read the same way.

| Situation | Approach |
|---|---|
| Architecture is a recombination of existing layers (GQA attention, SwiGLU/Dense FFN, short-conv, sigmoid-MoE, softmax-topk-MoE) | **Thin engine on shared kernels** — this guide. LFM2 needed 280 lines. |
| One layer type is new (e.g. a new attention variant) | First extract/extend a shared header (`nn/nn_*.h`) following `docs/supporting-layers.md`, then write the thin engine. |
| Radically different execution model | Write a fully self-contained engine (own kernels, own serving loop), compiled and registered like the rest. Last resort — large surface, no shared-kernel leverage. |

## 2. Write the engine: `engines/<name>.c`

The engine = config + layer shape + 10 runtime hooks. Skeleton:

```c
#define ENGINE_TAG "<arch>"          /* GGUF arch string */
#define ENGINE_EOT "<|im_end|>\n"    /* end-of-turn for chat building */

#include "io/st.h"
#include "io/gguf.h"
#include "nn/nn.h"                   /* kernels + Mat + MODEL_COMMON_FIELDS */
#include "nn/nn_rope.h"              /* if the arch uses RoPE */
#include "runtime/moe.h"             /* if MoE */
#include "tok/tok.h"
#include "util/compat.h"
#include "util/prof.h"

typedef struct { /* config fields beyond the common ones */ } Cfg;
typedef struct { /* per-layer Mats and vectors */ } Layer;
typedef struct { MODEL_COMMON_FIELDS; } Model;

/* ... hook implementations (step 3) ... */

#include "runtime/runtime.h"         /* pulls the scaffolding; calls hooks */
int main(int argc, char **argv) { return engine_main(argc, argv); }
```

Order matters: the hook *declarations* live in `runtime/runtime.h`, the
definitions come before the include (all static, one instance per TU).

## 3. The runtime hooks

`runtime/runtime.h` declares them; you implement all of them:

| Hook | Job | Notes |
|---|---|---|
| `load_cfg(Cfg*, snap)` | parse config fields (GGUF-synthesized JSON) | start with `cfg_common()` for hidden/layers/heads/inter/vocab/eps/eos, then read the arch-specific keys |
| `load_small(Model*)` | allocate + load norms, biases, per-layer small tensors, and all `Mat` metadata via `st_expect`/`ldm` | big Mats go through `layer_matrefs` (below); MoE experts usually load lazily via `runtime/moe.h` |
| `layer_matrefs(Model*, li, MatRef*)` | list the streamable Mats of layer `li` (name, O, I) — single source of truth for loader, `MEM_GB` streaming, prefetch | return count ≤ `MAX_LAYER_MATS` |
| `fixed_bytes(Model*, ctx)` | resident bytes independent of layer streaming (KV, recurrent state, embeddings) | drives the RAM budget math |
| `step(Model*, ids, S, pos_base)` | **the forward pass**: embed → per-layer → final norm + lm_head; return last-token logits | use `m->bscr` for step-lifetime buffers; shared kernels reset `m->scr` internally |
| `kv_alloc(Model*, max_t)` | allocate the KV cache (`K`,`V` f32 and/or `K8`,`V8`,`Ks`,`Vs` int8) | see any engine; `KV_BITS=8` support is nearly free |
| `state_reset(Model*)` | clear KV/recurrent state between conversations | |
| `build_turn(buf, cap, user)` | wrap user text in the chat template | |
| `stops_seed(Model*, Tok*)` | fill the stop-token set (`stop_add`) | |
| `banner(Model*)` | one startup line with arch, dims, load time | |

Optional macros (define before including `runtime/runtime.h`):

- `ENGINE_POST_INIT(m)` — run once after load; **use it to warm the expert
  cache in parallel** (`#pragma omp parallel for collapse(2)` over layers ×
  experts, like lfm2/qwenmoe). Without warmup the first tokens pay cold
  expert loads and benchmarks lie.
- `ENGINE_MICRO 1` — declare your `step()` can run with no resident
  embeddings (per-row disk gather).
- `ENGINE_LOGITS_HOOK(m, lo)` / `ENGINE_OBSERVE(m, tok)` — instrumentation
  (TTA experiments use these).

## 4. Compose the forward pass from shared layers

Everything below is paste-in: no intrinsics, no OpenMP management.

Layers are library calls (M3): the engine fills a small **view struct**
(weights + config + the per-layer storage from `MotyCommon`) and calls the
variant it wants — no macros, no paste-in:

- **GQA attention** — `nn/attn.h`: fill a `MotyAttnView`, then
  `moty_nn_attention(...)` or `moty_nn_attention_gated(...)` (gated =
  q_proj doubled [q|gate], output *= sigmoid(gate)). Fuses q/k/v into
  one VNNI region when weights are WF_I4G.
- **Dense FFN / SwiGLU** — `nn/ffn.h`: `MotyFfnView` →
  `moty_nn_dense_ffn(...)`.
- **Short convolution** — `nn/conv.h`: `MotyConvView` →
  `moty_nn_conv_layer(...)` (fused in_proj→depthwise-causal→out_proj,
  one parallel region).
- **MoE** — `nn/moe.h`: `MotyMoeView` (router, expert-bias, expert cache
  + your `load_expert` hook, shared expert as a flag) → the variant
  matching your gate: `moty_nn_moe_sigmoid_d1/_batch` (LFM2-style
  sigmoid+bias) or `moty_nn_moe_topk_d1/_batch` (Qwen-style softmax
  top-k).
- **Linear attention** — `nn/nn_deltanet.h` (Gated DeltaNet; still
  paste-in, engine-local).

See `engines/lfm2.c` for the `att_run`/`conv_run`/`ffn_run`/`moe_run`
adapter pattern — a ~15-line view fill per layer call.
- **Linear attention** — `nn/nn_deltanet.h` (Gated DeltaNet, Qwen3.5-style).

If your layer isn't here, extract it first — `docs/supporting-layers.md`.

## 5. Register in the multi-model binary

1. Compile rule + link in `c/Makefile`:

```make
<name>.o: engines/<name>.c $(NN_CORE) ...deps...
	$(CC) $(CFLAGS) -Dmain=<name>_main -c engines/<name>.c -o <name>.o
```

   (add `<name>.o` to the `moty` link line; optionally a standalone
   `<name>$(EXE)` target).

2. Registry in `c/moty.c` (dispatcher):

```c
int <name>_main(int argc, char **argv);          /* top */
{"<arch>",  <name>_main, "Human-readable family"},  /* models[] */
```

`moty` reads the GGUF arch string and dispatches; standalone builds
dispatch the same way with `GGUF=<file>`.

## 6. Validate

1. **Tokenizer parity first** — `make test` builds
   `tests/build/tok_oracle`: `./tok_oracle tokenizer.json < cases.tsv`
   (lines `TEXT\tID,ID,...`). A tokenizer mismatch poisons every later
   comparison.
2. **Greedy vs reference** — generate `ref.json` with transformers:

```python
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float32)
ids = tok(prompt, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=24, do_sample=False, num_beams=1)
json.dump({"prompt_ids": ids[0].tolist(), "full_ids": out[0].tolist()},
          open("ref.json","w"))
```

   Then `GGUF=<file> REF=ref.json PROMPT=... TEMP=0 ./moty` — the engine
   prints the id-by-id match count (exit 0 on full match; f32 build,
   `QBITS=0`, `IDOT=0` for exactness).
3. **Smoke quality** — a couple of greedy completions you can eyeball
   (`The capital of France is` → ` Paris.`).
4. **Logits and quality** — greedy ids can match while the logits are off
   by a missing scale. `tools/ref/` has the scripts: `hf_ref.py` writes
   reference ids *and* last-position logits (`REF_LOGITS=<file>` makes the
   engine dump its own; `cmp_logits.py` reports max |Δ| and top-10
   agreement — expect ~1e-5 at f32), `hf_ppl.py` + `PPL=` / `PPL_OUT=` give
   teacher-forced perplexity and top-1 agreement for the quantized paths.
5. **Throughput** — ≥384-token generations with `MOTY_NO_OMP_TUNE=1`;
   short runs are dominated by expert-cache warmup and lie. For small
   dense models on ARM boards use `IGNORE_EOS=1` for fixed-length runs and
   the `-DMOTY_PROF` per-operation table to see where time goes.

### int4 on ARM (Q4R4)

Nothing engine-specific is required: `load_mat_bits` packs every matrix
listed by `layer_matrefs` into Q4R4 under `QBITS=4` (aarch64 default) and
`mat_apply` dispatches it, and `SAVE_PACKED` writes whatever
`layer_matrefs` lists. Two optional steps pay off on in-order cores:

- **Fuse projections** in `ENGINE_POST_INIT`: `moty_mat_fuse_rows`
  concatenates q/k/v (and gate/up) of a resident Q4R4 layer into one
  matrix; point the layer's `.qkv` / `.gate_up` view at it and split the
  output rows (see `lfm2_fuse` / `qwen_fuse` and the attention/FFN
  `qkv` / `gate_up` view fields). Add a bit-exactness test against the
  unfused path (`lt_fused_bitexact`).
- **Use the `hw/hw_ops.h` row ops** (`moty_hw_rmsnorm`, `_silu_mul`,
  `_add`, ...) instead of scalar loops in `step()`; they are NEON on
  aarch64 and the original loops elsewhere.

## 7. Tests

Add a `tests/<name>_tests.c` + `<name>_gtest.cc` pair following the
existing suites (convention: plain-C functions returning 0/1/2-skip, gtest
glue only declares and wraps). Cover at minimum:

- config parsing of a synthetic GGUF (see `tiny_gguf.h`),
- your `step()` vs a serial reference on random small weights,
- any new layer's kernel-level invariants (batch-vs-incremental,
  causality) as in `tests/attn_tests.c` / `tests/conv_tests.c`.

Register in `tests/CMakeLists.txt`: `moty_test(test_<name> <name>_tests.c
<name>_gtest.cc)`.

## Checklist

```
[ ] engines/<name>.c: Cfg/Layer/Model + 10 hooks (+ ENGINE_POST_INIT warmup)
[ ] forward pass composed of nn/ shared layers
[ ] Makefile: <name>.o rule + moty link line (+ standalone target)
[ ] moty.c: models[] entry
[ ] tok_oracle parity on a real corpus sample
[ ] REF greedy full match (f32)
[ ] smoke: sensible greedy completions
[ ] tests/<name>_tests.c registered in CMakeLists; make check green
[ ] REF_LOGITS max |Δlogit| at f32 ~1e-5; PPL / top-1 of QBITS=8/4
[ ] perf: ≥384-tok run, no warmup distortion
```
