# Adding a new model to Moty

How to support a new architecture end-to-end: GGUF in, tokens out. Budget
1–3 days for a transformer variant that reuses existing layers; the shared
kernels and runtime do the heavy lifting.

Worked reference: `engines/lfm2.c` (~280 lines — the smallest complete
engine) and `engines/qwenmoe.c` (~730 lines — MoE with shared expert).
Read this alongside `docs/architecture.md` for the layering.

## 0. What you need before starting

- A GGUF of the model (llama.cpp's `gguf-py` converts most HF checkpoints).
  Know the **arch string** in its metadata (`general.architecture`).
- A greedy reference for validation: prompt ids + full ids from the
  original model (see "Validate" below).
- The modeling file from transformers (`modeling_<arch>.py`) — the forward
  pass ground truth.

## 1. Choose the integration level

| Situation | Approach |
|---|---|
| Architecture is a recombination of existing layers (GQA attention, SwiGLU/Dense FFN, short-conv, sigmoid-MoE, softmax-topk-MoE) | **Thin engine on shared kernels** — this guide. LFM2 needed 280 lines. |
| One layer type is new (e.g. a new attention variant) | First extract/extend a shared header (`nn/nn_*.h`) following `docs/supporting-layers.md`, then write the thin engine. |
| Radically different execution model | Study `engines/glm.c` — a fully self-contained engine (own kernels, own serving loop). Last resort. |

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

- **GQA attention** — `nn/nn_attn.h`: `attention(m, l, li, x, S, pos_base, out)`.
  Configures via `#define ENGINE_GATED_ATTN` (gated attention) and
  `ATTN_NORM` (pre-attention norm field name). Fuses q/k/v into one VNNI
  region when weights are WF_I4G.
- **Dense FFN / SwiGLU** — `nn/nn_ffn.h`: `dense_ffn(...)`.
- **Short convolution** (hybrid conv nets) — `nn/nn_conv.h`: `conv_layer(...)`
  (fused in_proj→depthwise-causal→out_proj, one parallel region).
- **MoE** — `nn/nn_moe_sigmoid.h`: `moe_decode1` / `moe_batch`. Configure:
  - `MOE_GATE_SIGMOID` — sigmoid+bias gating (LFM2-style) instead of
    softmax top-k (Qwen-style)
  - `MOE_SHARED_EXPERT` — layer has a gated shared expert
  - `MOE_LOAD_EXPERT` — your expert-bytes loader (from `runtime/moe.h`
    cache or disk)
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
4. **Throughput** — ≥384-token generations with `MOTY_NO_OMP_TUNE=1`;
   short runs are dominated by expert-cache warmup and lie.

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
[ ] perf: ≥384-tok run, no warmup distortion
```
