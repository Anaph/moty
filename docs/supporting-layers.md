# Supporting a new layer type

How to extract, write, or extend a shared layer in `nn/` so that every
engine can paste it in. Use this when a new architecture needs a layer
that doesn't exist yet (new attention variant, new gating scheme, new
recurrent block...), or when an engine-local implementation should become
shared.

Reference examples, in order of increasing complexity:

| Layer | File | What it demonstrates |
|---|---|---|
| Dense FFN | `nn/nn_ffn.h` | minimal: one function, arena scratch, no config |
| Short conv | `nn/nn_conv.h` | causal state, fused single parallel region + barriers |
| Attention | `nn/nn_attn.h` | compile-time config macros, fused VNNI path + fallback, KV store f32/int8 |
| MoE | `nn/nn_moe_sigmoid.h` | router variants behind macros, expert-cache hook, S=1 vs S>1 paths, prof instrumentation |
| DeltaNet | `nn/nn_deltanet.h` | recurrent (state, not KV), custom kernels behind `hw/` |

## Design rules

1. **Header-only, all-static, no .c.** Engines `#include` the header; each
   TU gets its own instance. The API is the function signature, documented
   in the header comment. No global state beyond what the macros need.
2. **Hardware access only through `hw/`.** If the layer needs a new
   primitive (e.g. a new dot variant), add it to `hw/hw.h`'s contract with
   an AVX512 implementation plus a portable fallback in `hw/hw_quant.h`
   (see `dot_i4g8p` for the pattern). Never let intrinsics leak into `nn/`.
3. **Scratch via the Model arena, reserve-before-take.**

```c
scr_reset(&m->scr);
scr_reserve(&m->scr, scr_al(a_bytes) + scr_al(b_bytes));  // ONE reserve
float *a = scr_take(&m->scr, scr_al(a_bytes));
float *b = scr_take(&m->scr, scr_al(b_bytes));            // never moves base
```

   Compute the *total* first, reserve once, then take. No `grow()`, no
   function-local statics — two Models must coexist in one process.
   Serial context only: never inside an OpenMP region.
4. **Engine variation = preprocessor macros.** The engine defines them
   before including:

```c
#define ENGINE_GATED_ATTN   /* q_proj doubled [query|gate] */
#define ATTN_NORM in_ln     /* pre-attention norm field name */
#include "nn/nn_attn.h"
```

   Keep the macro surface minimal (one per real architectural difference,
   not per engine).
5. **Quantization-aware from day one.** Route matmuls through `mat_apply`
   (`nn/nn_mat.h`) so the layer automatically runs f32 / int8 / int4 /
   grouped-int4 depending on the weights' `Mat.fmt` (`WF_F32/WF_I8/WF_I4/
   WF_I4G/WF_I2`). If you hand-roll a dot loop (like the fused paths do),
   provide both the VNNI fast path and a `mat_apply`-equivalent fallback —
   and make the fallback the reference for tests.
6. **Parallel regions: one per phase, not one per row.** Fork/join
   dominates at decode batch sizes. Pattern from `nn_moe_sigmoid.h`:
   2 regions total (all gate+up rows, then all down rows), `collapse`
   instead of div/mod, quantize shared activations once outside. If you
   need ordering between `omp for` sections inside one region, remember
   `omp for` synchronizes at *exit* — add explicit `#pragma omp barrier`
   before reading another section's output (the conv race this caused is
   now a regression test).

## Template

```c
/* nn/nn_<layer>.h — <one-line description>.
 * Include AFTER Layer/Cfg/Model are defined and after nn/nn.h.
 *
 * Config macros (define before including):
 *   <LAYER>_FLAG  — what it changes
 *
 * Requires Layer fields: <...>
 * Requires Cfg fields:   <...>
 * Contract: <batch-invariance / causality / reference guarantees>
 */
#ifndef NN_<LAYER>_H
#define NN_<LAYER>_H

#ifndef <LAYER>_FLAG
#define <LAYER>_FLAG 0
#endif

static void <layer>(Model *m, Layer *l, int li, const float *x, int S, float *out) {
    /* 1. sizes + total scratch, one reserve */
    /* 2. quantize shared activations once (qrow_i8 + px_sum + px_permute) */
    /* 3. one parallel region per phase; dots via hw/ kernels */
    /* 4. serial tail (weighted sums, gates) */
}

#endif /* NN_<LAYER>_H */
```

## Tests a layer must ship with

Add to `tests/<layer>_tests.c` (+ gtest glue), modeled on the existing
suites:

1. **Batch invariance / causality** — `<layer>(S=N) == N × <layer>(S=1)`
   bit-exact, plus any state equality (the conv suite checks the causal
   state too). This catches races, barrier bugs, and state-overwrite bugs.
2. **Independent serial reference** — a dumb scalar reimplementation in
   the test file (no shared kernels); compare with tolerance. This catches
   wiring errors the invariance test can't (wrong GQA mapping, RoPE order,
   missing scale).
3. **Quantized vs f32 path** — I4G/I8 weights vs F32 weights with a
   tolerance loose enough for the quantization but tight enough for
   wiring (see `at_fused_vs_f32`).
4. **Kernel-level** exactness if you added `hw/` primitives (vs dequantized
   reference, bit-exact where the algorithm allows).

Register: `moty_test(test_<layer> <layer>_tests.c <layer>_gtest.cc)` in
`tests/CMakeLists.txt`.

## Wiring a new hw/ primitive (checklist)

```
[ ] contract + range guarantees documented in hw/hw.h (e.g. dot_i8i8's
    x ∈ [-127,127])
[ ] AVX512 implementation (hw/hw_avx512.h) + AVX2/NEON/scalar fallbacks
    (or hw/hw_quant.h portable fallback, guarded by __AVX512VNNI__ etc.)
[ ] tests: exactness vs dequant reference, edge sizes (non-multiples of
    the vector width), bit-exactness where order-independent
[ ] make check (portable) AND make test-native both green
```
