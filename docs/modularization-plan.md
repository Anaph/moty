# Moty modularization plan

Rework the codebase from **paste-in headers** (every engine compiles its own
copy of everything, textually) into a **layered library model** with real
link boundaries, one implementation per module, and an ops-table engine
contract. This is the plan of record; execute top to bottom, one phase per
merge, gates after every phase.

## Why now — measured costs of the paste-in model

| Symptom | Measurement (2026-08) |
|---|---|
| Code duplication in binaries | `dot_i8i8` appears 2×, `matmul` OMP outlines 6× in the `moty` binary; 7 TUs each carry a full copy of nn/hw/runtime |
| TU explosion | `engines/qwenmoe.c` = 730 source lines → **14,760 preprocessed lines** (~20×); full engine compile 1.8 s |
| Textual contracts | `MODEL_COMMON_FIELDS` is a macro pasted into 6 different `Model` structs; `ATTN_NORM` selects a *field name* by preprocessor |
| Global-as-config | sampler (`g_temp`, `g_rng`), loader (`g_kv_bits`, `g_qgroup`) are per-TU statics — engine state is not instance state |
| Library code that can't return errors | 24 `exit(1)` calls inside `nn/`, `runtime/`, `io/` — a server embedding Moty cannot survive a bad GGUF |
| Tests include engines | `tests/qwen_tests.c` does `#include "../engines/qwen.c"` — tests can't link against modules, only against whole engines |
| First-build cost, cache misses | any touch to `nn/*.h` recompiles every engine (~13 s full rebuild of 7 TUs) |

What we keep (non-negotiable): zero runtime dependencies; compile-time SIMD
selection (no runtime CPUID dispatch); the portability ladder (x86-64 →
ARMv7 → AArch64 → PPC64, MinGW/macOS/*BSD); **performance parity** (LFM2
57 tok/s class on the 7700X); `make check` / `make test-native` green at
every step.

## Target model

```
        ┌────────────────────────────────────────────┐
        │  moty (dispatcher) + engines/*.c (thin)    │   ← ~300 lines each,
        │  implement MotyEngineOps, register in table│     no kernels, no IO
        └───────────────┬────────────────────────────┘
                        │ MotyEngineOps (vtable, 11 hooks)
        ┌───────────────┴────────────────────────────┐
        │ libmoty-runtime   load/budget/gen/env loops│   ← real .a, one copy
        │                   + MoE expert cache        │
        └───────┬───────────────────┬────────────────┘
                │ MotyModel/MatyMat │ engine-visible layer API
        ┌───────┴─────────┐ ┌───────┴────────────────┐
        │ libmoty-nn      │ │ libmoty-io             │
        │ attn/conv/moe/  │ │ st/gguf/uring/tier     │
        │ ffn/sample/mat  │ │                        │
        └───────┬─────────┘ └────────────────────────┘
                │
        ┌───────┴────────────────────────────────────┐
        │ libmoty-hw      dot_*/qrow/px_* kernels    │  ← ONE implementation
        └────────────────────────────────────────────┘   per -march tier
```

Rules that make the layers honest:

1. **Downward links only.** No layer includes or links upward. `hw` knows
   nothing; `io` knows nothing about models; `runtime` knows engines only
   through the ops table.
2. **One implementation.** A symbol exists once in the final binary. No
   `static` header that defines more than `static inline` accessors.
3. **Explicit specialization.** Today's per-engine macros
   (`ENGINE_GATED_ATTN`, `MOE_GATE_SIGMOID`, `MOE_SHARED_EXPERT`,
   `ATTN_NORM`) become *named variants* in the library
   (`attention()`, `attention_gated()`, `moe_sigmoid_decode1()`,
   `moe_topk_batch()`) selected by a layer-descriptor flag, not by
   pasting different code.
4. **Errors flow up.** Library calls return `moty_result`; the CLI maps
   nonzero to the current stderr format + exit code. `exit()` stays only
   in engines/binaries.
5. **Config is data.** `MotyConfig` struct (QBITS, KV_BITS, threads,
   temp/nucleus, chunking) owned by the caller, threaded through calls —
   no `g_*` statics in libraries.

## Phases

### M0 — Guardrails (½ day)

Before touching anything, freeze the invariants we'll migrate under.

- `tools/bench-gate.sh`: back-to-back 384-token LFM2 + Qwen runs, compares
  against recorded baselines (57.x / 20.x tok/s, ±3% tolerance given the
  shared machine); runs `make check`, `make test-native`, greedy smokes.
- `docs/symbol-map.md`: generated inventory (nm) of what each library will
  export — the future public surface, reviewed once here.
- Decide the naming convention now: exported symbols `moty_<module>_*`
  (`moty_hw_dot_i8i8`, `moty_nn_attention`, `moty_rt_load`), types
  `MotyModel`, `MotyMat`, `MotyEngineOps`.

**Gate:** script green on main.

### M1 — hw becomes a real library (1 day)

- Compile `hw/hw_avx512.h` etc. as .c units (thin: they include the header
  with `MOTY_HW_IMPL` defined once) into `libmoty-hw.a`; kernels lose
  `static`, gain `moty_hw_` prefix during a **transition macro**:

  ```c
  /* hw/hw.h */
  #ifdef MOTY_HW_LEGACY_NAMES
  #define dot_i8i8 moty_hw_dot_i8i8      /* engines keep compiling unchanged */
  #else
  int32_t moty_hw_dot_i8i8(const int8_t*, const int8_t*, int);
  #endif
  ```

- `moty` links the archive; duplicate `dot_i8i8` copies collapse to one.
- Makefile gains `libmoty-hw.a`; engines keep `-I.` includes for now
  (strangler pattern: new boundary, old include paths still work).

**Gate:** bench-gate green; `nm moty | grep -c 'dot_i8i8'` → 1; binary
size drops.

### M2 — core types: Mat, Scratch, quant (1–2 days)

- New `nn/mat.c` + `nn/alloc.c` + `nn/quant.c` compile into
  `libmoty-nn-core.a`. `Mat`, `Scratch`, `pack_int4_*`, `quantize_rows`
  become exported, single-copy.
- `MODEL_COMMON_FIELDS` is replaced by a real struct:

  ```c
  typedef struct MotyCommon {
      MotyMat lm_head; Scratch scr, bscr; /* … the current field list … */
  } MotyCommon;
  /* engine: */ typedef struct { MotyCommon base; Cfg c; Layer *L; … } Model;
  ```

  Engines migrate one at a time (`Model.base.kv_len` etc.); a deprecated
  macro keeps old spellings compiling during the phase.

- `nn_alloc`'s `exit(1)` → `moty_result` return; arenas grow a
  `moty_scr_reserve_ex` that reports failure (OOM behavior at the edges).

**Gate:** bench-gate; 118 tests; no `MODEL_COMMON_FIELDS` outside
`nn/mat.h`.

### M3 — layer library with explicit variants (2–3 days, the heart)

- Each `nn/nn_*.h` layer becomes `nn/<layer>.c` in `libmoty-nn.a`. The
  macro variants are **split at the pasting point into named functions**:

  | Today (macro) | Tomorrow (symbol) |
  |---|---|
  | `attention()` + `ENGINE_GATED_ATTN` | `moty_nn_attention()` and `moty_nn_attention_gated()` |
  | `ATTN_NORM` field-name macro | `MotyLayerView` descriptor: `{ const float *pre_norm; … }` |
  | `moe_decode1` + `MOE_GATE_SIGMOID` | `moty_nn_moe_sigmoid_d1()` / `moty_nn_moe_topk_d1()` |
  | `MOE_SHARED_EXPERT` | flag bit in `MotyMoeDesc` |
  | `MOE_LOAD_EXPERT` | function pointer in `MotyMoeDesc` (already de-facto) |

- Layers stop taking `Model*` (engine-shaped); they take
  `MotyCommon*` + the descriptor. Engines keep their own `Layer` arrays
  and fill descriptors per layer — the engine keeps the *shape*, the
  library owns the *math*.
- `nn_sample` config statics → `MotySampler` struct (temp/nucleus/rng as
  state — this also fixes "two models, one sampler" latent bug).
- Per-layer unit tests move from include-based (`tests/attn_tests.c`
  builds its own Model contract) to linking `libmoty-nn` directly;
  the minimal-contract scaffolding gets deleted as each header converts.

**Gate:** bench-gate (attention/MoE are the hot path — **no regression
accepted here**); batch-invariance + serial-reference tests still green;
`grep -r 'MOE_GATE_SIGMOID' engines/` empty.

### M4 — runtime as a library + ops table (2 days)

- The 11 static hooks become the vtable:

  ```c
  typedef struct MotyEngineOps {
      const char *arch;
      moty_result (*load_cfg)(MotyModel*, const char *snap);
      moty_result (*load_small)(MotyModel*);
      int         (*layer_matrefs)(MotyModel*, int li, MotyMatRef*);
      /* … load/step/kv_alloc/state_reset/build_turn/stops_seed/banner … */
      void        (*post_init)(MotyModel*);   /* expert warm-up */
  } MotyEngineOps;
  ```

- `runtime/rt_*.h` compile to `libmoty-runtime.a`; `engine_main()` becomes
  `moty_rt_serve(const MotyEngineOps*)`. `g_kv_bits`/`g_qgroup`/
  `g_prefill_chunk`/… move into `MotyConfig` filled by
  `moty_rt_config_from_env()` — one parse point, pass-by-pointer after.
- Engines shrink to: ops table + `main(){ return
  moty_rt_serve(&ops); }`.

**Gate:** bench-gate; `make check`; standalone engine binaries still build
and behave identically (env vars, exit codes).

### M5 — dispatcher & (optional) dynamic plugins (1–2 days)

- `moty.c` keeps the static registry but now registers `MotyEngineOps`
  tables instead of `*_main` functions — dispatch by arch string then one
  `moty_rt_serve`.
- Optional, behind `MOTY_PLUGINS=1`: `moty_engine_dlopen(path)` loads
  `moty_plugin_ops()` from a shared object; the Makefile learns a
  `plugin/<name>.so` target. Default builds stay fully static — this is
  an escape hatch for out-of-tree engines, not a requirement.

**Gate:** `moty` binary works exactly as before; one example plugin
compiles (can be `test_engines.so` used by tests).

### M6 — one build system, tests link modules (1–2 days)

- Root `CMakeLists.txt` becomes the build of record: targets
  `moty-hw`, `moty-nn`, `moty-runtime`, `moty` (+ optional plugins),
  `make` shim retained for muscle memory (`make` → `cmake --build`).
- The gtest suites link the libraries — `tests/qwen_tests.c` stops
  including `engines/qwen.c`; instead `libmoty-testengine.a` wraps an
  engine for white-box entry points. Compile time of the full matrix
  drops accordingly (~13 s → seconds for incremental).
- CI gets a matrix job (portable + native) building with CMake and
  exercising bench-gate's correctness half (not perf — CI runners are
  noisy).

**Gate:** `make check` and `cmake --build && ctest` produce the same 118+
results; Dockerfile switches to the CMake path and still produces the slim
image.

### M7 — style pass over the new surface (ongoing)

Applied **only** to code touched by M1–M6, in the same PRs where cheap:

- exported-symbol namespacing (`moty_`), documented in `docs/symbol-map.md`
- `moty_result` error returns; no `exit()` below the engine layer
- `const`-correct read-only pointers in the layer descriptors
- include hygiene: headers include what they use; engines include
  `"nn/attention.h"`-style module headers, never implementation headers
- comments for *why* (contracts), not *what* — preserve the Italian/
  English convention already in the tree

## Migration mechanics

- **Strangler, never big-bang.** Each phase leaves every engine compiling
  and every test green. Old include paths and old spellings survive as
  thin shims until the last consumer migrates, then the shim is deleted
  in the same phase that removes the final user.
- **One phase = one merge = one bench-gate run.** If a phase regresses
  perf, it does not merge until explained (acceptable: ±3% on LFM2/Qwen
  384-token medians, measured back-to-back per `docs/performance.md`
  methodology).
- **Tests migrate with their module.** A header converted in M3 moves its
  test from include-scaffolding to library-linking in the same PR — the
  suite count never drops.

## Explicit non-goals

- No runtime SIMD dispatch (compile-time ladder stays).
- No C++, no external libraries, no new runtime deps — including for the
  plugin path (dlopen is libc/ld.so).
- No public C ABI stability promise before M5 lands; after M5 the
  `MotyEngineOps` layout is versioned (`size_t ops_size` first field).
- `engines/glm.c` remains a self-contained engine (own serving stack,
  CUDA/Metal tiers) — it links `libmoty-hw` only if it wants to, and its
  modularization stays suspended per its header note until a GLM GGUF is
  available to validate against.
- No rewriting working engine code for taste; M7 touches only boundaries.

## Risks

| Risk | Mitigation |
|---|---|
| Out-of-line kernels slow hot loops (lost inlining) | hw dots are already leaf loops; bench-gate after M1 with zero-tolerance on kernel microbench (`tests/bench`) |
| Macro variants hide behavioral forks (e.g. sigmoid gate normalization) | M3 splits variants *mechanically* first (same body, name per macro), then unifies only where tests prove equivalence |
| `tests/*_tests.c` minimal-contract scaffolds encoded undocumented Layer layouts | Descriptors (M3) make those layouts explicit; port tests in the same commit |
| Docker/CI paths drift during CMake switch (M6) | Dockerfile + workflows updated in the M6 PR itself; `make` shim means CI can migrate lazily |
| Shared machine makes ±3% gates flaky | bench-gate uses back-to-back runs + median-of-3, refuses to judge when loadavg > 1.5 |

## Effort & order

| Phase | Est. | Depends on |
|---|---|---|
| M0 guardrails | ½ d | — |
| M1 hw library | 1 d | M0 |
| M2 core types | 1–2 d | M1 |
| M3 layer library | 2–3 d | M2 |
| M4 runtime + ops | 2 d | M3 |
| M5 dispatcher/plugins | 1–2 d | M4 |
| M6 build unification | 1–2 d | M4 (can overlap M5) |
| M7 style | ongoing | rides along |

Total: ~9–13 focused days. The order minimizes risk: value flows early
(M1 already dedupes the hot kernels and halves `moty`'s recompilation
cost), while the riskiest change (M3, hot-path layer conversion) lands
only after the types and tests it needs are in place.
