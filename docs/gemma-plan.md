# Engine review & improvement plan (gemma first)

Review date: 2026-08-28, post-modularization (M0–M7). Scope: the four
runtime-hook engines + olmoe; glm is compile-only and **excluded** (never
tested against a real model — see the note in `docs/architecture.md`).

## Current state review

| | lfm2 | qwenmoe | qwen | gemma | olmoe |
|---|---|---|---|---|---|
| Lines | 307 | 746 | 688 | 546 | 415 |
| Shared lib attention (`nn/attn.h`) | ✓ | ✓ (gated) | ✗ own | ✗ own | ✗ own |
| Shared MoE (`nn/moe.h`) | ✓ sigmoid | ✓ topk+shared | — dense | — dense | ✗ own (uses lib cache) |
| Scratch arena in hot path | ✓ | ✓ | ✓ | ✗ falloc/free per token | partial |
| Fused VNNI projections | ✓ | ✓ | ✗ | ✗ | ✗ |
| `ENGINE_POST_INIT` warm-up | ✓ | ✓ | ✓ | n/a (dense) | ✗ |
| `MOTY_PROF` instrumentation | ✓ | — | — | — | — |
| KV int8 (`KV_BITS=8`) | ✓ | ✓ | ✓ | ✓ | ✗ |
| On runtime hooks (`runtime.h`) | ✓ | ✓ | ✓ | ✓ | ✗ hand-rolled |
| On `MotyCommon` | ✓ | ✓ | ✓ | ✓ | ✗ |
| Validated vs real model | ✓ GGUF | ✓ GGUF | ? | **✗ VERIFY markers** | ? ref.json only |

**Verdict:** lfm2/qwenmoe are the reference implementations — thin engines
over the shared libraries with the full perf stack (arena, fused VNNI,
expert warm-up, int4). **gemma is the worst-positioned runtime engine**:
per-token `falloc/free` in attention, no fused VNNI path, no arena, and a
stack of unverified checkpoint conventions (`VERIFY` in `engines/gemma.c`:
`(1+w)` vs `w` norm, `rope_parameters` nesting, per-layer-embeddings norm)
that were never resolved because no Gemma model was ever run end-to-end.

Test coverage exists and is decent on primitives (rope/GeGLU/norm/sliding
vs brute force, tiny hybrid determinism, kv-shared aliasing, prefill chunk
bit-exactness, kv8 tolerance — 8 suites) but there is **no end-to-end REF
oracle test** and no attention batch-invariance/serial-reference coverage
of the real `attention()` (the `attn_tests.c` classes apply only to the
shared lib).

## Plan

### G0 — Real-model bring-up (blocking everything else) — ½–1 day

- Obtain a small Gemma GGUF (e.g. gemma-2-2b-it Q4_K_M class) and run it:
  `GGUF=... PROMPT="hi" ./moty`. First goal: tokenize correctly
  (`tests/build/tok_oracle` parity on a corpus sample — Gemma is
  SentencePiece with metaspace+byte-fallback, `tok/tok_unicode.h`).
- Resolve every `VERIFY` marker in `engines/gemma.c` against the real
  checkpoint; delete the ones confirmed, fix the ones wrong
  (`GEMMA_NORM_PLAIN` should become a decision, not an escape hatch).
- Produce `ref.json` (transformers greedy, recipe in
  `docs/adding-a-model.md`) and get a full id-by-id REF match at f32.
- Add the gemma baseline to `tools/bench-gate.sh` (tok/s gate like
  LFM2/Qwen have).

### G1 — Attention modernization onto the shared library — 2–3 days

The blocker was always "gemma attention is special". Make the special
parts **data** in `MotyAttnView` instead of a private engine copy:

- extend `nn/attn.h` with the gemma semantics, as fields + small variant
  functions (keep the lfm2/qwenmoe paths untouched):
  - `window` (sliding): scores masked to `t > qpos - window`; the KV store
    keeps everything, the mask is per-row (matches the current semantics,
    brute-force test exists)
  - dual head-dim / dual kv-heads: `hd_full/kv_full` vs `hd/kv` selected
    per layer — already a `Layer.type` distinction, becomes two view fills
  - per-layer `v`-norm (`vn`) and `k_eq_v` aliasing: two more view fields
    (`vn`, `k_eq_v`) consumed in the store path
  - p-RoPE (partial rotary, `rot_angles`): `rot` already a view field;
    the partial variant is `rot < hd/2`, same `rope_head`
- port the `attn_tests.c` classes onto the gemma variant:
  batch-vs-incremental bit-exact, independent serial reference (sliding
  window included), fused-I4G vs f32 tolerance.
- engine keeps: layer typing, view fill, `gnorm_row` conventions.

Gate: existing gemma suites still green + new invariance tests + REF match
from G0 unchanged + bench-gate.

### G2 — Hot-path allocation: arena, not malloc — 1 day

- Replace the per-call `falloc/free` pairs (attention q/k/v, GeGLU g/u,
  ple projections, `step()` xb/nb/tb) with `bscr` step-arena and `scr`
  kernel-arena usage exactly like lfm2's `step()`.
- This is where lfm2 gained measurable decode time (glibc malloc churn at
  3 allocs × layer × token); gemma currently does ~4-6 per layer.

### G3 — Quantization parity: fused VNNI + int4 lm_head — 1 day

- Once attention is on `mat_apply` with `WF_I4G` weights through the
  shared view, the fused q/k/v VNNI region (`dot_i4g8p`) comes for free
  for the non-gated path — gemma's mats just need `gs=32 && D%64==0`
  packing at load (the loader already supports it via `load_mat`).
- Optional `lm_head` int4 at QBITS=4 (tied-embedding models only), parity
  with lfm2's 262→131 MB/token logits win; gate on token-exact greedy vs
  int8 lm_head like lfm2 did (48/48 tokens).

### G4 — Sliding-window KV ring (optional, long contexts) — 1–2 days

`util/stw.h` (qwen's sliding-window KV) applied to gemma's LT_SLIDE
layers: KV memory drops from `layers × ctx × hd` to
`layers × window × hd` — at 128k ctx that is the difference between
"impossible" and "fine". Bit-exactness gate: ring KV == dense KV with
mask on a REF run. Do only if long-ctx matters.

### G5 — Perf + docs close-out — ½ day

- `MOTY_PROF` instrumentation in the gemma step (windowed phases).
- bench-gate baseline recorded; numbers into `docs/performance.md`.
- Update `docs/adding-a-model.md` example pointers from lfm2 to include
  the gemma attention-extension case (how to add semantics to a view).

### Parallel track (not blocking gemma)

- **qwen.c**: own attention exists because of the Qwen3.5 hybrid
  (DeltaNet + gated attention). The gated half could move onto
  `moty_nn_attention_gated` (it was built for exactly that shape);
  DeltaNet stays engine-local. Also: no prof instrumentation.
- **olmoe.c**: bring onto `MotyCommon` + runtime hooks (it still has a
  hand-rolled Model, no KV8, no ENGINE_POST_INIT warm-up — its expert
  cache misses pay cold disk loads on the first tokens). This is mostly
  mechanical now that `moe.h`/`kvcache.h` are libraries.
- **M4b** (from `docs/modularization-plan.md`): Cfg common prefix →
  `MotyEngineOps` vtable → `moty_rt_serve`. olmoe's conversion (above) is
  the natural forcing function.

## Suggested order

G0 → G2 → G1 → G3 → G5, with G4 optional and the parallel track
interleaved whenever a gemma step is blocked on measurement. G2 before G1
because it is mechanical, immediately measurable, and de-risks the
attention rewrite (the perf delta of each step stays attributable).
