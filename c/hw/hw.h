/* hw.h — Compute Backend Abstraction Layer (Level 0).
 *
 * This is the single include point for all hardware-specific compute kernels.
 * It selects the best available SIMD backend at compile time and provides a
 * uniform API regardless of the underlying hardware.
 *
 * Layered architecture:
 *
 *   ┌─────────────────────────────────────────────────────────┐
 *   │  L4: moty.c, lfm2.c, qwenmoe.c (engine entry pts)   │
 *   │  L3: runtime.h (engine scaffold, paste-in)              │
 *   │  L2: nn_attn.h, nn_conv.h, nn_ffn.h, nn_moe_sigmoid.h  │
 *   │      nn_deltanet.h, moe.h, st.h, gguf.h, tok.h         │
 *   │  L1: nn_gemm.h, nn_norm.h, nn_alloc.h, nn_sample.h     │
 *   │  ══════════════════════════════════════════════════════ │
 *   │  L0: THIS FILE (hw.h) + hw/hw_*.h                      │
 *   │      Backend selection + shared kernels                │
 *   │  ┌─────────┬─────────┬─────────┬─────────┬──────────┐  │
 *   │  │ AVX512  │ AVX2    │ NEON    │ SVE2    │ Scalar   │  │
 *   │  │ hw_*.h  │ hw_*.h  │ hw_*.h  │ hw_*.h  │ hw_*.h   │  │
 *   │  └─────────┴─────────┴─────────┴─────────┴──────────┘  │
 *   │  Placeholders: hw_opencl.h hw_vulkan.h hw_metal.h      │
 *   │               hw_cuda.h hw_wasm.h hw_backend.h          │
 *   └─────────────────────────────────────────────────────────┘
 *
 * Each backend implements the SAME function signatures:
 *   int32_t  dot_i8i8(const int8_t *w, const int8_t *x, int n);
 *   int32_t  dot_i4i8(const uint8_t *w4, const int8_t *x, int n);
 *   float    dot_f32(const float *a, const float *b, int n);
 *   float    dot_f32i8(const float *x, const int8_t *w, int n);
 *
 * To add a new CPU SIMD backend:
 *   1. Create hw/hw_<name>.h implementing the four dot_* functions.
 *   2. Add a selection branch below.
 *
 * To add a new GPU backend:
 *   1. Create hw/hw_<name>.h implementing the four dot_* functions.
 *   2. Add a selection branch below.
 *   3. For runtime CPU/GPU dispatch, include hw/hw_backend.h and use
 *      HW_DOT_I8I8() etc. macros instead of calling dot_i8i8() directly.
 */
#ifndef HW_H
#define HW_H

/* ============================================================ *
 *  Section 1: PUBLIC API (M1: libmoty-hw boundary).            *
 *  ONE compiled implementation lives in hw.c (via hw_impl.h,   *
 *  which picks the -march tier); engines link libmoty-hw.a and *
 *  include this header for prototypes + legacy name macros.    *
 *  The tier strings below must be compiled with the SAME       *
 *  -march as hw.c (both come from the same CFLAGS).            *
 * ============================================================ */
enum { WF_F32=0, WF_I8=1, WF_I4=2, WF_I2=3, WF_I4G=4, WF_Q4K=5, WF_Q6K=6, WF_Q4R4=7, WF_Q8R4=8 };

#include <stdint.h>
#include <math.h>

int32_t moty_hw_dot_i8i8(const int8_t *w, const int8_t *x, int n);
int32_t moty_hw_dot_i4i8(const uint8_t *w4, const int8_t *x, int I);
int32_t moty_hw_dot_i4i8p(const uint8_t *w4, const int8_t *xp, int32_t sxsum, int I);
float   moty_hw_dot_i4g8p(const uint8_t *w4, const float *scl,
                          const int8_t *x, const int32_t *xgsum, int I);
float   moty_hw_dot_f32(const float *a, const float *b, int n);
float   moty_hw_dot_f32i8(const float *x, const int8_t *w, int n);
float   moty_hw_qrow_i8(const float *x, int8_t *q, int n);
void    moty_hw_px_permute(const int8_t *x, int8_t *xp, int I);
int32_t moty_hw_px_sum(const int8_t *x, int I);
void    moty_hw_dn_row_decay_acc(float *restrict S, float dec, float ki,
                                 float *restrict kv, int dv);
void    moty_hw_dn_row_update_dot(float *restrict S, float ki,
                                  const float *restrict delta, float qi,
                                  float *restrict oh, int dv);
/* int4 Q4R4 (hw/hw_q4r4.h): 4-row blocks, groups of 32, f16 scales;
 * activations int8 per group of 32 (+ f32 scale, int32 group sum).
 * gemm: one 4-row block x ns tokens -> y[t*ys + r]; ns=1 is the decode GEMV. */
void    moty_hw_quant_g32(const float *x, int I, int8_t *xq, float *xs, int32_t *xsum);
void    moty_hw_q4r4_gemm(const uint8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                          const int32_t *xsum, int nb, int ns, float *y, int ys);
/* prefill tile: 4 tokens (activation rows ldx apart) with per-group scales
 * pre-arranged by moty_hw_q4r4_tile_scales (xst[g*4+t] = xs[t][g],
 * xct[g*4+t] = 8*xsum*xs) */
void    moty_hw_q4r4_gemm4t(const uint8_t *w, const uint16_t *d, const int8_t *xq, int64_t ldx,
                            const float *xst, const float *xct, int nb, float *y, int ys);
void    moty_hw_q4r4_gemm4t_ref(const uint8_t *w, const uint16_t *d, const int8_t *xq, int64_t ldx,
                                const float *xst, const float *xct, int nb, float *y, int ys);
void    moty_hw_q4r4_tile_scales(const float *xs, const int32_t *xsum, int nb, float *xst, float *xct);
/* Q8R4 (int8 codes in the Q4R4 block/scale layout, 128 B per block-group) */
void    moty_hw_q8r4_gemm(const int8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                          int nb, int ns, float *y, int ys);
void    moty_hw_q8r4_gemm_ref(const int8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                              int nb, int ns, float *y, int ys);
/* scalar references of the two above (always compiled: tests compare) */
void    moty_hw_quant_g32_ref(const float *x, int I, int8_t *xq, float *xs, int32_t *xsum);
void    moty_hw_q4r4_gemm_ref(const uint8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                              const int32_t *xsum, int nb, int ns, float *y, int ys);
float   moty_hw_f16_to_f32(uint16_t h);
/* non-matmul row ops (hw/hw_ops.h); *_ref = scalar reference (= portable kernel) */
void    moty_hw_rmsnorm(float *out, const float *x, const float *w, int n, float eps);
void    moty_hw_silu_mul(float *g, const float *u, int64_t n);          /* g = silu(g)*u */
void    moty_hw_softmax(float *x, int n);
void    moty_hw_axpy(float *y, float a, const float *x, int n);          /* y += a*x */
void    moty_hw_add(float *y, const float *x, int64_t n);                /* y += x */
void    moty_hw_shortconv_step(float *y, const float *b, const float *c, const float *x,
                               const float *w, float *state, int K, int c0, int c1);
void    moty_hw_rmsnorm_ref(float *out, const float *x, const float *w, int n, float eps);
void    moty_hw_silu_mul_ref(float *g, const float *u, int64_t n);
void    moty_hw_softmax_ref(float *x, int n);
void    moty_hw_axpy_ref(float *y, float a, const float *x, int n);
void    moty_hw_add_ref(float *y, const float *x, int64_t n);
void    moty_hw_shortconv_step_ref(float *y, const float *b, const float *c, const float *x,
                                   const float *w, float *state, int K, int c0, int c1);
/* attention rows (n rows hd floats apart): sc[t] = scale*<q,K[t]>, cx = Σ sc[t]*V[t] */
void    moty_hw_attn_scores(float *sc, const float *q, const float *K, int n, int hd, float scale);
void    moty_hw_attn_accum(float *cx, const float *sc, const float *V, int n, int hd);
void    moty_hw_attn_scores_ref(float *sc, const float *q, const float *K, int n, int hd, float scale);
void    moty_hw_attn_accum_ref(float *cx, const float *sc, const float *V, int n, int hd);
/* cnt[r*3+p] = popcount(b[r*bs ..] & pl[p*nbytes ..]) over nbytes (%16 == 0), r<4, p<3 */
void    moty_hw_popc4x3(const uint8_t *b, int64_t bs, const uint8_t *pl, int nbytes, uint32_t *cnt);
void    moty_hw_popc4x3_ref(const uint8_t *b, int64_t bs, const uint8_t *pl, int nbytes, uint32_t *cnt);

/* Legacy spellings: engines/nn headers keep calling dot_i8i8(...) —
 * rewritten to the exported symbol. Delete when M3/M4 migrate callers
 * (docs/symbol-map.md tracks the final names). */
#ifndef MOTY_HW_NO_LEGACY
#define dot_i8i8         moty_hw_dot_i8i8
#define dot_i4i8         moty_hw_dot_i4i8
#define dot_i4i8p        moty_hw_dot_i4i8p
#define dot_i4g8p        moty_hw_dot_i4g8p
#define dot_f32          moty_hw_dot_f32
#define dot_f32i8        moty_hw_dot_f32i8
#define qrow_i8          moty_hw_qrow_i8
#define px_permute       moty_hw_px_permute
#define px_sum           moty_hw_px_sum
#define dn_row_decay_acc moty_hw_dn_row_decay_acc
#define dn_row_update_dot moty_hw_dn_row_update_dot
#endif

/* --- GPU backends (compile-time opt-in via HW_<NAME>) --- */
#if defined(HW_OPENCL)
  /* tier impl: hw_impl.h (libmoty-hw) */

#elif defined(HW_VULKAN)
  /* tier impl: hw_impl.h (libmoty-hw) */

#elif defined(HW_METAL)
  /* tier impl: hw_impl.h (libmoty-hw) */

#elif defined(HW_CUDA)
  /* tier impl: hw_impl.h (libmoty-hw) */

#elif defined(HW_WASM)
  /* tier impl: hw_impl.h (libmoty-hw) */

/* --- x86-64 --- */
#elif defined(__AVX512VNNI__) && defined(__AVX512BW__)
  /* tier impl: hw_impl.h (libmoty-hw) */
  #define HW_IDOT_KERNEL "avx512-vnni"

#elif defined(__AVXVNNI__) && defined(__AVX2__)
  /* tier impl: hw_impl.h (libmoty-hw) */
  #define HW_IDOT_KERNEL "avx-vnni"

#elif defined(__AVX2__)
  /* tier impl: hw_impl.h (libmoty-hw) */
  #define HW_IDOT_KERNEL "avx2"

/* --- ARM --- */
#elif defined(__ARM_FEATURE_SVE2) && defined(__ARM_FEATURE_SVE)
  /* tier impl: hw_impl.h (libmoty-hw) */
  #define HW_IDOT_KERNEL "sve2-svdot"

#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  /* tier impl: hw_impl.h (libmoty-hw) */
  #define HW_IDOT_KERNEL "neon-dotprod"

#elif defined(__ARM_NEON)
  /* tier impl: hw_impl.h (libmoty-hw) */
  #define HW_IDOT_KERNEL "neon"

/* --- POWER --- */
#elif defined(__VSX__)
  /* tier impl: hw_impl.h (libmoty-hw) */
  #define HW_IDOT_KERNEL "vsx"

/* --- fallback --- */
#else
  /* tier impl: hw_impl.h (libmoty-hw) */
  #define HW_IDOT_KERNEL "scalar"
#endif

/* F32 kernel name (separate from IDOT) */
#ifndef HW_F32_KERNEL
  #if defined(__ARM_FEATURE_SVE)
    #define HW_F32_KERNEL "sve"
  #elif defined(__AVX512F__)
    #define HW_F32_KERNEL "avx512f"
  #elif defined(__AVX2__) && defined(__FMA__)
    #define HW_F32_KERNEL "avx2-fma"
  #elif defined(__ARM_NEON)
    #define HW_F32_KERNEL "neon"
  #else
    #define HW_F32_KERNEL "scalar"
  #endif
#endif

/* Backwards-compat aliases */
#define IDOT_KERNEL HW_IDOT_KERNEL
#define F32_KERNEL  HW_F32_KERNEL

#include "hw_backend.h"

#endif /* HW_H */