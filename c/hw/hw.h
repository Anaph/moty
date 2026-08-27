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
enum { WF_F32=0, WF_I8=1, WF_I4=2, WF_I2=3, WF_I4G=4, WF_Q4K=5, WF_Q6K=6 };

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