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
 *  Section 0a: Weight format enum.                             *
 *  Used by Mat in nn_gemm.h. Defined here so L1 can use it     *
 *  without pulling in any SIMD code.                           *
 * ============================================================ */
enum { WF_F32=0, WF_I8=1, WF_I4=2, WF_I2=3, WF_I4G=4, WF_Q4K=5, WF_Q6K=6 };

/* ============================================================ *
 *  Section 0b: System includes.                                *
 * ============================================================ */
#include <stdint.h>
#include <math.h>

#if defined(__AVX2__) || defined(__AVX512F__)
  #include <immintrin.h>
#elif defined(__ARM_FEATURE_SVE)
  #include <arm_sve.h>
  #include <arm_neon.h>
#elif defined(__ARM_NEON)
  #include <arm_neon.h>
#elif defined(__VSX__)
  #include <altivec.h>
  #undef vector
  #undef pixel
  #undef bool
#endif

/* ============================================================ *
 *  Section 1: Backend selection.                               *
 *                                                              *
 *  Priority: GPU > CPU SIMD. Within CPU SIMD:                  *
 *    AVX512-VNNI > AVX-VNNI > AVX2                            *
 *    SVE2 > NEON-DOTPROD > NEON                               *
 *    VSX                                                       *
 *    WASM-SIMD128                                              *
 *    scalar (fallback)                                        *
 * ============================================================ */

/* --- GPU backends (compile-time opt-in via HW_<NAME>) --- */
#if defined(HW_OPENCL)
  #include "hw/hw_opencl.h"

#elif defined(HW_VULKAN)
  #include "hw/hw_vulkan.h"

#elif defined(HW_METAL)
  #include "hw/hw_metal.h"

#elif defined(HW_CUDA)
  #include "hw/hw_cuda.h"

#elif defined(HW_WASM)
  #include "hw/hw_wasm.h"

/* --- x86-64 --- */
#elif defined(__AVX512VNNI__) && defined(__AVX512BW__)
  #include "hw/hw_avx512.h"
  #define HW_IDOT_KERNEL "avx512-vnni"

#elif defined(__AVXVNNI__) && defined(__AVX2__)
  #include "hw/hw_avx2.h"
  #define HW_IDOT_KERNEL "avx-vnni"

#elif defined(__AVX2__)
  #include "hw/hw_avx2.h"
  #define HW_IDOT_KERNEL "avx2"

/* --- ARM --- */
#elif defined(__ARM_FEATURE_SVE2) && defined(__ARM_FEATURE_SVE)
  #include "hw/hw_sve2.h"
  #define HW_IDOT_KERNEL "sve2-svdot"

#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  #include "hw/hw_neon.h"
  #define HW_IDOT_KERNEL "neon-dotprod"

#elif defined(__ARM_NEON)
  #include "hw/hw_neon.h"
  #define HW_IDOT_KERNEL "neon"

/* --- POWER --- */
#elif defined(__VSX__)
  #include "hw/hw_vsx.h"
  #define HW_IDOT_KERNEL "vsx"

/* --- fallback --- */
#else
  #include "hw/hw_scalar.h"
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

/* ============================================================ *
 *  Section 2: Shared kernels (not SIMD-specific).              *
 *  These use whatever dot_f32 / dot_i8i8 the backend provides. *
 * ============================================================ */
#include "hw/hw_quant.h"       /* qrow_i8 (portable) */
#include "hw/hw_deltanet.h"    /* dn_row_decay_acc, dn_row_update_dot */

/* ============================================================ *
 *  Section 3: Runtime dispatch (optional).                     *
 *  When HW_RUNTIME_DISPATCH is defined, includes hw_backend.h  *
 *  which provides function-pointer based dispatch for GPU/CPU  *
 *  selection at runtime. Otherwise this is a no-op.            *
 * ============================================================ */
#include "hw/hw_backend.h"

#endif /* HW_H */
