/* hw_impl.h — PRIVATE: tier ladder for hw.c (libmoty-hw).
 * Never include this from engines: it emits the kernel DEFINITIONS. */
#ifndef HW_IMPL_H
#define HW_IMPL_H
#include "hw.h"

/* Section 0b: system includes per-architecture (intrinsics) */
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
  #include "hw_opencl.h"

#elif defined(HW_VULKAN)
  #include "hw_vulkan.h"

#elif defined(HW_METAL)
  #include "hw_metal.h"

#elif defined(HW_CUDA)
  #include "hw_cuda.h"

#elif defined(HW_WASM)
  #include "hw_wasm.h"

/* --- x86-64 --- */
#elif defined(__AVX512VNNI__) && defined(__AVX512BW__)
  #include "hw_avx512.h"
  #define HW_IDOT_KERNEL "avx512-vnni"

#elif defined(__AVXVNNI__) && defined(__AVX2__)
  #include "hw_avx2.h"
  #define HW_IDOT_KERNEL "avx-vnni"

#elif defined(__AVX2__)
  #include "hw_avx2.h"
  #define HW_IDOT_KERNEL "avx2"

/* --- ARM --- */
#elif defined(__ARM_FEATURE_SVE2) && defined(__ARM_FEATURE_SVE)
  #include "hw_sve2.h"
  #define HW_IDOT_KERNEL "sve2-svdot"

#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  #include "hw_neon.h"
  #define HW_IDOT_KERNEL "neon-dotprod"

#elif defined(__ARM_NEON)
  #include "hw_neon.h"
  #define HW_IDOT_KERNEL "neon"

/* --- POWER --- */
#elif defined(__VSX__)
  #include "hw_vsx.h"
  #define HW_IDOT_KERNEL "vsx"

/* --- fallback --- */
#else
  #include "hw_scalar.h"
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
#include "hw_quant.h"       /* qrow_i8 (portable) */
#include "hw_deltanet.h"    /* dn_row_decay_acc, dn_row_update_dot */

/* ============================================================ *
 *  Section 3: Runtime dispatch (optional).                     *
 *  When HW_RUNTIME_DISPATCH is defined, includes hw_backend.h  *
 *  which provides function-pointer based dispatch for GPU/CPU  *
 *  selection at runtime. Otherwise this is a no-op.            *
 * ============================================================ */
#include "hw_backend.h"


#endif /* HW_IMPL_H */
