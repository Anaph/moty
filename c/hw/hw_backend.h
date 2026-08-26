/* hw_backend.h — runtime backend dispatch infrastructure.
 *
 * This header defines the abstraction layer for runtime-selectable compute
 * backends. On CPU, the backend is selected at compile time via hw.h's
 * #ifdef dispatch. On systems with GPU support, this header provides the
 * infrastructure to select between CPU and GPU backends at runtime.
 *
 * For CPU-only builds (the default), hw_backend.h is a no-op: the compile-time
 * static inline functions from hw.h are used directly. The function-pointer
 * dispatch is only activated when HW_RUNTIME_DISPATCH is defined.
 *
 * Backend lifecycle:
 *   ┌──────────────┐
 *   │  hw.h        │  compile-time: selects best CPU SIMD backend
 *   │  (L0)        │  static inline dot_i8i8, dot_i4i8, dot_f32, dot_f32i8
 *   ├──────────────┤
 *   │  hw_backend.h│  runtime: optional dispatch table
 *   │  (L0.5)      │  HwBackend struct + dispatch macros
 *   ├──────────────┤
 *   │  hw_opencl.h │  GPU backends (future)
 *   │  hw_vulkan.h │  each implements the same dot_* API
 *   │  hw_metal.h  │  but via GPU kernels
 *   │  hw_cuda.h   │
 *   └──────────────┘
 */
#ifndef HW_BACKEND_H
#define HW_BACKEND_H

/* ============================================================ *
 *  Backend descriptor.                                         *
 * ============================================================ */

typedef enum {
    HW_BACKEND_CPU       = 0,  /* CPU SIMD (AVX512, NEON, SVE2, ...) */
    HW_BACKEND_OPENCL     = 1,  /* OpenCL (AMD, Intel, ARM Mali) */
    HW_BACKEND_VULKAN     = 2,  /* Vulkan compute (portable GPU) */
    HW_BACKEND_METAL      = 3,  /* Apple Metal (M-series GPU) */
    HW_BACKEND_CUDA       = 4,  /* NVIDIA CUDA */
    HW_BACKEND_WASM_SIMD  = 5,  /* WebAssembly SIMD128 */
} HwBackendType;

typedef struct {
    HwBackendType type;
    const char *name;           /* "avx512-vnni", "opencl", "vulkan", ... */
    const char *device;         /* "AMD Radeon 780M", "Apple M2 GPU", ... */
    int compute_units;          /* SM count, EU count, or core count */
    int64_t memory_bytes;       /* device memory (0 = unified/shared) */

    /* Function pointers — populated by backend init.
     * NULL = not supported by this backend. */
    int32_t (*dot_i8i8)(const int8_t *w, const int8_t *x, int n);
    int32_t (*dot_i4i8)(const uint8_t *w4, const int8_t *x, int n);
    float   (*dot_f32)(const float *a, const float *b, int n);
    float   (*dot_f32i8)(const float *x, const int8_t *w, int n);

    /* Batch operations (future: GPU-accelerated matmul) */
    void (*matmul_i8)(float *y, const float *x, const int8_t *q, const float *scale,
                      int S, int I, int O);
    void (*matmul_i4)(float *y, const float *x, const uint8_t *q4, const float *scale,
                      int S, int I, int O);
} HwBackend;

/* ============================================================ *
 *  Dispatch macros.                                            *
 *                                                              *
 *  When HW_RUNTIME_DISPATCH is defined, dot_* calls go through  *
 *  function pointers. Otherwise, they call the static inline   *
 *  directly (zero overhead).                                   *
 * ============================================================ */

#ifdef HW_RUNTIME_DISPATCH
  /* Runtime: use function pointer if backend is active, else static inline */
  extern const HwBackend *g_hw_backend;
  #define HW_DOT_I8I8(w,x,n)  (g_hw_backend&&g_hw_backend->dot_i8i8 ? g_hw_backend->dot_i8i8(w,x,n)  : dot_i8i8(w,x,n))
  #define HW_DOT_I4I8(w,x,n)  (g_hw_backend&&g_hw_backend->dot_i4i8 ? g_hw_backend->dot_i4i8(w,x,n)  : dot_i4i8(w,x,n))
  #define HW_DOT_F32(a,b,n)   (g_hw_backend&&g_hw_backend->dot_f32  ? g_hw_backend->dot_f32(a,b,n)   : dot_f32(a,b,n))
  #define HW_DOT_F32I8(x,w,n) (g_hw_backend&&g_hw_backend->dot_f32i8? g_hw_backend->dot_f32i8(x,w,n) : dot_f32i8(x,w,n))
#else
  /* Compile-time: direct call, zero overhead */
  #define HW_DOT_I8I8(w,x,n)  dot_i8i8(w,x,n)
  #define HW_DOT_I4I8(w,x,n)  dot_i4i8(w,x,n)
  #define HW_DOT_F32(a,b,n)   dot_f32(a,b,n)
  #define HW_DOT_F32I8(x,w,n) dot_f32i8(x,w,n)
#endif

#endif /* HW_BACKEND_H */
