/* hw_opencl.h — OpenCL compute backend (placeholder).
 *
 * Implements the same dot_* API as the CPU backends, but offloads
 * computation to an OpenCL device (AMD, Intel, ARM Mali, Adreno, etc.).
 *
 * Activation: define HW_OPENCL at compile time. hw_backend.h will
 * detect the OpenCL backend and use it for matmul-heavy operations.
 *
 * Planned implementation:
 *   1. cl_init(): probe platforms, select device, compile kernels.
 *   2. dot_i8i8: enqueue a single-work-item kernel for small dot products.
 *   3. matmul_i8: NDRange kernel for large GEMV/GEMM (the real win).
 *   4. Memory: cl_mem buffers for weights, transferred once at load time.
 *
 * Dependencies (not yet in Makefile): -lOpenCL
 *
 * Key advantage: portable across AMD/Intel/ARM GPUs. The matmul kernels
 * can achieve 5-50× speedup over CPU for large vocab×hidden GEMV (lm_head).
 */
#ifndef HW_OPENCL_H
#define HW_OPENCL_H

#ifdef HW_OPENCL

/* Stub: all functions abort with a clear message until implemented.
 * When HW_OPENCL is defined but the backend isn't compiled in, these
 * provide a link-time error that guides the developer. */

#error "OpenCL backend not yet implemented. See hw/hw_opencl.h for the plan."

/* When implemented, this file will provide:
 *
 * static int hw_opencl_init(HwBackend *backend);
 *   // Probe CL platforms, create context+queue, compile kernels
 *
 * static int32_t hw_opencl_dot_i8i8(const int8_t *w, const int8_t *x, int n);
 *   // CL kernel: dot_i8_i8 (global_size=n/64, local_size=64)
 *
 * static int32_t hw_opencl_dot_i4i8(const uint8_t *w4, const int8_t *x, int n);
 *   // CL kernel: dot_i4_i8 (unpack nibbles on-GPU, VPDPBUSD-equivalent)
 *
 * static float hw_opencl_dot_f32(const float *a, const float *b, int n);
 *   // CL kernel: dot_f32 (standard FMA reduction)
 *
 * static void hw_opencl_matmul_i8(float *y, const float *x, const int8_t *q,
 *                                 const float *scale, int S, int I, int O);
 *   // CL kernel: matmul_i8_idot (global_size=O, dot per output row)
 *
 * static void hw_opencl_matmul_i4(float *y, const float *x, const uint8_t *q4,
 *                                 const float *scale, int S, int I, int O);
 *   // CL kernel: matmul_i4_idot (global_size=O, nibble unpack + dot)
 *
 * Planned OpenCL kernel source (embedded as string):
 *
 * __kernel void dot_i8_i8(__global const uchar *w, __global const char *x,
 *                          int n, __global int *result) {
 *     int gid = get_global_id(0);
 *     int sum = 0;
 *     for (int i = gid*64; i < min((gid+1)*64, n); i++)
 *         sum += (int)w[i] * (int)x[i];
 *     atomic_add(result, sum);
 * }
 */

#endif /* HW_OPENCL */
#endif /* HW_OPENCL_H */
