/* hw_metal.h — Apple Metal compute backend (placeholder).
 *
 * Implements the same dot_* API via Metal compute shaders. Runs on
 * Apple M-series unified memory GPUs (M1/M2/M3/M4) and A-series iOS.
 *
 * Activation: define HW_METAL at compile time. Link with -framework Metal.
 * Only available on macOS/iOS (guarded by __APPLE__).
 *
 * Key advantage on Apple Silicon:
 *   - Unified memory: zero-copy weight buffers (no PCIe transfer).
 *   - M2 GPU: 16-19 TFLOPS fp32, 2-4× faster than CPU for GEMV.
 *   - Metal Performance Shaders (MPS): hardware matmul via MPSMatrixMultiplication.
 *
 * Planned implementation:
 *   1. mt_init(): MTLCreateSystemDefaultDevice(), create command queue.
 *   2. Weights: id<MTLBuffer> with MTLResourceStorageModeShared (unified memory).
 *   3. Shaders: .metal source compiled at runtime via MTLLibrary.
 *   4. For large matmuls: MPSMatrixMultiplication (Apple's tuned GEMM).
 *   5. For small dot products: custom kernel with simdgroup_matrix (M2+).
 *
 * Metal shader (future .metal source):
 *
 * #include <metal_stdlib>
 * using namespace metal;
 *
 * kernel void dot_i8_i8(device const char *w [[buffer(0)]],
 *                       device const char *x [[buffer(1)]],
 *                       device float *result [[buffer(2)]],
 *                       constant int &n [[buffer(3)]],
 *                       uint tid [[thread_position_in_threadgroup]],
 *                       uint tg [[threadgroup_position_in_grid]],
 *                       uint ts [[threads_per_threadgroup]]) {
 *     threadgroup int partial[256];
 *     int sum = 0;
 *     for (uint i = tid; i < (uint)n; i += ts)
 *         sum += (int)w[i] * (int)x[i];
 *     partial[tid] = sum;
 *     threadgroup_barrier(mem_flags::mem_threadgroup);
 *     // reduction...
 * }
 */
#ifndef HW_METAL_H
#define HW_METAL_H

#ifdef HW_METAL
#error "Metal backend not yet implemented. See hw/hw_metal.h for the plan."
#endif

#endif /* HW_METAL_H */
