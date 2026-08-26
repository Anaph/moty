/* hw_cuda.h — NVIDIA CUDA compute backend (placeholder).
 *
 * Implements the same dot_* API via CUDA kernels. Runs on NVIDIA GPUs
 * (RTX series, data center A100/H100, Jetson embedded).
 *
 * Activation: define HW_CUDA at compile time. Compile with nvcc or use
 * the CUDA runtime API from host code. Link with -lcudart.
 *
 * Key advantage on NVIDIA:
 *   - Tensor Cores: int8 matmul via DP4A (4× int8 dot in 1 instruction).
 *   - H100: 1000+ TFLOPS int8, 50-100× faster than CPU for large GEMM.
 *   - High-bandwidth HBM (3 TB/s on H100): eliminates memory bottleneck.
 *
 * Planned implementation:
 *   1. cu_init(): cudaSetDevice, query SM count, allocate streams.
 *   2. Weight buffers: cudaMalloc + cudaMemcpy (one-time transfer).
 *   3. Kernels:
 *      - dot_i8i8: __dp4a intrinsic, warp-level reduction with __shfl.
 *      - matmul_i8: cublasGemmEx (int8 GEMM with Tensor Cores).
 *      - matmul_i4: custom kernel (int4 unpack + __dp4a).
 *   4. For pipeline overlap: CUDA streams for weight transfer + compute.
 *
 * CUDA kernel (future):
 *
 * __global__ void dot_i8_i8(const int8_t *w, const int8_t *x, int n, float *result) {
 *     extern __shared__ int warp_sums[];
 *     int tid = blockIdx.x * blockDim.x + threadIdx.x;
 *     int lane = threadIdx.x & 31;
 *     int warp = threadIdx.x >> 5;
 *     int sum = 0;
 *     for (int i = tid * 4; i < n; i += blockDim.x * gridDim.x * 4) {
 *         if (i + 3 < n)
 *             sum += __dp4a(*(int4*)(w+i), *(int4*)(x+i), 0);
 *     }
 *     // warp reduction
 *     for (int d = 16; d > 0; d >>= 1)
 *         sum += __shfl_xor_sync(0xffffffff, sum, d);
 *     if (lane == 0) warp_sums[warp] = sum;
 *     __syncthreads();
 *     // block reduction...
 * }
 */
#ifndef HW_CUDA_H
#define HW_CUDA_H

#ifdef HW_CUDA
#error "CUDA backend not yet implemented. See hw/hw_cuda.h for the plan."
#endif

#endif /* HW_CUDA_H */
