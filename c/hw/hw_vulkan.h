/* hw_vulkan.h — Vulkan compute backend (placeholder).
 *
 * Implements the same dot_* API via Vulkan compute shaders. Vulkan is the
 * most portable GPU API (works on NVIDIA, AMD, Intel, ARM Mali, Adreno,
 * PowerVR, Apple via MoltenVK).
 *
 * Activation: define HW_VULKAN at compile time. Link with -lvulkan.
 *
 * Planned implementation:
 *   1. vk_init(): create VkInstance/VkDevice/VkQueue, allocate descriptor pools.
 *     2. Shader modules: SPIR-V kernels compiled from GLSL at build time.
 *   3. Weight buffers: VkDeviceMemory with DEVICE_LOCAL for GPU-resident weights.
 *   4. Push constants for small params (n, I, O), descriptor sets for buffers.
 *
 * Key advantage: maximum portability. Single codebase runs on any Vulkan GPU.
 * The matmul kernels benefit from GPU's massive parallelism (thousands of ALUs).
 *
 * GLSL compute shader (future, compiled to SPIR-V offline):
 *
 * #version 450
 * layout(local_size_x=64) in;
 * layout(binding=0) readonly buffer WeightBuf { uint w_data[]; };
 * layout(binding=1) readonly buffer InputBuf  { int  x_data[]; };
 * layout(binding=2) writeonly buffer OutBuf   { float y_data[]; };
 * layout(push_constant) uniform Params { int n; int O; } params;
 *
 * shared int partial[64];
 *
 * void main() {
 *     int row = int(gl_WorkGroupID.x);
 *     int tid = int(gl_LocalInvocationID.x);
 *     int sum = 0;
 *     for (int i = tid; i < params.n; i += 64) {
 *         uint packed = w_data[row * (params.n/2) + i/2];
 *         int w = (i & 1) ? int(packed >> 4) - 8 : int(packed & 0xF) - 8;
 *         sum += w * x_data[i];
 *     }
 *     partial[tid] = sum;
 *     barrier();
 *     // tree reduction
 *     for (int s = 32; s > 0; s >>= 1)
 *         if (tid < s) partial[tid] += partial[tid + s];
 *     if (tid == 0) y_data[row] = float(partial[0]);
 * }
 */
#ifndef HW_VULKAN_H
#define HW_VULKAN_H

#ifdef HW_VULKAN
#error "Vulkan backend not yet implemented. See hw/hw_vulkan.h for the plan."
#endif

#endif /* HW_VULKAN_H */
