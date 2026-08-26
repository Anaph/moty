/* hw_wasm.h — WebAssembly SIMD128 backend (placeholder).
 *
 * Implements the same dot_* API via WASM SIMD128 intrinsics. Runs in
 * browsers (Chrome, Firefox, Safari) and Node.js with WASM SIMD enabled.
 *
 * Activation: define HW_WASM at compile time. Compile with
 *   clang --target=wasm32 -msimd128 -mbulk-memory
 *
 * Key WASM SIMD128 operations:
 *   - wasm_i8x16_dot_i7x16_saturate: int8 dot product (i32 from 16 pairs)
 *   - wasm_i16x8_extmul_low_i8x16: int8 → int16 widening multiply
 *   - wasm_f32x4_mul + wasm_f32x4_add: FMA equivalent
 *   - v128_load / v128_store: 128-bit memory operations
 *
 * WASM SIMD128 is narrower than AVX512 (128-bit vs 512-bit) but enables
 * in-browser inference at ~5-10× speedup over scalar WASM.
 *
 * Planned implementation:
 *   1. No init needed (WASM has no device discovery).
 *   2. dot_i8i8: wasm_i8x16_dot_i7x16_saturate per 16 elements + reduce.
 *   3. dot_i4i8: nibble unpack via wasm_v8x16_shuffle + dot.
 *   4. dot_f32: wasm_f32x4_mul + wasm_f32x4_add (16-wide unrolled).
 *
 * Note: WASM doesn't support OpenMP. Parallelism must be via
 * SharedArrayBuffer + Web Workers, or single-threaded with SIMD.
 */
#ifndef HW_WASM_H
#define HW_WASM_H

#ifdef HW_WASM
#include <wasm_simd128.h>

static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int n) {
    int32_t sum = 0; int i = 0;
    for (; i+16 <= n; i += 16) {
        v128_t wv = wasm_v128_load(w + i);
        v128_t xv = wasm_v128_load(x + i);
        sum += wasm_i8x16_dot_i7x16_saturate(wv, xv);
    }
    for (; i < n; i++) sum += (int32_t)w[i] * x[i];
    return sum;
}

static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum = 0; int i = 0;
    const v128_t m4 = wasm_i8x16_splat(0x0F);
    const v128_t b8 = wasm_i8x16_splat(8);
    for (; i+16 <= I; i += 16) {
        v128_t by = wasm_v128_load(w4 + (i >> 1));
        v128_t lo = wasm_v128_and(by, m4);
        v128_t hi = wasm_v128_and(wasm_u8x16_shr(by, 4), m4);
        /* interleave lo,hi via shuffle */
        v128_t wv = wasm_i8x16_sub(wasm_i8x16_shuffle(lo, hi,
            0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23), b8);
        v128_t xv = wasm_v128_load(x + i);
        sum += wasm_i8x16_dot_i7x16_saturate(wv, xv);
    }
    for (; i+1 < I; i += 2) { uint8_t b = w4[i>>1];
        sum += ((int)(b&0xF)-8)*x[i] + ((int)(b>>4)-8)*x[i+1]; }
    if (i < I) { uint8_t b = w4[i>>1]; sum += ((int)(b&0xF)-8)*x[i]; }
    return sum;
}

static inline float dot_f32(const float *a, const float *b, int n) {
    float s = 0; int i = 0;
    v128_t acc = wasm_f32x4_splat(0);
    for (; i+4 <= n; i += 4)
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(a+i), wasm_v128_load(b+i)));
    s = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
        wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline float dot_f32i8(const float *x, const int8_t *w, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * (float)w[i];
    return s;
}

#define HW_IDOT_KERNEL "wasm-simd128"
#define HW_F32_KERNEL  "wasm-simd128"

#endif /* HW_WASM */
#endif /* HW_WASM_H */
