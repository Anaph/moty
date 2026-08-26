/* hw_scalar.h — pure C fallback backend (no SIMD).
 * Used when no SIMD extension is available. Also serves as the reference
 * implementation that all SIMD backends must match numerically.
 */
#ifndef HW_SCALAR_H
#define HW_SCALAR_H

static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += (int32_t)w[i] * x[i];
    return sum;
}

static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum = 0; int i = 0;
    for (; i+1 < I; i += 2) {
        uint8_t b = w4[i>>1];
        sum += ((int)(b&0xF)-8)*x[i] + ((int)(b>>4)-8)*x[i+1];
    }
    if (i < I) { uint8_t b = w4[i>>1]; sum += ((int)(b&0xF)-8)*x[i]; }
    return sum;
}

static inline float dot_f32(const float *a, const float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline float dot_f32i8(const float *x, const int8_t *w, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * (float)w[i];
    return s;
}

#endif /* HW_SCALAR_H */
