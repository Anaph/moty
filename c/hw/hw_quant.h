/* hw_quant.h — quantization + packing (shared across all backends).
 * These functions use only scalar C or call the backend's dot product.
 * No SIMD intrinsics here — portable across all hw_*.h backends.
 */
#ifndef HW_QUANT_H
#define HW_QUANT_H

/* Symmetric per-row quantization: amax/127. Returns scale. */
static inline float qrow_i8(const float *x, int8_t *q, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    float s = amax / 127.f; if (s < 1e-12f) s = 1e-12f;
    float inv = 1.f / s;
    for (int i = 0; i < n; i++) q[i] = (int8_t)lrintf(x[i] * inv);
    return s;
}

#ifndef __AVX512VNNI__
/* fallback portabili (AVX2/scalar): l'unpack 128-bit dei backend non-AVX512
 * produce ordine SEQUENZIALE → px_permute = identita'; dot_i4i8p == dot_i4i8. */
static inline void px_permute(const int8_t *x, int8_t *xp, int I) {
    for (int i = 0; i < I; i++) xp[i] = x[i];
}
static inline int32_t px_sum(const int8_t *x, int I) {
    int32_t s = 0; for (int i = 0; i < I; i++) s += x[i]; return s;
}
static inline int32_t dot_i4i8p(const uint8_t *w4, const int8_t *xp, int32_t sxsum, int I) {
    (void)sxsum;
    return dot_i4i8(w4, xp, I);
}
/* grouped (gs=32) fallback: somma scalare per gruppo con scala f32.
 * matmul_i4_grouped_s non la usa fuori AVX512, ma nn_attn/nn_conv la
 * referenziano nelle loro regioni VNNI (guardate da fmt==WF_I4G&&gs==32,
 * ma il simbolo deve esistere). */
static inline float dot_i4g8p(const uint8_t *w4, const float *scl,
                              const int8_t *x, const int32_t *xgsum, int I) {
    float acc = 0;
    for (int g = 0; g*32 < I; g++) {
        int32_t d = 0;
        for (int j = 0; j < 32 && g*32+j < I; j++) {
            int e = g*32+j;
            uint8_t b = w4[e>>1];
            int v = (e & 1) ? (b>>4) : (b & 0xF);
            d += (v - 8) * x[e];
        }
        (void)xgsum;
        acc += scl[g] * (float)d;
    }
    return acc;
}
#endif

#endif /* HW_QUANT_H */
