/* hw_deltanet.h — DeltaNet recurrent kernels.
 * Parameterized by dot_f32 (provided by the active backend).
 * These are the same across all CPU backends; a GPU backend would
 * provide its own implementation.
 */
#ifndef HW_DELTANET_H
#define HW_DELTANET_H

/* S[i*dv + j] *= dec; kv[j] += S[i*dv + j] * ki */
static inline void dn_row_decay_acc(float *restrict S, float dec, float ki,
                                     float *restrict kv, int dv) {
    int j = 0;
#if defined(__AVX512F__)
    __m512 decv = _mm512_set1_ps(dec), kiv = _mm512_set1_ps(ki);
    for (; j + 16 <= dv; j += 16) {
        __m512 sv = _mm512_loadu_ps(S + j);
        sv = _mm512_mul_ps(sv, decv);
        _mm512_storeu_ps(S + j, sv);
        __m512 kvv = _mm512_loadu_ps(kv + j);
        kvv = _mm512_fmadd_ps(sv, kiv, kvv);
        _mm512_storeu_ps(kv + j, kvv);
    }
#elif defined(__ARM_NEON)
    float32x4_t decv = vdupq_n_f32(dec), kiv = vdupq_n_f32(dec);
    for (; j + 4 <= dv; j += 4) {
        float32x4_t sv = vld1q_f32(S + j);
        sv = vmulq_f32(sv, decv);
        vst1q_f32(S + j, sv);
        float32x4_t kvv = vld1q_f32(kv + j);
        kvv = vmlaq_f32(kvv, sv, kiv);
        vst1q_f32(kv + j, kvv);
    }
#endif
    for (; j < dv; j++) { S[j] *= dec; kv[j] += S[j] * ki; }
}

/* S[i*dv + j] += ki * delta[j]; oh[j] += S[i*dv + j] * qi */
static inline void dn_row_update_dot(float *restrict S, float ki,
                                     const float *restrict delta, float qi,
                                     float *restrict oh, int dv) {
    int j = 0;
#if defined(__AVX512F__)
    __m512 kiv = _mm512_set1_ps(ki), qiv = _mm512_set1_ps(qi);
    for (; j + 16 <= dv; j += 16) {
        __m512 sv = _mm512_loadu_ps(S + j);
        __m512 dv_ = _mm512_loadu_ps(delta + j);
        sv = _mm512_fmadd_ps(kiv, dv_, sv);
        _mm512_storeu_ps(S + j, sv);
        __m512 ohv = _mm512_loadu_ps(oh + j);
        ohv = _mm512_fmadd_ps(sv, qiv, ohv);
        _mm512_storeu_ps(oh + j, ohv);
    }
#elif defined(__ARM_NEON)
    float32x4_t kiv = vdupq_n_f32(ki), qiv = vdupq_n_f32(qi);
    for (; j + 4 <= dv; j += 4) {
        float32x4_t sv = vld1q_f32(S + j);
        float32x4_t dv_ = vld1q_f32(delta + j);
        sv = vmlaq_f32(sv, kiv, dv_);
        vst1q_f32(S + j, sv);
        float32x4_t ohv = vld1q_f32(oh + j);
        ohv = vmlaq_f32(ohv, sv, qiv);
        vst1q_f32(oh + j, ohv);
    }
#endif
    for (; j < dv; j++) { S[j] += ki * delta[j]; oh[j] += S[j] * qi; }
}

#endif /* HW_DELTANET_H */
