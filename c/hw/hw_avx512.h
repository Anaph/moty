/* hw_avx512.h — AVX512-VNNI + AVX512-BW backend.
 *
 * Implements: dot_i8i8 (VPDPBUSD, 64 int8/iter), dot_i4i8 (nibble unpack +
 * VPDPBUSD), dot_f32 (FMA), dot_f32i8 (dequant + FMA).
 * Multi-accumulator (4×256 elem/outer iter) for VPDPBUSD latency hiding.
 *
 * This file is included by hw.h when __AVX512VNNI__ && __AVX512BW__ are
 * both defined. It expects <immintrin.h> to already be in scope.
 */
#ifndef HW_AVX512_H
#define HW_AVX512_H
#include <string.h>

/* ---- hsum helpers (also used by nn_gemm.h matmul_i4_s AVX2 branch) ---- */
static inline float simd_hsum256_f32(__m256 v) {
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}

/* ---- dot_i8i8: int8 × int8 → int32 via VPDPBUSD ----
 * CONTRACT: x[i] must be in [-127, 127] (symmetric quant, e.g. qrow_i8).
 * The sign trick (|w|, sign(w)*x) overflows int8 when x == -128. ---- */
int32_t moty_hw_dot_i8i8(const int8_t *w, const int8_t *x, int n) {
    int32_t sum = 0; int i = 0;
    __m512i a0=_mm512_setzero_si512(),a1=_mm512_setzero_si512(),
            a2=_mm512_setzero_si512(),a3=_mm512_setzero_si512();
    for (; i+256 <= n; i += 256) {
        __m512i wv, xv, xs; __mmask64 neg;
        wv=_mm512_loadu_si512(w+i);     xv=_mm512_loadu_si512(x+i);
        neg=_mm512_movepi8_mask(wv);    xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        a0=_mm512_dpbusd_epi32(a0,_mm512_abs_epi8(wv),xs);
        wv=_mm512_loadu_si512(w+i+64);  xv=_mm512_loadu_si512(x+i+64);
        neg=_mm512_movepi8_mask(wv);    xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        a1=_mm512_dpbusd_epi32(a1,_mm512_abs_epi8(wv),xs);
        wv=_mm512_loadu_si512(w+i+128); xv=_mm512_loadu_si512(x+i+128);
        neg=_mm512_movepi8_mask(wv);    xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        a2=_mm512_dpbusd_epi32(a2,_mm512_abs_epi8(wv),xs);
        wv=_mm512_loadu_si512(w+i+192); xv=_mm512_loadu_si512(x+i+192);
        neg=_mm512_movepi8_mask(wv);    xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        a3=_mm512_dpbusd_epi32(a3,_mm512_abs_epi8(wv),xs);
    }
    __m512i acc=_mm512_add_epi32(_mm512_add_epi32(a0,a1),_mm512_add_epi32(a2,a3));
    for (; i+64 <= n; i += 64) {
        __m512i wv=_mm512_loadu_si512(w+i), xv=_mm512_loadu_si512(x+i);
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
    for (; i+1 < n; i += 2) sum += w[i]*x[i] + w[i+1]*x[i+1];
    if (i < n) sum += w[i]*x[i];
    return sum;
}

/* ---- chunk helpers (reused by tiled GEMV callers) ---- */
static inline __m512i i8i8_dpbusd_chunk(const int8_t *w, const int8_t *x, __m512i acc) {
    __m512i wv=_mm512_loadu_si512((const void*)w), xv=_mm512_loadu_si512((const void*)x);
    __mmask64 neg=_mm512_movepi8_mask(wv);
    __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
    return _mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
}
static inline __m512i i4i8_dpbusd_chunk(const uint8_t *w4, const int8_t *x,
                                         __m256i m4v, __m512i b8v, __m512i xidx, __m512i acc) {
    __m256i by=_mm256_loadu_si256((const __m256i*)w4);
    __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
    __m256i z0=_mm256_unpacklo_epi8(lo,hi), z1=_mm256_unpackhi_epi8(lo,hi);
    __m512i wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(z0),z1,1),b8v);
    __m512i xv=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)x));
    __mmask64 neg=_mm512_movepi8_mask(wv);
    __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
    return _mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
}
static inline __m512i i4i8_dpbusd_chunk_xv(const uint8_t *w4, __m512i xv,
                                            __m256i m4v, __m512i b8v, __m512i acc) {
    __m256i by=_mm256_loadu_si256((const __m256i*)w4);
    __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
    __m256i z0=_mm256_unpacklo_epi8(lo,hi), z1=_mm256_unpackhi_epi8(lo,hi);
    __m512i wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(z0),z1,1),b8v);
    __mmask64 neg=_mm512_movepi8_mask(wv);
    __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
    return _mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
}

/* ---- dot_i4i8p: int4(nibble, 0..15) × int8 con x PRE-PERMUTATO.
 * Scorciatoia algebrica: Σ(n_j-8)x_j = Σn_j·x_j - 8·Σx_j → niente
 * sub/mask/abs nel loop; VPDPBUSD(u8, s8) diretto sui nibble.
 * xp deve essere l'attivazione quantizzata permuted per granuli-64
 * [0,1,4,5,2,3,6,7] (px_permute()), sxsum = Σx[j] (scalare, pre-calcolato). */
void moty_hw_px_permute(const int8_t *x, int8_t *xp, int I) {
    int i = 0;
    for (; i + 64 <= I; i += 64) {
        const int8_t *p = x + i;
        int8_t *q = xp + i;
        /* ordine granuli dell'unpack nibble: [0-15, 32-47, 16-31, 48-63] */
        memcpy(q,      p,     16);
        memcpy(q + 16, p+32,  16);
        memcpy(q + 32, p+16,  16);
        memcpy(q + 48, p+48,  16);
    }
    if (i < I) memcpy(xp + i, x + i, I - i);   /* residuo: identita' */
}
int32_t moty_hw_px_sum(const int8_t *x, int I) {
    int32_t s = 0; for (int i = 0; i < I; i++) s += x[i]; return s;
}
int32_t moty_hw_dot_i4i8p(const uint8_t *w4, const int8_t *xp, int32_t sxsum, int I) {
    int32_t sum = 0; int i = 0;
    const __m256i m4v=_mm256_set1_epi8(0x0F);
    __m512i a0=_mm512_setzero_si512(),a1=_mm512_setzero_si512(),
            a2=_mm512_setzero_si512(),a3=_mm512_setzero_si512();
    for (; i+256 <= I; i += 256) {
        __m256i by=_mm256_loadu_si256((const __m256i*)(w4+((i   )>>1)));
        __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        a0=_mm512_dpbusd_epi32(a0,_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi8(lo,hi)),_mm256_unpackhi_epi8(lo,hi),1),
                               _mm512_loadu_si512((const void*)(xp+i)));
        by=_mm256_loadu_si256((const __m256i*)(w4+((i+64 )>>1)));
        lo=_mm256_and_si256(by,m4v); hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        a1=_mm512_dpbusd_epi32(a1,_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi8(lo,hi)),_mm256_unpackhi_epi8(lo,hi),1),
                               _mm512_loadu_si512((const void*)(xp+i+64)));
        by=_mm256_loadu_si256((const __m256i*)(w4+((i+128)>>1)));
        lo=_mm256_and_si256(by,m4v); hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        a2=_mm512_dpbusd_epi32(a2,_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi8(lo,hi)),_mm256_unpackhi_epi8(lo,hi),1),
                               _mm512_loadu_si512((const void*)(xp+i+128)));
        by=_mm256_loadu_si256((const __m256i*)(w4+((i+192)>>1)));
        lo=_mm256_and_si256(by,m4v); hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        a3=_mm512_dpbusd_epi32(a3,_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi8(lo,hi)),_mm256_unpackhi_epi8(lo,hi),1),
                               _mm512_loadu_si512((const void*)(xp+i+192)));
    }
    __m512i acc=_mm512_add_epi32(_mm512_add_epi32(a0,a1),_mm512_add_epi32(a2,a3));
    for (; i+64 <= I; i += 64) {
        __m256i by=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)));
        __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        acc=_mm512_dpbusd_epi32(acc,_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi8(lo,hi)),_mm256_unpackhi_epi8(lo,hi),1),
                                _mm512_loadu_si512((const void*)(xp+i)));
    }
    sum=_mm512_reduce_add_epi32(acc);
    /* coda scalare: px_permute copia il residuo identita' → xp[j]==x[j];
     * il -8·Σx va solo sul prefisso SIMD, quindi si toglie la coda da sxsum.
     * Nibble: elem pari = low, dispari = high (stessa convenzione di dot_i4i8). */
    int32_t sc = 0, tailx = 0;
    for (int j = i; j < I; j++) {
        uint8_t b = w4[j>>1];
        int v = (j & 1) ? (int)(b>>4) : (int)(b & 0xF);
        sc += (v - 8) * xp[j];
        tailx += xp[j];
    }
    return sum - 8*(sxsum - tailx) + sc;
}

/* ---- dot_i4g8p: int4 GROUPED (gs=32) × int8 con scale per gruppo.
 * Per chunk da 64 elem = 2 gruppi: l'unpack per-128 mantiene i gruppi
 * sequenziali → G0 = lanes 0-7, G1 = lanes 8-15 → masked reduce.
 * Trucco nibble-offset: Σ(n-8)x = Σn·x - 8·Σx per gruppo (xgsum[g]).
 * x NON serve permuted (carico diretto). I multiplo di 64, gs=32. */
float moty_hw_dot_i4g8p(const uint8_t *w4, const float *scl,
                              const int8_t *x, const int32_t *xgsum, int I) {
    float acc = 0;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    for (int i = 0; i + 64 <= I; i += 64) {
        __m128i byA = _mm_loadu_si128((const __m128i*)(w4 + (i>>1)));
        __m128i loA = _mm_and_si128(byA, m4), hiA = _mm_and_si128(_mm_srli_epi16(byA, 4), m4);
        __m256i G0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(loA, hiA)),
                                             _mm_unpackhi_epi8(loA, hiA), 1);   /* elems 0-31 */
        __m128i byB = _mm_loadu_si128((const __m128i*)(w4 + (i>>1) + 16));
        __m128i loB = _mm_and_si128(byB, m4), hiB = _mm_and_si128(_mm_srli_epi16(byB, 4), m4);
        __m256i G1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(loB, hiB)),
                                             _mm_unpackhi_epi8(loB, hiB), 1);   /* elems 32-63 */
        __m512i wv = _mm512_inserti64x4(_mm512_castsi256_si512(G0), G1, 1);
        __m512i dv = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wv,
                                         _mm512_loadu_si512((const void*)(x+i)));
        int g = i >> 5;
        acc += scl[g]   * (float)(_mm512_mask_reduce_add_epi32(0x00FF, dv) - 8*xgsum[g]);
        acc += scl[g+1] * (float)(_mm512_mask_reduce_add_epi32(0xFF00, dv) - 8*xgsum[g+1]);
    }
    return acc;
}

/* ---- dot_i4i8: int4 × int8 → int32 via nibble unpack + VPDPBUSD ----
 * 4 accumulator chains (latency hiding: vpdpbusd lat ~4-5cyc; una sola
 * catena seria 32 iter = ~160cyc → throughput-bound al 25%). Ogni chunk
 * ha la PROPRIA attivazione xv0..xv3 (bug storico: xv condiviso). */
int32_t moty_hw_dot_i4i8(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum = 0; int i = 0;
    const __m256i m4v=_mm256_set1_epi8(0x0F);
    const __m512i b8v=_mm512_set1_epi8(8);
    const __m512i xidx=_mm512_setr_epi64(0,1,4,5,2,3,6,7);
    __m512i a0=_mm512_setzero_si512(),a1=_mm512_setzero_si512(),
            a2=_mm512_setzero_si512(),a3=_mm512_setzero_si512();
    for (; i+256 <= I; i += 256) {
        __m512i xv0=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)(x+i)));
        __m512i xv1=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)(x+i+64)));
        __m512i xv2=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)(x+i+128)));
        __m512i xv3=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)(x+i+192)));
        __m256i by=_mm256_loadu_si256((const __m256i*)(w4+((i   )>>1)));
        __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        __m512i wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi8(lo,hi)),_mm256_unpackhi_epi8(lo,hi),1),b8v);
        __mmask64 neg=_mm512_movepi8_mask(wv);
        a0=_mm512_dpbusd_epi32(a0,_mm512_abs_epi8(wv),_mm512_mask_sub_epi8(xv0,neg,_mm512_setzero_si512(),xv0));
        by=_mm256_loadu_si256((const __m256i*)(w4+((i+64 )>>1)));
        lo=_mm256_and_si256(by,m4v); hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi8(lo,hi)),_mm256_unpackhi_epi8(lo,hi),1),b8v);
        neg=_mm512_movepi8_mask(wv);
        a1=_mm512_dpbusd_epi32(a1,_mm512_abs_epi8(wv),_mm512_mask_sub_epi8(xv1,neg,_mm512_setzero_si512(),xv1));
        by=_mm256_loadu_si256((const __m256i*)(w4+((i+128)>>1)));
        lo=_mm256_and_si256(by,m4v); hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi8(lo,hi)),_mm256_unpackhi_epi8(lo,hi),1),b8v);
        neg=_mm512_movepi8_mask(wv);
        a2=_mm512_dpbusd_epi32(a2,_mm512_abs_epi8(wv),_mm512_mask_sub_epi8(xv2,neg,_mm512_setzero_si512(),xv2));
        by=_mm256_loadu_si256((const __m256i*)(w4+((i+192)>>1)));
        lo=_mm256_and_si256(by,m4v); hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi8(lo,hi)),_mm256_unpackhi_epi8(lo,hi),1),b8v);
        neg=_mm512_movepi8_mask(wv);
        a3=_mm512_dpbusd_epi32(a3,_mm512_abs_epi8(wv),_mm512_mask_sub_epi8(xv3,neg,_mm512_setzero_si512(),xv3));
    }
    __m512i acc=_mm512_add_epi32(_mm512_add_epi32(a0,a1),_mm512_add_epi32(a2,a3));
    for (; i+64 <= I; i += 64) {
        __m256i by=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)));
        __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        __m512i wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi8(lo,hi)),_mm256_unpackhi_epi8(lo,hi),1),b8v);
        __m512i xv=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)(x+i)));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv));
    }
    sum=_mm512_reduce_add_epi32(acc);
    for (; i+1 < I; i += 2) { uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if (i < I) { uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}

/* ---- dot_f32: f32 × f32 → f32 via FMA ---- */
float moty_hw_dot_f32(const float *a, const float *b, int n) {
    float s = 0; int i = 0;
    __m512 s0=_mm512_setzero_ps(), s1=_mm512_setzero_ps();
    for (; i+32 <= n; i += 32) {
        s0=_mm512_fmadd_ps(_mm512_loadu_ps(a+i),    _mm512_loadu_ps(b+i),    s0);
        s1=_mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), s1);
    }
    for (; i+16 <= n; i += 16)
        s0=_mm512_fmadd_ps(_mm512_loadu_ps(a+i),_mm512_loadu_ps(b+i),s0);
    s=_mm512_reduce_add_ps(_mm512_add_ps(s0,s1));
    for (; i < n; i++) s += a[i]*b[i];
    return s;
}

/* ---- dot_f32i8: f32 × int8 → f32 (dequant + FMA) ---- */
float moty_hw_dot_f32i8(const float *x, const int8_t *w, int n) {
    float s = 0; int i = 0;
    __m512 acc = _mm512_setzero_ps();
    for (; i+16 <= n; i += 16) {
        __m512i wi = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w+i)));
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(x+i), _mm512_cvtepi32_ps(wi), acc);
    }
    s = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) s += x[i] * (float)w[i];
    return s;
}

#endif /* HW_AVX512_H */
