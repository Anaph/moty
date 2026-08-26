/* hw_avx2.h — AVX2 + AVX-VNNI backend.
 * Implements dot_i8i8 via vpmaddubsw (AVX2) or vpdpbusd (AVX-VNNI).
 * dot_i4i8 via nibble unpack + vpmaddubsw or vpdpbusd.
 * Also handles AVX-VNNI 128-bit variant.
 */
#ifndef HW_AVX2_H
#define HW_AVX2_H

/* ---- hsum helpers ---- */
#ifdef __AVX2__
static inline int simd_hsum256_i32(__m256i v) {
    __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
    lo=_mm_add_epi32(lo,hi); lo=_mm_hadd_epi32(lo,lo); lo=_mm_hadd_epi32(lo,lo);
    return _mm_cvtsi128_si32(lo);
}
static inline float simd_hsum256_f32(__m256 v) {
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#endif
#if defined(__AVXVNNI__) && defined(__AVX2__)
static inline int simd_hsum128_i32(__m128i v) {
    v=_mm_hadd_epi32(v,v); v=_mm_hadd_epi32(v,v); return _mm_cvtsi128_si32(v);
}
#endif

/* ---- dot_i8i8 ---- */
#if defined(__AVXVNNI__) && defined(__AVX2__)
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int n) {
    int32_t sum=0; int i=0;
    __m128i acc=_mm_setzero_si128();
    for(; i+16<=n; i+=16){
        __m128i wv=_mm_loadu_si128((const __m128i*)(w+i));
        __m128i xv=_mm_loadu_si128((const __m128i*)(x+i));
        __m128i xs=_mm_sign_epi8(xv,wv);
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(wv),xs);
    }
    sum=simd_hsum128_i32(acc);
    for(; i<n; i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}
#elif defined(__AVX2__)
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int n) {
    int32_t sum=0; int i=0;
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i acc=_mm256_setzero_si256();
    for(; i+32<=n; i+=32){
        __m256i wv=_mm256_loadu_si256((const __m256i*)(w+i));
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=simd_hsum256_i32(acc);
    for(; i<n; i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}
#endif

/* ---- dot_i4i8 ---- */
#if defined(__AVXVNNI__) && defined(__AVX2__)
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum=0; int i=0;
    const __m128i m4=_mm_set1_epi8(0x0F); const __m128i b8=_mm_set1_epi8(8);
    __m128i acc=_mm_setzero_si128();
    for(; i+32<=I; i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);
        __m128i w0=_mm_sub_epi8(n0,b8), w1=_mm_sub_epi8(n1,b8);
        __m128i x0=_mm_loadu_si128((const __m128i*)(x+i));
        __m128i x1=_mm_loadu_si128((const __m128i*)(x+i+16));
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(w0),_mm_sign_epi8(x0,w0));
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(w1),_mm_sign_epi8(x1,w1));
    }
    sum=simd_hsum128_i32(acc);
    for(; i+1<I; i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}
#elif defined(__AVX2__)
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum=0; int i=0;
    const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi8(8);
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i acc=_mm256_setzero_si256();
    for(; i+32<=I; i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);
        __m256i wv=_mm256_sub_epi8(_mm256_set_m128i(n1,n0),b8);
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=simd_hsum256_i32(acc);
    for(; i+1<I; i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}
#endif

/* ---- dot_f32 + dot_f32i8 ---- */
static inline float dot_f32(const float *a, const float *b, int n) {
    float s=0; int i=0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc=_mm256_setzero_ps();
    for(; i+8<=n; i+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),acc);
    s=simd_hsum256_f32(acc);
#endif
    for(; i<n; i++) s+=a[i]*b[i];
    return s;
}

static inline float dot_f32i8(const float *x, const int8_t *w, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * (float)w[i];
    return s;
}

#endif /* HW_AVX2_H */
