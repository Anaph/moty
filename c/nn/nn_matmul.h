/* nn_matmul.h — GEMV/GEMM kernels: f32, int8 (IDOT), int4 (IDOT4/f32), int4-grouped, int2.
 *
 * Depends on: hw.h (dot_i8i8, dot_i4i8, dot_f32), nn_alloc.h (grow, falloc).
 * nn_quant.h is NOT required (matmul reads already-packed weights).
 *
 * IDOT path (int8): quantize x once, then VPDPBUSD per weight row.
 * IDOT4 path (int4): same but with nibble unpack (opt-in via IDOT4=1).
 * f32 path: exact dequant-on-fly for int4/int2.
 */
#ifndef NN_MATMUL_H
#define NN_MATMUL_H

/* M1: hw.h non diffonde piu' le intrinsics — chi le usa se le include */
#if defined(__AVX2__)
  #include <immintrin.h>
  static inline float simd_hsum256_f32(__m256 v) {
      __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
      lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
      sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
  }
#endif

#define NN_QROW_MAX 16384

/* ---- f32 ---- */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++)
            y[(int64_t)s * O + o] = dot_f32(x + (int64_t)s*I, w, I);
    }
}

/* ---- int8 + per-row scale (Q8_0 style) with IDOT ---- */
static void matmul_q_s(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O) {
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I <= NN_QROW_MAX) {
        static int8_t *xi = NULL; static float *sx = NULL;
        static int64_t xcap = 0, scap = 0;
        grow((void **)&xi, &xcap, (int64_t)S*I, 1, "attivazioni int8");
        grow((void **)&sx, &scap, S, sizeof(float), "scale attivazioni");
        for (int s = 0; s < S; s++) sx[s] = qrow_i8(x + (int64_t)s*I, xi + (int64_t)s*I, I);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o*I;
            for (int s = 0; s < S; s++)
                y[(int64_t)s*O + o] = scale[o] * sx[s] * (float)dot_i8i8(w, xi + (int64_t)s*I, I);
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s*I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * (float)w[i];
            y[(int64_t)s*O + o] = acc * scale[o];
        }
    }
}

static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    matmul_q_s(y, x, q, scale, 1, I, O);
}

/* ---- int4 + per-row scale with IDOT4 ---- */
static void matmul_i4_s(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O) {
    static int idot4 = -1;
    if (idot4 < 0) { const char *e = getenv("IDOT4"); idot4 = e ? atoi(e) : 0; }
    int rb = (I+1)/2;
    if (idot4 && I <= NN_QROW_MAX) {
        static int8_t *xi = NULL; static float *sx = NULL;
        static int64_t xcap = 0, scap = 0;
        grow((void **)&xi, &xcap, (int64_t)S*I, 1, "idot4 xi");
        grow((void **)&sx, &scap, S, sizeof(float), "idot4 sx");
        for (int s = 0; s < S; s++) sx[s] = qrow_i8(x + (int64_t)s*I, xi + (int64_t)s*I, I);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *w4 = q4 + (int64_t)o*rb;
            for (int s = 0; s < S; s++)
                y[(int64_t)s*O + o] = scale[o] * sx[s] * (float)dot_i4i8(w4, xi + (int64_t)s*I, I);
        }
        return;
    }
    /* f32×int4 dequant-on-fly */
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o*rb; float sc = scale[o];
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s*I; float a = 0; int i = 0;
#if defined(__AVX2__)
            const __m128i m4 = _mm_set1_epi8(0x0F); const __m256i b8 = _mm256_set1_epi32(8);
            __m256 acc = _mm256_setzero_ps();
            for (; i+16 <= I; i += 16) {
                __m128i by = _mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                __m128i lo = _mm_and_si128(by,m4), hi = _mm_and_si128(_mm_srli_epi16(by,4),m4);
                __m128i nib = _mm_unpacklo_epi8(lo,hi);
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc);
            }
            a = simd_hsum256_f32(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m4=vdup_n_u8(0x0F); const int8x8_t b8=vdup_n_s8(8);
            float32x4_t ac0=vdupq_n_f32(0),ac1=vdupq_n_f32(0);
            for(; i+16<=I; i+=16){
                uint8x8_t by=vld1_u8(w+(i>>1));
                uint8x8x2_t z=vzip_u8(vand_u8(by,m4),vshr_n_u8(by,4));
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[0]),b8));
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[1]),b8));
                ac0=vfmaq_f32(ac0,vld1q_f32(xs+i),   vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1,vld1q_f32(xs+i+4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0,vld1q_f32(xs+i+8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1,vld1q_f32(xs+i+12),vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))));
            }
            a=neon_hsum_f32(vaddq_f32(ac0,ac1));
#endif
            for (; i+1 < I; i += 2) {
                uint8_t byte = w[i>>1]; int lo = (int)(byte&0xF)-8, hi = (int)(byte>>4)-8;
                a += xs[i]*(float)lo + xs[i+1]*(float)hi;
            }
            if (i < I) { uint8_t byte = w[i>>1]; a += xs[i]*(float)((int)(byte&0xF)-8); }
            y[(int64_t)s*O + o] = a*sc;
        }
    }
}

/* ---- int4 + per-group scale ---- */
static void matmul_i4_grouped_s(float *y, const float *x, const uint8_t *q4, const float *scale,
                                int S, int I, int O, int gs) {
    int rb = (I+1)/2, ng = (I+gs-1)/gs;
    /* IDOT4: VNNI per gruppo (dot_i4g8p). x quantizzato int8 per riga una
     * volta (stessa doppia-quant degli expert, validata A/B sui token). */
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
    {
        static int idot4 = -1;
        if (idot4 < 0) { const char *e = getenv("IDOT4"); idot4 = e ? atoi(e) : 0; }
        if (idot4 && gs == 32 && (I & 63) == 0 && I <= NN_QROW_MAX) {
            static int8_t *xi = NULL; static int32_t *xg = NULL; static float *sx = NULL;
            static int64_t xic = 0, xgc = 0, sxc = 0;
            grow((void**)&xi, &xic, (int64_t)S*I, 1, "g4 xi");
            grow((void**)&xg, &xgc, (int64_t)S*ng, 4, "g4 xg");
            grow((void**)&sx, &sxc, S, 4, "g4 sx");
            for (int s = 0; s < S; s++) {
                sx[s] = qrow_i8(x + (int64_t)s*I, xi + (int64_t)s*I, I);
                for (int g = 0; g < ng; g++) {
                    int32_t a = 0;
                    for (int j = 0; j < 32; j++) a += xi[(int64_t)s*I + g*32+j];
                    xg[(int64_t)s*ng + g] = a;
                }
            }
            {   /* regione esplicita: ogni thread prefetcha la testa del SUO
                 * chunk (ramp ~15-30us su matrici piccole da 2-6MB) */
                int nth = omp_get_max_threads();
                #pragma omp parallel
                {
                    int t = omp_get_thread_num();
                    int o0 = (int)((int64_t)O * t / nth);
                    const uint8_t *pf = q4 + (int64_t)o0*rb;
                    for (int k = 0; k < 512 && k < (O-o0)*rb; k += 64)
                        __builtin_prefetch(pf + k, 0, 3);
                    #pragma omp for schedule(static)
                    for (int o = 0; o < O; o++) {
                        const uint8_t *wr = q4 + (int64_t)o*rb;
                        const float *sr = scale + (int64_t)o*ng;
                        for (int s = 0; s < S; s++)
                            y[(int64_t)s*O + o] = sx[s] * dot_i4g8p(wr, sr, xi + (int64_t)s*I, xg + (int64_t)s*ng, I);
                    }
                }
            }
            return;
        }
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (int64_t)o*rb;
        const float *scl = scale + (int64_t)o*ng;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s*I; float a = 0;
            for (int g = 0; g*gs < I; g++) {
                int base = g*gs, glen = gs; if (base+glen > I) glen = I-base;
                float sc = scl[g]; int i = base;
                int done = 0;
#if defined(__AVX512F__) && defined(__AVX512VL__)
                /* AVX512: gs=32 → un intero gruppo per iterazione (16 byte
                 * nibble → 2×__m512 f32), UNA mul di scala a gruppo. */
                if (glen == 32) {
                    const __m128i m4 = _mm_set1_epi8(0x0F);
                    const __m512i b8 = _mm512_set1_epi32(8);
                    __m128i by = _mm_loadu_si128((const __m128i*)(w+(i>>1)));
                    __m128i lo = _mm_and_si128(by, m4), hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
                    __m128i n0 = _mm_unpacklo_epi8(lo, hi), n1 = _mm_unpackhi_epi8(lo, hi);
                    __m512 w0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n0), b8));
                    __m512 w1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n1), b8));
                    __m512 s0 = _mm512_mul_ps(_mm512_loadu_ps(xs+i),    w0);
                    __m512 s1 = _mm512_mul_ps(_mm512_loadu_ps(xs+i+16), w1);
                    a += _mm512_reduce_add_ps(_mm512_add_ps(s0, s1)) * sc;
                    i += 32; done = 1;
                }
#endif
                if (!done) {
#if defined(__AVX2__)
                const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
                __m256 acc=_mm256_setzero_ps();
                for(;i+16<=base+glen;i+=16){
                    __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                    __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i nib=_mm_unpacklo_epi8(lo,hi);
                    __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                    __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),  w0,acc);
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8),w1,acc);
                }
                a+=simd_hsum256_f32(acc)*sc;
#endif
                }
                for(;i<base+glen;i+=2){
                    if(i+1<base+glen){uint8_t byte=w[i>>1];
                        a+=(xs[i]*(float)((int)(byte&0xF)-8)+xs[i+1]*(float)((int)(byte>>4)-8))*sc;
                    }else{uint8_t byte=w[i>>1];a+=xs[i]*(float)((int)(byte&0xF)-8)*sc;}
                }
            }
            y[(int64_t)s*O + o] = a;
        }
    }
}

/* ---- int2 + per-row scale ---- */
static void matmul_i2_s(float *y, const float *x, const uint8_t *q2, const float *scale, int S, int I, int O) {
    int rb = (I+3)/4;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q2 + (int64_t)o*rb; float sc = scale[o];
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s*I; float a = 0; int i = 0;
#ifdef __AVX2__
            const __m128i m2=_mm_set1_epi8(0x03); const __m256i b2=_mm256_set1_epi32(2);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){
                __m128i by=_mm_cvtsi32_si128(*(const int*)(w+(i>>2)));
                __m128i p0=_mm_and_si128(by,m2),p1=_mm_and_si128(_mm_srli_epi16(by,2),m2);
                __m128i p2=_mm_and_si128(_mm_srli_epi16(by,4),m2),p3=_mm_and_si128(_mm_srli_epi16(by,6),m2);
                __m128i lo=_mm_unpacklo_epi8(p0,p1),hi=_mm_unpacklo_epi8(p2,p3);
                __m128i nib=_mm_unpacklo_epi16(lo,hi);
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b2));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b2));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),  w0,acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8),w1,acc);
            }
            a=simd_hsum256_f32(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m2v=vdup_n_u8(3); const int8x8_t b2v=vdup_n_s8(2);
            float32x4_t ac0=vdupq_n_f32(0),ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){
                uint32_t wd; memcpy(&wd,w+(i>>2),4);
                uint8x8_t by=vreinterpret_u8_u32(vdup_n_u32(wd));
                uint8x8x2_t z01=vzip_u8(vand_u8(by,m2v),vand_u8(vshr_n_u8(by,2),m2v));
                uint8x8x2_t z23=vzip_u8(vand_u8(vshr_n_u8(by,4),m2v),vshr_n_u8(by,6));
                uint16x4x2_t zz=vzip_u16(vreinterpret_u16_u8(z01.val[0]),vreinterpret_u16_u8(z23.val[0]));
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[0]),b2v));
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[1]),b2v));
                ac0=vfmaq_f32(ac0,vld1q_f32(xs+i),   vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1,vld1q_f32(xs+i+4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0,vld1q_f32(xs+i+8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1,vld1q_f32(xs+i+12),vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))));
            }
            a=neon_hsum_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++){uint8_t byte=w[i>>2];int sh=(i&3)*2;a+=xs[i]*(float)((int)((byte>>sh)&3)-2);}
            y[(int64_t)s*O+o]=a*sc;
        }
    }
}

/* Q4_K/Q6_K native matmul: dequant-on-fly using gguf_dq_* (need gguf.h before nn.h) */
#ifdef GGUF_H
static void matmul_q4k_native(float *y, const float *x, const uint8_t *raw,
                              int S, int I, int O) {
    int nblk = I / 256;
    /* dequant ONCE per (row, block), dot against ALL S tokens */
    #pragma omp parallel
    {
        float *acc = malloc((size_t)S * sizeof(float));
        float wtmp[256];
        #pragma omp for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *blk = raw + (int64_t)o * nblk * 144;
            for (int s = 0; s < S; s++) acc[s] = 0;
            for (int b = 0; b < nblk; b++, blk += 144) {
                gguf_dq_q4k(blk, 1, wtmp);
                for (int s = 0; s < S; s++) {
                    const float *xs = x + (int64_t)s * I + b * 256;
                    float a = 0;
                    for (int l = 0; l < 256; l++) a += xs[l] * wtmp[l];
                    acc[s] += a;
                }
            }
            for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = acc[s];
        }
        free(acc);
    }
}

static void matmul_q6k_native(float *y, const float *x, const uint8_t *raw,
                              int S, int I, int O) {
    int nblk = I / 256;
    #pragma omp parallel
    {
        float *acc = malloc((size_t)S * sizeof(float));
        float wtmp[256];
        #pragma omp for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *blk = raw + (int64_t)o * nblk * 210;
            for (int s = 0; s < S; s++) acc[s] = 0;
            for (int b = 0; b < nblk; b++, blk += 210) {
                gguf_dq_q6k(blk, 1, wtmp);
                for (int s = 0; s < S; s++) {
                    const float *xs = x + (int64_t)s * I + b * 256;
                    float a = 0;
                    for (int l = 0; l < 256; l++) a += xs[l] * wtmp[l];
                    acc[s] += a;
                }
            }
            for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = acc[s];
        }
        free(acc);
    }
}
#endif /* GGUF_H */

#endif /* NN_MATMUL_H */
