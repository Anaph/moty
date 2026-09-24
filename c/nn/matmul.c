/* matmul.c — M3 libmoty-nn: UNA copia dei GEMV/GEMM.
 * Include io/gguf.h: i path nativi Q4_K/Q6_K sono sempre presenti nella lib
 * (nei motori prima erano condizionali a #ifdef GGUF_H). */
#include "nn/nn_alloc.h"
#include "hw/hw.h"
#include "io/gguf.h"
#include "nn/nn_matmul.h"

/* one context for the row loops below: each fills y[s][o] for rows [o0, o1) */
typedef struct {
    float *y; const float *x; const float *W; const int8_t *q; const uint8_t *q4; const float *scale;
    const int8_t *xi; const float *sx; const int32_t *xg; int S, I, O, gs;
} RowJob;

static void f32_rows(void *c_, int64_t o0, int64_t o1, int tid) {
    const RowJob *c = c_; int S = c->S, I = c->I, O = c->O;
    for (int o = (int)o0; o < (int)o1; o++) {
        const float *w = c->W + (int64_t)o * I;
        for (int s = 0; s < S; s++)
            c->y[(int64_t)s * O + o] = dot_f32(c->x + (int64_t)s*I, w, I);
    }
}

void moty_matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    RowJob c = { .y = y, .x = x, .W = W, .S = S, .I = I, .O = O };
    moty_par_for(O, 0, f32_rows, &c);
}

static void q8_idot_rows(void *c_, int64_t o0, int64_t o1, int tid) {
    const RowJob *c = c_; int S = c->S, I = c->I, O = c->O;
    for (int o = (int)o0; o < (int)o1; o++) {
        const int8_t *w = c->q + (int64_t)o*I;
        for (int s = 0; s < S; s++)
            c->y[(int64_t)s*O + o] = c->scale[o] * c->sx[s] * (float)dot_i8i8(w, c->xi + (int64_t)s*I, I);
    }
}

static void q8_f32_rows(void *c_, int64_t o0, int64_t o1, int tid) {
    const RowJob *c = c_; int S = c->S, I = c->I, O = c->O;
    for (int o = (int)o0; o < (int)o1; o++) {
        const int8_t *w = c->q + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = c->x + (int64_t)s*I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * (float)w[i];
            c->y[(int64_t)s*O + o] = acc * c->scale[o];
        }
    }
}

void moty_matmul_q_s(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O) {
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I <= NN_QROW_MAX) {
        static int8_t *xi = NULL; static float *sx = NULL;
        static int64_t xcap = 0, scap = 0;
        grow((void **)&xi, &xcap, (int64_t)S*I, 1, "attivazioni int8");
        grow((void **)&sx, &scap, S, sizeof(float), "scale attivazioni");
        for (int s = 0; s < S; s++) sx[s] = qrow_i8(x + (int64_t)s*I, xi + (int64_t)s*I, I);
        RowJob c = { .y = y, .q = q, .scale = scale, .xi = xi, .sx = sx, .S = S, .I = I, .O = O };
        moty_par_for(O, 0, q8_idot_rows, &c);
        return;
    }
    RowJob c = { .y = y, .x = x, .q = q, .scale = scale, .S = S, .I = I, .O = O };
    moty_par_for(O, 0, q8_f32_rows, &c);
}

void moty_matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    matmul_q_s(y, x, q, scale, 1, I, O);
}

static void i4_idot_rows(void *c_, int64_t o0, int64_t o1, int tid);
static void i4_f32_rows(void *c_, int64_t o0, int64_t o1, int tid);

void moty_matmul_i4_s(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O) {
    static int idot4 = -1;
    if (idot4 < 0) { const char *e = getenv("IDOT4"); idot4 = e ? atoi(e) : 0; }
    if (idot4 && I <= NN_QROW_MAX) {
        static int8_t *xi = NULL; static float *sx = NULL;
        static int64_t xcap = 0, scap = 0;
        grow((void **)&xi, &xcap, (int64_t)S*I, 1, "idot4 xi");
        grow((void **)&sx, &scap, S, sizeof(float), "idot4 sx");
        for (int s = 0; s < S; s++) sx[s] = qrow_i8(x + (int64_t)s*I, xi + (int64_t)s*I, I);
        RowJob c = { .y = y, .q4 = q4, .scale = scale, .xi = xi, .sx = sx, .S = S, .I = I, .O = O };
        moty_par_for(O, 0, i4_idot_rows, &c);
        return;
    }
    /* f32×int4 dequant-on-fly */
    RowJob c = { .y = y, .x = x, .q4 = q4, .scale = scale, .S = S, .I = I, .O = O };
    moty_par_for(O, 0, i4_f32_rows, &c);
}

static void i4_idot_rows(void *c_, int64_t o0, int64_t o1, int tid) {
    const RowJob *c = c_; int S = c->S, I = c->I, O = c->O, rb = (I+1)/2;
    for (int o = (int)o0; o < (int)o1; o++) {
        const uint8_t *w4 = c->q4 + (int64_t)o*rb;
        for (int s = 0; s < S; s++)
            c->y[(int64_t)s*O + o] = c->scale[o] * c->sx[s] * (float)dot_i4i8(w4, c->xi + (int64_t)s*I, I);
    }
}

static void i4_f32_rows(void *c_, int64_t o0, int64_t o1, int tid) {
    const RowJob *c = c_; int S = c->S, I = c->I, O = c->O, rb = (I+1)/2;
    const float *x = c->x; float *y = c->y;
    for (int o = (int)o0; o < (int)o1; o++) {
        const uint8_t *w = c->q4 + (int64_t)o*rb; float sc = c->scale[o];
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

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
static void i4g_vnni_rows(void *c_, int64_t o0, int64_t o1, int tid) {
    const RowJob *c = c_; int S = c->S, I = c->I, O = c->O, rb = (I+1)/2, ng = I/32;
    const uint8_t *pf = c->q4 + o0*rb;
    for (int64_t k = 0; k < 512 && k < (o1-o0)*rb; k += 64)
        __builtin_prefetch(pf + k, 0, 3);
    for (int o = (int)o0; o < (int)o1; o++) {
        const uint8_t *wr = c->q4 + (int64_t)o*rb;
        const float *sr = c->scale + (int64_t)o*ng;
        for (int s = 0; s < S; s++)
            c->y[(int64_t)s*O + o] = c->sx[s] * dot_i4g8p(wr, sr, c->xi + (int64_t)s*I, c->xg + (int64_t)s*ng, I);
    }
}
#endif

static void i4g_f32_rows(void *c_, int64_t o0, int64_t o1, int tid);

void moty_matmul_i4_grouped_s(float *y, const float *x, const uint8_t *q4, const float *scale,
                                int S, int I, int O, int gs) {
    /* IDOT4: VNNI per gruppo (dot_i4g8p). x quantizzato int8 per riga una
     * volta (stessa doppia-quant degli expert, validata A/B sui token). */
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
    {
        static int idot4 = -1;
        if (idot4 < 0) { const char *e = getenv("IDOT4"); idot4 = e ? atoi(e) : 0; }
        if (idot4 && gs == 32 && (I & 63) == 0 && I <= NN_QROW_MAX) {
            int ng = I / 32;
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
            /* static split: each thread prefetches the head of its own
             * range (ramp ~15-30us on small 2-6 MB matrices) */
            RowJob c = { .y = y, .q4 = q4, .scale = scale, .xi = xi, .sx = sx, .xg = xg, .S = S, .I = I, .O = O };
            moty_par_for(O, 0, i4g_vnni_rows, &c);
            return;
        }
    }
#endif
    RowJob c = { .y = y, .x = x, .q4 = q4, .scale = scale, .S = S, .I = I, .O = O, .gs = gs };
    moty_par_for(O, 0, i4g_f32_rows, &c);
}

static void i4g_f32_rows(void *c_, int64_t o0, int64_t o1, int tid) {
    const RowJob *c = c_; int S = c->S, I = c->I, O = c->O, gs = c->gs, rb = (I+1)/2, ng = (I+gs-1)/gs;
    const float *x = c->x; float *y = c->y;
    for (int o = (int)o0; o < (int)o1; o++) {
        const uint8_t *w = c->q4 + (int64_t)o*rb;
        const float *scl = c->scale + (int64_t)o*ng;
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

static void i2_rows(void *c_, int64_t o0, int64_t o1, int tid);

void moty_matmul_i2_s(float *y, const float *x, const uint8_t *q2, const float *scale, int S, int I, int O) {
    RowJob c = { .y = y, .x = x, .q4 = q2, .scale = scale, .S = S, .I = I, .O = O };
    moty_par_for(O, 0, i2_rows, &c);
}

static void i2_rows(void *c_, int64_t o0, int64_t o1, int tid) {
    const RowJob *c = c_; int S = c->S, I = c->I, O = c->O, rb = (I+3)/4;
    const float *x = c->x; float *y = c->y;
    for (int o = (int)o0; o < (int)o1; o++) {
        const uint8_t *w = c->q4 + (int64_t)o*rb; float sc = c->scale[o];
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

static void q4k_rows(void *c_, int64_t o0, int64_t o1, int tid) {
    const RowJob *c = c_; int S = c->S, I = c->I, O = c->O, nblk = I / 256;
    float *acc = malloc((size_t)S * sizeof(float));
    float wtmp[256];
    for (int o = (int)o0; o < (int)o1; o++) {
        const uint8_t *blk = c->q4 + (int64_t)o * nblk * 144;
        for (int s = 0; s < S; s++) acc[s] = 0;
        for (int b = 0; b < nblk; b++, blk += 144) {
            gguf_dq_q4k(blk, 1, wtmp);
            for (int s = 0; s < S; s++) {
                const float *xs = c->x + (int64_t)s * I + b * 256;
                float a = 0;
                for (int l = 0; l < 256; l++) a += xs[l] * wtmp[l];
                acc[s] += a;
            }
        }
        for (int s = 0; s < S; s++) c->y[(int64_t)s * O + o] = acc[s];
    }
    free(acc);
}

void moty_matmul_q4k_native(float *y, const float *x, const uint8_t *raw,
                              int S, int I, int O) {
    /* dequant ONCE per (row, block), dot against ALL S tokens */
    RowJob c = { .y = y, .x = x, .q4 = raw, .S = S, .I = I, .O = O };
    moty_par_for(O, 0, q4k_rows, &c);
}

static void q6k_rows(void *c_, int64_t o0, int64_t o1, int tid) {
    const RowJob *c = c_; int S = c->S, I = c->I, O = c->O, nblk = I / 256;
    float *acc = malloc((size_t)S * sizeof(float));
    float wtmp[256];
    for (int o = (int)o0; o < (int)o1; o++) {
        const uint8_t *blk = c->q4 + (int64_t)o * nblk * 210;
        for (int s = 0; s < S; s++) acc[s] = 0;
        for (int b = 0; b < nblk; b++, blk += 210) {
            gguf_dq_q6k(blk, 1, wtmp);
            for (int s = 0; s < S; s++) {
                const float *xs = c->x + (int64_t)s * I + b * 256;
                float a = 0;
                for (int l = 0; l < 256; l++) a += xs[l] * wtmp[l];
                acc[s] += a;
            }
        }
        for (int s = 0; s < S; s++) c->y[(int64_t)s * O + o] = acc[s];
    }
    free(acc);
}

void moty_matmul_q6k_native(float *y, const float *x, const uint8_t *raw,
                              int S, int I, int O) {
    /* dequant ONCE per (row, block), dot against ALL S tokens */
    RowJob c = { .y = y, .x = x, .q4 = raw, .S = S, .I = I, .O = O };
    moty_par_for(O, 0, q6k_rows, &c);
}


/* Q4R4 (hw/hw_q4r4.h). Activations quantized per group of 32 once per call.
 * S=1 (decode): 4-row blocks split dynamically across the team (a core busy
 * with another process does not stall a static partition).
 * S>1 (prefill): tiles of Q4R4_TT tokens (their int8 activations stay in L2);
 * inside a tile every block is unpacked once per 4 tokens by the GEMM kernel.
 * Serial calling contract (static scratch), like matmul_q_s. */
#define Q4R4_TT 32
/* activation quantization for the R4 kernels, one token per index */
typedef struct { const float *x; int8_t *xq; float *xs; int32_t *xsum; int I; } QuantJob;
static void quant_rows(void *c_, int64_t s0, int64_t s1, int tid) {
    const QuantJob *c = c_; int I = c->I, nb = I / 32;
    for (int64_t s = s0; s < s1; s++)
        moty_hw_quant_g32(c->x + s*I, I, c->xq + s*I, c->xs + s*nb, c->xsum + s*nb);
}

/* one token tile of an R4 matmul: 4-row blocks [b0, b1) */
typedef struct {
    int q8; const uint8_t *w; const uint16_t *s16; const int8_t *xq_t; const float *xs_t; const int32_t *xm_t;
    const float *xst, *xct; float *y_t; int t0, ns, n4, I, O, nb, full;
} R4Job;
static void r4_blocks(void *c_, int64_t b0, int64_t b1, int tid) {
    const R4Job *c = c_; int I = c->I, O = c->O, nb = c->nb, full = c->full, ns = c->ns, n4 = c->n4, t0 = c->t0;
    size_t bw = (size_t)nb * (c->q8 ? 128 : 64), bd = (size_t)nb * 4;
    for (int ob = (int)b0; ob < (int)b1; ob++) {
        float tmp[4 * Q4R4_TT]; int ragged = ob == full;   /* rows O%4 of the last block */
        float *yo = ragged ? tmp : c->y_t + ob*4; int ys = ragged ? 4 : O;
        for (int t = 0; t < n4; t += 4) {
            const float *st = c->xst + ((int64_t)(t0 + t)/4)*nb*4;
            if (c->q8) moty_hw_q8r4_gemm4t((const int8_t *)c->w + ob*bw, c->s16 + ob*bd, c->xq_t + (int64_t)t*I, I, st, nb, yo + (int64_t)t*ys, ys);
            else moty_hw_q4r4_gemm4t(c->w + ob*bw, c->s16 + ob*bd, c->xq_t + (int64_t)t*I, I,
                                     st, c->xct + ((int64_t)(t0 + t)/4)*nb*4, nb, yo + (int64_t)t*ys, ys);
        }
        if (n4 < ns) {
            if (c->q8) moty_hw_q8r4_gemm((const int8_t *)c->w + ob*bw, c->s16 + ob*bd, c->xq_t + (int64_t)n4*I, c->xs_t + (int64_t)n4*nb,
                                         nb, ns - n4, yo + (int64_t)n4*ys, ys);
            else moty_hw_q4r4_gemm(c->w + ob*bw, c->s16 + ob*bd, c->xq_t + (int64_t)n4*I, c->xs_t + (int64_t)n4*nb, c->xm_t + (int64_t)n4*nb,
                                   nb, ns - n4, yo + (int64_t)n4*ys, ys);
        }
        if (ragged)
            for (int t = 0; t < ns; t++) for (int r = 0; r < O - full*4; r++) c->y_t[(int64_t)t*O + full*4 + r] = tmp[t*4 + r];
    }
}

void moty_matmul_q4r4_s(float *y, const float *x, const uint8_t *q4, const uint16_t *s16,
                        int S, int I, int O) {
    static int8_t *xq = NULL; static float *xs = NULL, *xst = NULL, *xct = NULL; static int32_t *xsum = NULL;
    static int64_t c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0;
    int nb = I / 32, O4 = (O + 3) / 4, full = O / 4;
    grow((void **)&xq, &c1, (int64_t)S*I, 1, "q4r4 xq");
    grow((void **)&xs, &c2, (int64_t)S*nb, sizeof(float), "q4r4 xs");
    grow((void **)&xsum, &c3, (int64_t)S*nb, sizeof(int32_t), "q4r4 xsum");
    QuantJob qj = { x, xq, xs, xsum, I };
    if (S >= 8) moty_par_for(S, 0, quant_rows, &qj);
    else quant_rows(&qj, 0, S, 0);
    /* 4-token groups: scales rearranged once (moty_hw_q4r4_gemm4t layout) */
    int S4 = S / 4;
    if (S4) {
        grow((void **)&xst, &c4, (int64_t)S4*nb*4, sizeof(float), "q4r4 xst");
        grow((void **)&xct, &c5, (int64_t)S4*nb*4, sizeof(float), "q4r4 xct");
        for (int k = 0; k < S4; k++)
            moty_hw_q4r4_tile_scales(xs + (int64_t)k*4*nb, xsum + (int64_t)k*4*nb, nb, xst + (int64_t)k*nb*4, xct + (int64_t)k*nb*4);
    }
    for (int t0 = 0; t0 < S; t0 += Q4R4_TT) {
        int ns = S - t0 < Q4R4_TT ? S - t0 : Q4R4_TT;
        R4Job c = { 0, q4, s16, xq + (int64_t)t0*I, xs + (int64_t)t0*nb, xsum + (int64_t)t0*nb, xst, xct,
                    y + (int64_t)t0*O, t0, ns, ns / 4 * 4, I, O, nb, full };
        moty_par_for(O4, 8, r4_blocks, &c);
    }
}

/* Q8R4 (hw/hw_q4r4.h): the int8 sibling for mixed precision; the same
 * group-32 activation quantization, 4-row blocks split dynamically. */
void moty_matmul_q8r4_s(float *y, const float *x, const int8_t *q8, const uint16_t *s16,
                        int S, int I, int O) {
    static int8_t *xq = NULL; static float *xs = NULL, *xst = NULL, *xct = NULL; static int32_t *xsum = NULL;
    static int64_t c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0;
    int nb = I / 32, O4 = (O + 3) / 4, full = O / 4;
    grow((void **)&xq, &c1, (int64_t)S*I, 1, "q8r4 xq");
    grow((void **)&xs, &c2, (int64_t)S*nb, sizeof(float), "q8r4 xs");
    grow((void **)&xsum, &c3, (int64_t)S*nb, sizeof(int32_t), "q8r4 xsum");
    QuantJob qj = { x, xq, xs, xsum, I };
    if (S >= 8) moty_par_for(S, 0, quant_rows, &qj);
    else quant_rows(&qj, 0, S, 0);
    int S4 = S / 4;                                   /* 4-token groups: scales rearranged once */
    if (S4) {
        grow((void **)&xst, &c4, (int64_t)S4*nb*4, sizeof(float), "q8r4 xst");
        grow((void **)&xct, &c5, (int64_t)S4*nb*4, sizeof(float), "q8r4 xct");
        for (int k = 0; k < S4; k++)
            moty_hw_q4r4_tile_scales(xs + (int64_t)k*4*nb, xsum + (int64_t)k*4*nb, nb, xst + (int64_t)k*nb*4, xct + (int64_t)k*nb*4);
    }
    for (int t0 = 0; t0 < S; t0 += Q4R4_TT) {
        int ns = S - t0 < Q4R4_TT ? S - t0 : Q4R4_TT;
        R4Job c = { 1, (const uint8_t *)q8, s16, xq + (int64_t)t0*I, xs + (int64_t)t0*nb, NULL, xst, NULL,
                    y + (int64_t)t0*O, t0, ns, ns / 4 * 4, I, O, nb, full };
        moty_par_for(O4, 8, r4_blocks, &c);
    }
}
