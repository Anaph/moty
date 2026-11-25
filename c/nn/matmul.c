/* matmul.c — M3 libmoty-nn: UNA copia dei GEMV/GEMM.
 * Include io/gguf.h: i path nativi Q4_K/Q6_K sono sempre presenti nella lib
 * (nei motori prima erano condizionali a #ifdef GGUF_H). */
#include "nn/nn_alloc.h"
#include "hw/hw.h"
#include "io/gguf.h"
#include "nn/nn_mat.h"
#include "nn/nn_matmul.h"
#include "io/q41_panels.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_FML)
#include <stdatomic.h>
#define MOTY_Q41_FMLAL
#endif

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
    if (idot < 0) { const char *e = moty_getenv("IDOT"); idot = !(e && *e == '0'); }
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
    if (idot4 < 0) { const char *e = moty_getenv("IDOT4"); idot4 = e ? atoi(e) : 0; }
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
        if (idot4 < 0) { const char *e = moty_getenv("IDOT4"); idot4 = e ? atoi(e) : 0; }
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

/* Q4_1 reuses each decoded block across a tile of activations. */
typedef struct {
    float *y;
    const float *x;
    const uint8_t *q4;
    int S, I, O;
    int tiled, prefill_tile, pp_tile, pp_xpack, small_batch, panel_dual, ffn_decode_rows, group64;
    float *xsum;
#ifdef MOTY_Q41_FMLAL
    int compact_pp, prefetch_decode, expand_panel_prefill, short_prefill;
    float16_t *x16;
    atomic_int nonfinite_half;
#endif
} Q41Job;

#if defined(__aarch64__) && defined(__ARM_NEON)
typedef float32x4_t Q41Vec;
#define Q41_LANES 4
enum { Q41_INPUTS_PER_SUM_LANE = Q4_1_ELEMS / Q41_LANES };

/* Center the codes before the dot product. The FP16 encodings of -8..7
 * all have a zero low byte, so one table maps sixteen codes to their exact
 * values. The scale and constant term are applied once per block. */
static inline void q41_codes_bytes(const uint8_t *bytes_ptr, float16x8_t *codes) {
    const uint8x16_t fp16_hi = {0xc8, 0xc7, 0xc6, 0xc5, 0xc4, 0xc2, 0xc0, 0xbc,
                               0, 0x3c, 0x40, 0x42, 0x44, 0x45, 0x46, 0x47};
    uint8x16_t bytes = vld1q_u8(bytes_ptr);
    uint8x16_t halves[] = { vandq_u8(bytes, vdupq_n_u8(15)), vshrq_n_u8(bytes, 4) };
    for (int half = 0; half < 2; half++) {
        uint8x16_t hi_byte = vqtbl1q_u8(fp16_hi, halves[half]);
        codes[half*2]     = vreinterpretq_f16_u8(vzip1q_u8(vdupq_n_u8(0), hi_byte));
        codes[half*2 + 1] = vreinterpretq_f16_u8(vzip2q_u8(vdupq_n_u8(0), hi_byte));
    }
}

static inline void q41_codes(const uint8_t *block, float16x8_t *codes) {
    q41_codes_bytes(block+4,codes);
}

static inline Q41Vec q41_params(uint32_t dm) {
    Q41Vec params = vcvt_f32_f16(vreinterpret_f16_u16(vcreate_u16(dm)));
    return vsetq_lane_f32(vgetq_lane_f32(params, 1) + 8.f * vgetq_lane_f32(params, 0), params, 1);
}

static inline Q41Vec q41_dot(const Q41Vec *w, const float *x) {
    Q41Vec a = vmulq_f32(vld1q_f32(x), w[0]);
    Q41Vec b = vmulq_f32(vld1q_f32(x+4), w[1]);
    for (int v = 2; v < 8; v += 2) {
        a = vfmaq_f32(a, vld1q_f32(x + 4*v), w[v]);
        b = vfmaq_f32(b, vld1q_f32(x + 4*(v+1)), w[v+1]);
    }
    return vaddq_f32(a, b);
}

static inline Q41Vec q41_apply_block(Q41Vec acc, Q41Vec p, Q41Vec dot, Q41Vec xsum) {
    /* sum(x*(d*q+m)) = d*sum(x*(q-8)) + (m+8*d)*sum(x).
     * The input sums are shared by every output row. */
    acc = vfmaq_laneq_f32(acc, xsum, p, 1);
    return vfmaq_laneq_f32(acc, dot, p, 0);
}

static void q41_sum_inputs(void *job, int64_t s0, int64_t s1, int tid) {
    const Q41Job *c = job;
    for (int64_t s = s0; s < s1; s++) {
        const float *x = c->x + s*c->I;
        float *out = c->xsum + s*(c->I / Q41_INPUTS_PER_SUM_LANE);
        for (int i = 0; i < c->I; i += Q4_1_ELEMS) {
            Q41Vec a = vaddq_f32(vld1q_f32(x+i), vld1q_f32(x+i+4));
            Q41Vec b = vaddq_f32(vld1q_f32(x+i+8), vld1q_f32(x+i+12));
            Q41Vec d = vaddq_f32(vld1q_f32(x+i+16), vld1q_f32(x+i+20));
            Q41Vec e = vaddq_f32(vld1q_f32(x+i+24), vld1q_f32(x+i+28));
            vst1q_f32(out + i / Q41_INPUTS_PER_SUM_LANE, vaddq_f32(vaddq_f32(a, b), vaddq_f32(d, e)));
        }
    }
}
#ifdef MOTY_Q41_FMLAL
static void q41_pack_fp16(void *job, int64_t s0, int64_t s1, int tid) {
    Q41Job *c = job;
    uint16x8_t max_abs = vdupq_n_u16(0);
    for (int64_t s = s0; s < s1; s++) {
        const float *x = c->x + s*c->I;
        int tile = c->pp_tile;
        int step = 1;
        int64_t base = s*c->I;
        if (c->pp_xpack && s < (c->S/tile)*tile) {
            step = tile;
            base = (s/tile)*tile*c->I+(s%tile)*Q4_1_ELEMS;
        }
        float16_t *x16 = c->x16 + base;
        float *xsum = c->xsum + base/Q41_INPUTS_PER_SUM_LANE;
        for (int i = 0; i < c->I; i += Q4_1_ELEMS) {
            Q41Vec sum[4];
            for (int v = 0; v < 4; v++) {
                int j = i + 8*v;
                float16x8_t h = vcvt_high_f16_f32(vcvt_f16_f32(vld1q_f32(x+j)), vld1q_f32(x+j+4));
                vst1q_f16(x16+i*step+8*v, h);
                uint16x8_t abs_bits = vandq_u16(vreinterpretq_u16_f16(h), vdupq_n_u16(0x7fff));
                max_abs = vmaxq_u16(max_abs, abs_bits);
                sum[v] = vaddq_f32(vcvt_f32_f16(vget_low_f16(h)), vcvt_high_f32_f16(h));
            }
            vst1q_f32(xsum + i*step / Q41_INPUTS_PER_SUM_LANE,
                      vaddq_f32(vaddq_f32(sum[0], sum[1]), vaddq_f32(sum[2], sum[3])));
        }
    }
    if (vmaxvq_u16(max_abs) >= 0x7c00) atomic_store_explicit(&c->nonfinite_half, 1, memory_order_relaxed);
}

static inline Q41Vec q41_dot_fp16(const float16x8_t *codes, const float16_t *x) {
    Q41Vec lo = vdupq_n_f32(0), hi = vdupq_n_f32(0);
    for (int v = 0; v < 4; v++) {
        float16x8_t h = vld1q_f16(x + 8*v);
        lo = vfmlalq_low_f16(lo, h, codes[v]);
        hi = vfmlalq_high_f16(hi, h, codes[v]);
    }
    return vaddq_f32(lo, hi);
}
#endif
#elif defined(__AVX2__) && defined(__FMA__)
typedef __m256 Q41Vec;
#define Q41_LANES 8
static inline void q41_vectors(const uint8_t *block, Q41Vec *weights) {
    uint16_t dh, mh;
    memcpy(&dh, block, sizeof(dh));
    memcpy(&mh, block + 2, sizeof(mh));
    Q41Vec d = _mm256_set1_ps(f16_to_f32(dh)), m = _mm256_set1_ps(f16_to_f32(mh));
    __m128i bytes = _mm_loadu_si128((const __m128i *)(block + 4));
    __m128i mask = _mm_set1_epi8(15);
    __m128i halves[] = { _mm_and_si128(bytes, mask), _mm_and_si128(_mm_srli_epi16(bytes, 4), mask) };
    for (int half = 0; half < 2; half++) {
        Q41Vec lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(halves[half]));
        Q41Vec hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(halves[half], 8)));
        weights[half*2]     = _mm256_fmadd_ps(d, lo, m);
        weights[half*2 + 1] = _mm256_fmadd_ps(d, hi, m);
    }
}
static inline Q41Vec q41_zero(void) { return _mm256_setzero_ps(); }
static inline Q41Vec q41_madd(Q41Vec acc, Q41Vec w, const float *x) { return _mm256_fmadd_ps(_mm256_loadu_ps(x), w, acc); }
static inline Q41Vec q41_add(Q41Vec a, Q41Vec b) { return _mm256_add_ps(a, b); }
static inline float q41_sum(Q41Vec acc) { return simd_hsum256_f32(acc); }
#endif

#ifdef Q41_LANES
enum { Q41_VECTORS = Q4_1_ELEMS / Q41_LANES, Q41_TILE = Q41_LANES == 4 ? 8 : 2 };

/* Full token tiles use a fixed register layout. Partial tiles retain the
 * same operations per token. */
static inline __attribute__((always_inline)) void q41_tile(const Q41Job *c, int64_t o,
                                                           const uint8_t *row, int blocks,
                                                           int s0, int ns, int fp16) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    Q41Vec acc[Q41_TILE];
    for (int s = 0; s < ns; s++) acc[s] = vdupq_n_f32(0);
    for (int b = 0; b < blocks; b++) {
        const uint8_t *block = row + b * Q4_1_BYTES;
        uint32_t dm;
        memcpy(&dm, block, sizeof(dm));
        Q41Vec p = q41_params(dm);
        float16x8_t codes[4];
        q41_codes(block, codes);
#ifdef MOTY_Q41_FMLAL
        if (fp16) {
            for (int s = 0; s < ns; s++) {
                int64_t i = (int64_t)(s0 + s) * c->I + b * Q4_1_ELEMS;
                Q41Vec xsum = vld1q_f32(c->xsum + i / Q41_INPUTS_PER_SUM_LANE);
                acc[s] = q41_apply_block(acc[s], p, q41_dot_fp16(codes, c->x16 + i), xsum);
            }
            continue;
        }
#endif
        Q41Vec w[Q41_VECTORS];
        for (int v = 0; v < 4; v++) {
            w[2*v] = vcvt_f32_f16(vget_low_f16(codes[v]));
            w[2*v+1] = vcvt_high_f32_f16(codes[v]);
        }
        for (int s = 0; s < ns; s++) {
            int64_t i = (int64_t)(s0 + s) * c->I + b * Q4_1_ELEMS;
            Q41Vec xsum = vld1q_f32(c->xsum + i / Q41_INPUTS_PER_SUM_LANE);
            acc[s] = q41_apply_block(acc[s], p, q41_dot(w, c->x + i), xsum);
        }
    }
    for (int s = 0; s < ns; s++) c->y[(int64_t)(s0 + s) * c->O + o] = neon_hsum_f32(acc[s]);
#else
    Q41Vec acc[Q41_TILE][4];
    for (int s = 0; s < ns; s++)
        for (int a = 0; a < 4; a++) acc[s][a] = q41_zero();
    for (int b = 0; b < blocks; b++) {
        Q41Vec w[Q41_VECTORS];
        q41_vectors(row + b * Q4_1_BYTES, w);
        for (int s = 0; s < ns; s++) {
            const float *x = c->x + (int64_t)(s0 + s) * c->I + b * Q4_1_ELEMS;
            for (int v = 0; v < Q41_VECTORS; v++)
                acc[s][v % 4] = q41_madd(acc[s][v % 4], w[v], x + v * Q41_LANES);
        }
    }
    for (int s = 0; s < ns; s++) {
        const Q41Vec *a = acc[s];
        c->y[(int64_t)(s0 + s) * c->O + o] = q41_sum(q41_add(q41_add(a[0], a[1]), q41_add(a[2], a[3])));
    }
#endif
}

#if defined(__aarch64__) && defined(__ARM_NEON)
#ifdef MOTY_Q41_FMLAL
/* Codes are exact centered integers. Callers choose the row layout and
 * share the activation loads, then use the same two FMLAL chains. */
static inline __attribute__((always_inline)) Q41Vec q41_dot_shared(
        const uint8_t *bytes, const float16x8_t act[4]) {
    float16x8_t codes[4];
    q41_codes_bytes(bytes,codes);
    Q41Vec lo = {0}, hi = {0};
    for (int d = 0; d < 4; d++) {
        lo = vfmlalq_low_f16(lo,act[d],codes[d]);
        hi = vfmlalq_high_f16(hi,act[d],codes[d]);
    }
    return vaddq_f32(lo,hi);
}

/* Specialize for two or four rows. Consume one row's nibble codes before
 * unpacking the next row, while keeping the input vectors shared. */
static inline __attribute__((always_inline)) void q41_group64_ffn_small(
        const Q41Job *c, float *gate, float *up, const uint8_t *gw,
        const uint8_t *uw, int s, int nr) {
    const int blocks = c->I/Q4_1_ELEMS, rb = blocks*Q4_1_BYTES;
    const float16_t *x = c->x16+(int64_t)s*c->I;
    const float *xsum = c->xsum+(int64_t)s*(c->I/Q41_INPUTS_PER_SUM_LANE);
    Q41Vec acc[4] = {0};
    for (int b = 0; b < blocks; b += 2) {
        if (c->prefetch_decode && !(b % 4) && b+8 < blocks) {
            for (int r = 0; r < nr; r++) {
                __builtin_prefetch(gw+r*rb+(b+8)*Q4_1_BYTES,0,1);
                __builtin_prefetch(uw+r*rb+(b+8)*Q4_1_BYTES,0,1);
            }
        }
        Q41Vec params[4];
        for (int half = 0; half < 2; half++) {
            const float16_t *xh = x+(b+half)*Q4_1_ELEMS;
            const float16x8_t act[] = {vld1q_f16(xh), vld1q_f16(xh+8),
                                       vld1q_f16(xh+16), vld1q_f16(xh+24)};
            Q41Vec sum = vld1q_f32(xsum+4*(b+half));
            for (int r = 0; r < nr*2; r++) {
                const uint8_t *w = (r < nr ? gw+r*rb : uw+(r-nr)*rb)+(b+half)*Q4_1_BYTES;
                if (!half) {
                    uint32_t dm;
                    memcpy(&dm,w,sizeof(dm));
                    params[r] = q41_params(dm);
                }
                acc[r] = q41_apply_block(acc[r],params[r],q41_dot_shared(w+4,act),sum);
            }
        }
    }
    for (int r = 0; r < nr; r++) {
        gate[r] = neon_hsum_f32(acc[r]);
        up[r] = neon_hsum_f32(acc[nr+r]);
    }
}

static inline __attribute__((always_inline)) void q41_group64_decode(const Q41Job *c,
        float *out, const uint8_t *rows, const uint8_t *other, float *out_other,
        int s, int nr, int pair) {
    int split = pair ? nr/2 : nr, blocks = c->I/Q4_1_ELEMS;
    int64_t row_bytes = (int64_t)blocks*Q4_1_BYTES;
    Q41Vec acc[8];
    for (int r = 0; r < nr; r++) acc[r] = vdupq_n_f32(0);
    const float16_t *x = c->x16+(int64_t)s*c->I;
    const float *xsum = c->xsum+(int64_t)s*(c->I/Q41_INPUTS_PER_SUM_LANE);
    for (int b = 0; b < blocks; b += 2) {
        Q41Vec params[8];
        for (int half = 0; half < 2; half++) {
            Q41Vec low[8], high[8];
            Q41Vec sum = vld1q_f32(xsum+4*(b+half));
            for (int r = 0; r < nr; r++) low[r] = high[r] = vdupq_n_f32(0);
            float16x8_t codes[8][4];
            for (int r = 0; r < nr; r++) {
                const uint8_t *row = r < split ? rows+r*row_bytes : other+(r-split)*row_bytes;
                const uint8_t *w = row+(b+half)*Q4_1_BYTES;
                if (!half) {
                    uint32_t dm;
                    memcpy(&dm,w,sizeof(dm));
                    params[r] = q41_params(dm);
                }
                q41_codes(w,codes[r]);
            }
            for (int d = 0; d < 4; d++) {
                float16x8_t h = vld1q_f16(x+(b+half)*Q4_1_ELEMS+8*d);
                for (int r = 0; r < nr; r++) {
                    low[r] = vfmlalq_low_f16(low[r],h,codes[r][d]);
                    high[r] = vfmlalq_high_f16(high[r],h,codes[r][d]);
                }
            }
            /* Share the decoded parameters, preserving the two FP32 updates. */
            for (int r = 0; r < nr; r++)
                acc[r] = q41_apply_block(acc[r],params[r],vaddq_f32(low[r],high[r]),sum);
        }
    }
    for (int r = 0; r < nr; r++) {
        float *dst = r < split ? out+r : out_other+r-split;
        *dst = neon_hsum_f32(acc[r]);
    }
}

#endif

static inline __attribute__((always_inline)) void q41_rows_neon(const Q41Job *c, float *out,
        int stride, const uint8_t *rows, int s0, int ns, int nr, int fp16, const uint8_t *other, float *out_other, int pair) {
#ifdef MOTY_Q41_FMLAL
    if (fp16 && ns == 1 && c->group64) {
        q41_group64_decode(c,out,rows,other,out_other,s0,nr,pair);
        return;
    }
#endif
    Q41Vec acc[Q41_TILE][8];
    int split = pair ? nr/2 : nr;
    const int blocks = c->I / Q4_1_ELEMS;
    const int64_t row_bytes = (int64_t)blocks * Q4_1_BYTES;
    for (int t = 0; t < ns; t++)
        for (int r = 0; r < nr; r++) acc[t][r] = vdupq_n_f32(0);

    for (int b = 0; b < blocks; b++) {
        Q41Vec params[8];
        float16x8_t codes[8][4];
        for (int r = 0; r < nr; r++) {
            const uint8_t *row = r < split ? rows + r*row_bytes : other + (r-split)*row_bytes;
            const uint8_t *block = row + b*Q4_1_BYTES;
            uint32_t dm;
            memcpy(&dm, block, sizeof(dm));
            params[r] = q41_params(dm);
            q41_codes(block, codes[r]);
        }
#ifdef MOTY_Q41_FMLAL
        if (fp16) {
            for (int t = 0; t < ns; t++) {
                int64_t i = (int64_t)(s0+t)*c->I + b*Q4_1_ELEMS;
                const float16_t *x = c->x16 + i;
                Q41Vec sum = vld1q_f32(c->xsum + i/Q41_INPUTS_PER_SUM_LANE);
                Q41Vec low[8], high[8];
                for (int r = 0; r < nr; r++) low[r] = high[r] = vdupq_n_f32(0);
                for (int v = 0; v < 4; v++) {
                    float16x8_t h = vld1q_f16(x + 8*v);
                    for (int r = 0; r < nr; r++) {
                        low[r] = vfmlalq_low_f16(low[r], h, codes[r][v]);
                        high[r] = vfmlalq_high_f16(high[r], h, codes[r][v]);
                    }
                }
                for (int r = 0; r < nr; r++)
                    acc[t][r] = q41_apply_block(acc[t][r], params[r], vaddq_f32(low[r], high[r]), sum);
            }
            continue;
        }
#endif
        Q41Vec w[8][8];
        for (int r = 0; r < nr; r++)
            for (int v = 0; v < 4; v++) {
                w[r][2*v] = vcvt_f32_f16(vget_low_f16(codes[r][v]));
                w[r][2*v+1] = vcvt_high_f32_f16(codes[r][v]);
            }
        for (int t = 0; t < ns; t++) {
            int64_t i = (int64_t)(s0+t)*c->I + b*Q4_1_ELEMS;
            const float *x = c->x + i;
            Q41Vec sum = vld1q_f32(c->xsum + i/Q41_INPUTS_PER_SUM_LANE);
            Q41Vec low[8], high[8];
            Q41Vec x0 = vld1q_f32(x), x1 = vld1q_f32(x+4);
            for (int r = 0; r < nr; r++) {
                low[r] = vmulq_f32(x0, w[r][0]);
                high[r] = vmulq_f32(x1, w[r][1]);
            }
            for (int v = 2; v < 8; v += 2) {
                x0 = vld1q_f32(x + 4*v);
                x1 = vld1q_f32(x + 4*(v+1));
                for (int r = 0; r < nr; r++) {
                    low[r] = vfmaq_f32(low[r], x0, w[r][v]);
                    high[r] = vfmaq_f32(high[r], x1, w[r][v+1]);
                }
            }
            for (int r = 0; r < nr; r++)
                acc[t][r] = q41_apply_block(acc[t][r], params[r], vaddq_f32(low[r], high[r]), sum);
        }
    }
    for (int t = 0; t < ns; t++)
        for (int r = 0; r < nr; r++) {
            float *dst = r < split ? out + r : out_other + r - split;
            dst[(int64_t)t*stride] = neon_hsum_f32(acc[t][r]);
        }
}

#ifdef MOTY_Q41_FMLAL
/* One output row keeps the live FP32 state bounded. Unpack the
 * block once, then use it for all tokens in the outer tile. */
static inline __attribute__((always_inline)) void q41_pp_row(
        const Q41Job *c, const uint8_t *row, int lane, int packed, int start,
        int count, float *out, int stride) {
    Q41Vec acc[32] = {0};
    for (int b = 0; b < c->I/32; b++) {
        const uint8_t *block = packed ? row+(b/2)*Q41_PANEL_BYTES : row+b*Q4_1_BYTES;
        const uint8_t *data = packed ? block+16+64*(b%2)+16*lane : block+4;
        uint32_t dm;
        memcpy(&dm,block+(packed ? 4*lane : 0),sizeof(dm));
        Q41Vec params = q41_params(dm);
        float16x8_t codes[4];
        q41_codes_bytes(data,codes);
        for (int j = 0; j < count; j += 8) {
            _Pragma("GCC unroll 8")
            for (int t = j; t < j+8; t++) {
                int64_t k = c->pp_xpack ? (int64_t)start*c->I+(int64_t)b*32*count+t*32 :
                                            (int64_t)(start+t)*c->I+b*32;
                const float16_t *x = c->x16+k;
                Q41Vec lo = vdupq_n_f32(0), hi = vdupq_n_f32(0);
                for (int d = 0; d < 4; d++) {
                    float16x8_t a = vld1q_f16(x+8*d);
                    lo = vfmlalq_low_f16(lo,a,codes[d]);
                    hi = vfmlalq_high_f16(hi,a,codes[d]);
                }
                acc[t] = q41_apply_block(acc[t],params,vaddq_f32(lo,hi),
                                         vld1q_f32(c->xsum+k/Q41_INPUTS_PER_SUM_LANE));
            }
        }
    }
    for (int t = 0; t < count; t++) out[(int64_t)t*stride] = neon_hsum_f32(acc[t]);
}

static inline __attribute__((always_inline)) void q41_pp_rows(const Q41Job *c,
        const uint8_t *rows, int n, int start, int count, float *out, int stride) {
    int64_t rb = (int64_t)(c->I/32)*Q4_1_BYTES;
    for (int r = 0; r < n; r++) {
        if (count == 32) q41_pp_row(c,rows+r*rb,0,0,start,32,out+r,stride);
        else q41_pp_row(c,rows+r*rb,0,0,start,16,out+r,stride);
    }
}
/* Two rows share each activation load. Limit persistent accumulators to
 * 2*ns, and finish both products before loading the next token's block. */
static inline __attribute__((always_inline)) void q41_compact_pp(
        const Q41Job *c, float *out, int stride, const uint8_t *row, int start, int ns) {
    const int blocks = c->I/Q4_1_ELEMS;
    const uint8_t *other = row+(int64_t)blocks*Q4_1_BYTES;
    Q41Vec acc0[6] = {0}, acc1[6] = {0};
    for (int b = 0; b < blocks; b++, row += Q4_1_BYTES, other += Q4_1_BYTES) {
        uint32_t dm0, dm1;
        memcpy(&dm0,row,sizeof(dm0));
        memcpy(&dm1,other,sizeof(dm1));
        Q41Vec p0 = q41_params(dm0), p1 = q41_params(dm1);
        float16x8_t w0[4], w1[4];
        q41_codes(row,w0);
        q41_codes(other,w1);
        _Pragma("GCC unroll 6")
        for (int t = 0; t < ns; t++) {
            int64_t i = (int64_t)(start+t)*c->I+b*Q4_1_ELEMS;
            Q41Vec lo0 = {0}, hi0 = {0}, lo1 = {0}, hi1 = {0};
            for (int d = 0; d < 4; d++) {
                float16x8_t x = vld1q_f16(c->x16+i+8*d);
                lo0 = vfmlalq_low_f16(lo0,x,w0[d]);
                hi0 = vfmlalq_high_f16(hi0,x,w0[d]);
                lo1 = vfmlalq_low_f16(lo1,x,w1[d]);
                hi1 = vfmlalq_high_f16(hi1,x,w1[d]);
            }
            Q41Vec sum = vld1q_f32(c->xsum+i/Q41_INPUTS_PER_SUM_LANE);
            acc0[t] = q41_apply_block(acc0[t],p0,vaddq_f32(lo0,hi0),sum);
            acc1[t] = q41_apply_block(acc1[t],p1,vaddq_f32(lo1,hi1),sum);
        }
    }
    for (int t = 0; t < ns; t++) {
        out[(int64_t)t*stride] = neon_hsum_f32(acc0[t]);
        out[(int64_t)t*stride+1] = neon_hsum_f32(acc1[t]);
    }
}

static __attribute__((noinline)) void q41_compact_pp4(
        const Q41Job *c, float *out, int stride, const uint8_t *rows, int start) {
    q41_compact_pp(c,out,stride,rows,start,4);
}

static __attribute__((noinline)) void q41_compact_pp6(
        const Q41Job *c, float *out, int stride, const uint8_t *rows, int start) {
    q41_compact_pp(c,out,stride,rows,start,6);
}

static int q41_compact_size(const char *name) {
    const char *value = moty_getenv(name);
    return value && !strcmp(value,"6") ? 6 : value && !strcmp(value,"4") ? 4 : 0;
}
#endif

static inline __attribute__((always_inline)) void q41_neon_range(const Q41Job *c,
        float *out, int stride, const uint8_t *rows, int n, int s, int ns, int fp16, int tiled) {
    int64_t rb = (int64_t)(c->I/Q4_1_ELEMS)*Q4_1_BYTES;
    int r = 0;
#ifdef MOTY_Q41_FMLAL
    if (fp16 && c->compact_pp && ns == c->compact_pp) {
        for (; n-r >= 2; r += 2) {
            if (ns == 6) q41_compact_pp6(c,out+r,stride,rows+r*rb,s);
            else q41_compact_pp4(c,out+r,stride,rows+r*rb,s);
        }
    }
#endif
    if (tiled) {
        if (fp16) {
            for (; n-r >= 4; r += 4) q41_rows_neon(c, out+r, stride, rows+r*rb, s, ns, 4, 1, NULL, NULL, 0);
        } else {
            for (; n-r >= 2; r += 2) q41_rows_neon(c, out+r, stride, rows+r*rb, s, ns, 2, 0, NULL, NULL, 0);
        }
    }
    for (; r < n; r++) q41_rows_neon(c, out+r, stride, rows+r*rb, s, ns, 1, fp16, NULL, NULL, 0);
}
#endif

/* Keep the activation tile hot across output rows on each worker. */
static inline __attribute__((always_inline)) void q41_tile_rows(const Q41Job *c, int64_t o0, int64_t o1,
                                                               int s, int ns, int fp16) {
    int blocks = c->I / Q4_1_ELEMS;
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (c->tiled) {
        q41_neon_range(c, c->y+(int64_t)s*c->O+o0, c->O,
                       c->q4+o0*blocks*Q4_1_BYTES, (int)(o1-o0), s, ns, fp16, 1);
        return;
    }
#endif
    for (int64_t o = o0; o < o1; o++) q41_tile(c, o, c->q4 + o * blocks * Q4_1_BYTES, blocks, s, ns, fp16);
}

static void q41_decode_rows(void *job, int64_t o0, int64_t o1, int tid) {
    q41_tile_rows(job, o0, o1, 0, 1, 0);
}

static inline __attribute__((always_inline)) void q41_prefill_rows(const Q41Job *c, int64_t o0,
                                                                  int64_t o1, int fp16) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (c->small_batch && c->tiled) {
        q41_tile_rows(c,o0,o1,0,c->S,fp16);
        return;
    }
#endif
    int s = 0;
#ifdef MOTY_Q41_FMLAL
    if (fp16 && c->compact_pp)
        for (; c->S-s >= c->compact_pp; s += c->compact_pp)
            q41_tile_rows(c,o0,o1,s,c->compact_pp,1);
    if (fp16 && c->pp_tile) {
        const uint8_t *rows = c->q4+o0*(c->I/32)*Q4_1_BYTES;
        for (; c->S-s >= c->pp_tile; s += c->pp_tile)
            q41_pp_rows(c,rows,(int)(o1-o0),s,c->pp_tile,c->y+(int64_t)s*c->O+o0,c->O);
    }
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (c->prefill_tile == 2) {
        for (; c->S - s >= 2; s += 2) q41_tile_rows(c, o0, o1, s, 2, fp16);
    } else if (c->prefill_tile == 4) {
        for (; c->S - s >= 4; s += 4) q41_tile_rows(c, o0, o1, s, 4, fp16);
    }
#endif
    for (; c->S - s >= Q41_TILE; s += Q41_TILE) q41_tile_rows(c, o0, o1, s, Q41_TILE, fp16);
    if (Q41_TILE > 4 && c->S - s >= 4) { q41_tile_rows(c, o0, o1, s, 4, fp16); s += 4; }
    if (Q41_TILE > 2 && c->S - s >= 2) { q41_tile_rows(c, o0, o1, s, 2, fp16); s += 2; }
    if (s < c->S) q41_tile_rows(c, o0, o1, s, 1, fp16);
}
#endif

#ifdef MOTY_Q41_FMLAL
static void q41_rows_fp16(void *job, int64_t o0, int64_t o1, int tid) {
    const Q41Job *c = job;
    if (c->S == 1) q41_tile_rows(c, o0, o1, 0, 1, 1);
    else q41_prefill_rows(c, o0, o1, 1);
}

#if defined(__APPLE__)
enum { Q41_CACHE_BYTES = 128 };
#else
enum { Q41_CACHE_BYTES = 64 };
#endif
enum { Q41_CACHE_ROWS = Q41_CACHE_BYTES/sizeof(float) };

static int q41_decode_head(const Q41Job *c) {
    return (int)((-(uintptr_t)c->y & (Q41_CACHE_BYTES-1))/sizeof(float));
}

static void q41_decode_aligned(void *job, int64_t first, int64_t end, int tid) {
    const Q41Job *c = job;
    int head = q41_decode_head(c);
    int64_t start = first ? (first-(head != 0))*Q41_CACHE_ROWS+head : 0;
    int64_t stop = (end-(head != 0))*Q41_CACHE_ROWS+head;
    q41_tile_rows(c,start,stop < c->O ? stop : c->O,0,1,1);
}
#endif

static void q41_rows(void *job, int64_t o0, int64_t o1, int tid) {
    const Q41Job *c = job;
#ifdef Q41_LANES
    q41_prefill_rows(c, o0, o1, 0);
#else
    int blocks = c->I / Q4_1_ELEMS;
    for (int64_t o = o0; o < o1; o++) {
        const uint8_t *row = c->q4 + o * blocks * Q4_1_BYTES;
        for (int s = 0; s < c->S; s++) {
            const float *x = c->x + (int64_t)s * c->I;
            float sum = 0.f;
            for (int b = 0; b < blocks; b++, x += Q4_1_ELEMS) {
                const uint8_t *block = row + b * Q4_1_BYTES;
                uint16_t dh, mh;
                memcpy(&dh, block, sizeof(dh));
                memcpy(&mh, block + 2, sizeof(mh));
                float d = f16_to_f32(dh), m = f16_to_f32(mh);
                const uint8_t *q = block + 4;
                for (int i = 0; i < 16; i++) sum += x[i] * (d * (q[i] & 15) + m);
                for (int i = 0; i < 16; i++) sum += x[i + 16] * (d * (q[i] >> 4) + m);
            }
            c->y[(int64_t)s * c->O + o] = sum;
        }
    }
#endif
}

static _Thread_local int q41_pack_fallback, q41_exact_batch;

void moty_q41_begin_exact_batch(void) {
    q41_exact_batch = 1;
    q41_pack_fallback = 0;
}

int moty_q41_end_exact_batch(void) {
    q41_exact_batch = 0;
    return q41_pack_fallback;
}

int moty_q41_uses_fp16(void) {
#ifdef MOTY_Q41_FMLAL
    const char *precision = moty_getenv("Q41_ACT");
    return !precision || !strcmp(precision, "f16");
#else
    return 0;
#endif
}

#if defined(__aarch64__) && defined(__ARM_NEON)
static int q41_prepare(Q41Job *c) {
    static float *xsum = NULL;
    static int64_t cap = 0;
    int S = c->S, I = c->I;
    grow((void **)&xsum, &cap, (int64_t)S*(I / Q41_INPUTS_PER_SUM_LANE), sizeof(float), "Q4_1 input sums");
    c->xsum = xsum;
    const char *rows = moty_getenv("MOTY_Q41_ROWS");
    const char *tile = moty_getenv("MOTY_Q41_TILE");
    int fp16 = moty_q41_uses_fp16();
    c->tiled = rows ? !strcmp(rows, "1") : fp16;
    c->small_batch = q41_exact_batch && S > 1 && S <= 5;
    c->prefill_tile = tile && !strcmp(tile, "4") ? 4 : tile && !strcmp(tile, "2") ? 2 : 8;
#ifdef MOTY_Q41_FMLAL
    if (fp16) {
        if (S >= 8 && S <= 32) {
            const char *short_prefill = moty_getenv("MOTY_Q41_SHORT_PREFILL");
            c->short_prefill = short_prefill && !strcmp(short_prefill,"1");
        }
        if (S >= 16 && !c->compact_pp) {
            const char *pp = moty_getenv("MOTY_Q41_PP_TILE");
            c->pp_tile = pp && !strcmp(pp,"16") ? 16 : pp && !strcmp(pp,"32") ? 32 : 0;
            const char *order = c->pp_tile ? moty_getenv("MOTY_Q41_PP_XPACK") : NULL;
            c->pp_xpack = order && !strcmp(order,"1");
        }
        static float16_t *x16 = NULL;
        static int64_t cap16 = 0;
        grow((void **)&x16, &cap16, (int64_t)S*I, sizeof(float16_t), "Q4_1 FP16 activations");
        c->x16 = x16;
        atomic_init(&c->nonfinite_half, 0);
        /* The worker dispatch costs more than packing a small tile locally. */
        if (S >= 16 && (!c->short_prefill || (int64_t)S*I >= 24*1024))
            moty_par_for(S, 0, q41_pack_fp16, c);
        else q41_pack_fp16(c, 0, S, 0);
        if (!atomic_load_explicit(&c->nonfinite_half, memory_order_relaxed)) return 1;
        q41_pack_fallback = 1;
    }
#endif
    if (S >= 16) moty_par_for(S, 0, q41_sum_inputs, c);
    else q41_sum_inputs(c, 0, S, 0);
    return 0;
}

enum { Q41_FFN_ROWS = 16 };
typedef struct { const Q41Job *mat; const uint8_t *up; int fp16; } Q41FfnJob;

static inline __attribute__((always_inline)) void q41_ffn_tile(const Q41FfnJob *f,
        int64_t first, int nr, int s, int ns, int fp16) {
    const Q41Job *c = f->mat;
    int64_t offset = first*(c->I/Q4_1_ELEMS)*Q4_1_BYTES;
    float local_gate[Q41_TILE*Q41_FFN_ROWS], up[Q41_TILE*Q41_FFN_ROWS];
    float *dst = c->y+(int64_t)s*c->O+first;
    float *gate = local_gate;
#ifdef MOTY_Q41_FMLAL
    if (fp16 && c->prefetch_decode && ns == 1) gate = dst;
#endif
    const uint8_t *gate_rows = c->q4+offset, *up_rows = f->up+offset;
    int r = 0;
    if (ns == 1 && c->tiled) {
        int64_t rb = (int64_t)(c->I/Q4_1_ELEMS)*Q4_1_BYTES;
        if (fp16) {
#ifdef MOTY_Q41_FMLAL
            if (c->group64 && c->ffn_decode_rows == 1) {
                for (; r < nr; r++)
                    q41_group64_ffn_small(c,gate+r,up+r,gate_rows+r*rb,up_rows+r*rb,s,1);
            } else if (c->group64 && c->ffn_decode_rows == 2) {
                for (; nr-r >= 2; r += 2)
                    q41_group64_ffn_small(c,gate+r,up+r,gate_rows+r*rb,up_rows+r*rb,s,2);
            }
#endif
            for (; nr-r >= 4; r += 4)
                q41_rows_neon(c,gate+r,nr,gate_rows+r*rb,s,1,8,1,up_rows+r*rb,up+r,1);
        } else {
            for (; nr-r >= 2; r += 2)
                q41_rows_neon(c,gate+r,nr,gate_rows+r*rb,s,1,4,0,up_rows+r*rb,up+r,1);
        }
        gate_rows += r*rb;
        up_rows += r*rb;
    }
    q41_neon_range(c, gate+r, nr, gate_rows, nr-r, s, ns, fp16, c->tiled);
    q41_neon_range(c, up+r, nr, up_rows, nr-r, s, ns, fp16, c->tiled);
    moty_hw_silu_mul(gate, up, ns*nr);
    if (gate != dst)
        for (int t = 0; t < ns; t++)
            memcpy(dst+(int64_t)t*c->O,gate+t*nr,(size_t)nr*sizeof(float));
}

static inline __attribute__((always_inline)) void q41_ffn_worker(const Q41FfnJob *f,
        int64_t first, int64_t end, int fp16) {
    for (int64_t o = first; o < end; o += Q41_FFN_ROWS) {
        int nr = (int)(end-o < Q41_FFN_ROWS ? end-o : Q41_FFN_ROWS);
        int s = 0, S = f->mat->S;
        if (f->mat->small_batch && f->mat->tiled) {
            q41_ffn_tile(f,o,nr,0,S,fp16);
            continue;
        }
#ifdef MOTY_Q41_FMLAL
        const Q41Job *c = f->mat;
        if (fp16 && c->compact_pp)
            for (; S-s >= c->compact_pp; s += c->compact_pp)
                q41_ffn_tile(f,o,nr,s,c->compact_pp,1);
        if (fp16 && c->pp_tile) {
            int64_t offset = o*(c->I/32)*Q4_1_BYTES;
            int count = c->pp_tile;
            for (; S-s >= count; s += count) {
                float gate[32*Q41_FFN_ROWS], up[32*Q41_FFN_ROWS];
                q41_pp_rows(c,c->q4+offset,nr,s,count,gate,nr);
                q41_pp_rows(c,f->up+offset,nr,s,count,up,nr);
                moty_hw_silu_mul(gate,up,count*nr);
                for (int t = 0; t < count; t++)
                    memcpy(c->y+(int64_t)(s+t)*c->O+o,gate+t*nr,nr*sizeof(float));
            }
        }
#endif
        if (f->mat->prefill_tile == 8)
            for (; S-s >= 8; s += 8) q41_ffn_tile(f, o, nr, s, 8, fp16);
        if (f->mat->prefill_tile != 2)
            for (; S-s >= 4; s += 4) q41_ffn_tile(f, o, nr, s, 4, fp16);
        for (; S-s >= 2; s += 2) q41_ffn_tile(f, o, nr, s, 2, fp16);
        if (s < S) q41_ffn_tile(f, o, nr, s, 1, fp16);
    }
}

static void q41_ffn_rows(void *job, int64_t first, int64_t end, int tid) {
    const Q41FfnJob *f = job;
    first *= Q41_FFN_ROWS;
    end *= Q41_FFN_ROWS;
    if (end > f->mat->O) end = f->mat->O;
#ifdef MOTY_Q41_FMLAL
    if (f->fp16) { q41_ffn_worker(f, first, end, 1); return; }
#endif
    q41_ffn_worker(f, first, end, 0);
}

static void q41_panel_rows(void *arg, int64_t a, int64_t b, int tid);

typedef struct {
    Q41Job *input;
    float *const *out;
    const Mat *const *proj;
    int fp16;
} Q41QkvJob;

static inline __attribute__((always_inline)) void q41_qkv_range(
        void *job, int64_t first, int64_t end, int tid, int allow_packed) {
    const Q41QkvJob *qkv = job;
    /* Intersect this worker's range with each matrix's rows. The outputs
     * keep their own token strides, so no concatenation or split is needed. */
    for (int i = 0; i < 3 && end > 0; i++) {
        const Mat *w = qkv->proj[i];
        int packed = allow_packed && w->gs == Q41_PACKED_GS;
        int jobs = packed ? w->O/Q41_PANEL_ROWS : w->O;
        if (first < jobs) {
            Q41Job c = *qkv->input;
            c.y = qkv->out[i]; c.q4 = w->q4; c.O = w->O; c.group64 = w->gs == 64;
            int64_t lo = first < 0 ? 0 : first, hi = end < jobs ? end : jobs;
            if (packed) {
                Q41FfnJob panel = { &c, NULL, qkv->fp16 };
                q41_panel_rows(&panel,lo,hi,tid);
            } else {
#ifdef MOTY_Q41_FMLAL
                if (qkv->fp16) q41_rows_fp16(&c,lo,hi,tid);
                else
#endif
                if (c.S == 1) q41_decode_rows(&c,lo,hi,tid);
                else q41_rows(&c,lo,hi,tid);
            }
        }
        first -= jobs;
        end -= jobs;
    }
}
static void q41_qkv_rows(void *job, int64_t first, int64_t end, int tid) {
    q41_qkv_range(job,first,end,tid,0);
}

static void q41_qkv_mixed_rows(void *job, int64_t first, int64_t end, int tid) {
    q41_qkv_range(job,first,end,tid,1);
}

#endif

int moty_q41_qkv(float *const out[3], const float *x, const Mat *const proj[3], int S) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    const char *enable = moty_getenv("MOTY_Q41_QKV");
    if (!(enable ? !strcmp(enable, "1") : moty_q41_uses_fp16())) return 0;
    const int I = proj[0]->I;
    int64_t rows = 0;
    int packed = 0;
    for (int i = 0; i < 3; i++) {
        const Mat *w = proj[i];
        if (w->fmt != WF_Q4_1 || !w->q4 || w->sh || w->I != I) return 0;
        if (w->gs == Q41_PACKED_GS) {
            if (w->O % Q41_PANEL_ROWS || I % Q41_PANEL_GROUP) return 0;
            rows += w->O/Q41_PANEL_ROWS;
            packed = 1;
        } else rows += w->O;
    }
    Q41Job input = { .x = x, .S = S, .I = I };
    Q41QkvJob job = { &input, out, proj, q41_prepare(&input) };
    moty_par_for(rows,0,packed ? q41_qkv_mixed_rows : q41_qkv_rows,&job);
    return 1;
#else
    return 0;
#endif
}

int moty_q41_ffn_enabled(void) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    const char *enable = moty_getenv("MOTY_Q41_FFN");
    return enable ? !strcmp(enable, "1") : moty_q41_uses_fp16();
#else
    return 0;
#endif
}

#ifdef MOTY_Q41_FMLAL
static inline void q41_set_ffn_rows(Q41Job *c) {
    if (c->S == 1 && c->group64) {
        const char *tile = moty_getenv("MOTY_Q41_FFN_DECODE_ROWS");
        /* Only the twelve-worker run cleared the whole-model rollout threshold. */
        c->ffn_decode_rows = tile ? atoi(tile) : (moty_par_threads() == 12 ? 2 : 4);
        const char *prefetch = moty_getenv("MOTY_Q41_GATE_DECODE");
        c->prefetch_decode = prefetch && !strcmp(prefetch,"1");
    }
}
#endif

int moty_q41_gate_up(float *y, const float *x, const uint8_t *gate, const uint8_t *up, int S, int I, int O, int group64) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (O % 4 || !moty_q41_ffn_enabled()) return 0;
    Q41Job c = { .y = y, .x = x, .q4 = gate, .S = S, .I = I, .O = O, .group64 = group64 };
#ifdef MOTY_Q41_FMLAL
    q41_set_ffn_rows(&c);
    if (S >= 4) c.compact_pp = q41_compact_size("MOTY_Q41_COMPACT_PP");
#endif
    Q41FfnJob f = { &c, up, q41_prepare(&c) };
    int jobs = (O-1)/Q41_FFN_ROWS+1;
    moty_par_for(jobs, 0, q41_ffn_rows, &f);
    return 1;
#else
    return 0;
#endif
}

#if defined(__aarch64__) && defined(__ARM_NEON)
static inline __attribute__((always_inline)) void q41_panel_tile(
        const Q41Job *c, const uint8_t *panel, int start, int count,
        float *out, int stride, int fp16) {
    Q41Vec acc[8][4] = {0};
    for (int g = 0; g < c->I/64; g++, panel += Q41_PANEL_BYTES) {
        for (int r = 0; r < 4; r++) {
            uint32_t dm;
            memcpy(&dm,panel+4*r,4);
            Q41Vec params = q41_params(dm);
            for (int half = 0; half < 2; half++) {
                float16x8_t codes[4];
                q41_codes_bytes(panel+16+64*half+16*r,codes);
#ifdef MOTY_Q41_FMLAL
                if (fp16) {
                    for (int t = 0; t < count; t++) {
                        int64_t i = (int64_t)(start+t)*c->I+g*64+half*32;
                        Q41Vec lo = vdupq_n_f32(0), hi = vdupq_n_f32(0);
                        for (int d = 0; d < 4; d++) {
                            float16x8_t x = vld1q_f16(c->x16+i+8*d);
                            lo = vfmlalq_low_f16(lo,x,codes[d]);
                            hi = vfmlalq_high_f16(hi,x,codes[d]);
                        }
                        acc[t][r] = q41_apply_block(acc[t][r],params,vaddq_f32(lo,hi),
                                                      vld1q_f32(c->xsum+i/Q41_INPUTS_PER_SUM_LANE));
                    }
                    continue;
                }
#endif
                Q41Vec w[8];
                for (int d = 0; d < 4; d++) {
                    w[2*d] = vcvt_f32_f16(vget_low_f16(codes[d]));
                    w[2*d+1] = vcvt_high_f32_f16(codes[d]);
                }
                for (int t = 0; t < count; t++) {
                    int64_t i = (int64_t)(start+t)*c->I+g*64+half*32;
                    acc[t][r] = q41_apply_block(acc[t][r],params,q41_dot(w,c->x+i),
                                                  vld1q_f32(c->xsum+i/Q41_INPUTS_PER_SUM_LANE));
                }
            }
        }
    }
    for (int t = 0; t < count; t++)
        for (int r = 0; r < 4; r++) out[(int64_t)t*stride+r] = neon_hsum_f32(acc[t][r]);
}

#ifdef MOTY_Q41_FMLAL
typedef struct { Q41Vec scale, bias; } Q41PanelParams;

static inline Q41PanelParams q41_panel_params(uint8x16_t bytes) {
    const float16x8_t dm = vreinterpretq_f16_u8(bytes);
    const Q41Vec lo = vcvt_f32_f16(vget_low_f16(dm));
    const Q41Vec hi = vcvt_high_f32_f16(dm);
    const Q41Vec scale = vuzp1q_f32(lo,hi);
    return (Q41PanelParams){scale,vfmaq_n_f32(vuzp2q_f32(lo,hi),scale,8.f)};
}

/* The NEON lane operands must be compile-time constants. */
#define Q41_UPDATE_PANEL_ROW(r, codes) do { \
    const Q41Vec dot = q41_dot_shared(codes,act); \
    acc[r] = vfmaq_laneq_f32(acc[r],sum,params.bias,r); \
    acc[r] = vfmaq_laneq_f32(acc[r],dot,params.scale,r); \
} while (0)

/* A fixed GEMV tile keeps the four accumulators and packed activations
 * live while consuming one row's codes at a time. */
static inline __attribute__((always_inline)) void q41_panel_gemv(
        const Q41Job *c, const uint8_t *w, float *out, const uint32_t *table,
        int indexed) {
    const int param_bytes = indexed ? 8 : 16;
    const int group_bytes = indexed ? Q41_INDEXED_BYTES : Q41_PANEL_BYTES;
    Q41Vec acc[4] = {0};
    for (int g = 0; g < c->I/64; g++, w += group_bytes) {
        /* Each lane supplies the scale and centered bias of one panel row. */
        uint8x16_t dm;
        if (indexed) {
            uint16_t id[4];
            memcpy(id,w,sizeof(id));
            uint32x4_t pair = {table[id[0]],table[id[1]],table[id[2]],table[id[3]]};
            dm = vreinterpretq_u8_u32(pair);
        } else dm = vld1q_u8(w);
        const Q41PanelParams params = q41_panel_params(dm);
        for (int half = 0; half < 2; half++) {
            int i = g*64+32*half;
            const float16_t *x = c->x16+i;
            const float16x8_t act[] = {vld1q_f16(x), vld1q_f16(x+8), vld1q_f16(x+16), vld1q_f16(x+24)};
            Q41Vec sum = vld1q_f32(c->xsum+i/Q41_INPUTS_PER_SUM_LANE);
            const uint8_t *codes = w+param_bytes+64*half;
            Q41_UPDATE_PANEL_ROW(0,codes);
            Q41_UPDATE_PANEL_ROW(1,codes+16);
            Q41_UPDATE_PANEL_ROW(2,codes+32);
            Q41_UPDATE_PANEL_ROW(3,codes+48);
        }
    }
    for (int r = 0; r < 4; r++) out[r] = neon_hsum_f32(acc[r]);
}

static inline __attribute__((always_inline)) void q41_panel_gemv_dual(
        const Q41Job *c, const uint8_t *gate, const uint8_t *up, int lane,
        float *out, float *out_up) {
    Q41Vec acc[4] = {0};
    for (int g = 0; g < c->I/64; g++, gate += Q41_PANEL_BYTES, up += Q41_PANEL_BYTES) {
        const Q41PanelParams params =
            q41_panel_params(vcombine_u8(vld1_u8(gate+4*lane),vld1_u8(up+4*lane)));
        for (int half = 0; half < 2; half++) {
            int i = 64*g+32*half;
            const float16_t *x = c->x16+i;
            const float16x8_t act[] = {vld1q_f16(x),vld1q_f16(x+8),vld1q_f16(x+16),vld1q_f16(x+24)};
            Q41Vec sum = vld1q_f32(c->xsum+i/Q41_INPUTS_PER_SUM_LANE);
            const uint8_t *gate_codes = gate+16+64*half+16*lane;
            const uint8_t *up_codes = up+16+64*half+16*lane;
            Q41_UPDATE_PANEL_ROW(0,gate_codes);
            Q41_UPDATE_PANEL_ROW(1,gate_codes+16);
            Q41_UPDATE_PANEL_ROW(2,up_codes);
            Q41_UPDATE_PANEL_ROW(3,up_codes+16);
        }
    }
    for (int r = 0; r < 2; r++) {
        out[r] = neon_hsum_f32(acc[r]);
        out_up[r] = neon_hsum_f32(acc[r+2]);
    }
}
#undef Q41_UPDATE_PANEL_ROW

static void q41_panel_decode(const Q41FfnJob *f, int64_t a, int64_t b) {
    const Q41Job *c = f->mat;
    int64_t bytes = (int64_t)(c->I/64)*Q41_PANEL_BYTES;
    for (int64_t r = a; r < b; r++) {
        float *out = c->y+4*r;
        if (f->up && c->panel_dual) {
            float up[4];
            q41_panel_gemv_dual(c,c->q4+r*bytes,f->up+r*bytes,0,out,up);
            q41_panel_gemv_dual(c,c->q4+r*bytes,f->up+r*bytes,2,out+2,up+2);
            moty_hw_silu_mul(out,up,4);
            continue;
        }
        q41_panel_gemv(c,c->q4+r*bytes,out,NULL,0);
        if (f->up) {
            float up[4];
            q41_panel_gemv(c,f->up+r*bytes,up,NULL,0);
            moty_hw_silu_mul(out,up,4);
        }
    }
}

enum { Q41_PANEL_PREFILL_MAX_INPUT = 2560 };

static void q41_expand_panel(uint8_t *dst, const uint8_t *panel, int input) {
    const int groups = input/Q41_PANEL_GROUP;
    const int row_bytes = input/Q4_1_ELEMS*Q4_1_BYTES;
    for (int r = 0; r < Q41_PANEL_ROWS; r++) {
        uint8_t *row = dst+r*row_bytes;
        for (int g = 0; g < groups; g++, row += 2*Q4_1_BYTES) {
            const uint8_t *src = panel+g*Q41_PANEL_BYTES;
            q41_panel_row_block(src,r,0,row);
            q41_panel_row_block(src,r,1,row+Q4_1_BYTES);
        }
    }
}

/* Unpack once for all tokens and reuse the raw row kernels. A worker
 * keeps only one panel per projection on its stack. */
static void q41_panel_expand_work(const Q41FfnJob *f, int64_t first, int64_t end) {
    const Q41Job *c = f->mat;
    const int packed_bytes = c->I/Q41_PANEL_GROUP*Q41_PANEL_BYTES;
    const int raw_bytes = Q41_PANEL_ROWS*(c->I/Q4_1_ELEMS)*Q4_1_BYTES;
    uint8_t raw[2*Q41_PANEL_ROWS*(Q41_PANEL_PREFILL_MAX_INPUT/Q4_1_ELEMS)*Q4_1_BYTES];
    for (int64_t r = first; r < end; r++) {
        Q41Job tile = *c;
        tile.y += Q41_PANEL_ROWS*r;
        tile.q4 = raw;
        tile.group64 = 1;
        q41_expand_panel(raw,c->q4+r*packed_bytes,c->I);
        if (f->up) {
            q41_expand_panel(raw+raw_bytes,f->up+r*packed_bytes,c->I);
            Q41FfnJob pair = {&tile,raw+raw_bytes,1};
            q41_ffn_worker(&pair,0,Q41_PANEL_ROWS,1);
        } else q41_prefill_rows(&tile,0,Q41_PANEL_ROWS,1);
    }
}
#endif

#ifdef MOTY_Q41_FMLAL
static __attribute__((noinline)) void q41_panel_rows8(
        const Q41Job *c, const uint8_t *w, int start, float *out, int stride) {
    /* One row at a time leaves room for all eight FP32 accumulators. */
    for (int row = 0; row < Q41_PANEL_ROWS; row++)
        q41_pp_row(c,w,row,1,start,8,out+row,stride);
}
#endif

static inline __attribute__((always_inline)) void q41_panel_prefill_tile(
        const Q41Job *c, const uint8_t *w, int start, int count, float *out, int stride, int fp16) {
#ifdef MOTY_Q41_FMLAL
    if (fp16 && c->short_prefill && count == 8 && !c->pp_xpack) {
        q41_panel_rows8(c,w,start,out,stride);
        return;
    }
#endif
    q41_panel_tile(c,w,start,count,out,stride,fp16);
}

static inline __attribute__((always_inline)) void q41_panel_work(
        const Q41FfnJob *f, int64_t a, int64_t b, int fp16) {
    const Q41Job *c = f->mat;
    const int64_t bytes = (int64_t)(c->I/64)*Q41_PANEL_BYTES;
    for (int64_t r = a; r < b; r++) {
        const uint8_t *w = c->q4 + r*bytes;
        int t = 0;
#ifdef MOTY_Q41_FMLAL
        if (fp16 && c->pp_tile) {
            int count = c->pp_tile;
            for (; c->S-t >= count; t += count) {
                float gate[128], up[128];
                float *out = c->y+(int64_t)t*c->O+4*r;
                for (int lane = 0; lane < 4; lane++) {
                    if (count == 32) {
                        q41_pp_row(c,w,lane,1,t,32,f->up ? gate+lane : out+lane,f->up ? 4 : c->O);
                        if (f->up) q41_pp_row(c,f->up+r*bytes,lane,1,t,32,up+lane,4);
                    } else {
                        q41_pp_row(c,w,lane,1,t,16,f->up ? gate+lane : out+lane,f->up ? 4 : c->O);
                        if (f->up) q41_pp_row(c,f->up+r*bytes,lane,1,t,16,up+lane,4);
                    }
                }
                if (f->up) {
                    moty_hw_silu_mul(gate,up,4*count);
                    for (int j = 0; j < count; j++) memcpy(out+(int64_t)j*c->O,gate+4*j,4*sizeof(float));
                }
            }
        }
#endif
        for (; t < c->S;) {
            int count = c->S-t >= 8 ? 8 : c->S-t;
            float *out = c->y+(int64_t)t*c->O+4*r;
            if (f->up) {
                float gate[32], up[32];
                q41_panel_prefill_tile(c,w,t,count,gate,4,fp16);
                q41_panel_prefill_tile(c,f->up+r*bytes,t,count,up,4,fp16);
                moty_hw_silu_mul(gate,up,count*4);
                for (int j = 0; j < count; j++) memcpy(out+(int64_t)j*c->O,gate+4*j,4*sizeof(float));
            } else q41_panel_prefill_tile(c,w,t,count,out,c->O,fp16);
            t += count;
        }
    }
}
static void q41_panel_rows(void *arg, int64_t a, int64_t b, int tid) {
    const Q41FfnJob *f = arg;
#ifdef MOTY_Q41_FMLAL
    if (f->fp16) {
        if (f->mat->S == 1) q41_panel_decode(f,a,b);
        else if (f->mat->expand_panel_prefill) q41_panel_expand_work(f,a,b);
        else q41_panel_work(f,a,b,1);
        return;
    }
#endif
    q41_panel_work(f,a,b,0);
}

static void q41_panel_prefill_option(Q41Job *c) {
#ifdef MOTY_Q41_FMLAL
    if (c->S >= 16 && c->I <= Q41_PANEL_PREFILL_MAX_INPUT) {
        const char *prefill = moty_getenv("MOTY_Q41_PANEL_PREFILL");
        c->expand_panel_prefill = prefill && !strcmp(prefill,"1");
    }
#endif
}

typedef struct { const Q41Job *input; const uint32_t *table; int fp16; } Q41IndexedHead;

/* Batched and FP32 callers use the regular kernel on one restored panel. */
static __attribute__((noinline,cold)) void q41_indexed_head_restore(
        const Q41IndexedHead *head, const uint8_t *w, int64_t row, int tid) {
    uint8_t panel[(Q41_INDEX_MAX_INPUT/Q41_PANEL_GROUP)*Q41_PANEL_BYTES];
    Q41Job c = *head->input;
    c.y += row*Q41_PANEL_ROWS;
    c.q4 = panel;
    for (int g = 0; g < c.I/Q41_PANEL_GROUP; g++, w += Q41_INDEXED_BYTES) {
        uint8_t *dst = panel+g*Q41_PANEL_BYTES;
        uint16_t ids[4];
        memcpy(ids,w,sizeof(ids));
        for (int r = 0; r < Q41_PANEL_ROWS; r++)
            memcpy(dst+4*r,head->table+ids[r],sizeof(uint32_t));
        memcpy(dst+16,w+8,128);
    }
    Q41FfnJob f = {&c,NULL,head->fp16};
    q41_panel_rows(&f,0,1,tid);
}

static void q41_indexed_head_rows(void *arg, int64_t first, int64_t end, int tid) {
    const Q41IndexedHead *head = arg;
    const Q41Job *c = head->input;
    const int bytes = (c->I/Q41_PANEL_GROUP)*Q41_INDEXED_BYTES;
#ifdef MOTY_Q41_FMLAL
    if (c->S == 1 && head->fp16) {
        for (int64_t r = first; r < end; r++)
            q41_panel_gemv(c,c->q4+r*bytes,c->y+Q41_PANEL_ROWS*r,head->table,1);
        return;
    }
#endif
    for (int64_t r = first; r < end; r++)
        q41_indexed_head_restore(head,c->q4+r*bytes,r,tid);
}
#endif

void moty_q41_indexed_head(float *y, const float *x, const uint8_t *panels,
                          int S, int I, int O) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (I <= 0 || I > Q41_INDEX_MAX_INPUT || I % Q41_PANEL_GROUP ||
        O <= 0 || O % Q41_PANEL_ROWS)
        moty_fail_code(MOTY_FAIL_FORMAT,"Invalid indexed Q4_1 head shape");
    Q41Job c = { .y = y, .x = x, .q4 = panels, .S = S, .I = I, .O = O };
    q41_panel_prefill_option(&c);
    Q41IndexedHead head = {&c,q41_indexed_table(panels,I,O),q41_prepare(&c)};
    moty_par_for(O/Q41_PANEL_ROWS,0,q41_indexed_head_rows,&head);
#else
    moty_fail_code(MOTY_FAIL_FORMAT,"Indexed Q4_1 head requires NEON");
#endif
}

void moty_q41_panels(float *y, const float *x, const uint8_t *panels, const uint8_t *up,
                     int S, int I, int O) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    Q41Job c = { .y = y, .x = x, .q4 = panels, .S = S, .I = I, .O = O };
    if (S == 1 && up) {
        const char *dual = moty_getenv("MOTY_Q41_PANEL_DUAL");
        c.panel_dual = dual && !strcmp(dual,"1");
    }
    q41_panel_prefill_option(&c);
    Q41FfnJob f = { &c, up, q41_prepare(&c) };
    moty_par_for(O/4,0,q41_panel_rows,&f);
#else
    moty_fail_code(MOTY_FAIL_FORMAT,"Q4_1 panels require NEON");
#endif
}

int moty_q41_ffn_team(float *out, float *intermediate, const float *x,
                       const Mat *gate, const Mat *up, const Mat *down) {
#if defined(_OPENMP) && defined(__aarch64__) && defined(__ARM_NEON)
    const char *enable = moty_getenv("MOTY_Q41_FFN_TEAM");
    if (!enable || strcmp(enable,"1") || omp_in_parallel() || omp_get_max_threads() == 1) return 0;
    int packed_gate = gate->gs == Q41_PACKED_GS;
    int packed_down = down->gs == Q41_PACKED_GS;
    if (down->fmt != WF_Q4_1 || !down->q4 || down->sh || down->I != gate->O) return 0;
    Q41Job g = { .y = intermediate, .x = x, .q4 = gate->q4, .S = 1, .I = gate->I,
                 .O = gate->O, .group64 = gate->gs == 64 && up->gs == 64 };
    Q41Job d = { .y = out, .x = intermediate, .q4 = down->q4, .S = 1, .I = down->I,
                 .O = down->O, .group64 = down->gs == 64 };
#ifdef MOTY_Q41_FMLAL
    q41_set_ffn_rows(&g);
#endif
    Q41FfnJob first = { &g, up->q4, q41_prepare(&g) }, second = { &d, NULL, 0 };
    #pragma omp parallel
    {
        int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        int64_t a, b;
        int jobs = packed_gate ? g.O/4 : (g.O-1)/Q41_FFN_ROWS+1;
        moty_par_range(jobs,tid,nt,&a,&b);
        if (a < b) {
            if (packed_gate) q41_panel_rows(&first,a,b,tid);
            else q41_ffn_rows(&first,a,b,tid);
        }
        #pragma omp barrier
        #pragma omp master
        { second.fp16 = q41_prepare(&d); }
        #pragma omp barrier
        moty_par_range(packed_down ? d.O/4 : d.O,tid,nt,&a,&b);
        if (a < b) {
            if (packed_down) q41_panel_rows(&second,a,b,tid);
#ifdef MOTY_Q41_FMLAL
            else if (second.fp16) q41_rows_fp16(&d,a,b,tid);
#endif
            else q41_decode_rows(&d,a,b,tid);
        }
    }
    return 1;
#else
    return 0;
#endif
}

void moty_q41_matmul(float *y, const float *x, const uint8_t *raw, int S, int I, int O, int group64) {
    Q41Job c = { .y = y, .x = x, .q4 = raw, .S = S, .I = I, .O = O, .group64 = group64 };
#if defined(__aarch64__) && defined(__ARM_NEON)
#ifdef MOTY_Q41_FMLAL
    if (S >= 4 && I > O) c.compact_pp = q41_compact_size("MOTY_Q41_COMPACT_DOWN");
    if (q41_prepare(&c)) {
        const char *align = S == 1 && I > O ? moty_getenv("MOTY_Q41_DOWN_ALIGN") : NULL;
        if (align && !strcmp(align,"1")) {
            int head = q41_decode_head(&c);
            int jobs = O <= head ? 1 : (O-head-1)/Q41_CACHE_ROWS+1+(head != 0);
            moty_par_for(jobs,0,q41_decode_aligned,&c);
        } else moty_par_for(O,0,q41_rows_fp16,&c);
        return;
    }
#else
    q41_prepare(&c);
#endif
#endif
#ifdef Q41_LANES
    if (S == 1) { moty_par_for(O, 0, q41_decode_rows, &c); return; }
#endif
    moty_par_for(O, 0, q41_rows, &c);
}

void moty_matmul_q4_1_native(float *y, const float *x, const uint8_t *raw, int S, int I, int O) {
    moty_q41_matmul(y,x,raw,S,I,O,0);
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
