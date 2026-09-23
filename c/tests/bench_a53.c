/* bench_a53.c — memory bandwidth + int4/int8 GEMV kernel candidates for
 * in-order ARMv8.0 cores (Cortex-A53: no SDOT, no fp16 arithmetic).
 * Used to PICK the Q4R4 kernels by measurement (docs/performance.md).
 *
 *   make CC=aarch64-linux-gnu-gcc ARCH=armv8-a libmoty-hw.a libmoty-nn.a
 *   aarch64-linux-gnu-gcc -O3 -fno-tree-vectorize -march=armv8-a -fopenmp -static -I. \
 *       tests/bench_a53.c libmoty-nn.a libmoty-hw.a -lm -o bench_a53
 *   ./bench_a53 bw   [MB]            read bandwidth (with/without PRFM), 1/2/4 threads
 *   ./bench_a53 mac                  int8 SMLAL multiply-accumulate peak
 *   ./bench_a53 gemv [O I MB]        GEMV candidates, weights >> cache
 *   ./bench_a53 drv  O I S [MB]      the real driver (moty_matmul_q4r4_s), S tokens
 *   ./bench_a53 gemm I S             hot prefill GEMM variants (intrinsics vs asm)
 * (-fno-tree-vectorize: GCC 12 ICEs vectorizing some of the scalar
 * reference loops; the kernels are explicit intrinsics anyway.)
 *
 * Every candidate is checked against a scalar reference on the same packed
 * data before it is timed; GB/s counts the weight bytes each variant must
 * stream per pass (nibbles + scales). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <arm_neon.h>
#include "hw/hw.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
static uint64_t rs = 88172645463325252ULL;
static float frnd(void) { rs ^= rs<<13; rs ^= rs>>7; rs ^= rs<<17; return (float)((rs>>11)*(1.0/9007199254740992.0)) - 0.5f; }
static void *amalloc(size_t n) { void *p = NULL; if (posix_memalign(&p, 64, n)) { perror("alloc"); exit(1); } memset(p, 0, n); return p; }

/* ------------------------------------------------------------------ */
/* bandwidth                                                           */
/* ------------------------------------------------------------------ */
/* spin all threads ~ms so the cpufreq governor (interactive) has ramped up */
static void warm(int ms) {
    #pragma omp parallel
    { double t0 = now(); volatile double z = 1; while (now() - t0 < ms*1e-3) z = z*1.0000001 + 1e-9; (void)z; }
}
static int cmpd(const void *a, const void *b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }

/* one read pass over [a, a+n) split across the team; pf = prefetch distance
 * (0 = none), keep = PLDL1KEEP instead of PLDL1STRM */
static uint32_t read_pass(const uint8_t *a, size_t n, int pf, int keep) {
    uint32_t tot = 0;
    #pragma omp parallel reduction(+:tot)
    {
        int t = omp_get_thread_num(), T = omp_get_num_threads();
        size_t s0 = n / T * t, s1 = t == T-1 ? n : n / T * (t+1);
        uint32x4_t acc0 = vdupq_n_u32(0), acc1 = vdupq_n_u32(0);
        for (size_t i = s0; i + 64 <= s1; i += 64) {
            if (pf) { if (keep) __builtin_prefetch(a + i + pf, 0, 3); else __builtin_prefetch(a + i + pf, 0, 0); }
            uint8x16x4_t v = vld1q_u8_x4(a + i);
            acc0 = vpadalq_u16(acc0, vpaddlq_u8(v.val[0]));
            acc1 = vpadalq_u16(acc1, vpaddlq_u8(v.val[1]));
            acc0 = vpadalq_u16(acc0, vpaddlq_u8(v.val[2]));
            acc1 = vpadalq_u16(acc1, vpaddlq_u8(v.val[3]));
        }
        tot += vaddvq_u32(vaddq_u32(acc0, acc1));
    }
    return tot;
}

static void bw(size_t mb) {
    size_t n = mb << 20;
    uint8_t *a = amalloc(n), *b = amalloc(n);
    for (size_t i = 0; i < n; i++) a[i] = (uint8_t)i;
    int ths[3] = {1, 2, 4};
    enum { R = 10 };
    const int cfg_pf[] = {0, 512, 1024, 2048, 1024, 2048}, cfg_keep[] = {0, 0, 0, 0, 1, 1};
    const char *cfg_nm[] = {"read", "pf512strm", "pf1024strm", "pf2048strm", "pf1024keep", "pf2048keep"};
    volatile uint32_t sink = 0;
    for (int ti = 0; ti < 3; ti++) {
        int nt = ths[ti]; omp_set_num_threads(nt);
        printf("bw %zu MB threads %d:", mb, nt);
        for (int c = 0; c < 6; c++) {
            double t[R];
            warm(200);
            for (int rep = 0; rep < R; rep++) { double t0 = now(); sink += read_pass(a, n, cfg_pf[c], cfg_keep[c]); t[rep] = now() - t0; }
            qsort(t, R, sizeof(double), cmpd);
            printf("  %s %.2f/%.2f", cfg_nm[c], n/t[0]/1e9, n/t[R/2]/1e9);
        }
        double tc[R];
        warm(200);
        for (int rep = 0; rep < R; rep++) {
            double t0 = now();
            #pragma omp parallel
            {
                int t = omp_get_thread_num(), T = omp_get_num_threads();
                size_t s0 = n / T * t, s1 = t == T-1 ? n : n / T * (t+1);
                memcpy(b + s0, a + s0, s1 - s0);
            }
            tc[rep] = now() - t0;
        }
        qsort(tc, R, sizeof(double), cmpd);
        printf("  copy(r+w) %.2f/%.2f  GB/s best/median\n", 2.0*n/tc[0]/1e9, 2.0*n/tc[R/2]/1e9);
    }
    (void)sink;
    free(a); free(b);
}

/* ------------------------------------------------------------------ */
/* activation quantization (per group of 32: int8 + f32 scale + sum)   */
/* ------------------------------------------------------------------ */
static void quant_x_g32(const float *x, int I, int8_t *xq, float *xs, int32_t *xsum) {
    for (int g = 0; g < I/32; g++) {
        float amax = 0; for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[g*32+j]));
        float s = amax / 127.f; if (s < 1e-30f) s = 1e-30f;
        float inv = 1.f / s; int32_t sm = 0;
        for (int j = 0; j < 32; j++) { int q = (int)lrintf(x[g*32+j]*inv); xq[g*32+j] = (int8_t)q; sm += q; }
        xs[g] = s; xsum[g] = sm;
    }
}

/* weight int4 group-32 symmetric: q = round(w/d)+8 in [0,15], d = amax/7 */
static void quant_w_g32(const float *w, int I, uint8_t *q, float *d) {
    for (int g = 0; g < I/32; g++) {
        float amax = 0; for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(w[g*32+j]));
        float s = amax / 7.f; if (s < 1e-30f) s = 1e-30f;
        /* round the scale through f16 first so the codes match the stored d */
        s = (float)(__fp16)s;
        for (int j = 0; j < 32; j++) { int v = (int)lrintf(w[g*32+j]/s); if (v < -8) v = -8; if (v > 7) v = 7; q[g*32+j] = (uint8_t)(v+8); }
        d[g] = s;
    }
}

/* ------------------------------------------------------------------ */
/* candidate kernels                                                   */
/* ------------------------------------------------------------------ */

/* Q4R4: 4-row blocks; per group: 4x16 bytes (row r: byte j = q[j] | q[j+16]<<4),
 * scales f16 [nb][4] in a separate stream. y[4]. */
static inline void k_q4r4(const uint8_t *w, const __fp16 *d, const int8_t *xq, const float *xs,
                          const int32_t *xsum, int nb, float *y, int pf) {
    float32x4_t acc = vdupq_n_f32(0);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int g = 0; g < nb; g++) {
        if (pf) __builtin_prefetch(w + pf, 0, 0);
        int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16); xq += 32;
        uint8x16_t b0 = vld1q_u8(w), b1 = vld1q_u8(w+16), b2 = vld1q_u8(w+32), b3 = vld1q_u8(w+48); w += 64;
        int8x16_t l, h; int16x8_t p0, p1, p2, p3;
        #define ROW(b, p) \
            l = vreinterpretq_s8_u8(vandq_u8(b, m4)); h = vreinterpretq_s8_u8(vshrq_n_u8(b, 4)); \
            p = vmull_s8(vget_low_s8(l), vget_low_s8(x0)); p = vmlal_high_s8(p, l, x0); \
            p = vmlal_s8(p, vget_low_s8(h), vget_low_s8(x1)); p = vmlal_high_s8(p, h, x1);
        ROW(b0, p0) ROW(b1, p1) ROW(b2, p2) ROW(b3, p3)
        #undef ROW
        /* |q|<=15, |x|<=127: a lane of p holds 4 products (<=7620); two
         * pairwise adds keep 16 products (<=30480) in int16, then widen */
        int16x8_t t = vpaddq_s16(vpaddq_s16(p0, p1), vpaddq_s16(p2, p3));
        int32x4_t s = vpaddlq_s16(t);
        s = vsubq_s32(s, vdupq_n_s32(8 * xsum[g]));
        float32x4_t sc = vmulq_n_f32(vcvt_f32_f16(vld1_f16(d)), xs[g]); d += 4;
        acc = vfmaq_f32(acc, vcvtq_f32_s32(s), sc);
    }
    vst1q_f32(y, acc);
}

/* Q4R4 + prefetch of the scale stream too (2 streams, both prefetched) */
static inline void k_q4r4pd(const uint8_t *w, const __fp16 *d, const int8_t *xq, const float *xs,
                            const int32_t *xsum, int nb, float *y) {
    float32x4_t acc = vdupq_n_f32(0);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int g = 0; g < nb; g++) {
        __builtin_prefetch(w + 1024, 0, 3);
        if ((g & 7) == 0) __builtin_prefetch(d + 4*8*16, 0, 3);
        int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16); xq += 32;
        uint8x16_t b0 = vld1q_u8(w), b1 = vld1q_u8(w+16), b2 = vld1q_u8(w+32), b3 = vld1q_u8(w+48); w += 64;
        int8x16_t l, h; int16x8_t p0, p1, p2, p3;
        #define ROW(b, p) \
            l = vreinterpretq_s8_u8(vandq_u8(b, m4)); h = vreinterpretq_s8_u8(vshrq_n_u8(b, 4)); \
            p = vmull_s8(vget_low_s8(l), vget_low_s8(x0)); p = vmlal_high_s8(p, l, x0); \
            p = vmlal_s8(p, vget_low_s8(h), vget_low_s8(x1)); p = vmlal_high_s8(p, h, x1);
        ROW(b0, p0) ROW(b1, p1) ROW(b2, p2) ROW(b3, p3)
        int32x4_t s = vpaddlq_s16(vpaddq_s16(vpaddq_s16(p0, p1), vpaddq_s16(p2, p3)));
        s = vsubq_s32(s, vdupq_n_s32(8 * xsum[g]));
        float32x4_t sc = vmulq_n_f32(vcvt_f32_f16(vld1_f16(d)), xs[g]); d += 4;
        acc = vfmaq_f32(acc, vcvtq_f32_s32(s), sc);
    }
    vst1q_f32(y, acc);
}
/* Q4R4 single stream: chunks of 8 groups = 512 B nibbles + 64 B scales */
static inline void k_q4r4i(const uint8_t *w, const int8_t *xq, const float *xs,
                           const int32_t *xsum, int nb, float *y) {
    float32x4_t acc = vdupq_n_f32(0);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int c = 0; c < nb; c += 8) {
        const __fp16 *d = (const __fp16 *)(w + 512);
        for (int g = c; g < c + 8; g++) {
            __builtin_prefetch(w + 1024, 0, 3);
            int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16); xq += 32;
            uint8x16_t b0 = vld1q_u8(w), b1 = vld1q_u8(w+16), b2 = vld1q_u8(w+32), b3 = vld1q_u8(w+48); w += 64;
            int8x16_t l, h; int16x8_t p0, p1, p2, p3;
            ROW(b0, p0) ROW(b1, p1) ROW(b2, p2) ROW(b3, p3)
            int32x4_t s = vpaddlq_s16(vpaddq_s16(vpaddq_s16(p0, p1), vpaddq_s16(p2, p3)));
            s = vsubq_s32(s, vdupq_n_s32(8 * xsum[g]));
            float32x4_t sc = vmulq_n_f32(vcvt_f32_f16(vld1_f16(d)), xs[g]); d += 4;
            acc = vfmaq_f32(acc, vcvtq_f32_s32(s), sc);
        }
        w += 64;                                           /* skip the scale line */
        #undef ROW
    }
    vst1q_f32(y, acc);
}

/* Q4R1: one row, split-nibble groups, 4 groups reduced together */
static inline float k_q4r1(const uint8_t *w, const __fp16 *d, const int8_t *xq, const float *xs,
                           const int32_t *xsum, int nb) {
    float32x4_t acc = vdupq_n_f32(0);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int g = 0; g + 4 <= nb; g += 4) {
        int16x8_t p[4];
        for (int k = 0; k < 4; k++) {
            int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16); xq += 32;
            uint8x16_t b = vld1q_u8(w); w += 16;
            int8x16_t l = vreinterpretq_s8_u8(vandq_u8(b, m4)), h = vreinterpretq_s8_u8(vshrq_n_u8(b, 4));
            int16x8_t q = vmull_s8(vget_low_s8(l), vget_low_s8(x0)); q = vmlal_high_s8(q, l, x0);
            q = vmlal_s8(q, vget_low_s8(h), vget_low_s8(x1)); p[k] = vmlal_high_s8(q, h, x1);
        }
        int32x4_t s = vpaddlq_s16(vpaddq_s16(vpaddq_s16(p[0], p[1]), vpaddq_s16(p[2], p[3])));
        s = vsubq_s32(s, vshlq_n_s32(vld1q_s32(xsum + g), 3));
        float32x4_t sc = vmulq_f32(vcvt_f32_f16(vld1_f16(d + g)), vld1q_f32(xs + g));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(s), sc);
    }
    return vaddvq_f32(acc);
}

/* F32: one row, split-nibble groups dequantized to f32 and FMLA'd with f32 x */
static inline float k_f32r1(const uint8_t *w, const __fp16 *d, const float *x, int nb) {
    float32x4_t acc = vdupq_n_f32(0);
    const uint8x16_t m4 = vdupq_n_u8(0x0F); const int8x16_t e8 = vdupq_n_s8(8);
    for (int g = 0; g < nb; g++) {
        uint8x16_t b = vld1q_u8(w); w += 16;
        int8x16_t l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, m4)), e8);
        int8x16_t h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), e8);
        float32x4_t a = vdupq_n_f32(0);
        int16x8_t v[4] = { vmovl_s8(vget_low_s8(l)), vmovl_high_s8(l), vmovl_s8(vget_low_s8(h)), vmovl_high_s8(h) };
        for (int k = 0; k < 4; k++) {
            a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v[k]))), vld1q_f32(x + 8*k));
            a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_high_s16(v[k])), vld1q_f32(x + 8*k + 4));
        }
        x += 32;
        acc = vfmaq_n_f32(acc, a, (float)d[g]);
    }
    return vaddvq_f32(acc);
}

/* ------------------------------------------------------------------ */
/* gemv bench                                                          */
/* ------------------------------------------------------------------ */
/* scalar references (GCC 12 aarch64 ICEs vectorizing these: keep them scalar) */
__attribute__((optimize("no-tree-vectorize")))
static void ref_q4(const uint8_t *qc, const float *dg, const int8_t *xq, const float *xs, int O, int I, double *ref) {
    int nb = I / 32;
    for (int o = 0; o < O; o++) {
        double a = 0;
        for (int g = 0; g < nb; g++) {
            int32_t s = 0; for (int j = 0; j < 32; j++) s += ((int)qc[(size_t)o*I+g*32+j] - 8) * xq[g*32+j];
            a += (double)(float)(__fp16)dg[(size_t)o*nb+g] * xs[g] * s;
        }
        ref[o] = a;
    }
}
__attribute__((optimize("no-tree-vectorize")))
static void ref_f32(const uint8_t *qc, const float *dg, const float *x, int O, int I, double *ref) {
    int nb = I / 32;
    for (int o = 0; o < O; o++) {
        double a = 0;
        for (int i = 0; i < I; i++) a += (double)(float)(__fp16)dg[(size_t)o*nb+i/32] * ((int)qc[(size_t)o*I+i]-8) * x[i];
        ref[o] = a;
    }
}

static void gemv(int O, int I, size_t mb) {
    int nb = I / 32;
    size_t q4_bytes = (size_t)O*I/2 + (size_t)O*nb*2;          /* nibbles + f16 scales */
    int nmat = (int)((mb << 20) / q4_bytes); if (nmat < 1) nmat = 1;
    printf("gemv O=%d I=%d: %d matrices, int4 %.1f MB, int8 %.1f MB\n", O, I, nmat,
           nmat*q4_bytes/1048576.0, nmat*(double)O*I/1048576.0);
    /* source floats for one matrix, reused for all copies (content does not
     * matter for bandwidth; numerics are checked on matrix 0) */
    float *wf = malloc(sizeof(float)*O*I), *x = malloc(sizeof(float)*I);
    for (size_t i = 0; i < (size_t)O*I; i++) wf[i] = frnd();
    for (int i = 0; i < I; i++) x[i] = frnd() * (i % 97 == 0 ? 20.f : 1.f);   /* a few outliers */
    uint8_t *qc = malloc((size_t)O*I); float *dg = malloc(sizeof(float)*O*nb);
    for (int o = 0; o < O; o++) quant_w_g32(wf + (size_t)o*I, I, qc + (size_t)o*I, dg + (size_t)o*nb);
    int8_t *xq = amalloc(I); float *xs = amalloc(sizeof(float)*nb); int32_t *xsum = amalloc(4*nb);
    quant_x_g32(x, I, xq, xs, xsum);
    double *ref = malloc(sizeof(double)*O);
    ref_q4(qc, dg, xq, xs, O, I, ref);
    /* --- pack Q4R4 --- */
    uint8_t **w4 = malloc(sizeof(void*)*nmat); __fp16 **d4 = malloc(sizeof(void*)*nmat);
    for (int m = 0; m < nmat; m++) {
        w4[m] = amalloc((size_t)O*I/2); d4[m] = amalloc((size_t)O*nb*2);
        for (int ob = 0; ob < O/4; ob++) for (int g = 0; g < nb; g++) for (int r = 0; r < 4; r++) {
            int o = ob*4 + r; uint8_t *dst = w4[m] + ((size_t)ob*nb + g)*64 + r*16;
            const uint8_t *src = qc + (size_t)o*I + g*32;
            for (int j = 0; j < 16; j++) dst[j] = src[j] | (src[j+16] << 4);
            d4[m][((size_t)ob*nb + g)*4 + r] = (__fp16)dg[(size_t)o*nb + g];
        }
    }
    /* --- pack Q4R4 interleaved (8 groups + scale line) --- */
    uint8_t **wi = malloc(sizeof(void*)*nmat);
    for (int m = 0; m < nmat; m++) {
        wi[m] = amalloc((size_t)O/4*nb/8*576);
        for (int ob = 0; ob < O/4; ob++) for (int c = 0; c < nb/8; c++) {
            uint8_t *dst = wi[m] + ((size_t)ob*(nb/8) + c)*576;
            memcpy(dst, w4[m] + ((size_t)ob*nb + c*8)*64, 512);
            memcpy(dst + 512, d4[m] + ((size_t)ob*nb + c*8)*4, 64);
        }
    }
    /* --- pack Q4R1 (row-major split nibbles) --- */
    uint8_t **w1 = malloc(sizeof(void*)*nmat); __fp16 **d1 = malloc(sizeof(void*)*nmat);
    for (int m = 0; m < nmat; m++) {
        w1[m] = amalloc((size_t)O*I/2); d1[m] = amalloc((size_t)O*nb*2);
        for (int o = 0; o < O; o++) for (int g = 0; g < nb; g++) {
            uint8_t *dst = w1[m] + (size_t)o*I/2 + g*16; const uint8_t *src = qc + (size_t)o*I + g*32;
            for (int j = 0; j < 16; j++) dst[j] = src[j] | (src[j+16] << 4);
            d1[m][(size_t)o*nb + g] = (__fp16)dg[(size_t)o*nb + g];
        }
    }
    /* --- existing moty int4 (pairs, per-row scale) and int8 (per-row) --- */
    uint8_t **we4 = malloc(sizeof(void*)*nmat); int8_t **we8 = malloc(sizeof(void*)*nmat);
    float *sr = malloc(sizeof(float)*O);
    int nm8 = nmat > 1 ? (nmat + 1) / 2 : 1;   /* int8 is 2x bigger: same MB budget */
    for (int m = 0; m < nmat; m++) {
        we4[m] = amalloc((size_t)O*I/2);
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i += 2)
            we4[m][(size_t)o*I/2 + i/2] = qc[(size_t)o*I+i] | (qc[(size_t)o*I+i+1] << 4);
    }
    for (int m = 0; m < nm8; m++) {
        we8[m] = amalloc((size_t)O*I);
        for (size_t i = 0; i < (size_t)O*I; i++) we8[m][i] = (int8_t)(((int)qc[i]-8)*16);
    }
    for (int o = 0; o < O; o++) sr[o] = 1.f;
    int8_t *xr = amalloc(I); float xrs = qrow_i8(x, xr, I);
    float *y = amalloc(sizeof(float)*O);

    /* numerics */
    { double md = 0, mr = 0;
      for (int ob = 0; ob < O/4; ob++) k_q4r4(w4[0] + (size_t)ob*nb*64, d4[0] + (size_t)ob*nb*4, xq, xs, xsum, nb, y + ob*4, 0);
      for (int o = 0; o < O; o++) { md = fmax(md, fabs(y[o]-ref[o])); mr = fmax(mr, fabs(ref[o])); }
      printf("  check q4r4: max|d| %.3g (max|ref| %.3g)\n", md, mr);
      { double m2 = 0, m3 = 0;
        for (int ob = 0; ob < O/4; ob++) k_q4r4pd(w4[0] + (size_t)ob*nb*64, d4[0] + (size_t)ob*nb*4, xq, xs, xsum, nb, y + ob*4);
        for (int o = 0; o < O; o++) m2 = fmax(m2, fabs(y[o]-ref[o]));
        for (int ob = 0; ob < O/4; ob++) k_q4r4i(wi[0] + (size_t)ob*(nb/8)*576, xq, xs, xsum, nb, y + ob*4);
        for (int o = 0; o < O; o++) m3 = fmax(m3, fabs(y[o]-ref[o]));
        printf("  check q4r4 pf w+d: %.3g  1-stream: %.3g\n", m2, m3); }
      md = 0;
      for (int o = 0; o < O; o++) { y[o] = k_q4r1(w1[0] + (size_t)o*I/2, d1[0] + (size_t)o*nb, xq, xs, xsum, nb); md = fmax(md, fabs(y[o]-ref[o])); }
      printf("  check q4r1: max|d| %.3g\n", md);
      md = 0; double mf = 0;
      double *rf = malloc(sizeof(double)*O);             /* f32 path vs exact f32-activation dot */
      ref_f32(qc, dg, x, O, I, rf);
      for (int o = 0; o < O; o++) {
          y[o] = k_f32r1(w1[0] + (size_t)o*I/2, d1[0] + (size_t)o*nb, x, nb); md = fmax(md, fabs(y[o]-rf[o])); mf = fmax(mf, fabs(rf[o])); }
      free(rf);
      printf("  check f32r1: max|d| %.3g (max|ref| %.3g)\n", md, mf);
    }

    int ths[3] = {1, 2, 4};
    enum { NV = 11, R = 12 };
    const char *names[NV] = {"q4r4", "q4r4+pf1024k", "q4r4+pf2048k", "q4r4 dyn8", "q4r4 dyn8+pf1024k",
                             "q4r1", "moty-i4(dot_i4i8)", "moty-i8(dot_i8i8)", "moty-i8 dyn32",
                             "q4r4 pf w+d", "q4r4 1-stream"};
    for (int v = 0; v < NV; v++) {
        for (int ti = 0; ti < 3; ti++) {
            omp_set_num_threads(ths[ti]);
            int nmv = (v == 7 || v == 8) ? nm8 : nmat;
            double bytes = (v == 7 || v == 8) ? (double)O*I + O*4.0 : v == 6 ? (double)O*I/2 + O*4.0 : (double)q4_bytes;
            double t[R];
            warm(150);
            for (int rep = 0; rep < R; rep++) {
                double t0 = now();
                for (int m = 0; m < nmv; m++) {
                    switch (v) {
                    case 0: case 1: case 2: {
                        int pf = v == 0 ? 0 : v == 1 ? 1024 : 2048;
                        #pragma omp parallel for schedule(static)
                        for (int ob = 0; ob < O/4; ob++)
                            k_q4r4(w4[m] + (size_t)ob*nb*64, d4[m] + (size_t)ob*nb*4, xq, xs, xsum, nb, y + ob*4, pf);
                        break; }
                    case 3: case 4: {
                        int pf = v == 3 ? 0 : 1024;
                        #pragma omp parallel for schedule(dynamic, 8)
                        for (int ob = 0; ob < O/4; ob++)
                            k_q4r4(w4[m] + (size_t)ob*nb*64, d4[m] + (size_t)ob*nb*4, xq, xs, xsum, nb, y + ob*4, pf);
                        break; }
                    case 5:
                        #pragma omp parallel for schedule(static)
                        for (int o = 0; o < O; o++) y[o] = k_q4r1(w1[m] + (size_t)o*I/2, d1[m] + (size_t)o*nb, xq, xs, xsum, nb);
                        break;
                    case 6:
                        #pragma omp parallel for schedule(static)
                        for (int o = 0; o < O; o++) y[o] = sr[o] * xrs * (float)dot_i4i8(we4[m] + (size_t)o*I/2, xr, I);
                        break;
                    case 7:
                        #pragma omp parallel for schedule(static)
                        for (int o = 0; o < O; o++) y[o] = sr[o] * xrs * (float)dot_i8i8(we8[m] + (size_t)o*I, xr, I);
                        break;
                    case 9:
                        #pragma omp parallel for schedule(static)
                        for (int ob = 0; ob < O/4; ob++)
                            k_q4r4pd(w4[m] + (size_t)ob*nb*64, d4[m] + (size_t)ob*nb*4, xq, xs, xsum, nb, y + ob*4);
                        break;
                    case 10:
                        #pragma omp parallel for schedule(static)
                        for (int ob = 0; ob < O/4; ob++)
                            k_q4r4i(wi[m] + (size_t)ob*(nb/8)*576, xq, xs, xsum, nb, y + ob*4);
                        break;
                    case 8:
                        #pragma omp parallel for schedule(dynamic, 32)
                        for (int o = 0; o < O; o++) y[o] = sr[o] * xrs * (float)dot_i8i8(we8[m] + (size_t)o*I, xr, I);
                        break;
                    }
                }
                t[rep] = (now() - t0) / nmv;
            }
            qsort(t, R, sizeof(double), cmpd);
            printf("  %-20s th=%d  best %7.3f ms  %5.2f GB/s %5.2f Gw/s | median %7.3f ms  %5.2f GB/s %5.2f Gw/s\n",
                   names[v], ths[ti], t[0]*1e3, bytes/t[0]/1e9, (double)O*I/t[0]/1e9,
                   t[R/2]*1e3, bytes/t[R/2]/1e9, (double)O*I/t[R/2]/1e9);
        }
    }
    fflush(stdout);
}

/* integer MAC peak: independent SMLAL/SMLAL2 chains (8x int8 MACs each),
 * enough accumulators to cover latency on an in-order core */
__attribute__((noinline)) static int32_t mac_loop(long iters, int8x16_t a, int8x16_t b) {
    int16x8_t c0 = vdupq_n_s16(0), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    for (long i = 0; i < iters; i++) {
        c0 = vmlal_s8(c0, vget_low_s8(a), vget_low_s8(b)); c1 = vmlal_high_s8(c1, a, b);
        c2 = vmlal_s8(c2, vget_low_s8(b), vget_low_s8(a)); c3 = vmlal_high_s8(c3, b, a);
        c4 = vmlal_s8(c4, vget_low_s8(a), vget_low_s8(a)); c5 = vmlal_high_s8(c5, a, a);
        c6 = vmlal_s8(c6, vget_low_s8(b), vget_low_s8(b)); c7 = vmlal_high_s8(c7, b, b);
        __asm__ volatile("" : "+w"(c0), "+w"(c1), "+w"(c2), "+w"(c3), "+w"(c4), "+w"(c5), "+w"(c6), "+w"(c7));
    }
    int16x8_t s = vaddq_s16(vaddq_s16(vaddq_s16(c0, c1), vaddq_s16(c2, c3)), vaddq_s16(vaddq_s16(c4, c5), vaddq_s16(c6, c7)));
    return vaddlvq_s16(s);
}
static void macpeak(void) {
    int8x16_t a = vdupq_n_s8(1), b = vdupq_n_s8(0);
    long iters = 20000000; volatile int32_t sink = 0;
    int ths[3] = {1, 2, 4};
    for (int ti = 0; ti < 3; ti++) {
        omp_set_num_threads(ths[ti]); warm(200);
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            double t0 = now();
            #pragma omp parallel
            { int32_t r = mac_loop(iters, a, b); if (r == 12345) sink = r; }
            double dt = now() - t0; if (dt < best) best = dt;
        }
        double macs = (double)iters * 8 * 8 * ths[ti];
        printf("macpeak threads %d: %.2f GMAC/s (int8 SMLAL, %.2f MAC/cycle/core at 1.608 GHz)\n",
               ths[ti], macs/best/1e9, macs/best/1.608e9/ths[ti]);
    }
    (void)sink;
}

/* ---- prefill GEMM prototypes: 4 rows x 4 tokens, one group unpacked once ---- */
/* one token's 4-row group dot, fixed temporaries v16-v23 (no spills of the
 * caller's accumulators): x at xp (32 B), corr = 8*xsum as int32 GP reg,
 * xs as float GP reg; acc += (dot - corr) * (dv * xs) */
#define TOK_ASM(acc, xp, corr, xsv) \
    __asm__ volatile( \
        "ldp q16, q17, [%[x]]\n\t" \
        "smull  v18.8h, %[l0].8b, v16.8b\n\t" \
        "smull  v19.8h, %[l1].8b, v16.8b\n\t" \
        "smull  v20.8h, %[l2].8b, v16.8b\n\t" \
        "smull  v21.8h, %[l3].8b, v16.8b\n\t" \
        "smlal2 v18.8h, %[l0].16b, v16.16b\n\t" \
        "smlal2 v19.8h, %[l1].16b, v16.16b\n\t" \
        "smlal2 v20.8h, %[l2].16b, v16.16b\n\t" \
        "smlal2 v21.8h, %[l3].16b, v16.16b\n\t" \
        "smlal  v18.8h, %[h0].8b, v17.8b\n\t" \
        "smlal  v19.8h, %[h1].8b, v17.8b\n\t" \
        "smlal  v20.8h, %[h2].8b, v17.8b\n\t" \
        "smlal  v21.8h, %[h3].8b, v17.8b\n\t" \
        "smlal2 v18.8h, %[h0].16b, v17.16b\n\t" \
        "smlal2 v19.8h, %[h1].16b, v17.16b\n\t" \
        "smlal2 v20.8h, %[h2].16b, v17.16b\n\t" \
        "smlal2 v21.8h, %[h3].16b, v17.16b\n\t" \
        "dup    v22.4s, %w[c]\n\t" \
        "addp   v18.8h, v18.8h, v19.8h\n\t" \
        "addp   v20.8h, v20.8h, v21.8h\n\t" \
        "dup    v23.4s, %w[s]\n\t" \
        "addp   v18.8h, v18.8h, v20.8h\n\t" \
        "fmul   v23.4s, v23.4s, %[dv].4s\n\t" \
        "saddlp v18.4s, v18.8h\n\t" \
        "sub    v18.4s, v18.4s, v22.4s\n\t" \
        "scvtf  v18.4s, v18.4s\n\t" \
        "fmla   %[a].4s, v18.4s, v23.4s\n\t" \
        : [a] "+w"(acc) \
        : [x] "r"(xp), [c] "r"(corr), [s] "r"(xsv), [dv] "w"(dv), \
          [l0] "w"(l0), [l1] "w"(l1), [l2] "w"(l2), [l3] "w"(l3), \
          [h0] "w"(h0), [h1] "w"(h1), [h2] "w"(h2), [h3] "w"(h3) \
        : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory")

static void gemm4_asm(const uint8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                      const int32_t *xsum, int nb, float *y, int ys) {
    int I = nb * 32;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int g = 0; g < nb; g++) {
        uint8x16_t b0 = vld1q_u8(w), b1 = vld1q_u8(w+16), b2 = vld1q_u8(w+32), b3 = vld1q_u8(w+48); w += 64;
        int8x16_t l0 = vreinterpretq_s8_u8(vandq_u8(b0, m4)), h0 = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4));
        int8x16_t l1 = vreinterpretq_s8_u8(vandq_u8(b1, m4)), h1 = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
        int8x16_t l2 = vreinterpretq_s8_u8(vandq_u8(b2, m4)), h2 = vreinterpretq_s8_u8(vshrq_n_u8(b2, 4));
        int8x16_t l3 = vreinterpretq_s8_u8(vandq_u8(b3, m4)), h3 = vreinterpretq_s8_u8(vshrq_n_u8(b3, 4));
        float32x4_t dv = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(d))); d += 4;
        float s0 = xs[g], s1 = xs[nb+g], s2 = xs[2*nb+g], s3 = xs[3*nb+g];
        uint32_t f0, f1, f2, f3; memcpy(&f0, &s0, 4); memcpy(&f1, &s1, 4); memcpy(&f2, &s2, 4); memcpy(&f3, &s3, 4);
        TOK_ASM(a0, xq + g*32,       8*xsum[g],        f0);
        TOK_ASM(a1, xq + I + g*32,   8*xsum[nb+g],     f1);
        TOK_ASM(a2, xq + 2*I + g*32, 8*xsum[2*nb+g],   f2);
        TOK_ASM(a3, xq + 3*I + g*32, 8*xsum[3*nb+g],   f3);
    }
    vst1q_f32(y, a0); vst1q_f32(y + ys, a1); vst1q_f32(y + 2*ys, a2); vst1q_f32(y + 3*ys, a3);
}

/* 4 rows x 8 tokens: each unpacked group reused for 8 tokens */
static void gemm8_intr(const uint8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                       const int32_t *xsum, int nb, float *y, int ys) {
    int I = nb * 32;
    float32x4_t a[8];
    for (int t = 0; t < 8; t++) a[t] = vdupq_n_f32(0);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int g = 0; g < nb; g++) {
        uint8x16_t b0 = vld1q_u8(w), b1 = vld1q_u8(w+16), b2 = vld1q_u8(w+32), b3 = vld1q_u8(w+48); w += 64;
        int8x16_t l0 = vreinterpretq_s8_u8(vandq_u8(b0, m4)), h0 = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4));
        int8x16_t l1 = vreinterpretq_s8_u8(vandq_u8(b1, m4)), h1 = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
        int8x16_t l2 = vreinterpretq_s8_u8(vandq_u8(b2, m4)), h2 = vreinterpretq_s8_u8(vshrq_n_u8(b2, 4));
        int8x16_t l3 = vreinterpretq_s8_u8(vandq_u8(b3, m4)), h3 = vreinterpretq_s8_u8(vshrq_n_u8(b3, 4));
        float32x4_t dv = vcvt_f32_f16(vld1_f16((const __fp16 *)d)); d += 4;
        for (int t = 0; t < 8; t++) {
            const int8_t *xp = xq + (size_t)t*I + g*32;
            int8x16_t x0 = vld1q_s8(xp), x1 = vld1q_s8(xp + 16);
            int16x8_t p0, p1, p2, p3;
            #define RW(l, h, p) p = vmull_s8(vget_low_s8(l), vget_low_s8(x0)); p = vmlal_high_s8(p, l, x0); \
                                p = vmlal_s8(p, vget_low_s8(h), vget_low_s8(x1)); p = vmlal_high_s8(p, h, x1);
            RW(l0, h0, p0) RW(l1, h1, p1) RW(l2, h2, p2) RW(l3, h3, p3)
            #undef RW
            int32x4_t sv = vpaddlq_s16(vpaddq_s16(vpaddq_s16(p0, p1), vpaddq_s16(p2, p3)));
            sv = vsubq_s32(sv, vdupq_n_s32(8 * xsum[(size_t)t*nb + g]));
            a[t] = vfmaq_f32(a[t], vcvtq_f32_s32(sv), vmulq_n_f32(dv, xs[(size_t)t*nb + g]));
        }
    }
    for (int t = 0; t < 8; t++) vst1q_f32(y + (size_t)t*ys, a[t]);
}

/* hot GEMM microbench: lib kernel (intrinsics) vs the asm prototype */
static void gemmbench(int I, int S) {
    int nb = I / 32, O = 64;
    uint8_t *w = amalloc((size_t)O/4*nb*64); uint16_t *d = amalloc((size_t)O/4*nb*8);
    for (size_t i = 0; i < (size_t)O/4*nb*64; i++) w[i] = (uint8_t)(frnd()*512);
    for (size_t i = 0; i < (size_t)O/4*nb*4; i++) d[i] = 0x2000 + (i % 64);
    int8_t *xq = amalloc((size_t)S*I); float *xs = amalloc(sizeof(float)*S*nb); int32_t *xm = amalloc(4*S*nb);
    float *xf = malloc(sizeof(float)*I);
    for (int t = 0; t < S; t++) { for (int i = 0; i < I; i++) xf[i] = frnd(); moty_hw_quant_g32(xf, I, xq + (size_t)t*I, xs + t*nb, xm + t*nb); }
    float *y0 = amalloc(sizeof(float)*S*O), *y1 = amalloc(sizeof(float)*S*O);
    for (int ob = 0; ob < O/4; ob++) {
        moty_hw_q4r4_gemm(w + (size_t)ob*nb*64, d + (size_t)ob*nb*4, xq, xs, xm, nb, S, y0 + ob*4, O);
        for (int t = 0; t + 4 <= S; t += 4)
            gemm4_asm(w + (size_t)ob*nb*64, d + (size_t)ob*nb*4, xq + (size_t)t*I, xs + t*nb, xm + t*nb, nb, y1 + (size_t)t*O + ob*4, O);
    }
    double md = 0, mr = 0; for (int i = 0; i < S*O; i++) { md = fmax(md, fabs(y0[i]-y1[i])); mr = fmax(mr, fabs(y0[i])); }
    printf("gemm I=%d S=%d: asm vs lib max|d| %.3g (max %.3g)\n", I, S, md, mr);
    for (int ob = 0; ob < O/4; ob++) for (int t = 0; t + 8 <= S; t += 8)
        gemm8_intr(w + (size_t)ob*nb*64, d + (size_t)ob*nb*4, xq + (size_t)t*I, xs + t*nb, xm + t*nb, nb, y1 + (size_t)t*O + ob*4, O);
    md = 0; for (int i = 0; i < S*O; i++) md = fmax(md, fabs(y0[i]-y1[i]));
    printf("gemm8 vs lib max|d| %.3g\n", md);
    warm(150);
    for (int v = 0; v < 3; v++) {
        double best = 1e9;
        for (int rep = 0; rep < 20; rep++) {
            double t0 = now();
            for (int ob = 0; ob < O/4; ob++) {
                if (v == 0) moty_hw_q4r4_gemm(w + (size_t)ob*nb*64, d + (size_t)ob*nb*4, xq, xs, xm, nb, S, y0 + ob*4, O);
                else if (v == 2) for (int t = 0; t + 8 <= S; t += 8)
                    gemm8_intr(w + (size_t)ob*nb*64, d + (size_t)ob*nb*4, xq + (size_t)t*I, xs + t*nb, xm + t*nb, nb, y1 + (size_t)t*O + ob*4, O);
                else for (int t = 0; t + 4 <= S; t += 4)
                    gemm4_asm(w + (size_t)ob*nb*64, d + (size_t)ob*nb*4, xq + (size_t)t*I, xs + t*nb, xm + t*nb, nb, y1 + (size_t)t*O + ob*4, O);
            }
            double dt = now() - t0; if (dt < best) best = dt;
        }
        double mac = (double)O*I*S;
        printf("  %-10s 1 thread: %.2f GMAC/s (%.2f MAC/cycle @1.608GHz)\n", v == 2 ? "gemm8" : v ? "asm" : "lib", mac/best/1e9, mac/best/1.608e9);
    }
}

/* the real moty driver (nn/matmul.c moty_matmul_q4r4_s): decode S=1 and
 * prefill S tokens on model shapes; GMAC/s and weight GB/s */
#include "nn/nn.h"
static void drv(int O, int I, int S, size_t mb) {
    int nb = I/32, O4 = (O+3)/4;
    size_t bytes = (size_t)O4*nb*(64+8);
    int nmat = (int)((mb << 20) / bytes); if (nmat < 1) nmat = 1;
    float *W = malloc(sizeof(float)*4*I), *x = malloc(sizeof(float)*S*I), *y = malloc(sizeof(float)*S*O);
    for (int i = 0; i < 4*I; i++) W[i] = frnd();
    for (int i = 0; i < S*I; i++) x[i] = frnd();
    uint8_t **q = malloc(sizeof(void*)*nmat); uint16_t **d = malloc(sizeof(void*)*nmat);
    for (int m = 0; m < nmat; m++) {
        q[m] = amalloc((size_t)O4*nb*64); d[m] = amalloc((size_t)O4*nb*8);
        for (int b = 0; b < O4; b++) moty_pack_q4r4_block(W, 4, I, q[m] + (size_t)b*nb*64, d[m] + (size_t)b*nb*4);
    }
    int ths[3] = {1, 2, 4};
    for (int ti = 0; ti < 3; ti++) {
        omp_set_num_threads(ths[ti]); warm(150);
        enum { R = 8 }; double t[R];
        for (int rep = 0; rep < R; rep++) {
            double t0 = now();
            for (int m = 0; m < nmat; m++) moty_matmul_q4r4_s(y, x, q[m], d[m], S, I, O);
            t[rep] = (now() - t0) / nmat;
        }
        qsort(t, R, sizeof(double), cmpd);
        printf("drv O=%d I=%d S=%d mats=%d th=%d: best %.3f ms (%.2f GMAC/s, %.2f GB/s) median %.3f ms (%.2f GMAC/s, %.2f GB/s)\n",
               O, I, S, nmat, ths[ti], t[0]*1e3, (double)O*I*S/t[0]/1e9, bytes/t[0]/1e9,
               t[R/2]*1e3, (double)O*I*S/t[R/2]/1e9, bytes/t[R/2]/1e9);
    }
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "mac")) { macpeak(); return 0; }
    if (argc > 1 && !strcmp(argv[1], "gemm")) { gemmbench(atoi(argv[2]), atoi(argv[3])); return 0; }
    if (argc > 1 && !strcmp(argv[1], "drv")) {
        drv(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), argc > 5 ? atoi(argv[5]) : 48); return 0; }
    if (argc > 1 && !strcmp(argv[1], "bw")) { bw(argc > 2 ? atoi(argv[2]) : 64); return 0; }
    if (argc > 1 && !strcmp(argv[1], "gemv")) {
        int O = argc > 2 ? atoi(argv[2]) : 4608, I = argc > 3 ? atoi(argv[3]) : 1536;
        gemv(O, I, argc > 4 ? atoi(argv[4]) : 48);
        return 0;
    }
    fprintf(stderr, "usage: %s bw [MB] | mac | gemv [O I MB] | drv O I S [MB] | gemm I S\n", argv[0]);
    return 1;
}
