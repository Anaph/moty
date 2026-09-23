/* Q4R4 int4 kernels (hw/hw_q4r4.h) + packing (nn/quant.c) + matmul driver.
 * Plain C, 0 = ok / 1 = fail; gtest glue in q4r4_gtest.cc. On hosts without
 * the NEON tier the public kernels ARE the references, so these tests check
 * the format/packing/driver; on aarch64 they check NEON vs the scalar
 * reference. Cross-build a standalone runner for the target with
 *   aarch64-linux-gnu-gcc -O2 -march=armv8-a -fopenmp -static -I. \
 *     -DQ4R4_TEST_MAIN tests/q4r4_tests.c libmoty-nn.a libmoty-hw.a -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nn/nn.h"

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

static uint64_t q4_rng = 1234567;
static float q4_frnd(void) {
    q4_rng ^= q4_rng << 13; q4_rng ^= q4_rng >> 7; q4_rng ^= q4_rng << 17;
    return (float)((q4_rng >> 11) * (1.0 / 9007199254740992.0)) - 0.5f;
}

/* f16 <-> f32: every finite half survives the round trip exactly */
int q4_f16_roundtrip(void) {
    for (uint32_t h = 0; h < 65536; h++) {
        if (((h >> 10) & 0x1f) == 0x1f) continue;             /* inf/nan */
        float f = moty_hw_f16_to_f32((uint16_t)h);
        uint16_t b = moty_f32_to_f16(f);
        if (b != h && !(f == 0.f && (b & 0x7fff) == 0)) { fprintf(stderr, "h=%04x f=%g back=%04x\n", h, f, b); return 1; }
    }
    CHECK(moty_f32_to_f16(1.0f) == 0x3c00);
    CHECK(moty_f32_to_f16(65504.f) == 0x7bff);
    CHECK(moty_f32_to_f16(1e6f) == 0x7c00);
    CHECK(moty_f32_to_f16(1.00048828125f) == 0x3c00);       /* tie -> even */
    CHECK(moty_f32_to_f16(1.00146484375f) == 0x3c02);       /* tie -> even (up) */
    return 0;
}

/* group-32 activation quantizer: kernel == reference, bit for bit */
int q4_quant_g32_exact(void) {
    enum { I = 32*37 };
    float x[I]; int8_t q0[I], q1[I]; float s0[I/32], s1[I/32]; int32_t m0[I/32], m1[I/32];
    for (int i = 0; i < I; i++) x[i] = q4_frnd() * (i % 53 == 0 ? 40.f : 3.f);
    for (int i = 64; i < 96; i++) x[i] = 0.f;                /* an all-zero group */
    moty_hw_quant_g32(x, I, q0, s0, m0);
    moty_hw_quant_g32_ref(x, I, q1, s1, m1);
    CHECK(!memcmp(q0, q1, I) && !memcmp(s0, s1, sizeof s0) && !memcmp(m0, m1, sizeof m0));
    for (int g = 0; g < I/32; g++) {
        int32_t sm = 0; int amax = 0;
        for (int j = 0; j < 32; j++) { sm += q0[g*32+j]; if (abs(q0[g*32+j]) > amax) amax = abs(q0[g*32+j]); }
        CHECK(sm == m0[g] && amax <= 127);
        if (g != 2) CHECK(amax == 127);                        /* the group max maps to 127 */
    }
    return 0;
}

/* packing: codes in range, the group max is represented exactly (no clip),
 * every other weight within half a step */
int q4_pack_noclip(void) {
    enum { I = 96 };
    float w[3*I];
    for (int i = 0; i < 3*I; i++) w[i] = q4_frnd();
    w[5] = 2.f; w[I+40] = -3.f; w[2*I+70] = 0.9f; w[2*I+71] = -0.8f;   /* outliers of both signs */
    uint8_t blk[3*64]; uint16_t d[3*4];
    moty_pack_q4r4_block(w, 3, I, blk, d);
    for (int r = 0; r < 4; r++) for (int g = 0; g < 3; g++) {
        const uint8_t *b = blk + g*64 + r*16;
        float dd = moty_hw_f16_to_f32(d[g*4 + r]);
        if (r == 3) { CHECK(dd == 0.f); for (int j = 0; j < 16; j++) CHECK(b[j] == 0x88); continue; }
        float mx = 0, opp = 0;
        for (int j = 0; j < 32; j++) if (fabsf(w[r*I + g*32+j]) > fabsf(mx)) mx = w[r*I + g*32+j];
        for (int j = 0; j < 32; j++) { float x = w[r*I + g*32+j]; if ((x > 0) != (mx > 0) && fabsf(x) > opp) opp = fabsf(x); }
        for (int j = 0; j < 32; j++) {
            int q = j < 16 ? (b[j] & 15) : (b[j-16] >> 4);
            float v = (q - 8) * dd, x = w[r*I + g*32+j];
            CHECK(fabsf(v - x) <= 0.5f * fabsf(dd) * 1.001f + 1e-7f);
        }
        /* finest step that clips neither sign; mx on the -8 side */
        float dm = fmaxf(fabsf(mx) / 8.f, opp / 7.f);
        CHECK(fabsf(fabsf(dd) - dm) <= dm * 1e-3f && (dd < 0) == (mx > 0));
    }
    return 0;
}

/* integer sums exact: scales 1, so the float result equals Σ(q-8)x exactly.
 * NS=6 covers the 4-token GEMM and the GEMV tail, NB=7 an odd group count;
 * row 1 at codes 15 against x = ±127 is the int16 lane worst case. */
int q4_gemm_int_exact(void) {
    enum { NB = 7, I = NB*32, NS = 6 };
    uint8_t w[NB*64]; uint16_t d[NB*4];
    for (int i = 0; i < NB*64; i++) w[i] = (uint8_t)(q4_frnd() * 512);
    for (int i = 0; i < NB*4; i++) d[i] = 0x3c00;             /* 1.0 */
    w[0] = 0xff; w[1] = 0x00;                                  /* extreme codes */
    for (int g = 0; g < NB; g++) memset(w + g*64 + 16, 0xff, 16);   /* row 1: all codes 15 */
    int8_t xq[NS*I]; float xs[NS*NB]; int32_t xm[NS*NB];
    for (int t = 0; t < NS; t++) for (int g = 0; g < NB; g++) {
        int32_t s = 0;
        for (int j = 0; j < 32; j++) {
            int v = (int)(q4_frnd() * 254);
            if (t == 0) v = (j & 1) ? 127 : -127;
            if (t == 1 || t == 5) v = 127;                     /* int16 lane limit: 16 x 15 x 127 */
            if (t == 2) v = -127;
            xq[t*I+g*32+j] = (int8_t)v; s += v; }
        xs[t*NB+g] = 1.f; xm[t*NB+g] = s;
    }
    float y[NS*4], yr[NS*4];
    moty_hw_q4r4_gemm(w, d, xq, xs, xm, NB, NS, y, 4);
    moty_hw_q4r4_gemm_ref(w, d, xq, xs, xm, NB, NS, yr, 4);
    for (int t = 0; t < NS; t++) for (int r = 0; r < 4; r++) {
        int64_t e = 0;
        for (int g = 0; g < NB; g++) for (int j = 0; j < 16; j++) {
            uint8_t b = w[g*64 + r*16 + j];
            e += ((b & 15) - 8) * xq[t*I+g*32+j] + ((b >> 4) - 8) * xq[t*I+g*32+16+j];
        }
        CHECK(y[t*4+r] == (float)e && yr[t*4+r] == (float)e);
    }
    return 0;
}

/* 4-token tile kernel: integer-exact with unit scales (incl. the int16 lane
 * worst case), and against the per-token reference with real scales */
int q4_gemm4t(void) {
    enum { NB = 9, I = NB*32 };
    uint8_t w[NB*64]; uint16_t d[NB*4]; int8_t xq[4*I]; float xs[4*NB], xst[NB*4], xct[NB*4], y[16], yr[16], y1[16];
    int32_t xm[4*NB];
    for (int i = 0; i < NB*64; i++) w[i] = (uint8_t)(q4_frnd() * 512);
    for (int g = 0; g < NB; g++) memset(w + g*64 + 32, 0xff, 16);   /* row 2: codes 15 */
    for (int i = 0; i < NB*4; i++) d[i] = 0x3c00;
    for (int t = 0; t < 4; t++) for (int g = 0; g < NB; g++) {
        int32_t s = 0;
        for (int j = 0; j < 32; j++) { int v = t == 1 ? 127 : t == 2 ? -127 : (int)(q4_frnd() * 254); xq[t*I+g*32+j] = (int8_t)v; s += v; }
        xs[t*NB+g] = 1.f; xm[t*NB+g] = s;
    }
    moty_hw_q4r4_tile_scales(xs, xm, NB, xst, xct);
    moty_hw_q4r4_gemm4t(w, d, xq, I, xst, xct, NB, y, 4);
    moty_hw_q4r4_gemm4t_ref(w, d, xq, I, xst, xct, NB, yr, 4);
    for (int t = 0; t < 4; t++) for (int r = 0; r < 4; r++) {
        int64_t e = 0;
        for (int g = 0; g < NB; g++) for (int j = 0; j < 16; j++) {
            uint8_t b = w[g*64 + r*16 + j];
            e += ((b & 15) - 8) * xq[t*I+g*32+j] + ((b >> 4) - 8) * xq[t*I+g*32+16+j];
        }
        CHECK(y[t*4+r] == (float)e && yr[t*4+r] == (float)e);
    }
    /* real scales: same result as the per-token GEMM reference up to f32 rounding */
    for (int i = 0; i < NB*4; i++) d[i] = moty_f32_to_f16(q4_frnd() * 0.02f);
    for (int t = 0; t < 4; t++) for (int g = 0; g < NB; g++) xs[t*NB+g] = 0.01f + q4_frnd() * 0.005f;
    moty_hw_q4r4_tile_scales(xs, xm, NB, xst, xct);
    moty_hw_q4r4_gemm4t(w, d, xq, I, xst, xct, NB, y, 4);
    moty_hw_q4r4_gemm_ref(w, d, xq, xs, xm, NB, 4, y1, 4);
    for (int k = 0; k < 16; k++) CHECK(fabsf(y[k] - y1[k]) <= 1e-4f * (fabsf(y1[k]) + 1.f));
    return 0;
}

/* the generic entry point on long rows: nb > 64 takes the chunked 4-token
 * path (activation rows stay I apart), ns = 6 adds a GEMV tail */
int q4_gemm_long_rows(void) {
    enum { NB = 150, I = NB*32, NS = 6 };
    static uint8_t w[NB*64]; static uint16_t d[NB*4]; static int8_t xq[NS*I];
    float xs[NS*NB], y[NS*4], yr[NS*4]; int32_t xm[NS*NB];
    for (int i = 0; i < NB*64; i++) w[i] = (uint8_t)(q4_frnd() * 512);
    for (int i = 0; i < NB*4; i++) d[i] = moty_f32_to_f16(q4_frnd() * 0.02f);
    for (int t = 0; t < NS; t++) for (int g = 0; g < NB; g++) {
        int32_t s = 0;
        for (int j = 0; j < 32; j++) { int v = (int)(q4_frnd() * 254); xq[t*I+g*32+j] = (int8_t)v; s += v; }
        xs[t*NB+g] = 0.01f + q4_frnd() * 0.005f; xm[t*NB+g] = s;
    }
    moty_hw_q4r4_gemm(w, d, xq, xs, xm, NB, NS, y, 4);
    moty_hw_q4r4_gemm_ref(w, d, xq, xs, xm, NB, NS, yr, 4);
    for (int k = 0; k < NS*4; k++) CHECK(fabsf(y[k] - yr[k]) <= 1e-4f * (fabsf(yr[k]) + 1.f));
    return 0;
}

/* full driver vs a dequantized double reference: ragged O, decode (S=1),
 * the 4-token GEMM + tails (S=7) and more than one 32-token tile (S=37) */
static int q4_matmul_case(int O, int I, int S) {
    int nb = I/32, O4 = (O+3)/4;
    float *W = malloc(sizeof(float)*O*I), *x = malloc(sizeof(float)*S*I), *y = malloc(sizeof(float)*S*O);
    uint8_t *q = malloc((size_t)O4*nb*64); uint16_t *d = malloc((size_t)O4*nb*8);
    for (int i = 0; i < O*I; i++) W[i] = q4_frnd();
    for (int i = 0; i < S*I; i++) x[i] = q4_frnd() * (i % 31 == 0 ? 8.f : 1.f);
    for (int b = 0; b < O4; b++) moty_pack_q4r4_block(W + (size_t)b*4*I, O - b*4 < 4 ? O - b*4 : 4, I, q + (size_t)b*nb*64, d + (size_t)b*nb*4);
    moty_matmul_q4r4_s(y, x, q, d, S, I, O);
    int8_t *xq = malloc(I); float *xs = malloc(sizeof(float)*nb); int32_t *xm = malloc(sizeof(int32_t)*nb);
    double worst = 0;
    for (int s = 0; s < S; s++) {
        moty_hw_quant_g32_ref(x + (size_t)s*I, I, xq, xs, xm);
        for (int o = 0; o < O; o++) {
            int b = o/4, r = o%4; double a = 0, mag = 0;
            for (int g = 0; g < nb; g++) {
                const uint8_t *wb = q + ((size_t)b*nb + g)*64 + r*16;
                double dd = moty_hw_f16_to_f32(d[((size_t)b*nb + g)*4 + r]) * xs[g];
                for (int j = 0; j < 16; j++) {
                    a += dd * (((wb[j]&15)-8) * xq[g*32+j] + ((wb[j]>>4)-8) * xq[g*32+16+j]);
                    mag += fabs(dd) * 8 * (abs(xq[g*32+j]) + abs(xq[g*32+16+j]));
                }
            }
            double e = fabs(y[(size_t)s*O+o] - a) / (mag + 1e-30);
            if (e > worst) worst = e;
        }
    }
    free(W); free(x); free(y); free(q); free(d); free(xq); free(xs); free(xm);
    if (worst > 1e-6) fprintf(stderr, "q4_matmul O=%d I=%d S=%d: worst rel %.3g\n", O, I, S, worst);
    return worst > 1e-6;
}
int q4_matmul_driver(void) {
    CHECK(q4_matmul_case(64, 96, 1) == 0);
    CHECK(q4_matmul_case(37, 64, 1) == 0);
    CHECK(q4_matmul_case(38, 128, 7) == 0);
    CHECK(q4_matmul_case(20, 64, 37) == 0);
    return 0;
}

#ifdef Q4R4_TEST_MAIN
int main(void) {
    struct { const char *n; int (*f)(void); } T[] = {
        {"f16_roundtrip", q4_f16_roundtrip}, {"quant_g32_exact", q4_quant_g32_exact},
        {"pack_noclip", q4_pack_noclip}, {"gemm_int_exact", q4_gemm_int_exact},
        {"matmul_driver", q4_matmul_driver}, {"gemm4t", q4_gemm4t}, {"gemm_long_rows", q4_gemm_long_rows} };
    int bad = 0;
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) {
        int r = T[i].f(); bad |= r;
        printf("[%s] %s\n", r ? "FAIL" : " OK ", T[i].n);
    }
    printf("tier: %s\n", HW_IDOT_KERNEL);
    return bad;
}
#endif
