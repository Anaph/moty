/* bench_q8row.c — prefill tile, one A53 core, hot caches: the current Q8R4 tile (group-32 weight
 * AND activation scales: reduce + convert + scale every group) against a per-row-weight /
 * per-token-activation tile (int32 accumulated over the whole row, one conversion per row x token).
 * Same 4-row block layout (32 int8 per row per group), exact int sums checked first.
 *
 *   make CC=aarch64-linux-gnu-gcc ARCH=armv8-a libmoty-hw.a
 *   aarch64-linux-gnu-gcc -O3 -fno-tree-vectorize -march=armv8-a -static -I. tests/bench_q8row.c libmoty-hw.a -lm -o bench_q8row
 *   ./bench_q8row I reps
 * Result and the quality side (per-token activation scales): docs/performance.md 5.16. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>
#include "hw/hw.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
static uint64_t rs = 88172645463325252ULL;
static int irnd(int m) { rs ^= rs<<13; rs ^= rs>>7; rs ^= rs<<17; return (int)(rs % (2*m+1)) - m; }

#define RS(wa, wb, x0, x1, acc) { \
    int16x8_t p_ = vmlal_high_s8(vmull_s8(vget_low_s8(wa), vget_low_s8(x0)), wa, x0); \
    int16x8_t q_ = vmlal_high_s8(vmull_s8(vget_low_s8(wb), vget_low_s8(x1)), wb, x1); \
    acc = vpadalq_s16(vpadalq_s16(acc, p_), q_); }

/* 4 rows x 4 tokens: 16 int32x4 accumulators, reduced once at the end.
 * out[t][r] = sum (int) -> f32 * wscale[r] * xscale[t] */
static void row4x4(const int8_t *w, const int8_t *xq, int64_t ldx, int nb, const float *wsc, const float *xsc, float *y, int ys) {
    int32x4_t a[4][4];
    for (int t = 0; t < 4; t++) for (int r = 0; r < 4; r++) a[t][r] = vdupq_n_s32(0);
    for (int g = 0; g < nb; g++) {
        int8x16_t w00 = vld1q_s8(w), w01 = vld1q_s8(w + 16), w10 = vld1q_s8(w + 32), w11 = vld1q_s8(w + 48);
        int8x16_t w20 = vld1q_s8(w + 64), w21 = vld1q_s8(w + 80), w30 = vld1q_s8(w + 96), w31 = vld1q_s8(w + 112);
        w += 128;
        for (int t = 0; t < 4; t++) {
            int8x16_t x0 = vld1q_s8(xq + t*ldx + g*32), x1 = vld1q_s8(xq + t*ldx + g*32 + 16);
            RS(w00, w01, x0, x1, a[t][0]) RS(w10, w11, x0, x1, a[t][1]) RS(w20, w21, x0, x1, a[t][2]) RS(w30, w31, x0, x1, a[t][3])
        }
    }
    float32x4_t ws = vld1q_f32(wsc);
    for (int t = 0; t < 4; t++) {
        int32x4_t s = vpaddq_s32(vpaddq_s32(a[t][0], a[t][1]), vpaddq_s32(a[t][2], a[t][3]));
        vst1q_f32(y + t*ys, vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(s), ws), xsc[t]));
    }
}
/* 4 rows x 2 tokens (8 accumulators: no register pressure) */
static void row4x2(const int8_t *w, const int8_t *xq, int64_t ldx, int nb, const float *wsc, const float *xsc, float *y, int ys) {
    int32x4_t a[2][4];
    for (int t = 0; t < 2; t++) for (int r = 0; r < 4; r++) a[t][r] = vdupq_n_s32(0);
    for (int g = 0; g < nb; g++) {
        int8x16_t w00 = vld1q_s8(w), w01 = vld1q_s8(w + 16), w10 = vld1q_s8(w + 32), w11 = vld1q_s8(w + 48);
        int8x16_t w20 = vld1q_s8(w + 64), w21 = vld1q_s8(w + 80), w30 = vld1q_s8(w + 96), w31 = vld1q_s8(w + 112);
        w += 128;
        for (int t = 0; t < 2; t++) {
            int8x16_t x0 = vld1q_s8(xq + t*ldx + g*32), x1 = vld1q_s8(xq + t*ldx + g*32 + 16);
            RS(w00, w01, x0, x1, a[t][0]) RS(w10, w11, x0, x1, a[t][1]) RS(w20, w21, x0, x1, a[t][2]) RS(w30, w31, x0, x1, a[t][3])
        }
    }
    float32x4_t ws = vld1q_f32(wsc);
    for (int t = 0; t < 2; t++) {
        int32x4_t s = vpaddq_s32(vpaddq_s32(a[t][0], a[t][1]), vpaddq_s32(a[t][2], a[t][3]));
        vst1q_f32(y + t*ys, vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(s), ws), xsc[t]));
    }
}

int main(int argc, char **argv) {
    int I = argc > 1 ? atoi(argv[1]) : 960, reps = argc > 2 ? atoi(argv[2]) : 20000, nb = I / 32;
    int8_t *w = aligned_alloc(64, (size_t)nb*128), *xq = aligned_alloc(64, (size_t)4*I);
    uint16_t *d = aligned_alloc(64, (size_t)nb*8); float *xst = aligned_alloc(64, (size_t)nb*16);
    float wsc[4] = {0.01f, 0.02f, 0.03f, 0.04f}, xsc[4] = {0.5f, 0.25f, 0.125f, 1.f};
    for (int i = 0; i < nb*128; i++) w[i] = (int8_t)irnd(127);
    for (int i = 0; i < 4*I; i++) xq[i] = (int8_t)irnd(127);
    for (int i = 0; i < nb*4; i++) { __fp16 h = (__fp16)0.01f; memcpy(d + i, &h, 2); }
    for (int i = 0; i < nb*4; i++) xst[i] = 0.5f;
    float ref[16], y4[16], y2[16], yc[16];
    for (int t = 0; t < 4; t++) for (int r = 0; r < 4; r++) {
        int64_t s = 0; for (int g = 0; g < nb; g++) for (int k = 0; k < 32; k++) s += w[g*128 + r*32 + k] * xq[t*I + g*32 + k];
        ref[t*4 + r] = (float)s * wsc[r] * xsc[t];
    }
    row4x4(w, xq, I, nb, wsc, xsc, y4, 4);
    row4x2(w, xq, I, nb, wsc, xsc, y2, 4); row4x2(w, xq + 2*I, I, nb, wsc, xsc + 2, y2 + 8, 4);
    double e4 = 0, e2 = 0;
    for (int i = 0; i < 16; i++) { e4 = fmax(e4, fabs(y4[i] - ref[i]) / fabs(ref[i])); e2 = fmax(e2, fabs(y2[i] - ref[i]) / fabs(ref[i])); }
    printf("I=%d check (rel): row4x4 %.2g, row4x2 %.2g\n", I, e4, e2);
    double macs = 16.0 * I * reps, t0, t1; volatile float sink = 0;
    for (int k = 0; k < 3; k++) {
        t0 = now(); for (int i = 0; i < reps; i++) { moty_hw_q8r4_gemm4t(w, d, xq, I, xst, nb, yc, 4); sink += yc[0]; } t1 = now();
        printf("q8r4 group tile : %.2f GMAC/s\n", macs / (t1 - t0) / 1e9);
        t0 = now(); for (int i = 0; i < reps; i++) { row4x4(w, xq, I, nb, wsc, xsc, y4, 4); sink += y4[0]; } t1 = now();
        printf("per-row 4x4     : %.2f GMAC/s\n", macs / (t1 - t0) / 1e9);
        t0 = now(); for (int i = 0; i < reps; i++) { row4x2(w, xq, I, nb, wsc, xsc, y2, 4); row4x2(w, xq + 2*I, I, nb, wsc, xsc + 2, y2 + 8, 4); sink += y2[0]; } t1 = now();
        printf("per-row 4x2 x2  : %.2f GMAC/s\n", macs / (t1 - t0) / 1e9);
    }
    return 0;
}
