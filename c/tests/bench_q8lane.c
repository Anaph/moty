/* bench_q8lane.c — Q8R4 prefill tile candidates on one A53 core, hot caches.
 *   cur : moty_hw_q8r4_gemm4t (4 rows x 4 tokens, SMULL/SMLAL int8 + SADDLP/SADALP/ADDP reduction per token)
 *   lane: the group transposed/widened once (4 rows interleaved, int16), then by-element
 *         SMLAL int16 -> int32 accumulators that already hold [row0..row3]: no reduction tree;
 *         activations pre-widened to int16 once per token (as the driver would)
 * Both checked against a scalar reference on the same data, then timed.
 *
 *   make CC=aarch64-linux-gnu-gcc ARCH=armv8-a libmoty-hw.a
 *   aarch64-linux-gnu-gcc -O3 -fno-tree-vectorize -march=armv8-a -static -I. tests/bench_q8lane.c libmoty-hw.a -lm -o bench_q8lane
 *   ./bench_q8lane I reps          (result: docs/performance.md 5.15) */
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

/* 4 rows x T tokens, T = 4 */
static void lane4t(const int8_t *w, const uint16_t *d, const int16_t *x16, int64_t ldx,
                   const float *xst, int nb, float *y, int ys) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int g = 0; g < nb; g++) {
        /* rows r0..r3, 32 int8 each -> 16 int16x8 [r0 r1 r2 r3 | same, next column] */
        int8x16_t A0 = vld1q_s8(w), A1 = vld1q_s8(w + 16), B0 = vld1q_s8(w + 32), B1 = vld1q_s8(w + 48);
        int8x16_t C0 = vld1q_s8(w + 64), C1 = vld1q_s8(w + 80), D0 = vld1q_s8(w + 96), D1 = vld1q_s8(w + 112);
        w += 128;
        int16x8_t W[16];
        #define TR(A, B, C, D, o) { \
            int8x16_t ab_l = vzip1q_s8(A, B), ab_h = vzip2q_s8(A, B), cd_l = vzip1q_s8(C, D), cd_h = vzip2q_s8(C, D); \
            int8x16_t q0 = vreinterpretq_s8_s16(vzip1q_s16(vreinterpretq_s16_s8(ab_l), vreinterpretq_s16_s8(cd_l))); \
            int8x16_t q1 = vreinterpretq_s8_s16(vzip2q_s16(vreinterpretq_s16_s8(ab_l), vreinterpretq_s16_s8(cd_l))); \
            int8x16_t q2 = vreinterpretq_s8_s16(vzip1q_s16(vreinterpretq_s16_s8(ab_h), vreinterpretq_s16_s8(cd_h))); \
            int8x16_t q3 = vreinterpretq_s8_s16(vzip2q_s16(vreinterpretq_s16_s8(ab_h), vreinterpretq_s16_s8(cd_h))); \
            W[o+0] = vmovl_s8(vget_low_s8(q0)); W[o+1] = vmovl_high_s8(q0); W[o+2] = vmovl_s8(vget_low_s8(q1)); W[o+3] = vmovl_high_s8(q1); \
            W[o+4] = vmovl_s8(vget_low_s8(q2)); W[o+5] = vmovl_high_s8(q2); W[o+6] = vmovl_s8(vget_low_s8(q3)); W[o+7] = vmovl_high_s8(q3); }
        TR(A0, B0, C0, D0, 0) TR(A1, B1, C1, D1, 8)
        #undef TR
        float32x4_t dv = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(d))); d += 4;
        float32x4_t xs4 = vld1q_f32(xst + 4*g);
        #define TOK(xp, lane, acc) { \
            const int16_t *xg = xp + g*32; int32x4_t s = vdupq_n_s32(0); \
            for (int h = 0; h < 4; h++) { int16x8_t X = vld1q_s16(xg + 8*h); \
                s = vmlal_laneq_s16(s, vget_low_s16(W[4*h+0]), X, 0); s = vmlal_high_laneq_s16(s, W[4*h+0], X, 1); \
                s = vmlal_laneq_s16(s, vget_low_s16(W[4*h+1]), X, 2); s = vmlal_high_laneq_s16(s, W[4*h+1], X, 3); \
                s = vmlal_laneq_s16(s, vget_low_s16(W[4*h+2]), X, 4); s = vmlal_high_laneq_s16(s, W[4*h+2], X, 5); \
                s = vmlal_laneq_s16(s, vget_low_s16(W[4*h+3]), X, 6); s = vmlal_high_laneq_s16(s, W[4*h+3], X, 7); } \
            acc = vfmaq_f32(acc, vcvtq_f32_s32(s), vmulq_laneq_f32(dv, xs4, lane)); }
        TOK(x16, 0, a0) TOK(x16 + ldx, 1, a1) TOK(x16 + 2*ldx, 2, a2) TOK(x16 + 3*ldx, 3, a3)
        #undef TOK
    }
    vst1q_f32(y, a0); vst1q_f32(y + ys, a1); vst1q_f32(y + 2*ys, a2); vst1q_f32(y + 3*ys, a3);
}

int main(int argc, char **argv) {
    int I = argc > 1 ? atoi(argv[1]) : 960, reps = argc > 2 ? atoi(argv[2]) : 20000, nb = I / 32;
    int8_t *w = aligned_alloc(64, (size_t)nb*128), *xq = aligned_alloc(64, (size_t)4*I);
    int16_t *x16 = aligned_alloc(64, (size_t)4*I*2);
    uint16_t *d = aligned_alloc(64, (size_t)nb*8); float *xst = aligned_alloc(64, (size_t)nb*16);
    for (int i = 0; i < nb*128; i++) w[i] = (int8_t)irnd(127);
    for (int i = 0; i < 4*I; i++) { xq[i] = (int8_t)irnd(127); x16[i] = xq[i]; }
    for (int i = 0; i < nb*4; i++) { __fp16 h = (__fp16)(0.001f * (1 + (i % 7))); memcpy(d + i, &h, 2); }
    for (int i = 0; i < nb*4; i++) xst[i] = 0.01f * (1 + (i % 5));
    /* reference: exact int sums per group, then the same f32 order as the kernels */
    float ref[16], yc[16], yl[16];
    for (int t = 0; t < 4; t++) for (int r = 0; r < 4; r++) {
        float acc = 0;
        for (int g = 0; g < nb; g++) {
            int32_t s = 0; for (int k = 0; k < 32; k++) s += w[g*128 + r*32 + k] * xq[t*I + g*32 + k];
            __fp16 h; memcpy(&h, d + g*4 + r, 2);
            acc = fmaf((float)s, (float)h * xst[g*4 + t], acc);
        }
        ref[t*4 + r] = acc;
    }
    moty_hw_q8r4_gemm4t(w, d, xq, I, xst, nb, yc, 4);
    lane4t(w, d, x16, I, xst, nb, yl, 4);
    double ec = 0, el = 0;
    for (int i = 0; i < 16; i++) { ec = fmax(ec, fabs(yc[i] - ref[i])); el = fmax(el, fabs(yl[i] - ref[i])); }
    printf("I=%d check: cur maxerr %.3g, lane maxerr %.3g (|ref| ~ %.3g)\n", I, ec, el, fabs(ref[0]));
    double macs = 16.0 * I * reps, t0, t1; volatile float sink = 0;
    for (int k = 0; k < 3; k++) {
        t0 = now(); for (int i = 0; i < reps; i++) { moty_hw_q8r4_gemm4t(w, d, xq, I, xst, nb, yc, 4); sink += yc[0]; } t1 = now();
        printf("cur : %.2f GMAC/s\n", macs / (t1 - t0) / 1e9);
        t0 = now(); for (int i = 0; i < reps; i++) { lane4t(w, d, x16, I, xst, nb, yl, 4); sink += yl[0]; } t1 = now();
        printf("lane: %.2f GMAC/s\n", macs / (t1 - t0) / 1e9);
    }
    return 0;
}
