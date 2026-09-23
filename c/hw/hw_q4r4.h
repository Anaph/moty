/* hw_q4r4.h — int4 "Q4R4" kernels (libmoty-hw, included by hw_impl.h).
 *
 * Format (built by moty_pack_q4r4, nn/quant.c): rows in blocks of 4, the
 * row dimension I in groups of 32. Per (block, group):
 *   w: 64 bytes = 4 rows x 16 bytes, row r byte j = q[r][j] | q[r][j+16] << 4
 *      (q in [0,15], value = (q-8)*d) — low/high nibble split so that the
 *      unpack is AND 0x0F / USHR 4 with no zip;
 *   d: 4 x f16 scales (one per row), in a separate stream [block][group][4].
 * Activations: int8 per group of 32 with an f32 scale xs[g] and the group
 * sum xsum[g] (so Σ(q-8)x = Σq·x - 8·Σx and q stays unsigned in SMULL).
 *
 *   y[t*ys + r] = Σ_g f16(d[g][r]) * xs[t][g] * (Σ_j q[r][g,j]·x[t][g,j] - 8·xsum[t][g])
 *
 * The integer group sums are exact (|q|<=15, |x|<=127: 16 products per
 * int16 lane <= 30480), so NEON and the scalar reference agree bit for
 * bit on them; the f32 accumulation may differ in order (the decode GEMV
 * keeps even and odd groups in two accumulators) and FMA contraction. */
#ifndef HW_Q4R4_H
#define HW_Q4R4_H
#include <string.h>

float moty_hw_f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
    if (e == 0) {
        if (m == 0) u = s;
        else {                                     /* subnormal: normalize */
            e = 1; while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3ff; u = s | ((e + 112) << 23) | (m << 13);
        }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}

/* ---------------- scalar references (always compiled) ---------------- */
void moty_hw_quant_g32_ref(const float *x, int I, int8_t *xq, float *xs, int32_t *xsum) {
    for (int g = 0; g < I / 32; g++) {
        const float *xg = x + g*32;
        float amax = 0; for (int j = 0; j < 32; j++) { float a = fabsf(xg[j]); if (a > amax) amax = a; }
        float s = amax / 127.f; if (s < 1e-30f) s = 1e-30f;
        float inv = 1.f / s; int32_t sm = 0;
        for (int j = 0; j < 32; j++) { int q = (int)lrintf(xg[j] * inv); xq[g*32+j] = (int8_t)q; sm += q; }
        xs[g] = s; xsum[g] = sm;
    }
}

void moty_hw_q4r4_gemm_ref(const uint8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                           const int32_t *xsum, int nb, int ns, float *y, int ys) {
    int I = nb * 32;
    for (int t = 0; t < ns; t++)
        for (int r = 0; r < 4; r++) {
            float acc = 0;
            for (int g = 0; g < nb; g++) {
                const uint8_t *wb = w + (size_t)g*64 + r*16;
                const int8_t *xg = xq + (size_t)t*I + g*32;
                int32_t s = 0;
                for (int j = 0; j < 16; j++) s += (int32_t)(wb[j] & 0xF) * xg[j] + (int32_t)(wb[j] >> 4) * xg[j+16];
                s -= 8 * xsum[(size_t)t*nb + g];
                acc += (float)s * (moty_hw_f16_to_f32(d[(size_t)g*4 + r]) * xs[(size_t)t*nb + g]);
            }
            y[(size_t)t*ys + r] = acc;
        }
}

/* prefill tile of 4 tokens (activation rows ldx int8 apart) with the
 * per-group activation scales laid out per group: xst[g*4 + t] = xs[t][g],
 * xct[g*4 + t] = 8 * xsum[t][g] * xs[t][g] (the offset term in float):
 *   y[t*ys + r] = Σ_g f16(d[g][r]) * (xs[t][g] * Σ_j q·x - xct[g*4+t]) */
void moty_hw_q4r4_gemm4t_ref(const uint8_t *w, const uint16_t *d, const int8_t *xq, int64_t ldx,
                             const float *xst, const float *xct, int nb, float *y, int ys) {
    for (int t = 0; t < 4; t++)
        for (int r = 0; r < 4; r++) {
            float acc = 0;
            for (int g = 0; g < nb; g++) {
                const uint8_t *wb = w + (size_t)g*64 + r*16;
                const int8_t *xg = xq + t*ldx + g*32;
                int32_t s = 0;
                for (int j = 0; j < 16; j++) s += (int32_t)(wb[j] & 0xF) * xg[j] + (int32_t)(wb[j] >> 4) * xg[j+16];
                float dd = moty_hw_f16_to_f32(d[(size_t)g*4 + r]);
                acc += (float)s * (dd * xst[g*4 + t]);
                acc -= dd * xct[g*4 + t];
            }
            y[(size_t)t*ys + r] = acc;
        }
}
/* per-group scales of 4 token rows -> the gemm4t layout */
void moty_hw_q4r4_tile_scales(const float *xs, const int32_t *xsum, int nb, float *xst, float *xct) {
    for (int g = 0; g < nb; g++)
        for (int t = 0; t < 4; t++) {
            xst[g*4 + t] = xs[(size_t)t*nb + g];
            xct[g*4 + t] = 8.f * (float)xsum[(size_t)t*nb + g] * xs[(size_t)t*nb + g];
        }
}

/* Q8R4: the int8 sibling for tensors that int4 damages (mixed precision).
 * Same 4-row blocks, groups of 32 and f16 scale stream as Q4R4; per
 * (block, group) 128 bytes, row r = 32 signed codes, value = q * d.
 * Activations: the Q4R4 group-32 int8 quantization (no offset term).
 *   y[t*ys + r] = Σ_g f16(d[g][r]) * xs[t][g] * Σ_j q[r][g,j] · x[t][g,j] */
void moty_hw_q8r4_gemm_ref(const int8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                           int nb, int ns, float *y, int ys) {
    int I = nb * 32;
    for (int t = 0; t < ns; t++)
        for (int r = 0; r < 4; r++) {
            float acc = 0;
            for (int g = 0; g < nb; g++) {
                const int8_t *wr = w + (size_t)g*128 + r*32;
                const int8_t *xg = xq + (size_t)t*I + g*32;
                int32_t s = 0;
                for (int j = 0; j < 32; j++) s += (int32_t)wr[j] * xg[j];
                acc += (float)s * (moty_hw_f16_to_f32(d[(size_t)g*4 + r]) * xs[(size_t)t*nb + g]);
            }
            y[(size_t)t*ys + r] = acc;
        }
}

#if defined(__aarch64__) && defined(__ARM_NEON)
/* ---------------- NEON (ARMv8.0: SMULL/SMLAL2, no SDOT) ---------------- */
void moty_hw_quant_g32(const float *x, int I, int8_t *xq, float *xs, int32_t *xsum) {
    for (int g = 0; g < I / 32; g++) {
        const float *xg = x + g*32;
        float32x4_t v[8];
        for (int k = 0; k < 8; k++) v[k] = vld1q_f32(xg + 4*k);
        float32x4_t m = vmaxq_f32(vmaxq_f32(vmaxq_f32(vabsq_f32(v[0]), vabsq_f32(v[1])), vmaxq_f32(vabsq_f32(v[2]), vabsq_f32(v[3]))),
                                  vmaxq_f32(vmaxq_f32(vabsq_f32(v[4]), vabsq_f32(v[5])), vmaxq_f32(vabsq_f32(v[6]), vabsq_f32(v[7]))));
        float amax = vmaxvq_f32(m);
        float s = amax / 127.f; if (s < 1e-30f) s = 1e-30f;
        float inv = 1.f / s;
        int32x4_t q[8];
        for (int k = 0; k < 8; k++) q[k] = vcvtnq_s32_f32(vmulq_n_f32(v[k], inv));   /* = lrintf (nearest-even) */
        int16x8_t h0 = vcombine_s16(vmovn_s32(q[0]), vmovn_s32(q[1])), h1 = vcombine_s16(vmovn_s32(q[2]), vmovn_s32(q[3]));
        int16x8_t h2 = vcombine_s16(vmovn_s32(q[4]), vmovn_s32(q[5])), h3 = vcombine_s16(vmovn_s32(q[6]), vmovn_s32(q[7]));
        vst1q_s8(xq + g*32,      vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
        vst1q_s8(xq + g*32 + 16, vcombine_s8(vmovn_s16(h2), vmovn_s16(h3)));
        xsum[g] = vaddlvq_s16(vaddq_s16(vaddq_s16(h0, h1), vaddq_s16(h2, h3)));
        xs[g] = s;
    }
}

/* prefetch distance for the weight stream (bytes ahead): measured on a
 * Cortex-A53 (tests/bench_a53.c), 1024 B with PLDL1KEEP beat none/256/512/2048 */
#define Q4R4_PF 1024

/* products of one 4-row group with one token's 32 activations -> int32x4
 * [row0..row3] (exact) */
#define Q4R4_ROW(l, h, x0, x1, p) \
    p = vmull_s8(vget_low_s8(l), vget_low_s8(x0)); p = vmlal_high_s8(p, l, x0); \
    p = vmlal_s8(p, vget_low_s8(h), vget_low_s8(x1)); p = vmlal_high_s8(p, h, x1);

/* one group of one 4-row block against one token's 32 activations ->
 * int32x4 [row0..row3] (exact; the offset -8*xsum is applied by the caller).
 * 16 products per int16 lane (<= 30480) before widening. */
#define Q4R4_GROUP(w, x0, x1, s) { \
    uint8x16_t b0_ = vld1q_u8(w), b1_ = vld1q_u8((w)+16), b2_ = vld1q_u8((w)+32), b3_ = vld1q_u8((w)+48); \
    int8x16_t l_, h_; int16x8_t p0_, p1_, p2_, p3_; \
    l_ = vreinterpretq_s8_u8(vandq_u8(b0_, m4)); h_ = vreinterpretq_s8_u8(vshrq_n_u8(b0_, 4)); Q4R4_ROW(l_, h_, x0, x1, p0_) \
    l_ = vreinterpretq_s8_u8(vandq_u8(b1_, m4)); h_ = vreinterpretq_s8_u8(vshrq_n_u8(b1_, 4)); Q4R4_ROW(l_, h_, x0, x1, p1_) \
    l_ = vreinterpretq_s8_u8(vandq_u8(b2_, m4)); h_ = vreinterpretq_s8_u8(vshrq_n_u8(b2_, 4)); Q4R4_ROW(l_, h_, x0, x1, p2_) \
    l_ = vreinterpretq_s8_u8(vandq_u8(b3_, m4)); h_ = vreinterpretq_s8_u8(vshrq_n_u8(b3_, 4)); Q4R4_ROW(l_, h_, x0, x1, p3_) \
    s = vpaddlq_s16(vpaddq_s16(vpaddq_s16(p0_, p1_), vpaddq_s16(p2_, p3_))); }

/* decode: two groups per iteration with two f32 accumulators. The in-order
 * A53 cannot overlap the per-group reduction chain (ADDP..SCVTF..FMLA) of one
 * group with the next; interleaving two independent groups in one body lets
 * the compiler do it: 68 -> 56 cycles per group with hot caches, +18 %
 * weight GB/s on one core (tests/bench_a53.c "gemv"). */
static inline void q4r4_gemv(const uint8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                             const int32_t *xsum, int nb, float *y) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    int g = 0;
    for (; g + 2 <= nb; g += 2) {
        __builtin_prefetch(w + Q4R4_PF, 0, 3);
        __builtin_prefetch(w + Q4R4_PF + 64, 0, 3);
        /* the f16 scale stream (8 B/group) is a second sequential stream: one
         * PRFM per 64 B line of it; measured +23%/+9%/+13% weight GB/s at
         * 1/2/4 threads on the A53 (bench_a53 "pf w+d") */
        if ((g & 7) == 0) __builtin_prefetch(d + Q4R4_PF / 2, 0, 3);
        int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16), x2 = vld1q_s8(xq + 32), x3 = vld1q_s8(xq + 48); xq += 64;
        int32x4_t s0, s1;
        Q4R4_GROUP(w, x0, x1, s0)
        Q4R4_GROUP(w + 64, x2, x3, s1)
        w += 128;
        s0 = vsubq_s32(s0, vdupq_n_s32(8 * xsum[g])); s1 = vsubq_s32(s1, vdupq_n_s32(8 * xsum[g+1]));
        float16x8_t dh = vreinterpretq_f16_u16(vld1q_u16(d)); d += 8;
        a0 = vfmaq_f32(a0, vcvtq_f32_s32(s0), vmulq_n_f32(vcvt_f32_f16(vget_low_f16(dh)), xs[g]));
        a1 = vfmaq_f32(a1, vcvtq_f32_s32(s1), vmulq_n_f32(vcvt_high_f32_f16(dh), xs[g+1]));
    }
    if (g < nb) {                                      /* odd group count */
        int8x16_t x0 = vld1q_s8(xq), x1 = vld1q_s8(xq + 16);
        int32x4_t s0;
        Q4R4_GROUP(w, x0, x1, s0)
        s0 = vsubq_s32(s0, vdupq_n_s32(8 * xsum[g]));
        a0 = vfmaq_f32(a0, vcvtq_f32_s32(s0), vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(d))), xs[g]));
    }
    vst1q_f32(y, vaddq_f32(a0, a1));
}

/* prefill: 4 rows x 4 tokens; every group is unpacked once and reused for
 * the 4 tokens. The 4 tokens' scales come as one vector per group (xst,
 * xct: FMUL/FMLS by lane) instead of scalar loads, broadcasts and an
 * integer offset subtraction per token: 156 -> ~120 instructions per group,
 * 2.67 -> 3.05 MAC/cycle at I = 1024 on one A53 core. */
void moty_hw_q4r4_gemm4t(const uint8_t *w, const uint16_t *d, const int8_t *xq, int64_t ldx,
                         const float *xst, const float *xct, int nb, float *y, int ys) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8_t *x0p = xq, *x1p = xq + ldx, *x2p = xq + 2*ldx, *x3p = xq + 3*ldx;
    for (int g = 0; g < nb; g++) {
        uint8x16_t b0 = vld1q_u8(w), b1 = vld1q_u8(w+16), b2 = vld1q_u8(w+32), b3 = vld1q_u8(w+48); w += 64;
        int8x16_t l0 = vreinterpretq_s8_u8(vandq_u8(b0, m4)), h0 = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4));
        int8x16_t l1 = vreinterpretq_s8_u8(vandq_u8(b1, m4)), h1 = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
        int8x16_t l2 = vreinterpretq_s8_u8(vandq_u8(b2, m4)), h2 = vreinterpretq_s8_u8(vshrq_n_u8(b2, 4));
        int8x16_t l3 = vreinterpretq_s8_u8(vandq_u8(b3, m4)), h3 = vreinterpretq_s8_u8(vshrq_n_u8(b3, 4));
        float32x4_t dv = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(d))); d += 4;
        float32x4_t xs4 = vld1q_f32(xst + 4*g), c4 = vld1q_f32(xct + 4*g);
        #define Q4R4_TOK(xp, lane, acc) { \
            int8x16_t x0 = vld1q_s8(xp + g*32), x1 = vld1q_s8(xp + g*32 + 16); int16x8_t p0, p1, p2, p3; \
            Q4R4_ROW(l0, h0, x0, x1, p0) Q4R4_ROW(l1, h1, x0, x1, p1) \
            Q4R4_ROW(l2, h2, x0, x1, p2) Q4R4_ROW(l3, h3, x0, x1, p3) \
            float32x4_t s = vcvtq_f32_s32(vpaddlq_s16(vpaddq_s16(vpaddq_s16(p0, p1), vpaddq_s16(p2, p3)))); \
            acc = vfmaq_f32(acc, s, vmulq_laneq_f32(dv, xs4, lane)); \
            acc = vfmsq_laneq_f32(acc, dv, c4, lane); }
        Q4R4_TOK(x0p, 0, a0) Q4R4_TOK(x1p, 1, a1) Q4R4_TOK(x2p, 2, a2) Q4R4_TOK(x3p, 3, a3)
        #undef Q4R4_TOK
    }
    vst1q_f32(y, a0); vst1q_f32(y + ys, a1); vst1q_f32(y + 2*ys, a2); vst1q_f32(y + 3*ys, a3);
}

void moty_hw_q4r4_gemm(const uint8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                       const int32_t *xsum, int nb, int ns, float *y, int ys) {
    int I = nb * 32, t = 0;
    for (; t + 4 <= ns; t += 4) {                 /* the driver builds the tile scales once per tile */
        float st[4*64], ct[4*64];
        for (int g0 = 0; g0 < nb; g0 += 64) {     /* 64-group chunks: bounded stack */
            int n = nb - g0 < 64 ? nb - g0 : 64; float yp[16];
            for (int g = 0; g < n; g++) for (int k = 0; k < 4; k++) {
                st[g*4 + k] = xs[(size_t)(t+k)*nb + g0 + g];
                ct[g*4 + k] = 8.f * (float)xsum[(size_t)(t+k)*nb + g0 + g] * xs[(size_t)(t+k)*nb + g0 + g];
            }
            moty_hw_q4r4_gemm4t(w + (size_t)g0*64, d + (size_t)g0*4, xq + (size_t)t*I + g0*32, I, st, ct, n, yp, 4);
            for (int k = 0; k < 4; k++) for (int r = 0; r < 4; r++)
                y[(size_t)(t+k)*ys + r] = g0 ? y[(size_t)(t+k)*ys + r] + yp[k*4 + r] : yp[k*4 + r];
        }
    }
    for (; t < ns; t++) {
        float tmp[4];
        q4r4_gemv(w, d, xq + (size_t)t*I, xs + (size_t)t*nb, xsum + (size_t)t*nb, nb, tmp);
        memcpy(y + (size_t)t*ys, tmp, sizeof tmp);
    }
}
/* Q8R4 one group, 4 rows -> int32x4 [row0..row3]: two products per int16
 * lane (|q| <= 128, |x| <= 127: <= 32512), then widened (SADDLP/SADALP) */
#define Q8R4_ROWSUM(w, x0, x1, s) { \
    int8x16_t w0_ = vld1q_s8(w), w1_ = vld1q_s8((w) + 16); \
    int16x8_t p_ = vmlal_high_s8(vmull_s8(vget_low_s8(w0_), vget_low_s8(x0)), w0_, x0); \
    int16x8_t q_ = vmlal_high_s8(vmull_s8(vget_low_s8(w1_), vget_low_s8(x1)), w1_, x1); \
    s = vpadalq_s16(vpaddlq_s16(p_), q_); }
static inline int32x4_t q8r4_group(const int8_t *w, int8x16_t x0, int8x16_t x1) {
    int32x4_t s0, s1, s2, s3;
    Q8R4_ROWSUM(w, x0, x1, s0) Q8R4_ROWSUM(w + 32, x0, x1, s1)
    Q8R4_ROWSUM(w + 64, x0, x1, s2) Q8R4_ROWSUM(w + 96, x0, x1, s3)
    return vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
}
/* one token; two groups per iteration (two accumulators, as the Q4R4 GEMV) */
static inline void q8r4_gemv(const int8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                             int nb, float *y) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
    int g = 0;
    for (; g + 2 <= nb; g += 2) {
        __builtin_prefetch(w + 2*Q4R4_PF, 0, 3); __builtin_prefetch(w + 2*Q4R4_PF + 64, 0, 3);
        __builtin_prefetch(w + 2*Q4R4_PF + 128, 0, 3); __builtin_prefetch(w + 2*Q4R4_PF + 192, 0, 3);
        if ((g & 7) == 0) __builtin_prefetch(d + Q4R4_PF / 2, 0, 3);
        int32x4_t s0 = q8r4_group(w, vld1q_s8(xq), vld1q_s8(xq + 16));
        int32x4_t s1 = q8r4_group(w + 128, vld1q_s8(xq + 32), vld1q_s8(xq + 48));
        w += 256; xq += 64;
        float16x8_t dh = vreinterpretq_f16_u16(vld1q_u16(d)); d += 8;
        a0 = vfmaq_f32(a0, vcvtq_f32_s32(s0), vmulq_n_f32(vcvt_f32_f16(vget_low_f16(dh)), xs[g]));
        a1 = vfmaq_f32(a1, vcvtq_f32_s32(s1), vmulq_n_f32(vcvt_high_f32_f16(dh), xs[g+1]));
    }
    if (g < nb) {
        int32x4_t s0 = q8r4_group(w, vld1q_s8(xq), vld1q_s8(xq + 16));
        a0 = vfmaq_f32(a0, vcvtq_f32_s32(s0), vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(d))), xs[g]));
    }
    vst1q_f32(y, vaddq_f32(a0, a1));
}
/* prefill: token by token over the block (4 rows x I int8 stay in L1) */
void moty_hw_q8r4_gemm(const int8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                       int nb, int ns, float *y, int ys) {
    int I = nb * 32;
    for (int t = 0; t < ns; t++) {
        float tmp[4];
        q8r4_gemv(w, d, xq + (size_t)t*I, xs + (size_t)t*nb, nb, tmp);
        memcpy(y + (size_t)t*ys, tmp, sizeof tmp);
    }
}
#undef Q8R4_ROWSUM
#undef Q4R4_ROW
#undef Q4R4_GROUP
#else
/* ---------------- portable: the reference is the kernel ---------------- */
void moty_hw_quant_g32(const float *x, int I, int8_t *xq, float *xs, int32_t *xsum) {
    moty_hw_quant_g32_ref(x, I, xq, xs, xsum);
}
void moty_hw_q4r4_gemm(const uint8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                       const int32_t *xsum, int nb, int ns, float *y, int ys) {
    moty_hw_q4r4_gemm_ref(w, d, xq, xs, xsum, nb, ns, y, ys);
}
void moty_hw_q4r4_gemm4t(const uint8_t *w, const uint16_t *d, const int8_t *xq, int64_t ldx,
                         const float *xst, const float *xct, int nb, float *y, int ys) {
    moty_hw_q4r4_gemm4t_ref(w, d, xq, ldx, xst, xct, nb, y, ys);
}
void moty_hw_q8r4_gemm(const int8_t *w, const uint16_t *d, const int8_t *xq, const float *xs,
                       int nb, int ns, float *y, int ys) {
    moty_hw_q8r4_gemm_ref(w, d, xq, xs, nb, ns, y, ys);
}
#endif

#endif /* HW_Q4R4_H */
