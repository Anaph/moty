/* hw_ops.h — non-matmul row ops (libmoty-hw, included by hw_impl.h):
 * RMSNorm, SiLU*up, softmax, y+=a*x, y+=x, short-conv step (depthwise
 * causal conv1d, K taps, state [D][K-1]), bit-plane popcounts.
 *
 * Every op has a *_ref scalar reference that is ALSO the portable kernel
 * (bit-identical to the loops it replaced in nn/ and the engines, so non-NEON
 * builds keep their numerics); the aarch64 NEON versions differ only in f32
 * summation order and in the exp approximation (rel. error < 2e-7 on the
 * range used, tests/ops_tests.c). */
#ifndef HW_OPS_H
#define HW_OPS_H

/* ---------------- scalar references ---------------- */
void moty_hw_rmsnorm_ref(float *out, const float *x, const float *w, int n, float eps) {
    double ms = 0; for (int i = 0; i < n; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}
void moty_hw_silu_mul_ref(float *g, const float *u, int64_t n) {
    for (int64_t i = 0; i < n; i++) { float v = g[i]; g[i] = (v / (1.f + expf(-v))) * u[i]; }
}
void moty_hw_softmax_ref(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}
void moty_hw_axpy_ref(float *y, float a, const float *x, int n) {
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}
void moty_hw_add_ref(float *y, const float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] += x[i];
}
/* one token: bx = b*x; y = c * (Σ_t state[ch][t]*w[ch][t] + bx*w[ch][K-1]);
 * state shifts left and takes bx (channels [c0,c1)) */
void moty_hw_shortconv_step_ref(float *y, const float *b, const float *c, const float *x,
                                const float *w, float *state, int K, int c0, int c1) {
    int dc = K - 1;
    for (int ch = c0; ch < c1; ch++) {
        float bx = b[ch]*x[ch], acc = 0;
        for (int tt = 0; tt < dc; tt++) acc += state[ch*dc+tt] * w[ch*K+tt];
        acc += bx * w[ch*K+dc];
        y[ch] = c[ch] * acc;
        if (dc > 0) {
            for (int tt = 0; tt < dc-1; tt++) state[ch*dc+tt] = state[ch*dc+tt+1];
            state[ch*dc+dc-1] = bx;
        }
    }
}

/* attention rows: sc[t] = scale * <q, K[t]> for t < n (rows hd floats apart),
 * and cx = Σ_t sc[t] * V[t]. The references are the per-row loops over
 * dot_f32 / axpy that nn/attn_kernels.c used. */
void moty_hw_attn_scores_ref(float *sc, const float *q, const float *K, int n, int hd, float scale) {
    for (int t = 0; t < n; t++) sc[t] = moty_hw_dot_f32(q, K + (int64_t)t*hd, hd) * scale;
}
void moty_hw_attn_accum_ref(float *cx, const float *sc, const float *V, int n, int hd) {
    for (int d = 0; d < hd; d++) cx[d] = 0;
    for (int t = 0; t < n; t++) moty_hw_axpy(cx, sc[t], V + (int64_t)t*hd, hd);
}

/* bit-plane popcounts of a 1-bit-weight dot product (the lm_head shortlist,
 * nn/head.c): 4 weight-sign rows b + r*bs and 3 activation bit-planes
 * pl + p*nbytes; cnt[r*3 + p] = popcount(b_r & pl_p) over nbytes bytes
 * (nbytes a multiple of 16). */
void moty_hw_popc4x3_ref(const uint8_t *b, int64_t bs, const uint8_t *pl, int nbytes, uint32_t *cnt) {
    for (int r = 0; r < 4; r++) for (int p = 0; p < 3; p++) {
        uint32_t c = 0;
        for (int i = 0; i < nbytes; i++) c += (uint32_t)__builtin_popcount(b[r*bs + i] & pl[p*nbytes + i]);
        cnt[r*3 + p] = c;
    }
}

#if defined(__aarch64__) && defined(__ARM_NEON)
/* exp for f32x4: x = n*ln2 + r, |r| <= ln2/2, 2^n via the exponent field,
 * degree-6 polynomial on r (Taylor-exact coefficients) */
static inline float32x4_t hw_expq(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3f)), vdupq_n_f32(88.3f));
    float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693145751953125f));       /* ln2 hi */
    r = vfmsq_f32(r, n, vdupq_n_f32(1.428606765330187045e-06f));             /* ln2 lo */
    float32x4_t p = vdupq_n_f32(1.f/720);
    p = vfmaq_f32(vdupq_n_f32(1.f/120), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f/24), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f/6), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f), p, r);
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

void moty_hw_rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0 = vld1q_f32(x+i), v1 = vld1q_f32(x+i+4), v2 = vld1q_f32(x+i+8), v3 = vld1q_f32(x+i+12);
        a0 = vfmaq_f32(a0, v0, v0); a1 = vfmaq_f32(a1, v1, v1); a2 = vfmaq_f32(a2, v2, v2); a3 = vfmaq_f32(a3, v3, v3);
    }
    float ms = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; i++) ms += x[i]*x[i];
    float r = 1.f / sqrtf(ms / n + eps);
    i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(out+i, vmulq_f32(vmulq_n_f32(vld1q_f32(x+i), r), vld1q_f32(w+i)));
    for (; i < n; i++) out[i] = x[i] * r * w[i];
}
/* silu(g)*u: four independent vectors per iteration. One exp polynomial is
 * a chain of six dependent FMLAs; the in-order A53 only overlaps chains that
 * sit in the same loop body (80 -> 35 us for 4608 elements). Per element the
 * operations are unchanged, so the result is bit-identical. */
static inline float32x4_t hw_silu_q(float32x4_t v) {
    return vdivq_f32(v, vaddq_f32(vdupq_n_f32(1.f), hw_expq(vnegq_f32(v))));
}
void moty_hw_silu_mul(float *g, const float *u, int64_t n) {
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0 = vld1q_f32(g+i), v1 = vld1q_f32(g+i+4), v2 = vld1q_f32(g+i+8), v3 = vld1q_f32(g+i+12);
        v0 = hw_silu_q(v0); v1 = hw_silu_q(v1); v2 = hw_silu_q(v2); v3 = hw_silu_q(v3);
        vst1q_f32(g+i,    vmulq_f32(v0, vld1q_f32(u+i)));
        vst1q_f32(g+i+4,  vmulq_f32(v1, vld1q_f32(u+i+4)));
        vst1q_f32(g+i+8,  vmulq_f32(v2, vld1q_f32(u+i+8)));
        vst1q_f32(g+i+12, vmulq_f32(v3, vld1q_f32(u+i+12)));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(g+i);
        float32x4_t s = vdivq_f32(v, vaddq_f32(vdupq_n_f32(1.f), hw_expq(vnegq_f32(v))));
        vst1q_f32(g+i, vmulq_f32(s, vld1q_f32(u+i)));
    }
    for (; i < n; i++) { float v = g[i]; g[i] = (v / (1.f + expf(-v))) * u[i]; }
}
void moty_hw_softmax(float *x, int n) {
    int i = 0; float32x4_t mv = vdupq_n_f32(-1e30f);
    for (; i + 4 <= n; i += 4) mv = vmaxq_f32(mv, vld1q_f32(x+i));
    float m = vmaxvq_f32(mv); for (; i < n; i++) if (x[i] > m) m = x[i];
    float32x4_t sv = vdupq_n_f32(0), mm = vdupq_n_f32(m);
    i = 0;
    float32x4_t sv1 = sv, sv2 = sv, sv3 = sv;         /* 4 exp chains per iteration, as in silu */
    for (; i + 16 <= n; i += 16) {
        float32x4_t e0 = hw_expq(vsubq_f32(vld1q_f32(x+i), mm)), e1 = hw_expq(vsubq_f32(vld1q_f32(x+i+4), mm));
        float32x4_t e2 = hw_expq(vsubq_f32(vld1q_f32(x+i+8), mm)), e3 = hw_expq(vsubq_f32(vld1q_f32(x+i+12), mm));
        vst1q_f32(x+i, e0); vst1q_f32(x+i+4, e1); vst1q_f32(x+i+8, e2); vst1q_f32(x+i+12, e3);
        sv = vaddq_f32(sv, e0); sv1 = vaddq_f32(sv1, e1); sv2 = vaddq_f32(sv2, e2); sv3 = vaddq_f32(sv3, e3);
    }
    sv = vaddq_f32(vaddq_f32(sv, sv1), vaddq_f32(sv2, sv3));
    for (; i + 4 <= n; i += 4) { float32x4_t e = hw_expq(vsubq_f32(vld1q_f32(x+i), mm)); vst1q_f32(x+i, e); sv = vaddq_f32(sv, e); }
    float s = vaddvq_f32(sv); for (; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    float inv = 1.f / s; i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(x+i, vmulq_n_f32(vld1q_f32(x+i), inv));
    for (; i < n; i++) x[i] *= inv;
}
void moty_hw_axpy(float *y, float a, const float *x, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y+i,   vfmaq_n_f32(vld1q_f32(y+i),   vld1q_f32(x+i),   a));
        vst1q_f32(y+i+4, vfmaq_n_f32(vld1q_f32(y+i+4), vld1q_f32(x+i+4), a));
    }
    for (; i < n; i++) y[i] += a * x[i];
}
void moty_hw_add(float *y, const float *x, int64_t n) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(y+i, vaddq_f32(vld1q_f32(y+i), vld1q_f32(x+i)));
    for (; i < n; i++) y[i] += x[i];
}
/* hd == 64 (LFM2, Qwen3-0.6B, ...): q (scores) or the 64 accumulators
 * (accum) live in 16 registers for the whole row sweep. The per-row dot_f32
 * call + horizontal reduction cost ~140 cycles per row, the axpy version
 * reloaded and stored the accumulators every row (~240 cycles); these run
 * two rows per iteration. Other hd: the references. */
void moty_hw_attn_scores(float *sc, const float *q, const float *K, int n, int hd, float scale) {
    if (hd != 64) { moty_hw_attn_scores_ref(sc, q, K, n, hd, scale); return; }
    float32x4_t qv[16];
    _Pragma("GCC unroll 16") for (int i = 0; i < 16; i++) qv[i] = vld1q_f32(q + 4*i);
    int t = 0;
    for (; t + 2 <= n; t += 2) {
        const float *k0 = K + (int64_t)t*64, *k1 = k0 + 64;
        float32x4_t a0 = vmulq_f32(qv[0], vld1q_f32(k0)), a1 = vmulq_f32(qv[1], vld1q_f32(k0+4));
        float32x4_t b0 = vmulq_f32(qv[0], vld1q_f32(k1)), b1 = vmulq_f32(qv[1], vld1q_f32(k1+4));
        _Pragma("GCC unroll 8") for (int i = 2; i < 16; i += 2) {
            a0 = vfmaq_f32(a0, qv[i], vld1q_f32(k0+4*i)); a1 = vfmaq_f32(a1, qv[i+1], vld1q_f32(k0+4*i+4));
            b0 = vfmaq_f32(b0, qv[i], vld1q_f32(k1+4*i)); b1 = vfmaq_f32(b1, qv[i+1], vld1q_f32(k1+4*i+4));
        }
        sc[t] = vaddvq_f32(vaddq_f32(a0, a1)) * scale;
        sc[t+1] = vaddvq_f32(vaddq_f32(b0, b1)) * scale;
    }
    if (t < n) moty_hw_attn_scores_ref(sc + t, q, K + (int64_t)t*64, n - t, 64, scale);
}
void moty_hw_attn_accum(float *cx, const float *sc, const float *V, int n, int hd) {
    if (hd != 64) { moty_hw_attn_accum_ref(cx, sc, V, n, hd); return; }
    float32x4_t a[16];
    _Pragma("GCC unroll 16") for (int i = 0; i < 16; i++) a[i] = vdupq_n_f32(0);
    int t = 0;
    for (; t + 2 <= n; t += 2) {
        const float *v0 = V + (int64_t)t*64, *v1 = v0 + 64; float s0 = sc[t], s1 = sc[t+1];
        _Pragma("GCC unroll 16") for (int i = 0; i < 16; i++) a[i] = vfmaq_n_f32(a[i], vld1q_f32(v0 + 4*i), s0);
        _Pragma("GCC unroll 16") for (int i = 0; i < 16; i++) a[i] = vfmaq_n_f32(a[i], vld1q_f32(v1 + 4*i), s1);
    }
    for (; t < n; t++) { const float *v0 = V + (int64_t)t*64; float s0 = sc[t];
        _Pragma("GCC unroll 16") for (int i = 0; i < 16; i++) a[i] = vfmaq_n_f32(a[i], vld1q_f32(v0 + 4*i), s0); }
    _Pragma("GCC unroll 16") for (int i = 0; i < 16; i++) vst1q_f32(cx + 4*i, a[i]);
}

/* 12 named u8 accumulators (an indexed array would live on the stack):
 * per 16 bytes 4 row loads + 3 plane loads, 12 AND + CNT + ADD */
#define POPC_ROW(w, a0, a1, a2) \
    a0 = vaddq_u8(a0, vcntq_u8(vandq_u8(w, p0))); a1 = vaddq_u8(a1, vcntq_u8(vandq_u8(w, p1))); \
    a2 = vaddq_u8(a2, vcntq_u8(vandq_u8(w, p2)));
void moty_hw_popc4x3(const uint8_t *b, int64_t bs, const uint8_t *pl, int nbytes, uint32_t *cnt) {
    for (int k = 0; k < 12; k++) cnt[k] = 0;
    for (int c0 = 0; c0 < nbytes; c0 += 16*16) {      /* u8 lanes: <= 16 chunks x 8 bits */
        int c1 = nbytes - c0 < 16*16 ? nbytes : c0 + 16*16;
        uint8x16_t z = vdupq_n_u8(0);
        uint8x16_t a00 = z, a01 = z, a02 = z, a10 = z, a11 = z, a12 = z;
        uint8x16_t a20 = z, a21 = z, a22 = z, a30 = z, a31 = z, a32 = z;
        for (int i = c0; i < c1; i += 16) {
            uint8x16_t p0 = vld1q_u8(pl + i), p1 = vld1q_u8(pl + nbytes + i), p2 = vld1q_u8(pl + 2*nbytes + i);
            uint8x16_t w0 = vld1q_u8(b + i), w1 = vld1q_u8(b + bs + i), w2 = vld1q_u8(b + 2*bs + i), w3 = vld1q_u8(b + 3*bs + i);
            POPC_ROW(w0, a00, a01, a02) POPC_ROW(w1, a10, a11, a12)
            POPC_ROW(w2, a20, a21, a22) POPC_ROW(w3, a30, a31, a32)
        }
        cnt[0] += vaddlvq_u8(a00); cnt[1] += vaddlvq_u8(a01); cnt[2] += vaddlvq_u8(a02);
        cnt[3] += vaddlvq_u8(a10); cnt[4] += vaddlvq_u8(a11); cnt[5] += vaddlvq_u8(a12);
        cnt[6] += vaddlvq_u8(a20); cnt[7] += vaddlvq_u8(a21); cnt[8] += vaddlvq_u8(a22);
        cnt[9] += vaddlvq_u8(a30); cnt[10] += vaddlvq_u8(a31); cnt[11] += vaddlvq_u8(a32);
    }
}
#undef POPC_ROW
void moty_hw_shortconv_step(float *y, const float *b, const float *c, const float *x,
                            const float *w, float *state, int K, int c0, int c1) {
    int ch = c0;
    if (K == 3) {                                   /* LFM2: state [ch][2], w [ch][3] */
        for (; ch + 4 <= c1; ch += 4) {
            float32x4x2_t st = vld2q_f32(state + ch*2);
            float32x4x3_t wv = vld3q_f32(w + ch*3);
            float32x4_t bx = vmulq_f32(vld1q_f32(b+ch), vld1q_f32(x+ch));
            float32x4_t acc = vmulq_f32(st.val[0], wv.val[0]);
            acc = vfmaq_f32(acc, st.val[1], wv.val[1]);
            acc = vfmaq_f32(acc, bx, wv.val[2]);
            vst1q_f32(y+ch, vmulq_f32(vld1q_f32(c+ch), acc));
            float32x4x2_t ns = { { st.val[1], bx } };
            vst2q_f32(state + ch*2, ns);
        }
    }
    if (ch < c1) moty_hw_shortconv_step_ref(y, b, c, x, w, state, K, ch, c1);
}
#else
void moty_hw_rmsnorm(float *out, const float *x, const float *w, int n, float eps) { moty_hw_rmsnorm_ref(out, x, w, n, eps); }
void moty_hw_silu_mul(float *g, const float *u, int64_t n) { moty_hw_silu_mul_ref(g, u, n); }
void moty_hw_softmax(float *x, int n) { moty_hw_softmax_ref(x, n); }
void moty_hw_axpy(float *y, float a, const float *x, int n) { moty_hw_axpy_ref(y, a, x, n); }
void moty_hw_add(float *y, const float *x, int64_t n) { moty_hw_add_ref(y, x, n); }
void moty_hw_shortconv_step(float *y, const float *b, const float *c, const float *x,
                            const float *w, float *state, int K, int c0, int c1) {
    moty_hw_shortconv_step_ref(y, b, c, x, w, state, K, c0, c1);
}
void moty_hw_popc4x3(const uint8_t *b, int64_t bs, const uint8_t *pl, int nbytes, uint32_t *cnt) {
    moty_hw_popc4x3_ref(b, bs, pl, nbytes, cnt);
}
void moty_hw_attn_scores(float *sc, const float *q, const float *K, int n, int hd, float scale) {
    moty_hw_attn_scores_ref(sc, q, K, n, hd, scale);
}
void moty_hw_attn_accum(float *cx, const float *sc, const float *V, int n, int hd) {
    moty_hw_attn_accum_ref(cx, sc, V, n, hd);
}
#endif

#endif /* HW_OPS_H */
