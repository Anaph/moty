/* quant.c — unica implementazione della quantizzazione (M2, libmoty-nn).
 * Firme moty_*; nn/nn_quant.h dichiara i prototipi + le macro legacy. */
#include "nn/nn_quant.h"
#include "nn/par.h"
#include <string.h>
#include "hw/hw.h"

/* the packers below: rows [o0, o1) of w [O][I] */
typedef struct { const float *w; void *q; float *scale; int I, bits, gs; } PackJob;

static void quantize_rows_part(void *c_, int64_t o0, int64_t o1, int tid) {
    const PackJob *c = c_; const float *w = c->w; int8_t *q = c->q; float *scale = c->scale; int I = c->I;
    int qmax = (1 << (c->bits - 1)) - 1;
    for (int o = (int)o0; o < (int)o1; o++) {
        const float *wr = w + (int64_t)o * I;
        float amax = 0.f; for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax / qmax; if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        int8_t *qr = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i] / s);
            if (v >  qmax) v =  qmax;
            if (v < -qmax-1) v = -qmax-1;
            qr[i] = (int8_t)v;
        }
    }
}

void moty_quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits) {
    PackJob c = { w, q, scale, I, bits, 0 };
    moty_par_for(O, 0, quantize_rows_part, &c);
}

static void pack_int4_part(void *c_, int64_t o0, int64_t o1, int tid) {
    const PackJob *c = c_; const float *w = c->w; uint8_t *q4 = c->q; float *scale = c->scale; int I = c->I;
    int rb = (I+1)/2;
    for (int o = (int)o0; o < (int)o1; o++) {
        const float *wr = w + (int64_t)o*I; float amax = 0;
        for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax/7.f; if (s < 1e-8f) s = 1e-8f; scale[o] = s;
        uint8_t *qr = q4 + (int64_t)o*rb;
        for (int i = 0; i < I; i += 2) {
            int v0 = (int)lrintf(wr[i]/s); if (v0 > 7) v0 = 7; if (v0 < -8) v0 = -8;
            int v1 = 0;
            if (i+1 < I) { v1 = (int)lrintf(wr[i+1]/s); if (v1 > 7) v1 = 7; if (v1 < -8) v1 = -8; }
            qr[i>>1] = (uint8_t)((v0+8) | ((v1+8)<<4));
        }
    }
}

void moty_pack_int4(const float *w, uint8_t *q4, float *scale, int O, int I) {
    PackJob c = { w, q4, scale, I, 4, 0 };
    moty_par_for(O, 0, pack_int4_part, &c);
}

static void pack_int4_grouped_part(void *c_, int64_t o0, int64_t o1, int tid) {
    const PackJob *c = c_; const float *w = c->w; uint8_t *q4 = c->q; float *scale = c->scale;
    int I = c->I, gs = c->gs, rb = (I+1)/2, ng = (I+gs-1)/gs;
    for (int o = (int)o0; o < (int)o1; o++) {
        const float *wr = w + (int64_t)o*I;
        uint8_t *qr = q4 + (int64_t)o*rb;
        float *scl = scale + (int64_t)o*ng;
        for (int g = 0; g*gs < I; g++) {
            int base = g*gs, glen = gs; if (base+glen > I) glen = I-base;
            float amax = 0;
            for (int i = base; i < base+glen; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
            float s = amax/7.f; if (s < 1e-8f) s = 1e-8f; scl[g] = s;
            for (int i = base; i < base+glen; i += 2) {
                int v0 = (int)lrintf(wr[i]/s); if (v0 > 7) v0 = 7; if (v0 < -8) v0 = -8;
                int v1 = 0;
                if (i+1 < base+glen) { v1 = (int)lrintf(wr[i+1]/s); if (v1 > 7) v1 = 7; if (v1 < -8) v1 = -8; }
                qr[i>>1] = (uint8_t)((v0+8) | ((v1+8)<<4));
            }
        }
    }
}

void moty_pack_int4_grouped(const float *w, uint8_t *q4, float *scale, int O, int I, int gs) {
    if (gs % 16) { fprintf(stderr, "pack_int4_grouped: gs=%d non multiplo di 16\n", gs); exit(1); }
    PackJob c = { w, q4, scale, I, 4, gs };
    moty_par_for(O, 0, pack_int4_grouped_part, &c);
}

static void pack_int2_part(void *c_, int64_t o0, int64_t o1, int tid) {
    const PackJob *c = c_; const float *w = c->w; uint8_t *q2 = c->q; float *scale = c->scale; int I = c->I;
    int qmax = (1 << (c->bits - 1)) - 1, rb = (I+3)/4;
    for (int o = (int)o0; o < (int)o1; o++) {
        const float *wr = w + (int64_t)o*I; float amax = 0;
        for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax/qmax; if (s < 1e-8f) s = 1e-8f; scale[o] = s;
        uint8_t *qr = q2 + (int64_t)o*rb;
        for (int i = 0; i < I; i += 4) {
            uint8_t byte = 0;
            for (int k = 0; k < 4 && i+k < I; k++) {
                int v = (int)lrintf(wr[i+k]/s); if (v > qmax) v = qmax; if (v < -2) v = -2;
                byte |= (uint8_t)((v+2) << (k*2));
            }
            qr[i>>2] = byte;
        }
    }
}

void moty_pack_int2(const float *w, uint8_t *q2, float *scale, int O, int I, int bits) {
    PackJob c = { w, q2, scale, I, bits, 0 };
    moty_par_for(O, 0, pack_int2_part, &c);
}

/* f32 -> IEEE half, round to nearest even (normal, subnormal, inf/nan) */
uint16_t moty_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
    if (e == 0xff) return (uint16_t)(sign | 0x7c00 | (m ? 0x200 : 0));
    int ne = (int)e - 127 + 15;
    if (ne >= 31) return (uint16_t)(sign | 0x7c00);
    if (ne <= 0) {                                      /* subnormal half */
        if (ne < -10) return (uint16_t)sign;
        m |= 0x800000; int sh = 14 - ne;
        uint32_t h = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)ne << 10) | (m >> 13), rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;   /* may carry into the exponent: correct */
    return (uint16_t)(sign | h);
}

/* Scale rule "no-clip Q4_0": the group's largest |w| (mx) maps to level -8
 * (finest step that keeps it), but the step never clips the opposite sign:
 * d = -sign(mx) * max(|mx|/8, opp/7), opp = largest |w| of the other sign.
 * Chosen by HF fake-quant on 2048 tokens (tools/ref/hf_qsim.py, KL vs f32 /
 * top-1): LFM2.5-350M amax/7 1.056/56.9%, Q4_0 0.844/60.4%, no-clip
 * 0.747/62.4%; MiniCPM5-1B 0.312/72.2%, 0.267/74.6%, 0.271/73.8%. A squared-
 * error scale search was worse still (it clips the group outliers). */
void moty_pack_q8r4_block(const float *w, int nr, int I, int8_t *blk, uint16_t *d) {
    int nb = I / 32;
    for (int g = 0; g < nb; g++)
        for (int r = 0; r < 4; r++) {
            int8_t *dst = blk + (size_t)g*128 + r*32;
            if (r >= nr) { memset(dst, 0, 32); d[(size_t)g*4 + r] = 0; continue; }
            const float *wg = w + (size_t)r*I + g*32;
            float mx = 0;
            for (int j = 0; j < 32; j++) if (fabsf(wg[j]) > fabsf(mx)) mx = wg[j];
            float opp = 0;
            for (int j = 0; j < 32; j++) if ((wg[j] > 0) != (mx > 0) && fabsf(wg[j]) > opp) opp = fabsf(wg[j]);
            float dm = fmaxf(fabsf(mx) / 128.f, opp / 127.f);
            uint16_t h = moty_f32_to_f16(mx > 0 ? -dm : dm);
            float dd = moty_hw_f16_to_f32(h), inv = dd != 0.f ? 1.f / dd : 0.f;
            for (int j = 0; j < 32; j++) {
                int v = (int)lrintf(wg[j] * inv); if (v < -128) v = -128; if (v > 127) v = 127;
                dst[j] = (int8_t)v;
            }
            d[(size_t)g*4 + r] = h;
        }
}

void moty_pack_q4r4_block(const float *w, int nr, int I, uint8_t *blk, uint16_t *d) {
    int nb = I / 32;
    for (int g = 0; g < nb; g++)
        for (int r = 0; r < 4; r++) {
            uint8_t *dst = blk + (size_t)g*64 + r*16;
            if (r >= nr) { memset(dst, 0x88, 16); d[(size_t)g*4 + r] = 0; continue; }
            const float *wg = w + (size_t)r*I + g*32;
            float mx = 0;
            for (int j = 0; j < 32; j++) if (fabsf(wg[j]) > fabsf(mx)) mx = wg[j];
            float opp = 0;
            for (int j = 0; j < 32; j++) if ((wg[j] > 0) != (mx > 0) && fabsf(wg[j]) > opp) opp = fabsf(wg[j]);
            float dm = fmaxf(fabsf(mx) / 8.f, opp / 7.f);
            uint16_t h = moty_f32_to_f16(mx > 0 ? -dm : dm);
            float dd = moty_hw_f16_to_f32(h), inv = dd != 0.f ? 1.f / dd : 0.f;
            uint8_t q[32];
            for (int j = 0; j < 32; j++) {
                int v = (int)lrintf(wg[j] * inv); if (v < -8) v = -8; if (v > 7) v = 7;
                q[j] = (uint8_t)(v + 8);
            }
            d[(size_t)g*4 + r] = h;
            for (int j = 0; j < 16; j++) dst[j] = (uint8_t)(q[j] | (q[j+16] << 4));
        }
}
