/* quant.c — unica implementazione della quantizzazione (M2, libmoty-nn).
 * Firme moty_*; nn/nn_quant.h dichiara i prototipi + le macro legacy. */
#include "nn/nn_quant.h"

void moty_quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits) {
    int qmax = (1 << (bits - 1)) - 1;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
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

void moty_pack_int4(const float *w, uint8_t *q4, float *scale, int O, int I) {
    int rb = (I+1)/2;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
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

void moty_pack_int4_grouped(const float *w, uint8_t *q4, float *scale, int O, int I, int gs) {
    if (gs % 16) { fprintf(stderr, "pack_int4_grouped: gs=%d non multiplo di 16\n", gs); exit(1); }
    int rb = (I+1)/2, ng = (I+gs-1)/gs;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
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

void moty_pack_int2(const float *w, uint8_t *q2, float *scale, int O, int I, int bits) {
    int qmax = (1 << (bits - 1)) - 1, rb = (I+3)/4;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
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
