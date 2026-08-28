/* attn_kernels.c — M3 libmoty-nn */
#include "hw/hw.h"
#include "nn/nn_attn_kernels.h"

void moty_att_scores_f32(float *sc, const float *qv, const float *K,
                                  int64_t kvbase, int t0, int qpos, int hd, float scale) {
    for (int t = t0; t <= qpos; t++)
        sc[t-t0] = dot_f32(qv, K + (kvbase + t)*hd, hd) * scale;
}

void moty_att_scores_i8(float *sc, const float *qv, const int8_t *K8,
                                 const float *Ks, int64_t kvbase, int t0, int qpos,
                                 int hd, float scale) {
    for (int t = t0; t <= qpos; t++) {
        int64_t slot = kvbase + t;
        sc[t-t0] = Ks[slot] * dot_f32i8(qv, K8 + slot*hd, hd) * scale;
    }
}

void moty_att_accum_f32(float *cx, const float *sc, const float *V,
                                 int64_t kvbase, int t0, int qpos, int hd) {
    for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
    for (int t = t0; t <= qpos; t++) {
        const float *vr = V + (kvbase + t)*hd;
        float a = sc[t-t0];
        for (int dd = 0; dd < hd; dd++) cx[dd] += a * vr[dd];
    }
}

void moty_att_accum_i8(float *cx, const float *sc, const int8_t *V8,
                                const float *Vs, int64_t kvbase, int t0, int qpos, int hd) {
    for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
    for (int t = t0; t <= qpos; t++) {
        int64_t slot = kvbase + t;
        const int8_t *vr = V8 + slot*hd;
        float a = sc[t-t0] * Vs[slot];
        for (int dd = 0; dd < hd; dd++) cx[dd] += a * (float)vr[dd];
    }
}
