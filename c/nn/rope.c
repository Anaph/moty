/* rope.c — M3 libmoty-nn */
#include "nn/nn_rope.h"

void moty_l2norm_head(float *x, int d) {
    double s = 0; for (int i = 0; i < d; i++) s += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)s + 1e-6f);
    for (int i = 0; i < d; i++) x[i] *= r;
}

/* RoPE Neox-style half-split sui primi `rot` dimensioni di un head. */
void moty_rope_head(float *x, int pos, float theta, int rot) {
    for (int i = 0; i < rot / 2; i++) {
        float angle = pos / powf(theta, (float)(2 * i) / rot);
        float c = cosf(angle), s = sinf(angle);
        float a = x[i], b = x[i + rot / 2];
        x[i] = a * c - b * s;
        x[i + rot / 2] = a * s + b * c;
    }
}
