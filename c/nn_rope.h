/* nn_rope.h — RoPE + activation primitives. Pure scalar C, no SIMD.
 * Include from engine .c files before nn_attn.h / nn_deltanet.h. */
#ifndef NN_ROPE_H
#define NN_ROPE_H

#include <math.h>

static inline float softplusf(float x) { return x > 20.f ? x : logf(1.f + expf(x)); }
static inline float sigmoidf(float x) { return x > 20.f ? 1.f : x < -20.f ? 0.f : 1.f / (1.f + expf(-x)); }

static inline void l2norm_head(float *x, int d) {
    double s = 0; for (int i = 0; i < d; i++) s += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)s + 1e-6f);
    for (int i = 0; i < d; i++) x[i] *= r;
}

/* RoPE Neox-style half-split sui primi `rot` dimensioni di un head.
 * Modifica x[in] in-place. theta = base frequency, pos = position. */
static inline void rope_head(float *x, int pos, float theta, int rot) {
    for (int i = 0; i < rot / 2; i++) {
        float angle = pos / powf(theta, (float)(2 * i) / rot);
        float c = cosf(angle), s = sinf(angle);
        float a = x[i], b = x[i + rot / 2];
        x[i] = a * c - b * s;
        x[i + rot / 2] = a * s + b * c;
    }
}

#endif /* NN_ROPE_H */
