/* norm.c — M3 libmoty-nn */
#include "nn/nn_norm.h"
#include "hw/hw.h"

/* bodies in hw/hw_ops.h (NEON on aarch64; the scalar reference elsewhere
 * is the exact former loop, so non-NEON numerics are unchanged) */
void moty_rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    moty_hw_rmsnorm(out, x, w, D, eps);
}
void moty_softmax_row(float *x, int n) { moty_hw_softmax(x, n); }
