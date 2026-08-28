/* ffn.c — M3 libmoty-nn: dense SwiGLU (da nn_ffn.h, 1:1). */
#include "nn/ffn.h"

void moty_nn_dense_ffn(const MotyFfnView *f, const float *x, int S, float *out) {
    int I = f->inter;
    scr_reset(f->scr);
    scr_reserve(f->scr, 2*scr_al((int64_t)S*I*4));
    float *gb = scr_take(f->scr, (int64_t)S*I*4), *ub = scr_take(f->scr, (int64_t)S*I*4);
    mat_apply(gb, x, f->gate, S); mat_apply(ub, x, f->up, S);
    for (int64_t i = 0; i < (int64_t)S*I; i++) { float v=gb[i]; gb[i]=(v/(1.f+expf(-v)))*ub[i]; }
    mat_apply(out, gb, f->down, S);
}
