/* ffn.c — M3 libmoty-nn: dense SwiGLU (da nn_ffn.h, 1:1). */
#include "util/prof.h"
#include "nn/ffn.h"

void moty_nn_dense_ffn(const MotyFfnView *f, const float *x, int S, float *out) {
    int I = f->inter;
    scr_reset(f->scr);
    scr_reserve(f->scr, 2*scr_al((int64_t)S*I*4));
    float *gb = scr_take(f->scr, (int64_t)S*I*4), *ub = scr_take(f->scr, (int64_t)S*I*4);
    OP_T(t0);
    mat_apply(gb, x, f->gate, S); mat_apply(ub, x, f->up, S);
    OP_ACC(OP_FFN_GATE_UP, t0);
    OP_T(t1);
    for (int64_t i = 0; i < (int64_t)S*I; i++) { float v=gb[i]; gb[i]=(v/(1.f+expf(-v)))*ub[i]; }
    OP_ACC(OP_FFN_SILU, t1);
    OP_T(t2);
    mat_apply(out, gb, f->down, S);
    OP_ACC(OP_FFN_DOWN, t2);
}
