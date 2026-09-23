/* ffn.c — M3 libmoty-nn: dense SwiGLU (da nn_ffn.h, 1:1). */
#include "util/prof.h"
#include "nn/ffn.h"
#include <string.h>

/* silu(g)*u over S token rows: g/u rows have strides gs/us, out is [S][I].
 * Prefill: tokens split across the team (NEON row kernel per token). */
static void silu_rows(float *out, const float *g, const float *u, int S, int I, int64_t gs, int64_t us) {
    #pragma omp parallel for schedule(static) if (S >= 4)
    for (int s = 0; s < S; s++) {
        float *o = out + (int64_t)s*I;
        if (o != g + (int64_t)s*gs) memcpy(o, g + (int64_t)s*gs, (size_t)I*sizeof(float));
        moty_hw_silu_mul(o, u + (int64_t)s*us, I);
    }
}

void moty_nn_dense_ffn(const MotyFfnView *f, const float *x, int S, float *out) {
    int I = f->inter;
    if (f->gate_up) {
        scr_reset(f->scr);
        scr_reserve(f->scr, scr_al((int64_t)S*2*I*4) + scr_al((int64_t)S*I*4));
        float *gu = scr_take(f->scr, (int64_t)S*2*I*4), *hb = scr_take(f->scr, (int64_t)S*I*4);
        OP_T(t0);
        mat_apply(gu, x, f->gate_up, S);
        OP_ACC(OP_FFN_GATE_UP, t0);
        OP_T(t1);
        silu_rows(hb, gu, gu + I, S, I, 2*(int64_t)I, 2*(int64_t)I);
        OP_ACC(OP_FFN_SILU, t1);
        OP_T(t2);
        mat_apply(out, hb, f->down, S);
        OP_ACC(OP_FFN_DOWN, t2);
        return;
    }
    scr_reset(f->scr);
    scr_reserve(f->scr, 2*scr_al((int64_t)S*I*4));
    float *gb = scr_take(f->scr, (int64_t)S*I*4), *ub = scr_take(f->scr, (int64_t)S*I*4);
    OP_T(t0);
    mat_apply(gb, x, f->gate, S); mat_apply(ub, x, f->up, S);
    OP_ACC(OP_FFN_GATE_UP, t0);
    OP_T(t1);
    silu_rows(gb, gb, ub, S, I, I, I);
    OP_ACC(OP_FFN_SILU, t1);
    OP_T(t2);
    mat_apply(out, gb, f->down, S);
    OP_ACC(OP_FFN_DOWN, t2);
}
