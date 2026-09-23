/* nn/ffn.h — dense FFN/SwiGLU, M3 public API. */
#ifndef MOTY_NN_FFN_H
#define MOTY_NN_FFN_H

#include "nn/nn_mat.h"

typedef struct MotyFfnView {
    const Mat *gate, *up, *down;   /* [I,D], [I,D], [D,I] */
    const Mat *gate_up;            /* optional fused [gate|up] [2I,D] (moty_mat_fuse_rows) */
    int inter;                     /* I */
    Scratch *scr;
} MotyFfnView;

void moty_nn_dense_ffn(const MotyFfnView *f, const float *x, int S, float *out);

#endif /* MOTY_NN_FFN_H */
