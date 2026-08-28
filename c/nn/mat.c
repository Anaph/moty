/* mat.c — M3 libmoty-nn: dispatch mat_apply + KV store row. */
#include "nn/nn_mat.h"
#include "io/gguf.h"
#include "nn/nn_matmul.h"

static MotyMatStreamFn g_stream = NULL;
void moty_nn_set_stream_fn(MotyMatStreamFn fn) { g_stream = fn; }

void moty_kv_store_row(int8_t *dst, float *scale_slot, const float *src, int hd) {
    *scale_slot = qrow_i8(src, dst, hd);
}

void moty_mat_apply(float *y, const float *x, const Mat *w, int S) {
    if (g_stream && w->sh) { g_stream(y, x, w, S); return; }
    switch (w->fmt) {
        case WF_I4G: matmul_i4_grouped_s(y, x, w->q4, w->qs, S, w->I, w->O, w->gs); return;
        case WF_I4:  matmul_i4_s(y, x, w->q4, w->qs, S, w->I, w->O); return;
        case WF_I8:  matmul_q_s(y, x, w->q, w->qs, S, w->I, w->O); return;
        case WF_I2:  matmul_i2_s(y, x, w->q4, w->qs, S, w->I, w->O); return;
        case WF_Q4K: matmul_q4k_native(y, x, w->q4, S, w->I, w->O); return;
        case WF_Q6K: matmul_q6k_native(y, x, w->q4, S, w->I, w->O); return;
        default:     matmul(y, x, w->f, S, w->I, w->O); return;
    }
}
