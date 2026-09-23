/* mat.c — M3 libmoty-nn: dispatch mat_apply + KV store row. */
#include "nn/nn_mat.h"
#include <string.h>
#include "nn/nn_alloc.h"
#include "io/gguf.h"
#include "nn/nn_matmul.h"

static MotyMatStreamFn g_stream = NULL;
void moty_nn_set_stream_fn(MotyMatStreamFn fn) { g_stream = fn; }

void moty_kv_store_row(int8_t *dst, float *scale_slot, const float *src, int hd) {
    *scale_slot = qrow_i8(src, dst, hd);
}

int moty_mat_fuse_rows(Mat *dst, Mat *const *src, int n) {
    int I = src[0]->I, O = 0;
    for (int i = 0; i < n; i++) {
        if (src[i]->fmt != WF_Q4R4 || src[i]->I != I || src[i]->O % 4 || !src[i]->q4 || src[i]->sh) return 0;
        O += src[i]->O;
    }
    int nb = I / 32;
    uint8_t *q = balloc((int64_t)O/4*nb*64, "fused q4r4");
    uint16_t *d = balloc((int64_t)O/4*nb*4*sizeof(uint16_t), "fused q4r4 scales");
    int64_t oq = 0, od = 0;
    for (int i = 0; i < n; i++) {
        int64_t bq = (int64_t)src[i]->O/4*nb*64, bd = (int64_t)src[i]->O/4*nb*4;
        memcpy(q + oq, src[i]->q4, bq); memcpy(d + od, src[i]->s16, bd*sizeof(uint16_t));
        free(src[i]->q4); free(src[i]->s16);
        src[i]->q4 = q + oq; src[i]->s16 = d + od;         /* views */
        oq += bq; od += bd;
    }
    mat_reset_storage(dst);
    dst->fmt = WF_Q4R4; dst->O = O; dst->I = I; dst->q4 = q; dst->s16 = d;
    return 1;
}

void moty_mat_apply(float *y, const float *x, const Mat *w, int S) {
    if (g_stream && w->sh) { g_stream(y, x, w, S); return; }
    switch (w->fmt) {
        case WF_Q4R4: moty_matmul_q4r4_s(y, x, w->q4, w->s16, S, w->I, w->O); return;
        case WF_I4G: matmul_i4_grouped_s(y, x, w->q4, w->qs, S, w->I, w->O, w->gs); return;
        case WF_I4:  matmul_i4_s(y, x, w->q4, w->qs, S, w->I, w->O); return;
        case WF_I8:  matmul_q_s(y, x, w->q, w->qs, S, w->I, w->O); return;
        case WF_I2:  matmul_i2_s(y, x, w->q4, w->qs, S, w->I, w->O); return;
        case WF_Q4K: matmul_q4k_native(y, x, w->q4, S, w->I, w->O); return;
        case WF_Q6K: matmul_q6k_native(y, x, w->q4, S, w->I, w->O); return;
        default:     matmul(y, x, w->f, S, w->I, w->O); return;
    }
}
