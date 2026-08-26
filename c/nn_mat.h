/* nn_mat.h — Mat struct, MODEL_COMMON_FIELDS, mat_apply dispatch, kv_store_row.
 *
 * The Mat struct is the universal weight descriptor used by all engines.
 * mat_apply() dispatches to the appropriate matmul kernel from nn_matmul.h.
 * MODEL_COMMON_FIELDS is the convention contract for engine Model structs.
 *
 * Depends on: hw.h (WF_* enum), nn_alloc.h, nn_matmul.h (for mat_apply dispatch).
 */
#ifndef NN_MAT_H
#define NN_MAT_H

#include <stdint.h>

typedef struct Mat { int fmt;             /* WF_F32/WF_I8/WF_I4/WF_I4G/WF_I2 */
                     float *f; int8_t *q; float *qs; int O, I;
                     uint8_t *q4; int gs;
                     const void *sh; const char *sname; } Mat;

static void (*g_mat_stream_fn)(float *y, const float *x, const struct Mat *w, int S) = NULL;

#define MODEL_COMMON_FIELDS \
    Cfg c;                                          \
    shards S;                                       \
    int qbits;                                      \
    float *embed, *final_norm;                      \
    int8_t *embed_q; float *embed_qs;               \
    Mat lm_head; int lm_tied;                       \
    Layer *L;                                       \
    float **K, **V; int kv_len, max_t;              \
    int8_t **K8, **V8; float **Ks, **Vs;            \
    float *att_sc;                                  \
    int n_resident;                                 \
    float *stream_buf;                              \
    int8_t *stream_q; float *stream_qs;             \
    Scratch scr, bscr;   /* P5: kernel scratch (reset/kernel) e stream (reset/step) */ \
    double load_s

static inline void mat_reset_storage(Mat *w) {
    w->fmt = WF_F32;
    w->f = NULL; w->q = NULL; w->qs = NULL;
    w->q4 = NULL; w->gs = 0;
    w->sh = NULL; w->sname = NULL;
}

static inline void kv_store_row(int8_t *dst, float *scale_slot, const float *src, int hd) {
    *scale_slot = qrow_i8(src, dst, hd);
}

static void mat_apply(float *y, const float *x, const Mat *w, int S) {
    if (g_mat_stream_fn && w->sh) { g_mat_stream_fn(y, x, w, S); return; }
    switch (w->fmt) {
        case WF_I4G: matmul_i4_grouped_s(y, x, w->q4, w->qs, S, w->I, w->O, w->gs); return;
        case WF_I4:  matmul_i4_s(y, x, w->q4, w->qs, S, w->I, w->O); return;
        case WF_I8:  matmul_q_s(y, x, w->q, w->qs, S, w->I, w->O); return;
        case WF_I2:  matmul_i2_s(y, x, w->q4, w->qs, S, w->I, w->O); return;
#ifdef GGUF_H
        case WF_Q4K: matmul_q4k_native(y, x, w->q4, S, w->I, w->O); return;
        case WF_Q6K: matmul_q6k_native(y, x, w->q4, S, w->I, w->O); return;
#endif
        default:     matmul(y, x, w->f, S, w->I, w->O); return;
    }
}

#endif /* NN_MAT_H */
