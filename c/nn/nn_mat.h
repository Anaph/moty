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
#include <stddef.h>
#include "hw/hw.h"            /* WF_* enum */
#include "nn/nn_alloc.h"      /* Scratch (MotyCommon) */

#include <stdint.h>

typedef struct Mat { int fmt;             /* WF_F32/WF_I8/WF_I4/WF_I4G/WF_I2 */
                     float *f; int8_t *q; float *qs; int O, I;
                     uint8_t *q4; int gs;
                     const void *sh; const char *sname; } Mat;

/* M3: implementazioni in nn/mat.c (libmoty-nn) */
typedef void (*MotyMatStreamFn)(float *y, const float *x, const struct Mat *w, int S);
void  moty_nn_set_stream_fn(MotyMatStreamFn fn);   /* MEM_GB: hook dello streamer */
void  moty_mat_apply(float *y, const float *x, const Mat *w, int S);
void  moty_kv_store_row(int8_t *dst, float *scale_slot, const float *src, int hd);
#ifndef MOTY_CORE_NO_LEGACY
#define mat_apply     moty_mat_apply
#define kv_store_row  moty_kv_store_row
#endif

/* M2: il contratto runtime/engine e' un TIPO reale, non solo testo. Gli
 * engine scrivono `typedef struct { MODEL_COMMON_FIELDS; ...extra... }
 * Model;` — l'accesso ai campi comuni passa da m->base.<campo> (il resto
 * del motore — Cfg/Layer/extra — resta del motore). docs/symbol-map.md. */
typedef struct MotyCommon {
    int qbits;                                       /* 0 f32 / 8 i8 / 4 i4 / -1 nativo GGUF */
    float *embed, *final_norm;
    int8_t *embed_q; float *embed_qs;                /* QBITS=8: embedding int8 */
    Mat lm_head; int lm_tied;
    float **K, **V; int kv_len, max_t;               /* KV f32: [li][h*max_t*hd] */
    int8_t **K8, **V8; float **Ks, **Vs;             /* KV_BITS=8: [li] + scale per (h,t) */
    float *att_sc;                                   /* score per-thread: nth*max_t */
    int n_resident;                                  /* MEM_GB: layer residenti */
    float *stream_buf;                               /* streaming layer f32 */
    int8_t *stream_q; float *stream_qs;              /* QBITS=8: layer streamato int8 */
    Scratch scr, bscr;   /* P5: kernel scratch (reset/kernel) e stream (reset/step) */
    double load_s;
} MotyCommon;

#define MODEL_COMMON_FIELDS \
    MotyCommon base;                               \
    Cfg c;                                          \
    shards S;                                       \
    Layer *L;

static inline void mat_reset_storage(Mat *w) {
    w->fmt = WF_F32;
    w->f = NULL; w->q = NULL; w->qs = NULL;
    w->q4 = NULL; w->gs = 0;
    w->sh = NULL; w->sname = NULL;
}




#endif /* NN_MAT_H */
