/* mat.c — M3 libmoty-nn: dispatch mat_apply + KV store row. */
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif
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

/* the whole pages inside [p, p+n) of a read-only file mapping are no longer
 * used by this process: let the kernel drop them (they are clean file pages;
 * a later access would just re-read them). Not for anonymous memory. */
static void mat_unmap_hint(const void *p, int64_t n) {
#ifndef _WIN32
    long pg = sysconf(_SC_PAGESIZE);
    uintptr_t a = ((uintptr_t)p + pg - 1) & ~(uintptr_t)(pg - 1), b = ((uintptr_t)p + n) & ~(uintptr_t)(pg - 1);
    if (b > a) madvise((void *)a, b - a, MADV_DONTNEED);
#else
    (void)p; (void)n;
#endif
}

int moty_mat_fuse_rows(Mat *dst, Mat *const *src, int n) {
    int I = src[0]->I, O = 0, fmt = src[0]->fmt;
    if (fmt != WF_Q4R4 && fmt != WF_Q8R4) return 0;
    for (int i = 0; i < n; i++) {
        if (src[i]->fmt != fmt || src[i]->I != I || src[i]->O % 4 || !src[i]->q4 || src[i]->sh) return 0;
        O += src[i]->O;
    }
    int nb = I / 32, gb = fmt == WF_Q8R4 ? 128 : 64;   /* bytes per block-group */
    /* sources already back to back (codes and scales, e.g. a container
     * written in fused order and mapped): the fused matrix is a view */
    int adj = 1;
    for (int i = 0; i + 1 < n; i++) {
        int64_t bq = (int64_t)src[i]->O/4*nb*gb, bd = (int64_t)src[i]->O/4*nb*4;
        if (src[i]->q4 + bq != src[i+1]->q4 || src[i]->s16 + bd != src[i+1]->s16) adj = 0;
    }
    mat_reset_storage(dst);
    dst->fmt = fmt; dst->O = O; dst->I = I;
    if (adj) {
        dst->q4 = src[0]->q4; dst->s16 = src[0]->s16; dst->borrowed = 1;   /* the sources keep their blocks */
        return 1;
    }
    uint8_t *q = balloc((int64_t)O/4*nb*gb, "fused r4 codes");
    uint16_t *d = balloc((int64_t)O/4*nb*4*sizeof(uint16_t), "fused r4 scales");
    int64_t oq = 0, od = 0;
    for (int i = 0; i < n; i++) {
        int64_t bq = (int64_t)src[i]->O/4*nb*gb, bd = (int64_t)src[i]->O/4*nb*4;
        memcpy(q + oq, src[i]->q4, bq); memcpy(d + od, src[i]->s16, bd*sizeof(uint16_t));
        if (!src[i]->borrowed) { moty_bfree(src[i]->q4); moty_bfree(src[i]->s16); }
        else if (src[i]->borrowed == 2) {          /* a file mapping (v1 container): drop the copied pages */
            mat_unmap_hint(src[i]->q4, bq); mat_unmap_hint(src[i]->s16, bd*(int64_t)sizeof(uint16_t));
        }
        src[i]->q4 = q + oq; src[i]->s16 = d + od; src[i]->borrowed = 1;   /* views */
        oq += bq; od += bd;
    }
    dst->q4 = q; dst->s16 = d;
    return 1;
}

void moty_mat_apply(float *y, const float *x, const Mat *w, int S) {
    if (g_stream && w->sh) { g_stream(y, x, w, S); return; }
    switch (w->fmt) {
        case WF_Q4R4: moty_matmul_q4r4_s(y, x, w->q4, w->s16, S, w->I, w->O); return;
        case WF_Q8R4: moty_matmul_q8r4_s(y, x, (const int8_t *)w->q4, w->s16, S, w->I, w->O); return;
        case WF_I4G: matmul_i4_grouped_s(y, x, w->q4, w->qs, S, w->I, w->O, w->gs); return;
        case WF_I4:  matmul_i4_s(y, x, w->q4, w->qs, S, w->I, w->O); return;
        case WF_I8:  matmul_q_s(y, x, w->q, w->qs, S, w->I, w->O); return;
        case WF_I2:  matmul_i2_s(y, x, w->q4, w->qs, S, w->I, w->O); return;
        case WF_Q4K: matmul_q4k_native(y, x, w->q4, S, w->I, w->O); return;
        case WF_Q6K: matmul_q6k_native(y, x, w->q4, S, w->I, w->O); return;
        default:     matmul(y, x, w->f, S, w->I, w->O); return;
    }
}
