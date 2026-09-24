/* head.c — two-stage lm_head (see nn/head.h). */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nn/head.h"
#include "nn/nn_alloc.h"

typedef struct { MotyHeadSL *h; const Mat *head; } SlBuildJob;
static void sl_build_part(void *c_, int64_t v0, int64_t v1, int tid);

int moty_nn_head_sl_build(MotyHeadSL *h, const Mat *head, int K) {
    memset(h, 0, sizeof *h);
    if (head->fmt != WF_Q4R4 || head->I % 32 || K <= 0) return 0;
    int V = head->O, D = head->I, rb = D / 8;
    if (K > V) K = V;
    h->V = V; h->D = D; h->K = K;
    h->bits = malloc((size_t)V * rb); h->scale = malloc(sizeof(float) * V); h->pc = malloc(sizeof(int32_t) * V);
    if (!h->bits || !h->scale || !h->pc) { moty_nn_head_sl_free(h); return 0; }
    /* from the packed codes: value = (q-8)*d, q = low/high nibble */
    SlBuildJob c = { h, head };
    moty_par_for(V, 0, sl_build_part, &c);
    return 1;
}

static void sl_build_part(void *c_, int64_t v0, int64_t v1, int tid) {
    const SlBuildJob *c = c_; MotyHeadSL *h = c->h; const Mat *head = c->head;
    int D = h->D, nb = D / 32, rb = D / 8;
    for (int v = (int)v0; v < (int)v1; v++) {
        int b = v / 4, r = v % 4;
        uint8_t *row = h->bits + (size_t)v * rb;
        memset(row, 0, rb);
        double sa = 0; int32_t pc = 0;
        for (int g = 0; g < nb; g++) {
            const uint8_t *wb = head->q4 + ((size_t)b*nb + g)*64 + r*16;
            float d = moty_hw_f16_to_f32(head->s16[((size_t)b*nb + g)*4 + r]);
            for (int j = 0; j < 32; j++) {
                int q = j < 16 ? (wb[j] & 15) : (wb[j-16] >> 4);
                float w = (float)(q - 8) * d;
                int jj = g*32 + j;
                if (w >= 0) { row[jj >> 3] |= (uint8_t)(1u << (jj & 7)); pc++; }
                sa += fabsf(w);
            }
        }
        h->scale[v] = (float)(sa / D); h->pc[v] = pc;
    }
}

void moty_nn_head_sl_free(MotyHeadSL *h) {
    free(h->bits); free(h->scale); free(h->pc);
    memset(h, 0, sizeof *h);
}

/* x -> 3-bit unsigned codes u = clamp(round(x/s), -3, 3) + 4 in [1,7] as
 * 3 bit-planes; returns Σu. Σ_j sign_j (u_j - 4) = 2 Σ_{bit} u - 8 pc - Σu + 4 D,
 * and Σ_{bit} u = Σ_p 2^p popcount(bits & plane_p). 3 bits rank as well as
 * 4 or 8 for a 512-row shortlist (top-1 recall 99.9 %, HF hidden states of
 * LFM2.5-350M) at 3/4 of the popcount work. */
static int32_t head_planes(const float *x, int D, uint8_t *pl) {
    float amax = 0; for (int j = 0; j < D; j++) { float a = fabsf(x[j]); if (a > amax) amax = a; }
    float inv = amax > 0 ? 3.f / amax : 0.f;
    int rb = D / 8; int32_t su = 0;
    memset(pl, 0, (size_t)3 * rb);
    for (int j = 0; j < D; j++) {
        int u = (int)lrintf(x[j] * inv); u = u < -3 ? -3 : u > 3 ? 3 : u; u += 4; su += u;
        for (int p = 0; p < 3; p++) if (u >> p & 1) pl[p*rb + (j >> 3)] |= (uint8_t)(1u << (j & 7));
    }
    return su;
}

typedef struct { const MotyHeadSL *h; const uint8_t *pl; int32_t su; float *score; } ScoreJob;
static void scores_part(void *c_, int64_t b0, int64_t b1, int tid) {
    const ScoreJob *c = c_; const MotyHeadSL *h = c->h; const uint8_t *pl = c->pl; int32_t su = c->su; float *score = c->score;
    int V = h->V, D = h->D, rb = D / 8, full = V / 4;
    for (int b = (int)b0; b < (int)b1; b++) {
        uint32_t cnt[12];
        if (b < full) moty_hw_popc4x3(h->bits + (size_t)b*4*rb, rb, pl, rb, cnt);
        else {                                         /* ragged tail: rows one by one */
            for (int r = 0; r < 4; r++) {
                int v = b*4 + r; if (v >= V) break;
                for (int p = 0; p < 3; p++) {
                    uint32_t c = 0;
                    for (int i = 0; i < rb; i++) c += (uint32_t)__builtin_popcount(h->bits[(size_t)v*rb + i] & pl[p*rb + i]);
                    cnt[r*3 + p] = c;
                }
            }
        }
        for (int r = 0; r < 4 && b*4 + r < V; r++) {
            int v = b*4 + r;
            int32_t sb = (int32_t)(cnt[r*3] + 2*cnt[r*3+1] + 4*cnt[r*3+2]);
            score[v] = h->scale[v] * (float)(2*sb - 8*h->pc[v] - su + 4*D);
        }
    }
}

static void head_scores(const MotyHeadSL *h, const uint8_t *pl, int32_t su, float *score) {
    ScoreJob c = { h, pl, su, score };
    moty_par_for((h->V + 3) / 4, 64, scores_part, &c);
}

void moty_nn_head_sl_scores(const MotyHeadSL *h, const float *x, float *score) {
    uint8_t *pl = malloc((size_t)h->D / 8 * 3);
    int32_t su = head_planes(x, h->D, pl);
    head_scores(h, pl, su, score);
    free(pl);
}

/* K-th largest of a[0..n) (destroys a): iterative quickselect */
static float kth_largest(float *a, int n, int k) {
    int lo = 0, hi = n - 1, want = k - 1;
    while (lo < hi) {
        float piv = a[lo + (hi - lo) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (a[i] > piv) i++;
            while (a[j] < piv) j--;
            if (i <= j) { float t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        }
        if (want <= j) hi = j; else if (want >= i) lo = i; else break;
    }
    return a[want];
}

/* stage 2 of the head: exact logits of candidate blocks [i0, i1) */
typedef struct { float *logit; const Mat *head; const int32_t *blk; const int8_t *xq; const float *xs; const int32_t *xm; int nb, V; } ExactJob;
static void exact_part(void *c_, int64_t i0, int64_t i1, int tid) {
    const ExactJob *c = c_; int nb = c->nb;
    for (int64_t i = i0; i < i1; i++) {
        int b = c->blk[i]; float y4[4];
        moty_hw_q4r4_gemm(c->head->q4 + (size_t)b*nb*64, c->head->s16 + (size_t)b*nb*4, c->xq, c->xs, c->xm, nb, 1, y4, 4);
        for (int r = 0; r < 4 && b*4 + r < c->V; r++) c->logit[b*4 + r] = y4[r];
    }
}

void moty_nn_head_apply(float *logit, const float *x, const Mat *head, const MotyHeadSL *h) {
    if (!h || !h->bits) { mat_apply(logit, x, head, 1); return; }
    static uint8_t *pl = NULL, *mark = NULL; static float *score = NULL, *tmp = NULL;
    static int8_t *xq = NULL; static float *xs = NULL; static int32_t *xm = NULL, *blk = NULL;
    static int64_t c0, c1, c2, c3, c4, c5, c6, c7;
    int V = h->V, D = h->D, nb = D / 32, NB = (V + 3) / 4;
    grow((void **)&pl, &c0, D / 8 * 3, 1, "head planes");
    grow((void **)&score, &c1, V, sizeof(float), "head scores");
    grow((void **)&tmp, &c2, V, sizeof(float), "head select");
    grow((void **)&mark, &c3, NB, 1, "head marks");
    grow((void **)&blk, &c4, NB, sizeof(int32_t), "head blocks");
    grow((void **)&xq, &c5, D, 1, "head xq");
    grow((void **)&xs, &c6, nb, sizeof(float), "head xs");
    grow((void **)&xm, &c7, nb, sizeof(int32_t), "head xsum");
    /* stage 1: approximate scores, K-th largest as the threshold */
    int32_t su = head_planes(x, D, pl);
    head_scores(h, pl, su, score);
    /* threshold: a strided sample gives a provisional bound below the K-th
     * score (rank 2K/stride in the sample); if at least K scores clear it,
     * the exact K-th largest is among them, else fall back to all V */
    enum { STRIDE = 16 };
    int ns = 0;
    for (int v = 0; v < V; v += STRIDE) tmp[ns++] = score[v];
    int ks = 2 * h->K / STRIDE + 1; if (ks > ns) ks = ns;
    float thr0 = kth_largest(tmp, ns, ks);
    int nc = 0;
    for (int v = 0; v < V; v++) if (score[v] >= thr0) tmp[nc++] = score[v];
    float thr;
    if (nc >= h->K) thr = kth_largest(tmp, nc, h->K);
    else { memcpy(tmp, score, sizeof(float) * V); thr = kth_largest(tmp, V, h->K); }
    memset(mark, 0, NB);
    int nblk = 0;
    for (int v = 0; v < V; v++)
        if (score[v] >= thr && !mark[v >> 2]) { mark[v >> 2] = 1; blk[nblk++] = v >> 2; }
    /* stage 2: exact Q4R4 logits of the candidate blocks, -1e30 elsewhere */
    moty_hw_quant_g32(x, D, xq, xs, xm);
    for (int v = 0; v < V; v++) logit[v] = -1e30f;
    ExactJob c = { logit, head, blk, xq, xs, xm, nb, V };
    moty_par_for(nblk, 4, exact_part, &c);
}
