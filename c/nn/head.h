/* nn/head.h — two-stage lm_head for single-token decode (HEAD_TOPK).
 *
 * A tied 65536 x 1024 head is ~19 % of the bytes an int4 decode step reads.
 * Stage 1 scores every vocabulary row with a 1-bit copy of the head (the
 * sign of each Q4R4 weight, one f32 scale = mean |w| per row) against the
 * activation quantized to 4 bits, as bit-plane popcounts (8x fewer bytes
 * than the int4 head). Stage 2 computes exact Q4R4 logits for the K best
 * rows (their 4-row blocks). Every other logit is set to -1e30, so greedy
 * decoding and sampling see a top-K-truncated distribution.
 *
 * Not exact: the int4 argmax is outside the shortlist at a small fraction
 * of positions (docs/performance.md). REF= and PPL= always use the full
 * head. */
#ifndef MOTY_NN_HEAD_H
#define MOTY_NN_HEAD_H
#include "nn/nn_mat.h"

typedef struct MotyHeadSL {
    int V, D, K;
    uint8_t *bits;        /* [V][D/8]: bit j%8 of byte j/8 = (w[v][j] >= 0) */
    float *scale;         /* [V] mean |w| of the row */
    int32_t *pc;          /* [V] popcount of the row's bits */
} MotyHeadSL;

/* build from a WF_Q4R4 head (D % 32 == 0); returns 0 when it does not apply */
int  moty_nn_head_sl_build(MotyHeadSL *h, const Mat *head, int K);
void moty_nn_head_sl_free(MotyHeadSL *h);
/* logit[V] for one activation row x[D]: shortlist when h != NULL, else the
 * full head (mat_apply). Serial calling context. */
void moty_nn_head_apply(float *logit, const float *x, const Mat *head, const MotyHeadSL *h);
/* stage 1 alone (tests): approximate scores of all rows, same ranking the
 * shortlist uses */
void moty_nn_head_sl_scores(const MotyHeadSL *h, const float *x, float *score);

#endif
