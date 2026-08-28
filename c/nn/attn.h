/* nn/attn.h — shared GQA attention with QK-norm + RoPE (M3 public API).
 *
 * The engine fills a MotyAttnView (weights + config + per-layer KV storage
 * from MotyCommon) and calls one of the two variants; no Layer field-name
 * macros, no paste-in. The gated variant (q_proj doubled [query|gate],
 * output *= sigmoid(gate)) is a named function instead of ENGINE_GATED_ATTN.
 *
 *   - NON-gated + all mats WF_I4G(gs=32, D%64==0): q/k/v projections fused
 *     in ONE parallel region via dot_i4g8p (VNNI). 1 ramp instead of 3.
 *   - Gated: sequential projections (the interleaved [q|gate] layout
 *     prevents the fused region).
 *   - o_proj always separate (different input: ctx).
 * KV store: f32 (K/V) or int8 (K8/V8 + per-(head,pos) scales Ks/Vs). */
#ifndef MOTY_NN_ATTN_H
#define MOTY_NN_ATTN_H

#include "nn/nn_mat.h"      /* Mat, MotyCommon, mat_apply */
#include "nn/nn_norm.h"
#include "nn/nn_rope.h"
#include "nn/nn_attn_kernels.h"

typedef struct MotyAttnView {
    const Mat *q, *k, *v, *o;      /* proiezioni + out */
    const float *qn, *kn;          /* pesi QK-norm per head [head_dim] */
    int n_heads, n_kv_heads, head_dim;
    float theta, eps; int rot;     /* RoPE base, norm eps, rotary dim */
    /* per-layer storage (da MotyCommon) */
    float **K, **V;                /* [li] f32 */
    int8_t **K8, **V8; float **Ks, **Vs;
    float *att_sc;                 /* score scratch per-thread: nth*max_t */
    int max_t;
    Scratch *scr;
    int li;
} MotyAttnView;

void moty_nn_attention(const MotyAttnView *a, const float *x, int S, int pos_base, float *out);
void moty_nn_attention_gated(const MotyAttnView *a, const float *x, int S, int pos_base, float *out);

#endif /* MOTY_NN_ATTN_H */
