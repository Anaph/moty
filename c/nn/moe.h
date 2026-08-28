/* nn/moe.h — shared MoE decode/batch, M3 public API.
 *
 * The engine fills a MotyMoeView and calls the variant matching its gate:
 *   moty_nn_moe_sigmoid_d1/_batch  — sigmoid gating + expert_bias, pesi
 *                                    normalizzati (LFM2-style)
 *   moty_nn_moe_topk_d1/_batch     — softmax + greedy top-K (Qwen-style)
 * Shared expert (gated) e' un flag nella view, non una macro.
 *
 * S=1 fast path: 2 fork/join TOTAL (vs 3×K): region 1 = tutte le righe
 * gate+up di tutti gli expert; region 2 = tutte le righe down. x quantizzato
 * UNA volta (contratto dot_i8i8: [-127,127]). ebits<=4: gli expert stanno in
 * int4 packed (nibble) con scale per riga — righe servite da dot_i4i8p/dot_i8i8. */
#ifndef MOTY_NN_MOE_H
#define MOTY_NN_MOE_H

#include "nn/nn_mat.h"
#include "runtime/moe.h"     /* ExpertCache, ExpertSlot, LoadExpertFn, moe_topk */

typedef struct MotyMoeView {
    const Mat *router;             /* [E,D] */
    const float *expert_bias;      /* [E] (sigmoid gate; NULL per topk) */
    ExpertCache *ec;               /* cache per-layer */
    LoadExpertFn load_expert;      /* come leggere i byte di un expert */
    void *ectx;                    /* ctx opaco per load_expert (il Model*) */
    int E, K, I, D;                /* n_experts, topk, moe_inter, hidden */
    int li;                        /* layer index (per expert_get) */
    int ebits;                     /* 8 int8 per-riga; <=4 int4 packed */
    /* shared expert gated (opzionale) */
    int has_shared;
    const Mat *sh_gate, *sh_up, *sh_down, *sh_router_gate;
    int sh_inter;
    Scratch *scr;
} MotyMoeView;

void moty_nn_moe_sigmoid_d1(const MotyMoeView *v, const float *x, float *out);
void moty_nn_moe_topk_d1(const MotyMoeView *v, const float *x, float *out);
void moty_nn_moe_sigmoid_batch(const MotyMoeView *v, const float *x, int S, float *out);
void moty_nn_moe_topk_batch(const MotyMoeView *v, const float *x, int S, float *out);

#endif /* MOTY_NN_MOE_H */
