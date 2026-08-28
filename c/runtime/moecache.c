/* moecache.c — M3: cache expert LFRU/pin + routing top-K (libmoty-nn). */
#include "runtime/moe.h"

void moty_expert_cache_init(ExpertCache *lc, int cap, int n_experts) {
    if (cap < 1) cap = 1;
    lc->cap = cap; lc->n = 0; lc->n_experts = n_experts; lc->clock = 0;
    lc->hits = lc->misses = 0;
    lc->slots = (ExpertSlot *)calloc(cap, sizeof(ExpertSlot));
    for (int i = 0; i < cap; i++) lc->slots[i].eid = -1;
    if (n_experts > 0) {
        lc->heat = (uint32_t *)calloc(n_experts, sizeof(uint32_t));
        lc->last = (uint32_t *)calloc(n_experts, sizeof(uint32_t));
        lc->pinned = (uint8_t *)calloc(n_experts, sizeof(uint8_t));
    } else { lc->heat = lc->last = NULL; lc->pinned = NULL; }
}

void moty_expert_cache_free(ExpertCache *lc) {
    for (int i = 0; i < lc->n; i++) {
        free(lc->slots[i].g); free(lc->slots[i].u); free(lc->slots[i].d);
        free(lc->slots[i].gs); free(lc->slots[i].us); free(lc->slots[i].ds);
    }
    free(lc->slots); free(lc->heat); free(lc->last); free(lc->pinned);
}

ExpertSlot *moty_expert_pin(ExpertCache *lc, void *ctx, int layer, int eid,
                              LoadExpertFn fn, int inter, int hidden) {
    if (lc->pinned && eid >= 0 && eid < lc->n_experts) lc->pinned[eid] = 1;
    /* precarica in uno slot (un pinned conta nel cap; l'eviction lo salta) */
    ExpertSlot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
    } else {
        /* cap satura: trova un non-pinned da sacrificare (un pinned non va mai qui) */
        int v = -1;
        for (int i = 0; i < lc->n; i++) if (!lc->pinned || !lc->pinned[lc->slots[i].eid]) { v = i; break; }
        s = (v >= 0) ? &lc->slots[v] : &lc->slots[0];
    }
    int64_t ng = (int64_t)inter*hidden, nd = (int64_t)hidden*inter;
    if (!s->g) { s->g = (int8_t *)malloc(ng>0?ng:1); s->u = (int8_t *)malloc(ng>0?ng:1);
                 s->d = (int8_t *)malloc(nd>0?nd:1); s->gs = falloc(inter); s->us = falloc(inter); s->ds = falloc(hidden);
#ifdef MOTY_MADVISE
                 if (ng > (1<<18)) { madvise(s->g, ng, MADV_HUGEPAGE); madvise(s->u, ng, MADV_HUGEPAGE); }
                 if (nd > (1<<18)) { madvise(s->d, nd, MADV_HUGEPAGE); }
#endif
    }
    fn(ctx, layer, eid, s, inter, hidden);    /* engine riempie g/u/d + scale */
    s->eid = eid;
    return s;
}

ExpertSlot *moty_expert_get(ExpertCache *lc, void *ctx, int layer, int eid,
                              LoadExpertFn fn, int inter, int hidden) {
    /* track heat (frequency) + last (recency) per LFRU */
    if (lc->heat && eid >= 0 && eid < lc->n_experts) {
        if (lc->heat[eid] < 0xFFFFFFFF) lc->heat[eid]++;
        lc->last[eid] = lc->clock++;
    }
    /* HIT */
    for (int i = 0; i < lc->n; i++)
        if (lc->slots[i].eid == eid) { lc->hits++; return &lc->slots[i]; }
    /* MISS */
    lc->misses++;
    ExpertSlot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
    } else {
        /* eviction victim: il resident NON-pinned con LFRU-score minore.
         * tier_lfru_score = (heat<<8)|recent; recency spezza i pareggi. */
        int v = 0;
        if (lc->heat) {
            uint64_t worst = (uint64_t)-1;
            for (int i = 0; i < lc->n; i++) {
                int e = lc->slots[i].eid;
                if (lc->pinned && e >= 0 && e < lc->n_experts && lc->pinned[e]) continue;
                uint64_t sc = tier_lfru_score(lc->heat[e], lc->last[e], lc->clock);
                if (sc < worst) { worst = sc; v = i; }
            }
        } else {  /* fallback LRU-by-fill: il primo slot (piu' vecchio) */
            for (int i = 1; i < lc->n; i++) if (!lc->pinned || !lc->pinned[lc->slots[i].eid]) { v = i; break; }
        }
        s = &lc->slots[v];
    }
    int64_t ng = (int64_t)inter*hidden, nd = (int64_t)hidden*inter;
    if (!s->g) { s->g = (int8_t *)malloc(ng>0?ng:1); s->u = (int8_t *)malloc(ng>0?ng:1);
                 s->d = (int8_t *)malloc(nd>0?nd:1); s->gs = falloc(inter); s->us = falloc(inter); s->ds = falloc(hidden); }
    fn(ctx, layer, eid, s, inter, hidden);    /* engine riempie g/u/d + scale */
    s->eid = eid;
    return s;
}

void moty_moe_route_bias(float *logits, const ExpertCache *lc, float bias) {
    if (bias <= 0 || !lc) return;
    for (int i = 0; i < lc->n; i++)
        if (lc->slots[i].eid >= 0) logits[lc->slots[i].eid] += bias;
}

void moty_moe_topk(const float *logits, int E, int K, int *idx, float *w, int norm_topk) {
    for (int kk = 0; kk < K; kk++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int j = 0; j < kk; j++) if (idx[j] == e) { taken = 1; break; }
            if (!taken && logits[e] > bv) { bv = logits[e]; best = e; }
        }
        idx[kk] = best; w[kk] = bv;
    }
    if (norm_topk) {
        float sm = 0; for (int kk = 0; kk < K; kk++) sm += w[kk];
        if (sm > 0) for (int kk = 0; kk < K; kk++) w[kk] /= sm;
    }
}

