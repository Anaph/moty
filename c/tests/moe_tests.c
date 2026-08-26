/* Test di c/moe.h: topk greedy bit-identico al naive, cache LFRU (heat+recency)
 * con hit/miss/eviction e riuso slot, PIN (expert mai evicto), CACHE_ROUTE
 * (bias del router verso i residenti). Convenzione 0=pass 1=fail 2=skip. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../runtime/moe.h"

static uint64_t mt_rng = 0xABCDEF12345ULL;
static uint32_t mt_rnd(void){ mt_rng = mt_rng*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t)(mt_rng>>33); }
static float mt_fr(void){ return (float)((int)mt_rnd()%2000 - 1000) / 137.0f; }

/* moe_topk vs naive greedy O(K*E). */
int moe_topk_ref(void){
    int idx[64], idx2[64]; float w[64], w2[64];
    for (int rep = 0; rep < 200; rep++) {
        int E = 2 + mt_rnd()%40, K = 1 + mt_rnd()% (E<8?E-1:8); if (K>64) K=64;
        float lg[64];
        for (int e = 0; e < E; e++) lg[e] = mt_fr();
        int nt = mt_rnd()&1;
        moe_topk(lg, E, K, idx, w, nt);
        for (int kk = 0; kk < K; kk++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0; for (int j = 0; j < kk; j++) if (idx2[j]==e){taken=1;break;}
                if (!taken && lg[e] > bv) { bv = lg[e]; best = e; }
            }
            idx2[kk] = best; w2[kk] = bv;
        }
        if (nt) { float sm=0; for(int kk=0;kk<K;kk++) sm+=w2[kk]; if(sm>0) for(int kk=0;kk<K;kk++) w2[kk]/=sm; }
        for (int kk = 0; kk < K; kk++)
            if (idx[kk] != idx2[kk] || w[kk] != w2[kk]) {
                fprintf(stderr,"moe_topk rep=%d kk=%d: idx %d/%d w %.8f/%.8f\n",rep,kk,idx[kk],idx2[kk],w[kk],w2[kk]);
                return 1;
            }
    }
    return 0;
}

static int g_calls = 0;
static void mock_load(void *ctx, int layer, int eid, ExpertSlot *s, int inter, int hidden){
    (void)ctx; (void)layer;
    g_calls++;
    for (int i = 0; i < inter*hidden; i++) s->g[i] = (int8_t)(eid & 0x7F);
    for (int i = 0; i < inter*hidden; i++) s->u[i] = (int8_t)((eid*7) & 0x7F);
    for (int i = 0; i < hidden*inter; i++) s->d[i] = (int8_t)((eid*13) & 0x7F);
    for (int i = 0; i < inter; i++) s->gs[i] = 1.0f, s->us[i] = 1.0f;
    for (int i = 0; i < hidden; i++) s->ds[i] = 1.0f;
}

/* LFRU: heat (frequency) primario, recency spezza i pareggi. cap=2, n_experts=16.
 * Sequenza: e0,e1,e0,e2(evict e1: e0=2 accessi, e1=1), e0(hit), e1(evict e2). */
int moe_cache_lfru(void){
    ExpertCache lc; expert_cache_init(&lc, 2, 16);
    int inter = 4, hidden = 3;
    ExpertSlot *a, *b;
    a = expert_get(&lc, NULL, 0, 0, mock_load, inter, hidden);
    if (a->eid != 0 || lc.hits || lc.misses != 1) { fprintf(stderr,"lfru: e0 hit=%lld miss=%lld\n",lc.hits,lc.misses); return 1; }
    void *g0 = a->g;
    b = expert_get(&lc, NULL, 0, 1, mock_load, inter, hidden);
    if (b->eid != 1 || b->g == g0 || lc.misses != 2) { fprintf(stderr,"lfru: e1 slot\n"); return 1; }
    a = expert_get(&lc, NULL, 0, 0, mock_load, inter, hidden);   /* HIT e0 */
    if (a->eid != 0 || a->g != g0 || lc.hits != 1 || lc.misses != 2) { fprintf(stderr,"lfru: e0 hit\n"); return 1; }
    /* e2: cache piena {e0,e1}, e1 e' il piu' freddo (1 accesso vs 2 di e0) -> evict e1 */
    b = expert_get(&lc, NULL, 0, 2, mock_load, inter, hidden);
    if (b->eid != 2 || lc.hits != 1 || lc.misses != 3) { fprintf(stderr,"lfru: e2 evict\n"); return 1; }
    /* e0 ancora residente (non era il cold) */
    a = expert_get(&lc, NULL, 0, 0, mock_load, inter, hidden);
    if (a->eid != 0 || a->g != g0 || lc.hits != 2 || lc.misses != 3) { fprintf(stderr,"lfru: e0 still resident\n"); return 1; }
    /* e1: rimosso prima -> miss; ora il cold tra {e0,e2}: e2 ha 1 accesso, e0 ne ha 3 -> evict e2 */
    b = expert_get(&lc, NULL, 0, 1, mock_load, inter, hidden);
    if (b->eid != 1 || lc.hits != 2 || lc.misses != 4) { fprintf(stderr,"lfru: e1 reload\n"); return 1; }
    if (b->g[0] != (int8_t)(1 & 0x7F)) { fprintf(stderr,"lfru: content marker g[0]=%d\n",b->g[0]); return 1; }
    expert_cache_free(&lc);
    return 0;
}

/* cap<1 -> 1: un solo slot, ogni expert diverso evicta il precedente. */
int moe_cache_cap1(void){
    ExpertCache lc; expert_cache_init(&lc, 0, 8);   /* 0 -> cap 1 */
    if (lc.cap != 1) { fprintf(stderr,"cap1: cap=%d\n", lc.cap); return 1; }
    ExpertSlot *a = expert_get(&lc, NULL, 0, 1, mock_load, 2, 2);
    ExpertSlot *b = expert_get(&lc, NULL, 0, 2, mock_load, 2, 2);
    if (a != b || lc.hits || lc.misses != 2) { fprintf(stderr,"cap1: same slot\n"); return 1; }
    expert_cache_free(&lc);
    return 0;
}

/* PIN: expert marcato permanent non viene mai evicto, anche sotto pressione. */
int moe_cache_pin(void){
    ExpertCache lc; expert_cache_init(&lc, 2, 8);   /* cap 2 */
    /* pin e0 (occupa slot 0, mai evicto), poi riempi con e1, poi richiedi e2 */
    expert_pin(&lc, NULL, 0, 0, mock_load, 2, 2);
    expert_get(&lc, NULL, 0, 1, mock_load, 2, 2);   /* slot 1 = e1 */
    /* e2: cache piena, ma e0 e' pinned -> deve evictare e1 (non e0) */
    ExpertSlot *s2 = expert_get(&lc, NULL, 0, 2, mock_load, 2, 2);
    if (s2->eid != 2) { fprintf(stderr,"pin: e2 not loaded\n"); return 1; }
    /* e0 ancora residente (pinned, non toccato) */
    ExpertSlot *s0 = expert_get(&lc, NULL, 0, 0, mock_load, 2, 2);
    if (s0->eid != 0 || lc.hits < 1) { fprintf(stderr,"pin: e0 evicted (should be pinned!)\n"); return 1; }
    expert_cache_free(&lc);
    return 0;
}

/* CACHE_ROUTE: bias dei logit verso i residenti sposta l'argmax. */
int moe_route_bias_test(void){
    ExpertCache lc; expert_cache_init(&lc, 2, 8);
    expert_get(&lc, NULL, 0, 3, mock_load, 2, 2);   /* residente: solo eid 3 */
    float lg[8]; for (int i = 0; i < 8; i++) lg[i] = (float)i;   /* max = e7 */
    int idx[1]; float w[1];
    moe_topk(lg, 8, 1, idx, w, 0);
    if (idx[0] != 7) { fprintf(stderr,"bias: pre argmax=%d (!=7)\n",idx[0]); return 1; }
    moe_route_bias(lg, &lc, 100.0f);                /* e3 ora >> altri */
    moe_topk(lg, 8, 1, idx, w, 0);
    if (idx[0] != 3) { fprintf(stderr,"bias: post argmax=%d (!=3 residente)\n",idx[0]); return 1; }
    expert_cache_free(&lc);
    return 0;
}
