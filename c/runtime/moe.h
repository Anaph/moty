/* moe.h — infrastruttura MoE riusabile: cache expert hot/LFRU + routing top-K.
 *
 * Seed per nuovi motori MoE: un engine include questo header, fornisce l'hook
 * load_expert_fn (come leggere g/u/d di un expert) e ottiene:
 *   - cache per-layer con eviction LFRU (frequency×256 + recency, vedi tier.h):
 *     gli expert "caldi" (scelti spesso dal router) restano residenti, i freddi
 *     vengono riletii dal disco solo se ri-selezionati. cap>=n_experts => tutto
 *     residente, NESSUNA eviction (fast path su RAM larga, zero overhead).
 *   - PIN: expert marcati permanenti (precaricati, mai evicti) -> EXPERT_PIN/
 *     AUTOPIN. tier.h gestisce gia' il set residente + hysteresis.
 *   - CACHE_ROUTE: bias del router verso gli expert gia' residenti -> taglia
 *     l'I/O su disco (moe_route_bias prima di softmax/topk).
 *   - moe_topk: greedy top-K bit-identico al naive.
 * glm.c mantiene la sua cache ancora piu' ricca (io_uring/PIPE/CUDA) legata al
 * suo path tile+DSA; questa e' la baseline dependency-free (porta pin/LRU/CACHE
 * di glm nel core condiviso).
 *
 * Lo slot modella il MLP di un expert come int8 per-riga + scala (schema Q8_0,
 * dequant-on-use via matmul_q di nn.h). gate/up [inter,D], down [D,inter];
 * l'engine li applica con la sua attivazione (SwiGLU/GeGLU/...).
 *
 * Tutto static (un'istanza per translation unit), stesso pattern di nn.h. */
#ifndef MOE_H
#define MOE_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32) && defined(MADV_HUGEPAGE)
#define MOTY_MADVISE 1
#include <sys/mman.h>        /* madvise(MADV_HUGEPAGE) per i buffer expert (riduce TLB-miss) */
#endif
#include "nn/nn.h"            /* falloc per le scale; matmul_q lo usa il chiamante */
#include "io/tier.h"          /* tier_lfru_score per l'eviction frequency+recency */

typedef struct {
    int eid;                /* expert id, -1 = slot vuoto */
    int8_t *g, *u, *d;      /* gate [inter*D], up [inter*D], down [D*inter] */
    float   *gs, *us, *ds;  /* scale per riga: [inter], [inter], [D] */
} ExpertSlot;

typedef struct {
    ExpertSlot *slots;      /* [cap] */
    int n, cap;             /* n = slot riempiti */
    /* hot-caching. Sempre allocati se n_experts>0 (2KB/layer per 256 expert):
     * a costo nullo danno eviction LFRU + pin. n_experts==0 -> LRU-by-clock. */
    uint32_t *heat;         /* [n_experts] contatori frequenza (tier.h) */
    uint32_t *last;         /* [n_experts] tick di ultimo accesso */
    uint8_t  *pinned;       /* [n_experts] 1 = permanent-pin (mai evicto) */
    int n_experts;
    uint32_t clock;         /* clock LFRU interno */
    long long hits, misses;
} ExpertCache;

/* hook: l'engine legge i pesi di UN expert dal disco e riempie s->g/u/d/gs/us/ds.
 * ctx opaco (Model* / shards*). inter/hidden dimensionano. */
typedef void (*LoadExpertFn)(void *ctx, int layer, int eid, ExpertSlot *s, int inter, int hidden);

/* alloca una cache per layer con `cap` slot e tracking heat/pin per n_experts.
 * cap<1 -> 1; n_experts<0 -> 0 (LRU semplice senza heat). */




/* marca `eid` come permanent-pin e lo precarica subito (slot dedicato, mai
 * evicto). Per EXPERT_PIN=<lista> / AUTOPIN: chiamare dopo init sui top-hot. */


/* ritorna lo slot per (layer,eid): HIT (aggiorna heat/last) o MISS (carica via
 * fn, evictando il cold piu' FREDDO tra i non-pinned). Primo get di uno slot
 * alloca i buffer g/u/d/gs/us/ds una volta per tutte (riusati dopo ogni
 * eviction). cap>=n_experts => niente eviction mai (fast path residente). */


/* CACHE_ROUTE: aggiungi `bias` ai logits degli expert gia' residenti (in slot)
 * -> il router softmax li preferisce, tagliando i miss/IO. bias<=0 = no-op.
 * Chiamare PRIMA di softmax/topk, per-token. Piccolo compromesso numerico
 * (le probabilita' si spostano verso i residenti): attivo solo sotto MEM_GB. */


/* top-K greedy da logits[E] (gia' softmaxati) -> idx[K], w[K].
 * Algoritmo IDENTICO alla selezione greedy O(K*E) originale di olmoe.c: per
 * ogni kk il max non ancora preso, tie-break sull'indice piu' basso. ->
 * bit-identico al naive, estratto per riuso. norm_topk: normalizza a somma 1. */



/* M3: implementazioni in runtime/moecache.c (libmoty-nn) */
void moty_expert_cache_init(ExpertCache *lc, int cap, int n_experts);
void moty_expert_cache_free(ExpertCache *lc);
ExpertSlot * moty_expert_pin(ExpertCache *lc, void *ctx, int layer, int eid,
                              LoadExpertFn fn, int inter, int hidden);
ExpertSlot * moty_expert_get(ExpertCache *lc, void *ctx, int layer, int eid,
                              LoadExpertFn fn, int inter, int hidden);
void moty_moe_route_bias(float *logits, const ExpertCache *lc, float bias);
void moty_moe_topk(const float *logits, int E, int K, int *idx, float *w, int norm_topk);

#ifndef MOTY_CORE_NO_LEGACY
#define expert_cache_init moty_expert_cache_init
#define expert_cache_free moty_expert_cache_free
#define expert_pin moty_expert_pin
#define expert_get moty_expert_get
#define moe_route_bias moty_moe_route_bias
#define moe_topk moty_moe_topk
#endif

#endif /* MOE_H */
