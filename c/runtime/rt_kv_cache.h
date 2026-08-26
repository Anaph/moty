/* KV-cache: kv_arrays_alloc/kv_layer_alloc, embed_row (+g_kv_bits).
 * Estratto da runtime.h (P2). Include DOPO Cfg/Layer/Model
 * e le dichiarazioni degli hook. Un'istanza per TU (tutto static). */
#ifndef RT_KV_CACHE_H
#define RT_KV_CACHE_H

/* ---------- KV-cache: pezzi comuni di kv_alloc (qwen e gemma) ----------
 * Prologo: array di puntatori per-layer + scratch degli score. Lo scratch e'
 * dimensionato QUI perche' THREADS viene applicato prima (engine_main) e
 * OMP_DYNAMIC=FALSE tiene il team fisso: alzare i thread dopo kv_alloc non e'
 * supportato. Al motore restano solo le decisioni per-layer (quale saltare,
 * eventuale aliasing kv-shared). */
static void kv_arrays_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    m->max_t = max_t; m->kv_len = 0;
    m->att_sc = falloc((int64_t)omp_get_max_threads() * max_t);
    m->K  = bzalloc(c->n_layers * sizeof(float*),  "array K");
    m->V  = bzalloc(c->n_layers * sizeof(float*),  "array V");
    m->K8 = bzalloc(c->n_layers * sizeof(int8_t*), "array K8");
    m->V8 = bzalloc(c->n_layers * sizeof(int8_t*), "array V8");
    m->Ks = bzalloc(c->n_layers * sizeof(float*),  "array Ks");
    m->Vs = bzalloc(c->n_layers * sizeof(float*),  "array Vs");
}

/* un layer di KV: f32, oppure int8 + scala per (testa, pos) con KV_BITS=8 */
static void kv_layer_alloc(Model *m, int i, int KV, int hd, int max_t) {
    int64_t n = (int64_t)KV * max_t * hd;
    if (g_kv_bits == 8) {                    /* 4x meno RAM per la cache */
        m->K8[i] = balloc(n, "KV int8"); m->V8[i] = balloc(n, "KV int8");
        m->Ks[i] = falloc((int64_t)KV * max_t);
        m->Vs[i] = falloc((int64_t)KV * max_t);
    } else {
        m->K[i] = falloc(n);
        m->V[i] = falloc(n);
    }
}

#endif /* RT_KV_CACHE_H */
