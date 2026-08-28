/* KV-cache: wrapper sottili sui pezzi di libmoty-runtime (M4).
 * Include DOPO Cfg/Layer/Model. I motori chiamano ancora kv_arrays_alloc/
 * kv_layer_alloc(m, ...): il wrapper instrada su MotyCommon. */
#ifndef RT_KV_CACHE_H
#define RT_KV_CACHE_H
#include "runtime/kvcache.h"

static void kv_arrays_alloc(Model *m, int max_t) {
    moty_rt_kv_arrays_alloc(&m->base, m->c.n_layers, max_t);
}

static void kv_layer_alloc(Model *m, int i, int KV, int hd, int max_t) {
    moty_rt_kv_layer_alloc(&m->base, i, KV, hd, max_t, g_kv_bits == 8);
}

#endif /* RT_KV_CACHE_H */
