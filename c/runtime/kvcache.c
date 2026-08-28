/* kvcache.c — M4 libmoty-runtime: allocazione KV (f32/int8) + score scratch.
 * Corpo portato da rt_kv_cache.h; opera su MotyCommon, non sul Model. */
#include "runtime/kvcache.h"

void moty_rt_kv_arrays_alloc(MotyCommon *mc, int n_layers, int max_t) {
    mc->max_t = max_t; mc->kv_len = 0;
    mc->att_sc = falloc((int64_t)omp_get_max_threads() * max_t);
    mc->K  = bzalloc(n_layers * sizeof(float*),  "array K");
    mc->V  = bzalloc(n_layers * sizeof(float*),  "array V");
    mc->K8 = bzalloc(n_layers * sizeof(int8_t*), "array K8");
    mc->V8 = bzalloc(n_layers * sizeof(int8_t*), "array V8");
    mc->Ks = bzalloc(n_layers * sizeof(float*),  "array Ks");
    mc->Vs = bzalloc(n_layers * sizeof(float*),  "array Vs");
}

void moty_rt_kv_layer_alloc(MotyCommon *mc, int i, int KV, int hd, int max_t, int kv8) {
    int64_t n = (int64_t)KV * max_t * hd;
    if (kv8) {
        mc->K8[i] = balloc(n, "KV int8"); mc->V8[i] = balloc(n, "KV int8");
        mc->Ks[i] = falloc((int64_t)KV * max_t);
        mc->Vs[i] = falloc((int64_t)KV * max_t);
    } else {
        mc->K[i] = falloc(n);
        mc->V[i] = falloc(n);
    }
}
