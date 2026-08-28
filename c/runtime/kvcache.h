/* kvcache.h — M4: KV alloc su MotyCommon (libmoty-runtime). */
#ifndef MOTY_RT_KVCACHE_H
#define MOTY_RT_KVCACHE_H
#include "nn/nn_mat.h"          /* MotyCommon, falloc/bzalloc (macro legacy ok) */
#ifdef _OPENMP
#include <omp.h>
#endif

void moty_rt_kv_arrays_alloc(MotyCommon *mc, int n_layers, int max_t);
void moty_rt_kv_layer_alloc(MotyCommon *mc, int i, int KV, int hd, int max_t, int kv8);

#endif /* MOTY_RT_KVCACHE_H */
