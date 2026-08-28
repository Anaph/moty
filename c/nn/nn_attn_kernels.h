/* nn_attn_kernels.h — attention row primitives: score + accumulate.
 *
 * Shared by all attention implementations. Works with both f32 and int8 KV.
 * Depends on: hw.h (dot_f32, dot_f32i8), nn_alloc.h.
 */
#ifndef NN_ATTN_KERNELS_H
#include <stdint.h>
#define NN_ATTN_KERNELS_H
#include <stdint.h>









void moty_att_scores_f32(float *sc, const float *qv, const float *K,
                                  int64_t kvbase, int t0, int qpos, int hd, float scale);
void moty_att_scores_i8(float *sc, const float *qv, const int8_t *K8,
                                 const float *Ks, int64_t kvbase, int t0, int qpos,
                                 int hd, float scale);
void moty_att_accum_f32(float *cx, const float *sc, const float *V,
                                 int64_t kvbase, int t0, int qpos, int hd);
void moty_att_accum_i8(float *cx, const float *sc, const int8_t *V8,
                                const float *Vs, int64_t kvbase, int t0, int qpos, int hd);

#ifndef MOTY_CORE_NO_LEGACY
#define att_scores_f32 moty_att_scores_f32
#define att_scores_i8 moty_att_scores_i8
#define att_accum_f32 moty_att_accum_f32
#define att_accum_i8 moty_att_accum_i8
#endif

#endif /* NN_ATTN_KERNELS_H */
