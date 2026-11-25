/* nn_attn_kernels.h — attention row primitives: score + accumulate.
 *
 * Shared by all attention implementations. Works with both f32 and int8 KV.
 * Depends on: hw.h (dot_f32, dot_f32i8), nn_alloc.h.
 */
#ifndef NN_ATTN_KERNELS_H
#include <stdint.h>
#define NN_ATTN_KERNELS_H
#include <stdint.h>

struct Scratch;
void moty_attn_prefill_group(float *out, int stride, const float *q,
                            float *const *scores, const float *k, const float *v,
                            int n, int count, float scale);
void moty_att_scores_kv4_pair(float *s0, float *s1, const float *q0, const float *q1,
                              const float *k, int n, float scale);
void moty_att_scores_kv4(float *scores, const float *q, const float *k, int n, float scale);
int moty_attn_gqa64_decode_batch(float *out, const float *q, const float *k, const float *v,
                                int kv_heads, int stride, int visible, int n, float scale, struct Scratch *scratch);
int moty_attn_gqa64_layout(float *out, const float *q, const float *k, const float *v,
                           int kv_heads, int kv_stride, int visible, float scale, struct Scratch *scratch, int kv4);
int moty_attn_gqa64(float *out, const float *q, const float *k, const float *v,
                   int kv_heads, int kv_stride, int visible, float scale, struct Scratch *scratch);








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
