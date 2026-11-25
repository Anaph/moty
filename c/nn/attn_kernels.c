/* attn_kernels.c — M3 libmoty-nn */
#include "hw/hw.h"
#include "nn/nn_attn_kernels.h"
#include "nn/nn_alloc.h"
#include "nn/nn_norm.h"
#include "nn/nn_matmul.h"
#include <string.h>
#ifdef MOTY_THREADPOOL
#include <stdatomic.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

enum { GQA_DIM = 64, GQA_GROUP = 7, GQA_BLOCK = 128, GQA_MIN_VISIBLE = 2048 };

typedef enum { GQA_SCORE_PAIR, GQA_SCORE_PAIR_WIDE, GQA_SCORE_TRIPLE } GqaScoreTile;
typedef struct {
    GqaScoreTile score_tile;
    int block_first, merge_wide, team;
} GqaOptions;

typedef struct {
    float *out, *scores, *partial;
    const float *q, *k, *v;
    int stride, visible, blocks, groups, kv4;
    float scale;
    GqaOptions options;
} GqaJob;

static inline float32x4_t kv4_reduce(const float32x4_t *a, float scale) {
    float32x4_t s01 = vaddq_f32(vaddq_f32(a[0],a[4]),vaddq_f32(a[1],a[5]));
    float32x4_t s23 = vaddq_f32(vaddq_f32(a[2],a[6]),vaddq_f32(a[3],a[7]));
    return vmulq_n_f32(vaddq_f32(s01,s23),scale);
}

/* Each vector lane is a position. Preserve the two chains and the
 * horizontal addition tree of the row-major FP32 score kernel. */
static void att_scores_kv4(float *scores, const float *q, const float *key, int n, float scale) {
    int t = 0;
    for (; t < (n & ~1); t += 4, key += 4*GQA_DIM) {
        float32x4_t a[8];
        for (int d = 0; d < 8; d++) a[d] = vmulq_n_f32(vld1q_f32(key+4*d),q[d]);
        for (int d = 8; d < GQA_DIM; d += 8)
            for (int j = 0; j < 8; j++) a[j] = vfmaq_n_f32(a[j],vld1q_f32(key+4*(d+j)),q[d+j]);
        float32x4_t v = kv4_reduce(a,scale);
        if (t+4 <= (n & ~1)) vst1q_f32(scores+t,v);
        else vst1_f32(scores+t,vget_low_f32(v));
    }
    if (n & 1) {
        t = n-1;
        key -= (t%4 ? 4*GQA_DIM : 0);
        float row[GQA_DIM];
        for (int d = 0; d < GQA_DIM; d++) row[d] = key[4*d+t%4];
        moty_hw_attn_scores(scores+t,q,row,1,GQA_DIM,scale);
    }
}

static void att_scores_kv4_pair(float *s0, float *s1, const float *q0, const float *q1,
                                 const float *k, int n0, int n1, float scale) {
    int prefix = n0 & ~3;
    for (int t = 0; t < prefix; t += 4, k += 4*GQA_DIM) {
        float32x4_t a[8], b[8];
        for (int j = 0; j < 8; j++) {
            float32x4_t w = vld1q_f32(k+4*j);
            a[j] = vmulq_n_f32(w,q0[j]); b[j] = vmulq_n_f32(w,q1[j]);
        }
        for (int d = 8; d < GQA_DIM; d += 8)
            for (int j = 0; j < 8; j++) {
                float32x4_t w = vld1q_f32(k+4*(d+j));
                a[j] = vfmaq_n_f32(a[j],w,q0[d+j]);
                b[j] = vfmaq_n_f32(b[j],w,q1[d+j]);
            }
        vst1q_f32(s0+t,kv4_reduce(a,scale));
        vst1q_f32(s1+t,kv4_reduce(b,scale));
    }
    att_scores_kv4(s0+prefix,q0,k,n0-prefix,scale);
    att_scores_kv4(s1+prefix,q1,k,n1-prefix,scale);
}

/* Match the two-accumulator FP32 score kernel's reduction for each query. */
static void gqa_scores_pair(float *s0, float *s1, const float *q0, const float *q1,
                            const float *key, int n, float scale) {
    int t = 0;
    for (; t + 1 < n; t += 2) {
        const float *k0 = key + (int64_t)t*GQA_DIM, *k1 = k0+GQA_DIM;
        float32x4_t q00 = vld1q_f32(q0), q01 = vld1q_f32(q0+4);
        float32x4_t q10 = vld1q_f32(q1), q11 = vld1q_f32(q1+4);
        float32x4_t k00 = vld1q_f32(k0), k01 = vld1q_f32(k0+4);
        float32x4_t k10 = vld1q_f32(k1), k11 = vld1q_f32(k1+4);
        float32x4_t a0 = vmulq_f32(q00,k00), a1 = vmulq_f32(q01,k01);
        float32x4_t b0 = vmulq_f32(q10,k00), b1 = vmulq_f32(q11,k01);
        float32x4_t c0 = vmulq_f32(q00,k10), c1 = vmulq_f32(q01,k11);
        float32x4_t d0 = vmulq_f32(q10,k10), d1 = vmulq_f32(q11,k11);
        _Pragma("GCC unroll 8")
        for (int d = 8; d < GQA_DIM; d += 8) {
            q00 = vld1q_f32(q0+d); q01 = vld1q_f32(q0+d+4);
            q10 = vld1q_f32(q1+d); q11 = vld1q_f32(q1+d+4);
            k00 = vld1q_f32(k0+d); k01 = vld1q_f32(k0+d+4);
            k10 = vld1q_f32(k1+d); k11 = vld1q_f32(k1+d+4);
            a0 = vfmaq_f32(a0,q00,k00); a1 = vfmaq_f32(a1,q01,k01);
            b0 = vfmaq_f32(b0,q10,k00); b1 = vfmaq_f32(b1,q11,k01);
            c0 = vfmaq_f32(c0,q00,k10); c1 = vfmaq_f32(c1,q01,k11);
            d0 = vfmaq_f32(d0,q10,k10); d1 = vfmaq_f32(d1,q11,k11);
        }
        s0[t] = vaddvq_f32(vaddq_f32(a0,a1))*scale;
        s1[t] = vaddvq_f32(vaddq_f32(b0,b1))*scale;
        s0[t+1] = vaddvq_f32(vaddq_f32(c0,c1))*scale;
        s1[t+1] = vaddvq_f32(vaddq_f32(d0,d1))*scale;
    }
    if (t < n) {
        moty_hw_attn_scores(s0+t,q0,key+(int64_t)t*GQA_DIM,1,GQA_DIM,scale);
        moty_hw_attn_scores(s1+t,q1,key+(int64_t)t*GQA_DIM,1,GQA_DIM,scale);
    }
}

/* Each (head, position) keeps the two original FP32 chains. Only the
 * order of loading and advancing independent chains changes. */
static inline __attribute__((always_inline)) void gqa_scores_tile(
        float *scores, int stride, const float *q, const float *key,
        int n, float scale, int heads, int positions) {
    int t = 0;
    for (; n-t >= positions; t += positions) {
        const float *k = key+(int64_t)t*GQA_DIM;
        float32x4_t lo[3][4], hi[3][4];
        for (int d = 0; d < GQA_DIM; d += 8) {
            float32x4_t q0[3], q1[3];
            for (int h = 0; h < heads; h++) {
                q0[h] = vld1q_f32(q+h*GQA_DIM+d);
                q1[h] = vld1q_f32(q+h*GQA_DIM+d+4);
            }
            for (int pos = 0; pos < positions; pos++) {
                float32x4_t k0 = vld1q_f32(k+pos*GQA_DIM+d);
                float32x4_t k1 = vld1q_f32(k+pos*GQA_DIM+d+4);
                for (int h = 0; h < heads; h++) {
                    lo[h][pos] = d ? vfmaq_f32(lo[h][pos],q0[h],k0) : vmulq_f32(q0[h],k0);
                    hi[h][pos] = d ? vfmaq_f32(hi[h][pos],q1[h],k1) : vmulq_f32(q1[h],k1);
                }
            }
        }
        for (int h = 0; h < heads; h++)
            for (int pos = 0; pos < positions; pos++)
                scores[(int64_t)h*stride+t+pos] = vaddvq_f32(vaddq_f32(lo[h][pos],hi[h][pos]))*scale;
    }
    for (int h = 0; t < n && h < heads; h++)
        moty_hw_attn_scores(scores+(int64_t)h*stride+t,q+h*GQA_DIM,key+(int64_t)t*GQA_DIM,n-t,GQA_DIM,scale);
}

static void gqa_values_pair(float *out0, float *out1, const float *s0, const float *s1,
                            const float *value, int n) {
    /* Two half-head sweeps leave room for both sets of accumulators. */
    for (int d = 0; d < GQA_DIM; d += 32) {
        float32x4_t a[8], b[8];
        for (int i = 0; i < 8; i++) a[i] = b[i] = vdupq_n_f32(0);
        int t = 0;
        for (; t + 1 < n; t += 2) {
            const float *v = value + (int64_t)t*GQA_DIM+d;
            float w0 = s0[t], w1 = s1[t], w2 = s0[t+1], w3 = s1[t+1];
            _Pragma("GCC unroll 8")
            for (int i = 0; i < 8; i++) {
                float32x4_t x = vld1q_f32(v+4*i), y = vld1q_f32(v+GQA_DIM+4*i);
                a[i] = vfmaq_n_f32(vfmaq_n_f32(a[i],x,w0),y,w2);
                b[i] = vfmaq_n_f32(vfmaq_n_f32(b[i],x,w1),y,w3);
            }
        }
        if (t < n) {
            const float *v = value + (int64_t)t*GQA_DIM+d;
            for (int i = 0; i < 8; i++) {
                float32x4_t x = vld1q_f32(v+4*i);
                a[i] = vfmaq_n_f32(a[i],x,s0[t]); b[i] = vfmaq_n_f32(b[i],x,s1[t]);
            }
        }
        for (int i = 0; i < 8; i++) {
            vst1q_f32(out0+d+4*i,a[i]); vst1q_f32(out1+d+4*i,b[i]);
        }
    }
}

static void gqa_scores_four(float *const *score, const float *q, const float *key,
                            int n, float scale) {
    int t = 0;
    for (; t+1 < n; t += 2) {
        const float *k = key+(int64_t)t*GQA_DIM;
        float32x4_t lo[4], hi[4], next_lo[4], next_hi[4];
        float32x4_t k0 = vld1q_f32(k), k1 = vld1q_f32(k+4);
        float32x4_t k2 = vld1q_f32(k+GQA_DIM), k3 = vld1q_f32(k+GQA_DIM+4);
        for (int h = 0; h < 4; h++) {
            const float *qh = q+h*GQA_DIM;
            float32x4_t q0 = vld1q_f32(qh), q1 = vld1q_f32(qh+4);
            lo[h] = vmulq_f32(q0,k0); hi[h] = vmulq_f32(q1,k1);
            next_lo[h] = vmulq_f32(q0,k2); next_hi[h] = vmulq_f32(q1,k3);
        }
        _Pragma("GCC unroll 4")
        for (int d = 8; d < GQA_DIM; d += 8) {
            k0 = vld1q_f32(k+d); k1 = vld1q_f32(k+d+4);
            k2 = vld1q_f32(k+d+GQA_DIM); k3 = vld1q_f32(k+d+GQA_DIM+4);
            for (int h = 0; h < 4; h++) {
                const float *qh = q+h*GQA_DIM+d;
                float32x4_t q0 = vld1q_f32(qh), q1 = vld1q_f32(qh+4);
                lo[h] = vfmaq_f32(lo[h],q0,k0); hi[h] = vfmaq_f32(hi[h],q1,k1);
                next_lo[h] = vfmaq_f32(next_lo[h],q0,k2); next_hi[h] = vfmaq_f32(next_hi[h],q1,k3);
            }
        }
        for (int h = 0; h < 4; h++) {
            float *row = score[h]+t;
            row[0] = vaddvq_f32(vaddq_f32(lo[h],hi[h]))*scale;
            row[1] = vaddvq_f32(vaddq_f32(next_lo[h],next_hi[h]))*scale;
        }
    }
    if (t < n)
        for (int h = 0; h < 4; h++)
            moty_hw_attn_scores(score[h]+t,q+h*GQA_DIM,key+(int64_t)t*GQA_DIM,1,GQA_DIM,scale);
}

static void gqa_values_four(float *out, int out_stride, const float *const *scores,
                            const float *value, int n) {
    for (int d = 0; d < GQA_DIM; d += 16) {
        float32x4_t acc[4][4] = {0};
        for (int t = 0; t < n; t++) {
            const float *v = value+(int64_t)t*GQA_DIM+d;
            const float32x4_t x[] = {vld1q_f32(v), vld1q_f32(v+4),
                                     vld1q_f32(v+8), vld1q_f32(v+12)};
            for (int h = 0; h < 4; h++) {
                float w = scores[h][t];
                for (int i = 0; i < 4; i++) acc[h][i] = vfmaq_n_f32(acc[h][i],x[i],w);
            }
        }
        for (int h = 0; h < 4; h++)
            for (int i = 0; i < 4; i++) vst1q_f32(out+(int64_t)h*out_stride+d+4*i,acc[h][i]);
    }
}

/* Share each V load across three heads. Each lane still applies the
 * FP32 FMAs in position order within the original block. */
static void gqa_values_three(float *out, int out_stride, const float *scores, int stride,
                             const float *value, int n) {
    const float *s1 = scores+stride, *s2 = s1+stride;
    for (int d = 0; d < GQA_DIM; d += 16) {
        float32x4_t a[4] = {0}, b[4] = {0}, c[4] = {0};
        for (int t = 0; t < n; t++) {
            const float *v = value+(int64_t)t*GQA_DIM+d;
            float w0 = scores[t], w1 = s1[t], w2 = s2[t];
            _Pragma("GCC unroll 4")
            for (int i = 0; i < 4; i++) {
                float32x4_t x = vld1q_f32(v+4*i);
                a[i] = vfmaq_n_f32(a[i],x,w0);
                b[i] = vfmaq_n_f32(b[i],x,w1);
                c[i] = vfmaq_n_f32(c[i],x,w2);
            }
        }
        for (int i = 0; i < 4; i++) {
            vst1q_f32(out+d+4*i,a[i]);
            vst1q_f32(out+out_stride+d+4*i,b[i]);
            vst1q_f32(out+2*(int64_t)out_stride+d+4*i,c[i]);
        }
    }
}

enum { GQA_VALUE_GROUPS = 3 };

static void gqa_value_three_work(void *job, int64_t first, int64_t last, int tid) {
    const GqaJob *c = job;
    for (int64_t task = first; task < last; task++) {
        /* Interleave both triple-head tasks and the lighter singleton
         * for each block so the static worker ranges stay balanced. */
        int slot = (int)(task%GQA_VALUE_GROUPS);
        int64_t tile = task/GQA_VALUE_GROUPS;
        int kv = (int)(tile/c->blocks), block = (int)(tile%c->blocks);
        int h = kv*GQA_GROUP+3*slot;
        int begin = block*GQA_BLOCK, n = c->visible-begin;
        if (n > GQA_BLOCK) n = GQA_BLOCK;
        const float *v = c->v+((int64_t)kv*c->stride+begin)*GQA_DIM;
        const float *scores = c->scores+(int64_t)h*c->visible+begin;
        float *out = c->partial+((int64_t)h*c->blocks+block)*GQA_DIM;
        if (slot < 2) gqa_values_three(out,c->blocks*GQA_DIM,scores,c->visible,v,n);
        else moty_hw_attn_accum(out,scores,v,n,GQA_DIM);
    }
}

/* Head spans within one KV head. The last span is always the singleton. */
static inline int gqa_head_offset(int groups, int slot) {
    return 2*slot + (groups == 3 && slot > 0 ? 2 : 0);
}

static inline int gqa_head_count(int groups, int slot) {
    return slot == groups-1 ? 1 : groups == 3 && slot == 0 ? 4 : 2;
}

static int gqa_score_groups(const GqaJob *c) {
    return c->options.score_tile == GQA_SCORE_TRIPLE ? GQA_VALUE_GROUPS : c->groups;
}

static void gqa_blocks(const GqaJob *c, int64_t first, int64_t last, int values, int pair_owned) {
    int groups = values ? c->groups : gqa_score_groups(c);
    int triple = !values && c->options.score_tile == GQA_SCORE_TRIPLE;
    int block_first = !values && !pair_owned && c->options.block_first;
    for (int64_t task = first; task < last; task++) {
        int block, kv, slot;
        if (block_first) {
            slot = (int)(task%groups);
            int64_t tile = task/groups;
            block = (int)(tile%c->blocks); kv = (int)(tile/c->blocks);
        } else {
            block = (int)(task%c->blocks);
            int64_t pair = task/c->blocks;
            kv = (int)(pair/groups); slot = (int)(pair%groups);
        }
        int h = kv*GQA_GROUP + (triple ? 3*slot : gqa_head_offset(groups,slot));
        int count = triple ? (slot < 2 ? 3 : 1) : gqa_head_count(groups,slot);
        int begin = block*GQA_BLOCK, n = c->visible-begin;
        if (n > GQA_BLOCK) n = GQA_BLOCK;
        int64_t offset = ((int64_t)kv*c->stride+begin)*GQA_DIM;
        float *s0 = c->scores+(int64_t)h*c->visible+begin;
        if (values) {
            float *v0 = c->partial+((int64_t)h*c->blocks+block)*GQA_DIM;
            if (count == 4) {
                const float *scores[4];
                for (int j = 0; j < 4; j++) scores[j] = s0+(int64_t)j*c->visible;
                gqa_values_four(v0,c->blocks*GQA_DIM,scores,c->v+offset,n);
            } else if (count == 2) gqa_values_pair(v0,v0+(int64_t)c->blocks*GQA_DIM,s0,s0+c->visible,c->v+offset,n);
            else moty_hw_attn_accum(v0,s0,c->v+offset,n,GQA_DIM);
        } else {
            const float *q0 = c->q+(int64_t)h*GQA_DIM;
            if (c->kv4) {
                int64_t stride = ((int64_t)c->stride+3)&~INT64_C(3);
                const float *key = c->k+(kv*stride+begin)*GQA_DIM;
                int j = 0;
                for (; j+1 < count; j += 2)
                    att_scores_kv4_pair(s0+(int64_t)j*c->visible,s0+(int64_t)(j+1)*c->visible,
                                         q0+j*GQA_DIM,q0+(j+1)*GQA_DIM,key,n,n,c->scale);
                if (j < count) att_scores_kv4(s0+(int64_t)j*c->visible,q0+j*GQA_DIM,key,n,c->scale);
            } else if (count == 4) {
                float *scores[4];
                for (int j = 0; j < 4; j++) scores[j] = s0+(int64_t)j*c->visible;
                gqa_scores_four(scores,q0,c->k+offset,n,c->scale);
            } else if (count == 3)
                gqa_scores_tile(s0,c->visible,q0,c->k+offset,n,c->scale,3,2);
            else if (count == 2) {
                if (c->options.score_tile == GQA_SCORE_PAIR_WIDE)
                    gqa_scores_tile(s0,c->visible,q0,c->k+offset,n,c->scale,2,4);
                else gqa_scores_pair(s0,s0+c->visible,q0,q0+GQA_DIM,c->k+offset,n,c->scale);
            } else moty_hw_attn_scores(s0,q0,c->k+offset,n,GQA_DIM,c->scale);
        }
    }
}

static void gqa_score_work(void *job, int64_t a, int64_t b, int tid) { gqa_blocks(job,a,b,0,0); }
static void gqa_value_work(void *job, int64_t a, int64_t b, int tid) { gqa_blocks(job,a,b,1,0); }
static void gqa_softmax_work(void *job, int64_t a, int64_t b, int tid) {
    GqaJob *c = job;
    for (int64_t h = a; h < b; h++) softmax_row(c->scores+h*c->visible,c->visible);
}

static void gqa_merge_work(void *job, int64_t a, int64_t b, int tid) {
    GqaJob *c = job;
    for (int64_t h = a; h < b; h++) {
        float *out = c->out+h*GQA_DIM;
        const float *partial = c->partial+h*c->blocks*GQA_DIM;
        if (c->options.merge_wide) {
            for (int d = 0; d < GQA_DIM; d += 16) {
                float32x4_t sum[4];
                for (int i = 0; i < 4; i++) sum[i] = vld1q_f32(partial+d+4*i);
                for (int b = 1; b < c->blocks; b++)
                    for (int i = 0; i < 4; i++)
                        sum[i] = vaddq_f32(sum[i],vld1q_f32(partial+b*GQA_DIM+d+4*i));
                for (int i = 0; i < 4; i++) vst1q_f32(out+d+4*i,sum[i]);
            }
            continue;
        }
        for (int d = 0; d < GQA_DIM; d += 4) {
            float32x4_t sum = vld1q_f32(partial+d);
            for (int b = 1; b < c->blocks; b++) sum = vaddq_f32(sum,vld1q_f32(partial+b*GQA_DIM+d));
            vst1q_f32(out+d,sum);
        }
    }
}

static void gqa_pair_work(void *job, int64_t a, int64_t b, int tid) {
    GqaJob *c = job;
    /* This worker owns both score rows and all value partials for the pair. */
    for (int64_t pair = a; pair < b; pair++) {
        int slot = (int)(pair % c->groups);
        int64_t h = pair / c->groups * GQA_GROUP + gqa_head_offset(c->groups,slot);
        int64_t last = h + gqa_head_count(c->groups,slot);
        int64_t first_block = pair * c->blocks, last_block = first_block + c->blocks;
        gqa_blocks(c, first_block, last_block, 0,1);
        gqa_softmax_work(c, h, last, tid);
        gqa_blocks(c, first_block, last_block, 1,1);
        gqa_merge_work(c, h, last, tid);
    }
}

static int gqa_flag(const char *name, int default_value) {
    const char *value = moty_getenv(name);
    return value ? !strcmp(value,"1") : default_value;
}

static GqaOptions gqa_options(int kv4, int use_twelve_worker_defaults) {
    GqaScoreTile score_tile = use_twelve_worker_defaults ? GQA_SCORE_TRIPLE : GQA_SCORE_PAIR;
    const char *tile = kv4 ? NULL : moty_getenv("MOTY_Q41_GQA_SCORE");
    if (tile) {
        if (!strcmp(tile,"24")) score_tile = GQA_SCORE_PAIR_WIDE;
        else if (!strcmp(tile,"32")) score_tile = GQA_SCORE_TRIPLE;
        else score_tile = GQA_SCORE_PAIR;
    }
    int team_default = use_twelve_worker_defaults;
#ifdef MOTY_THREADPOOL
    team_default = 0;
#endif
    return (GqaOptions){
        .score_tile = score_tile,
        .block_first = gqa_flag("MOTY_Q41_GQA_BLOCK_FIRST",use_twelve_worker_defaults),
        .merge_wide = gqa_flag("MOTY_Q41_GQA_MERGE",use_twelve_worker_defaults),
        .team = gqa_flag("MOTY_Q41_GQA_TEAM",team_default)
    };
}

/* Single-token and lookahead paths select the same four phase kernels. */
typedef struct {
    int groups, pair_owned, value_three, twelve_worker_defaults;
} GqaSchedule;

static GqaSchedule gqa_schedule(int kv_heads, int fp16) {
    GqaSchedule schedule = { .groups = gqa_flag("MOTY_Q41_GQA_FOUR",0) ? 3 : 4 };
    const int threads = moty_par_threads();
    const int value_default = fp16 && threads == 12 && schedule.groups == 4;
    schedule.pair_owned = gqa_flag("MOTY_Q41_GQA_PAIR",fp16 && threads <= kv_heads*schedule.groups);
    schedule.value_three = gqa_flag("MOTY_Q41_GQA_V3",value_default);
#if defined(__APPLE__) && (defined(_OPENMP) || defined(MOTY_THREADPOOL))
    schedule.twelve_worker_defaults = value_default && kv_heads == 2;
#endif
    return schedule;
}

static int gqa_enabled(int fp16) {
    const char *enable = moty_getenv("MOTY_Q41_GQA");
    return enable ? !strcmp(enable,"1") : fp16;
}

static int gqa_block_count(int visible) {
    return (visible-1)/GQA_BLOCK+1;
}

static int64_t gqa_scores_bytes(int heads, int visible) {
    return (int64_t)heads*visible*sizeof(float);
}

static int64_t gqa_partial_bytes(int heads, int blocks) {
    return (int64_t)heads*blocks*GQA_DIM*sizeof(float);
}

typedef enum { GQA_SCORE, GQA_SOFTMAX, GQA_VALUE, GQA_MERGE, GQA_PHASES } GqaPhaseId;
typedef struct {
    moty_par_fn run;
    int64_t tasks;
} GqaPhase;

static GqaPhase gqa_phase(const GqaJob *c, int heads, int kv_heads, int v3, GqaPhaseId phase) {
    switch (phase) {
        case GQA_SCORE: return (GqaPhase){gqa_score_work,(int64_t)kv_heads*c->blocks*gqa_score_groups(c)};
        case GQA_SOFTMAX: return (GqaPhase){gqa_softmax_work,heads};
        case GQA_VALUE: return (GqaPhase){v3 ? gqa_value_three_work : gqa_value_work,
                                   (int64_t)kv_heads*c->blocks*(v3 ? GQA_VALUE_GROUPS : c->groups)};
        default: return (GqaPhase){gqa_merge_work,heads};
    }
}

#ifdef MOTY_THREADPOOL
typedef struct {
    GqaJob *job;
    const GqaPhase *phases;
    int workers;
    _Alignas(64) atomic_int arrived;
} GqaTeam;

static void gqa_phase_worker(void *ctx, int64_t a, int64_t b, int tid) {
    GqaTeam *team = ctx;
    for (int phase = 0; phase < GQA_PHASES; phase++) {
        const GqaPhase *op = &team->phases[phase];
        moty_par_range(op->tasks,tid,team->workers,&a,&b);
        if (a < b) op->run(team->job,a,b,tid);
        if (phase < GQA_MERGE) {
            /* Wait for every worker's writes before starting the next phase. */
            const int target = (phase+1)*team->workers;
            atomic_fetch_add_explicit(&team->arrived,1,memory_order_acq_rel);
            while (atomic_load_explicit(&team->arrived,memory_order_acquire) < target)
                __asm__ __volatile__("yield" ::: "memory");
        }
    }
}
#endif

static void gqa_run_phases(GqaJob *c, int heads, int kv_heads, int v3) {
    GqaPhase phases[GQA_PHASES];
    for (int phase = 0; phase < GQA_PHASES; phase++)
        phases[phase] = gqa_phase(c,heads,kv_heads,v3,phase);
#ifdef _OPENMP
    if (c->options.team && !omp_in_parallel() && omp_get_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num(), workers = omp_get_num_threads();
            for (int phase = 0; phase < GQA_PHASES; phase++) {
                int64_t a, b;
                moty_par_range(phases[phase].tasks,tid,workers,&a,&b);
                if (a < b) phases[phase].run(c,a,b,tid);
                if (phase < GQA_MERGE) {
                    #pragma omp barrier
                }
            }
        }
        return;
    }
#elif defined(MOTY_THREADPOOL)
    if (c->options.team && !moty_par_active()) {
        int workers = moty_par_threads();
        if (workers > 1 && workers <= moty_par_procs()) {
            /* One task per worker so even empty block ranges join the barriers. */
            GqaTeam team = {.job=c,.phases=phases,.workers=workers};
            atomic_init(&team.arrived,0);
            moty_par_for(workers,0,gqa_phase_worker,&team);
            return;
        }
    }
#endif
    for (int phase = 0; phase < GQA_PHASES; phase++)
        moty_par_for(phases[phase].tasks,0,phases[phase].run,c);
}

#endif

/* Consecutive queries share the even prefix of their score sweep and
 * all V positions visible to the first query. Finish each causal tail
 * with the same scalar-row and FP32 FMA kernels as the separate sweep. */
void moty_attn_prefill_group(float *out, int stride, const float *q,
                            float *const *scores, const float *k, const float *v,
                            int n, int count, float scale) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    int even = n & ~1;
    if (count == 4) {
        float query[4*GQA_DIM];
        for (int j = 0; j < 4; j++) memcpy(query+j*GQA_DIM,q+(int64_t)j*stride,GQA_DIM*sizeof(float));
        gqa_scores_four(scores,query,k,even,scale);
    } else gqa_scores_pair(scores[0],scores[1],q,q+stride,k,even,scale);
    for (int j = 0; j < count; j++) {
        moty_hw_attn_scores(scores[j]+even,q+(int64_t)j*stride,k+(int64_t)even*GQA_DIM,
                            n+j-even,GQA_DIM,scale);
        softmax_row(scores[j],n+j);
    }
    if (count == 4) {
        const float *rows[] = { scores[0], scores[1], scores[2], scores[3] };
        gqa_values_four(out,stride,rows,v,n);
    } else gqa_values_pair(out,out+stride,scores[0],scores[1],v,n);
    for (int j = 1; j < count; j++)
        for (int t = n; t < n+j; t++)
            moty_hw_axpy(out+(int64_t)j*stride,scores[j][t],v+(int64_t)t*GQA_DIM,GQA_DIM);
#else
    for (int j = 0; j < count; j++) {
        moty_hw_attn_scores(scores[j],q+(int64_t)j*stride,k,n+j,64,scale);
        softmax_row(scores[j],n+j);
        moty_hw_attn_accum(out+(int64_t)j*stride,scores[j],v,n+j,64);
    }
#endif
}

int moty_attn_gqa64_layout(float *out, const float *q, const float *k, const float *v,
                    int kv_heads, int kv_stride, int visible, float scale, Scratch *scratch, int kv4) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (visible < GQA_MIN_VISIBLE) return 0;
    const int fp16 = moty_q41_uses_fp16();
    if (!gqa_enabled(fp16)) return 0;
    const GqaSchedule schedule = gqa_schedule(kv_heads,fp16);
    int heads = kv_heads*GQA_GROUP, blocks = gqa_block_count(visible);
    int64_t scores = gqa_scores_bytes(heads,visible);
    int64_t partial = gqa_partial_bytes(heads,blocks);
    scr_reset(scratch);
    scr_reserve(scratch,scr_al(scores)+scr_al(partial));
    GqaJob c = { .out = out, .scores = scr_take(scratch,scores), .partial = scr_take(scratch,partial),
                 .q = q, .k = k, .v = v, .stride = kv_stride, .visible = visible,
                 .blocks = blocks, .groups = schedule.groups, .kv4 = kv4, .scale = scale };
    if (schedule.pair_owned) {
        moty_par_for((int64_t)kv_heads*schedule.groups,0,gqa_pair_work,&c);
        return 1;
    }
    c.options = gqa_options(kv4,!kv4 && schedule.twelve_worker_defaults);
    gqa_run_phases(&c,heads,kv_heads,schedule.value_three);
    return 1;
#else
    return 0;
#endif
}

#if defined(__aarch64__) && defined(__ARM_NEON)
enum { GQA_BATCH_MAX = 5 };
typedef struct {
    GqaJob row[GQA_BATCH_MAX];
    int n, heads, kv_heads, v3;
} GqaBatch;

/* Each position owns its own score and partial buffers. Flatten only
 * the worker ranges; each attention phase keeps its original kernels. */
static void gqa_batch_phase(GqaBatch *batch, int64_t a, int64_t b, int tid, GqaPhaseId phase) {
    for (int i = 0; i < batch->n && b > 0; i++) {
        GqaJob *c = &batch->row[i];
        GqaPhase work = gqa_phase(c,batch->heads,batch->kv_heads,batch->v3,phase);
        int64_t jobs = work.tasks;
        if (a < jobs) work.run(c,a > 0 ? a : 0,b < jobs ? b : jobs,tid);
        a -= jobs; b -= jobs;
    }
}
static void gqa_batch_scores(void *p, int64_t a, int64_t b, int tid) { gqa_batch_phase(p,a,b,tid,GQA_SCORE); }
static void gqa_batch_softmax(void *p, int64_t a, int64_t b, int tid) { gqa_batch_phase(p,a,b,tid,GQA_SOFTMAX); }
static void gqa_batch_values(void *p, int64_t a, int64_t b, int tid) { gqa_batch_phase(p,a,b,tid,GQA_VALUE); }
static void gqa_batch_merge(void *p, int64_t a, int64_t b, int tid) { gqa_batch_phase(p,a,b,tid,GQA_MERGE); }
#endif

int moty_attn_gqa64_decode_batch(float *out, const float *q, const float *k, const float *v,
                                int kv_heads, int stride, int visible, int n, float scale, Scratch *scratch) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (visible < GQA_MIN_VISIBLE || n < 2 || n > GQA_BATCH_MAX || visible > stride-n+1) return 0;
    const int fp16 = moty_q41_uses_fp16();
    if (!gqa_enabled(fp16)) return 0;
    const GqaSchedule schedule = gqa_schedule(kv_heads,fp16);
    if (schedule.pair_owned) return 0;
    GqaBatch batch = { .n = n, .heads = kv_heads*GQA_GROUP, .kv_heads = kv_heads,
                        .v3 = schedule.value_three };
    const GqaOptions options = gqa_options(0,schedule.twelve_worker_defaults);
    int64_t bytes = 0, tasks[GQA_PHASES] = {0};
    for (int i = 0; i < n; i++) {
        int positions = visible+i, blocks = gqa_block_count(positions);
        bytes += scr_al(gqa_scores_bytes(batch.heads,positions));
        bytes += scr_al(gqa_partial_bytes(batch.heads,blocks));
    }
    scr_reset(scratch);
    scr_reserve(scratch,bytes);
    for (int i = 0; i < n; i++) {
        int positions = visible+i, blocks = gqa_block_count(positions);
        int64_t offset = (int64_t)i*batch.heads*GQA_DIM;
        float *scores = scr_take(scratch,gqa_scores_bytes(batch.heads,positions));
        float *partial = scr_take(scratch,gqa_partial_bytes(batch.heads,blocks));
        batch.row[i] = (GqaJob){ .out = out+offset, .scores = scores, .partial = partial,
                                .q = q+offset, .k = k, .v = v, .stride = stride, .visible = positions,
                                .blocks = blocks, .groups = schedule.groups, .scale = scale, .options = options };
        for (int phase = 0; phase < GQA_PHASES; phase++)
            tasks[phase] += gqa_phase(&batch.row[i],batch.heads,kv_heads,batch.v3,phase).tasks;
    }
    const moty_par_fn run[] = {gqa_batch_scores,gqa_batch_softmax,gqa_batch_values,gqa_batch_merge};
    for (int phase = 0; phase < GQA_PHASES; phase++) moty_par_for(tasks[phase],0,run[phase],&batch);
    return 1;
#else
    return 0;
#endif
}

int moty_attn_gqa64(float *out, const float *q, const float *k, const float *v,
                    int kv_heads, int kv_stride, int visible, float scale, Scratch *scratch) {
    return moty_attn_gqa64_layout(out,q,k,v,kv_heads,kv_stride,visible,scale,scratch,0);
}

void moty_att_scores_kv4(float *scores, const float *q, const float *k, int n, float scale) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    att_scores_kv4(scores,q,k,n,scale);
#else
    moty_fail_code(MOTY_FAIL_FORMAT,"KV4 requires NEON");
#endif
}

void moty_att_scores_kv4_pair(float *s0, float *s1, const float *q0, const float *q1,
                              const float *k, int n, float scale) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    att_scores_kv4_pair(s0,s1,q0,q1,k,n,n+1,scale);
#else
    moty_fail_code(MOTY_FAIL_FORMAT,"KV4 requires NEON");
#endif
}

void moty_att_scores_f32(float *sc, const float *qv, const float *K,
                                  int64_t kvbase, int t0, int qpos, int hd, float scale) {
    moty_hw_attn_scores(sc, qv, K + (kvbase + t0)*hd, qpos - t0 + 1, hd, scale);
}

void moty_att_scores_i8(float *sc, const float *qv, const int8_t *K8,
                                 const float *Ks, int64_t kvbase, int t0, int qpos,
                                 int hd, float scale) {
    for (int t = t0; t <= qpos; t++) {
        int64_t slot = kvbase + t;
        sc[t-t0] = Ks[slot] * dot_f32i8(qv, K8 + slot*hd, hd) * scale;
    }
}

void moty_att_accum_f32(float *cx, const float *sc, const float *V,
                                 int64_t kvbase, int t0, int qpos, int hd) {
    moty_hw_attn_accum(cx, sc, V + (kvbase + t0)*hd, qpos - t0 + 1, hd);
}

void moty_att_accum_i8(float *cx, const float *sc, const int8_t *V8,
                                const float *Vs, int64_t kvbase, int t0, int qpos, int hd) {
    for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
    for (int t = t0; t <= qpos; t++) {
        int64_t slot = kvbase + t;
        const int8_t *vr = V8 + slot*hd;
        float a = sc[t-t0] * Vs[slot];
        for (int dd = 0; dd < hd; dd++) cx[dd] += a * (float)vr[dd];
    }
}
