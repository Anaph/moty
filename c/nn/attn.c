/* attn.c — M3 libmoty-nn: implementazione condivisa dell'attenzione GQA.
 * Trasformata 1:1 da nn_attn.h (paste-in) alla MotyAttnView: stesse
 * regioni OpenMP, stesso ordine delle operazioni, stesse scelte VNNI. */
#include "util/prof.h"
#include "nn/attn.h"

/* scores + accumulation for (head, token) pairs [r0, r1), r = head*S + token */
typedef struct {
    const MotyAttnView *a; const float *q; float *ctx; int S, G, hd, li, pos_base, kv8; int64_t qw; float scale;
} AttJob;
static void att_part(void *c_, int64_t r0, int64_t r1, int tid) {
    const AttJob *c = c_; const MotyAttnView *a = c->a; int hd = c->hd, li = c->li;
    float *sc = a->att_sc + (int64_t)tid*a->max_t;       /* per-thread score row */
    for (int64_t r = r0; r < r1; r++) {
        int hh = (int)(r / c->S), s = (int)(r % c->S);
        int kvh = hh / c->G, qpos = c->pos_base + s;
        const float *qv = c->q + s*c->qw + (int64_t)hh*hd;
        int64_t kvbase = (int64_t)kvh * a->max_t;
        if (c->kv8) att_scores_i8(sc, qv, a->K8[li], a->Ks[li], kvbase, 0, qpos, hd, c->scale);
        else        att_scores_f32(sc, qv, a->K[li], kvbase, 0, qpos, hd, c->scale);
        softmax_row(sc, qpos+1);
        float *cx = c->ctx + s*c->qw + (int64_t)hh*hd;
        if (c->kv8) att_accum_i8(cx, sc, a->V8[li], a->Vs[li], kvbase, 0, qpos, hd);
        else        att_accum_f32(cx, sc, a->V[li], kvbase, 0, qpos, hd);
    }
}

/* coda comune: QK-norm + RoPE → KV store → scores/accum → (gate) → o_proj */
static void attn_tail(const MotyAttnView *a, float *q, float *k, float *vv,
                      const float *x, int S, int pos_base, float *out, const float *gate) {
    int H = a->n_heads, KV = a->n_kv_heads, hd = a->head_dim, G = H/KV;
    int64_t qw = (int64_t)H*hd, kw = (int64_t)KV*hd;
    int li = a->li;
    /* QK-norm + RoPE */
    OP_T(t_qk);
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) {
            if (a->qn) rmsnorm_row(q + s*qw + hh*hd, q + s*qw + hh*hd, a->qn, hd, a->eps);
            rope_head(q + s*qw + hh*hd, pos, a->theta, a->rot);
        }
        for (int hh = 0; hh < KV; hh++) {
            if (a->kn) rmsnorm_row(k + s*kw + hh*hd, k + s*kw + hh*hd, a->kn, hd, a->eps);
            rope_head(k + s*kw + hh*hd, pos, a->theta, a->rot);
        }
    }
    OP_ACC(OP_QKNORM_ROPE, t_qk);
    /* KV store */
    OP_T(t_kv);
    int kv8 = a->K8[li] != NULL;
    for (int s = 0; s < S; s++) for (int hh = 0; hh < KV; hh++) {
        int t = pos_base + s; int64_t slot = (int64_t)hh*a->max_t + t;
        if (kv8) {
            kv_store_row(a->K8[li] + slot*hd, &a->Ks[li][slot], k + s*kw + hh*hd, hd);
            kv_store_row(a->V8[li] + slot*hd, &a->Vs[li][slot], vv + s*kw + hh*hd, hd);
        } else {
            memcpy(a->K[li] + slot*hd, k + s*kw + hh*hd, hd*sizeof(float));
            memcpy(a->V[li] + slot*hd, vv + s*kw + hh*hd, hd*sizeof(float));
        }
    }
    OP_ACC(OP_KV_STORE, t_kv);
    /* scores + accumulation */
    OP_T(t_at);
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = scr_take(a->scr, (int64_t)S*qw*4);
    AttJob aj = { a, q, ctx, S, G, hd, li, pos_base, kv8, qw, scale };
    moty_par_for((int64_t)H*S, 0, att_part, &aj);
    if (gate)
        for (int64_t i = 0; i < (int64_t)S*qw; i++) ctx[i] *= 1.f/(1.f + expf(-gate[i]));
    OP_ACC(OP_ATTN_CORE, t_at);
    OP_T(t_o);
    mat_apply(out, ctx, a->o, S);
    OP_ACC(OP_O_PROJ, t_o);
}

/* q/k/v rows of one region on the VNNI path, r over [q rows | k rows | v rows] */
typedef struct {
    const MotyAttnView *a; float *q, *k, *vv; const int8_t *axi; const int32_t *axg; const float *asx;
    int S, D; int64_t qw, kw, nk; int ng, rb;
} QkvJob;
static void qkv_vnni_part(void *c_, int64_t r0, int64_t r1, int tid) {
    const QkvJob *c = c_; const MotyAttnView *a = c->a; int S = c->S, D = c->D, ng = c->ng, rb = c->rb;
    int64_t qw = c->qw, kw = c->kw, nk = c->nk;
    for (int64_t r = r0; r < r1; r++) {
        int mi, o; int64_t s;
        if (r < (int64_t)S*qw)       { mi = 0; s = r / qw;       o = (int)(r - s*qw); }
        else if (r < (int64_t)S*qw+nk){ mi = 1; s = (r-S*qw)/kw;  o = (int)(r-S*qw - s*kw); }
        else                          { mi = 2; s = (r-S*qw-nk)/kw;o = (int)(r-S*qw-nk - s*kw); }
        const Mat *w = mi == 0 ? a->q : mi == 1 ? a->k : a->v;
        float *dst = mi == 0 ? c->q : mi == 1 ? c->k : c->vv;
        dst[(int64_t)s*(mi == 0 ? qw : kw) + o] =
            c->asx[s] * dot_i4g8p(w->q4 + (int64_t)o*rb, w->qs + (int64_t)o*ng,
                                  c->axi + (int64_t)s*D, c->axg + (int64_t)s*ng, D);
    }
}

void moty_nn_attention(const MotyAttnView *a, const float *x, int S, int pos_base, float *out) {
    int H = a->n_heads, KV = a->n_kv_heads, hd = a->head_dim;
    int64_t qw = (int64_t)H*hd, kw = (int64_t)KV*hd;
    int D = a->q->I;                     /* input dim = hidden */
    /* P5: arena per-view. Reserve PRIMA dei take: nessun take muove la base */
    int vnni_all = (a->q->fmt == WF_I4G && a->k->fmt == WF_I4G && a->v->fmt == WF_I4G
                    && a->q->gs == 32 && (D & 63) == 0);
    {
        int64_t aux = vnni_all
            ? scr_al((int64_t)S*D) + scr_al((int64_t)S*((D+31)/32)*4) + scr_al((int64_t)S*4)
            : 0;
        scr_reset(a->scr);
        scr_reserve(a->scr, scr_al((int64_t)S*qw*4) + 2*scr_al((int64_t)S*kw*4) + aux);
    }
    float *q = scr_take(a->scr, (int64_t)S*qw*4);
    float *k = scr_take(a->scr, (int64_t)S*kw*4), *vv = scr_take(a->scr, (int64_t)S*kw*4);
    /* q/k/v in UNA regione quando tutte WF_I4G gs=32 D%64==0 (VNNI) */
    OP_T(t_qkv);
    {
        int64_t nk = (int64_t)S*kw;
        int64_t tot = (int64_t)S*qw + 2*nk;
        int gs = a->q->gs, ng = (D+gs-1)/gs, rb = (D+1)/2;
        if (a->qkv) {                  /* fused rows [q|k|v]: one GEMV, then split per token */
            float *qkvb = falloc((int64_t)S*(qw + 2*kw));
            mat_apply(qkvb, x, a->qkv, S);
            for (int s = 0; s < S; s++) {
                const float *r = qkvb + (int64_t)s*(qw + 2*kw);
                memcpy(q + (int64_t)s*qw, r, qw*sizeof(float));
                memcpy(k + (int64_t)s*kw, r + qw, kw*sizeof(float));
                memcpy(vv + (int64_t)s*kw, r + qw + kw, kw*sizeof(float));
            }
            free(qkvb);
        } else if (vnni_all) {
            int8_t *axi = scr_take(a->scr, scr_al((int64_t)S*D));
            int32_t *axg = scr_take(a->scr, scr_al((int64_t)S*ng*4));
            float   *asx = scr_take(a->scr, scr_al((int64_t)S*4));
            for (int s = 0; s < S; s++) {
                asx[s] = qrow_i8(x + (int64_t)s*D, axi + (int64_t)s*D, D);
                for (int g = 0; g < ng; g++) {
                    int32_t acc = 0;
                    for (int j = 0; j < 32; j++) acc += axi[(int64_t)s*D + g*32+j];
                    axg[(int64_t)s*ng + g] = acc;
                }
            }
            QkvJob qj = { a, q, k, vv, axi, axg, asx, S, D, qw, kw, nk, ng, rb };
            moty_par_for(tot, 0, qkv_vnni_part, &qj);
        } else {
            mat_apply(q, x, a->q, S);
            mat_apply(k, x, a->k, S);
            mat_apply(vv, x, a->v, S);
        }
    }
    OP_ACC(OP_QKV, t_qkv);
    attn_tail(a, q, k, vv, x, S, pos_base, out, NULL);
}

void moty_nn_attention_gated(const MotyAttnView *a, const float *x, int S, int pos_base, float *out) {
    int H = a->n_heads, KV = a->n_kv_heads, hd = a->head_dim;
    int64_t qw = (int64_t)H*hd, kw = (int64_t)KV*hd;
    int D = a->q->I;
    int vnni_all = (a->q->fmt == WF_I4G && a->k->fmt == WF_I4G && a->v->fmt == WF_I4G
                    && a->q->gs == 32 && (D & 63) == 0);
    (void)vnni_all;   /* il layout [q|gate] interleaved impedisce la fusione */
    scr_reset(a->scr);
    scr_reserve(a->scr, scr_al((int64_t)S*qw*4) + 2*scr_al((int64_t)S*kw*4));
    float *q = scr_take(a->scr, (int64_t)S*qw*4);
    float *k = scr_take(a->scr, (int64_t)S*kw*4), *vv = scr_take(a->scr, (int64_t)S*kw*4);
    float *gate = NULL;
    {   /* q_proj raddoppiata [query|gate] interleaved per head: sequenziale */
        float *qg = falloc((int64_t)S*2*qw);
        mat_apply(qg, x, a->q, S);
        gate = falloc((int64_t)S*qw);
        for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) {
            memcpy(q    + (int64_t)s*qw + (int64_t)hh*hd, qg + (int64_t)s*2*qw + (int64_t)hh*2*hd,      hd*sizeof(float));
            memcpy(gate + (int64_t)s*qw + (int64_t)hh*hd, qg + (int64_t)s*2*qw + (int64_t)hh*2*hd + hd, hd*sizeof(float));
        }
        free(qg);
    }
    mat_apply(k, x, a->k, S);
    mat_apply(vv, x, a->v, S);
    attn_tail(a, q, k, vv, x, S, pos_base, out, gate);
    free(gate);
}
