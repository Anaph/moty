/* attn.c — M3 libmoty-nn: implementazione condivisa dell'attenzione GQA.
 * Trasformata 1:1 da nn_attn.h (paste-in) alla MotyAttnView: stesse
 * regioni OpenMP, stesso ordine delle operazioni, stesse scelte VNNI. */
#include "nn/attn.h"

/* coda comune: QK-norm + RoPE → KV store → scores/accum → (gate) → o_proj */
static void attn_tail(const MotyAttnView *a, float *q, float *k, float *vv,
                      const float *x, int S, int pos_base, float *out, const float *gate) {
    int H = a->n_heads, KV = a->n_kv_heads, hd = a->head_dim, G = H/KV;
    int64_t qw = (int64_t)H*hd, kw = (int64_t)KV*hd;
    int li = a->li;
    /* QK-norm + RoPE */
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) {
            rmsnorm_row(q + s*qw + hh*hd, q + s*qw + hh*hd, a->qn, hd, a->eps);
            rope_head(q + s*qw + hh*hd, pos, a->theta, a->rot);
        }
        for (int hh = 0; hh < KV; hh++) {
            rmsnorm_row(k + s*kw + hh*hd, k + s*kw + hh*hd, a->kn, hd, a->eps);
            rope_head(k + s*kw + hh*hd, pos, a->theta, a->rot);
        }
    }
    /* KV store */
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
    /* scores + accumulation */
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = scr_take(a->scr, (int64_t)S*qw*4);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) for (int s = 0; s < S; s++) {
        float *sc = a->att_sc + (int64_t)omp_get_thread_num()*a->max_t;
        int kvh = hh / G, qpos = pos_base + s;
        const float *qv = q + s*qw + (int64_t)hh*hd;
        int64_t kvbase = (int64_t)kvh * a->max_t;
        if (kv8) att_scores_i8(sc, qv, a->K8[li], a->Ks[li], kvbase, 0, qpos, hd, scale);
        else     att_scores_f32(sc, qv, a->K[li], kvbase, 0, qpos, hd, scale);
        softmax_row(sc, qpos+1);
        float *cx = ctx + s*qw + (int64_t)hh*hd;
        if (kv8) att_accum_i8(cx, sc, a->V8[li], a->Vs[li], kvbase, 0, qpos, hd);
        else     att_accum_f32(cx, sc, a->V[li], kvbase, 0, qpos, hd);
    }
    if (gate)
        for (int64_t i = 0; i < (int64_t)S*qw; i++) ctx[i] *= 1.f/(1.f + expf(-gate[i]));
    mat_apply(out, ctx, a->o, S);
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
    {
        int64_t nk = (int64_t)S*kw;
        int64_t tot = (int64_t)S*qw + 2*nk;
        int gs = a->q->gs, ng = (D+gs-1)/gs, rb = (D+1)/2;
        if (vnni_all) {
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
            #pragma omp parallel for schedule(static)
            for (int64_t r = 0; r < tot; r++) {
                int mi, o; int64_t s;
                if (r < (int64_t)S*qw)       { mi = 0; s = r / qw;       o = (int)(r - s*qw); }
                else if (r < (int64_t)S*qw+nk){ mi = 1; s = (r-S*qw)/kw;  o = (int)(r-S*qw - s*kw); }
                else                          { mi = 2; s = (r-S*qw-nk)/kw;o = (int)(r-S*qw-nk - s*kw); }
                const Mat *w = mi == 0 ? a->q : mi == 1 ? a->k : a->v;
                float *dst = mi == 0 ? q : mi == 1 ? k : vv;
                dst[(int64_t)s*(mi == 0 ? qw : kw) + o] =
                    asx[s] * dot_i4g8p(w->q4 + (int64_t)o*rb, w->qs + (int64_t)o*ng,
                                       axi + (int64_t)s*D, axg + (int64_t)s*ng, D);
            }
        } else {
            mat_apply(q, x, a->q, S);
            mat_apply(k, x, a->k, S);
            mat_apply(vv, x, a->v, S);
        }
    }
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
