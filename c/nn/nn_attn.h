/* Shared GQA attention with QK-norm + RoPE. Include AFTER Layer/Cfg/Model are
 * defined and after nn_rope.h.
 *
 * Config macros (define before including):
 *   ENGINE_GATED_ATTN  — q_proj doubled [query|gate], output *= sigmoid(gate)
 *   ATTN_NORM          — field name for pre-attention norm (default: in_ln)
 *
 * Structure:
 *   - NON-gated + all mats WF_I4G(gs=32, D%64==0): q/k/v projections fused
 *     in ONE parallel region via dot_i4g8p (VNNI). 1 ramp instead of 3.
 *   - Gated (ENGINE_GATED_ATTN): sequential projections (q doubled) — the
 *     interleaved [q|gate] layout prevents the fused region; correctness
 *     first, the gated engines have few full-attn layers.
 *   - o_proj always separate (different input: ctx).
 *
 * Requires Layer fields: q,k,v,o, qn,kn
 * Requires Cfg fields: n_heads, n_kv_heads, head_dim, theta, rot, eps
 */
#ifndef NN_ATTN_H
#define NN_ATTN_H

#ifndef ATTN_NORM
#define ATTN_NORM in_ln
#endif

static void attention(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c;
    int H = c->n_heads, KV = c->n_kv_heads, hd = c->head_dim, G = H/KV;
    int64_t qw = (int64_t)H*hd, kw = (int64_t)KV*hd;
    /* P5: arena per-Model. Reserve PRIMA dei take: nessun take muove la base */
    int vnni_all = (l->q.fmt == WF_I4G && l->k.fmt == WF_I4G && l->v.fmt == WF_I4G
                    && l->q.gs == 32 && (c->hidden & 63) == 0);
    {
        int64_t aux = vnni_all
            ? scr_al((int64_t)S*c->hidden) + scr_al((int64_t)S*((c->hidden+31)/32)*4) + scr_al((int64_t)S*4)
            : 0;
        scr_reset(&m->base.scr);
        scr_reserve(&m->base.scr, scr_al((int64_t)S*qw*4) + 2*scr_al((int64_t)S*kw*4) + aux);
    }
    float *q = scr_take(&m->base.scr, (int64_t)S*qw*4);
    float *k = scr_take(&m->base.scr, (int64_t)S*kw*4), *vv = scr_take(&m->base.scr, (int64_t)S*kw*4);
#ifdef ENGINE_GATED_ATTN
    float *gate = NULL;
    {   /* q_proj raddoppiata [query|gate] interleaved per head: sequenziale */
        float *qg = falloc((int64_t)S*2*qw);
        mat_apply(qg, x, &l->q, S);
        gate = falloc((int64_t)S*qw);
        for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) {
            memcpy(q    + (int64_t)s*qw + (int64_t)hh*hd, qg + (int64_t)s*2*qw + (int64_t)hh*2*hd,      hd*sizeof(float));
            memcpy(gate + (int64_t)s*qw + (int64_t)hh*hd, qg + (int64_t)s*2*qw + (int64_t)hh*2*hd + hd, hd*sizeof(float));
        }
        free(qg);
    }
    mat_apply(k, x, &l->k, S);
    mat_apply(vv, x, &l->v, S);
#else
    /* q/k/v in UNA regione quando tutte WF_I4G gs=32 D%64==0 (VNNI) */
    {
        int64_t nk = (int64_t)S*kw;
        int64_t tot = (int64_t)S*qw + 2*nk;
        int D = c->hidden, gs = l->q.gs, ng = (D+gs-1)/gs, rb = (D+1)/2;
        int vnni = (l->q.fmt == WF_I4G && l->k.fmt == WF_I4G && l->v.fmt == WF_I4G
                    && gs == 32 && (D & 63) == 0);
        if (vnni) {
            int8_t *axi = scr_take(&m->base.scr, scr_al((int64_t)S*D));
            int32_t *axg = scr_take(&m->base.scr, scr_al((int64_t)S*ng*4));
            float   *asx = scr_take(&m->base.scr, scr_al((int64_t)S*4));
            for (int s = 0; s < S; s++) {
                asx[s] = qrow_i8(x + (int64_t)s*D, axi + (int64_t)s*D, D);
                for (int g = 0; g < ng; g++) {
                    int32_t a = 0;
                    for (int j = 0; j < 32; j++) a += axi[(int64_t)s*D + g*32+j];
                    axg[(int64_t)s*ng + g] = a;
                }
            }
            #pragma omp parallel for schedule(static)
            for (int64_t r = 0; r < tot; r++) {
                int mi, o; int64_t s;
                if (r < (int64_t)S*qw)       { mi = 0; s = r / qw;       o = (int)(r - s*qw); }
                else if (r < (int64_t)S*qw+nk){ mi = 1; s = (r-S*qw)/kw;  o = (int)(r-S*qw - s*kw); }
                else                          { mi = 2; s = (r-S*qw-nk)/kw;o = (int)(r-S*qw-nk - s*kw); }
                Mat *w = mi == 0 ? &l->q : mi == 1 ? &l->k : &l->v;
                float *dst = mi == 0 ? q : mi == 1 ? k : vv;
                dst[(int64_t)s*(mi == 0 ? qw : kw) + o] =
                    asx[s] * dot_i4g8p(w->q4 + (int64_t)o*rb, w->qs + (int64_t)o*ng,
                                       axi + (int64_t)s*D, axg + (int64_t)s*ng, D);
            }
        } else {
            mat_apply(q, x, &l->q, S);
            mat_apply(k, x, &l->k, S);
            mat_apply(vv, x, &l->v, S);
        }
    }
#endif
    /* QK-norm + RoPE */
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) {
            rmsnorm_row(q + s*qw + hh*hd, q + s*qw + hh*hd, l->qn, hd, c->eps);
            rope_head(q + s*qw + hh*hd, pos, c->theta, c->rot);
        }
        for (int hh = 0; hh < KV; hh++) {
            rmsnorm_row(k + s*kw + hh*hd, k + s*kw + hh*hd, l->kn, hd, c->eps);
            rope_head(k + s*kw + hh*hd, pos, c->theta, c->rot);
        }
    }
    /* KV store */
    int kv8 = m->base.K8[li] != NULL;
    for (int s = 0; s < S; s++) for (int hh = 0; hh < KV; hh++) {
        int t = pos_base + s; int64_t slot = (int64_t)hh*m->base.max_t + t;
        if (kv8) {
            kv_store_row(m->base.K8[li] + slot*hd, &m->base.Ks[li][slot], k + s*kw + hh*hd, hd);
            kv_store_row(m->base.V8[li] + slot*hd, &m->base.Vs[li][slot], vv + s*kw + hh*hd, hd);
        } else {
            memcpy(m->base.K[li] + slot*hd, k + s*kw + hh*hd, hd*sizeof(float));
            memcpy(m->base.V[li] + slot*hd, vv + s*kw + hh*hd, hd*sizeof(float));
        }
    }
    /* scores + accumulation */
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = scr_take(&m->base.scr, (int64_t)S*qw*4);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) for (int s = 0; s < S; s++) {
        float *sc = m->base.att_sc + (int64_t)omp_get_thread_num()*m->base.max_t;
        int kvh = hh / G, qpos = pos_base + s;
        const float *qv = q + s*qw + (int64_t)hh*hd;
        int64_t kvbase = (int64_t)kvh * m->base.max_t;
        if (kv8) att_scores_i8(sc, qv, m->base.K8[li], m->base.Ks[li], kvbase, 0, qpos, hd, scale);
        else     att_scores_f32(sc, qv, m->base.K[li], kvbase, 0, qpos, hd, scale);
        softmax_row(sc, qpos+1);
        float *cx = ctx + s*qw + (int64_t)hh*hd;
        if (kv8) att_accum_i8(cx, sc, m->base.V8[li], m->base.Vs[li], kvbase, 0, qpos, hd);
        else     att_accum_f32(cx, sc, m->base.V[li], kvbase, 0, qpos, hd);
    }
#ifdef ENGINE_GATED_ATTN
    for (int64_t i = 0; i < (int64_t)S*qw; i++) ctx[i] *= 1.f/(1.f + expf(-gate[i]));
    free(gate);
#endif
    mat_apply(out, ctx, &l->o, S);
}

#undef ATTN_NORM
#endif /* NN_ATTN_H */
