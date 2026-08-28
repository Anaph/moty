/* Shared MoE dispatch (2 fork/join: regione gate+up di tutti gli expert,
 * poi tutte le righe down; x quantizzato UNA volta). Include AFTER
 * Layer/Cfg/Model, dopo moe.h.
 *
 * Config macros (prima dell'include):
 *   MOE_LOAD_EXPERT fn      — hook caricamento expert (default ldm_layer_load_expert)
 *   MOE_GATE_SIGMOID        — gating sigmoid + expert_bias (LFM2; default)
 *                             altrimenti softmax top-K pesata (Qwen)
 *   MOE_SHARED_EXPERT       — Layer ha sh_gate/sh_up/sh_down/sh_router_gate
 *                             + Cfg.sh_inter: shared expert gated sigmoid,
 *                             sommato PRIMA dei routed (default off)
 *
 * Requires Layer: router (Mat), ec (ExpertCache*) [, expert_bias se SIGMOID]
 * Requires Cfg: hidden, n_experts, topk, moe_inter [, sh_inter se SHARED]
 * Requires: g_ebits (int globale)
 */
#ifndef NN_MOE_SIGMOID_H
#define NN_MOE_SIGMOID_H

#ifndef MOE_LOAD_EXPERT
#define MOE_LOAD_EXPERT ldm_layer_load_expert
#endif

/* S=1 fast path: 2 fork/join TOTAL (vs 3×K). Region 1 = tutte le righe
 * gate+up di tutti gli expert (righe indipendenti); region 2 = tutte le
 * righe down. x quantizzato UNA volta (contratto dot_i8i8: [-127,127]). */
static void moe_decode1(Model *m, Layer *l, int li, const float *x, float *out) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->moe_inter;
#ifdef MOTY_PROF
    static double _r1=0,_r2=0,_ser=0,_setup=0,_tail=0; static int _nn=0;
    double _t0e = 0, _t0 = 0;
#else
# define _on 0
    double _t0e = 0, _t0 = 0; (void)_t0e; (void)_t0;
#endif
    /* P5: arena per-Model (tutte le size sono costanti del modello) */
    int Sh_ = 0;
#ifdef MOE_SHARED_EXPERT
    Sh_ = c->sh_inter;
#endif
    scr_reset(&m->base.scr);
    scr_reserve(&m->base.scr, scr_al((int64_t)E*4) + 2*scr_al((int64_t)K*I*4) + scr_al((int64_t)K*D*4)
                      + 2*scr_al(D) + 2*scr_al((int64_t)K*I) + 2*scr_al((int64_t)Sh_*4));
    float *lb = scr_take(&m->base.scr, (int64_t)E*4);
    float *gb = scr_take(&m->base.scr, (int64_t)K*I*4), *ub = scr_take(&m->base.scr, (int64_t)K*I*4);
    float *hb = scr_take(&m->base.scr, (int64_t)K*D*4);
    int8_t *xi = scr_take(&m->base.scr, scr_al(D)), *gi = NULL;
#ifdef MOTY_PROF
    static double _sA=0,_sB=0,_sC=0; double _a = now_s();
#endif
    mat_apply(lb, x, &l->router, 1);
#ifdef MOTY_PROF
    _sA += now_s()-_a; _a = now_s();
#endif
    int idx[64]; float w[64];
#ifdef MOE_GATE_SIGMOID
    float probs[256]; for (int e = 0; e < E; e++) probs[e] = 1.f/(1.f+expf(-lb[e]));
    for (int kk = 0; kk < K; kk++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            float v = probs[e] + l->expert_bias[e]; int used = 0;
            for (int j = 0; j < kk; j++) if (idx[j] == e) used = 1;
            if (!used && v > bv) { bv = v; best = e; }
        }
        idx[kk] = best;
    }
    float ws = 0;
    for (int kk = 0; kk < K; kk++) { w[kk] = probs[idx[kk]]; ws += w[kk]; }
    if (ws > 0) for (int kk = 0; kk < K; kk++) w[kk] /= ws;
#else
    /* softmax top-K pesata (Qwen): probabilita' gia' normalizzate */
    softmax_row(lb, E);
    moe_topk(lb, E, K, idx, w, 1);
#endif
    int e4 = (g_ebits <= 4);
#ifdef MOE_SHARED_EXPERT
    {   /* shared expert gated: fuori dalle regioni (pesi ~2MB, mat_apply) */
        int Sh = c->sh_inter;
        float *sgb = scr_take(&m->base.scr, scr_al((int64_t)Sh*4));
        float *sub = scr_take(&m->base.scr, scr_al((int64_t)Sh*4));
        mat_apply(sgb, x, &l->sh_gate, 1); mat_apply(sub, x, &l->sh_up, 1);
        for (int i = 0; i < Sh; i++) { float gv=sgb[i]; sgb[i] = (gv/(1.f+expf(-gv)))*sub[i]; }
        mat_apply(out, sgb, &l->sh_down, 1);
        float shw[1]; mat_apply(shw, x, &l->sh_router_gate, 1);
        float shgate = sigmoidf(shw[0]);
        for (int d = 0; d < D; d++) out[d] *= shgate;
    }
#else
    memset(out, 0, D*sizeof(float));
#endif
    ExpertSlot *es[64];
    for (int kk = 0; kk < K; kk++) es[kk] = expert_get(l->ec, m, li, idx[kk], MOE_LOAD_EXPERT, I, D);
#ifdef MOTY_PROF
    _sB += now_s()-_a; _a = now_s();
#endif
    /* x -> int8 una volta, condiviso da gate/up di tutti gli expert;
     * px: versione permuted per dot_i4i8p (niente permutex nel loop) */
    int8_t *xip = scr_take(&m->base.scr, scr_al(D));
    int8_t *gip = scr_take(&m->base.scr, scr_al((int64_t)K*I));
#ifdef MOTY_PROF
    _sC += now_s()-_a;
#endif
    float sx = qrow_i8(x, xi, D);
    int32_t sxsum = px_sum(xi, D);
    px_permute(xi, xip, D);
    int rbD = (D+1)/2, rbI = (I+1)/2;
#ifdef MOTY_PROF
    _setup += now_s() - _t0e; _t0 = now_s();
#endif
    /* regione 1 */
    #pragma omp parallel for collapse(3) schedule(static)
    for (int kk = 0; kk < K; kk++) for (int up = 0; up < 2; up++) for (int i = 0; i < I; i++) {
        ExpertSlot *e = es[kk];
        float v;
        if (e4) {
            const uint8_t *w4 = up ? (uint8_t*)e->u + (int64_t)i*rbD : (uint8_t*)e->g + (int64_t)i*rbD;
            float sc = up ? e->us[i] : e->gs[i];
            v = sc * sx * (float)dot_i4i8p(w4, xip, sxsum, D);
        } else {
            const int8_t *w8 = up ? e->u + (int64_t)i*D : e->g + (int64_t)i*D;
            float sc = up ? e->us[i] : e->gs[i];
            v = sc * sx * (float)dot_i8i8(w8, xi, D);
        }
        (up ? ub : gb)[(int64_t)kk*I + i] = v;
    }
#ifdef MOTY_PROF
    _r1 += now_s()-_t0; _t0 = now_s();
#endif
    /* silu-mul seriale + quant g per expert */
    gi = scr_take(&m->base.scr, scr_al((int64_t)K*I));
    static float sgb[64]; static int32_t sgb_sum[64];
    /* silu+quant per expert: PARALLELI (era ~250us seriali/call = 5.5ms/tok) */
    #pragma omp parallel for schedule(static)
    for (int kk = 0; kk < K; kk++) {
        float *g = gb + (int64_t)kk*I, *u = ub + (int64_t)kk*I;
        for (int i = 0; i < I; i++) { float v = g[i]; g[i] = (v/(1.f+expf(-v)))*u[i]; }
        sgb[kk] = qrow_i8(g, gi + (int64_t)kk*I, I);
        sgb_sum[kk] = px_sum(gi + (int64_t)kk*I, I);
        px_permute(gi + (int64_t)kk*I, gip + (int64_t)kk*I, I);
    }
    const float *sg = sgb;
#ifdef MOTY_PROF
    _ser += now_s()-_t0; _nn++; static double _pl=0; double _nw=now_s();
    if (_nw-_pl > 2.0) { _pl=_nw; fprintf(stderr,"[M] n=%d setup=%.3f [router=%.3f topk+sigmoid=%.3f eget=%.3f quant=%.3f] r1=%.3f r2=%.3f\n", _nn, _setup*1000/_nn, _sA*1000/_nn, (_setup-_sA-_sB-_sC)*1000/_nn, _sB*1000/_nn, _sC*1000/_nn, _r1*1000/_nn, _r2*1000/_nn);
        fprintf(stderr, "[C] L%d hits=%lld misses=%lld n=%d cap=%d\n", li, (long long)l->ec->hits, (long long)l->ec->misses, l->ec->n, l->ec->cap); }
    _t0 = now_s();
#endif
    /* regione 2: K*D righe down */
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < K*D; r++) {
        int kk = r/D, d = r - kk*D;
        ExpertSlot *e = es[kk];
        const int8_t *xk = gi + (int64_t)kk*I;
        float v;
        if (e4) v = e->ds[d] * sg[kk] * (float)dot_i4i8p((uint8_t*)e->d + (int64_t)d*rbI, gip + (int64_t)kk*I, sgb_sum[kk], I);
        else    v = e->ds[d] * sg[kk] * (float)dot_i8i8(e->d + (int64_t)d*I, xk, I);
        hb[(int64_t)kk*D + d] = v;
    }
#ifdef MOTY_PROF
    _r2 += now_s()-_t0; _t0 = now_s();
#endif
    /* somma pesata (seriale) */
    for (int kk = 0; kk < K; kk++) {
        const float *hh = hb + (int64_t)kk*D;
        for (int d = 0; d < D; d++) out[d] += w[kk] * hh[d];
    }
#ifdef MOTY_PROF
    _tail += now_s()-_t0;
#endif
}

/* S>1 batch path: batch-union over experts */
static void moe_batch(Model *m, Layer *l, int li, const float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->moe_inter;
    float *logits = falloc((int64_t)S*E); mat_apply(logits, x, &l->router, S);
    int *idx = malloc((int64_t)S*K*sizeof(int)); float *rw = malloc((int64_t)S*K*sizeof(float));
    for (int s = 0; s < S; s++) {
        float *lp = logits + (int64_t)s*E;
#ifdef MOE_GATE_SIGMOID
        float probs[256]; for (int e = 0; e < E; e++) probs[e] = 1.f/(1.f+expf(-lp[e]));
        for (int kk = 0; kk < K; kk++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                float v = probs[e] + l->expert_bias[e]; int used = 0;
                for (int j = 0; j < kk; j++) if (idx[(int64_t)s*K+j] == e) used = 1;
                if (!used && v > bv) { bv = v; best = e; }
            }
            idx[(int64_t)s*K+kk] = best; rw[(int64_t)s*K+kk] = probs[best];
        }
        float ws = 0; for (int kk = 0; kk < K; kk++) ws += rw[(int64_t)s*K+kk];
        if (ws > 0) for (int kk = 0; kk < K; kk++) rw[(int64_t)s*K+kk] /= ws;
#else
        softmax_row(lp, E);
        moe_topk(lp, E, K, idx + (int64_t)s*K, rw + (int64_t)s*K, 1);
#endif
    }
    free(logits);
#ifdef MOE_SHARED_EXPERT
    {   /* shared expert batched su S */
        int Sh = c->sh_inter;
        float *sg = falloc((int64_t)S*Sh), *su = falloc((int64_t)S*Sh);
        mat_apply(sg, x, &l->sh_gate, S); mat_apply(su, x, &l->sh_up, S);
        for (int64_t i = 0; i < (int64_t)S*Sh; i++) { float gv=sg[i]; sg[i]=(gv/(1.f+expf(-gv)))*su[i]; }
        mat_apply(out, sg, &l->sh_down, S);
        float *shg = falloc(S); mat_apply(shg, x, &l->sh_router_gate, S);
        for (int s = 0; s < S; s++) { float g = sigmoidf(shg[s]);
            for (int d = 0; d < D; d++) out[(int64_t)s*D+d] *= g; }
        free(sg); free(su); free(shg);
    }
#else
    memset(out, 0, (int64_t)S*D*sizeof(float));
#endif
    int e4 = (g_ebits <= 4);
    int *tlist = malloc(S*sizeof(int));
    float *gw = falloc((int64_t)S*I), *uu = falloc((int64_t)S*I), *hh = falloc((int64_t)S*D), *xsub = falloc((int64_t)S*D);
    for (int eid = 0; eid < E; eid++) {
        int nt = 0;
        for (int s = 0; s < S; s++) for (int kk = 0; kk < K; kk++)
            if (idx[(int64_t)s*K+kk] == eid) { tlist[nt++] = s; break; }
        if (nt == 0) continue;
        for (int t = 0; t < nt; t++) memcpy(xsub + (int64_t)t*D, x + (int64_t)tlist[t]*D, D*sizeof(float));
        ExpertSlot *e = expert_get(l->ec, m, li, eid, MOE_LOAD_EXPERT, I, D);
        if (e4) { matmul_i4_s(gw, xsub, (uint8_t*)e->g, e->gs, nt, D, I);
                  matmul_i4_s(uu, xsub, (uint8_t*)e->u, e->us, nt, D, I); }
        else    { matmul_q_s(gw, xsub, e->g, e->gs, nt, D, I);
                  matmul_q_s(uu, xsub, e->u, e->us, nt, D, I); }
        for (int64_t i = 0; i < (int64_t)nt*I; i++) { float v = gw[i]; gw[i] = (v/(1.f+expf(-v)))*uu[i]; }
        if (e4) matmul_i4_s(hh, gw, (uint8_t*)e->d, e->ds, nt, I, D);
        else    matmul_q_s(hh, gw, e->d, e->ds, nt, I, D);
        for (int t = 0; t < nt; t++) for (int kk = 0; kk < K; kk++)
            if (idx[(int64_t)tlist[t]*K+kk] == eid)
                for (int d = 0; d < D; d++) out[(int64_t)tlist[t]*D+d] += rw[(int64_t)tlist[t]*K+kk] * hh[(int64_t)t*D+d];
    }
    free(tlist); free(gw); free(uu); free(hh); free(xsub); free(idx); free(rw);
}

#undef MOE_LOAD_EXPERT
#endif /* NN_MOE_SIGMOID_H */
