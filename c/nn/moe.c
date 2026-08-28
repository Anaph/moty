/* moe.c — M3 libmoty-nn: MoE decode/batch condivisa.
 * Trasformata da nn_moe_sigmoid.h (paste-in): le macro MOE_GATE_SIGMOID /
 * MOE_SHARED_EXPERT / MOE_LOAD_EXPERT diventano la variante chiamata e i
 * campi della MotyMoeView. Regioni OpenMP e ordine delle operazioni 1:1. */
#include "nn/moe.h"
#include "nn/nn_rope.h"      /* sigmoidf (inline pura) */
#include "util/prof.h"

/* ---- routing: sigmoid+bias (LFM2) — pesi normalizzati ---- */
static void route_sigmoid(const float *lb, const float *bias, int E, int K, int *idx, float *w) {
    float probs[256]; for (int e = 0; e < E; e++) probs[e] = 1.f/(1.f+expf(-lb[e]));
    for (int kk = 0; kk < K; kk++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            float v = probs[e] + (bias ? bias[e] : 0.f); int used = 0;
            for (int j = 0; j < kk; j++) if (idx[j] == e) used = 1;
            if (!used && v > bv) { bv = v; best = e; }
        }
        idx[kk] = best;
    }
    float ws = 0;
    for (int kk = 0; kk < K; kk++) { w[kk] = probs[idx[kk]]; ws += w[kk]; }
    if (ws > 0) for (int kk = 0; kk < K; kk++) w[kk] /= ws;
}

/* ---- routing: softmax + greedy top-K (Qwen) ---- */
static void route_topk(float *lb, int E, int K, int *idx, float *w) {
    softmax_row(lb, E);
    moe_topk(lb, E, K, idx, w, 1);
}

/* ---- S=1 core: le due varianti condividono tutto tranne il router ---- */
static void moe_d1_core(const MotyMoeView *v, const float *x, float *out, int sig) {
    int D = v->D, E = v->E, K = v->K, I = v->I;
#ifdef MOTY_PROF
    static double _r1=0,_r2=0,_ser=0,_setup=0,_tail=0; static int _nn=0;
    double _t0e = 0, _t0 = 0;
#else
    double _t0e = 0, _t0 = 0; (void)_t0e; (void)_t0;
#endif
    Scratch *scr = v->scr;
    int e4 = (v->ebits <= 4);
    int Sh_ = v->has_shared ? v->sh_inter : 0;
    scr_reset(scr);
    scr_reserve(scr, scr_al((int64_t)E*4) + 2*scr_al((int64_t)K*I*4) + scr_al((int64_t)K*D*4)
                      + 2*scr_al(D) + 2*scr_al((int64_t)K*I) + 2*scr_al((int64_t)Sh_*4));
    float *lb = scr_take(scr, (int64_t)E*4);
    float *gb = scr_take(scr, (int64_t)K*I*4), *ub = scr_take(scr, (int64_t)K*I*4);
    float *hb = scr_take(scr, (int64_t)K*D*4);
    int8_t *xi = scr_take(scr, scr_al(D)), *gi = NULL;
#ifdef MOTY_PROF
    static double _sA=0,_sB=0,_sC=0; double _a = now_s();
#endif
    mat_apply(lb, x, v->router, 1);
#ifdef MOTY_PROF
    _sA += now_s()-_a; _a = now_s();
#endif
    int idx[64]; float w[64];
    if (sig) route_sigmoid(lb, v->expert_bias, E, K, idx, w);
    else     route_topk(lb, E, K, idx, w);
    if (v->has_shared) {
        /* shared expert gated: fuori dalle regioni (pesi ~2MB, mat_apply) */
        int Sh = v->sh_inter;
        float *sgb = scr_take(scr, scr_al((int64_t)Sh*4));
        float *sub = scr_take(scr, scr_al((int64_t)Sh*4));
        mat_apply(sgb, x, v->sh_gate, 1); mat_apply(sub, x, v->sh_up, 1);
        for (int i = 0; i < Sh; i++) { float gv=sgb[i]; sgb[i] = (gv/(1.f+expf(-gv)))*sub[i]; }
        mat_apply(out, sgb, v->sh_down, 1);
        float shw[1]; mat_apply(shw, x, v->sh_router_gate, 1);
        float shgate = sigmoidf(shw[0]);
        for (int d = 0; d < D; d++) out[d] *= shgate;
    } else {
        memset(out, 0, D*sizeof(float));
    }
    ExpertSlot *es[64];
    for (int kk = 0; kk < K; kk++) es[kk] = expert_get(v->ec, v->ectx, v->li, idx[kk], v->load_expert, I, D);
#ifdef MOTY_PROF
    _sB += now_s()-_a; _a = now_s();
#endif
    /* x -> int8 una volta, condiviso da gate/up di tutti gli expert;
     * px: versione permuted per dot_i4i8p (niente permutex nel loop) */
    int8_t *xip = scr_take(scr, scr_al(D));
    int8_t *gip = scr_take(scr, scr_al((int64_t)K*I));
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
        float val;
        if (e4) {
            const uint8_t *w4 = up ? (uint8_t*)e->u + (int64_t)i*rbD : (uint8_t*)e->g + (int64_t)i*rbD;
            float sc = up ? e->us[i] : e->gs[i];
            val = sc * sx * (float)dot_i4i8p(w4, xip, sxsum, D);
        } else {
            const int8_t *w8 = up ? e->u + (int64_t)i*D : e->g + (int64_t)i*D;
            float sc = up ? e->us[i] : e->gs[i];
            val = sc * sx * (float)dot_i8i8(w8, xi, D);
        }
        (up ? ub : gb)[(int64_t)kk*I + i] = val;
    }
#ifdef MOTY_PROF
    _r1 += now_s()-_t0; _t0 = now_s();
#endif
    /* silu-mul + quant g per expert */
    gi = scr_take(scr, scr_al((int64_t)K*I));
    static float sgb[64]; static int32_t sgb_sum[64];
    /* silu+quant per expert: PARALLELI (era ~250us seriali/call = 5.5ms/tok) */
    #pragma omp parallel for schedule(static)
    for (int kk = 0; kk < K; kk++) {
        float *g = gb + (int64_t)kk*I, *u = ub + (int64_t)kk*I;
        for (int i = 0; i < I; i++) { float val = g[i]; g[i] = (val/(1.f+expf(-val)))*u[i]; }
        sgb[kk] = qrow_i8(g, gi + (int64_t)kk*I, I);
        sgb_sum[kk] = px_sum(gi + (int64_t)kk*I, I);
        px_permute(gi + (int64_t)kk*I, gip + (int64_t)kk*I, I);
    }
    const float *sg = sgb;
#ifdef MOTY_PROF
    _ser += now_s()-_t0; _nn++; static double _pl=0; double _nw=now_s();
    if (_nw-_pl > 2.0) { _pl=_nw; fprintf(stderr,"[M] n=%d setup=%.3f [router=%.3f topk+sigmoid=%.3f eget=%.3f quant=%.3f] r1=%.3f r2=%.3f\n", _nn, _setup*1000/_nn, _sA*1000/_nn, (_setup-_sA-_sB-_sC)*1000/_nn, _sB*1000/_nn, _sC*1000/_nn, _r1*1000/_nn, _r2*1000/_nn); }
    _t0 = now_s();
#endif
    /* regione 2: K*D righe down */
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < K*D; r++) {
        int kk = r/D, d = r - kk*D;
        ExpertSlot *e = es[kk];
        const int8_t *xk = gi + (int64_t)kk*I;
        float val;
        if (e4) val = e->ds[d] * sg[kk] * (float)dot_i4i8p((uint8_t*)e->d + (int64_t)d*rbI, gip + (int64_t)kk*I, sgb_sum[kk], I);
        else    val = e->ds[d] * sg[kk] * (float)dot_i8i8(e->d + (int64_t)d*I, xk, I);
        hb[(int64_t)kk*D + d] = val;
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

void moty_nn_moe_sigmoid_d1(const MotyMoeView *v, const float *x, float *out) { moe_d1_core(v, x, out, 1); }
void moty_nn_moe_topk_d1(const MotyMoeView *v, const float *x, float *out) { moe_d1_core(v, x, out, 0); }

/* ---- S>1 batch core: batch-union over experts ---- */
static void moe_batch_core(const MotyMoeView *v, const float *x, int S, float *out, int sig) {
    int D = v->D, E = v->E, K = v->K, I = v->I;
    float *logits = falloc((int64_t)S*E); mat_apply(logits, x, v->router, S);
    int *idx = malloc((int64_t)S*K*sizeof(int)); float *rw = malloc((int64_t)S*K*sizeof(float));
    for (int s = 0; s < S; s++) {
        float *lp = logits + (int64_t)s*E;
        if (sig) route_sigmoid(lp, v->expert_bias, E, K, idx + (int64_t)s*K, rw + (int64_t)s*K);
        else     route_topk(lp, E, K, idx + (int64_t)s*K, rw + (int64_t)s*K);
    }
    free(logits);
    if (v->has_shared) {
        /* shared expert batched su S */
        int Sh = v->sh_inter;
        float *sg = falloc((int64_t)S*Sh), *su = falloc((int64_t)S*Sh);
        mat_apply(sg, x, v->sh_gate, S); mat_apply(su, x, v->sh_up, S);
        for (int64_t i = 0; i < (int64_t)S*Sh; i++) { float gv=sg[i]; sg[i]=(gv/(1.f+expf(-gv)))*su[i]; }
        mat_apply(out, sg, v->sh_down, S);
        float *shg = falloc(S); mat_apply(shg, x, v->sh_router_gate, S);
        for (int s = 0; s < S; s++) { float g = sigmoidf(shg[s]);
            for (int d = 0; d < D; d++) out[(int64_t)s*D+d] *= g; }
        free(sg); free(su); free(shg);
    } else {
        memset(out, 0, (int64_t)S*D*sizeof(float));
    }
    int e4 = (v->ebits <= 4);
    int *tlist = malloc(S*sizeof(int));
    float *gw = falloc((int64_t)S*I), *uu = falloc((int64_t)S*I), *hh = falloc((int64_t)S*D), *xsub = falloc((int64_t)S*D);
    for (int eid = 0; eid < E; eid++) {
        int nt = 0;
        for (int s = 0; s < S; s++) for (int kk = 0; kk < K; kk++)
            if (idx[(int64_t)s*K+kk] == eid) { tlist[nt++] = s; break; }
        if (nt == 0) continue;
        for (int t = 0; t < nt; t++) memcpy(xsub + (int64_t)t*D, x + (int64_t)tlist[t]*D, D*sizeof(float));
        ExpertSlot *e = expert_get(v->ec, v->ectx, v->li, eid, v->load_expert, I, D);
        if (e4) { matmul_i4_s(gw, xsub, (uint8_t*)e->g, e->gs, nt, D, I);
                  matmul_i4_s(uu, xsub, (uint8_t*)e->u, e->us, nt, D, I); }
        else    { matmul_q_s(gw, xsub, e->g, e->gs, nt, D, I);
                  matmul_q_s(uu, xsub, e->u, e->us, nt, D, I); }
        for (int64_t i = 0; i < (int64_t)nt*I; i++) { float val = gw[i]; gw[i] = (val/(1.f+expf(-val)))*uu[i]; }
        if (e4) matmul_i4_s(hh, gw, (uint8_t*)e->d, e->ds, nt, I, D);
        else    matmul_q_s(hh, gw, e->d, e->ds, nt, I, D);
        for (int t = 0; t < nt; t++) for (int kk = 0; kk < K; kk++)
            if (idx[(int64_t)tlist[t]*K+kk] == eid)
                for (int d = 0; d < D; d++) out[(int64_t)tlist[t]*D+d] += rw[(int64_t)tlist[t]*K+kk] * hh[(int64_t)t*D+d];
    }
    free(tlist); free(gw); free(uu); free(hh); free(xsub); free(idx); free(rw);
}

void moty_nn_moe_sigmoid_batch(const MotyMoeView *v, const float *x, int S, float *out) { moe_batch_core(v, x, S, out, 1); }
void moty_nn_moe_topk_batch(const MotyMoeView *v, const float *x, int S, float *out) { moe_batch_core(v, x, S, out, 0); }
