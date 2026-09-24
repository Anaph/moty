/* Logica dei test del motore lfm2 (C puro; glue gtest in lfm2_gtest.cc).
 * Ogni funzione lt_* ritorna 0=ok, 1=fail; dettagli su stderr.
 * Include il motore intero: accesso diretto a static e strutture.
 * Modello sintetico con NOMI HF (Lfm2ForCausalLM, come LFM2.5-350M):
 * conv/attn alternati, QK-norm, FFN con block_auto_adjust_ff_dim. */
#define LFM2_TEST
#include <malloc.h>
#include "../engines/lfm2.c"
#include "tiny_st.h"
#include "ref_dense.h"

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

#include "tiny_lfm2.h"

/* independent reference: HF Lfm2 forward over the whole sequence, all
 * positions' logits -> lo[T][V] */
static void lt_ref_logits(shards *S, const int *ids, int T, double *lo) {
    char nm[128];
    int D = LD, qw = LH*LHD, kw = LKV*LHD;
    double *E = rd_load(S, "model.embed_tokens.weight", (int64_t)LV*D);
    double *x = calloc((size_t)T*D, sizeof(double)), *h = malloc(sizeof(double)*D);
    double *nrm = malloc(sizeof(double)*T*D), *o = malloc(sizeof(double)*T*D);
    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) x[t*D+d] = E[(int64_t)ids[t]*D+d];
    #define LDW(suffix, n) (snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i), rd_load(S,nm,(n)))
    for (int i = 0; i < LL; i++) {
        double *on = LDW("operator_norm.weight", D), *fn = LDW("ffn_norm.weight", D);
        for (int t = 0; t < T; t++) rd_rmsnorm(nrm+t*D, x+t*D, on, D, 1e-5);
        if (lt_is_attn[i]) {
            double *Wq = LDW("self_attn.q_proj.weight", qw*D), *Wk = LDW("self_attn.k_proj.weight", kw*D);
            double *Wv = LDW("self_attn.v_proj.weight", kw*D), *Wo = LDW("self_attn.out_proj.weight", D*qw);
            double *qn = LDW("self_attn.q_layernorm.weight", LHD), *kn = LDW("self_attn.k_layernorm.weight", LHD);
            double *q = malloc(sizeof(double)*T*qw), *k = malloc(sizeof(double)*T*kw);
            double *v = malloc(sizeof(double)*T*kw), *cx = malloc(sizeof(double)*T*qw);
            for (int t = 0; t < T; t++) {
                rd_matvec(q+t*qw, Wq, nrm+t*D, qw, D); rd_matvec(k+t*kw, Wk, nrm+t*D, kw, D);
                rd_matvec(v+t*kw, Wv, nrm+t*D, kw, D);
                for (int hh = 0; hh < LH; hh++) { double *p = q+t*qw+hh*LHD; rd_rmsnorm(p, p, qn, LHD, 1e-5); rd_rope(p, t, 10000.0, LHD); }
                for (int hh = 0; hh < LKV; hh++) { double *p = k+t*kw+hh*LHD; rd_rmsnorm(p, p, kn, LHD, 1e-5); rd_rope(p, t, 10000.0, LHD); }
            }
            rd_attention(cx, q, k, v, T, LH, LKV, LHD);
            for (int t = 0; t < T; t++) rd_matvec(o+t*D, Wo, cx+t*qw, D, qw);
            free(Wq); free(Wk); free(Wv); free(Wo); free(qn); free(kn); free(q); free(k); free(v); free(cx);
        } else {
            double *Wi = LDW("conv.in_proj.weight", 3*D*D), *Wout = LDW("conv.out_proj.weight", D*D);
            double *cw = LDW("conv.conv.weight", D*LK);                 /* [D,1,K] */
            double *bcx = malloc(sizeof(double)*T*3*D), *bx = malloc(sizeof(double)*T*D), *y = malloc(sizeof(double)*D);
            for (int t = 0; t < T; t++) {
                rd_matvec(bcx+t*3*D, Wi, nrm+t*D, 3*D, D);
                for (int c = 0; c < D; c++) bx[t*D+c] = bcx[t*3*D+c] * bcx[t*3*D+2*D+c];   /* B * x */
            }
            for (int t = 0; t < T; t++) {
                for (int c = 0; c < D; c++) {
                    double a = 0;                  /* Conv1d(padding=K-1)[..., :T]: causal */
                    for (int kk = 0; kk < LK; kk++) { int u = t - (LK-1) + kk; if (u >= 0) a += cw[c*LK+kk] * bx[u*D+c]; }
                    y[c] = bcx[t*3*D+D+c] * a;                                             /* C * conv */
                }
                rd_matvec(o+t*D, Wout, y, D, D);
            }
            free(Wi); free(Wout); free(cw); free(bcx); free(bx); free(y);
        }
        for (int j = 0; j < T*D; j++) x[j] += o[j];
        double *W1 = LDW("feed_forward.w1.weight", LI_EFF*D), *W3 = LDW("feed_forward.w3.weight", LI_EFF*D);
        double *W2 = LDW("feed_forward.w2.weight", D*LI_EFF);
        for (int t = 0; t < T; t++) {
            rd_rmsnorm(h, x+t*D, fn, D, 1e-5);
            rd_swiglu(o+t*D, W1, W3, W2, h, D, LI_EFF);
        }
        for (int j = 0; j < T*D; j++) x[j] += o[j];
        free(on); free(fn); free(W1); free(W3); free(W2);
    }
    #undef LDW
    double *fw = rd_load(S, "model.embedding_norm.weight", D);
    for (int t = 0; t < T; t++) { rd_rmsnorm(h, x+t*D, fw, D, 1e-5); rd_matvec(lo+(int64_t)t*LV, E, h, LV, D); }
    free(fw); free(E); free(x); free(h); free(nrm); free(o);
}

static const int lt_ids[10] = {1, 5, 9, 3, 17, 22, 8, 31, 2, 39};

/* engine logits at every position: prefill of `np` tokens, then decode */
static int lt_engine_logits(Model *m, int np, int T, float *lo) {
    float *l = step(m, lt_ids, np, 0);
    memcpy(lo + (int64_t)(np-1)*LV, l, LV*sizeof(float)); free(l);
    for (int t = np; t < T; t++) {
        l = step(m, &lt_ids[t], 1, t);
        memcpy(lo + (int64_t)t*LV, l, LV*sizeof(float)); free(l);
    }
    return 0;
}

int lt_cfg_dense(void) {
    const char *dir = tst_dir("lfm2_tiny_hf");
    lt_write_dir(dir);
    Model m; model_init(&m, dir, 0);
    CHECK(g_lfm_hf == 1);
    CHECK(m.c.inter == LI_EFF);                 /* block_auto_adjust_ff_dim */
    CHECK(fabsf(m.c.eps - 1e-5f) < 1e-12f);     /* norm_eps, not rms_norm_eps */
    CHECK(fabsf(m.c.theta - 10000.f) < 1e-3f);  /* rope_parameters.rope_theta */
    CHECK(m.c.n_experts == 0 && m.c.conv_L == LK);
    CHECK(m.base.lm_tied);
    for (int i = 0; i < LL; i++) {
        CHECK(m.L[i].is_moe == 0);
        CHECK(m.L[i].type == (lt_is_attn[i] ? LT_FULL : LT_CONV));
    }
    CHECK(m.L[0].in_proj.O == 3*LD && m.L[1].o.I == LH*LHD && m.L[3].down.I == LI_EFF);
    return 0;
}

/* load_small must not load the layer Mats (the runtime does it from
 * layer_matrefs; loading them in both places leaked a full copy) */
int lt_load_small_no_mats(void) {
    const char *dir = tst_dir("lfm2_tiny_hf");
    lt_write_dir(dir);
    Model m; memset(&m, 0, sizeof m);
    load_cfg(&m.c, dir); st_init(&m.S, dir);
    load_small(&m);
    for (int i = 0; i < LL; i++) {
        Layer *l = &m.L[i];
        CHECK(l->gate.f == NULL && l->up.f == NULL && l->down.f == NULL);
        CHECK(l->q.f == NULL && l->in_proj.f == NULL && l->out_proj.f == NULL);
        CHECK(l->attn_norm != NULL && l->ffn_norm != NULL);
    }
    return 0;
}

/* f32 engine vs double reference at every position, prefill+decode */
int lt_ref_f32(void) {
    const char *dir = tst_dir("lfm2_tiny_hf");
    lt_write_dir(dir);
    int T = 10;
    double *ref = malloc(sizeof(double)*T*LV);
    Model m; model_init(&m, dir, 0);
    lt_ref_logits(&m.S, lt_ids, T, ref);
    for (int np = 1; np <= 6; np += 5) {        /* token-by-token and 6-token prefill */
        float *lo = calloc((size_t)T*LV, sizeof(float));
        Model e; model_init(&e, dir, 0); kv_alloc(&e, 32);
        lt_engine_logits(&e, np, T, lo);
        double md = 0;
        for (int t = np-1; t < T; t++) for (int v = 0; v < LV; v++) {
            double d = fabs(lo[t*LV+v] - ref[t*LV+v]); if (d > md) md = d;
        }
        if (md > 2e-4) fprintf(stderr, "lt_ref_f32 np=%d: max |dlogit| %.3g\n", np, md);
        CHECK(md < 2e-4);
        free(lo);
    }
    free(ref);
    return 0;
}

/* QBITS=8 (+ int8 KV): close to the reference in relative L2 */
int lt_ref_q8(void) {
    const char *dir = tst_dir("lfm2_tiny_hf");
    lt_write_dir(dir);
    int T = 10;
    double *ref = malloc(sizeof(double)*T*LV);
    Model m; model_init(&m, dir, 0);
    lt_ref_logits(&m.S, lt_ids, T, ref);
    int save = g_kv_bits;
    for (int kv = 0; kv <= 8; kv += 8) {
        g_kv_bits = kv;
        float *lo = calloc((size_t)T*LV, sizeof(float));
        Model e; model_init(&e, dir, 8); kv_alloc(&e, 32);
        lt_engine_logits(&e, 4, T, lo);
        double num = 0, den = 0;
        for (int t = 3; t < T; t++) for (int v = 0; v < LV; v++) {
            double d = lo[t*LV+v] - ref[t*LV+v]; num += d*d; den += ref[t*LV+v]*ref[t*LV+v];
        }
        fprintf(stderr, "lt_ref_q8 kv=%d: rel L2 %.4f\n", kv, sqrt(num/den));
        CHECK(sqrt(num/den) < 0.05);
        free(lo);
    }
    g_kv_bits = save;
    free(ref);
    return 0;
}

/* KV_BITS=8 per-(head,pos) scale arrays must hold KV*max_t entries: the
 * kv_layer_alloc(KV, max_t, hd) swap sized them KV*hd (heap overflow once
 * the context passes head_dim). */
int lt_kv8_alloc(void) {
    const char *dir = tst_dir("lfm2_tiny_hf");
    lt_write_dir(dir);
    int save = g_kv_bits; g_kv_bits = 8;
    Model m; model_init(&m, dir, 8);
    int max_t = 5 * LHD;                          /* context well past head_dim */
    kv_alloc(&m, max_t);
    for (int i = 0; i < LL; i++) {
        if (!lt_is_attn[i]) { CHECK(m.base.Ks[i] == NULL); continue; }
        CHECK(malloc_usable_size(m.base.Ks[i]) >= (size_t)LKV*max_t*sizeof(float));
        CHECK(malloc_usable_size(m.base.Vs[i]) >= (size_t)LKV*max_t*sizeof(float));
    }
    g_kv_bits = save;
    return 0;
}

/* fused q/k/v and gate/up (lfm2_fuse, Q4R4): bit-identical to the separate
 * projections — same activation quantization, same per-block kernel */
int lt_fused_bitexact(void) {
    const char *dir = tst_dir("lfm2_tiny_hf");
    lt_write_dir(dir);
    int save = g_q4fmt; g_q4fmt = 1;
    Model a, b; model_init(&a, dir, 4); model_init(&b, dir, 4);
    lfm2_fuse(&b);
    CHECK(b.L[1].qkv.q4 != NULL && b.L[1].qkv.O == (LH + 2*LKV)*LHD && b.L[0].gate_up.O == 2*LI_EFF);
    CHECK(a.L[1].qkv.q4 == NULL && b.L[0].qkv.q4 == NULL);      /* conv layer: no qkv */
    int T = 8;
    float *la = calloc((size_t)T*LV, sizeof(float)), *lb = calloc((size_t)T*LV, sizeof(float));
    kv_alloc(&a, 16); kv_alloc(&b, 16);
    lt_engine_logits(&a, 3, T, la); lt_engine_logits(&b, 3, T, lb);
    CHECK(!memcmp(la + 2*LV, lb + 2*LV, (size_t)(T-2)*LV*sizeof(float)));
    g_q4fmt = save;
    free(la); free(lb);
    return 0;
}

/* SAVE_PACKED container: QBITS=4 Q4R4 model saved, reloaded from the
 * container (raw packed read) -> bit-identical logits, prefill + decode */
/* mixed precision: Q8_TENSORS picks Q8R4 for layer 0 and the tied head,
 * SAVE_PACKED writes both formats, the container reloads them without the
 * policy (the stored sizes decide) and gives the same logits */
int lt_mixed_roundtrip(void) {
    const char *dir = tst_dir("lfm2_tiny_hf_mix");
    lt_write_dir(dir);
    char src[512]; snprintf(src, sizeof src, "%s", dir);
    const char *pk = tst_dir("lfm2_tiny_packed_mix");
    int save = g_q4fmt; const char *save8 = g_q8_tensors;
    g_q4fmt = 1; g_q8_tensors = "model.layers.0.*,lm_head.weight";
    Model a; model_init(&a, src, 4);
    CHECK(a.L[0].in_proj.fmt == WF_Q8R4 && a.L[0].out_proj.fmt == WF_Q8R4 && a.L[1].q.fmt == WF_Q4R4);
    CHECK(a.base.lm_head.fmt == WF_Q8R4);
    model_save_packed(&a, src, pk);
    g_q8_tensors = NULL;
    Model b; model_init(&b, pk, 4);
    CHECK(b.L[0].in_proj.fmt == WF_Q8R4 && b.L[1].q.fmt == WF_Q4R4 && b.base.lm_head.fmt == WF_Q8R4);
    int T = 8;
    float *la = calloc((size_t)T*LV, sizeof(float)), *lb = calloc((size_t)T*LV, sizeof(float));
    kv_alloc(&a, 16); kv_alloc(&b, 16);
    lt_engine_logits(&a, 3, T, la); lt_engine_logits(&b, 3, T, lb);
    CHECK(!memcmp(la + 2*LV, lb + 2*LV, (size_t)(T-2)*LV*sizeof(float)));
    g_q4fmt = save; g_q8_tensors = save8;
    free(la); free(lb);
    return 0;
}

/* a multimodal wrapper (text_config + "model.language_model." names) loads
 * the same text model: identical logits */
int lt_vl_wrapped(void) {
    const char *d0 = tst_dir("lfm2_tiny_plain"), *d1 = tst_dir("lfm2_tiny_vlwrap");
    lt_write_dir_w(d0, 0); lt_write_dir_w(d1, 1);
    char s0[512], s1[512]; snprintf(s0, sizeof s0, "%s", d0); snprintf(s1, sizeof s1, "%s", d1);
    Model a, b; model_init(&a, s0, 0); model_init(&b, s1, 0);
    CHECK(b.c.n_layers == a.c.n_layers && b.c.vocab == a.c.vocab && b.base.lm_tied);
    int T = 6;
    float *la = calloc((size_t)T*LV, sizeof(float)), *lb = calloc((size_t)T*LV, sizeof(float));
    kv_alloc(&a, 16); kv_alloc(&b, 16);
    lt_engine_logits(&a, 3, T, la); lt_engine_logits(&b, 3, T, lb);
    CHECK(!memcmp(la + 2*LV, lb + 2*LV, (size_t)(T-2)*LV*sizeof(float)));
    free(la); free(lb);
    return 0;
}

/* EMBEDS: rows injected at placeholder positions replace the token
 * embedding. Injecting token X's own embedding row at a placeholder gives
 * exactly the logits of the prompt with X there. */
int lt_embed_inject(void) {
    const char *dir = tst_dir("lfm2_tiny_inject");
    lt_write_dir(dir);
    char src[512]; snprintf(src, sizeof src, "%s", dir);
    enum { NP = 6, P = 38 };
    Model a, b; model_init(&a, src, 0); model_init(&b, src, 0);
    int ids[NP], idp[NP];
    for (int i = 0; i < NP; i++) ids[i] = idp[i] = lt_ids[i];
    idp[1] = P; idp[4] = P;                        /* two placeholders, consumed in order */
    float rows[2*LD];
    memcpy(rows, a.base.embed + (int64_t)ids[1]*LD, LD*sizeof(float));
    memcpy(rows + LD, a.base.embed + (int64_t)ids[4]*LD, LD*sizeof(float));
    b.base.inj = rows; b.base.inj_n = 2; b.base.inj_used = 0; b.base.inj_tok = P;
    kv_alloc(&a, 16); kv_alloc(&b, 16);
    float *la = step(&a, ids, NP, 0), *lb = step(&b, idp, NP, 0);
    CHECK(b.base.inj_used == 2);
    CHECK(!memcmp(la, lb, LV*sizeof(float)));
    free(la); free(lb);
    return 0;
}

int lt_packed_roundtrip(void) {
    const char *dir = tst_dir("lfm2_tiny_hf");
    lt_write_dir(dir);
    char src[512]; snprintf(src, sizeof src, "%s", dir);
    const char *pk = tst_dir("lfm2_tiny_packed");
    int save = g_q4fmt; g_q4fmt = 1;
    Model a; model_init(&a, src, 4);
    CHECK(a.L[0].in_proj.fmt == WF_Q4R4 && a.base.lm_head.fmt == WF_Q4R4);
    model_save_packed(&a, src, pk);
    g_q4fmt = 0;                              /* a container is read as R4 whatever Q4FMT says */
    Model b; model_init(&b, pk, 4);
    CHECK(b.L[1].q.fmt == WF_Q4R4 && b.base.lm_head.fmt == WF_Q4R4);
    int T = 8;
    float *la = calloc((size_t)T*LV, sizeof(float)), *lb = calloc((size_t)T*LV, sizeof(float));
    kv_alloc(&a, 16); kv_alloc(&b, 16);
    lt_engine_logits(&a, 3, T, la); lt_engine_logits(&b, 3, T, lb);
    CHECK(!memcmp(la + 2*LV, lb + 2*LV, (size_t)(T-2)*LV*sizeof(float)));
    g_q4fmt = save;
    free(la); free(lb);
    return 0;
}
