/* Test di nn_attn.h — l'attenzione condivisa GQA (QK-norm + RoPE) usata da
 * lfm2 e qwenmoe. Include un contratto MINIMALE Cfg/Layer/Model come da
 * documento d'intestazione di nn_attn.h. Verifica:
 *   1. CAUSALITÀ/BATCH-INVARIANCE: attention(S=N, pos 0) == N × attention(S=1)
 *      incrementale, BIT-EXACT sia su KV f32 che KV int8 (la riga k/v si
 *      quantizza allo store in entrambi i cammini)
 *   2. RIFERIMENTO SERIALE: attention (path f32) == implementazione di
 *      riferimento indipendente (proiezioni → QK-norm → RoPE → softmax →
 *      mix → o_proj, scalare, senza kernel) a tolleranza float — cattura
 *      errori di wiring GQA (h → h/G), ordine RoPE, scala 1/sqrt(hd)
 *   3. PATH FUSO VNNI (WF_I4G gs=32, D%64==0) == path f32 a tolleranza
 *      della quantizzazione int4 raggruppata
 * Convenzione: 0 = pass, 1 = fail. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#include "../nn_alloc.h"
#include "../simd.h"
#include "../nn_quant.h"
#include "../nn_matmul.h"
#include "../nn_mat.h"       /* Mat, mat_apply, mat_reset_storage, kv_store_row */
#include "../nn_attn_kernels.h"  /* att_scores_*, att_accum_* */
#include "../nn_norm.h"
#include "../nn_rope.h"

typedef struct { int hidden, n_heads, n_kv_heads, head_dim, max_t; float eps, theta; int rot; } Cfg;
typedef struct { Mat q, k, v, o; float *qn, *kn; } Layer;
typedef struct { Cfg c; float **K, **V; int8_t **K8, **V8; float **Ks, **Vs;
                 float *att_sc; int kv_len, max_t; Scratch scr; } Model;

#include "../nn_attn.h"

static uint64_t at_lcg = 0x1234ABCD5678EF90ULL;
static uint32_t at_rnd(void) {
    at_lcg = at_lcg*6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(at_lcg >> 33);
}
static float at_frnd(void) { return (float)at_rnd()/2147483648.0f - 1.0f; }

#define AT_D  64      /* D%64==0: esercita anche il path VNNI fuso */
#define AT_H  4
#define AT_KV 2       /* GQA: G=2 */
#define AT_HD 16      /* H*hd = D */
#define AT_ROT 16
#define AT_T  9       /* token del test di causalità */

typedef struct { Model m; Layer l; } AtCtx;

static void at_fill_mat(Mat *w, int O, int I, int grouped) {
    mat_reset_storage(w);
    w->O = O; w->I = I;
    float *f = malloc((size_t)O*I*sizeof(float));
    for (int i = 0; i < O*I; i++) f[i] = at_frnd()*0.3f;
    if (grouped) {
        int rb = (I+1)/2, ng = I/32;
        w->gs = 32;
        w->q4 = malloc((size_t)O*rb); w->qs = malloc((size_t)O*ng*sizeof(float));
        pack_int4_grouped(f, w->q4, w->qs, O, I, 32);
        w->fmt = WF_I4G;
        free(f);
    } else {
        w->f = f; w->fmt = WF_F32;
    }
}

static void at_ctx_init(AtCtx *ctx, int kv8, int grouped) {
    memset(ctx, 0, sizeof *ctx);
    ctx->m.c = (Cfg){ AT_D, AT_H, AT_KV, AT_HD, AT_T + 4, 1e-5f, 10000.f, AT_ROT };
    at_fill_mat(&ctx->l.q, AT_H*AT_HD, AT_D, grouped);
    at_fill_mat(&ctx->l.k, AT_KV*AT_HD, AT_D, grouped);
    at_fill_mat(&ctx->l.v, AT_KV*AT_HD, AT_D, grouped);
    at_fill_mat(&ctx->l.o, AT_D, AT_D, grouped);
    ctx->l.qn = malloc(AT_HD*sizeof(float)); ctx->l.kn = malloc(AT_HD*sizeof(float));
    for (int i = 0; i < AT_HD; i++) { ctx->l.qn[i] = 1.f + 0.1f*at_frnd(); ctx->l.kn[i] = 1.f + 0.1f*at_frnd(); }
    int max_t = ctx->m.c.max_t;
    ctx->m.max_t = max_t;
    ctx->m.K = calloc(1, sizeof(float*)); ctx->m.V = calloc(1, sizeof(float*));
    ctx->m.K[0] = calloc((size_t)AT_KV*max_t*AT_HD, sizeof(float));
    ctx->m.V[0] = calloc((size_t)AT_KV*max_t*AT_HD, sizeof(float));
    ctx->m.K8 = calloc(1, sizeof(int8_t*)); ctx->m.V8 = calloc(1, sizeof(int8_t*));
    ctx->m.Ks = calloc(1, sizeof(float*)); ctx->m.Vs = calloc(1, sizeof(float*));
    if (kv8) {
        ctx->m.K8[0] = calloc((size_t)AT_KV*max_t*AT_HD, 1);
        ctx->m.V8[0] = calloc((size_t)AT_KV*max_t*AT_HD, 1);
        ctx->m.Ks[0] = calloc((size_t)AT_KV*max_t, sizeof(float));
        ctx->m.Vs[0] = calloc((size_t)AT_KV*max_t, sizeof(float));
    } else {
        ctx->m.K8[0] = NULL; ctx->m.V8[0] = NULL;
    }
    int nth = omp_get_max_threads();
    ctx->m.att_sc = calloc((size_t)nth*max_t, sizeof(float));
}

static void at_ctx_free(AtCtx *ctx) {
    Mat *ms[4] = { &ctx->l.q, &ctx->l.k, &ctx->l.v, &ctx->l.o };
    for (int i = 0; i < 4; i++) { free((void*)ms[i]->f); free(ms[i]->q4); free(ms[i]->qs); }
    free(ctx->l.qn); free(ctx->l.kn);
    free(ctx->m.K[0]); free(ctx->m.V[0]); free(ctx->m.K); free(ctx->m.V);
    free(ctx->m.K8[0]); free(ctx->m.V8[0]); free(ctx->m.K8); free(ctx->m.V8);
    free(ctx->m.Ks[0]); free(ctx->m.Vs[0]); free(ctx->m.Ks); free(ctx->m.Vs);
    free(ctx->m.att_sc); scr_free(&ctx->m.scr);
}

static void at_mk_input(float *x, int S) {
    for (int i = 0; i < S*AT_D; i++) x[i] = at_frnd();
}

/* ---- riferimento seriale indipendente (path f32, no kernel condivisi) ---- */
static void at_reference(AtCtx *ctx, const float *x, int S, float *out) {
    Cfg *c = &ctx->m.c;
    int H = AT_H, KV = AT_KV, hd = AT_HD, G = H/KV;
    int64_t qw = (int64_t)H*hd, kw = (int64_t)KV*hd;
    float *q = malloc((size_t)S*qw*sizeof(float)), *k = malloc((size_t)S*kw*sizeof(float));
    float *v = malloc((size_t)S*kw*sizeof(float)), *ctxv = malloc((size_t)S*qw*sizeof(float));
    for (int s = 0; s < S; s++) {   /* proiezioni come mat_apply f32 */
        for (int64_t o = 0; o < qw; o++) { float a = 0; for (int i = 0; i < AT_D; i++) a += ctx->l.q.f[o*AT_D+i]*x[s*AT_D+i]; q[s*qw+o] = a; }
        for (int64_t o = 0; o < kw; o++) { float a = 0; for (int i = 0; i < AT_D; i++) a += ctx->l.k.f[o*AT_D+i]*x[s*AT_D+i]; k[s*kw+o] = a; }
        for (int64_t o = 0; o < kw; o++) { float a = 0; for (int i = 0; i < AT_D; i++) a += ctx->l.v.f[o*AT_D+i]*x[s*AT_D+i]; v[s*kw+o] = a; }
    }
    for (int s = 0; s < S; s++) {   /* QK-norm + RoPE, stesse formule di nn_attn */
        int pos = s;
        for (int hh = 0; hh < H; hh++) { rmsnorm_row(q+s*qw+hh*hd, q+s*qw+hh*hd, ctx->l.qn, hd, c->eps); rope_head(q+s*qw+hh*hd, pos, c->theta, c->rot); }
        for (int hh = 0; hh < KV; hh++) { rmsnorm_row(k+s*kw+hh*hd, k+s*kw+hh*hd, ctx->l.kn, hd, c->eps); rope_head(k+s*kw+hh*hd, pos, c->theta, c->rot); }
    }
    float scale = 1.f/sqrtf((float)hd);
    for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) {
        int kvh = hh / G;   /* wiring GQA: la testa query hh serve la kv h/G */
        float *sc = malloc((s+1)*sizeof(float));
        for (int t = 0; t <= s; t++) {
            float a = 0; for (int i = 0; i < hd; i++) a += q[s*qw+(int64_t)hh*hd+i]*k[t*kw+(int64_t)kvh*hd+i];
            sc[t] = a*scale;
        }
        softmax_row(sc, s+1);
        for (int i = 0; i < hd; i++) {
            float a = 0; for (int t = 0; t <= s; t++) a += sc[t]*v[t*kw+(int64_t)kvh*hd+i];
            ctxv[s*qw+(int64_t)hh*hd+i] = a;
        }
        free(sc);
    }
    for (int s = 0; s < S; s++) for (int o = 0; o < AT_D; o++) {
        float a = 0; for (int i = 0; i < AT_D; i++) a += ctx->l.o.f[o*AT_D+i]*ctxv[s*AT_D+i];
        out[s*AT_D+o] = a;
    }
    free(q); free(k); free(v); free(ctxv);
}

/* ---- 1. causalità/batch-invariance: S=N in un colpo vs N step S=1 ---- */
static int at_causality_impl(int kv8, int grouped) {
    AtCtx A, B; at_ctx_init(&A, kv8, grouped); at_ctx_init(&B, kv8, grouped);
    /* stessi pesi: ricopia i Mat da A a B (i buffer kv restano freschi) */
    memcpy(B.l.qn, A.l.qn, AT_HD*sizeof(float)); memcpy(B.l.kn, A.l.kn, AT_HD*sizeof(float));
    Mat *ma[4] = { &A.l.q, &A.l.k, &A.l.v, &A.l.o }, *mb[4] = { &B.l.q, &B.l.k, &B.l.v, &B.l.o };
    for (int i = 0; i < 4; i++) {
        free((void*)mb[i]->f); free(mb[i]->q4); free(mb[i]->qs);
        *mb[i] = *ma[i];
        if (ma[i]->fmt == WF_F32) { mb[i]->f = malloc((size_t)ma[i]->O*ma[i]->I*sizeof(float)); memcpy((void*)mb[i]->f, ma[i]->f, (size_t)ma[i]->O*ma[i]->I*sizeof(float)); }
        else { int rb = (ma[i]->I+1)/2, ng = ma[i]->I/32;
               mb[i]->q4 = malloc((size_t)ma[i]->O*rb); memcpy(mb[i]->q4, ma[i]->q4, (size_t)ma[i]->O*rb);
               mb[i]->qs = malloc((size_t)ma[i]->O*ng*sizeof(float)); memcpy(mb[i]->qs, ma[i]->qs, (size_t)ma[i]->O*ng*sizeof(float)); }
    }
    float x[AT_T*AT_D]; at_mk_input(x, AT_T);
    float outb[AT_T*AT_D], outi[AT_T*AT_D], o1[AT_D];

    attention(&A.m, &A.l, 0, x, AT_T, 0, outb);                 /* batch */
    for (int s = 0; s < AT_T; s++) attention(&B.m, &B.l, 0, x + (int64_t)s*AT_D, 1, s, o1),
        memcpy(outi + (int64_t)s*AT_D, o1, AT_D*sizeof(float)); /* incrementale */
    int bad = 0;
    for (int i = 0; i < AT_T*AT_D; i++)
        if (outb[i] != outi[i]) { if (!bad) fprintf(stderr, "mismatch @%d: %g vs %g (kv8=%d grouped=%d)\n", i, outb[i], outi[i], kv8, grouped); bad = 1; }
    /* lo stato kv deve coincidere nei due cammini */
    if (!bad && !kv8) {
        for (int i = 0; i < AT_KV*AT_T*AT_HD; i++)
            if (A.m.K[0][i] != B.m.K[0][i] || A.m.V[0][i] != B.m.V[0][i]) { fprintf(stderr, "kv mismatch @%d\n", i); bad = 1; break; }
    } else if (!bad) {
        for (int i = 0; i < AT_KV*AT_T*AT_HD; i++)
            if (A.m.K8[0][i] != B.m.K8[0][i] || A.m.V8[0][i] != B.m.V8[0][i]) { fprintf(stderr, "kv8 mismatch @%d\n", i); bad = 1; break; }
    }
    at_ctx_free(&A); at_ctx_free(&B);
    return bad;
}
int at_causality_f32(void)  { return at_causality_impl(0, 0); }
int at_causality_kv8(void)  { return at_causality_impl(1, 0); }

/* ---- 2. riferimento seriale (path f32): wiring GQA/RoPE/norm/scala ---- */
int at_vs_reference(void) {
    AtCtx A; at_ctx_init(&A, 0, 0);
    float x[AT_T*AT_D]; at_mk_input(x, AT_T);
    float got[AT_T*AT_D], ref[AT_T*AT_D];
    attention(&A.m, &A.l, 0, x, AT_T, 0, got);
    at_reference(&A, x, AT_T, ref);
    double num = 0, den = 0;
    for (int i = 0; i < AT_T*AT_D; i++) { double d = got[i]-ref[i]; num += d*d; den += (double)ref[i]*ref[i]; }
    double rel = sqrt(num/(den + 1e-30));
    at_ctx_free(&A);
    if (rel > 1e-5) { fprintf(stderr, "rel err vs riferimento: %.3g\n", rel); return 1; }
    return 0;
}

/* ---- 3. path VNNI fuso (I4G gs=32) vs f32 a tolleranza di quant ---- */
int at_fused_vs_f32(void) {
    AtCtx A; at_ctx_init(&A, 0, 0);   /* riferimento f32 */
    AtCtx B; at_ctx_init(&B, 0, 1);   /* tutto I4G: q/k/v fusi + o_proj */
    float x[AT_T*AT_D]; at_mk_input(x, AT_T);
    float got_f32[AT_T*AT_D], got_i4g[AT_T*AT_D];
    attention(&A.m, &A.l, 0, x, AT_T, 0, got_f32);
    attention(&B.m, &B.l, 0, x, AT_T, 0, got_i4g);
    double num = 0, den = 0;
    for (int i = 0; i < AT_T*AT_D; i++) { double d = got_f32[i]-got_i4g[i]; num += d*d; den += (double)got_f32[i]*got_f32[i]; }
    double rel = sqrt(num/(den + 1e-30));
    at_ctx_free(&A); at_ctx_free(&B);
    if (rel > 0.2) { fprintf(stderr, "rel err i4g vs f32: %.3g\n", rel); return 1; }
    return 0;
}
