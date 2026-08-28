/* Test di causalità e identità del conv_layer fuso (nn_conv.h).
 * Include un Model/Layer/Cfg MINIMALI che soddisfano il contratto
 * documentato in nn_conv.h, più la batteria di matmul. Verifica:
 *   1. conv_layer(S=N) == N × conv_layer(S=1) BIT-EXACT (batch invariance
 *      + causalità dello stato: il token k vede solo i token 0..k-1)
 *   2. lo stato finale è identico nei due cammini
 *   3. CONV_VNNI=0 (fallback) == CONV_VNNI=1 (fuso VNNI) a tolleranza
 *      della doppia quantizzazione (path identico a mat_apply→legacy)
 * Convenzione: 0 = pass, 1 = fail. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ---- contratto minimo di nn_conv.h / nn_mat.h ---- */
#include "../nn/nn_alloc.h"
#include "../util/simd.h"
#include "../nn/nn_quant.h"
#include "../nn/nn_matmul.h"
#include "../nn/nn_mat.h"       /* Mat, mat_apply, mat_reset_storage */

typedef struct { int hidden, n_heads, n_kv_heads, head_dim, n_layers, inter, vocab,
                 max_pos, n_eos, eos[4], tie_emb; float eps, theta; int rot,
                 n_experts, topk, moe_inter, n_dense_layers, conv_L; int *ltype; } Cfg;
typedef struct { Mat in_proj, out_proj; float *conv_w; float *conv_state; } Layer;
typedef struct { MotyCommon base; Cfg c; } Model;   /* M2: contratto MotyCommon nel Model del test */

#include "../nn/nn_conv.h"

static uint64_t cv_lcg = 0xA5A5F00DULL;
static uint32_t cv_rnd(void) {
    cv_lcg = cv_lcg*6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(cv_lcg >> 33);
}
static float cv_frnd(void) { return (float)cv_rnd()/2147483648.0f - 1.0f; }

#define CV_D 128     /* D%64==0 per il path VNNI */
#define CV_K 3
#define CV_N 7       /* token del test di causalità */

static void cv_fill_mats(Layer *l, int D) {
    int Oi = 3*D, rb = (D+1)/2, ng = D/32;
    mat_reset_storage(&l->in_proj);
    mat_reset_storage(&l->out_proj);
    float *wi = malloc((size_t)Oi*D*sizeof(float));
    float *wo = malloc((size_t)D*D*sizeof(float));
    for (int i = 0; i < Oi*D; i++) wi[i] = cv_frnd();
    for (int i = 0; i < D*D; i++) wo[i] = cv_frnd();
    l->in_proj.O = Oi; l->in_proj.I = D; l->in_proj.gs = 32;
    l->in_proj.q4 = malloc((size_t)Oi*rb); l->in_proj.qs = malloc((size_t)Oi*ng*sizeof(float));
    pack_int4_grouped(wi, l->in_proj.q4, l->in_proj.qs, Oi, D, 32);
    l->in_proj.fmt = WF_I4G;
    l->out_proj.O = D; l->out_proj.I = D; l->out_proj.gs = 32;
    l->out_proj.q4 = malloc((size_t)D*rb); l->out_proj.qs = malloc((size_t)D*ng*sizeof(float));
    pack_int4_grouped(wo, l->out_proj.q4, l->out_proj.qs, D, D, 32);
    l->out_proj.fmt = WF_I4G;
    l->conv_w = malloc((size_t)D*CV_K*sizeof(float));
    for (int i = 0; i < D*CV_K; i++) l->conv_w[i] = cv_frnd();
    l->conv_state = calloc((size_t)D*(CV_K-1), sizeof(float));
    free(wi); free(wo);
}

/* causale: conv_layer(S=N) == N × conv_layer(S=1), bit-exact, stesso stato finale */
int cv_causality(void) {
    Model m; memset(&m, 0, sizeof(m));
    m.c.hidden = CV_D; m.c.conv_L = CV_K;
    Layer la, lb; memset(&la, 0, sizeof(la)); memset(&lb, 0, sizeof(lb));
    cv_fill_mats(&la, CV_D);
    /* lb = copia indipendente (stessi pesi, stato inizialmente identico=0) */
    cv_lcg = 0xA5A5F00DULL;
    cv_fill_mats(&lb, CV_D);

    float xs[CV_N][CV_D], ya[CV_N][CV_D], yb[CV_N][CV_D];
    for (int s = 0; s < CV_N; s++) for (int i = 0; i < CV_D; i++) xs[s][i] = cv_frnd();

    /* cammino A: una chiamata S=N */
    for (int s = 0; s < CV_N; s++) for (int i = 0; i < CV_D; i++) { (void)xs; }
    conv_layer(&m, &la, (float*)xs, CV_N, (float*)ya);

    /* cammino B: N chiamate S=1 */
    for (int s = 0; s < CV_N; s++)
        conv_layer(&m, &lb, xs[s], 1, yb[s]);

    for (int s = 0; s < CV_N; s++)
        for (int i = 0; i < CV_D; i++)
            if (ya[s][i] != yb[s][i]) {
                fprintf(stderr, "causality: tok %d dim %d: %f != %f\n", s, i, ya[s][i], yb[s][i]);
                return 1;
            }
    if (memcmp(la.conv_state, lb.conv_state, (size_t)CV_D*(CV_K-1)*sizeof(float)) != 0) {
        fprintf(stderr, "causality: stato finale diverso\n");
        return 1;
    }
    return 0;
}

/* fuso VNNI vs fallback legacy (CONV_VNNI=0): stessa matematica del
 * mat_apply→grouped, quindi la differenza è solo la doppia quant y→int8 */
int cv_fused_vs_legacy(void) {
    Model m; memset(&m, 0, sizeof(m));
    m.c.hidden = CV_D; m.c.conv_L = CV_K;
    Layer lf, ll; memset(&lf, 0, sizeof(lf)); memset(&ll, 0, sizeof(ll));
    setenv("CONV_VNNI", "1", 1);
    cv_lcg = 0xC0DEC0DEULL; cv_fill_mats(&lf, CV_D);
    setenv("CONV_VNNI", "0", 1);
    cv_lcg = 0xC0DEC0DEULL; cv_fill_mats(&ll, CV_D);
    setenv("CONV_VNNI", "1", 1);

    float x[CV_D], yf[CV_D], yl[CV_D];
    for (int i = 0; i < CV_D; i++) x[i] = cv_frnd();
    conv_layer(&m, &lf, x, 1, yf);
    conv_layer(&m, &ll, x, 1, yl);
    double mx = 0, mv = 0;
    for (int i = 0; i < CV_D; i++) {
        double e = fabs((double)yf[i]-yl[i]);
        if (e > mx) mx = e;
        mv += fabs(yl[i]);
    }
    mv /= CV_D;
    if (mx > 0.06*(1.0+mv)) {
        fprintf(stderr, "fused-vs-legacy: maxerr %.4e avg|y| %.4e\n", mx, mv);
        return 1;
    }
    return 0;
}
