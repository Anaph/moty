/* LFM2 (Liquid Foundation Model 2.5-8B-A1B, lfm2moe arch).
 * Hybrid: short-conv (depthwise conv1d + gate) + GQA attention.
 * MoE: sigmoid gating + expert_bias + weight normalization.
 *
 * Uses shared headers: nn_attn.h (attention), nn_conv.h (shortconv),
 * nn_ffn.h (dense SwiGLU), nn_moe_sigmoid.h (MoE dispatch).
 * ref: llama.cpp src/models/lfm2.cpp */
#define ENGINE_TAG "lfm2moe"
#define ENGINE_MICRO 0
#define ENGINE_EOT "<|im_end|>\n"

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "io/st.h"
#include "io/gguf.h"
#include "nn/nn.h"
#include "nn/nn_rope.h"
#include "runtime/moe.h"
#include "tok/tok.h"
#include "util/compat.h"
#include "util/prof.h"

enum { LT_CONV = 0, LT_FULL = 1 };

typedef struct {
    int hidden, n_heads, n_kv_heads, head_dim;
    int n_layers, inter, vocab;
    int max_pos, n_eos, eos[4], tie_emb;
    float eps, theta;
    int rot;
    int n_experts, topk, moe_inter;
    int n_dense_layers, conv_L;
    int *ltype;
} Cfg;

typedef struct {
    int type, is_full, is_moe;
    float *attn_norm, *ffn_norm;
    Mat q, k, v, o;
    float *qn, *kn;
    Mat in_proj, out_proj;
    float *conv_w, *conv_state;
    Mat router;
    float *expert_bias;
    Mat gate, up, down;
    ExpertCache *ec;
} Layer;

typedef struct { MODEL_COMMON_FIELDS; } Model;

static void ldm_layer_load_expert(void *ctx, int layer, int eid, ExpertSlot *s, int inter, int hidden);
#define ENGINE_POST_INIT(m) do { \
    Cfg *_c = &(m)->c; \
    _Pragma("omp parallel for collapse(2) schedule(dynamic)") \
    for (int _i = 0; _i < _c->n_layers; _i++) \
        for (int _e = 0; _e < _c->n_experts; _e++) \
            if ((m)->L[_i].ec) \
                expert_get((m)->L[_i].ec, (m), _i, _e, ldm_layer_load_expert, _c->moe_inter, _c->hidden); \
} while (0)
#include "runtime/runtime.h"

/* ---------- config ---------- */
static void load_cfg(Cfg *c, const char *snap) {
    jval *root; char *buf;
    jval *r = cfg_slurp(snap, &root, &buf);
    cfg_common(r, c);
    c->theta = json_get(r,"rope_theta") ? (float)json_get(r,"rope_theta")->num : 5000000.f;
    c->rot = c->head_dim;
    c->n_experts = json_get(r,"num_experts") ? (int)json_get(r,"num_experts")->num : 0;
    c->topk = json_get(r,"num_experts_per_tok") ? (int)json_get(r,"num_experts_per_tok")->num : 0;
    c->moe_inter = json_get(r,"moe_intermediate_size") ? (int)json_get(r,"moe_intermediate_size")->num : 0;
    c->n_dense_layers = json_get(r,"num_dense_layers") ? (int)json_get(r,"num_dense_layers")->num : 0;
    c->conv_L = json_get(r,"conv_L_cache") ? (int)json_get(r,"conv_L_cache")->num : 0;
    if (c->inter == 0 && c->moe_inter > 0) c->inter = c->moe_inter;
    CKR("num_experts", c->n_experts, 0, 4096);
    CKR("topk", c->topk, 0, 64);
    CKR("conv_L", c->conv_L, 0, 16);
    c->ltype = calloc(c->n_layers, sizeof(int));
    for (int i = 0; i < c->n_layers; i++) c->ltype[i] = LT_FULL;
}

static const char *LN(char *b, int sz, int i, const char *s) { snprintf(b,sz,"blk.%d.%s",i,s); return b; }
static const char *HFN(char *b, int sz, int i, const char *s) { snprintf(b,sz,"model.layers.%d.%s",i,s); return b; }

/* ---------- weight loading ---------- */
static void load_small(Model *m) {
    Cfg *c = &m->c;
    int D = c->hidden, L = c->n_layers, convK = c->conv_L;
    int KV = c->n_kv_heads, hd = c->head_dim;
    int64_t kw = (int64_t)KV*hd;
    m->L = calloc(L, sizeof(Layer));
    char nm[128]; int cap = getenv("EXPERT_CACHE") ? atoi(getenv("EXPERT_CACHE")) : 0;
    if (cap < 1) cap = c->n_experts;
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        l->attn_norm = load_t(m, HFN(nm,sizeof(nm),i,"input_layernorm.weight"), D);
        l->ffn_norm  = load_t(m, HFN(nm,sizeof(nm),i,"post_attention_layernorm.weight"), D);
        l->is_full = (st_find(&m->S, HFN(nm,sizeof(nm),i,"self_attn.q_proj.weight")) != NULL);
        l->type = l->is_full ? LT_FULL : LT_CONV;
        c->ltype[i] = l->type;
        l->is_moe = (i >= c->n_dense_layers);
        if (l->is_full) {
            load_mat(m, &l->q, HFN(nm,sizeof(nm),i,"self_attn.q_proj.weight"), D, D);
            load_mat(m, &l->k, HFN(nm,sizeof(nm),i,"self_attn.k_proj.weight"), kw, D);
            load_mat(m, &l->v, HFN(nm,sizeof(nm),i,"self_attn.v_proj.weight"), kw, D);
            load_mat(m, &l->o, HFN(nm,sizeof(nm),i,"self_attn.o_proj.weight"), D, D);
            l->qn = load_t(m, HFN(nm,sizeof(nm),i,"self_attn.q_norm.weight"), hd);
            l->kn = load_t(m, HFN(nm,sizeof(nm),i,"self_attn.k_norm.weight"), hd);
        } else {
            load_mat(m, &l->in_proj,  LN(nm,sizeof(nm),i,"shortconv.in_proj.weight"),  3*D, D);
            load_mat(m, &l->out_proj, LN(nm,sizeof(nm),i,"shortconv.out_proj.weight"), D, D);
            l->conv_w = load_t(m, LN(nm,sizeof(nm),i,"shortconv.conv.weight"), (int64_t)convK*D);
            l->conv_state = falloc((int64_t)(convK-1)*D);
        }
        if (l->is_moe && c->n_experts > 0) {
            load_mat(m, &l->router, LN(nm,sizeof(nm),i,"ffn_gate_inp.weight"), c->n_experts, D);
            snprintf(nm,sizeof(nm),"blk.%d.exp_probs_b.bias",i);
            if (st_find(&m->S, nm)) l->expert_bias = load_t(m, nm, c->n_experts);
            else { l->expert_bias = falloc(c->n_experts); memset(l->expert_bias, 0, c->n_experts*sizeof(float)); }
            l->ec = (ExpertCache*)malloc(sizeof(ExpertCache));
            expert_cache_init(l->ec, cap, c->n_experts);
        } else {
            load_mat(m, &l->gate, HFN(nm,sizeof(nm),i,"mlp.gate_proj.weight"), c->inter, D);
            load_mat(m, &l->up,   HFN(nm,sizeof(nm),i,"mlp.up_proj.weight"),   c->inter, D);
            load_mat(m, &l->down, HFN(nm,sizeof(nm),i,"mlp.down_proj.weight"), D, c->inter);
        }
    }
    m->base.final_norm = load_t(m, "token_embd_norm.weight", D);
}

/* ---------- expert load hook ---------- */
static int g_ebits;
static void ldm_layer_load_expert(void *ctx, int layer, int eid, ExpertSlot *s, int inter, int hidden) {
    Model *m = (Model*)ctx; char nm[128];
    int64_t slen = (int64_t)inter*hidden, dlen = (int64_t)hidden*inter;
    float *tmp = falloc(slen > dlen ? slen : dlen);
    int e4 = (g_ebits <= 4);
    snprintf(nm,sizeof(nm),"blk.%d.ffn_gate_exps.weight",layer);
    st_read_slice_f32(&m->S, nm, (int64_t)eid*slen, slen, tmp, 0);
    if (e4) { s->g=(int8_t*)balloc((int64_t)inter*((hidden+1)/2),"ex.g"); s->gs=falloc(inter);
        pack_int4(tmp,(uint8_t*)s->g,s->gs,inter,hidden); }
    else { s->g=balloc(slen,"ex.g"); s->gs=falloc(inter); quantize_rows(tmp,s->g,s->gs,inter,hidden,8); }
    snprintf(nm,sizeof(nm),"blk.%d.ffn_up_exps.weight",layer);
    st_read_slice_f32(&m->S, nm, (int64_t)eid*slen, slen, tmp, 0);
    if (e4) { s->u=(int8_t*)balloc((int64_t)inter*((hidden+1)/2),"ex.u"); s->us=falloc(inter);
        pack_int4(tmp,(uint8_t*)s->u,s->us,inter,hidden); }
    else { s->u=balloc(slen,"ex.u"); s->us=falloc(inter); quantize_rows(tmp,s->u,s->us,inter,hidden,8); }
    snprintf(nm,sizeof(nm),"blk.%d.ffn_down_exps.weight",layer);
    st_read_slice_f32(&m->S, nm, (int64_t)eid*dlen, dlen, tmp, 0);
    if (e4) { s->d=(int8_t*)balloc((int64_t)hidden*((inter+1)/2),"ex.d"); s->ds=falloc(hidden);
        pack_int4(tmp,(uint8_t*)s->d,s->ds,hidden,inter); }
    else { s->d=balloc(dlen,"ex.d"); s->ds=falloc(hidden); quantize_rows(tmp,s->d,s->ds,hidden,inter,8); }
    free(tmp);
}

/* ---------- shared layer compute ---------- */
#define ATTN_NORM attn_norm
#include "nn/nn_attn.h"       /* attention() */
#include "nn/nn_conv.h"        /* conv_token(), conv_layer() */
#include "nn/nn_ffn.h"         /* dense_ffn() */
#define MOE_LOAD_EXPERT ldm_layer_load_expert
#define MOE_GATE_SIGMOID
#include "nn/nn_moe_sigmoid.h" /* moe_decode1(), moe_batch() */

/* ---------- forward pass ---------- */
static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    /* P5: stream buffers per-Step nell'arena bscr (vive attraverso le
     * chiamate kernel del layer loop; m->base.scr si resetta dentro i kernel) */
    scr_reset(&m->base.bscr);
    scr_reserve(&m->base.bscr, 3*scr_al((int64_t)S*D*4));
    float *xb = scr_take(&m->base.bscr, (int64_t)S*D*4);
    float *nb = scr_take(&m->base.bscr, (int64_t)S*D*4);
    float *tb = scr_take(&m->base.bscr, (int64_t)S*D*4);
    float *x=xb,*nrm=nb,*tmp=tb;
    for (int s = 0; s < S; s++) embed_row(m, ids[s], 1.f, x + (int64_t)s*D);
    int strm = m->base.stream_buf != NULL || m->base.stream_q != NULL;
    if (strm && m->base.n_resident < c->n_layers) layer_prefetch(m, m->base.n_resident);
    PROF_DECL();
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (strm && i >= m->base.n_resident) { layer_stream_in(m, i); if (i+1 < c->n_layers && i+1 >= m->base.n_resident) layer_prefetch(m, i+1); }
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->attn_norm, D, c->eps);
        double c0 = 0;
        if (PROF_ON) c0 = now_s();
        else (void)c0;
        if (l->type == LT_CONV) conv_layer(m, l, nrm, S, tmp); else attention(m, l, i, nrm, S, pos_base, tmp);
        if (l->type == LT_CONV) { PROF_ACC(conv, c0); } else { PROF_ACC(attn, c0); }
        if (PROF_ON) c0 = now_s();
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->ffn_norm, D, c->eps);
        if (l->is_moe) { if (S == 1) moe_decode1(m, l, i, nrm, tmp); else moe_batch(m, l, i, nrm, S, tmp); }
        else dense_ffn(m, l, nrm, S, tmp);
        if (l->is_moe) { PROF_ACC(moe, c0); } else { PROF_ACC(ffn, c0); }
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    PROF_COUNT();
    m->base.kv_len = pos_base + S;
    if (g_skip_logits) return NULL;
    double t_c1 = 0; if (PROF_ON) t_c1 = now_s(); else (void)t_c1;
    float *last = falloc(D); rmsnorm_row(last, x + (int64_t)(S-1)*D, m->base.final_norm, D, c->eps);
    float *logit = falloc(c->vocab); mat_apply(logit, last, &m->base.lm_head, 1);
    PROF_ACC(log, t_c1);
    PROF_WINDOW(m->c.vocab);
    free(last); return logit;
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c; kv_arrays_alloc(m, max_t);
    for (int i = 0; i < c->n_layers; i++) if (c->ltype[i] == LT_FULL) kv_layer_alloc(m, i, c->n_kv_heads, max_t, c->head_dim);
    state_reset(m);
}
static void state_reset(Model *m) {
    Cfg *c = &m->c; if (!m->L) return;
    for (int i = 0; i < c->n_layers; i++) { Layer *l = &m->L[i];
        if (l->conv_state) memset(l->conv_state, 0, (int64_t)(c->conv_L-1)*c->hidden*sizeof(float)); }
}
static int64_t fixed_bytes(Model *m, int ctx) {
    Cfg *c = &m->c; int nfull = 0;
    for (int i = 0; i < c->n_layers; i++) if (c->ltype[i] == LT_FULL) nfull++;
    int64_t rows = (int64_t)nfull * 2 * c->n_kv_heads * ctx;
    return g_kv_bits == 8 ? rows*c->head_dim + rows*4 : rows*c->head_dim*4;
}
static int layer_matrefs(Model *m, int li, MatRef *r) {
    Cfg *c = &m->c; Layer *l = &m->L[li]; int n = 0, D = c->hidden;
    int H = c->n_heads, KV = c->n_kv_heads, hd = c->head_dim;
    #define MR(field, fmt, O_, I_) do { r[n].mat=&l->field; \
        snprintf(r[n].name,sizeof(r[n].name),fmt,li); r[n].O=(O_); r[n].I=(I_); n++; } while(0)
    if (l->is_full) {
        MR(q, "model.layers.%d.self_attn.q_proj.weight", H*hd, D);
        MR(k, "model.layers.%d.self_attn.k_proj.weight", KV*hd, D);
        MR(v, "model.layers.%d.self_attn.v_proj.weight", KV*hd, D);
        MR(o, "model.layers.%d.self_attn.o_proj.weight", D, H*hd);
    } else {
        MR(in_proj,  "blk.%d.shortconv.in_proj.weight", 3*D, D);
        MR(out_proj, "blk.%d.shortconv.out_proj.weight", D, D);
    }
    if (l->is_moe) { MR(router, "blk.%d.ffn_gate_inp.weight", c->n_experts, D); }
    else { MR(gate, "model.layers.%d.mlp.gate_proj.weight", c->inter, D);
           MR(up,   "model.layers.%d.mlp.up_proj.weight",   c->inter, D);
           MR(down, "model.layers.%d.mlp.down_proj.weight", D, c->inter); }
    #undef MR
    return n;
}
static int build_turn(char *buf, int cap, const char *user) {
    return snprintf(buf, cap, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user);
}
static void stops_seed(Model *m, Tok *T) { (void)m;
    stop_add(tok_id_of(T, "<|im_end|>")); stop_add(tok_id_of(T, "<|endoftext|>"));
}
static void banner(Model *m) {
    Cfg *c = &m->c; int nf = 0, nc = 0;
    for (int i = 0; i < c->n_layers; i++) (c->ltype[i] == LT_FULL) ? nf++ : nc++;
    fprintf(stderr, "[lfm2moe] %d layer (%d conv/%d full), hidden %d, %d/%d heads (hd %d), "
        "%d expert top-%d (moe_inter %d, dense_inter %d, %d dense), conv_L %d, vocab %d | "
        "load %.1fs | RSS %.2f GB | idot %s | f32 %s\n",
        c->n_layers, nc, nf, c->hidden, c->n_heads, c->n_kv_heads, c->head_dim,
        c->n_experts, c->topk, c->moe_inter, c->inter, c->n_dense_layers, c->conv_L, c->vocab,
        m->base.load_s, rss_gb(), IDOT_KERNEL, F32_KERNEL);
}

int main(int argc, char **argv) {
    mallopt(M_MMAP_THRESHOLD, 8*1024*1024);   /* logits 512KB: niente mmap/munmap per token */
    g_ebits = getenv("EBITS") ? atoi(getenv("EBITS")) : 4;
    setenv("QBITS", "4", 0);   /* int4 dense grouped: 20.7 tok/s @ 11.6 GB */
    setenv("EBITS", "4", 0);   /* int4 experts via VPDPBUSD */
    setenv("IDOT4", "1", 0);   /* dot_i4i8 for expert GEMV */
    setenv("CTX", "32768", 0);
    if (getenv("THREADS")) { int t = atoi(getenv("THREADS")); if (t > 0) omp_set_num_threads(t); }
    else { int nc = omp_get_num_procs(); if (nc > 12) omp_set_num_threads(nc*3/4); }  /* 8C/16T: 12 > 8 */
    return engine_main(argc, argv);
}
