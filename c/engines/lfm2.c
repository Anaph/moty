/* LFM2 (Liquid Foundation Model 2 / 2.5): lfm2moe (8B-A1B, GGUF) and dense
 * lfm2 (e.g. LFM2.5-350M, HF safetensors snapshot or GGUF).
 * Hybrid: short-conv (depthwise conv1d + gate) + GQA attention.
 * MoE: sigmoid gating + expert_bias + weight normalization.
 * Tensor names: GGUF (after gguf_map_name) and HF transformers Lfm2 are both
 * accepted; the source is detected once from the final-norm tensor name.
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
    Mat qkv, gate_up;          /* fused views (Q4R4, resident layers): lfm2_fuse */
    ExpertCache *ec;
} Layer;

typedef struct { MODEL_COMMON_FIELDS; } Model;

static void ldm_layer_load_expert(void *ctx, int layer, int eid, ExpertSlot *s, int inter, int hidden);
static void lfm2_fuse(Model *m);
#define ENGINE_POST_INIT(m) do { \
    lfm2_fuse(m); \
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
    /* HF Lfm2Config: rope_theta nested under rope_parameters, eps is norm_eps,
     * tying is tie_embedding (not tie_word_embeddings) */
    jval *rp = json_get(r,"rope_parameters");
    if (rp && rp->t == J_OBJ && json_get(rp,"rope_theta")) c->theta = (float)json_get(rp,"rope_theta")->num;
    if (json_get(r,"norm_eps")) c->eps = (float)json_get(r,"norm_eps")->num;
    jval *te = json_get(r,"tie_embedding");
    c->tie_emb = te && te->t == J_BOOL ? te->boolean : 1;
    /* Lfm2MLP: block_auto_adjust_ff_dim rescales intermediate_size
     * (int(2I/3), * multiplier, round up to block_multiple_of) */
    jval *aa = json_get(r,"block_auto_adjust_ff_dim");
    if (aa && aa->t == J_BOOL && aa->boolean && !json_get(r,"moe_intermediate_size")) {
        int I = (int)(2 * (int64_t)c->inter / 3);
        jval *mu = json_get(r,"block_ffn_dim_multiplier");
        jval *mo = json_get(r,"block_multiple_of");
        if (mu && mu->t == J_NUM) {
            int mult = mo ? (int)mo->num : 1;
            I = (int)(mu->num * I);
            if (mult > 0) I = mult * ((I + mult - 1) / mult);
        }
        c->inter = I;
    }
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

/* per-layer tensor names: [0] GGUF (as seen after gguf_map_name), [1] HF Lfm2 */
enum { N_ATTN_NORM, N_FFN_NORM, N_Q, N_K, N_V, N_O, N_QN, N_KN,
       N_CONV_IN, N_CONV_OUT, N_CONV_W, N_GATE, N_UP, N_DOWN, N_COUNT };
static const char *const g_lfm_names[N_COUNT][2] = {
    [N_ATTN_NORM] = { "model.layers.%d.input_layernorm.weight",          "model.layers.%d.operator_norm.weight" },
    [N_FFN_NORM]  = { "model.layers.%d.post_attention_layernorm.weight", "model.layers.%d.ffn_norm.weight" },
    [N_Q]         = { "model.layers.%d.self_attn.q_proj.weight",         "model.layers.%d.self_attn.q_proj.weight" },
    [N_K]         = { "model.layers.%d.self_attn.k_proj.weight",         "model.layers.%d.self_attn.k_proj.weight" },
    [N_V]         = { "model.layers.%d.self_attn.v_proj.weight",         "model.layers.%d.self_attn.v_proj.weight" },
    [N_O]         = { "model.layers.%d.self_attn.o_proj.weight",         "model.layers.%d.self_attn.out_proj.weight" },
    [N_QN]        = { "model.layers.%d.self_attn.q_norm.weight",         "model.layers.%d.self_attn.q_layernorm.weight" },
    [N_KN]        = { "model.layers.%d.self_attn.k_norm.weight",         "model.layers.%d.self_attn.k_layernorm.weight" },
    [N_CONV_IN]   = { "blk.%d.shortconv.in_proj.weight",                 "model.layers.%d.conv.in_proj.weight" },
    [N_CONV_OUT]  = { "blk.%d.shortconv.out_proj.weight",                "model.layers.%d.conv.out_proj.weight" },
    [N_CONV_W]    = { "blk.%d.shortconv.conv.weight",                    "model.layers.%d.conv.conv.weight" },
    [N_GATE]      = { "model.layers.%d.mlp.gate_proj.weight",            "model.layers.%d.feed_forward.w1.weight" },
    [N_UP]        = { "model.layers.%d.mlp.up_proj.weight",              "model.layers.%d.feed_forward.w3.weight" },
    [N_DOWN]      = { "model.layers.%d.mlp.down_proj.weight",            "model.layers.%d.feed_forward.w2.weight" },
};
static int g_lfm_hf;   /* 1: HF snapshot naming (set by load_small) */
static const char *TN(char *b, int sz, int i, int which) {
    snprintf(b, sz, g_lfm_names[which][g_lfm_hf], i); return b;
}

/* ---------- weight loading ---------- */
static void load_small(Model *m) {
    Cfg *c = &m->c;
    int D = c->hidden, L = c->n_layers, convK = c->conv_L;
    int hd = c->head_dim;
    m->L = calloc(L, sizeof(Layer));
    g_lfm_hf = st_has(&m->S, "model.embedding_norm.weight");
    char nm[128]; int cap = getenv("EXPERT_CACHE") ? atoi(getenv("EXPERT_CACHE")) : 0;
    if (cap < 1) cap = c->n_experts;
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        l->attn_norm = load_t(m, TN(nm,sizeof(nm),i,N_ATTN_NORM), D);
        l->ffn_norm  = load_t(m, TN(nm,sizeof(nm),i,N_FFN_NORM), D);
        l->is_full = (st_find(&m->S, TN(nm,sizeof(nm),i,N_Q)) != NULL);
        l->type = l->is_full ? LT_FULL : LT_CONV;
        c->ltype[i] = l->type;
        l->is_moe = c->n_experts > 0 && i >= c->n_dense_layers;   /* dense lfm2: no MoE layers */
        if (l->is_full) {
            l->qn = load_t(m, TN(nm,sizeof(nm),i,N_QN), hd);
            l->kn = load_t(m, TN(nm,sizeof(nm),i,N_KN), hd);
        } else {
            l->conv_w = load_t(m, TN(nm,sizeof(nm),i,N_CONV_W), (int64_t)convK*D);
            l->conv_state = falloc((int64_t)(convK-1)*D);
        }
        if (l->is_moe && c->n_experts > 0) {
            snprintf(nm,sizeof(nm),"blk.%d.exp_probs_b.bias",i);
            if (st_find(&m->S, nm)) l->expert_bias = load_t(m, nm, c->n_experts);
            else { l->expert_bias = falloc(c->n_experts); memset(l->expert_bias, 0, c->n_experts*sizeof(float)); }
            l->ec = (ExpertCache*)malloc(sizeof(ExpertCache));
            expert_cache_init(l->ec, cap, c->n_experts);
        }
        /* the layer Mats (q/k/v/o, conv in/out, router, gate/up/down) are loaded by
         * the runtime from layer_matrefs: loading them here too leaked a full copy */
    }
    m->base.final_norm = load_t(m, g_lfm_hf ? "model.embedding_norm.weight" : "token_embd_norm.weight", D);
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

/* q/k/v and gate/up of resident Q4R4 layers -> one matrix each (one GEMV,
 * one activation quantization, one parallel region). Streamed layers
 * (MEM_GB) and other formats keep the separate path. */
static void lfm2_fuse(Model *m) {
    for (int i = 0; i < m->c.n_layers && i < m->base.n_resident && !g_micro; i++) {
        Layer *l = &m->L[i];
        if (l->is_full) { Mat *s3[3] = { &l->q, &l->k, &l->v }; moty_mat_fuse_rows(&l->qkv, s3, 3); }
        if (!l->is_moe) { Mat *s2[2] = { &l->gate, &l->up }; moty_mat_fuse_rows(&l->gate_up, s2, 2); }
    }
}

/* ---------- shared layer compute (M3: view → libmoty-nn) ---------- */
#include "nn/attn.h"
#include "nn/conv.h"
#include "nn/ffn.h"
#include "nn/moe.h"

static void att_run(Model *m, Layer *l, int li, const float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c;
    MotyAttnView a = { .q=&l->q, .k=&l->k, .v=&l->v, .o=&l->o, .qn=l->qn, .kn=l->kn,
        .qkv = l->qkv.q4 ? &l->qkv : NULL,
        .n_heads=c->n_heads, .n_kv_heads=c->n_kv_heads, .head_dim=c->head_dim,
        .theta=c->theta, .eps=c->eps, .rot=c->rot,
        .K=m->base.K, .V=m->base.V, .K8=m->base.K8, .V8=m->base.V8,
        .Ks=m->base.Ks, .Vs=m->base.Vs,
        .att_sc=m->base.att_sc, .max_t=m->base.max_t, .scr=&m->base.scr, .li=li };
    moty_nn_attention(&a, x, S, pos_base, out);
}
static void conv_run(Model *m, Layer *l, const float *x, int S, float *out) {
    MotyConvView cv = { .in_proj=&l->in_proj, .out_proj=&l->out_proj,
        .conv_w=l->conv_w, .conv_state=l->conv_state,
        .hidden=m->c.hidden, .conv_L=m->c.conv_L, .scr=&m->base.scr };
    moty_nn_conv_layer(&cv, x, S, out);
}
static void ffn_run(Model *m, Layer *l, const float *x, int S, float *out) {
    MotyFfnView f = { .gate=&l->gate, .up=&l->up, .down=&l->down,
        .gate_up = l->gate_up.q4 ? &l->gate_up : NULL,
        .inter=m->c.inter, .scr=&m->base.scr };
    moty_nn_dense_ffn(&f, x, S, out);
}
static void moe_run(Model *m, Layer *l, int li, const float *x, int S, float *out) {
    Cfg *c = &m->c;
    MotyMoeView v = { .router=&l->router, .expert_bias=l->expert_bias, .ec=l->ec,
        .load_expert=ldm_layer_load_expert, .ectx=m,
        .E=c->n_experts, .K=c->topk, .I=c->moe_inter, .D=c->hidden,
        .li=li, .ebits=g_ebits,
        .has_shared=0, .scr=&m->base.scr };
    if (S == 1) moty_nn_moe_sigmoid_d1(&v, x, out);
    else        moty_nn_moe_sigmoid_batch(&v, x, S, out);
}

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
    OP_T(t_e);
    for (int s = 0; s < S; s++) embed_row(m, ids[s], 1.f, x + (int64_t)s*D);
    OP_ACC(OP_EMBED, t_e);
    int strm = m->base.stream_buf != NULL || m->base.stream_q != NULL;
    if (strm && m->base.n_resident < c->n_layers) layer_prefetch(m, m->base.n_resident);
    PROF_DECL();
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (strm && i >= m->base.n_resident) { layer_stream_in(m, i); if (i+1 < c->n_layers && i+1 >= m->base.n_resident) layer_prefetch(m, i+1); }
        OP_T(t_n1);
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->attn_norm, D, c->eps);
        OP_ACC(OP_NORM, t_n1);
        double c0 = 0;
        if (PROF_ON) c0 = now_s();
        else (void)c0;
        if (l->type == LT_CONV) conv_run(m, l, nrm, S, tmp); else att_run(m, l, i, nrm, S, pos_base, tmp);
        if (l->type == LT_CONV) { PROF_ACC(conv, c0); } else { PROF_ACC(attn, c0); }
        if (PROF_ON) c0 = now_s();
        OP_T(t_r1);
        moty_hw_add(x, tmp, (int64_t)S*D);
        OP_ACC(OP_RESID, t_r1);
        OP_T(t_n2);
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->ffn_norm, D, c->eps);
        OP_ACC(OP_NORM, t_n2);
        if (l->is_moe) moe_run(m, l, i, nrm, S, tmp);
        else ffn_run(m, l, nrm, S, tmp);
        if (l->is_moe) { PROF_ACC(moe, c0); } else { PROF_ACC(ffn, c0); }
        OP_T(t_r2);
        moty_hw_add(x, tmp, (int64_t)S*D);
        OP_ACC(OP_RESID, t_r2);
    }
    PROF_COUNT();
    m->base.kv_len = pos_base + S;
    if (g_skip_logits) return NULL;
    double t_c1 = 0; if (PROF_ON) t_c1 = now_s(); else (void)t_c1;
    OP_T(t_h);
    float *last = falloc(D); rmsnorm_row(last, x + (int64_t)(S-1)*D, m->base.final_norm, D, c->eps);
    float *logit = falloc(c->vocab); mat_apply(logit, last, &m->base.lm_head, 1);
    OP_ACC(OP_LM_HEAD, t_h);
    PROF_ACC(log, t_c1);
    PROF_WINDOW(m->c.vocab);
    free(last); return logit;
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c; kv_arrays_alloc(m, max_t);
    for (int i = 0; i < c->n_layers; i++) if (c->ltype[i] == LT_FULL) kv_layer_alloc(m, i, c->n_kv_heads, c->head_dim, max_t);
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
        MR(q, g_lfm_names[N_Q][g_lfm_hf], H*hd, D);
        MR(k, g_lfm_names[N_K][g_lfm_hf], KV*hd, D);
        MR(v, g_lfm_names[N_V][g_lfm_hf], KV*hd, D);
        MR(o, g_lfm_names[N_O][g_lfm_hf], D, H*hd);
    } else {
        MR(in_proj,  g_lfm_names[N_CONV_IN][g_lfm_hf],  3*D, D);
        MR(out_proj, g_lfm_names[N_CONV_OUT][g_lfm_hf], D, D);
    }
    if (l->is_moe) { MR(router, "blk.%d.ffn_gate_inp.weight", c->n_experts, D); }
    else { MR(gate, g_lfm_names[N_GATE][g_lfm_hf], c->inter, D);
           MR(up,   g_lfm_names[N_UP][g_lfm_hf],   c->inter, D);
           MR(down, g_lfm_names[N_DOWN][g_lfm_hf], D, c->inter); }
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

#ifndef LFM2_TEST
int main(int argc, char **argv) {
#ifdef M_MMAP_THRESHOLD                    /* glibc: logits 512KB, niente mmap/munmap per token */
    mallopt(M_MMAP_THRESHOLD, 8*1024*1024);
#endif
    g_ebits = getenv("EBITS") ? atoi(getenv("EBITS")) : 4;
    setenv("QBITS", "4", 0);   /* int4 dense grouped: 20.7 tok/s @ 11.6 GB */
    setenv("EBITS", "4", 0);   /* int4 experts via VPDPBUSD */
    setenv("IDOT4", "1", 0);   /* dot_i4i8 for expert GEMV */
    setenv("CTX", "32768", 0);
    if (getenv("THREADS")) { int t = atoi(getenv("THREADS")); if (t > 0) omp_set_num_threads(t); }
    else { int nc = omp_get_num_procs(); if (nc > 12) omp_set_num_threads(nc*3/4); }  /* 8C/16T: 12 > 8 */
    return engine_main(argc, argv);
}
#endif /* LFM2_TEST */
