/* Motore di inferenza Qwen3.5-MoE (qwen35moe: Qwen3.6-35B-A3B) in C puro.
 * Ibrido: layer linear_attention (Gated DeltaNet, 3/4 dei layer) alternati a
 * full_attention (Gated Attention con output-gate, ogni 4°). MLP = MoE:
 *   - router softmax -> top-K (default 8 di 256) normalizzato
 *   - shared expert sempre attivo, gated da sigmoid(shared_expert_gate(x))
 *   - Σ top-K expert (SwiGLU ciascuno)
 * Architettura confermata dal modeling_qwen3_5_moe.py di transformers 4.57:
 *   - GatedDeltaNet (recurrent_gated_delta_rule) == matematica di deltanet_token
 *   - Attention: q_proj doubled [query|gate], qk-RMSNorm (1+w), p-RoPE,
 *     output *= sigmoid(gate) prima di o_proj
 *   - RMSNorm (1+w) per layernorm/qk-norm; dn_norm = w·silu(z) (gated)
 *   - mrope per text-only (T=H=W=pos) riduce a RoPE standard (rot=dimension_count)
 *
 * Uso (env): SNAP=<gguf o dir>, GGUF=<file>, PROMPT/REF/chat, NGEN/CTX/TEMP/...,
 *   EXPERT_CACHE=<n>/EBITS=<2..8> (expert residency + quant), AUTOPIN/EXPERT_PIN
 *   (hot expert), MEM_GB/RESIDENT_LAYERS (dense residency).
 *
 * Matura su runtime.h (hook system): load_cfg/step/kv_alloc/... + engine_main,
 * REF, chat, KV_BITS, streaming, micro-RSS. MoE via moe.h (cache LFRU + pin). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "io/st.h"
#include "io/gguf.h"
#include "nn/nn.h"
#include "nn/nn_rope.h"
#include "runtime/moe.h"
#include "nn/nn_norm.h"
#include "tok/tok.h"

/* env-knob globali (letti in main prima di engine_main) */
static int g_ebits = 4;          /* EBITS: bit expert. 4=int4 packed (2x meno RAM+banda),
                                 * 8=int8 (VPDPBUSD diretto). Default 4 per massima velocita'. */
static int g_expert_cap = 0;     /* EXPERT_CACHE: slot/layer LFRU (0 = tutti residenti) */
static int g_spec_k = 0;        /* SPEC_K: n-gram speculation draft size (0=off, 4=consigliato) */
static int g_spec_n = 3;        /* SPEC_N: n-gram order (3=trigram, default) */

/* qwen35moe GGUF memorizza le value-heads in ordine TILED (vpk outer, k inner)
 * via _reorder_v_heads del convert (qwen.py); l'engine usa GROUPED (hk=hv/R).
 * Converte tiled->grouped (involuzione). Permuta blocchi di head_dim. */
static void vhead_ungroup(float *data, int n_v_heads, int head_dim, int num_k_heads) {
    int R = n_v_heads / num_k_heads;
    if (R <= 1) return;
    float *tmp = (float *)malloc((int64_t)n_v_heads*head_dim*sizeof(float));
    for (int g = 0; g < n_v_heads; g++) {     /* g = grouped (engine) */
        int t = (g % R)*num_k_heads + g/R;    /* tiled (GGUF) -> g */
        memcpy(tmp + (int64_t)g*head_dim, data + (int64_t)t*head_dim, (size_t)head_dim*sizeof(float));
    }
    memcpy(data, tmp, (size_t)n_v_heads*head_dim*sizeof(float));
    free(tmp);
}

#define ENGINE_TAG "qwenmoe"
#define ENGINE_EOT "<|im_end|>\n"
#define ENGINE_MICRO 0

enum { LT_FULL = 0, LT_LINEAR = 1 };

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim, inter, vocab, max_pos;
    float theta, eps;
    int tie_emb;
    int eos[4], n_eos;
    int rot;                    /* dim ruotate dal p-RoPE (= rope.dimension_count) */
    int n_experts, topk, moe_inter, sh_inter;
    int full_interval;          /* full-attention ogni `interval` layer */
    int lin_hk, lin_hv, lin_dk, lin_dv, lin_conv;
    int *ltype;                 /* [n_layers] LT_FULL / LT_LINEAR */
} Cfg;

/* norm: il GGUF di qwen35moe immagazzina il peso EFFETTIVO (llama.cpp ha gia'
 * cotto (1+w) nel peso: i valori sono ~1.0, non ~0). Quindi si usa rmsnorm_row
 * di nn.h (w-style: x*r*w) ovunque, ANCHE per layernorm/qk-norm/final. dn_norm
 * (RMSNormGated) e' nativamente w-style. Niente (1+w) a runtime qui. */

typedef struct {
    int type;                   /* LT_FULL / LT_LINEAR */
    float *in_ln, *post_ln;     /* (1+w) RMSNorm */
    /* full attention (gated output) */
    int gated;                  /* q_proj doubled [query|gate] */
    float *qn, *kn;             /* per-head, head_dim, (1+w) */
    Mat q, k, v, o;
    /* linear attention (Gated DeltaNet) */
    Mat aqkv, az, ab, aa, aout; /* ab=beta(ssm_beta), aa=a-gate(ssm_alpha) */
    float *conv_w, *conv_b, *dt_bias, *A_log, *dn_norm;   /* conv_b=NULL (no bias); dn_norm = w-style (gated) */
    float *conv_state, *Sstate;                   /* ricorrenti */
    /* MoE (comune) */
    Mat router;                 /* ffn_gate_inp [n_experts, hidden] */
    Mat sh_gate, sh_up, sh_down, sh_router_gate;  /* shared expert + its sigmoid gate [1,hidden] */
    ExpertCache *ec;            /* routed experts (moe.h) */
} Layer;

typedef struct {
    MODEL_COMMON_FIELDS;
} Model;

/* Shared DeltaNet kernel (after Model/Layer defined). ENGINE_PRECOOKED_A:
 * qwen35moe GGUF stores -exp(A_log) directly, unlike raw A_log in qwen.c. */
#define ENGINE_PRECOOKED_A
#define ENGINE_GATED_ATTN
#define ATTN_NORM in_ln
#include "nn/nn_attn.h"
#include "nn/nn_deltanet.h"

/* (gli hook load_cfg/load_small/... sono dichiarati forward da runtime.h subito sotto,
 * e definiti dopo l'include: stesso pattern di qwen.c. MatRef vive in runtime.h.) */
static void qwenmoe_load_expert(void *ctx, int layer, int eid, ExpertSlot *s, int inter, int hidden);
#define ENGINE_POST_INIT(m) do { \
    /* warm-up cache expert in parallelo al load: 256 expert/layer, ~7.5GB */ \
    Cfg *_c = &(m)->c; \
    _Pragma("omp parallel for collapse(2) schedule(dynamic)") \
    for (int _i = 0; _i < _c->n_layers; _i++) \
        for (int _e = 0; _e < _c->n_experts; _e++) \
            if ((m)->L[_i].ec) \
                expert_get((m)->L[_i].ec, (m), _i, _e, qwenmoe_load_expert, _c->moe_inter, _c->hidden); \
} while (0)
#include "runtime/runtime.h"

/* ---------- config ---------- */
static void load_cfg(Cfg *c, const char *snap) {
    jval *root; char *buf;
    jval *r = cfg_slurp(snap, &root, &buf);
    cfg_common(r, c);
    jval *th = json_get(r,"rope_theta");    c->theta = th ? (float)th->num : 1000000.f;
    jval *prf = json_get(r,"partial_rotary_factor");
    c->rot = prf ? (int)(c->head_dim * prf->num + 0.5) : c->head_dim;
    if (c->rot < 2 || c->rot > c->head_dim || (c->rot & 1)) {
        fprintf(stderr,"config: p-RoPE incoerente (rot=%d, hd=%d)\n", c->rot, c->head_dim); exit(1);
    }
    /* MoE */
    jval *ne = json_get(r,"num_experts");            c->n_experts = ne ? (int)ne->num : 0;
    jval *tk = json_get(r,"num_experts_per_tok");    c->topk      = tk ? (int)tk->num : 0;
    jval *mi = json_get(r,"moe_intermediate_size");  c->moe_inter = mi ? (int)mi->num : 0;
    jval *si = json_get(r,"shared_expert_intermediate_size"); c->sh_inter = si ? (int)si->num : 0;
    if (c->sh_inter == 0) c->sh_inter = c->moe_inter;
    if (c->n_experts <= 0) { fprintf(stderr,"qwenmoe: config non-MoE (num_experts assente)\n"); exit(1); }
    /* hybrid linear-attention */
    c->lin_hk = json_get(r,"linear_num_key_heads")  ? (int)json_get(r,"linear_num_key_heads")->num : 0;
    c->lin_hv = json_get(r,"linear_num_value_heads")? (int)json_get(r,"linear_num_value_heads")->num: 0;
    c->lin_dk = json_get(r,"linear_key_head_dim")   ? (int)json_get(r,"linear_key_head_dim")->num: 0;
    c->lin_dv = json_get(r,"linear_value_head_dim") ? (int)json_get(r,"linear_value_head_dim")->num: 0;
    c->lin_conv= json_get(r,"linear_conv_kernel_dim")? (int)json_get(r,"linear_conv_kernel_dim")->num:4;
    c->full_interval = json_get(r,"full_attention_interval") ? (int)json_get(r,"full_attention_interval")->num : 0;
    c->ltype = calloc(c->n_layers, sizeof(int));
    jval *lt = json_get(r,"layer_types");
    if (lt && lt->t==J_ARR) {
        for (int i = 0; i < c->n_layers && i < lt->len; i++)
            c->ltype[i] = (lt->kids[i]->t==J_STR && !strcmp(lt->kids[i]->str,"full_attention")) ? LT_FULL : LT_LINEAR;
    } else if (c->full_interval > 0) {
        for (int i = 0; i < c->n_layers; i++) c->ltype[i] = ((i+1) % c->full_interval) ? LT_LINEAR : LT_FULL;
    } else {  /* tutto full attention (fallback: MoE denso-attention) */
        for (int i = 0; i < c->n_layers; i++) c->ltype[i] = LT_FULL;
    }
    int nlin = 0; for (int i = 0; i < c->n_layers; i++) if (c->ltype[i] == LT_LINEAR) nlin++;
    if (nlin && (c->lin_hv<=0 || c->lin_hk<=0 || c->lin_dk<=0 || c->lin_dv<=0 ||
                 c->lin_hv % c->lin_hk || c->lin_conv<1 || c->lin_conv>8)) {
        fprintf(stderr,"config: parametri linear_attention mancanti/incoerenti\n"); exit(1);
    }
    CKR("num_experts", c->n_experts, 1, 4096);
    CKR("num_experts_per_tok", c->topk, 1, 64);
    CKR("moe_intermediate_size", c->moe_inter, 1, 1<<16);
    json_free(root); free(buf);
}

/* helper: nome tensore con fallback (mixed naming HF model.layers.N.* vs GGUF blk.N.*). */
static const char *lname(Model *m, char *buf, int cap, int li, const char *hf_suffix, const char *blk_suffix) {
    if (blk_suffix) snprintf(buf, cap, "blk.%d.%s", li, blk_suffix);
    else snprintf(buf, cap, "model.layers.%d.%s", li, hf_suffix);
    if (st_has(&m->S, buf)) return buf;
    /* prova l'altro schema */
    if (blk_suffix) snprintf(buf, cap, "model.layers.%d.%s", li, hf_suffix ? hf_suffix : "");
    else snprintf(buf, cap, "blk.%d.%s", li, blk_suffix ? blk_suffix : "");
    return buf;
}

static void load_small(Model *m) {
    Cfg *c = &m->c; int D = c->hidden;
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        l->type = c->ltype[i];
        snprintf(nm,sizeof(nm),"model.layers.%d.input_layernorm.weight",i);
        l->in_ln = load_t(m, nm, D);
        /* post_attention_norm: HF per full, blk per linear */
        if (l->type == LT_FULL)
            snprintf(nm,sizeof(nm),"model.layers.%d.post_attention_layernorm.weight",i);
        else
            snprintf(nm,sizeof(nm),"blk.%d.post_attention_norm.weight",i);
        l->post_ln = st_has(&m->S, nm) ? load_t(m, nm, D)
                    : load_t(m, lname(m,nm,sizeof(nm),i,"post_attention_layernorm.weight","post_attention_norm.weight"), D);
        if (l->type == LT_FULL) {
            int hd = c->head_dim;
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.q_norm.weight",i); l->qn = load_t(m,nm,hd);
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.k_norm.weight",i); l->kn = load_t(m,nm,hd);
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.q_proj.weight",i);
            l->gated = (st_numel(&m->S, nm) == (int64_t)2*c->n_heads*hd*D);
        } else {
            int kd = c->lin_hk*c->lin_dk, vd = c->lin_hv*c->lin_dv, cd = 2*kd+vd, K = c->lin_conv;
            int hk=c->lin_hk, hv=c->lin_hv, dk=c->lin_dk, dv=c->lin_dv;
            /* conv1d + stati; conv_w V-canali (dopo 2*kd) value-head-ordered -> ungroup */
            l->conv_w = load_t(m, lname(m,nm,sizeof(nm),i,NULL,"ssm_conv1d.weight"), (int64_t)cd*K);
            vhead_ungroup(l->conv_w + (int64_t)2*kd*K, hv, dv*K, hk);
            snprintf(nm,sizeof(nm),"blk.%d.ssm_dt.bias",i); l->dt_bias = load_t(m,nm,hv); vhead_ungroup(l->dt_bias, hv, 1, hk);
            snprintf(nm,sizeof(nm),"blk.%d.ssm_a",i);       l->A_log   = load_t(m,nm,hv); vhead_ungroup(l->A_log,   hv, 1, hk);
            snprintf(nm,sizeof(nm),"blk.%d.ssm_norm.weight",i); l->dn_norm = load_t(m,nm,dv);  /* condiviso, no ungroup */
            l->conv_state = falloc((int64_t)cd*K);
            l->Sstate = falloc((int64_t)hv*dk*dv);
            /* proiezioni linear-attn: load f32 -> ungroup value-head (tiled->grouped)
             * -> quantizza per qbits (int8 di default: 4x meno traffico decode). */
            #define LDMQ(field, suffix, O_, I_) do { mat_reset_storage(&l->field); \
                snprintf(nm,sizeof(nm),"blk.%d." suffix, i); \
                l->field.f = load_t(m,nm,(int64_t)(O_)*(I_)); l->field.O=(O_); l->field.I=(I_); \
                l->field.fmt=WF_F32; } while(0)   /* poi quantizza dopo ungroup (sotto) */
            LDMQ(aqkv, "attn_qkv.weight", cd, D);  vhead_ungroup(l->aqkv.f + (int64_t)2*kd*D, hv, dv*D, hk);
            LDMQ(az,   "attn_gate.weight", vd, D);  vhead_ungroup(l->az.f, hv, dv*D, hk);
            LDMQ(ab,   "ssm_beta.weight", hv, D);   vhead_ungroup(l->ab.f, hv, D, hk);
            LDMQ(aa,   "ssm_alpha.weight", hv, D);  vhead_ungroup(l->aa.f, hv, D, hk);
            LDMQ(aout, "ssm_out.weight", D, vd);
            { for (int r=0;r<D;r++) vhead_ungroup(l->aout.f + (int64_t)r*vd, hv, dv, hk); }
            #undef LDMQ
            /* quantizza le Mat f32 (post-ungroup) se qbits>0: int8 resident */
            if (m->base.qbits) {
                #define QZ(field, O_, I_) do { Mat *w=&l->field; int8_t *q=balloc((int64_t)(O_)*(I_),#field); \
                    float *qs=falloc(O_); quantize_rows(w->f,q,qs,(O_),(I_),8); \
                    free(w->f); w->f=NULL; w->q=q; w->qs=qs; w->fmt=WF_I8; } while(0)
                QZ(aqkv, cd, D); QZ(az, vd, D); QZ(ab, hv, D); QZ(aa, hv, D); QZ(aout, D, vd);
                #undef QZ
            }
        }
        /* MoE router + shared expert (+ suo gate sigmoid). Comuni a tutti i layer. */
        load_mat(m, &l->router, lname(m,nm,sizeof(nm),i,NULL,"ffn_gate_inp.weight"), c->n_experts, D);
        load_mat(m, &l->sh_gate, lname(m,nm,sizeof(nm),i,NULL,"ffn_gate_shexp.weight"), c->sh_inter, D);
        load_mat(m, &l->sh_up,   lname(m,nm,sizeof(nm),i,NULL,"ffn_up_shexp.weight"),   c->sh_inter, D);
        load_mat(m, &l->sh_down, lname(m,nm,sizeof(nm),i,NULL,"ffn_down_shexp.weight"), D, c->sh_inter);
        load_mat(m, &l->sh_router_gate, lname(m,nm,sizeof(nm),i,NULL,"ffn_gate_inp_shexp.weight"), 1, D);
        /* ExpertCache routed-expert: cap da EXPERT_CACHE (0 = tutti residenti, fast path su RAM larga) */
        int cap = g_expert_cap > 0 ? g_expert_cap : c->n_experts;
        l->ec = (ExpertCache *)malloc(sizeof(ExpertCache));
        expert_cache_init(l->ec, cap, c->n_experts);
    }
}

/* ---------- RoPE neox (half-split) su una testa, rot dims (le restanti passano) ---------- */
/* attention(): ora fornita da nn_attn.h (ENGINE_GATED_ATTN + regione VNNI fusa) */

/* ---------- expert load hook: legge l'expert E dal tensore fuso ffn_*_exps[256,...]
 * (slice block-aligned, dequant), quantizza int8/int4 per riga nello slot. ---------- */
static void qwenmoe_load_expert(void *ctx, int layer, int eid, ExpertSlot *s, int inter, int hidden) {
    Model *m = (Model *)ctx;
    int64_t slen = (int64_t)inter*hidden;   /* = hidden*inter per down */
    float *tmp = falloc(slen);
    char nm[128];
    /* int4-packed quando EBITS<=4: 2x meno RAM, dot_i4i8 (VPDPBUSD su nibble
     * unpacked). Entrambi i path (decode + batch) devono dispatchare sullo
     * stesso fmt: moe_decode1 usa dot_i4i8, moe_batch usa matmul_i4_s. */
    int e4 = (g_ebits <= 4);
    snprintf(nm,sizeof(nm),"blk.%d.ffn_gate_exps.weight",layer);
    st_read_slice_f32(&m->S, nm, (int64_t)eid*slen, slen, tmp, 0);
    if (e4) pack_int4(tmp, (uint8_t*)s->g, s->gs, inter, hidden); else quantize_rows(tmp, s->g, s->gs, inter, hidden, 8);
    snprintf(nm,sizeof(nm),"blk.%d.ffn_up_exps.weight",layer);
    st_read_slice_f32(&m->S, nm, (int64_t)eid*slen, slen, tmp, 0);
    if (e4) pack_int4(tmp, (uint8_t*)s->u, s->us, inter, hidden); else quantize_rows(tmp, s->u, s->us, inter, hidden, 8);
    snprintf(nm,sizeof(nm),"blk.%d.ffn_down_exps.weight",layer);
    st_read_slice_f32(&m->S, nm, (int64_t)eid*slen, slen, tmp, 0);
    if (e4) pack_int4(tmp, (uint8_t*)s->d, s->ds, hidden, inter); else quantize_rows(tmp, s->d, s->ds, hidden, inter, 8);
    free(tmp);
}

#define MOE_LOAD_EXPERT qwenmoe_load_expert
#define MOE_SHARED_EXPERT
#include "nn/nn_moe_sigmoid.h"

/* ---------- MoE: router(top-K) + shared(gated) + Σ expert ---------- */
/* g_expert_cap e' dichiarato sopra (prima di runtime.h/load_small) */

/* moe_decode1: fornita da nn_moe_sigmoid.h (MOE_SHARED_EXPERT, softmax gating) */

/* MoE BATCH (S>1): batch-union — ogni expert caricato 1 volta e applicato a
 * TUTTI i token che lo usano (matmul_q_s batched). Per prefill e spec-decode
 * verification: il peso expert (3MB) e' letto 1 volta per N token → N× meno
 * bandwidth rispetto al per-token. shared expert e router gia' batched su S. */
/* moe_batch: fornita da nn_moe_sigmoid.h */

static void moe(Model *m, Layer *l, int li, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->moe_inter;
    float *logits = falloc((int64_t)S*E);
    mat_apply(logits, x, &l->router, S);            /* router [n_experts, hidden] */
    int Sh = c->sh_inter;
    float *sg = falloc((int64_t)S*Sh), *su = falloc((int64_t)S*Sh);
    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
    int idx[64]; float w[64];
    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s*E;
        softmax_row(pr, E);
        moe_topk(pr, E, K, idx, w, 1);              /* norm_topk (pesi sommano a 1) */
        const float *xs = x + (int64_t)s*D;
        /* shared expert (always-on SwiGLU), gated da sigmoid(shared_router_gate(x)) [1,hidden] */
        float shw[1]; mat_apply(shw, xs, &l->sh_router_gate, 1);
        float *sgi = sg + (int64_t)s*Sh, *sui = su + (int64_t)s*Sh;
        mat_apply(sgi, xs, &l->sh_gate, 1);
        mat_apply(sui, xs, &l->sh_up,   1);
        for (int i = 0; i < Sh; i++) { float gv=sgi[i]; sgi[i] = (gv/(1.f+expf(-gv)))*sui[i]; }
        float *os = out + (int64_t)s*D;
        mat_apply(os, sgi, &l->sh_down, 1);
        float shgate = sigmoidf(shw[0]);
        for (int d = 0; d < D; d++) os[d] *= shgate;
        /* routed expert (top-K via moe.h: cache LFRU + pin) */
        for (int kk = 0; kk < K; kk++) {
            ExpertSlot *e = expert_get(l->ec, m, li, idx[kk], qwenmoe_load_expert, I, D);
            matmul_q(g, xs, e->g, e->gs, D, I);     /* gate [I,D] */
            matmul_q(u, xs, e->u, e->us, D, I);     /* up   [I,D] */
            for (int i = 0; i < I; i++) { float gv=g[i]; g[i]=(gv/(1.f+expf(-gv)))*u[i]; }
            matmul_q(hh, g, e->d, e->ds, I, D);     /* down [D,I] */
            float weight = w[kk];
            for (int d = 0; d < D; d++) os[d] += weight * hh[d];
        }
    }
    free(logits); free(sg); free(su); free(g); free(u); free(hh);
}

/* forward declarations per speculation (defined dopo step) */
static int *step_argmax_all(Model *m, const int *ids, int S, int pos_base);

/* ---------- n-gram speculation engine ----------
 * Hash-table trigram (o n-gram generico) -> token successivo piu' frequente.
 * Costruita dal testo generato; consultata a ogni decode-step per draft K
 * token da verificare in un singolo batched forward (S=K+1). Per MoE il
 * peso expert (3MB) viene letto UNA volta per K+1 token -> ~Kx bandwidth.
 *
 * Modulare: l'hash-table vive in static qui; il draft+verify loop e' in
 * gen_turn_spec (hook che runtime.h chiama se ENGINE_SPEC_ENABLED).
 * L'interazione col model e' solo via step() (batched). */

#define SPEC_NGRAM_N 4      /* n-gram order (hardcoded 4 = quadgram: safe) */
#define SPEC_NGRAM_CAP 65536   /* must be power of 2 */
#define SPEC_NGRAM_MASK (SPEC_NGRAM_CAP - 1)

typedef struct {
    uint64_t key;       /* hash of last N tokens */
    int next;           /* most-recent token that followed this n-gram */
    int used;           /* validity */
} SpecEntry;

static SpecEntry g_spec_table[SPEC_NGRAM_CAP];
static int g_spec_initialized = 0;

static void spec_init(void) {
    if (!g_spec_initialized) {
        memset(g_spec_table, 0, sizeof(g_spec_table));
        g_spec_initialized = 1;
    }
}

/* hash N tokens (rolling): shift + token + mix */
static inline uint64_t spec_hash_n(const int *toks, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= (unsigned)toks[i]; h *= 1099511628211ULL; }
    return h;
}

/* record: after generating 'next' following the n-gram ending at hist[len-1] */
static void spec_observe(const int *hist, int len, int next) {
    if (len < SPEC_NGRAM_N) return;
    uint64_t key = spec_hash_n(hist + len - SPEC_NGRAM_N, SPEC_NGRAM_N);
    SpecEntry *e = &g_spec_table[key & SPEC_NGRAM_MASK];
    e->key = key; e->next = next; e->used = 1;
}

/* draft: look up the n-gram ending at hist[len-1], return up to K continuation
 * tokens (greedy n-gram chain: each drafted token extends the lookup). */
static int spec_draft(const int *hist, int len, int *draft, int maxk) {
    if (len < SPEC_NGRAM_N) return 0;
    /* chain: draft token, then look up the extended n-gram */
    int ndraft = 0;
    int tmp[SPEC_NGRAM_N + 16];   /* working n-gram + drafts */
    memcpy(tmp, hist + len - SPEC_NGRAM_N, SPEC_NGRAM_N * sizeof(int));
    int tmplen = SPEC_NGRAM_N;
    while (ndraft < maxk) {
        uint64_t key = spec_hash_n(tmp + tmplen - SPEC_NGRAM_N, SPEC_NGRAM_N);
        SpecEntry *e = &g_spec_table[key & SPEC_NGRAM_MASK];
        if (!e->used || e->key != key) break;
        draft[ndraft++] = e->next;
        tmp[tmplen++] = e->next;
    }
    return ndraft;
}

/* hook: run a batched step that processes the verified-token + drafts.
 * Ritorna gli argmax (greedy) per ogni posizione [0..S-1] (malloc'd).
 * Il chiamante confronta con i draft e accetta i match. */
/* (spec_verify removed: replaced by step_argmax_all + inline verify in gen_turn_spec) */

/* save/restore DeltaNet recurrent state (per speculation round).
 * Buffer allocato UNA volta (spec_state_alloc) e riusato con memcpy.
 * DeltaNet e' append-only: draft rejected → stato corrotto → restore. */
typedef struct { float *conv_state, *Sstate; int64_t total_conv, total_S; } SpecStateBackup;

static void spec_state_alloc(Model *m, SpecStateBackup *b) {
    Cfg *c = &m->c; b->total_conv = b->total_S = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->ltype[i] != LT_LINEAR) continue;
        int cd = 2*c->lin_hk*c->lin_dk + c->lin_hv*c->lin_dv;
        b->total_conv += (int64_t)cd * c->lin_conv;
        b->total_S += (int64_t)c->lin_hk * c->lin_dk * c->lin_dv;
    }
    b->conv_state = malloc(b->total_conv * sizeof(float));
    b->Sstate = malloc(b->total_S * sizeof(float));
}
static void spec_state_save(Model *m, SpecStateBackup *b) {
    Cfg *c = &m->c; int64_t oc=0, os=0;
    for (int i = 0; i < c->n_layers; i++) { Layer *l=&m->L[i];
        if (l->type != LT_LINEAR) continue;
        int cd=2*c->lin_hk*c->lin_dk+c->lin_hv*c->lin_dv;
        int64_t csz=(int64_t)cd*c->lin_conv, ssz=(int64_t)c->lin_hk*c->lin_dk*c->lin_dv;
        memcpy(b->conv_state+oc, l->conv_state, csz*sizeof(float)); oc+=csz;
        memcpy(b->Sstate+os, l->Sstate, ssz*sizeof(float)); os+=ssz; }
}
static void spec_state_restore(Model *m, SpecStateBackup *b) {
    Cfg *c = &m->c; int64_t oc=0, os=0;
    for (int i = 0; i < c->n_layers; i++) { Layer *l=&m->L[i];
        if (l->type != LT_LINEAR) continue;
        int cd=2*c->lin_hk*c->lin_dk+c->lin_hv*c->lin_dv;
        int64_t csz=(int64_t)cd*c->lin_conv, ssz=(int64_t)c->lin_hk*c->lin_dk*c->lin_dv;
        memcpy(l->conv_state, b->conv_state+oc, csz*sizeof(float)); oc+=csz;
        memcpy(l->Sstate, b->Sstate+os, ssz*sizeof(float)); os+=ssz; }
}
static void spec_state_free(SpecStateBackup *b) { free(b->conv_state); free(b->Sstate); }

/* gen_turn con speculation: usa step_argmax_all per verificare ogni draft.
 * DeltaNet state save/restore: se un draft viene rejectato, ripristina lo
 * stato e replaya i token accettati uno-per-uno (forward singolo).
 * Full-attention KV: le posizioni dei draft sono sovrascrivibili (causal mask).
 * ~Kx bandwidth amortization su MoE quando i draft sono accettati. */
static int gen_turn_spec(Model *m, Tok *T, const int *prompt, int np,
                        int n_new, int *out, int echo, int *stopped) {
    Cfg *c = &m->c;
    spec_init();
    int len = np, maxk = g_spec_k;
    int pos = np;
    *stopped = 0;
    float *logit = step(m, prompt, np, 0);
    pos = np;
    int ng_gen = 0;
    SpecStateBackup backup; spec_state_alloc(m, &backup);
    for (int si = 0; si < n_new; ) {
        int tok = pick_tok(&m->base.scr, logit, c->vocab); free(logit); logit = NULL;
        out[len++] = tok; ng_gen++; pos++;
        if (g_tokens_dump) fprintf(stderr, "%d ", tok);
        if (echo && T) { char buf[64]; int bn = tok_decode(T, &tok, 1, buf, 63); fwrite(buf,1,bn,stdout); fflush(stdout); }
        spec_observe(out, len, tok);
        if (is_stop(tok)) { *stopped = 1; break; }
        si++;
        if (si >= n_new) break;
        /* speculation: draft K token */
        int draft[16];
        int ndraft = spec_draft(out, len, draft, maxk);
        if (ndraft > 0 && ndraft <= 16) {
            int batch[17]; batch[0] = tok;
            for (int d = 0; d < ndraft; d++) batch[d+1] = draft[d];
            int S = ndraft + 1;
            /* save DeltaNet state before batched forward */
            spec_state_save(m, &backup);
            /* batched forward: expert weights read ONCE for S tokens */
            int *am = step_argmax_all(m, batch, S, pos - 1);
            /* verify: am[d] should match draft[d] */
            int naccept = 0;
            for (int d = 0; d < ndraft && si < n_new; d++) {
                if (am[d] != draft[d]) break;
                naccept++;
            }
            free(am);
            if (naccept < ndraft) {
                /* rejection: restore state, replay accepted tokens */
                spec_state_restore(m, &backup);
                for (int d = 0; d < naccept && si < n_new; d++) {
                    logit = step(m, &out[len-1], 1, pos - 1);
                    int real_tok = argmax_v(logit, c->vocab);
                    free(logit); logit = NULL;
                    out[len++] = real_tok; ng_gen++; pos++;
                    if (g_tokens_dump) fprintf(stderr, "%d ", real_tok);
                    if (echo && T) { char buf[64]; int bn = tok_decode(T, &real_tok, 1, buf, 63); fwrite(buf,1,bn,stdout); fflush(stdout); }
                    if (is_stop(real_tok)) { *stopped = 1; break; }
                    si++;
                }
                if (*stopped) break;
                logit = step(m, &out[len-1], 1, pos - 1);
            } else {
                /* all accepted: state is correct, advance */
                pos += S - 1;
                for (int d = 0; d < ndraft && si < n_new; d++) {
                    out[len++] = draft[d]; ng_gen++;
                    if (g_tokens_dump) fprintf(stderr, "%d ", draft[d]);
                    if (echo && T) { char buf[64]; int bn = tok_decode(T, &draft[d], 1, buf, 63); fwrite(buf,1,bn,stdout); fflush(stdout); }
                    if (is_stop(draft[d])) { *stopped = 1; si++; break; }
                    si++;
                }
                if (*stopped) break;
                logit = step(m, &out[len-1], 1, pos - 1);
            }
            continue;
        }
        logit = step(m, &out[len-1], 1, pos - 1);
    }
    if (logit) free(logit);
    spec_state_free(&backup);
    if (g_tokens_dump) fprintf(stderr, "\n");
    return ng_gen;
}

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
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (strm && i >= m->base.n_resident) {
            layer_stream_in(m, i);
            if (i+1 < c->n_layers && i+1 >= m->base.n_resident) layer_prefetch(m, i+1);
        }
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        if (l->type == LT_LINEAR) deltanet(m, l, nrm, S, tmp);
        else attention(m, l, i, nrm, S, pos_base, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        if (S == 1) moe_decode1(m, l, i, nrm, tmp);
        else moe_batch(m, l, i, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    m->base.kv_len = pos_base + S;
    if (g_skip_logits) return NULL;
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->base.final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    mat_apply(logit, last, &m->base.lm_head, 1);
    free(last);
    return logit;
}

/* step_batch_logits: come step ma ritorna gli argmax greedy per OGNI posizione
 * [0..S-1], non solo l'ultimo. Usato da speculation per verificare i draft.
 * MODULAR: separa il forward (step) dal verify (argmax per posizione).
 * Per DeltaNet (ricorrente), ogni posizione ha gia' calcolato il suo hidden
 * nel forward batched; li scarto salvando l'hidden normato ad ogni posizione. */
static int *step_argmax_all(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) embed_row(m, ids[s], 1.f, x + (int64_t)s*D);
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        if (l->type == LT_LINEAR) deltanet(m, l, nrm, S, tmp);
        else attention(m, l, i, nrm, S, pos_base, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        if (S == 1) moe_decode1(m, l, i, nrm, tmp);
        else moe_batch(m, l, i, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    m->base.kv_len = pos_base + S;
    /* argmax per ogni posizione: lm_head su ogni hidden normato */
    int *am = malloc(S * sizeof(int));
    float *hn = falloc(D);
    for (int s = 0; s < S; s++) {
        rmsnorm_row(hn, x + (int64_t)s*D, m->base.final_norm, D, c->eps);
        float *lo = falloc(c->vocab);
        mat_apply(lo, hn, &m->base.lm_head, 1);
        am[s] = argmax_v(lo, c->vocab);
        free(lo);
    }
    free(x); free(nrm); free(tmp); free(hn);
    return am;
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    kv_arrays_alloc(m, max_t);
    for (int i = 0; i < c->n_layers; i++)
        if (c->ltype[i] == LT_FULL) kv_layer_alloc(m, i, c->n_kv_heads, c->head_dim, max_t);
    state_reset(m);
}

static void state_reset(Model *m) {
    Cfg *c = &m->c;
    if (!m->L) return;
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (l->type != LT_LINEAR) continue;
        int cd = 2*c->lin_hk*c->lin_dk + c->lin_hv*c->lin_dv;
        memset(l->conv_state, 0, (int64_t)cd*c->lin_conv*sizeof(float));
        memset(l->Sstate, 0, (int64_t)c->lin_hv*c->lin_dk*c->lin_dv*sizeof(float));
    }
}

/* matrici streamabili dense di un layer (attention/shared/router; NON gli expert
 * routed, che vivono nella ExpertCache). Solo i layer full hanno q/k/v/o. */
static int layer_matrefs(Model *m, int li, MatRef *r) {
    Cfg *c = &m->c; Layer *l = &m->L[li];
    int n = 0, D = c->hidden, hd = c->head_dim, H = c->n_heads, KV = c->n_kv_heads;
    #define MR(field, fmt, O_, I_) do { r[n].mat=&l->field; \
        snprintf(r[n].name,sizeof(r[n].name),fmt,li); r[n].O=(O_); r[n].I=(I_); n++; } while(0)
    if (l->type == LT_FULL) {
        MR(q, "model.layers.%d.self_attn.q_proj.weight", l->gated?2*H*hd:H*hd, D);
        MR(k, "model.layers.%d.self_attn.k_proj.weight", KV*hd, D);
        MR(v, "model.layers.%d.self_attn.v_proj.weight", KV*hd, D);
        MR(o, "model.layers.%d.self_attn.o_proj.weight", D, H*hd);
    } else {
        /* aqkv/az/ab/aa/aout sono caricati in load_small (con ungroup value-head),
         * NON streamati via layer_matrefs. Nessuna matrice dense qui. */
    }
    MR(router, "blk.%d.ffn_gate_inp.weight", c->n_experts, D);
    MR(sh_gate, "blk.%d.ffn_gate_shexp.weight", c->sh_inter, D);
    MR(sh_up,   "blk.%d.ffn_up_shexp.weight",   c->sh_inter, D);
    MR(sh_down, "blk.%d.ffn_down_shexp.weight", D, c->sh_inter);
    MR(sh_router_gate, "blk.%d.ffn_gate_inp_shexp.weight", 1, D);
    #undef MR
    return n;
}

static int64_t fixed_bytes(Model *m, int ctx) {
    Cfg *c = &m->c;
    int nfull = 0; for (int i = 0; i < c->n_layers; i++) nfull += (c->ltype[i]==LT_FULL);
    int64_t rows = (int64_t)nfull * 2 * c->n_kv_heads * ctx;
    return g_kv_bits == 8 ? rows*c->head_dim + rows*4 : rows*c->head_dim*4;
}

static int build_turn(char *buf, int cap, const char *user) {
    return snprintf(buf, cap, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user);
}
static void stops_seed(Model *m, Tok *T) { (void)m;
    stop_add(tok_id_of(T, "<|im_end|>"));
    stop_add(tok_id_of(T, "<|endoftext|>"));
}
static void banner(Model *m) {
    int nfull=0,nlin=0; for(int i=0;i<m->c.n_layers;i++){ if(m->c.ltype[i]==LT_FULL) nfull++; else nlin++; }
    fprintf(stderr,"[" ENGINE_TAG "] %d layer (%d full/%d deltanet), hidden %d, %d/%d teste (hd %d, rot %d), "
        "%d expert top-%d (moe_inter %d, shared %d), vocab %d%s | load %.1fs | RSS %.2f GB | idot %s | f32 %s\n",
        m->c.n_layers, nfull, nlin, m->c.hidden, m->c.n_heads, m->c.n_kv_heads, m->c.head_dim, m->c.rot,
        m->c.n_experts, m->c.topk, m->c.moe_inter, m->c.sh_inter, m->c.vocab,
        m->base.lm_tied?" | lm_head=embed":"", m->base.load_s, rss_gb(), IDOT_KERNEL, F32_KERNEL);
}

#ifndef QWENMOE_TEST
int main(int argc, char **argv) {
    setenv("QBITS", "8", 0);   /* int8 dense re-quant: 7.9 GB (vs 15.4 Q4_K native) */
    setenv("KV_BITS", "8", 0);
    setenv("CTX", "32768", 0);
    /* EBITS: bit expert (2..8). EXPERT_CACHE: slot/layer LFRU (0 = tutti residenti).
     * THREADS di default = core fisici (su SMT, logical/2): i tiny expert-matmul
     * soffrono l'oversubscription SMT.
     * SPEC_K: n-gram speculation draft size (0=off, 3-4=consigliato per testo ripetitivo). */
    if (getenv("EBITS")) { g_ebits = atoi(getenv("EBITS")); if (g_ebits<2||g_ebits>8) { fprintf(stderr,"EBITS 2..8\n"); return 1; } }
    if (getenv("EXPERT_CACHE")) g_expert_cap = atoi(getenv("EXPERT_CACHE"));
    if (getenv("SPEC_K")) g_spec_k = atoi(getenv("SPEC_K"));
    if (getenv("SPEC_N")) g_spec_n = atoi(getenv("SPEC_N"));
    if (!getenv("THREADS")) { int n = omp_get_max_threads(); omp_set_num_threads(n > 12 ? n/2 : n); }
    /* se SPEC_K>0 e non REF: usa il speculation gen loop invece di engine_main */
    if (g_spec_k > 0 && !getenv("REF")) {
        Model m; RunEnv e;
        if (!parse_env(&e)) return 1;
        int maxctx = e.maxctx;
        model_init_ex(&m, e.snap, e.qbits, e.budget, maxctx);
        banner(&m);
        ENGINE_POST_INIT(&m);
        if (m.c.max_pos > 0 && maxctx > m.c.max_pos) maxctx = m.c.max_pos;
        Tok T; char tokpath[2048];
        if (g_gguf) tok_load_gguf(&T, &g_gguf_meta);
        else { snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", e.snap); tok_load(&T, tokpath); }
        stops_seed(&m, &T);
        for (int i = 0; i < m.c.n_eos; i++) stop_add(m.c.eos[i]);
        kv_alloc(&m, maxctx);
        int *hist = malloc(maxctx * sizeof(int));
        char *buf = malloc(1<<16);
        const char *prompt = getenv("PROMPT");
        int ngen = e.ngen;
        if (prompt) {
            int bl = e.templ ? snprintf(buf, 1<<16, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt)
                             : snprintf(buf, 1<<16, "%s", prompt);
            int k = tok_encode(&T, buf, bl, hist, maxctx - 2);
            int cur = ngen; if (k + cur + 1 > maxctx) cur = maxctx - k - 1;
            int stopped;
            double t0 = now_s();
            int ng = gen_turn_spec(&m, &T, hist, k, cur, hist, 1, &stopped);
            double dt = now_s() - t0;
            fprintf(stderr, "\n[qwenmoe+spec] decode %d tok in %.2fs (%.2f tok/s) | RSS %.2f GB\n",
                    ng, dt, ng/(dt>1e-9?dt:1e-9), rss_gb());
            printf("\n");
            return 0;
        }
        fprintf(stderr, "[qwenmoe+spec] chat interattiva con speculation (SPEC_K=%d)\n", g_spec_k);
        int len = 0; char *line = NULL; size_t lcap = 0;
        for (;;) {
            fprintf(stderr, "\n> "); fflush(stderr);
            ssize_t nr = getline(&line, &lcap, stdin);
            if (nr < 0) break;
            while (nr > 0 && (line[nr-1]=='\n' || line[nr-1]=='\r')) line[--nr]=0;
            if (!nr) continue;
            int k = tok_encode(&T, line, (int)nr, hist + len, maxctx - len - 2);
            if (len + k + 8 >= maxctx) {
                fprintf(stderr, "[qwenmoe] contesto pieno, reset\n");
                len = 0; m.base.kv_len = 0; state_reset(&m);
                k = tok_encode(&T, line, (int)nr, hist, maxctx - 2);
            }
            int cur = ngen; if (len + k + cur + 1 > maxctx) cur = maxctx - len - k - 1;
            int stopped;
            double t0 = now_s();
            gen_turn_spec(&m, &T, hist + len, k, cur, hist + len, 1, &stopped);
            double dt = now_s() - t0;
            len += k + cur;
            fprintf(stderr, "\n[qwenmoe+spec] %.2f tok/s\n", cur/(dt>1e-9?dt:1e-9));
        }
        return 0;
    }
    return engine_main(argc, argv);
}
#endif
