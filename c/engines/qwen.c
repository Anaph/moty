/* Motore di inferenza Qwen3 in C puro (denso: 0.6B..32B).
 * Derivato da olmoe.c: GQA + qk-norm PER TESTA + RoPE (theta 1e6) + SwiGLU denso.
 * Tutti i pesi residenti in RAM (f32, oppure int8 con QBITS=8).
 *
 * Uso (variabili d'ambiente, stile glm/olmoe):
 *   SNAP=<dir snapshot HF>            config.json + tokenizer.json + *.safetensors
 *   GGUF=<file.gguf>                  in alternativa a SNAP: modello single-file (pesi, config
 *                                     e tokenizer dai metadati; F32/F16/BF16/Q8_0/Q4_0/Q4-6_K)
 *   PROMPT="..."                      one-shot; senza PROMPT ne' REF -> chat interattiva su stdin
 *   NGEN=256 CTX=4096                 limiti di generazione/contesto
 *   TEMP=0.7 NUCLEUS=0.95 SEED=n      sampling (TEMP=0 -> greedy)
 *   CHAT_TEMPLATE=1 THINK=0           template chat Qwen3 (<|im_start|>...); THINK=0 chiude il blocco think
 *   QBITS=8|4                         quantizza i pesi al load: int8 (~4x meno RAM) o int4
 *                                     (~8x; embed/lm_head restano int8; QGROUP=32 gruppo scale)
 *   MICRO=1                           micro-RSS: NESSUN peso residente, tutto streamato dal disco
 *                                     (MICRO_DROP=0 per lasciare vivere la page cache)
 *   REF=ref.json                      validazione: greedy sui prompt_ids, confronto con full_ids
 *   TOKENS=1                          dump degli id generati su stderr
 */
#define _GNU_SOURCE
#include "nn/track.h"   /* first: heap calls go through the open tracker */
#include "util/prof.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "nn/nn.h"
#include "nn/nn_rope.h"
#include "io/st.h"
#include "io/gguf.h"
#include "util/stw.h"
#include "tok/tok.h"

#define ENGINE_TAG "qwen"
#define ENGINE_EOT "<|im_end|>\n"

/* tipi di layer (valori del config: layer_types). Attenzione alla polarita':
 * in qwen 1 = linear_attention, in gemma 1 = full_attention. */
enum { LT_FULL = 0, LT_LINEAR = 1 };

/* tetto sui head_dim della parte lineare: dimensiona i buffer su stack di
 * deltanet_token, garantito dal check in load_cfg */
#define MAX_LIN_DV 1024

/* ---------- config (config.json HF di Qwen3 / Qwen3.5) ---------- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim, inter, vocab, max_pos;
    float theta, eps;
    int tie_emb;
    int eos[4], n_eos;
    /* ibrido Qwen3.5 (lignaggio Qwen3-Next): layer linear_attention (Gated
     * DeltaNet) intervallati da full_attention (Gated Attention). */
    int hybrid;
    int rot;                    /* dimensioni ruotate dal RoPE (partial_rotary_factor*head_dim) */
    int lin_hv, lin_hk;         /* teste value / key della parte lineare */
    int lin_dk, lin_dv;         /* head_dim key / value della parte lineare */
    int lin_conv;               /* kernel della conv1d causale */
    int *ltype;                 /* [n_layers] LT_FULL / LT_LINEAR */
} Cfg;

/* adattatore rank-r: y += (alpha/r) * B·(A·x); A[r,I], B[O,r]; B=0 => no-op */
#define LORA_MAX_R 64
typedef struct { float *A, *B; int r; float alpha; } Lora;
typedef struct { Lora q, k, v, o, gate, up, down; } LoraLayer;

typedef struct {
    int type;                              /* LT_FULL / LT_LINEAR */
    float *in_ln, *post_ln;
    /* full attention (anche gated: q_proj raddoppiato con il gate per testa) */
    int gated;
    float *qn, *kn;                        /* qn/kn: lunghezza head_dim (per testa) */
    Mat q, k, v, o;
    /* linear attention (Gated DeltaNet, proiezioni separate stile Qwen3.5) */
    Mat aqkv, az, ab, aa, aout;
    float *conv_w, *conv_b;                /* [conv_dim][K] depthwise (+bias opzionale) */
    float *dt_bias, *A_log;                /* [lin_hv] */
    float *dn_norm;                        /* [lin_dv] rmsnorm gated per testa */
    float *conv_state;                     /* [conv_dim * K] persistente */
    float *Sstate;                         /* [lin_hv * lin_dk * lin_dv] persistente */
    /* mlp (comune) */
    Mat gate, up, down;
    Mat qkv, gate_up;                      /* fused views (Q4R4, resident, no LoRA): qwen_fuse */
    LoraLayer *lo;                         /* adattatori LoRA (NULL = spenti) */
} Layer;

#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
typedef struct { struct timespec modified, changed; } QwenQ41PanelStamp;
#endif

typedef struct {
    Mat *mat;
    const st_tensor *tensor;
    int check_group64, resident;
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
    int raw_gs, panel_fd, panel_verified;
    int64_t panel_bytes;
    uint64_t panel_hash, panel_version;
    QwenQ41PanelStamp panel_stamp;
#endif
} QwenQ41Input;

typedef struct {
    QwenQ41Input ffn[3];
    int64_t bytes;
} QwenQ41Layer;

#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
typedef struct {
    uint8_t *address;
    size_t capacity, page_size;
    int writable;
} QwenQ41FileWindow;
#endif

typedef struct {
    const st_tensor *tensor;
    uint8_t *dst;
    int error;
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
    const QwenQ41Input *panel;
    QwenQ41FileWindow *file_window;
    uint8_t *copy_dst;
    int packed, map_panel, mapped_panel, map_failed, invalid_panel, expect_panel;
    QwenQ41PanelStamp panel_stamp;
#endif
} QwenQ41Read;

#ifdef __APPLE__
typedef struct QwenQ41Queue QwenQ41Queue;
enum { Q41_READERS = 4, Q41_WINDOWS = 3, Q41_MAX_WINDOWS = 4,
       Q41_FFN_MATRICES = 3, Q41_ATTN_MATRICES = 4 };
#endif

typedef struct {
    MODEL_COMMON_FIELDS;                   /* contratto con runtime.h (nn.h) */
    QwenQ41Layer *q41_layers;
    uint8_t *q41_window;
    int64_t q41_window_bytes;
    int q41_window_count, q41_prefill_panels;
#ifdef __APPLE__
    QwenQ41Queue *q41_queue;
#endif
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
    const st_tensor *(*q41_attention)[4];
    uint8_t *q41_attention_raw;
    int q41_attention_group64;
    int q41_panels_requested, q41_decoding;
    unsigned q41_mapped_gate, q41_mapped_down, q41_fallbacks;
#endif
    int kv4;
    float *kv4_scores, *pp_scores;
    int pp_attn, exact_batch;
    int prefill_workspace, prefill_parallel, auto_prefill_chunk;
    Lora lm_lora;                          /* adattatore LoRA sull'lm_head (r=0 = spento) */
} Model;

#include "nn/nn_deltanet.h"
#include "nn/ffn.h"

/* --- TTA sperimentale: adattamento lento a runtime (docs/online-learning.md).
 * TTA=cache -> neural cache (senza gradienti); TTA=bias -> bias sui logit con
 * gradiente in forma chiusa. Default SPENTO: gli hook costano un branch. */
static void tta_adjust(Model *m, float *lo);
static void tta_observe(Model *m, int tok);
static void lora_load(Model *m);
#define ENGINE_LOGITS_HOOK(m, lo) tta_adjust((m), (lo))
#define ENGINE_OBSERVE(m, tok)    tta_observe((m), (tok))
static void qwen_fuse(Model *m);
#define ENGINE_POST_INIT(m)       do { lora_load(m); qwen_fuse(m); } while (0)
#define ENGINE_MICRO 1            /* step() sa girare con embed NULL (gather per riga) */
/* library (runtime/engine_api.h): Qwen3 / Qwen2 / Llama-family dense (MiniCPM5-1B is "llama") */
/* lookahead verification (engine_api.h verify): the decode math per position
 * is kept only when nothing but full attention is stateful and the head is the
 * plain one (no shortlist, no online adapter, no TTA, all layers resident) */
#define ENGINE_VERIFY_OK(m) qwen_verify_ok(m)
static int qwen_verify_ok(Model *m);
#define ENGINE_CLI_VERIFY_OK(m) qwen_cli_verify_ok(m)
#define ENGINE_CLI_VERIFY(m, ids, n, pos) qwen_cli_verify((m),(ids),(n),(pos))
static int qwen_cli_verify_ok(Model *m);
static float *qwen_cli_verify(Model *m, const int *ids, int n, int pos);

#define ENGINE_API_ID qwen
#define ENGINE_API_TYPES "qwen3", "qwen2", "llama"

#define ENGINE_NATIVE_Q41 1
#define ENGINE_Q41_PANELS 1
#define ENGINE_Q41_STREAM 1
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
#define ENGINE_DECODE_PHASE(m, active) ((m)->q41_decoding = (active))
#endif
#ifdef __APPLE__
static void qwen_q41_stream_close(Model *m);
#define ENGINE_CLI_CLEANUP(m) qwen_q41_stream_close(m)
#endif
#include "runtime/runtime.h"

enum { TTA_OFF = 0, TTA_CACHE = 1, TTA_BIAS = 2, TTA_LORA = 3 };
static struct {
    int init, alloc, mode;
    int n, len, head;           /* ring del cache: capacita', riempimento, prossimo slot */
    float lr, lambda, theta;
    float *h;                   /* [n][D] hidden normalizzati */
    int   *tok;                 /* [n] token osservato dopo h_i */
    float *h_cur; int h_valid;  /* [D] hidden della predizione corrente (stash di step) */
    float *bias;                /* [V] (modo bias) */
    float *lA, *lB;             /* (modo lora) adattatore lm_head: A[r,D], B[V,r] */
    int lr_rank; float l_alpha; /* rank r e alpha (=2r) dell'adattatore online */
    float lt[LORA_MAX_R];       /* stash di A·h_cur calcolato in adjust (serve a observe) */
    float *p, *pc, *sc;         /* scratch: softmax corrente [V], distr. cache [V], scores [n] */
    int V, D;                   /* dimensioni al momento dell'alloc */
} g_tta;

static void tta_ensure(Model *m) {
    if (!g_tta.init) {
        g_tta.init = 1;
        const char *e = moty_getenv("TTA");
        g_tta.mode = !e || !*e || !strcmp(e,"0") ? TTA_OFF
                   : !strcmp(e,"cache") ? TTA_CACHE
                   : !strcmp(e,"bias")  ? TTA_BIAS
                   : !strcmp(e,"lora")  ? TTA_LORA : TTA_OFF;
        g_tta.n      = moty_getenv("TTA_N")      ? atoi(moty_getenv("TTA_N"))            : 2048;
        g_tta.lr     = moty_getenv("TTA_LR")     ? (float)atof(moty_getenv("TTA_LR"))
                                            : (g_tta.mode == TTA_LORA ? 1e-3f : 0.1f);
        g_tta.lambda = moty_getenv("TTA_LAMBDA") ? (float)atof(moty_getenv("TTA_LAMBDA")) : 0.1f;
        g_tta.theta  = moty_getenv("TTA_THETA")  ? (float)atof(moty_getenv("TTA_THETA"))  : 1.0f;
        if (g_tta.lambda < 0) g_tta.lambda = 0;
        if (g_tta.lambda > 0.5f) g_tta.lambda = 0.5f;   /* il cache non puo' dominare */
        if (g_tta.n < 1) g_tta.n = 1;
        if (g_tta.mode) fprintf(stderr, "[qwen] TTA sperimentale: %s (n=%d lr=%g lambda=%g theta=%g)\n",
                                g_tta.mode==TTA_CACHE?"cache":g_tta.mode==TTA_BIAS?"bias":"lora",
                                g_tta.n, g_tta.lr, g_tta.lambda, g_tta.theta);
    }
    if (g_tta.mode && !g_tta.alloc) {
        int V = m->c.vocab, D = m->c.hidden;
        g_tta.h     = falloc((int64_t)g_tta.n * D);
        g_tta.tok   = malloc(g_tta.n * sizeof(int));
        g_tta.h_cur = falloc(D);
        g_tta.bias  = calloc(V, sizeof(float));
        g_tta.p     = falloc(V);
        g_tta.pc    = falloc(V);
        g_tta.sc    = falloc(g_tta.n);
        g_tta.V = V; g_tta.D = D;
        if (g_tta.mode == TTA_LORA) {
            /* adattatore online sull'lm_head: A fissato casuale (deterministico),
             * B parte a zero -> no-op finche' non si osservano token */
            int r = moty_getenv("TTA_RANK") ? atoi(moty_getenv("TTA_RANK")) : 4;
            if (r < 1) r = 1;
            if (r > LORA_MAX_R) r = LORA_MAX_R;
            g_tta.lr_rank = r; g_tta.l_alpha = 2.f * r;
            g_tta.lA = falloc((int64_t)r * D);
            uint64_t rs = 0x5EEDCAFE12345ULL;
            float scn = 1.f / sqrtf((float)D);
            for (int64_t i = 0; i < (int64_t)r * D; i++) {
                rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
                g_tta.lA[i] = ((float)((rs >> 11) * (1.0/9007199254740992.0)) * 2.f - 1.f) * scn;
            }
            g_tta.lB = calloc((int64_t)V * r, sizeof(float));
        }
        g_tta.alloc = 1;
    }
}

static void tta_reset(void) {
    g_tta.len = 0; g_tta.head = 0; g_tta.h_valid = 0;
    if (g_tta.alloc) memset(g_tta.bias, 0, g_tta.V*sizeof(float));
    /* lora: azzerare B rende l'adattatore un no-op esatto; A (proiezione
     * casuale fissa) si conserva */
    if (g_tta.alloc && g_tta.lB)
        memset(g_tta.lB, 0, (int64_t)g_tta.V * g_tta.lr_rank * sizeof(float));
}

/* aggiusta i logits della predizione corrente (chiamato da gen_turn dopo step) */
static void tta_adjust(Model *m, float *lo) {
    tta_ensure(m);
    if (g_tta.mode == TTA_OFF) return;
    int V = m->c.vocab, D = m->c.hidden;
    if (g_tta.mode == TTA_BIAS) {
        for (int v = 0; v < V; v++) lo[v] += g_tta.bias[v];
        memcpy(g_tta.p, lo, V*sizeof(float));      /* softmax per l'update in observe */
        softmax_row(g_tta.p, V);
        return;
    }
    if (g_tta.mode == TTA_LORA) {
        /* logits += (alpha/r) * B·(A·h_cur); t = A·h_cur stashato per observe */
        if (g_tta.h_valid) {
            int r = g_tta.lr_rank;
            float sc = g_tta.l_alpha / r;
            for (int j = 0; j < r; j++)
                g_tta.lt[j] = dot_f32(g_tta.lA + (int64_t)j*D, g_tta.h_cur, D);
            for (int v = 0; v < V; v++)
                lo[v] += sc * dot_f32(g_tta.lB + (int64_t)v*r, g_tta.lt, r);
        }
        memcpy(g_tta.p, lo, V*sizeof(float));      /* softmax per l'update in observe */
        softmax_row(g_tta.p, V);
        return;
    }
    /* cache: logits' = log((1-l)*p_model + l*p_cache) */
    if (!g_tta.h_valid || g_tta.len == 0 || g_tta.lambda <= 0) return;
    double s2 = 0; for (int d = 0; d < D; d++) s2 += (double)g_tta.h_cur[d]*g_tta.h_cur[d];
    float r = 1.f/sqrtf((float)s2 + 1e-12f);
    float *hn = g_tta.pc;                          /* riuso momentaneo dello scratch */
    for (int d = 0; d < D; d++) hn[d] = g_tta.h_cur[d]*r;
    for (int i = 0; i < g_tta.len; i++)
        g_tta.sc[i] = g_tta.theta * dot_f32(hn, g_tta.h + (int64_t)i*D, D);
    softmax_row(g_tta.sc, g_tta.len);
    memset(g_tta.pc, 0, V*sizeof(float));
    for (int i = 0; i < g_tta.len; i++) g_tta.pc[g_tta.tok[i]] += g_tta.sc[i];
    memcpy(g_tta.p, lo, V*sizeof(float));
    softmax_row(g_tta.p, V);
    float om = 1.f - g_tta.lambda;
    for (int v = 0; v < V; v++) lo[v] = logf(om*g_tta.p[v] + g_tta.lambda*g_tta.pc[v] + 1e-30f);
}

/* osserva il token successivo fissato (chiamato da gen_turn dopo pick_tok) */
static void tta_observe(Model *m, int tok) {
    if (g_tta.mode == TTA_OFF || !g_tta.alloc) return;
    int V = m->c.vocab, D = m->c.hidden;
    if (g_tta.mode == TTA_BIAS) {
        /* SGD esatto sulla CE col bias: b += lr*(e_x - softmax(logit+b)) */
        for (int v = 0; v < V; v++) g_tta.bias[v] -= g_tta.lr * g_tta.p[v];
        g_tta.bias[tok] += g_tta.lr;
        return;
    }
    if (g_tta.mode == TTA_LORA) {
        /* SGD sulla CE dell'adattatore: g = p - e_tok (gradiente sui logit).
         * dB[v,:] = sc*g_v*t, dA[j,:] = sc*bg_j*h con bg = B^T g calcolato
         * PRIMA di aggiornare B (altrimenti il gradiente di A sarebbe sporco). */
        if (!g_tta.h_valid) return;
        int r = g_tta.lr_rank;
        float sc = g_tta.l_alpha / r, step = g_tta.lr * sc;
        float bg[LORA_MAX_R];
        for (int j = 0; j < r; j++) bg[j] = 0;
        for (int v = 0; v < V; v++) {
            float gv = g_tta.p[v] - (v == tok ? 1.f : 0.f);
            const float *Bv = g_tta.lB + (int64_t)v*r;
            for (int j = 0; j < r; j++) bg[j] += gv * Bv[j];
        }
        for (int v = 0; v < V; v++) {
            float gv = g_tta.p[v] - (v == tok ? 1.f : 0.f);
            float *Bv = g_tta.lB + (int64_t)v*r;
            for (int j = 0; j < r; j++) Bv[j] -= step * gv * g_tta.lt[j];
        }
        for (int j = 0; j < r; j++) {
            float cj = step * bg[j];
            float *Aj = g_tta.lA + (int64_t)j*D;
            for (int d = 0; d < D; d++) Aj[d] -= cj * g_tta.h_cur[d];
        }
        return;
    }
    if (!g_tta.h_valid) return;
    double s2 = 0; for (int d = 0; d < D; d++) s2 += (double)g_tta.h_cur[d]*g_tta.h_cur[d];
    float r = 1.f/sqrtf((float)s2 + 1e-12f);
    float *dst = g_tta.h + (int64_t)g_tta.head * D;
    for (int d = 0; d < D; d++) dst[d] = g_tta.h_cur[d]*r;
    g_tta.tok[g_tta.head] = tok;
    g_tta.head = (g_tta.head + 1) % g_tta.n;
    if (g_tta.len < g_tta.n) g_tta.len++;
}

/* ---------- LoRA runtime ----------
 * Adattatori low-rank sul percorso denso: y += (alpha/r)*B·(A·x) dopo ogni
 * proiezione adattata. Caricati da LORA=<file|dir safetensors> con nomi
 * lora.layers.N.self_attn.{q,k,v,o}_proj.{A,B}, lora.layers.N.mlp.{gate,up,
 * down}_proj.{A,B}, lora.lm_head.{A,B} e scalare opzionale lora.alpha. */

/* applica l'adattatore su y[S,O] con input x[S,I]; seriale (r piccolo) */
static void lora_apply(const Lora *lo, float *y, const float *x, int S, int I, int O) {
    if (!lo || !lo->A || lo->r <= 0) return;
    float sc = lo->alpha / lo->r;
    float t[LORA_MAX_R];                        /* r <= LORA_MAX_R garantito al load */
    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s*I;
        for (int j = 0; j < lo->r; j++) t[j] = dot_f32(lo->A + (int64_t)j*I, xs, I);
        float *ys = y + (int64_t)s*O;
        for (int o = 0; o < O; o++) ys[o] += sc * dot_f32(lo->B + (int64_t)o*lo->r, t, lo->r);
    }
}

/* carica uno slot base.A/base.B se presente; valida rank e forme contro le
 * dimensioni della matrice base. Ritorna r caricato, 0 se assente. */
static int lora_slot(shards *LS, char *used, const char *base, Lora *lo, int I, int O, float alpha) {
    char na[192], nb[192];
    snprintf(na, sizeof(na), "%s.A", base);
    snprintf(nb, sizeof(nb), "%s.B", base);
    st_tensor *ta = st_find(LS, na);
    if (!ta) return 0;
    st_tensor *tb = st_find(LS, nb);
    if (!tb) { moty_fail_code(MOTY_FAIL_FORMAT, "[qwen] LoRA: %s presente ma manca %s\n", na, nb); }
    if (I <= 0 || ta->numel % I) {
        moty_fail_code(MOTY_FAIL_FORMAT, "[qwen] LoRA: %s numel %lld incompatibile con I=%d\n", na, (long long)ta->numel, I); }
    int r = (int)(ta->numel / I);
    if (r < 1 || r > LORA_MAX_R || tb->numel != (int64_t)O*r) {
        moty_fail_code(MOTY_FAIL_FORMAT, "[qwen] LoRA: %s: rank %d fuori range [1,%d] oppure %s numel %lld != %d*%d\n",
                na, r, LORA_MAX_R, nb, (long long)tb->numel, O, r); }
    lo->r = r;
    lo->alpha = alpha > 0 ? alpha : 2.0f*r;     /* default: alpha = 2r */
    lo->A = falloc(ta->numel); st_read_f32(LS, na, lo->A, 0);
    lo->B = falloc(tb->numel); st_read_f32(LS, nb, lo->B, 0);
    used[ta - LS->t] = 1; used[tb - LS->t] = 1;
    return r;
}

/* legge LORA=<path> (file singolo o directory safetensors) e attacca gli
 * adattatori al modello. Fallisce RUMOROSAMENTE su tensori non riconosciuti. */
static void lora_load(Model *m) {
    const char *path = moty_getenv("LORA");
    if (!path || !*path) return;
    struct stat sb;
    if (stat(path, &sb) != 0) { moty_fail_code(MOTY_FAIL_IO, "%s: %s", path, strerror(errno)); }
    shards LS;
    if (S_ISDIR(sb.st_mode)) st_init(&LS, path);
    else st_init_file(&LS, path);
    char *used = calloc(LS.n ? LS.n : 1, 1);
    float alpha = 0.f;                          /* 0 = non impostato -> default 2r per slot */
    st_tensor *tal = st_find(&LS, "lora.alpha");
    if (tal) {
        if (tal->numel != 1) { moty_fail_code(MOTY_FAIL_FORMAT, "[qwen] LoRA: lora.alpha deve essere scalare\n"); }
        st_read_f32(&LS, "lora.alpha", &alpha, 0);
        used[tal - LS.t] = 1;
    }
    int nt = 0, rload = 0;
    int D = m->c.hidden, inter = m->c.inter;
    for (int i = 0; i < m->c.n_layers; i++) {
        Layer *l = &m->L[i];
        LoraLayer tmp; memset(&tmp, 0, sizeof(tmp));
        char base[160];
        int got = 0;
        #define SLOT(field, sub, I_, O_) do { \
            snprintf(base, sizeof(base), "lora.layers.%d." sub, i); \
            int r_ = lora_slot(&LS, used, base, &tmp.field, (I_), (O_), alpha); \
            if (r_) { got = 1; nt += 2; rload = r_; } } while (0)
        SLOT(q,    "self_attn.q_proj", D, l->q.O);
        SLOT(k,    "self_attn.k_proj", D, l->k.O);
        SLOT(v,    "self_attn.v_proj", D, l->v.O);
        SLOT(o,    "self_attn.o_proj", l->o.I, D);
        SLOT(gate, "mlp.gate_proj", D, inter);
        SLOT(up,   "mlp.up_proj",   D, inter);
        SLOT(down, "mlp.down_proj", inter, D);
        #undef SLOT
        if (got) {
            l->lo = calloc(1, sizeof(LoraLayer));
            if (!l->lo) { moty_fail_code(MOTY_FAIL_OOM, "[qwen] LoRA: OOM\n"); }
            *l->lo = tmp;
        }
    }
    if (lora_slot(&LS, used, "lora.lm_head", &m->lm_lora, D, m->c.vocab, alpha)) nt += 2;
    /* ogni tensore lora.* del file deve essere stato consumato: un nome storto
     * (typo, layout diverso) sarebbe altrimenti ignorato in silenzio */
    int bad = 0;
    for (int i = 0; i < LS.n; i++)
        if (!used[i]) { fprintf(stderr, "[qwen] LoRA: tensore non riconosciuto: %s\n", LS.t[i].name); bad = 1; }
    if (bad) moty_fail_code(MOTY_FAIL_FORMAT, "[qwen] LoRA: unrecognized tensors in the adapter file");
    free(used);
    fprintf(stderr, "[qwen] LoRA: %d tensori, r=%d, alpha=%g, layer adattati:", nt, rload,
            alpha > 0 ? alpha : 2.0f*rload);
    for (int i = 0; i < m->c.n_layers; i++) if (m->L[i].lo) fprintf(stderr, " %d", i);
    if (m->lm_lora.r) fprintf(stderr, " +lm_head");
    fprintf(stderr, "\n");
}

/* q/k/v and gate/up of resident Q4R4 layers -> one matrix each (see
 * moty_mat_fuse_rows). Layers with LoRA adapters keep separate projections
 * (the adapters are applied per matrix); gated attention keeps its split. */
static int qwen_verify_ok(Model *m) {
    for (int i = 0; i < m->c.n_layers; i++) if (m->c.ltype[i] == LT_LINEAR || m->L[i].lo) return 0;
    return !m->base.head_sl && !m->lm_lora.r && g_tta.mode == TTA_OFF && !g_micro && m->base.n_resident == m->c.n_layers;
}

static void qwen_fuse(Model *m) {
    for (int i = 0; i < m->c.n_layers && i < m->base.n_resident && !g_micro; i++) {
        Layer *l = &m->L[i];
        if (l->lo) continue;
        if (l->type == LT_FULL && !l->gated) { Mat *s3[3] = { &l->q, &l->k, &l->v }; moty_mat_fuse_rows(&l->qkv, s3, 3); }
        Mat *s2[2] = { &l->gate, &l->up }; moty_mat_fuse_rows(&l->gate_up, s2, 2);
    }
}

/* ---------- caricamento config ---------- */
static void load_cfg(Cfg *c, const char *snap) {
    jval *root; char *buf;
    jval *r = cfg_slurp(snap, &root, &buf);
    cfg_common(r, c);
    jval *th = json_get(r,"rope_theta");   c->theta = th ? (float)th->num : 1000000.f;
    jval *te = json_get(r,"tie_word_embeddings"); c->tie_emb = (te && te->t==J_BOOL) ? te->boolean : 0;
    /* --- parte ibrida (Qwen3.5): presente solo se il config la dichiara --- */
    jval *prf = json_get(r,"partial_rotary_factor");
    c->rot = prf ? (int)(c->head_dim * prf->num + 0.5) : c->head_dim;
    if (c->rot < 2 || c->rot > c->head_dim || (c->rot & 1)) {
        moty_fail_code(MOTY_FAIL_FORMAT, "config: partial_rotary_factor incoerente (rot=%d, head_dim=%d)\n", c->rot, c->head_dim);
    }
    jval *lv = json_get(r,"linear_num_value_heads");
    c->lin_hv  = lv ? (int)lv->num : 0;
    jval *lk = json_get(r,"linear_num_key_heads");   c->lin_hk  = lk ? (int)lk->num : 0;
    jval *ld = json_get(r,"linear_key_head_dim");    c->lin_dk  = ld ? (int)ld->num : 0;
    jval *le = json_get(r,"linear_value_head_dim");  c->lin_dv  = le ? (int)le->num : 0;
    jval *lc = json_get(r,"linear_conv_kernel_dim"); c->lin_conv= lc ? (int)lc->num : 4;
    c->ltype = calloc(c->n_layers, sizeof(int));
    jval *lt = json_get(r,"layer_types");
    if (lt && lt->t==J_ARR) {
        for (int i = 0; i < c->n_layers && i < lt->len; i++)
            c->ltype[i] = (lt->kids[i]->t==J_STR && !strcmp(lt->kids[i]->str,"linear_attention")) ? LT_LINEAR : LT_FULL;
    } else if (c->lin_hv > 0) {
        jval *fi = json_get(r,"full_attention_interval");
        int interval = fi ? (int)fi->num : 4;
        for (int i = 0; i < c->n_layers; i++) c->ltype[i] = ((i+1) % interval) ? LT_LINEAR : LT_FULL;
    }
    c->hybrid = 0;
    for (int i = 0; i < c->n_layers; i++) if (c->ltype[i] == LT_LINEAR) c->hybrid = 1;
    if (c->hybrid && (c->lin_hv<=0 || c->lin_hk<=0 || c->lin_dk<=0 || c->lin_dv<=0 ||
                      c->lin_dk>MAX_LIN_DV || c->lin_dv>MAX_LIN_DV ||
                      c->lin_hv % c->lin_hk || c->lin_conv<1 || c->lin_conv>8)) {
        moty_fail_code(MOTY_FAIL_FORMAT, "config: parametri linear_attention mancanti o incoerenti\n");
    }
    json_free(root); free(buf);       /* Cfg non trattiene puntatori nel JSON */
}

/* elenco delle MATRICI di un layer (unica fonte per loader, streamer e
 * prefetcher). Richiede l->type/l->gated gia' impostati. Ritorna il numero. */
static int layer_matrefs(Model *m, int li, MatRef *r) {
    Cfg *c = &m->c; Layer *l = &m->L[li];
    int n = 0, D = c->hidden, hd = c->head_dim, H = c->n_heads, KV = c->n_kv_heads;
    #define MR(field, fmt, O_, I_) do { r[n].mat=&l->field; \
        snprintf(r[n].name,sizeof(r[n].name),"model.layers.%d." fmt,li); \
        r[n].O=(O_); r[n].I=(I_); n++; } while(0)
    if (l->type == LT_FULL) {
        MR(q, "self_attn.q_proj.weight", l->gated ? 2*H*hd : H*hd, D);
        MR(k, "self_attn.k_proj.weight", KV*hd, D);
        MR(v, "self_attn.v_proj.weight", KV*hd, D);
        MR(o, "self_attn.o_proj.weight", D, H*hd);
    } else {
        int kd = c->lin_hk*c->lin_dk, vd = c->lin_hv*c->lin_dv, cd = 2*kd + vd;
        MR(aqkv, "linear_attn.in_proj_qkv.weight", cd, D);
        MR(az,   "linear_attn.in_proj_z.weight",   vd, D);
        MR(ab,   "linear_attn.in_proj_b.weight",   c->lin_hv, D);
        MR(aa,   "linear_attn.in_proj_a.weight",   c->lin_hv, D);
        MR(aout, "linear_attn.out_proj.weight",    D, vd);
    }
    MR(gate, "mlp.gate_proj.weight", c->inter, D);
    MR(up,   "mlp.up_proj.weight",   c->inter, D);
    MR(down, "mlp.down_proj.weight", D, c->inter);
    #undef MR
    return n;
}

/* parte piccola SEMPRE residente: norme, vettori e stati deltanet */
static void load_small(Model *m) {
    Cfg *c = &m->c;
    int D = c->hidden, hd = c->head_dim, H = c->n_heads;
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        l->type = c->ltype[i];
        #define LDT(field, suffix, n_) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm,n_)
        LDT(in_ln,  "input_layernorm.weight", D);
        LDT(post_ln,"post_attention_layernorm.weight", D);
        if (l->type == LT_FULL) {
            /* QK-norm per head (Qwen3). Llama-family checkpoints (e.g. MiniCPM5,
             * LlamaForCausalLM) have none: qn/kn stay NULL and are skipped. */
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.q_norm.weight",i);
            if (st_has(&m->S, nm)) {
                LDT(qn, "self_attn.q_norm.weight", hd);  /* per testa, NON per hidden */
                LDT(kn, "self_attn.k_norm.weight", hd);
            }
            /* Qwen3.5: q_proj raddoppiato = [query|gate]. Rilevato dalla forma. */
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.q_proj.weight",i);
            l->gated = (st_numel(&m->S, nm) == (int64_t)2*H*hd*D);
        } else {
            int kd = c->lin_hk*c->lin_dk, vd = c->lin_hv*c->lin_dv;
            int cd = 2*kd + vd, K = c->lin_conv;
            LDT(conv_w,  "linear_attn.conv1d.weight", (int64_t)cd*K);   /* [cd,1,K] depthwise */
            snprintf(nm,sizeof(nm),"model.layers.%d.linear_attn.conv1d.bias",i);
            l->conv_b = st_has(&m->S, nm) ? load_t(m, nm, cd) : NULL;
            LDT(dt_bias, "linear_attn.dt_bias", c->lin_hv);
            LDT(A_log,   "linear_attn.A_log",   c->lin_hv);
            LDT(dn_norm, "linear_attn.norm.weight", c->lin_dv);
            l->conv_state = falloc((int64_t)cd*K);
            l->Sstate = falloc((int64_t)c->lin_hv*c->lin_dk*c->lin_dv);
        }
        #undef LDT
    }
}

/* parte fissa del budget specifica del motore: KV-cache dei layer full */
static int64_t fixed_bytes(Model *m, int ctx) {
    Cfg *c = &m->c;
    int nfull = 0; for (int i = 0; i < c->n_layers; i++) if (c->ltype[i] == LT_FULL) nfull++;
    int64_t rows = (int64_t)nfull * 2 * c->n_kv_heads * ctx;
    return g_kv_bits == 8 ? rows*c->head_dim + rows*4      /* int8 + scala per riga */
                          : rows*c->head_dim*4;
}

#include "qwen_q41_stream.h"

/* azzera gli stati ricorrenti dei layer lineari (inizio generazione / reset
 * contesto) e lo stato TTA: l'adattamento non sopravvive al reset. */
static void state_reset(Model *m) {
    Cfg *c = &m->c;
    tta_reset();
    if (!m->L || !c->hybrid) return;
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (l->type != LT_LINEAR) continue;
        int cd = 2*c->lin_hk*c->lin_dk + c->lin_hv*c->lin_dv;
        memset(l->conv_state, 0, (int64_t)cd*c->lin_conv*sizeof(float));
        memset(l->Sstate, 0, (int64_t)c->lin_hv*c->lin_dk*c->lin_dv*sizeof(float));
    }
}

/* attenzione GQA sui token nuovi x[S,hidden]; pos_base = posizione del primo token nuovo.
 * Con l->gated (Qwen3.5): q_proj emette [query|gate] per testa; il contesto viene
 * moltiplicato per sigmoid(gate) prima di o_proj. */
/* scores + accumulation for (head, token) pairs [r0, r1), r = head*S + token;
 * per-thread score row from att_sc (kv_alloc): no malloc in the hot region */
typedef struct { Model *m; const float *q; float *ctx; int S, G, hd, layer, pos_base, kv8; int64_t qw; float scale; } QAttJob;
static void qwen_att_part(void *c_, int64_t r0, int64_t r1, int tid) {
    const QAttJob *c = c_; Model *m = c->m; int hd = c->hd, layer = c->layer;
    float *sc = m->base.att_sc + (int64_t)tid * m->base.max_t;
    for (int64_t r = r0; r < r1; r++) {
        int hh = (int)(r / c->S), s = (int)(r % c->S);
        int kvh = hh / c->G;                  /* GQA: testa kv condivisa */
        int qpos = c->pos_base + s;
        const float *qv = c->q + s*c->qw + (int64_t)hh*hd;
        int64_t kvbase = (int64_t)kvh * m->base.max_t;
        int pp = m->pp_attn;
        if (pp && hd == 64 && !m->kv4 && !c->kv8 && r1-r >= pp && c->S-s >= pp) {
            float *scores[4] = {sc};
            int64_t extra = (int64_t)tid*(pp-1)*m->base.max_t;
            for (int j = 1; j < pp; j++) scores[j] = m->pp_scores+extra+(int64_t)(j-1)*m->base.max_t;
            moty_attn_prefill_group(c->ctx+s*c->qw+(int64_t)hh*hd,(int)c->qw,qv,scores,
                                    m->base.K[layer]+kvbase*hd,m->base.V[layer]+kvbase*hd,
                                    qpos+1,pp,c->scale);
            r += pp-1;
            continue;
        }
        if (c->kv8) att_scores_i8(sc, qv, m->base.K8[layer], m->base.Ks[layer], kvbase, 0, qpos, hd, c->scale);
        else if (m->kv4) {
            int64_t stride = ((int64_t)m->base.max_t+3)&~INT64_C(3);
            const float *key = m->base.K[layer]+kvh*stride*hd;
            if (m->kv4_scores && r+1 < r1 && s+1 < c->S) {
                float *sc1 = m->kv4_scores+(int64_t)tid*m->base.max_t;
                moty_att_scores_kv4_pair(sc,sc1,qv,qv+c->qw,key,qpos+1,c->scale);
                softmax_row(sc1,qpos+2);
                att_accum_f32(c->ctx+(s+1)*c->qw+(int64_t)hh*hd,sc1,m->base.V[layer],kvbase,0,qpos+1,hd);
                r++;
            } else moty_att_scores_kv4(sc,qv,key,qpos+1,c->scale);
        } else att_scores_f32(sc, qv, m->base.K[layer], kvbase, 0, qpos, hd, c->scale);
        softmax_row(sc, qpos+1);
        float *cx = c->ctx + s*c->qw + (int64_t)hh*hd;
        if (c->kv8) att_accum_i8(cx, sc, m->base.V8[layer], m->base.Vs[layer], kvbase, 0, qpos, hd);
        else        att_accum_f32(cx, sc, m->base.V[layer], kvbase, 0, qpos, hd);
    }
}

static float *qwen_rows_alloc(Scratch *work, int64_t n) {
    return work ? scr_take(work,n*sizeof(float)) : falloc(n);
}

/* A layer reuses the same frame for attention and FFN. The caller reserves
 * the whole frame before taking any pointers, including the residuals. */
static void qwen_rows_release(Scratch *work, int64_t mark, float *a, float *b,
                              float *c, float *d) {
    if (work) work->used = mark;
    else { free(a); free(b); free(c); free(d); }
}

typedef struct {
    Model *m;
    const Layer *layer;
    float *q, *k;
    const float *v, *rope;
    int pos, li;
} QwenPrefillKeys;

static void qwen_prefill_keys(void *arg, int64_t first, int64_t end, int tid) {
    (void)tid;
    const QwenPrefillKeys *j = arg;
    const Model *m = j->m;
    const Cfg *c = &m->c;
    const int hd = c->head_dim, heads = c->n_heads, kv = c->n_kv_heads;
    for (int64_t s = first; s < end; s++) {
        const float *rope = j->rope+s*c->rot;
        for (int h = 0; h < heads; h++) {
            float *row = j->q+(s*heads+h)*hd;
            if (j->layer->qn) rmsnorm_row(row,row,j->layer->qn,hd,c->eps);
            moty_rope_apply_row(row,c->rot,rope);
        }
        for (int h = 0; h < kv; h++) {
            float *row = j->k+(s*kv+h)*hd;
            if (j->layer->kn) rmsnorm_row(row,row,j->layer->kn,hd,c->eps);
            moty_rope_apply_row(row,c->rot,rope);
            int64_t slot = (int64_t)h*m->base.max_t+j->pos+s;
            memcpy(m->base.K[j->li]+slot*hd,row,hd*sizeof(float));
            memcpy(m->base.V[j->li]+slot*hd,j->v+(s*kv+h)*hd,hd*sizeof(float));
        }
    }
}

static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out, Scratch *work) {
    int64_t mark = work ? work->used : 0;
    Cfg *c = &m->c;
    int H = c->n_heads, KV = c->n_kv_heads, hd = c->head_dim;
    int G = H / KV;                       /* teste query per testa kv */
    int64_t qw = (int64_t)H*hd, kw = (int64_t)KV*hd;
    float *q, *gate = NULL;
    float *k = qwen_rows_alloc(work,S*kw), *vv = qwen_rows_alloc(work,S*kw);
    int joint_qkv = 0;
    OP_T(t_qkv);
    if (l->gated) {
        float *qg = falloc(S*2*qw);
        mat_apply(qg, x, &l->q, S);
        /* LoRA sul buffer fuso [query|gate] PRIMA dello split (O = l->q.O = 2*H*hd) */
        if (l->lo) lora_apply(&l->lo->q, qg, x, S, l->q.I, l->q.O);
        q = falloc(S*qw); gate = falloc(S*qw);
        for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) {
            memcpy(q    + s*qw + (int64_t)hh*hd, qg + s*2*qw + (int64_t)hh*2*hd,      hd*sizeof(float));
            memcpy(gate + s*qw + (int64_t)hh*hd, qg + s*2*qw + (int64_t)hh*2*hd + hd, hd*sizeof(float));
        }
        free(qg);
    } else {
        q = qwen_rows_alloc(work,S*qw);
        if (!l->qkv.q4) {
            float *const out_qkv[] = { q, k, vv };
            const Mat *const proj[] = { &l->q, &l->k, &l->v };
            joint_qkv = l->q.fmt == WF_Q4_1 && moty_q41_qkv(out_qkv, x, proj, S);
            if (!joint_qkv) mat_apply(q, x, &l->q, S);
        }
        if (l->lo) lora_apply(&l->lo->q, q, x, S, l->q.I, l->q.O);
    }
    if (l->qkv.q4) {                        /* fused [q|k|v] (qwen_fuse): one GEMV, split per token */
        float *qkvb = falloc((int64_t)S*(qw + 2*kw));
        mat_apply(qkvb, x, &l->qkv, S);
        for (int s = 0; s < S; s++) {
            const float *r = qkvb + (int64_t)s*(qw + 2*kw);
            memcpy(q + (int64_t)s*qw, r, qw*sizeof(float));
            memcpy(k + (int64_t)s*kw, r + qw, kw*sizeof(float));
            memcpy(vv + (int64_t)s*kw, r + qw + kw, kw*sizeof(float));
        }
        free(qkvb);
    } else if (!joint_qkv) {
        mat_apply(k,  x, &l->k, S);
        mat_apply(vv, x, &l->v, S);
    }
    if (l->lo) {
        lora_apply(&l->lo->k, k,  x, S, l->k.I, l->k.O);
        lora_apply(&l->lo->v, vv, x, S, l->v.I, l->v.O);
    }
    OP_ACC(OP_QKV, t_qkv);
    /* qk-norm PER TESTA (Qwen3), poi RoPE per testa */
    OP_T(t_qk);
    int kv8 = m->base.K8[layer] != NULL;
    const float *rope = m->prefill_parallel && S >= 128 && !kv8 && !m->kv4
                      ? moty_rope_rows(c->theta,c->rot,pos_base,S) : NULL;
    if (rope) {
        QwenPrefillKeys job = {m,l,q,k,vv,rope,pos_base,layer};
        moty_par_for(S,0,qwen_prefill_keys,&job);
    }
    for (int s = 0; !rope && s < S; s++) {
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) {
            float *qh = q + s*qw + (int64_t)hh*hd;
            if (l->qn) rmsnorm_row(qh, qh, l->qn, hd, c->eps);
            rope_head(qh, pos, c->theta, c->rot);
        }
        for (int hh = 0; hh < KV; hh++) {
            float *kh = k + s*kw + (int64_t)hh*hd;
            if (l->kn) rmsnorm_row(kh, kh, l->kn, hd, c->eps);
            rope_head(kh, pos, c->theta, c->rot);
        }
    }
    OP_ACC(OP_QKNORM_ROPE, t_qk);
    /* scrive k,v nella kv-cache alle posizioni pos_base..pos_base+S-1 */
    OP_T(t_kv);
    for (int s = 0; !rope && s < S; s++) for (int hh = 0; hh < KV; hh++) {
        int t = pos_base + s;
        int64_t slot = (int64_t)hh*m->base.max_t + t;
        if (kv8) {
            kv_store_row(m->base.K8[layer] + slot*hd, &m->base.Ks[layer][slot], k + s*kw + (int64_t)hh*hd, hd);
            kv_store_row(m->base.V8[layer] + slot*hd, &m->base.Vs[layer][slot], vv + s*kw + (int64_t)hh*hd, hd);
        } else {
            const float *src = k + s*kw + (int64_t)hh*hd;
            if (m->kv4) {
                int64_t stride = ((int64_t)m->base.max_t+3)&~INT64_C(3);
                float *dst = m->base.K[layer]+(hh*stride+(t&~3))*hd;
                int lane = t%4;
                if (!lane) memset(dst,0,4*hd*sizeof(float));
                for (int d = 0; d < hd; d++) dst[4*d+lane] = src[d];
            } else memcpy(m->base.K[layer]+slot*hd,src,hd*sizeof(float));
            memcpy(m->base.V[layer] + slot*hd, vv + s*kw + (int64_t)hh*hd, hd*sizeof(float));
        }
    }
    OP_ACC(OP_KV_STORE, t_kv);
    OP_T(t_at);
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = qwen_rows_alloc(work,S*qw);
    QAttJob aj = { m, q, ctx, S, G, hd, layer, pos_base, kv8, qw, scale };
    int block_decode = hd == 64 && G == 7 && !kv8 && m->base.lm_head.fmt == WF_Q4_1;
    if (m->exact_batch && S > 1) {
        int batched = block_decode && !m->kv4 &&
                     moty_attn_gqa64_decode_batch(ctx,q,m->base.K[layer],m->base.V[layer],KV,
                                                  m->base.max_t,pos_base+1,S,scale,&m->base.bscr);
        for (int s = 0; !batched && s < S; s++) {
            const float *query = q+(int64_t)s*qw;
            float *result = ctx+(int64_t)s*qw;
            if (block_decode &&
                moty_attn_gqa64_layout(result,query,m->base.K[layer],m->base.V[layer],KV,
                                      m->base.max_t,pos_base+s+1,scale,&m->base.bscr,m->kv4)) continue;
            QAttJob single = { m,query,result,1,G,hd,layer,pos_base+s,kv8,qw,scale };
            moty_par_for(H,0,qwen_att_part,&single);
        }
    } else if (S != 1 || !block_decode ||
               !moty_attn_gqa64_layout(ctx,q,m->base.K[layer],m->base.V[layer],KV,
                                     m->base.max_t,pos_base+1,scale,&m->base.bscr,m->kv4))
        moty_par_for((int64_t)H*S, 0, qwen_att_part, &aj);
    if (gate) {                            /* Qwen3.5: gating per-elemento sull'output */
        for (int64_t i = 0; i < S*qw; i++) ctx[i] *= 1.f/(1.f + expf(-gate[i]));
        free(gate);
    }
    OP_ACC(OP_ATTN_CORE, t_at);
    OP_T(t_o);
    mat_apply(out, ctx, &l->o, S);
    if (l->lo) lora_apply(&l->lo->o, out, ctx, S, l->o.I, l->o.O);
    OP_ACC(OP_O_PROJ, t_o);
    qwen_rows_release(work,mark,q,k,vv,ctx);
}

/* ---------- Gated DeltaNet (Qwen3.5, layer linear_attention) ----------
 * Ricorrenza per token (fp32), stato per layer:
 *   conv_state [cd][K]  finestra scorrevole della conv1d causale depthwise
 *   S [hv][dk][dv]      stato ricorrente (sostituisce la KV-cache: NON cresce)
 * Formule (transformers, torch_recurrent_gated_delta_rule + modular_qwen3_5):
 *   q,k l2-normalizzate per testa; q *= dk^-0.5
 *   g = -exp(A_log)*softplus(a+dt_bias); beta = sigmoid(b)
 *   S *= exp(g); kv = k.S; delta = (v-kv)*beta; S += k(x)delta; o = q.S
 *   out = rmsnorm(o)*w * silu(z) per testa -> out_proj */

/* SwiGLU denso: out[S,D] = down( silu(gate(x)) * up(x) ), in batch sugli S
 * token: ogni riga di peso viene letta una volta sola per l'intero batch */
static void mlp(Model *m, Layer *l, float *x, int S, float *out, Scratch *work,
                int pending_down) {
    int64_t mark = work ? work->used : 0;
    Cfg *c = &m->c; int I = c->inter;
    if ((!l->lo || (!l->lo->gate.r && !l->lo->up.r)) && l->gate.fmt == WF_Q4_1 && l->up.fmt == WF_Q4_1 &&
        !l->gate.sh && !l->up.sh && l->gate.q4 && l->up.q4 && I % 4 == 0 &&
        ((l->gate.gs == Q41_PACKED_GS && l->up.gs == Q41_PACKED_GS) ||
         (l->gate.gs != Q41_PACKED_GS && l->up.gs != Q41_PACKED_GS && moty_q41_ffn_enabled()))) {
        float *g = qwen_rows_alloc(work,(int64_t)S*I);
        OP_T(t0);
        if (pending_down < 0 && S == 1 && !l->lo && moty_q41_ffn_team(out,g,x,&l->gate,&l->up,&l->down)) {
            OP_ACC(OP_FFN_GATE_UP, t0);  /* this measurement includes down */
            qwen_rows_release(work,mark,g,NULL,NULL,NULL);
            return;
        }
        if (l->gate.gs == Q41_PACKED_GS)
            moty_q41_panels(g,x,l->gate.q4,l->up.q4,S,l->gate.I,I);
        else moty_q41_gate_up(g,x,l->gate.q4,l->up.q4,S,l->gate.I,I,l->gate.gs == 64 && l->up.gs == 64);
        OP_ACC(OP_FFN_GATE_UP, t0);
        qwen_q41_finish_down(m,pending_down);
        OP_T(t1);
        mat_apply(out, g, &l->down, S);
        if (l->lo) lora_apply(&l->lo->down, out, g, S, l->down.I, l->down.O);
        OP_ACC(OP_FFN_DOWN, t1);
        qwen_rows_release(work,mark,g,NULL,NULL,NULL);
        return;
    }
    if (l->gate_up.q4) {                    /* fused [gate|up] -> shared dense FFN layer */
        qwen_q41_finish_down(m,pending_down);
        MotyFfnView f = { .gate=&l->gate, .up=&l->up, .down=&l->down, .gate_up=&l->gate_up,
                          .inter=I, .scr=&m->base.scr };
        moty_nn_dense_ffn(&f, x, S, out);
        return;
    }
    float *g = qwen_rows_alloc(work,(int64_t)S*I), *u = qwen_rows_alloc(work,(int64_t)S*I);
    OP_T(t0);
    mat_apply(g, x, &l->gate, S);
    mat_apply(u, x, &l->up,   S);
    if (l->lo) {
        lora_apply(&l->lo->gate, g, x, S, l->gate.I, l->gate.O);
        lora_apply(&l->lo->up,   u, x, S, l->up.I,   l->up.O);
    }
    OP_ACC(OP_FFN_GATE_UP, t0);
    OP_T(t1);
    moty_hw_silu_mul(g, u, (int64_t)S*I);
    OP_ACC(OP_FFN_SILU, t1);
    qwen_q41_finish_down(m,pending_down);
    OP_T(t2);
    mat_apply(out, g, &l->down, S);
    if (l->lo) lora_apply(&l->lo->down, out, g, S, l->down.I, l->down.O);  /* input = g fuso */
    OP_ACC(OP_FFN_DOWN, t2);
    qwen_rows_release(work,mark,g,u,NULL,NULL);
}

typedef struct {
    float *out;
    const float *x, *weights;
    int dim;
    float eps;
} QwenNormRows;

static void qwen_norm_worker(void *arg, int64_t first, int64_t end, int tid) {
    (void)tid;
    const QwenNormRows *j = arg;
    for (int64_t s = first; s < end; s++)
        rmsnorm_row(j->out+s*j->dim,j->x+s*j->dim,j->weights,j->dim,j->eps);
}

static void qwen_norm_rows(const Model *m, float *out, const float *x, const float *weights, int S) {
    int dim = m->c.hidden;
    if (m->prefill_parallel && S >= 128) {
        QwenNormRows job = {out,x,weights,dim,m->c.eps};
        moty_par_for(S,0,qwen_norm_worker,&job);
    } else {
        for (int s = 0; s < S; s++)
            rmsnorm_row(out+(int64_t)s*dim,x+(int64_t)s*dim,weights,dim,m->c.eps);
    }
}

static void qwen_attention_step(Model *m, Layer *l, int layer, float *x, int S,
                                int pos, float *nrm, float *tmp, Scratch *work) {
    Cfg *c = &m->c;
    int D = c->hidden;
    OP_T(t_n1);
    qwen_norm_rows(m,nrm,x,l->in_ln,S);
    OP_ACC(OP_NORM, t_n1);
    if (l->type == LT_LINEAR) deltanet(m,l,nrm,S,tmp);
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
    else if (S > 1 && m->q41_attention_raw) {
        /* Keep the decode panels resident and use one raw window across
         * prefill layers. The temporary view shares the original KV. */
        Layer view = *l;
        Mat *proj[] = {&view.q,&view.k,&view.v,&view.o};
        uint8_t *dst = m->q41_attention_raw;
        QwenQ41Batch *batch = m->q41_queue ? &m->q41_queue->attention : NULL;
        if (batch) dispatch_group_wait(batch->ready,DISPATCH_TIME_FOREVER);
        int job_index = 0;
        for (int j = 0; j < 4; j++) {
            const st_tensor *t = m->q41_attention[layer][j];
            if (!t) continue;
            int err = batch ? batch->jobs[job_index++].error : qwen_q41_read_bytes(t->fd,dst,t->off,t->nbytes);
            if (err) moty_fail_code(MOTY_FAIL_IO,"Cannot read native Q4_1 tensor %s: %s",t->name,strerror(err));
            proj[j]->q4 = dst;
            proj[j]->gs = m->q41_attention_group64;
            dst += scr_al(t->nbytes);
        }
        attention(m,&view,layer,nrm,S,pos,tmp,work);
    }
#endif
    else attention(m,l,layer,nrm,S,pos,tmp,work);
    OP_T(t_r1);
    moty_hw_add(x,tmp,(int64_t)S*D);
    OP_ACC(OP_RESID, t_r1);
}

static void qwen_ffn_step(Model *m, Layer *l, float *x, int S,
                          float *nrm, float *tmp, Scratch *work, int pending_down) {
    Cfg *c = &m->c;
    int D = c->hidden;
    OP_T(t_n2);
    qwen_norm_rows(m,nrm,x,l->post_ln,S);
    OP_ACC(OP_NORM, t_n2);
    mlp(m,l,nrm,S,tmp,work,pending_down);
    OP_T(t_r2);
    moty_hw_add(x,tmp,(int64_t)S*D);
    OP_ACC(OP_RESID, t_r2);
}

static int64_t qwen_prefill_bytes(const Cfg *c, int tokens) {
    int64_t hidden = scr_al((int64_t)tokens*c->hidden*sizeof(float));
    int64_t queries = scr_al((int64_t)tokens*c->n_heads*c->head_dim*sizeof(float));
    int64_t kv = scr_al((int64_t)tokens*c->n_kv_heads*c->head_dim*sizeof(float));
    int64_t attn = 2*(queries+kv);
    int64_t ffn = 2*scr_al((int64_t)tokens*c->inter*sizeof(float));
    return 2*hidden+(attn > ffn ? attn : ffn);
}

static void qwen_step_release(Scratch *work, Scratch *shared, float *x, float *nrm, float *tmp) {
    if (work) {
        if (work != shared) scr_free(work);
    }
    else { free(x); free(nrm); free(tmp); }
}

/* un passo: token nuovi ids[S] a posizione pos_base. Ritorna logits dell'ultimo token (malloc'd). */
static float *qwen_step_batch(Model *m, const int *ids, int S, int pos_base, Scratch *shared) {
    Cfg *c = &m->c; int D = c->hidden;
    Scratch arena = {0};
    Scratch *work = S > 1 && m->prefill_workspace ? (shared ? shared : &arena) : NULL;
    if (work) {
        scr_reset(work);
        scr_reserve(work,qwen_prefill_bytes(c,S));
    }
    float *x = qwen_rows_alloc(work,(int64_t)S*D);
    OP_T(t_e);
    for (int s = 0; s < S; s++) embed_row(m, ids[s], 1.f, x + (int64_t)s*D);
    OP_ACC(OP_EMBED, t_e);
    float *nrm = qwen_rows_alloc(work,(int64_t)S*D);
    /* The dense Pocket workspace consumes the normalized input in Q/K/V
     * or gate/up before the output projection overwrites it. */
    float *tmp = work ? nrm : qwen_rows_alloc(NULL,(int64_t)S*D);
    int strm = m->base.stream_buf != NULL || m->base.stream_q != NULL;
    if (strm && m->base.n_resident < c->n_layers) layer_prefetch(m,m->base.n_resident);
#ifdef __APPLE__
    QwenQ41Queue *queue = qwen_q41_begin_batch(m,S);
#endif
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (strm && i >= m->base.n_resident) {
            layer_stream_in(m,i);
            if (i + 1 < c->n_layers && i + 1 >= m->base.n_resident) layer_prefetch(m,i+1);
        }
        qwen_attention_step(m,l,i,x,S,pos_base,nrm,tmp,work);
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
        if (queue && S > 1 && m->q41_attention_raw && i+1 < c->n_layers)
            qwen_q41_attention_prefetch(m,i+1);
#endif
        const int cold = m->q41_window && i >= m->base.n_resident;
        int pending_down = cold ? qwen_q41_load_ffn(m,i,S) : -1;
        qwen_ffn_step(m,l,x,S,nrm,tmp,work,pending_down);
#ifdef __APPLE__
        /* Reuse the window after all FFN workers have returned. */
        if (cold && queue) {
            int next = i+m->q41_window_count;
            if (next < c->n_layers) qwen_q41_stream_prefetch(m,next,S);
            else if (queue->read_ahead && S == 1 && !g_skip_logits && !g_all_logits) {
                /* This slot has finished its last FFN of the current token. */
                int slot = (i-m->base.n_resident)%m->q41_window_count;
                if (slot < queue->read_ahead) qwen_q41_stage_decode(m,slot);
            }
        }
#endif
    }
    m->base.kv_len = pos_base + S;
    /* blocco intermedio di un prefill a blocchi: KV/stati aggiornati, ma
     * niente final-norm/lm_head/stash TTA (validi solo per l'ultimo token) */
    if (g_skip_logits) { qwen_step_release(work,shared,x,nrm,tmp); return NULL; }
#ifdef __APPLE__
    if (queue && S > 1 && !g_all_logits)
        for (int j = 0; j < queue->read_ahead && m->base.n_resident+j < c->n_layers; j++)
            qwen_q41_stage_decode(m,j);
#endif
    OP_T(t_h);
    if (g_all_logits) {                /* lookahead verification: every position, the full head */
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, m->base.final_norm, D, c->eps);
        float *lg = falloc((int64_t)S * c->vocab);
        mat_apply(lg, nrm, &m->base.lm_head, S);
        OP_ACC(OP_LM_HEAD, t_h);
        qwen_step_release(work,shared,x,nrm,tmp);
        return lg;
    }
    float *last = nrm;  /* Reuse the normalization buffer after the final layer. */
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->base.final_norm, D, c->eps);
    if ((g_tta.mode == TTA_CACHE || g_tta.mode == TTA_LORA) && g_tta.alloc) {
        memcpy(g_tta.h_cur, last, D*sizeof(float)); /* stash per cache / adattatore online */
        g_tta.h_valid = 1;
    }
    float *logit = falloc(c->vocab);
    /* the shortlist ranks rows of the base head: off with an lm_head adapter */
    moty_nn_head_apply(logit, last, &m->base.lm_head, m->lm_lora.r ? NULL : m->base.head_sl);
    lora_apply(&m->lm_lora, logit, last, 1, m->base.lm_head.I, m->base.lm_head.O);
    OP_ACC(OP_LM_HEAD, t_h);
    qwen_step_release(work,shared,x,nrm,tmp);
    return logit;
}

/* Dense causal layers can be advanced by contiguous chunks. If any
 * projection rejects FP16 packing, discard the provisional suffix and
 * replay the original batch from the input IDs, preserving batch-wide
 * precision selection. Every replayed K/V row is overwritten in order. */
static float *step(Model *m, const int *ids, int S, int pos_base) {
    int chunk = m->auto_prefill_chunk;
    if (chunk < 4 || S < 1024 || S <= chunk || g_prefill_chunk > 0 ||
        g_all_logits || m->exact_batch || m->base.inj)
        return qwen_step_batch(m,ids,S,pos_base,NULL);
    Scratch shared = {0};
    int skip_last = g_skip_logits;
    float *logits = NULL;
    for (int start = 0; start < S;) {
        int count = S-start;
        if (count > chunk) count = count == chunk+1 ? chunk-1 : chunk;
        g_skip_logits = skip_last || start+count < S;
        moty_q41_begin_exact_batch();
        logits = qwen_step_batch(m,ids+start,count,pos_base+start,&shared);
        if (moty_q41_end_exact_batch()) {
            free(logits);
            scr_free(&shared);
            g_skip_logits = skip_last;
            m->base.kv_len = pos_base;
            return qwen_step_batch(m,ids,S,pos_base,NULL);
        }
        start += count;
    }
    g_skip_logits = skip_last;
    scr_free(&shared);
    return logits;
}

static int qwen_cli_verify_ok(Model *m) {
    return m->base.lm_head.fmt == WF_Q4_1 && m->c.head_dim == 64 && m->c.n_heads == 14 &&
           m->c.n_kv_heads == 2 && !m->kv4 && g_kv_bits != 8 && qwen_verify_ok(m);
}

static float *qwen_cli_verify(Model *m, const int *ids, int n, int pos) {
    m->exact_batch = 1;
    moty_q41_begin_exact_batch();
    float *lo = verify_batch(m,ids,n,pos);
    m->exact_batch = 0;
    if (moty_q41_end_exact_batch()) {
        free(lo);
        m->base.kv_len = pos;
        return NULL;
    }
#ifdef MOTY_Q41_VERIFY_CHECK
    if (lo) {
        for (int j = 0; j < n; j++) {
            float *single = step(m,ids+j,1,pos+j);
            if (memcmp(single,lo+(int64_t)j*m->c.vocab,m->c.vocab*sizeof(float)))
                moty_fail_code(MOTY_FAIL_FORMAT,"Q4_1 batch logits differ at position %d",pos+j);
            free(single);
        }
    }
#endif
    return lo;
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    kv_arrays_alloc(m, max_t);
    m->pp_attn = 0;
    m->pp_scores = NULL;
    m->prefill_workspace = 0;
    m->prefill_parallel = 0;
    m->auto_prefill_chunk = 0;
    if (g_gguf && gguf_find(&g_gguf_meta,"pocket.qat_conversion") &&
        m->base.lm_head.fmt == WF_Q4_1 && !c->hybrid &&
        (m->base.n_resident == c->n_layers || m->q41_window)) {
        int compatible = 1;
        for (int l = 0; l < c->n_layers; l++)
            if (m->L[l].gated || m->L[l].qkv.q4 || m->L[l].lo || m->L[l].type == LT_LINEAR) {
                compatible = 0;
                break;
            }
        if (compatible) {
            const char *work = moty_getenv("MOTY_Q41_WORKSPACE");
            const char *chunk = moty_getenv("MOTY_Q41_AUTO_CHUNK");
#if defined(__APPLE__) && defined(__ARM_NEON)
            int workspace_default = 1;
            int chunk_default = moty_q41_uses_fp16() && moty_par_threads() == 12 ? 256 : 0;
#else
            int workspace_default = 0;
            int chunk_default = 0;
#endif
            m->prefill_workspace = work ? !strcmp(work,"1") : workspace_default;
            const char *parallel = moty_getenv("MOTY_Q41_PREFILL_PAR");
            m->prefill_parallel = parallel && !strcmp(parallel,"1") && moty_q41_uses_fp16();
            m->auto_prefill_chunk = chunk ? atoi(chunk) : chunk_default;
        }
    }
#if defined(__aarch64__) && defined(__ARM_NEON)
    const char *kv4 = moty_getenv("MOTY_Q41_KV4");
    m->kv4 = kv4 && !strcmp(kv4,"1") && g_kv_bits != 8 && c->head_dim == 64 &&
             c->n_heads == 14 && c->n_kv_heads == 2 && m->base.lm_head.fmt == WF_Q4_1 && max_t <= INT_MAX-3;
    const char *pp = moty_getenv("MOTY_Q41_PP_ATTN");
    if (!m->kv4 && g_kv_bits != 8 && c->head_dim == 64 && m->base.lm_head.fmt == WF_Q4_1) {
        int pp_default = max_t >= 2048 && c->n_heads == 14 && c->n_kv_heads == 2 &&
                         moty_q41_uses_fp16() && moty_par_threads() == 12;
        m->pp_attn = pp ? (!strcmp(pp,"2") ? 2 : !strcmp(pp,"4") ? 4 : 0) : pp_default ? 4 : 0;
    }
    if (m->pp_attn)
        m->pp_scores = falloc((int64_t)(m->pp_attn-1)*moty_par_threads()*max_t);
#endif
    const char *pair = m->kv4 ? moty_getenv("MOTY_Q41_KV4_PREFILL_PAIR") : NULL;
    m->kv4_scores = pair && !strcmp(pair,"1") ? falloc((int64_t)moty_par_threads()*max_t) : NULL;
    int slots = m->kv4 ? (max_t+3)&~3 : max_t;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->ltype[i] == LT_LINEAR) continue;  /* i layer lineari usano lo stato, non la KV */
        kv_layer_alloc(m, i, c->n_kv_heads, c->head_dim, slots);
    }
    state_reset(m);
}

/* THINK=0 pre-closes the reasoning block for Qwen3/MiniCPM5. Some models
 * (MNN-Pocket) retain <think> in the vocabulary but omit it from ChatML. */
static int build_turn(Tok *T, char *buf, int cap, const char *user) {
    int think = moty_getenv("THINK") ? atoi(moty_getenv("THINK")) : 0;
    int bl = snprintf(buf, cap, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user);
    if (!think && !T->chat_omits_think && tok_id_of(T, "<think>") >= 0 && bl >= 0 && bl < cap)
        bl += snprintf(buf+bl, cap-bl, "<think>\n\n</think>\n\n");
    return bl;
}

/* semina gli stop token del template ChatML */
static void stops_seed(Model *m, Tok *T) {
    (void)m;
    stop_add(tok_id_of(T, "<|im_end|>"));
    stop_add(tok_id_of(T, "<|endoftext|>"));
}

static void banner(Model *m) {
    int nlin = 0; for (int i = 0; i < m->c.n_layers; i++) nlin += (m->c.ltype[i] == LT_LINEAR);
    fprintf(stderr, "[qwen] %d layer (%d deltanet), hidden %d, %d/%d teste (hd %d, rot %d), inter %d, vocab %d%s | load %.1fs | RSS %.2f GB | idot %s | f32 %s\n",
            m->c.n_layers, nlin, m->c.hidden, m->c.n_heads, m->c.n_kv_heads, m->c.head_dim, m->c.rot, m->c.inter, m->c.vocab,
            m->base.lm_tied ? " | lm_head=embed" : "", m->base.load_s, rss_gb(), IDOT_KERNEL, F32_KERNEL);
}

#include "qwen_train.h"

#if !defined(QWEN_TEST) && !defined(MOTY_NO_MAIN)   /* MOTY_NO_MAIN: libmoty */
int main(int argc, char **argv) {
    /* TRAIN=<corpus.txt> -> fine-tuning LoRA (qwen_train.h) invece della generazione */
    if (getenv("TRAIN") && *getenv("TRAIN")) return train_main(argc, argv);
    return engine_main(argc, argv);
}
#endif /* QWEN_TEST */
