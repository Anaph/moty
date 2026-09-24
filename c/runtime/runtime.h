/* runtime.h — impalcatura dei motori (umbrella, P2).
 * Hook contract + GGUF/config + engine_main; la logica operativa vive nei
 * moduli rt_*.h inclusi qui (ordine = dipendenze):
 *   rt_model_load.h  pesi/budget/micro-RSS   rt_kv_cache.h  KV/embed_row
 *   rt_gen_loop.h    prefill/gen/ref          rt_env_cfg.h   env/threads
 *
 * Meccanismo a hook invariato: il motore definisce ENGINE_TAG/ENGINE_EOT e i
 * typedef Cfg/Layer/Model PRIMA di includere questo header, e implementa gli
 * hook DOPO. Tutto static, un'istanza per translation unit. */
#ifndef RUNTIME_H
#define RUNTIME_H

/* elenco delle MATRICI streamabili di un layer (unica fonte per loader,
 * streamer e prefetcher); riempito dall'hook layer_matrefs del motore. */
typedef struct { Mat *mat; char name[96]; int O, I; } MatRef;
#define MAX_LAYER_MATS 16

/* ---------- hook del motore (implementati dopo l'include) ---------- */
static void load_cfg(Cfg *c, const char *snap);
static void load_small(Model *m);
static int layer_matrefs(Model *m, int li, MatRef *r);
static int64_t fixed_bytes(Model *m, int ctx);
static float *step(Model *m, const int *ids, int S, int pos_base);
static void kv_alloc(Model *m, int max_t);
static void state_reset(Model *m);
static int build_turn(Tok *T, char *buf, int cap, const char *user);
static void stops_seed(Model *m, Tok *T);
static void banner(Model *m);

/* hook opzionali di strumentazione: no-op se il motore non li definisce */
#ifndef ENGINE_LOGITS_HOOK
#define ENGINE_LOGITS_HOOK(m, lo) ((void)0)
#endif
#ifndef ENGINE_OBSERVE
#define ENGINE_OBSERVE(m, tok) ((void)0)
#endif
/* chiamato una volta dopo model_init+banner (sia REF che generazione):
 * il motore puo' caricare stato extra (es. adattatori LoRA) */
#ifndef ENGINE_POST_INIT
#define ENGINE_POST_INIT(m) ((void)0)
#endif
/* il motore dichiara ENGINE_MICRO 1 se il suo step() sa girare senza embed
 * residente (gather per riga dal disco). Senza dichiarazione MICRO=1 fallisce
 * rumorosamente invece di crashare su embed NULL. */
#ifndef ENGINE_MICRO
#define ENGINE_MICRO 0
#endif

/* ---------- sorgente GGUF (GGUF=<file> al posto di SNAP=<dir>) ----------
 * gguf_index riempie lo stesso indice shards con nomi HF: da qui in poi il
 * runtime non distingue le due sorgenti, salvo config (sintetico dai
 * metadati) e tokenizer (tokenizer.ggml.*). */
#include "runtime/config.h"      /* M4: g_gguf/micro/... esportati dalla lib */
static GgufMeta g_gguf_meta;

/* ---------- config: range check ---------- */
#define CKR(name, v, lo, hi) do { long _v=(long)(v); if(_v<(lo)||_v>(hi)){ \
    moty_fail_code(MOTY_FAIL_FORMAT, "config.json: %s=%ld fuori range [%ld,%ld]\n",name,_v,(long)(lo),(long)(hi));} } while(0)

/* legge e parsa config.json; i rilasci multimodali annidano il config testo
 * sotto text_config. Ritorna l'oggetto config; *root_out e' la RADICE parsata
 * (puo' differire per il reparent text_config) e *buf_out il testo: il
 * chiamante li libera con json_free/free a parsing dei campi concluso. */
static jval *cfg_slurp(const char *snap, jval **root_out, char **buf_out) {
    char *buf;
    if (g_gguf) {
        buf = gguf_synth_config(&g_gguf_meta);      /* metadati -> JSON con chiavi HF */
    } else {
        char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
        buf = slurp_file(path, NULL);
    }
    jval *root = json_parse(buf);
    jval *r = root;
    jval *tc = json_get(root,"text_config"); if (tc && tc->t==J_OBJ) r = tc;
    *root_out = root; *buf_out = buf;
    return r;
}

/* campi del config parsati in modo identico da tutti i motori */
static void cfg_common(jval *r, Cfg *c) {
    c->hidden    = (int)json_get(r,"hidden_size")->num;
    c->n_layers  = (int)json_get(r,"num_hidden_layers")->num;
    c->n_heads   = (int)json_get(r,"num_attention_heads")->num;
    c->n_kv_heads= (int)json_get(r,"num_key_value_heads")->num;
    c->inter     = (int)json_get(r,"intermediate_size")->num;
    c->vocab     = (int)json_get(r,"vocab_size")->num;
    jval *hd = json_get(r,"head_dim");
    c->head_dim  = hd ? (int)hd->num : c->hidden / c->n_heads;
    jval *mp = json_get(r,"max_position_embeddings"); c->max_pos = mp ? (int)mp->num : 32768;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps = ep ? (float)ep->num : 1e-6f;
    c->n_eos = 0;
    jval *eo = json_get(r,"eos_token_id");
    if (eo) {
        if (eo->t==J_ARR) { for (int i=0;i<eo->len && c->n_eos<4;i++) c->eos[c->n_eos++]=(int)eo->kids[i]->num; }
        else c->eos[c->n_eos++]=(int)eo->num;
    }
    CKR("hidden_size",          c->hidden,     8, 65536);
    CKR("num_hidden_layers",    c->n_layers,   1, 256);
    CKR("num_attention_heads",  c->n_heads,    1, 256);
    CKR("num_key_value_heads",  c->n_kv_heads, 1, c->n_heads);
    CKR("head_dim",             c->head_dim,   2, 1024);
    CKR("intermediate_size",    c->inter,      8, 262144);
    CKR("vocab_size",           c->vocab,     16, 2000000);
    if (c->n_heads % c->n_kv_heads) { moty_fail_code(MOTY_FAIL_FORMAT, "config: n_heads %% n_kv_heads != 0\n"); }
}

#include "runtime/rt_model_load.h"
#include "runtime/rt_kv_cache.h"
#include "runtime/rt_gen_loop.h"
#include "runtime/rt_env_cfg.h"

#ifdef ENGINE_API_ID
static int cli_api_oneshot(void);
#endif

/* ---------- main condiviso ---------- */
static int engine_main(int argc, char **argv) {
    (void)argc;
    omp_hot_tune(argv);
#ifdef ENGINE_API_ID
    /* one-shot generation (PROMPT / PROMPT_IDS, optional EMBEDS) goes through
     * the public library API: the CLI is its first user and measures it.
     * Validation, perplexity, packing, streaming, GGUF and chat keep the
     * direct path below. */
    {
        const char *q = getenv("QBITS"); int qb = q ? atoi(q) : 0;
        int oneshot = (getenv("PROMPT") || (getenv("PROMPT_IDS") && *getenv("PROMPT_IDS")))
                      && getenv("SNAP") && !getenv("GGUF") && !getenv("REF") && !(getenv("PPL") && *getenv("PPL"))
                      && !(getenv("SAVE_PACKED") && *getenv("SAVE_PACKED")) && !getenv("MEM_GB") && !getenv("MEM_FRAC")
                      && !(getenv("MICRO") && atoi(getenv("MICRO")) > 0) && !getenv("TTA") && !getenv("Q8_TENSORS")
                      && !getenv("Q4FMT") && !getenv("QGROUP") && !getenv("EMBED_Q8") && (qb == 4 || qb == 8)
                      && !(getenv("MOTY_CLI_DIRECT") && atoi(getenv("MOTY_CLI_DIRECT")));
        if (oneshot) return cli_api_oneshot();
    }
#endif
    /* THREADS: tetto sul team OpenMP (batte OMP_NUM_THREADS), applicato PRIMA
     * di qualunque allocazione dipendente dal numero di thread. */
    const char *th_ = getenv("THREADS");
    if (th_ && atoi(th_) > 0) moty_par_set_threads(atoi(th_));
    RunEnv e;
    if (!parse_env(&e)) return 1;
    const char *snap = e.snap;
    int ngen = e.ngen, maxctx = e.maxctx, templ = e.templ;

    Model m;
    if (getenv("MMAP")) g_st_map = atoi(getenv("MMAP"));   /* 0 copy, 1 map pre-packed tensors (default), 2 + populate */
    model_init_ex(&m, snap, e.qbits, e.budget, maxctx);
    banner(&m);
    /* banner precede il ramo REF: l'hook gira UNA volta per entrambi i percorsi */
    ENGINE_POST_INIT(&m);
    /* after the fusion: the writer keeps each fused group's codes back to back */
    const char *savep = getenv("SAVE_PACKED");
    if (savep && *savep) { model_save_packed(&m, snap, savep); return 0; }
    if (m.c.max_pos > 0 && maxctx > m.c.max_pos) maxctx = m.c.max_pos;

    /* EMBEDS=<file>: raw little-endian f32 rows [N][hidden] that replace the
     * token embedding at every EMBEDS_TOKEN position of the prompt, in order
     * (e.g. projected image features of a vision-language model; default
     * token: the snapshot config's image_token_id) */
    const char *embp = getenv("EMBEDS");
    if (embp && *embp) {
        long nbytes = 0; char *eb = slurp_file(embp, &nbytes);
        int D = m.c.hidden;
        if (!eb || nbytes <= 0 || nbytes % ((long)D * 4)) {
            fprintf(stderr, "[" ENGINE_TAG "] EMBEDS %s: need N x %d little-endian f32 (%ld bytes)\n", embp, D, nbytes); return 1;
        }
        m.base.inj = (const float *)eb; m.base.inj_n = (int)(nbytes / ((long)D * 4)); m.base.inj_used = 0;
        m.base.inj_tok = getenv("EMBEDS_TOKEN") ? atoi(getenv("EMBEDS_TOKEN")) : -1;
        if (m.base.inj_tok < 0 && snap) {           /* multimodal config: image_token_id at the root */
            char cp[2048]; snprintf(cp, sizeof cp, "%s/config.json", snap);
            char *cb = slurp_file(cp, NULL); jval *cr = cb ? json_parse(cb) : NULL;
            jval *it = cr ? json_get(cr, "image_token_id") : NULL;
            if (it && it->t == J_NUM) m.base.inj_tok = (int)it->num;
            if (cr) json_free(cr); free(cb);
        }
        if (m.base.inj_tok < 0) { fprintf(stderr, "[" ENGINE_TAG "] EMBEDS: set EMBEDS_TOKEN=<placeholder id>\n"); return 1; }
        fprintf(stderr, "[" ENGINE_TAG "] EMBEDS: %d rows for token %d\n", m.base.inj_n, m.base.inj_tok);
    }
    const char *refpath = getenv("REF");
    if (refpath) return run_ref(&m, refpath);
    const char *pplpath = getenv("PPL");
    if (pplpath && *pplpath) return run_ppl(&m, pplpath);
    /* after REF/PPL: validation and perplexity always use the full head */
    static MotyHeadSL head_sl;
    if (g_head_topk > 0) {
        if (moty_nn_head_sl_build(&head_sl, &m.base.lm_head, g_head_topk)) {
            m.base.head_sl = &head_sl;
            fprintf(stderr, "[" ENGINE_TAG "] HEAD_TOPK=%d: 1-bit head %.1f MB\n", head_sl.K,
                    (double)head_sl.V * head_sl.D / 8 / 1048576.0);
        } else fprintf(stderr, "[" ENGINE_TAG "] HEAD_TOPK ignored: needs a Q4R4 lm_head (QBITS=4 Q4FMT=r4)\n");
    }

    Tok T;
    if (g_gguf) tok_load_gguf(&T, &g_gguf_meta);   /* single-file: vocab/merges dai metadati */
    else {
        char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
        tok_load(&T, tokpath);
    }
    stops_seed(&m, &T);
    for (int i = 0; i < m.c.n_eos; i++) stop_add(m.c.eos[i]);
    fprintf(stderr, "[" ENGINE_TAG "] stop tokens:"); for (int i=0;i<g_nstop;i++) fprintf(stderr," %d",g_stop[i]); fprintf(stderr,"\n");

    kv_alloc(&m, maxctx);
    int *hist = malloc(maxctx * sizeof(int));
    char *buf = malloc(1<<16);

    /* PROMPT_IDS=<json {"ids":[...]}>: a ready token sequence (e.g. a
     * processor's expansion of text + image placeholders), no template */
    const char *pids = getenv("PROMPT_IDS");
    if (pids && *pids) {
        char *jb = slurp_file(pids, NULL); jval *jr = jb ? json_parse(jb) : NULL;
        int k = 0; int *ids = jr ? read_int_array(jr, "ids", &k) : NULL;
        if (!ids || k <= 0 || k + 2 > maxctx) { fprintf(stderr, "[" ENGINE_TAG "] PROMPT_IDS %s: need {\"ids\":[...]} within CTX\n", pids); return 1; }
        memcpy(hist, ids, k * sizeof(int));
        int cur = ngen; if (k + cur + 1 > maxctx) cur = maxctx - k - 1;
        int stopped;
        gen_turn(&m, &T, hist, 0, k, cur, 1, &stopped);
        printf("\n");
        return 0;
    }
    const char *prompt = getenv("PROMPT");
    if (prompt) {                                   /* one-shot */
        int bl = templ ? build_turn(&T, buf, 1<<16, prompt)
                       : snprintf(buf, 1<<16, "%s", prompt);
        int k = 0;
        if (T.bos_id >= 0) hist[k++] = T.bos_id;    /* HF add_special_tokens=True */
        k += tok_encode(&T, buf, bl, hist + k, maxctx - 2 - k);
        int cur = ngen; if (k + cur + 1 > maxctx) cur = maxctx - k - 1;
        int stopped;
        gen_turn(&m, &T, hist, 0, k, cur, 1, &stopped);
        printf("\n");
        return 0;
    }

    /* chat interattiva: KV persistente, storia append-only */
    fprintf(stderr, "[" ENGINE_TAG "] chat interattiva: scrivi e premi invio (Ctrl-D per uscire)\n");
    int len = 0;                                    /* token gia' in KV */
    char *line = NULL; size_t lcap = 0;
    for (;;) {
        fprintf(stderr, "\n> "); fflush(stderr);
        ssize_t nr = getline(&line, &lcap, stdin);
        if (nr < 0) break;
        while (nr > 0 && (line[nr-1]=='\n' || line[nr-1]=='\r')) line[--nr]=0;
        if (!nr) continue;
        int bl = templ ? build_turn(&T, buf, 1<<16, line)
                       : snprintf(buf, 1<<16, "%s", line);
        int k = 0;
        if (len == 0 && T.bos_id >= 0) hist[k++] = T.bos_id;   /* BOS once per conversation */
        k += tok_encode(&T, buf, bl, hist + len + k, maxctx - len - 2 - k);
        if (len + k + 8 >= maxctx) {                /* contesto pieno: reset conversazione */
            fprintf(stderr, "[" ENGINE_TAG "] contesto pieno, reset della conversazione\n");
            len = 0; m.base.kv_len = 0; state_reset(&m);  /* lo stato ricorrente non e' troncabile */
            k = 0;
            if (T.bos_id >= 0) hist[k++] = T.bos_id;
            k += tok_encode(&T, buf, bl, hist + k, maxctx - 2 - k);
        }
        int cur = ngen; if (len + k + cur + 1 > maxctx) cur = maxctx - len - k - 1;
        int stopped;
        int ng = gen_turn(&m, &T, hist, len, k, cur, 1, &stopped);
        len += k + ng;
        /* chiude il blocco assistant nel transcript: i token del suffisso entrano
         * in KV col prefill del turno successivo */
        const char *suffix = stopped ? "\n" : ENGINE_EOT;
        len += tok_encode(&T, suffix, (int)strlen(suffix), hist + len, maxctx - len);
    }
    return 0;
}

/* ---------- library entry points (api/api.c, runtime/engine_api.h) ----------
 * An engine that defines ENGINE_API_ID (and ENGINE_API_TYPES, the config
 * model_type strings it serves) exports moty_engine_<ID>: the same hooks,
 * loader and tokenizer as engine_main, driven by the public API instead of
 * the environment. ENGINE_API_CHECK(m) may reject a config the library
 * does not serve (a failure message and MOTY_ERR_UNSUPPORTED). */
#ifdef ENGINE_API_ID
#include "runtime/engine_api.h"
#include "nn/head.h"
#ifndef ENGINE_API_CHECK
#define ENGINE_API_CHECK(m) NULL
#endif
typedef struct {
    Model m; Tok T; int has_tok;
    int stops[16], nstop;
    MotyHeadSL head_sl;
    int ctx, image_tok;
} EngInst;

static int eng_accepts(const char *type) {
    static const char *types[] = { ENGINE_API_TYPES };
    for (size_t i = 0; i < sizeof types / sizeof *types; i++) if (!strcmp(type, types[i])) return 1;
    return 0;
}
static void *eng_open(const char *dir, const MotyEngineOpen *o) {
    EngInst *e = calloc(1, sizeof *e);
    if (!e) moty_fail_code(MOTY_FAIL_OOM, "OOM: model instance");
    if (o->inst_out) *o->inst_out = e;
    g_gguf = NULL; g_micro = 0; g_qgroup = 32;
    g_kv_bits = o->kv_bits; g_embed_disk = o->embed_disk; g_head_topk = 0;
    g_q4fmt = o->q4fmt; g_q8_tensors = o->q8_tensors; g_prefill_chunk = 0; g_st_map = o->mmap;
    g_nstop = 0;                                   /* this TU's stop list: filled below, copied */
    model_init_ex(&e->m, dir, o->qbits, 0, o->ctx);
    const char *why = ENGINE_API_CHECK(&e->m);
    if (why) moty_fail_code(MOTY_FAIL_FORMAT, "%s", why);
    double tp = now_s();
    ENGINE_POST_INIT(&e->m);
    LP_MARK(&e->m, LP_FUSE, tp);
    e->ctx = o->ctx;
    if (e->m.c.max_pos > 0 && e->ctx > e->m.c.max_pos) e->ctx = e->m.c.max_pos;
    if (o->head_topk > 0 && moty_nn_head_sl_build(&e->head_sl, &e->m.base.lm_head, o->head_topk))
        e->m.base.head_sl = &e->head_sl;
    char p[2048]; snprintf(p, sizeof p, "%s/tokenizer.json", dir);
    FILE *tf = fopen(p, "rb");
    if (tf) { fclose(tf); tok_load(&e->T, p); e->has_tok = 1; stops_seed(&e->m, &e->T); }
    LP_MARK(&e->m, LP_TOK, tp);
    for (int i = 0; i < e->m.c.n_eos; i++) stop_add(e->m.c.eos[i]);
    for (int i = 0; i < g_nstop && e->nstop < 16; i++) e->stops[e->nstop++] = g_stop[i];
    g_nstop = 0;
    e->image_tok = -1;                             /* multimodal config: image_token_id at the root */
    snprintf(p, sizeof p, "%s/config.json", dir);
    char *cb = slurp_file(p, NULL); jval *cr = cb ? json_parse(cb) : NULL;
    jval *it = cr ? json_get(cr, "image_token_id") : NULL;
    if (it && it->t == J_NUM) e->image_tok = (int)it->num;
    if (cr) json_free(cr);
    free(cb);
    kv_alloc(&e->m, e->ctx);
    LP_MARK(&e->m, LP_KV, tp);
    if (o->log_level >= 2) {
        banner(&e->m);
        char lb[256]; int n = 0;
        for (int i = 0; i < LP_N; i++) n += snprintf(lb + n, sizeof lb - n, " %s %.2f", lp_name[i], e->m.base.load_ph[i]);
        fprintf(stderr, "[" ENGINE_TAG "] open phases (s):%s\n", lb);
    }
    return e;
}
static void eng_close(void *p) {
    EngInst *e = p;
    st_close_fds(&e->m.S);
    scr_free(&e->m.base.scr); scr_free(&e->m.base.bscr);
    if (e->m.base.head_sl) moty_nn_head_sl_free(&e->head_sl);
}
static int eng_vocab(void *p)  { return ((EngInst *)p)->m.c.vocab; }
static int eng_hidden(void *p) { return ((EngInst *)p)->m.c.hidden; }
static int eng_ctx(void *p)    { return ((EngInst *)p)->ctx; }
static int eng_image_token(void *p) { return ((EngInst *)p)->image_tok; }
static int eng_has_tok(void *p) { return ((EngInst *)p)->has_tok; }
static int eng_bos(void *p) { EngInst *e = p; return e->has_tok ? e->T.bos_id : -1; }
static int eng_is_stop(void *p, int t) {
    EngInst *e = p;
    for (int i = 0; i < e->nstop; i++) if (e->stops[i] == t) return 1;
    return 0;
}
static float *eng_prefill(void *p, const int *ids, int n, int pos, const float *rows, int n_rows, int inj_tok,
                          int chunk, const atomic_int *abort) {
    EngInst *e = p; Model *m = &e->m;
    m->base.inj = rows; m->base.inj_n = rows ? n_rows : 0; m->base.inj_used = 0; m->base.inj_tok = rows ? inj_tok : -1;
    float *lo = NULL; int done = 0;
    if (chunk <= 0) chunk = n;
    while (done < n) {
        int c = n - done < chunk ? n - done : chunk, last = done + c == n;
        g_skip_logits = !last;
        lo = step(m, ids + done, c, pos + done);
        g_skip_logits = 0;
        done += c;
        if (!last) { if (lo) free(lo); lo = NULL; if (atomic_load(abort)) break; }
    }
    m->base.inj = NULL; m->base.inj_n = 0; m->base.inj_tok = -1;
    return lo;
}
static float *eng_step(void *p, int tok, int pos) { return step(&((EngInst *)p)->m, &tok, 1, pos); }
static void *eng_scratch(void *p) { return &((EngInst *)p)->m.base.scr; }
static void eng_reset(void *p) { EngInst *e = p; e->m.base.kv_len = 0; e->m.base.inj_used = 0; state_reset(&e->m); }
static int eng_encode(void *p, const char *text, int add_bos, int chat, int *ids, int cap) {
    EngInst *e = p;
    if (!e->has_tok) return -1;
    int tl = (int)strlen(text), bcap = tl + 256;
    char *buf = malloc((size_t)bcap);
    if (!buf) moty_fail_code(MOTY_FAIL_OOM, "OOM: tokenize");
    int bl = chat ? build_turn(&e->T, buf, bcap, text) : snprintf(buf, (size_t)bcap, "%s", text);
    int k = 0;
    if (add_bos && e->T.bos_id >= 0 && cap > 0) ids[k++] = e->T.bos_id;
    k += tok_encode(&e->T, buf, bl, ids + k, cap - k);
    free(buf);
    return k;
}
static int eng_piece(void *p, int tok, char *buf, int cap) {
    EngInst *e = p;
    if (!e->has_tok || tok < 0 || tok >= e->T.n_ids) return 0;
    return tok_decode(&e->T, &tok, 1, buf, cap);
}
#define ENG_OPS_NAME_(id) moty_engine_##id
#define ENG_OPS_NAME(id) ENG_OPS_NAME_(id)
#define ENG_STR_(x) #x
#define ENG_STR(x) ENG_STR_(x)
const MotyEngineOps ENG_OPS_NAME(ENGINE_API_ID) = {
    ENG_STR(ENGINE_API_ID), eng_accepts, eng_open, eng_close, eng_vocab, eng_hidden, eng_ctx, eng_image_token,
    eng_has_tok, eng_bos, eng_is_stop, eng_prefill, eng_step, eng_scratch, eng_reset, eng_encode, eng_piece
};

/* ---------- the command-line one-shot through the public API ---------- */
#include "api/api_internal.h"
static int cli_env_int(const char *k, int d) { const char *v = getenv(k); return v && *v ? atoi(v) : d; }
typedef struct { int dump; } CliOut;
static int cli_on_token(void *u, int32_t tok, const char *piece, int n) {
    CliOut *o = u;
    if (o->dump) fprintf(stderr, "%d ", tok);
    if (n > 0) { fwrite(piece, 1, (size_t)n, stdout); fflush(stdout); }
    return 0;
}
static int cli_api_oneshot(void) {
    const char *snap = getenv("SNAP");
    moty_options o; moty_options_init(&o);
    o.threads = cli_env_int("THREADS", 0); o.threads_decode = cli_env_int("THREADS_DECODE", 0);
    o.ctx = cli_env_int("CTX", 4096); o.qbits = cli_env_int("QBITS", 4); o.kv_bits = cli_env_int("KV_BITS", 0);
    o.embed_disk = getenv("EMBED") && !strcmp(getenv("EMBED"), "disk");
    o.head_topk = cli_env_int("HEAD_TOPK", 0); o.prefill_chunk = cli_env_int("PREFILL_CHUNK", 0);
    o.mmap_weights = cli_env_int("MMAP", 1);
    o.log_level = 2;
    moty_model *h; char err[512];
    moty_status rc = moty_model_open_with(&ENG_OPS_NAME(ENGINE_API_ID), NULL, snap, &o, &h, err, sizeof err);
    if (rc) { fprintf(stderr, "[" ENGINE_TAG "] %s\n", err); return 1; }
    int ctx = moty_model_ctx(h), n = 0;
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)ctx);
    const char *pj = getenv("PROMPT_IDS");
    if (pj && *pj) {                                  /* a ready sequence, no template */
        char *jb = slurp_file(pj, NULL); jval *jr = jb ? json_parse(jb) : NULL;
        jval *a = jr ? json_get(jr, "ids") : NULL;
        if (!a || a->t != J_ARR || a->len <= 0 || a->len > ctx) { fprintf(stderr, "[" ENGINE_TAG "] PROMPT_IDS %s: need {\"ids\":[...]} within CTX\n", pj); return 1; }
        for (int i = 0; i < a->len; i++) ids[n++] = (int32_t)a->kids[i]->num;
        json_free(jr); free(jb);
    } else {
        n = moty_tokenize(h, getenv("PROMPT"), 1, cli_env_int("CHAT_TEMPLATE", 1), ids, ctx);
        if (n < 0) { fprintf(stderr, "[" ENGINE_TAG "] tokenize: %s\n", moty_last_error(h)); return 1; }
    }
    const char *ep = getenv("EMBEDS"); float *rows = NULL; int nrows = 0, etok = -1;
    if (ep && *ep) {
        long nb = 0; rows = (float *)slurp_file(ep, &nb);
        int D = moty_model_hidden(h);
        if (!rows || nb <= 0 || nb % ((long)D * 4)) { fprintf(stderr, "[" ENGINE_TAG "] EMBEDS %s: need N x %d little-endian f32\n", ep, D); return 1; }
        nrows = (int)(nb / ((long)D * 4));
        etok = getenv("EMBEDS_TOKEN") ? atoi(getenv("EMBEDS_TOKEN")) : moty_model_image_token(h);
        if (etok < 0) { fprintf(stderr, "[" ENGINE_TAG "] EMBEDS: set EMBEDS_TOKEN=<placeholder id>\n"); return 1; }
        fprintf(stderr, "[" ENGINE_TAG "] EMBEDS: %d rows for token %d\n", nrows, etok);
    }
    moty_sampling s; moty_sampling_init(&s);
    s.temperature = getenv("TEMP") ? (float)atof(getenv("TEMP")) : 0.7f;
    s.top_p = getenv("NUCLEUS") ? (float)atof(getenv("NUCLEUS")) : 0.95f;
    s.seed = getenv("SEED") ? strtoull(getenv("SEED"), NULL, 10) : 0;
    s.ignore_eos = cli_env_int("IGNORE_EOS", 0);
    s.max_new_tokens = cli_env_int("NGEN", 256);
    if (n + s.max_new_tokens > ctx) s.max_new_tokens = ctx - n;
    CliOut out = { cli_env_int("TOKENS", 0) };
    moty_stats st;
    rc = moty_generate(h, ids, n, rows, nrows, etok, &s, cli_on_token, &out, &st);
    if (out.dump) fprintf(stderr, "\n");
    printf("\n");
    if (rc) { fprintf(stderr, "[" ENGINE_TAG "] generate: %s\n", moty_last_error(h)); return 1; }
    fprintf(stderr, "\n[" ENGINE_TAG "] prefill %d tok in %.2fs (%.1f tok/s) | decode %d tok in %.2fs (%.2f tok/s) | RSS %.2f GB\n",
            st.prompt_tokens, st.prefill_s, st.prompt_tokens / (st.prefill_s > 1e-9 ? st.prefill_s : 1e-9),
            st.new_tokens, st.decode_s, st.new_tokens / (st.decode_s > 1e-9 ? st.decode_s : 1e-9), rss_gb());
    free(rows); free(ids);
    moty_model_close(h);
    return 0;
}
#endif /* ENGINE_API_ID */

#endif /* RUNTIME_H */
