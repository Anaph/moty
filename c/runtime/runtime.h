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
static int build_turn(char *buf, int cap, const char *user);
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
static const char *g_gguf = NULL;
static GgufMeta g_gguf_meta;

/* ---------- config: range check ---------- */
#define CKR(name, v, lo, hi) do { long _v=(long)(v); if(_v<(lo)||_v>(hi)){ \
    fprintf(stderr,"config.json: %s=%ld fuori range [%ld,%ld]\n",name,_v,(long)(lo),(long)(hi)); exit(1);} } while(0)

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
    if (c->n_heads % c->n_kv_heads) { fprintf(stderr,"config: n_heads %% n_kv_heads != 0\n"); exit(1); }
}

#include "runtime/rt_model_load.h"
#include "runtime/rt_kv_cache.h"
#include "runtime/rt_gen_loop.h"
#include "runtime/rt_env_cfg.h"

/* ---------- main condiviso ---------- */
static int engine_main(int argc, char **argv) {
    (void)argc;
    omp_hot_tune(argv);
    /* THREADS: tetto sul team OpenMP (batte OMP_NUM_THREADS), applicato PRIMA
     * di qualunque allocazione dipendente dal numero di thread. */
    const char *th_ = getenv("THREADS");
    if (th_ && atoi(th_) > 0) omp_set_num_threads(atoi(th_));
    RunEnv e;
    if (!parse_env(&e)) return 1;
    const char *snap = e.snap;
    int ngen = e.ngen, maxctx = e.maxctx, templ = e.templ;

    Model m;
    model_init_ex(&m, snap, e.qbits, e.budget, maxctx);
    banner(&m);
    /* banner precede il ramo REF: l'hook gira UNA volta per entrambi i percorsi */
    ENGINE_POST_INIT(&m);
    if (m.c.max_pos > 0 && maxctx > m.c.max_pos) maxctx = m.c.max_pos;

    const char *refpath = getenv("REF");
    if (refpath) return run_ref(&m, refpath);

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

    const char *prompt = getenv("PROMPT");
    if (prompt) {                                   /* one-shot */
        int bl = templ ? build_turn(buf, 1<<16, prompt)
                       : snprintf(buf, 1<<16, "%s", prompt);
        int k = tok_encode(&T, buf, bl, hist, maxctx - 2);
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
        int bl = templ ? build_turn(buf, 1<<16, line)
                       : snprintf(buf, 1<<16, "%s", line);
        int k = tok_encode(&T, buf, bl, hist + len, maxctx - len - 2);
        if (len + k + 8 >= maxctx) {                /* contesto pieno: reset conversazione */
            fprintf(stderr, "[" ENGINE_TAG "] contesto pieno, reset della conversazione\n");
            len = 0; m.base.kv_len = 0; state_reset(&m);  /* lo stato ricorrente non e' troncabile */
            k = tok_encode(&T, buf, bl, hist, maxctx - 2);
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

#endif /* RUNTIME_H */
