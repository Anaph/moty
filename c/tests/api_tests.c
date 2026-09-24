/* The public library API (api/moty.h) on a synthetic LFM2 snapshot
 * (tiny_lfm2.h): lifecycle, determinism across handles, continuation,
 * reset, abort (from the callback and from another thread), embedding
 * injection, error paths. Plain C, 0 = ok / 1 = fail; gtest glue in
 * api_gtest.cc. Built also with -fsanitize=address / thread (docs/api.md). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include "api/moty.h"
#include "tiny_lfm2.h"
#include "tiny_llama.h"

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

static const int32_t ap_ids[6] = {1, 5, 9, 3, 17, 22};

typedef struct { int32_t toks[512]; int n; moty_model *h; int abort_at; int sleep_us; } Col;
static int col_cb(void *u, int32_t t, const char *piece, int n) {
    (void)piece; (void)n;
    Col *c = u;
    if (c->n < 512) c->toks[c->n] = t;
    __atomic_add_fetch(&c->n, 1, __ATOMIC_RELEASE);      /* read by the aborting thread */
    if (c->abort_at && c->n == c->abort_at) moty_abort(c->h);
    if (c->sleep_us) { struct timespec ts = {0, c->sleep_us * 1000L}; nanosleep(&ts, NULL); }
    return 0;
}

static moty_model *ap_open(const char *dir, int ctx) {
    moty_options o; moty_options_init(&o);
    o.threads = 2; o.ctx = ctx; o.log_level = 0;
    moty_model *h = NULL; char err[256];
    if (moty_model_open(dir, &o, &h, err, sizeof err) != MOTY_OK) { fprintf(stderr, "open: %s\n", err); return NULL; }
    return h;
}
static moty_status ap_gen(moty_model *h, const int32_t *ids, int n, int ntok, Col *c) {
    moty_sampling s; moty_sampling_init(&s);
    s.max_new_tokens = ntok; s.ignore_eos = 1;
    c->h = h;
    return moty_generate(h, ids, n, NULL, 0, -1, &s, col_cb, c, NULL);
}
static const char *ap_dir(void) {
    static char d[600];
    if (!d[0]) { snprintf(d, sizeof d, "%s", tst_dir("moty_api")); lt_write_dir(d); }
    return d;
}

/* open/generate/close twice in one process, two handles in sequence: the
 * same greedy tokens; the model info reflects the config */
int ap_lifecycle(void) {
    Col a = {0}, b = {0};
    for (int round = 0; round < 2; round++) {
        moty_model *h = ap_open(ap_dir(), 256);
        CHECK(h);
        CHECK(moty_model_vocab(h) == LV && moty_model_hidden(h) == LD && moty_model_ctx(h) == 256);
        CHECK(moty_model_image_token(h) == -1);
        Col *c = round ? &b : &a;
        moty_stats st;
        moty_sampling s; moty_sampling_init(&s); s.max_new_tokens = 12; s.ignore_eos = 1;
        c->h = h;
        CHECK(moty_generate(h, ap_ids, 6, NULL, 0, -1, &s, col_cb, c, &st) == MOTY_OK);
        CHECK(st.prompt_tokens == 6 && st.new_tokens == 12 && st.stop == MOTY_STOP_LENGTH);
        CHECK(moty_model_n_past(h) == 6 + 11);        /* the last token waits for the next call */
        moty_model_close(h);
    }
    CHECK(a.n == 12 && b.n == 12 && !memcmp(a.toks, b.toks, sizeof(int32_t) * 12));
    moty_release_scratch();
    return 0;
}

/* 4 tokens, then 8 more continuing (the pending token goes first) == 12 at once */
int ap_continuation(void) {
    moty_model *h = ap_open(ap_dir(), 256); CHECK(h);
    Col one = {0}, two = {0};
    CHECK(ap_gen(h, ap_ids, 6, 12, &one) == MOTY_OK);
    CHECK(moty_reset(h) == MOTY_OK && moty_model_n_past(h) == 0);
    CHECK(ap_gen(h, ap_ids, 6, 4, &two) == MOTY_OK);
    CHECK(ap_gen(h, NULL, 0, 8, &two) == MOTY_OK);
    CHECK(two.n == 12 && !memcmp(one.toks, two.toks, sizeof(int32_t) * 12));
    /* after a reset the same prompt gives the same answer again */
    Col three = {0};
    CHECK(moty_reset(h) == MOTY_OK);
    CHECK(ap_gen(h, ap_ids, 6, 12, &three) == MOTY_OK && !memcmp(one.toks, three.toks, sizeof(int32_t) * 12));
    moty_model_close(h);
    return 0;
}

/* abort from the callback: MOTY_ERR_ABORTED, conversation reset, handle usable */
int ap_abort_callback(void) {
    moty_model *h = ap_open(ap_dir(), 256); CHECK(h);
    Col c = {0}; c.abort_at = 3;
    moty_stats st;
    moty_sampling s; moty_sampling_init(&s); s.max_new_tokens = 50; s.ignore_eos = 1;
    c.h = h;
    CHECK(moty_generate(h, ap_ids, 6, NULL, 0, -1, &s, col_cb, &c, &st) == MOTY_ERR_ABORTED);
    CHECK(c.n == 3 && st.stop == MOTY_STOP_ABORT && moty_model_n_past(h) == 0);
    CHECK(strstr(moty_last_error(h), "aborted") != NULL);
    /* sticky: refused until cleared */
    Col d = {0};
    CHECK(ap_gen(h, ap_ids, 6, 5, &d) == MOTY_ERR_ABORTED && d.n == 0);
    moty_abort_clear(h);
    CHECK(ap_gen(h, ap_ids, 6, 5, &d) == MOTY_OK && d.n == 5);
    /* an abort that lands before the call starts; moty_reset clears it too */
    moty_abort(h);
    Col e = {0};
    CHECK(ap_gen(h, ap_ids, 6, 5, &e) == MOTY_ERR_ABORTED && e.n == 0);
    CHECK(moty_reset(h) == MOTY_OK);
    CHECK(ap_gen(h, ap_ids, 6, 5, &e) == MOTY_OK && e.n == 5);
    moty_model_close(h);
    return 0;
}

/* abort from another thread while generating */
typedef struct { moty_model *h; moty_status rc; Col c; } GenArg;
static void *gen_thread(void *p) {
    GenArg *g = p;
    moty_sampling s; moty_sampling_init(&s); s.max_new_tokens = 200; s.ignore_eos = 1;
    g->c.h = g->h; g->c.sleep_us = 2000;          /* ~0.4 s of generation */
    g->rc = moty_generate(g->h, ap_ids, 6, NULL, 0, -1, &s, col_cb, &g->c, NULL);
    return NULL;
}
int ap_abort_thread(void) {
    moty_model *h = ap_open(ap_dir(), 256); CHECK(h);
    GenArg g; memset(&g, 0, sizeof g); g.h = h;
    pthread_t th; CHECK(pthread_create(&th, NULL, gen_thread, &g) == 0);
    /* abort once the answer is streaming (a loaded machine may take a while
     * to prefill), then check it stops long before its 200 tokens */
    struct timespec ts = {0, 1000000L};
    for (int i = 0; i < 20000 && __atomic_load_n(&g.c.n, __ATOMIC_ACQUIRE) < 3; i++) nanosleep(&ts, NULL);
    moty_abort(h);
    pthread_join(th, NULL);
    CHECK(g.rc == MOTY_ERR_ABORTED && g.c.n > 0 && g.c.n < 200);
    moty_model_close(h);
    return 0;
}

/* LFM2-VL layout: rows injected at the image-token positions; other rows,
 * other answer; a row count that does not match the placeholders is
 * MOTY_ERR_ARG and the handle stays usable */
int ap_inject(void) {
    char dir[600]; snprintf(dir, sizeof dir, "%s", tst_dir("moty_api_vl")); lt_write_dir_w(dir, 1);
    moty_model *h = ap_open(dir, 256); CHECK(h);
    int it = moty_model_image_token(h); CHECK(it == 38);
    int32_t ids[8] = {1, 5, it, it, it, 9, 3, 17};
    float rows[3 * LD], rows2[3 * LD];
    for (int i = 0; i < 3 * LD; i++) { rows[i] = (float)((i * 37) % 11 - 5) * 0.1f; rows2[i] = -rows[i]; }
    moty_sampling s; moty_sampling_init(&s); s.max_new_tokens = 10; s.ignore_eos = 1;
    Col a = {0}, b = {0}, c = {0};
    a.h = b.h = c.h = h;
    CHECK(moty_generate(h, ids, 8, rows, 3, it, &s, col_cb, &a, NULL) == MOTY_OK);
    CHECK(moty_reset(h) == MOTY_OK);
    CHECK(moty_generate(h, ids, 8, rows, 3, it, &s, col_cb, &b, NULL) == MOTY_OK);
    CHECK(!memcmp(a.toks, b.toks, sizeof(int32_t) * 10));
    CHECK(moty_reset(h) == MOTY_OK);
    CHECK(moty_generate(h, ids, 8, rows2, 3, it, &s, col_cb, &c, NULL) == MOTY_OK);
    CHECK(memcmp(a.toks, c.toks, sizeof(int32_t) * 10) != 0);
    CHECK(moty_reset(h) == MOTY_OK);
    /* rows must match the placeholder count exactly: never read past them */
    CHECK(moty_generate(h, ids, 8, rows, 2, it, &s, NULL, NULL, NULL) == MOTY_ERR_ARG);
    CHECK(strstr(moty_last_error(h), "placeholder") != NULL && moty_model_n_past(h) == 0);
    CHECK(moty_generate(h, ids, 8, rows, 4, it, &s, NULL, NULL, NULL) == MOTY_ERR_ARG);
    Col d = {0}; d.h = h;
    CHECK(moty_generate(h, ids, 8, rows, 3, it, &s, col_cb, &d, NULL) == MOTY_OK);
    CHECK(!memcmp(a.toks, d.toks, sizeof(int32_t) * 10));
    moty_model_close(h);
    return 0;
}

/* errors come back as codes and messages, nothing exits, nothing leaks */
int ap_errors(void) {
    moty_model *h = (moty_model *)1; char err[256] = "";
    CHECK(moty_model_open("/nonexistent/moty", NULL, &h, err, sizeof err) == MOTY_ERR_IO && h == NULL && strstr(err, "config.json"));
    char dir[600]; snprintf(dir, sizeof dir, "%s", tst_dir("moty_api_bad"));
    tst_write_text(dir, "config.json", "{\"model_type\":\"mamba\"}");
    CHECK(moty_model_open(dir, NULL, &h, err, sizeof err) == MOTY_ERR_UNSUPPORTED && strstr(err, "mamba"));
    /* a layer's tensor missing from the file: the loader fails midway */
    lt_write_dir(dir);
    tst_write_text(dir, "config.json",
        "{\"architectures\":[\"Lfm2ForCausalLM\"],\"model_type\":\"lfm2\",\"hidden_size\":32,\"num_hidden_layers\":5,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"intermediate_size\":100,\"block_auto_adjust_ff_dim\":true,"
        "\"block_ffn_dim_multiplier\":1.0,\"block_multiple_of\":32,\"vocab_size\":40,\"norm_eps\":1e-05,\"conv_L_cache\":3,"
        "\"conv_bias\":false,\"rope_parameters\":{\"rope_theta\":10000.0},\"tie_embedding\":true,\"eos_token_id\":7,"
        "\"max_position_embeddings\":256,\"layer_types\":[\"conv\",\"full_attention\",\"conv\",\"full_attention\",\"conv\"]}");
    moty_options o; moty_options_init(&o); o.log_level = 0;
    CHECK(moty_model_open(dir, &o, &h, err, sizeof err) == MOTY_ERR_FORMAT && strstr(err, "missing tensor"));
    moty_options bad; moty_options_init(&bad); bad.size = 3;
    CHECK(moty_model_open(ap_dir(), &bad, &h, err, sizeof err) == MOTY_ERR_ARG);
    moty_options_init(&bad); bad.qbits = 3;
    CHECK(moty_model_open(ap_dir(), &bad, &h, err, sizeof err) == MOTY_ERR_ARG);
    moty_options_init(&bad); bad.mmap_weights = 3;
    CHECK(moty_model_open(ap_dir(), &bad, &h, err, sizeof err) == MOTY_ERR_ARG);
    /* a caller built against v1 passes the v1 size: fields added since keep their defaults */
    moty_options v1; moty_options_init(&v1); v1.log_level = 0; v1.size = MOTY_OPTIONS_V1_SIZE; v1.mmap_weights = 99;
    CHECK(moty_model_open(ap_dir(), &v1, &h, err, sizeof err) == MOTY_OK); moty_model_close(h);
    moty_options_init(&bad); bad.size = MOTY_OPTIONS_V1_SIZE + 2;
    CHECK(moty_model_open(ap_dir(), &bad, &h, err, sizeof err) == MOTY_ERR_ARG);
    h = ap_open(ap_dir(), 16); CHECK(h);
    int32_t big[40] = {0}; int32_t oob[2] = {1, LV};
    moty_sampling s; moty_sampling_init(&s); s.max_new_tokens = 8;
    CHECK(moty_generate(h, big, 10, NULL, 0, -1, &s, NULL, NULL, NULL) == MOTY_ERR_CONTEXT);
    CHECK(moty_generate(h, oob, 2, NULL, 0, -1, &s, NULL, NULL, NULL) == MOTY_ERR_ARG);
    CHECK(moty_generate(h, NULL, 0, NULL, 0, -1, &s, NULL, NULL, NULL) == MOTY_ERR_ARG);
    int32_t t[4];
    CHECK(moty_tokenize(h, "hi", 1, 0, t, 4) == MOTY_ERR_UNSUPPORTED);   /* no tokenizer.json */
    CHECK(moty_generate(NULL, big, 1, NULL, 0, -1, &s, NULL, NULL, NULL) == MOTY_ERR_ARG);
    moty_model_close(h);
    moty_model_close(NULL);
    CHECK(strstr(moty_version(), "moty") != NULL);
    moty_release_scratch();
    return 0;
}

/* the pool between requests: joined by moty_release_scratch, restarted by
 * the next call; open/generate/close cycles keep RSS flat (nothing kept) */
static long rss_kb(void) {
    FILE *f = fopen("/proc/self/statm", "r"); long pages = 0, res = 0;
    if (f) { if (fscanf(f, "%ld %ld", &pages, &res) != 2) res = 0; fclose(f); }
    return res * (sysconf(_SC_PAGESIZE) / 1024);
}
static int n_threads(void) {
    FILE *f = fopen("/proc/self/status", "r"); char l[256]; int n = -1;
    while (f && fgets(l, sizeof l, f)) if (sscanf(l, "Threads: %d", &n) == 1) break;
    if (f) fclose(f);
    return n;
}
/* joined threads can still be listed for a moment after pthread_join (the
 * kernel tears the task down asynchronously): wait for the count to settle */
static int n_threads_settled(void) {
    int n = n_threads(); struct timespec ts = {0, 5000000L};
    for (int i = 0; i < 200; i++) { nanosleep(&ts, NULL); int m = n_threads(); if (m == n) return n; n = m; }
    return n;
}
int ap_cycles(void) {
    Col ref = {0};
    long base = 0;
    moty_release_scratch();                           /* earlier tests in this process may have left workers */
    int t0 = n_threads_settled();
    int pool = strstr(moty_version(), "pthread pool") != NULL;   /* OpenMP keeps its own threads */
    for (int cycle = 0; cycle < 4; cycle++) {
        moty_options o; moty_options_init(&o); o.threads = 3; o.ctx = 256; o.log_level = 0; o.pool_spin_us = 0;
        moty_model *h = NULL; char err[256];
        CHECK(moty_model_open(ap_dir(), &o, &h, err, sizeof err) == MOTY_OK);
        Col c = {0};
        CHECK(ap_gen(h, ap_ids, 6, 12, &c) == MOTY_OK);
        if (cycle == 0) ref = c; else CHECK(!memcmp(ref.toks, c.toks, sizeof(int32_t) * 12));
        if (pool) CHECK(n_threads() >= t0 + 2);       /* the pool's workers are running */
        moty_model_close(h);
        moty_release_scratch();
        if (pool) CHECK(n_threads_settled() == t0);   /* ... and joined */
        long r = rss_kb();
        if (cycle == 1) base = r;
#if !defined(__SANITIZE_THREAD__)                    /* TSan's shadow memory grows with every mapping */
        if (cycle > 1) CHECK(r - base < 2048);        /* flat within 2 MB */
#else
        (void)r; (void)base;
#endif
    }
    return 0;
}

/* lookahead decoding (moty_sampling.draft): with draft_k <= 2 the answer is
 * identical to plain greedy whatever the draft (the model's own answer, a
 * shifted copy, garbage); the budget, the stop at EOS and a continuation
 * after a lookahead call behave as without it; an engine that cannot verify
 * (LFM2: recurrent conv state) ignores the draft; a v1-sized struct works */
static const char *tl_dir(void) {
    static char d[600];
    if (!d[0]) { snprintf(d, sizeof d, "%s", tst_dir("moty_api_llama")); tl_write_dir(d); }
    return d;
}
static moty_status la_gen(moty_model *h, const int32_t *ids, int n, int ntok, int eos, const int32_t *draft, int nd, int k, Col *c,
                          moty_stats *st) {
    moty_sampling s; moty_sampling_init(&s);
    s.max_new_tokens = ntok; s.ignore_eos = !eos; s.draft = draft; s.n_draft = nd; s.draft_k = k;
    c->h = h;
    return moty_generate(h, ids, n, NULL, 0, -1, &s, col_cb, c, st);
}
static int la_steps_logged;
static void la_log(void *u, int level, const char *msg) {
    (void)u; (void)level; int n = 0;
    if (sscanf(msg, "lookahead: %d verify steps", &n) == 1) la_steps_logged += n;
}
int ap_lookahead(void) {
    static const int32_t ids[5] = {1, 7, 11, 3, 29};
    const char *dirs[2] = { tl_dir(), ap_dir() };
    for (int e = 0; e < 2; e++) {                        /* e = 0: Llama (verifies), 1: LFM2 (falls back) */
        moty_model *h = ap_open(dirs[e], 256);
        CHECK(h);
        Col ref = {0};
        moty_reset(h); CHECK(la_gen(h, ids, 5, 24, 0, NULL, 0, 0, &ref, NULL) == MOTY_OK); CHECK(ref.n == 24);
        int32_t garbage[24], shifted[24];
        for (int i = 0; i < 24; i++) { garbage[i] = (i * 13 + 5) % 40; shifted[i] = ref.toks[(i + 3) % 24]; }
        const int32_t *drafts[3] = { ref.toks, shifted, garbage };
        for (int d = 0; d < 3; d++) for (int k = 1; k <= 2; k++) {
            Col c = {0};
            moty_reset(h); CHECK(la_gen(h, ids, 5, 24, 0, drafts[d], 24, k, &c, NULL) == MOTY_OK);
            CHECK(c.n == 24 && !memcmp(c.toks, ref.toks, sizeof(int32_t) * 24));
        }
        {                                               /* the Llama engine really verifies; LFM2 never */
            moty_options o; moty_options_init(&o); o.threads = 2; o.ctx = 256; o.log_level = 2;
            moty_model *h2 = NULL; char err[256]; Col c = {0};
            CHECK(moty_model_open(dirs[e], &o, &h2, err, sizeof err) == MOTY_OK);
            moty_set_log(la_log, NULL); la_steps_logged = 0;
            CHECK(la_gen(h2, ids, 5, 24, 0, ref.toks, 24, 2, &c, NULL) == MOTY_OK);
            moty_set_log(NULL, NULL);
            CHECK(e == 0 ? la_steps_logged > 0 && la_steps_logged < 24 : la_steps_logged == 0);
            moty_model_close(h2);
        }
        Col c4 = {0};                                   /* k = 4: the 4-token tile, near-exact: runs, full budget */
        moty_reset(h); CHECK(la_gen(h, ids, 5, 24, 0, ref.toks, 24, 4, &c4, NULL) == MOTY_OK); CHECK(c4.n == 24);
        Col b5 = {0};                                   /* budget: exactly max_new_tokens */
        moty_reset(h); CHECK(la_gen(h, ids, 5, 5, 0, ref.toks, 24, 2, &b5, NULL) == MOTY_OK);
        CHECK(b5.n == 5 && !memcmp(b5.toks, ref.toks, sizeof(int32_t) * 5));
        Col pe = {0}, le = {0}; moty_stats sp, sl;       /* stop at EOS: the same tokens and reason */
        moty_reset(h); CHECK(la_gen(h, ids, 5, 40, 1, NULL, 0, 0, &pe, &sp) == MOTY_OK);
        moty_reset(h); CHECK(la_gen(h, ids, 5, 40, 1, ref.toks, 24, 2, &le, &sl) == MOTY_OK);
        CHECK(pe.n == le.n && !memcmp(pe.toks, le.toks, sizeof(int32_t) * (size_t)pe.n) && sp.stop == sl.stop);
        Col c1 = {0}, c2 = {0};                         /* continuation: 10 with a draft + 14 more = the plain 24 */
        moty_reset(h);
        CHECK(la_gen(h, ids, 5, 10, 0, ref.toks, 24, 2, &c1, NULL) == MOTY_OK);
        CHECK(la_gen(h, NULL, 0, 14, 0, NULL, 0, 0, &c2, NULL) == MOTY_OK);
        CHECK(c1.n == 10 && c2.n == 14 && !memcmp(c1.toks, ref.toks, 40) && !memcmp(c2.toks, ref.toks + 10, 56));
        moty_model_close(h);
    }
    moty_model *h = ap_open(tl_dir(), 256);             /* argument checks, v1-sized struct */
    CHECK(h);
    moty_sampling s; moty_sampling_init(&s); s.max_new_tokens = 4; s.n_draft = 3;   /* n_draft without draft */
    CHECK(moty_generate(h, ids, 5, NULL, 0, -1, &s, NULL, NULL, NULL) == MOTY_ERR_ARG);
    moty_sampling_init(&s); s.draft_k = -1;
    CHECK(moty_generate(h, ids, 5, NULL, 0, -1, &s, NULL, NULL, NULL) == MOTY_ERR_ARG);
    moty_sampling v1; moty_sampling_init(&v1); v1.size = MOTY_SAMPLING_V1_SIZE; v1.max_new_tokens = 4; v1.n_draft = 99;
    moty_reset(h); CHECK(moty_generate(h, ids, 5, NULL, 0, -1, &v1, NULL, NULL, NULL) == MOTY_OK);
    moty_model_close(h);
    return 0;
}
