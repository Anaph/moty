/* api.c — the public library (api/moty.h) over the engines' ops
 * (runtime/engine_api.h): one process-wide lock, a MotyTrap per call, the
 * open tracker for ownership, the generation loop with its callback. */
#define _GNU_SOURCE
#include "api/moty.h"
#include "api/api_internal.h"
#include "nn/nn_alloc.h"
#include "nn/nn_sample.h"
#include "nn/par.h"
#include "util/json.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifndef MOTY_VERSION_STR
#define MOTY_VERSION_STR "0.2"
#endif

struct moty_model {
    const MotyEngineOps *ops;
    void *inst;
    MotyTrack track;                    /* everything the instance owns */
    moty_options opt;
    int n_past;
    int pending;                        /* last sampled token, not yet in the KV (-1: none) */
    double load_s;
    atomic_int abort;
    char err[512];
    /* per-call state that lives across setjmp/longjmp: heap, not locals */
    MotyTrap trap;
    MotyTrack call;
    float *lo;
    moty_stats stats;
    moty_status rc;
    int32_t *gen; int ng;               /* lookahead: the answer so far (suffix lookup in the draft) */
    int la_steps, la_acc;               /* lookahead verification steps / accepted proposals (log level 2) */
};

static pthread_mutex_t g_api_mu = PTHREAD_MUTEX_INITIALIZER;

/* logging: a process-wide callback, stderr by default */
static moty_log_cb g_log_cb; static void *g_log_user;
void moty_set_log(moty_log_cb cb, void *user) { g_log_cb = cb; g_log_user = user; }
static void api_log(int level, const char *fmt, ...) {
    char msg[768]; va_list ap; va_start(ap, fmt); vsnprintf(msg, sizeof msg, fmt, ap); va_end(ap);
    if (g_log_cb) g_log_cb(g_log_user, level, msg);
    else fprintf(stderr, "[moty] %s\n", msg);
}

static double api_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static moty_status status_of(int code) {
    switch (code) {
    case MOTY_FAIL_IO: return MOTY_ERR_IO;
    case MOTY_FAIL_OOM: return MOTY_ERR_OOM;
    default: return MOTY_ERR_FORMAT;
    }
}
static void set_err(char *dst, size_t n, const char *fmt, ...) {
    if (!dst || !n) return;
    va_list ap; va_start(ap, fmt); vsnprintf(dst, n, fmt, ap); va_end(ap);
}

void moty_options_init(moty_options *o) {
    memset(o, 0, sizeof *o);
    o->size = sizeof *o; o->qbits = 4; o->log_level = 1;
    o->prefill_chunk = 64;          /* bounds abort latency: 0.86 s measured on a Cortex-A53 */
    o->pool_spin_us = 1000; o->pin_threads = 1;
    o->mmap_weights = 1;
}
void moty_sampling_init(moty_sampling *s) {
    memset(s, 0, sizeof *s);
    s->size = sizeof *s; s->top_p = 0.95f; s->max_new_tokens = 64;
}

/* config.json model_type (a multimodal root keeps its own type) */
static int model_type_of(const char *dir, char *type, size_t n, char *err, size_t errn) {
    char p[4096]; snprintf(p, sizeof p, "%s/config.json", dir);
    FILE *f = fopen(p, "rb");
    if (!f) { set_err(err, errn, "%s: %s", p, strerror(errno)); return MOTY_ERR_IO; }
    fclose(f);
    char *b = slurp_file(p, NULL); jval *r = b ? json_parse(b) : NULL;
    jval *t = r ? json_get(r, "model_type") : NULL;
    int ok = t && t->t == J_STR;
    if (ok) snprintf(type, n, "%s", t->str);
    if (r) json_free(r);
    free(b);
    if (!ok) { set_err(err, errn, "%s: no model_type", p); return MOTY_ERR_FORMAT; }
    return MOTY_OK;
}

moty_status moty_model_open_with(const MotyEngineOps *ops, const MotyEngineOps *const *engines, const char *path,
                                 const moty_options *opt, moty_model **out, char *err, size_t err_len) {
    if (!path || !out) { set_err(err, err_len, "moty_model_open: NULL argument"); return MOTY_ERR_ARG; }
    *out = NULL;
    moty_options o; moty_options_init(&o);
    if (opt) {
        /* any published version of the struct: fields it lacks keep their defaults */
        if (opt->size != (int)sizeof o && opt->size != MOTY_OPTIONS_V1_SIZE) {
            set_err(err, err_len, "moty_options.size mismatch (moty_options_init?)"); return MOTY_ERR_ARG; }
        memcpy(&o, opt, (size_t)opt->size); o.size = (int)sizeof o;
    }
    if (o.qbits != 4 && o.qbits != 8) { set_err(err, err_len, "qbits must be 4 or 8"); return MOTY_ERR_ARG; }
    if (o.kv_bits != 0 && o.kv_bits != 8) { set_err(err, err_len, "kv_bits must be 0 or 8"); return MOTY_ERR_ARG; }
    if (o.mmap_weights < 0 || o.mmap_weights > 2) { set_err(err, err_len, "mmap_weights must be 0, 1 or 2"); return MOTY_ERR_ARG; }
    if (o.threads < 0 || o.threads_decode < 0 || o.ctx < 0 || o.head_topk < 0 || o.prefill_chunk < 0 || o.pool_spin_us < 0) {
        set_err(err, err_len, "negative option"); return MOTY_ERR_ARG; }
    if (!o.ctx) o.ctx = 4096;
    char type[128];
    moty_status st = model_type_of(path, type, sizeof type, err, err_len);
    if (st) return st;
    if (!ops) {
        for (int i = 0; engines && engines[i] && !ops; i++) if (engines[i]->accepts(type)) ops = engines[i];
        if (!ops) { set_err(err, err_len, "model_type \"%s\" is not served by the library", type); return MOTY_ERR_UNSUPPORTED; }
    } else if (!ops->accepts(type)) {
        set_err(err, err_len, "model_type \"%s\" is not served by the %s engine", type, ops->id); return MOTY_ERR_UNSUPPORTED;
    }
    moty_model *m = calloc(1, sizeof *m);
    if (!m) { set_err(err, err_len, "OOM"); return MOTY_ERR_OOM; }
    m->ops = ops; m->opt = o; m->pending = -1;
    double t_open = api_now();
    MotyEngineOpen eo = { o.qbits, o.ctx, o.kv_bits, o.embed_disk, o.head_topk,
#if defined(__aarch64__) && defined(__ARM_NEON)
                          1,
#else
                          0,
#endif
                          o.log_level, NULL, &m->inst, o.mmap_weights };
    pthread_mutex_lock(&g_api_mu);
    moty_par_set_threads(o.threads ? o.threads : moty_par_procs());
    moty_par_config(o.pool_spin_us, o.pin_threads);
    MotyParCaller pc; moty_par_enter(&pc);
    MotyTrap *prev = moty_trap_arm(&m->trap);
    m->track.big = 1;                        /* the model's blocks: mmap, returned on close */
    MotyTrack *prevt = moty_track_set(&m->track);
    if (MOTY_TRAP_TRY(&m->trap)) {
        m->inst = m->ops->open(path, &eo);
        moty_track_set(prevt); moty_trap_disarm(prev);
        moty_par_leave(&pc);
        m->load_s = api_now() - t_open;
        if (o.log_level >= 2)
            api_log(2, "%s: %s engine, vocab %d, hidden %d, ctx %d, load %.2f s", path, m->ops->id,
                    m->ops->vocab(m->inst), m->ops->hidden(m->inst), m->ops->ctx(m->inst), m->load_s);
        pthread_mutex_unlock(&g_api_mu);
        *out = m;
        return MOTY_OK;
    }
    /* failed: whatever was allocated is in the tracker; files may be open */
    moty_track_set(prevt); moty_trap_disarm(prev);
    moty_par_leave(&pc);
    set_err(err, err_len, "%s", m->trap.msg);
    if (m->opt.log_level >= 1) api_log(1, "open %s: %s", path, m->trap.msg);
    if (m->inst) m->ops->close(m->inst);          /* files opened before the failure */
    moty_track_free_all(&m->track);
    pthread_mutex_unlock(&g_api_mu);
    moty_status rc = status_of(m->trap.code);
    free(m);
    return rc;
}

void moty_model_close(moty_model *m) {
    if (!m) return;
    pthread_mutex_lock(&g_api_mu);
    if (m->inst) m->ops->close(m->inst);
    moty_track_free_all(&m->track);
    pthread_mutex_unlock(&g_api_mu);
    free(m);
}

int moty_model_vocab(const moty_model *m)       { return m ? m->ops->vocab(m->inst) : MOTY_ERR_ARG; }
int moty_model_hidden(const moty_model *m)      { return m ? m->ops->hidden(m->inst) : MOTY_ERR_ARG; }
int moty_model_ctx(const moty_model *m)         { return m ? m->ops->ctx(m->inst) : MOTY_ERR_ARG; }
int moty_model_image_token(const moty_model *m) { return m ? m->ops->image_token(m->inst) : MOTY_ERR_ARG; }
int moty_model_n_past(const moty_model *m)      { return m ? m->n_past : MOTY_ERR_ARG; }
double moty_model_load_s(const moty_model *m)   { return m ? m->load_s : 0; }
const char *moty_last_error(const moty_model *m) { return m ? m->err : "NULL handle"; }

int moty_tokenize(moty_model *m, const char *text, int add_bos, int chat_template, int32_t *ids, int cap) {
    if (!m || !text || (!ids && cap > 0) || cap < 0) return MOTY_ERR_ARG;
    if (!m->ops->has_tokenizer(m->inst)) { set_err(m->err, sizeof m->err, "no tokenizer.json in the model directory"); return MOTY_ERR_UNSUPPORTED; }
    pthread_mutex_lock(&g_api_mu);
    MotyTrap *prev = moty_trap_arm(&m->trap);
    memset(&m->call, 0, sizeof m->call);
    MotyTrack *prevt = moty_track_set(&m->call);
    if (MOTY_TRAP_TRY(&m->trap)) {
        m->rc = m->ops->encode(m->inst, text, add_bos, chat_template, ids, cap);
    } else {
        set_err(m->err, sizeof m->err, "%s", m->trap.msg); m->rc = status_of(m->trap.code);
    }
    moty_track_set(prevt); moty_trap_disarm(prev);
    moty_track_free_all(&m->call);                /* temporaries of a failed call; empty otherwise */
    int n = m->rc;
    pthread_mutex_unlock(&g_api_mu);
    return n;
}

int moty_token_piece(moty_model *m, int32_t id, char *buf, int cap) {
    if (!m || !buf || cap <= 0) return MOTY_ERR_ARG;
    return m->ops->piece(m->inst, id, buf, cap);
}

void moty_abort(moty_model *m) { if (m) atomic_store(&m->abort, 1); }
void moty_abort_clear(moty_model *m) { if (m) atomic_store(&m->abort, 0); }

moty_status moty_reset(moty_model *m) {
    if (!m) return MOTY_ERR_ARG;
    pthread_mutex_lock(&g_api_mu);
    m->ops->reset(m->inst); m->n_past = 0; m->pending = -1;
    atomic_store(&m->abort, 0);
    pthread_mutex_unlock(&g_api_mu);
    return MOTY_OK;
}

/* one produced token: count, callback, stop and budget checks (shared by the
 * plain and the lookahead decode); returns 1 when generation must stop */
static int emit_tok(moty_model *m, const moty_sampling *s, moty_token_cb cb, void *user, int t, char *piece, int cap) {
    m->stats.new_tokens++; m->pending = t;
    if (m->gen) m->gen[m->ng++] = t;
    int stop = !s->ignore_eos && m->ops->is_stop(m->inst, t);
    int pl = cb && !stop ? m->ops->piece(m->inst, t, piece, cap) : 0;
    if (cb && cb(user, t, piece, pl)) { m->stats.stop = MOTY_STOP_CALLBACK; return 1; }
    if (stop) { m->stats.stop = MOTY_STOP_EOS; return 1; }
    return (int)m->stats.new_tokens >= s->max_new_tokens;
}

moty_status moty_generate(moty_model *m, const int32_t *ids, int n, const float *embeds, int n_rows, int32_t embed_token,
                          const moty_sampling *sp, moty_token_cb cb, void *user, moty_stats *st) {
    if (!m || (!ids && n > 0) || n < 0 || n_rows < 0 || (embeds && n_rows == 0)) return MOTY_ERR_ARG;
    moty_sampling s; moty_sampling_init(&s);
    if (sp) {
        if (sp->size != (int)sizeof s && sp->size != MOTY_SAMPLING_V1_SIZE) {
            set_err(m->err, sizeof m->err, "moty_sampling.size mismatch"); return MOTY_ERR_ARG; }
        memcpy(&s, sp, (size_t)sp->size); s.size = (int)sizeof s;   /* fields added since keep their defaults */
    }
    if (s.max_new_tokens < 0 || s.temperature < 0 || s.n_draft < 0 || s.draft_k < 0 || (s.n_draft > 0 && !s.draft)) {
        set_err(m->err, sizeof m->err, "bad sampling parameters"); return MOTY_ERR_ARG; }
    int V = m->ops->vocab(m->inst), ctx = m->ops->ctx(m->inst);
    int n_ph = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= V) { set_err(m->err, sizeof m->err, "token id %d out of range", ids[i]); return MOTY_ERR_ARG; }
        n_ph += embeds && ids[i] == embed_token;
    }
    if (embeds && n_ph != n_rows) {       /* never read past the caller's rows */
        set_err(m->err, sizeof m->err, "embeds: %d placeholder positions (token %d) but %d rows", n_ph, embed_token, n_rows);
        return MOTY_ERR_ARG;
    }
    if (atomic_load(&m->abort)) {          /* sticky: a stop may land before the call starts */
        set_err(m->err, sizeof m->err, "aborted (moty_abort_clear or moty_reset to run again)");
        if (st) { memset(st, 0, sizeof *st); st->stop = MOTY_STOP_ABORT; }
        return MOTY_ERR_ABORTED;
    }
    int np = n + (m->pending >= 0);                     /* the previous answer's last token goes first */
    if (np == 0) { set_err(m->err, sizeof m->err, "empty prompt on an empty conversation"); return MOTY_ERR_ARG; }
    if (m->n_past + np + s.max_new_tokens > ctx) {
        set_err(m->err, sizeof m->err, "context: %d past + %d prompt + %d new > %d", m->n_past, np, s.max_new_tokens, ctx);
        return MOTY_ERR_CONTEXT;
    }
    int *pr = malloc(sizeof(int) * (size_t)np);
    if (!pr) { set_err(m->err, sizeof m->err, "OOM"); return MOTY_ERR_OOM; }
    int k0 = 0;
    if (m->pending >= 0) pr[k0++] = m->pending;
    for (int i = 0; i < n; i++) pr[k0 + i] = ids[i];
    pthread_mutex_lock(&g_api_mu);
    memset(&m->stats, 0, sizeof m->stats); m->stats.prompt_tokens = np;
    int th = m->opt.threads ? m->opt.threads : moty_par_procs();
    int thd = m->opt.threads_decode ? m->opt.threads_decode : th;
    moty_par_set_threads(th);
    moty_par_config(m->opt.pool_spin_us, m->opt.pin_threads);
    MotyParCaller pc; moty_par_enter(&pc);
    moty_g_temp = s.temperature; moty_g_nuc = s.top_p;
    if (s.seed) moty_g_rng = s.seed | 1u;
    MotyTrap *prev = moty_trap_arm(&m->trap);
    memset(&m->call, 0, sizeof m->call);
    MotyTrack *prevt = moty_track_set(&m->call);
    m->rc = MOTY_OK; m->lo = NULL;
    if (MOTY_TRAP_TRY(&m->trap)) {
        double t0 = api_now();
        m->lo = m->ops->prefill(m->inst, pr, np, m->n_past, embeds, n_rows, embed_token, m->opt.prefill_chunk, &m->abort);
        m->n_past += np; m->pending = -1;
        m->stats.prefill_s = api_now() - t0;
        moty_par_set_threads(thd);
        t0 = api_now();
        char piece[256];
        /* lookahead (greedy + draft + an engine that can verify): each step feeds the
         * pending token and up to draft_k proposed ones in one forward and keeps the
         * proposals the model itself predicts; without it one token per step */
        int la = s.draft && s.n_draft > 0 && s.temperature == 0 && m->ops->verify;
        int dk = s.draft_k > 0 ? s.draft_k : 2;
        m->ng = 0; m->la_steps = m->la_acc = 0;
        if (la && !(m->gen = malloc(sizeof(int32_t) * (size_t)(s.max_new_tokens + 1)))) moty_fail_code(MOTY_FAIL_OOM, "OOM: lookahead");
        int t = m->lo ? moty_pick_tok((Scratch *)m->ops->scratch(m->inst), m->lo, V) : -1;
        if (m->lo) { moty_trk_free(m->lo); m->lo = NULL; }
        while (t >= 0 && s.max_new_tokens > 0) {
            if (atomic_load(&m->abort)) break;
            if (emit_tok(m, &s, cb, user, t, piece, sizeof piece)) break;
            int prop[16], np2 = 0, budget = s.max_new_tokens - (int)m->stats.new_tokens;
            if (la && budget > 0) {                    /* the tokens after the longest (<= 3) answer suffix found in the draft */
                int kmax = dk < 15 ? dk : 15; if (kmax > budget) kmax = budget;
                for (int len = m->ng < 3 ? m->ng : 3; len > 0 && !np2; len--)
                    for (int j = 0; j + len < s.n_draft && !np2; j++) {
                        if (memcmp(s.draft + j, m->gen + m->ng - len, sizeof(int32_t) * (size_t)len)) continue;
                        while (np2 < kmax && j + len + np2 < s.n_draft) { prop[np2] = s.draft[j + len + np2]; np2++; }
                    }
            }
            if (np2 > 0) {
                int seq[17]; seq[0] = t; memcpy(seq + 1, prop, sizeof(int) * (size_t)np2);
                m->lo = m->ops->verify(m->inst, seq, np2 + 1, m->n_past);
                if (m->lo) {                           /* m->lo: freed by the call's cleanup if a check longjmps */
                    int a = 0, stopped = 0;
                    m->n_past++; m->pending = -1;          /* t is in the KV now */
                    for (; a < np2; a++) {
                        int y = moty_pick_tok((Scratch *)m->ops->scratch(m->inst), m->lo + (size_t)a * V, V);
                        if (y != prop[a]) { t = y; break; }
                        if (emit_tok(m, &s, cb, user, y, piece, sizeof piece)) { stopped = 1; break; }
                        m->n_past++; m->pending = -1;      /* an accepted proposal is in the KV */
                    }
                    if (!stopped && a == np2) t = moty_pick_tok((Scratch *)m->ops->scratch(m->inst), m->lo + (size_t)np2 * V, V);
                    moty_trk_free(m->lo); m->lo = NULL;
                    m->la_steps++; m->la_acc += a;
                    if (stopped) break;
                    continue;
                }
            }
            m->lo = m->ops->step(m->inst, t, m->n_past);
            m->n_past++; m->pending = -1;
            t = moty_pick_tok((Scratch *)m->ops->scratch(m->inst), m->lo, V);
            moty_trk_free(m->lo); m->lo = NULL;
        }
        if (la && m->opt.log_level >= 2)
            api_log(2, "lookahead: %d verify steps, %d proposals accepted, %d tokens", m->la_steps, m->la_acc, (int)m->stats.new_tokens);
        m->stats.decode_s = api_now() - t0;
        if (atomic_load(&m->abort)) { m->rc = MOTY_ERR_ABORTED; m->stats.stop = MOTY_STOP_ABORT; }
    } else {
        m->rc = status_of(m->trap.code);
        set_err(m->err, sizeof m->err, "%s", m->trap.msg);
        if (m->opt.log_level >= 1) api_log(1, "generate: %s", m->trap.msg);
    }
    if (m->lo) { moty_trk_free(m->lo); m->lo = NULL; }
    free(m->gen); m->gen = NULL;
    moty_track_set(prevt); moty_trap_disarm(prev);
    if (m->rc == MOTY_OK) moty_track_merge(&m->track, &m->call);   /* allocations meant to persist */
    else moty_track_free_all(&m->call);                            /* the failed call's temporaries */
    moty_par_set_threads(th);
    moty_par_leave(&pc);
    if (m->rc != MOTY_OK) { m->ops->reset(m->inst); m->n_past = 0; m->pending = -1; }  /* the KV is inconsistent */
    if (m->rc == MOTY_ERR_ABORTED) set_err(m->err, sizeof m->err, "aborted (moty_abort_clear or moty_reset to run again)");
    moty_status rc = m->rc;
    if (st) *st = m->stats;
    pthread_mutex_unlock(&g_api_mu);
    free(pr);
    return rc;
}

void moty_release_scratch(void) {
    pthread_mutex_lock(&g_api_mu);
    moty_release_scratch_all();
    moty_par_shutdown();
    pthread_mutex_unlock(&g_api_mu);
}

const char *moty_version(void) {
    static char v[64];
    snprintf(v, sizeof v, "moty " MOTY_VERSION_STR " (%s)", moty_par_backend());
    return v;
}
