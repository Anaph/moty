#define MOTY_NO_TRACK_MACROS   /* this file implements the wrappers */
/* alloc.c — unica implementazione degli allocatori e dell'arena Scratch
 * (M2, libmoty-nn core). Le firme pubbliche sono moty_*; nn/nn_alloc.h
 * dichiara i prototipi e tiene le macro legacy (balloc -> moty_balloc). */
#include "nn/nn_alloc.h"

#include "nn/fail.h"
#include <stdarg.h>
#include <pthread.h>

/* ---------- fatal errors (nn/fail.h) ---------- */
static _Thread_local MotyTrap *tl_trap;
MotyTrap *moty_trap_arm(MotyTrap *t) { MotyTrap *prev = tl_trap; t->code = 0; t->msg[0] = 0; tl_trap = t; return prev; }
void moty_trap_disarm(MotyTrap *prev) { tl_trap = prev; }
void moty_fail_code(int code, const char *fmt, ...) {
    char msg[512]; va_list ap; va_start(ap, fmt); vsnprintf(msg, sizeof msg, fmt, ap); va_end(ap);
    size_t ml = strlen(msg); while (ml && msg[ml - 1] == '\n') msg[--ml] = 0;   /* the callers' old fprintf newline */
    MotyTrap *t = tl_trap;
    if (t) {                                   /* library call: back to its entry */
        t->code = code ? code : MOTY_FAIL_GENERIC;
        snprintf(t->msg, sizeof t->msg, "%s", msg);
        tl_trap = NULL;                        /* a failure while unwinding must not loop */
        longjmp(t->jb, 1);
    }
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

/* ---------- environment: the command-line engines read their knobs from it;
 * the library (moty_model_open) switches it off for the whole process —
 * a host's environment is shared by every plugin it loads ---------- */
#include <stdatomic.h>
static atomic_int g_env_off;
void moty_env_disable(void) { atomic_store(&g_env_off, 1); }
const char *moty_getenv(const char *name) { return atomic_load(&g_env_off) ? NULL : getenv(name); }

/* ---------- big blocks straight from mmap ----------
 * What a model keeps (weights, KV cache, scratch arenas, the kernels' grow
 * statics) is allocated with mmap when it is at least MOTY_BIG bytes, so a
 * close really returns it to the OS: glibc's dynamic mmap threshold would
 * otherwise leave such blocks in the heap. A registry maps them to their
 * size; every free in moty (the tracker wrappers) checks it first. */
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#define MOTY_BIG ((size_t)256 << 10)
static pthread_mutex_t g_big_mu = PTHREAD_MUTEX_INITIALIZER;
typedef struct { void *p; size_t n; } BigEnt;
#define BIG_DEAD ((void *)1)
static BigEnt *g_big; static int64_t g_big_cap, g_big_used;
static size_t big_h(const void *p, int64_t cap) { uintptr_t x = (uintptr_t)p >> 12; x *= 0x9E3779B97F4A7C15ull; return (size_t)(x >> 7) & (size_t)(cap - 1); }
static void big_put(void *p, size_t n);
static int64_t g_big_live;
static void big_grow(void) {                         /* rehash; double only when live entries need it */
    int64_t oc = g_big_cap; BigEnt *old = g_big;
    g_big_cap = !oc ? 256 : (g_big_live + 1) * 4 > oc ? oc * 2 : oc;
    g_big = calloc((size_t)g_big_cap, sizeof *g_big); g_big_used = 0; g_big_live = 0;
    if (!g_big) { g_big = old; g_big_cap = oc; moty_fail_code(MOTY_FAIL_OOM, "OOM: big-block registry"); }
    for (int64_t i = 0; i < oc; i++) if (old[i].p && old[i].p != BIG_DEAD) big_put(old[i].p, old[i].n);
    free(old);
}
static void big_put(void *p, size_t n) {
    if ((g_big_used + 1) * 2 > g_big_cap) big_grow();
    size_t i = big_h(p, g_big_cap);
    while (g_big[i].p && g_big[i].p != BIG_DEAD) i = (i + 1) & (size_t)(g_big_cap - 1);
    if (!g_big[i].p) g_big_used++;
    g_big[i].p = p; g_big[i].n = n; g_big_live++;
}
static size_t big_find(void *p, int take) {          /* 0: not a big block */
    if (!p || !g_big_cap) return 0;
    size_t i = big_h(p, g_big_cap);
    while (g_big[i].p) {
        if (g_big[i].p == p) { size_t n = g_big[i].n; if (take) { g_big[i].p = BIG_DEAD; g_big_live--; } return n; }
        i = (i + 1) & (size_t)(g_big_cap - 1);
    }
    return 0;
}
static void *big_alloc(size_t n) {                   /* zero-filled */
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    pthread_mutex_lock(&g_big_mu); big_put(p, n); pthread_mutex_unlock(&g_big_mu);
    return p;
}
static int big_free(void *p) {
    if (!p) return 0;
    pthread_mutex_lock(&g_big_mu); size_t n = big_find(p, 1); pthread_mutex_unlock(&g_big_mu);
    if (!n) return 0;
    munmap(p, n);
    return 1;
}
static size_t big_len(void *p) { pthread_mutex_lock(&g_big_mu); size_t n = big_find(p, 0); pthread_mutex_unlock(&g_big_mu); return n; }
#else
#define MOTY_BIG ((size_t)-1)
static void *big_alloc(size_t n) { return calloc(1, n); }
static int big_free(void *p) { (void)p; return 0; }
static size_t big_len(void *p) { (void)p; return 0; }
#endif
/* any moty block: a big one, else libc */
static void mfree(void *p) { if (!big_free(p)) free(p); }
/* resize, contents kept. A big block stays one; a heap block that grows to
 * >= MOTY_BIG with `big` set moves into one (realloc first: its old size is
 * unknown here, the realloc'd block has n valid bytes to copy). */
static void *mrealloc(void *p, size_t n, int big) {
    size_t on = big_len(p);
    if (!on && !(big && n >= MOTY_BIG)) return realloc(p, n);
    if (on && n <= on) return p;
    if (!on && p) { void *r = realloc(p, n); if (!r) return NULL; p = r; on = n; }
    void *q = big_alloc(n);
    if (!q) return NULL;
    if (p) { memcpy(q, p, on < n ? on : n); mfree(p); }
    return q;
}

/* ---------- open tracker: a pointer set (open addressing, tombstones) ---------- */
static _Thread_local MotyTrack *tl_track;
#define TRK_DEAD ((void *)1)
MotyTrack *moty_track_set(MotyTrack *t) { MotyTrack *prev = tl_track; tl_track = t; return prev; }
static size_t trk_h(const void *p, int64_t cap) { uintptr_t x = (uintptr_t)p >> 4; x ^= x >> 17; x *= 0x9E3779B97F4A7C15ull; return (size_t)(x & (uintptr_t)(cap - 1)); }
static void trk_add(MotyTrack *t, void *p);
static void trk_grow(MotyTrack *t) {
    int64_t oc = t->cap; void **op = t->p;
    t->cap = oc ? oc * 2 : 1024; t->p = calloc((size_t)t->cap, sizeof(void *)); t->n = t->live = 0;
    if (!t->p) moty_fail_code(MOTY_FAIL_OOM, "OOM: allocation tracker (%lld entries)", (long long)t->cap);
    for (int64_t i = 0; i < oc; i++) if (op[i] && op[i] != TRK_DEAD) trk_add(t, op[i]);
    free(op);
}
static void trk_add(MotyTrack *t, void *p) {
    if (!p) return;
    if ((t->n + 1) * 2 > t->cap) trk_grow(t);
    size_t i = trk_h(p, t->cap);
    while (t->p[i] && t->p[i] != TRK_DEAD) i = (i + 1) & (size_t)(t->cap - 1);
    if (!t->p[i]) t->n++;
    t->p[i] = p; t->live++;
}
static int trk_del(MotyTrack *t, void *p) {
    if (!p || !t->cap) return 0;
    size_t i = trk_h(p, t->cap);
    while (t->p[i]) { if (t->p[i] == p) { t->p[i] = TRK_DEAD; t->live--; return 1; } i = (i + 1) & (size_t)(t->cap - 1); }
    return 0;
}
void moty_track_free_all(MotyTrack *t) {
    for (int64_t i = 0; i < t->cap; i++) if (t->p[i] && t->p[i] != TRK_DEAD) mfree(t->p[i]);
    free(t->p); t->p = NULL; t->n = t->cap = t->live = 0;
}
void moty_track_merge(MotyTrack *dst, MotyTrack *src) {   /* src's entries become dst's */
    for (int64_t i = 0; i < src->cap; i++) if (src->p[i] && src->p[i] != TRK_DEAD) trk_add(dst, src->p[i]);
    free(src->p); src->p = NULL; src->n = src->cap = src->live = 0;
}
/* a model's persistent blocks (tracker with `big`, i.e. during open) come from mmap */
static int want_big(size_t n) { return tl_track && tl_track->big && n >= MOTY_BIG; }
void *moty_trk_malloc(size_t n) { void *p = want_big(n) ? big_alloc(n) : malloc(n); if (tl_track) trk_add(tl_track, p); return p; }
void *moty_trk_calloc(size_t n, size_t sz) {
    size_t b = n * sz; void *p = want_big(b) ? big_alloc(b) : calloc(n, sz);
    if (tl_track) trk_add(tl_track, p);
    return p;
}
void *moty_trk_realloc(void *p, size_t n) {
    int owned = tl_track && trk_del(tl_track, p);
    void *q = mrealloc(p, n, tl_track && tl_track->big);
    if (tl_track && (owned || !p)) trk_add(tl_track, q ? q : p);   /* failed realloc keeps p */
    return q;
}
char *moty_trk_strdup(const char *s) { char *p = strdup(s); if (tl_track) trk_add(tl_track, p); return p; }
void moty_trk_free(void *p) { if (tl_track) trk_del(tl_track, p); mfree(p); }

/* ---------- the grow() statics: registered so they can be released ---------- */
static pthread_mutex_t g_scr_mu = PTHREAD_MUTEX_INITIALIZER;
static struct { void **p; int64_t *cap; } g_scr[128]; static int g_nscr;
void moty_release_scratch_all(void) {
    pthread_mutex_lock(&g_scr_mu);
    for (int i = 0; i < g_nscr; i++) { mfree(*g_scr[i].p); *g_scr[i].p = NULL; *g_scr[i].cap = 0; }
    g_nscr = 0;
    pthread_mutex_unlock(&g_scr_mu);
}

void *moty_grow(void **p, int64_t *cap, int64_t need, size_t esz, const char *what) {
    if (need <= *cap) return *p;
    if (!*p) {                                 /* first allocation of this buffer: register it */
        pthread_mutex_lock(&g_scr_mu);
        int k = 0; while (k < g_nscr && g_scr[k].p != p) k++;
        if (k == g_nscr && g_nscr < (int)(sizeof g_scr / sizeof *g_scr)) { g_scr[g_nscr].p = p; g_scr[g_nscr].cap = cap; g_nscr++; }
        pthread_mutex_unlock(&g_scr_mu);
    }
    void *q = mrealloc(*p, (size_t)need * esz, 1);       /* process-lifetime: big blocks from mmap */
    if (!q) moty_fail_code(MOTY_FAIL_OOM, "OOM %s (%lld x %lu byte)", what, (long long)need, (unsigned long)esz);
    *p = q; *cap = need;
    return q;
}
void *moty_balloc(int64_t n, const char *what) {
    void *p = want_big((size_t)n) ? big_alloc((size_t)n) : malloc(n > 0 ? n : 1);
    if (!p) moty_fail_code(MOTY_FAIL_OOM, "OOM %s (%lld byte)", what, (long long)n);
    if (tl_track) trk_add(tl_track, p);
    return p;
}
void *moty_bzalloc(int64_t n, const char *what) {
    void *p = want_big((size_t)n) ? big_alloc((size_t)n) : calloc(1, n > 0 ? n : 1);
    if (!p) moty_fail_code(MOTY_FAIL_OOM, "OOM %s (%lld byte)", what, (long long)n);
    if (tl_track) trk_add(tl_track, p);
    return p;
}
void moty_bfree(void *p) { if (tl_track) trk_del(tl_track, p); mfree(p); }
void moty_scr_reset(Scratch *s) { s->used = 0; }
void moty_scr_free(Scratch *s) { mfree(s->raw); s->raw = NULL; s->base = NULL; s->off = 0; s->cap = s->used = 0; }
void moty_scr_reserve(Scratch *s, int64_t bytes) {
    if (bytes <= s->cap) return;
    int64_t nc = s->cap ? s->cap : (int64_t)1 << 20;   /* 1MB iniziale, x2 */
    while (nc < bytes) nc *= 2;
    char *nr = mrealloc(s->raw, (size_t)nc + 63, 1);   /* the model's arena: a big block */
    if (!nr) moty_fail_code(MOTY_FAIL_OOM, "OOM scratch (%lld byte)", (long long)nc);
    int noff = (int)((64 - ((uintptr_t)nr & 63)) & 63);
    if (s->base && noff != s->off && s->used > 0) {
        int64_t live = s->used < s->cap ? s->used : s->cap;
        memmove(nr + noff, nr + s->off, (size_t)live);
    }
    s->raw = nr; s->off = noff; s->base = nr + noff; s->cap = nc;
}
void *moty_scr_take(Scratch *s, int64_t bytes) {
    bytes = (bytes + 63) & ~(int64_t)63;
    void *p = s->base + s->used;
    s->used += bytes;
    return p;
}

#include "util/prof.h"
double moty_prof_op[OP_N];   /* util/prof.h per-op table */
