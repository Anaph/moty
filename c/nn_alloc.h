/* nn_alloc.h — fondazione condivisa: tempo (now_s), RSS, allocatori con check,
 * scratch statico grow + stub OpenMP. Estratto da nn.h per granularita': un
 * motore che vuole solo gli allocatori (niente gemm/sampler) include questo.
 * Dipende solo da libc; niente simd.h, niente Mat. */
#ifndef NN_ALLOC_H
#define NN_ALLOC_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#endif

/* OpenMP: header reale se compilato con -fopenmp, altrimenti stub inline a
 * un thread, cosi' i chiamanti (THREADS, scratch per-thread) non hanno
 * bisogno di #ifdef sparsi. Vive qui (fondazione) perche' gemm/att lo usano. */
#ifdef _OPENMP
#include <omp.h>
#else
static inline int  omp_get_max_threads(void) { return 1; }
static inline int  omp_get_thread_num(void)  { return 0; }
static inline void omp_set_num_threads(int n) { (void)n; }
#endif

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
#endif
/* allocatori con check: UN solo formato di messaggio OOM (prima: sette
 * varianti scritte a mano nei call site) */
static void *balloc(int64_t n, const char *what) {
    void *p = malloc(n > 0 ? n : 1);
    if (!p) { fprintf(stderr, "OOM %s (%lld byte)\n", what, (long long)n); exit(1); }
    return p;
}
static void *bzalloc(int64_t n, const char *what) {
    void *p = calloc(1, n > 0 ? n : 1);
    if (!p) { fprintf(stderr, "OOM %s (%lld byte)\n", what, (long long)n); exit(1); }
    return p;
}
static float *falloc(int64_t n) { return (float *)balloc(n*sizeof(float), "f32"); }

/* scratch che cresce e basta (l'idioma static-grow dei kernel: contratto di
 * chiamata SERIALE, mai da dentro una regione parallela) */
static void grow(void **p, int64_t *cap, int64_t need, size_t esz, const char *what) {
    if (need <= *cap) return;
    *cap = need;
    *p = realloc(*p, (size_t)need * esz);
    if (!*p) { fprintf(stderr, "OOM %s (%lld x %zu byte)\n", what, (long long)need, esz); exit(1); }
}

/* ---------- P5: arena scratch per-Model ----------
 * Sostituisce gli static-grow function-local nei kernel condivisi: due Model
 * nello stesso processo non si calpestano piu' la memoria.
 * Contratto (come grow(), SERIALE, mai dentro una regione parallela):
 *   1. scr_reset()   all'ingresso del kernel/step
 *   2. scr_reserve() PRIMA di tutti i take della fase: la reserve e' l'unica
 *      che puo' reallocare; un take dopo la reserve non muove mai la base,
 *      quindi i puntatori consegnati prima restano validi
 *   3. scr_take()    consegna chunk allineati a 64 byte (cache line) */
typedef struct Scratch { char *base; int64_t cap, used; } Scratch;

static void scr_reset(Scratch *s) { s->used = 0; }
static void scr_free(Scratch *s) { free(s->base); s->base = NULL; s->cap = s->used = 0; }

static void scr_reserve(Scratch *s, int64_t bytes) {
    if (bytes <= s->cap) return;
    int64_t nc = s->cap ? s->cap : (int64_t)1 << 20;   /* 1MB iniziale, x2 */
    while (nc < bytes) nc *= 2;
    s->base = realloc(s->base, (size_t)nc);
    if (!s->base) { fprintf(stderr, "OOM scratch (%lld byte)\n", (long long)nc); exit(1); }
    s->cap = nc;
}
static void *scr_take(Scratch *s, int64_t bytes) {
    bytes = (bytes + 63) & ~(int64_t)63;
    void *p = s->base + s->used;
    s->used += bytes;
    return p;
}
/* allineamento a 64 per il calcolo dei totale di reserve */
static inline int64_t scr_al(int64_t bytes) { return (bytes + 63) & ~(int64_t)63; }

#endif /* NN_ALLOC_H */
