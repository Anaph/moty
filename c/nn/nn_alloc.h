/* nn_alloc.h — fondazione condivisa: tempo (now_s), RSS, allocatori con check,
 * scratch arena. M2: le IMPLEMENTAZIONI vivono in nn/alloc.c (libmoty-nn);
 * qui restano i tipi, le utility pure inline e le macro legacy che riscrivono
 * i vecchi nomi (balloc -> moty_balloc) finche' i chiamanti non migrano
 * (docs/symbol-map.md).
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
#define MOTY_HAVE_RUSAGE 1
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
static inline int  omp_get_num_procs(void)    { return 1; }
#endif

/* ---------- utility pure (restano inline) ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#ifdef MOTY_HAVE_RUSAGE
  #if defined(__APPLE__)
  static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }
  #else
  static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
  #endif
#else
  /* Windows/niente-rusage: picco RSS via compat (GetProcessMemoryInfo) se
   * disponibile, altrimenti 0 — e' solo reporting, non funzionale */
  double moty_compat_rss_gb(void);   /* weak: definita da util/compat.h se usato */
  static double rss_gb(void) { return 0.0; }
#endif

typedef struct Scratch { char *raw; int off; char *base; int64_t cap, used; } Scratch;

/* ---------- allocatori + arena: prototipi (nn/alloc.c) ---------- */
void *moty_grow(void **p, int64_t *cap, int64_t need, size_t esz, const char *what);
void *moty_balloc(int64_t n, const char *what);
void *moty_bzalloc(int64_t n, const char *what);
static inline float *moty_falloc(int64_t n) { return (float *)moty_balloc(n*sizeof(float), "f32"); }

void  moty_scr_reset(Scratch *s);
void  moty_scr_free(Scratch *s);
void  moty_scr_reserve(Scratch *s, int64_t bytes);
void *moty_scr_take(Scratch *s, int64_t bytes);

/* Contratto dell'arena (SERIALE, mai dentro una regione parallela):
 *   1. scr_reset()   all'ingresso del kernel/step
 *   2. scr_reserve() PRIMA di tutti i take della fase: la reserve e' l'unica
 *      che puo' reallocare; un take dopo la reserve non muove mai la base
 *   3. scr_take()    consegna chunk allineati a 64 byte (cache line) */
/* allineamento a 64 per il calcolo dei totale di reserve */
static inline int64_t scr_al(int64_t bytes) { return (bytes + 63) & ~(int64_t)63; }

/* macro legacy: i vecchi nomi continuano a comporre (M2 strangler) */
#ifndef MOTY_CORE_NO_LEGACY
#define grow       moty_grow
#define balloc     moty_balloc
#define bzalloc    moty_bzalloc
#define falloc     moty_falloc
#define scr_reset  moty_scr_reset
#define scr_free   moty_scr_free
#define scr_reserve moty_scr_reserve
#define scr_take   moty_scr_take
#endif

#endif /* NN_ALLOC_H */