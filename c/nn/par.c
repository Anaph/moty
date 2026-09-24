/* par.c — backends of nn/par.h. */
#define _GNU_SOURCE
#include "nn/par.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#if defined(_OPENMP) && defined(MOTY_THREADPOOL)
#error "MOTY_THREADPOOL replaces OpenMP: build without -fopenmp"
#endif

#if defined(_OPENMP)
#include <omp.h>

void moty_par_for(int64_t n, int chunk, moty_par_fn fn, void *ctx) {
    if (n <= 0) return;
    if (omp_in_parallel() || omp_get_max_threads() == 1) { fn(ctx, 0, n, omp_get_thread_num()); return; }
    if (chunk <= 0) {
        #pragma omp parallel
        {
            int64_t i0, i1;
            moty_par_range(n, omp_get_thread_num(), omp_get_num_threads(), &i0, &i1);
            if (i0 < i1) fn(ctx, i0, i1, omp_get_thread_num());
        }
        return;
    }
    int64_t nc = (n + chunk - 1) / chunk;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t c = 0; c < nc; c++) {
        int64_t i0 = c * chunk, i1 = i0 + chunk < n ? i0 + chunk : n;
        fn(ctx, i0, i1, omp_get_thread_num());
    }
}
int  moty_par_threads(void)      { return omp_get_max_threads(); }
void moty_par_set_threads(int n) { if (n > 0) omp_set_num_threads(n); }
int  moty_par_procs(void)        { return omp_get_num_procs(); }
int  moty_par_tid(void)          { return omp_get_thread_num(); }
int  moty_par_active(void)       { return omp_in_parallel(); }
const char *moty_par_backend(void) { return "openmp"; }
void moty_par_config(long spin_us, int pin) { (void)spin_us; (void)pin; }
void moty_par_enter(MotyParCaller *c) { (void)c; }
void moty_par_leave(MotyParCaller *c) { (void)c; }
void moty_par_shutdown(void) {}

#elif defined(MOTY_THREADPOOL)
#include <pthread.h>
#include <string.h>
#include <errno.h>
#include "nn/fail.h"
#include <sched.h>
#include <stdatomic.h>
#include <time.h>

#define PAR_MAX 64

#if defined(__aarch64__) || defined(__arm__)
#define cpu_relax() __asm__ __volatile__("yield" ::: "memory")
#elif defined(__x86_64__) || defined(__i386__)
#define cpu_relax() __asm__ __volatile__("pause" ::: "memory")
#else
#define cpu_relax() ((void)0)
#endif

typedef struct {                    /* one cache line per worker */
    pthread_t th;
    atomic_int sleeping;
    unsigned start_gen;             /* `gen` when the worker was created */
    pthread_mutex_t mu;
    pthread_cond_t cv;
    char pad[64];
} Worker;

static struct {
    int want;                       /* team size of the next region */
    int started;                    /* workers running (tids 1..started) */
    atomic_long spin_us;            /* a worker polls this long after a region, then sleeps */
    atomic_int pin;                 /* both set by each library call while idle workers read them */
    int configured;
    atomic_int quit;                /* moty_par_shutdown */
    /* the current region, published by `gen` */
    moty_par_fn fn; void *ctx; int64_t n; int chunk;
    atomic_int team;
    _Alignas(64) atomic_uint gen;
    _Alignas(64) atomic_llong next;
    _Alignas(64) atomic_int done;
    Worker w[PAR_MAX];
} P = { .want = 0 };

static _Thread_local int tl_tid = 0, tl_in = 0;

#if defined(__linux__)
static cpu_set_t base_set;          /* the process's CPUs, read before any pinning */
static int base_ok = -1;
static int base_mask(void) {
    if (base_ok < 0) base_ok = sched_getaffinity(0, sizeof base_set, &base_set) == 0;
    return base_ok;
}
#endif

static void pin_to(int k) {         /* k-th CPU of the process's affinity mask */
#if defined(__linux__)
    if (!base_mask()) return;
    int seen = 0;
    for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, &base_set) && seen++ == k) {
            cpu_set_t one; CPU_ZERO(&one); CPU_SET(c, &one);
            pthread_setaffinity_np(pthread_self(), sizeof one, &one);
            return;
        }
#else
    (void)k;
#endif
}

static void run_share(int tid) {
    tl_tid = tid; tl_in = 1;
    moty_par_fn fn = P.fn; void *ctx = P.ctx; int64_t n = P.n;
    if (P.chunk <= 0) {
        int64_t i0, i1;
        moty_par_range(n, tid, atomic_load_explicit(&P.team, memory_order_relaxed), &i0, &i1);
        if (i0 < i1) fn(ctx, i0, i1, tid);
    } else {
        for (;;) {
            int64_t i0 = atomic_fetch_add_explicit(&P.next, P.chunk, memory_order_relaxed);
            if (i0 >= n) break;
            fn(ctx, i0, i0 + P.chunk < n ? i0 + P.chunk : n, tid);
        }
    }
    tl_in = 0;
}

static double par_now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec * 1e-3; }

static void *worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    Worker *me = &P.w[tid];
    if (atomic_load_explicit(&P.pin, memory_order_relaxed)) pin_to(tid);
    /* not atomic_load(&P.gen): the region that made the caller start this
     * worker may be published before the thread first runs */
    unsigned seen = me->start_gen;
    for (;;) {
        unsigned g = seen;
        /* in the team: poll for the next region for spin_us, then sleep; out
         * of it (THREADS_DECODE below THREADS): sleep at once, leaving the
         * core to others. Asleep, a worker costs no CPU. */
        long spin_us = atomic_load_explicit(&P.spin_us, memory_order_relaxed);
        if (tid < atomic_load_explicit(&P.team, memory_order_relaxed) && spin_us > 0) {
            double t0 = par_now_us();
            for (long i = 1;; i++) {
                g = atomic_load_explicit(&P.gen, memory_order_acquire);
                if (g != seen || atomic_load_explicit(&P.quit, memory_order_relaxed)) break;
                cpu_relax();
                if (!(i & 63) && par_now_us() - t0 > spin_us) break;
            }
        }
        if (atomic_load(&P.quit)) return NULL;
        if (g == seen) {
            pthread_mutex_lock(&me->mu);
            atomic_store(&me->sleeping, 1);             /* seq_cst: pairs with the publisher */
            for (;;) {
                if (atomic_load(&P.quit)) { atomic_store(&me->sleeping, 0); pthread_mutex_unlock(&me->mu); return NULL; }
                g = atomic_load(&P.gen);
                if (g != seen) { if (tid < atomic_load(&P.team)) break; seen = g; }
                pthread_cond_wait(&me->cv, &me->mu);
            }
            atomic_store(&me->sleeping, 0);
            pthread_mutex_unlock(&me->mu);
        } else if (tid >= atomic_load(&P.team)) {       /* a region without this worker */
            seen = g; continue;
        }
        seen = g;
        run_share(tid);
        atomic_fetch_add_explicit(&P.done, 1, memory_order_release);
    }
    return NULL;
}

static void par_defaults(void) {             /* command line: MOTY_POOL_SPIN_US / MOTY_POOL_PIN */
    if (P.configured) return;
    const char *e = moty_getenv("MOTY_POOL_SPIN_US");
    atomic_store_explicit(&P.spin_us, e ? atol(e) : 1000, memory_order_relaxed);
    e = moty_getenv("MOTY_POOL_PIN");
    atomic_store_explicit(&P.pin, !(e && *e == '0'), memory_order_relaxed);
    P.configured = 1;
}
void moty_par_config(long spin_us, int pin) {
    atomic_store_explicit(&P.spin_us, spin_us < 0 ? 0 : spin_us, memory_order_relaxed);
    atomic_store_explicit(&P.pin, pin, memory_order_relaxed);
    P.configured = 1;
}

static void start_workers(int team) {
    par_defaults();
    for (int t = P.started + 1; t < team; t++) {
        Worker *w = &P.w[t];
        pthread_mutex_init(&w->mu, NULL); pthread_cond_init(&w->cv, NULL);
        w->start_gen = atomic_load(&P.gen);
        if (pthread_create(&w->th, NULL, worker, (void *)(intptr_t)t))
            moty_fail_code(MOTY_FAIL_OOM, "moty_par: pthread_create: %s", strerror(errno));
        P.started = t;
    }
}

/* join every worker (the library between requests: an idle process keeps no
 * moty threads); the next region starts them again */
void moty_par_shutdown(void) {
    if (!P.started) return;
    atomic_store(&P.quit, 1);
    for (int t = 1; t <= P.started; t++) {
        pthread_mutex_lock(&P.w[t].mu); pthread_cond_signal(&P.w[t].cv); pthread_mutex_unlock(&P.w[t].mu);
    }
    for (int t = 1; t <= P.started; t++) {
        pthread_join(P.w[t].th, NULL);
        pthread_mutex_destroy(&P.w[t].mu); pthread_cond_destroy(&P.w[t].cv);
        atomic_store(&P.w[t].sleeping, 0);
    }
    P.started = 0;
    atomic_store(&P.quit, 0);
}

/* the calling thread is tid 0 of every region: pinned to the first CPU of
 * the process mask for the duration of a library call (like
 * OMP_PROC_BIND=close), its own affinity restored afterwards */
void moty_par_enter(MotyParCaller *c) {
    par_defaults();
    c->pinned = 0;
#if defined(__linux__)
    if (atomic_load_explicit(&P.pin, memory_order_relaxed) && pthread_getaffinity_np(pthread_self(), sizeof c->mask, (cpu_set_t *)c->mask) == 0) {
        c->pinned = 1; pin_to(0);
    }
#endif
}
void moty_par_leave(MotyParCaller *c) {
#if defined(__linux__)
    if (c->pinned) pthread_setaffinity_np(pthread_self(), sizeof c->mask, (const cpu_set_t *)c->mask);
#endif
    c->pinned = 0;
}

void moty_par_for(int64_t n, int chunk, moty_par_fn fn, void *ctx) {
    if (n <= 0) return;
    int team = moty_par_threads();
    if (tl_in || team == 1) { fn(ctx, 0, n, tl_tid); return; }
    if (team > P.started + 1) start_workers(team);
    P.fn = fn; P.ctx = ctx; P.n = n; P.chunk = chunk;
    atomic_store_explicit(&P.team, team, memory_order_relaxed);
    atomic_store_explicit(&P.next, 0, memory_order_relaxed);
    atomic_store_explicit(&P.done, 0, memory_order_relaxed);
    atomic_fetch_add(&P.gen, 1);                        /* seq_cst: publish, then look for sleepers */
    for (int t = 1; t < team; t++)
        if (atomic_load(&P.w[t].sleeping)) {
            pthread_mutex_lock(&P.w[t].mu); pthread_cond_signal(&P.w[t].cv); pthread_mutex_unlock(&P.w[t].mu);
        }
    run_share(0);
    while (atomic_load_explicit(&P.done, memory_order_acquire) < team - 1) cpu_relax();
}

int moty_par_procs(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
#if defined(__linux__)
    if (base_mask() && CPU_COUNT(&base_set) < n) n = CPU_COUNT(&base_set);
#endif
    return n < 1 ? 1 : n > PAR_MAX ? PAR_MAX : (int)n;
}
int  moty_par_threads(void)      { if (!P.want) P.want = moty_par_procs(); return P.want; }
void moty_par_set_threads(int n) { if (n > 0) P.want = n > PAR_MAX ? PAR_MAX : n; }
int  moty_par_tid(void)          { return tl_tid; }
int  moty_par_active(void)       { return tl_in; }
const char *moty_par_backend(void) { return "pthread pool"; }

#else  /* neither: one thread */

void moty_par_for(int64_t n, int chunk, moty_par_fn fn, void *ctx) { (void)chunk; if (n > 0) fn(ctx, 0, n, 0); }
int  moty_par_threads(void)      { return 1; }
void moty_par_set_threads(int n) { (void)n; }
int  moty_par_procs(void)        { return 1; }
int  moty_par_tid(void)          { return 0; }
int  moty_par_active(void)       { return 0; }
const char *moty_par_backend(void) { return "serial"; }
void moty_par_config(long spin_us, int pin) { (void)spin_us; (void)pin; }
void moty_par_enter(MotyParCaller *c) { (void)c; }
void moty_par_leave(MotyParCaller *c) { (void)c; }
void moty_par_shutdown(void) {}
#endif
