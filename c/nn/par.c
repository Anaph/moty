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

#elif defined(MOTY_THREADPOOL)
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>

#define PAR_MAX 64

#if defined(__aarch64__) || defined(__arm__)
#define cpu_relax() __asm__ __volatile__("yield" ::: "memory")
#elif defined(__x86_64__) || defined(__i386__)
#define cpu_relax() __asm__ __volatile__("pause" ::: "memory")
#else
#define cpu_relax() ((void)0)
#endif

typedef struct {                    /* one cache line per worker */
    atomic_int sleeping;
    unsigned start_gen;             /* `gen` when the worker was created */
    pthread_mutex_t mu;
    pthread_cond_t cv;
    char pad[64];
} Worker;

static struct {
    int want;                       /* team size of the next region */
    int started;                    /* workers running (tids 1..started) */
    long spin;                      /* polls before a worker sleeps */
    int pin;
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

static void *worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    Worker *me = &P.w[tid];
    if (P.pin) pin_to(tid);
    /* not atomic_load(&P.gen): the region that made the caller start this
     * worker may be published before the thread first runs */
    unsigned seen = me->start_gen;
    for (;;) {
        unsigned g = seen;
        /* in the team: spin for the next region; out of it (THREADS_DECODE
         * below THREADS): sleep at once, leaving the core to others */
        if (tid < atomic_load_explicit(&P.team, memory_order_relaxed))
            for (long i = 0; i < P.spin; i++) {
                g = atomic_load_explicit(&P.gen, memory_order_acquire);
                if (g != seen) break;
                cpu_relax();
            }
        if (g == seen) {
            pthread_mutex_lock(&me->mu);
            atomic_store(&me->sleeping, 1);             /* seq_cst: pairs with the publisher */
            for (;;) {
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

static void start_workers(int team) {
    if (!P.spin) {
        const char *e = getenv("MOTY_POOL_SPIN");
        P.spin = e ? atol(e) : 200000;                  /* ~GOMP_SPINCOUNT of the OpenMP hot tune */
        if (P.spin < 1) P.spin = 1;
        e = getenv("MOTY_POOL_PIN");
        P.pin = !(e && *e == '0');
        if (P.pin) pin_to(0);                           /* caller = tid 0, like OMP_PROC_BIND=close */
    }
    for (int t = P.started + 1; t < team; t++) {
        Worker *w = &P.w[t];
        pthread_mutex_init(&w->mu, NULL); pthread_cond_init(&w->cv, NULL);
        w->start_gen = atomic_load(&P.gen);
        pthread_t th; pthread_attr_t at; pthread_attr_init(&at);
        if (pthread_create(&th, &at, worker, (void *)(intptr_t)t)) { perror("moty_par: pthread_create"); exit(1); }
        pthread_detach(th); pthread_attr_destroy(&at);
        P.started = t;
    }
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
#endif
