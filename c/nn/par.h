/* par.h — the parallel loops of libmoty-nn, one API for two backends.
 *
 * Default build (-fopenmp): thin wrappers over OpenMP, the same schedules
 * the kernels used as pragmas. MOTY_THREADPOOL build (no OpenMP runtime,
 * e.g. a clang toolchain without libomp): a small pool of persistent
 * pthread workers, pinned one per CPU, spinning between regions then
 * sleeping on a condition variable (nn/par.c).
 *
 *   moty_par_for(n, 0, fn, ctx)      static: each thread gets one contiguous
 *                                    range, split like schedule(static)
 *   moty_par_for(n, chunk, fn, ctx)  dynamic: chunks of `chunk` indices
 *                                    handed out in order, like
 *                                    schedule(dynamic, chunk)
 *
 * fn(ctx, i0, i1, tid) runs indices [i0, i1) on thread tid (0 = caller);
 * a region ends when every index is done (implicit barrier). Called from
 * inside a region, moty_par_for runs serially on the calling thread. One
 * caller thread at a time (moty's engines are single-threaded outside the
 * kernels). */
#ifndef NN_PAR_H
#define NN_PAR_H
#include <stdint.h>

typedef void (*moty_par_fn)(void *ctx, int64_t i0, int64_t i1, int tid);

void moty_par_for(int64_t n, int chunk, moty_par_fn fn, void *ctx);
int  moty_par_threads(void);       /* team size of the next region */
void moty_par_set_threads(int n);  /* THREADS / THREADS_DECODE */
int  moty_par_procs(void);         /* online CPUs */
int  moty_par_tid(void);           /* 0 outside a region */
int  moty_par_active(void);        /* inside a region */
const char *moty_par_backend(void);

/* the static split of [0, n) for thread t of nt (libgomp's: the first
 * n % nt threads get one more index) */
static inline void moty_par_range(int64_t n, int t, int nt, int64_t *i0, int64_t *i1) {
    int64_t q = n / nt, r = n % nt;
    *i0 = t * q + (t < r ? t : r);
    *i1 = *i0 + q + (t < r);
}
#endif
