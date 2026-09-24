/* nn/par.c: the parallel loops (OpenMP wrappers by default, the pthread
 * pool with MOTY_THREADPOOL). Plain C, 0 = ok / 1 = fail; gtest glue in
 * par_gtest.cc. Checks: every index exactly once, the static split,
 * dynamic chunks, team size changes (THREADS_DECODE), nested calls, many
 * short regions on the sleep path, and a Q4R4 matmul that is bit-identical
 * at 1 and 4 threads (each output row is computed by one thread). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include "nn/nn.h"

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

typedef struct {
    atomic_int *hits; atomic_int calls; int *tid_of; int64_t *lo, *hi; int bad;
} Cov;

static void cov_fn(void *c_, int64_t i0, int64_t i1, int tid) {
    Cov *c = c_;
    atomic_fetch_add(&c->calls, 1);
    if (tid < 0 || tid >= 64 || i0 >= i1) c->bad = 1;
    if (c->lo) { c->lo[tid] = i0; c->hi[tid] = i1; }
    for (int64_t i = i0; i < i1; i++) { atomic_fetch_add(&c->hits[i], 1); if (c->tid_of) c->tid_of[i] = tid; }
}

/* static: one contiguous range per thread, split like schedule(static) */
int par_static_split(void) {
    const int64_t ns[] = { 1, 3, 7, 64, 1000, 65537 };
    for (int nt = 1; nt <= 6; nt++) {
        moty_par_set_threads(nt);
        CHECK(moty_par_threads() == nt);
        for (size_t k = 0; k < sizeof ns / sizeof *ns; k++) {
            int64_t n = ns[k];
            Cov c = { calloc(n, sizeof(atomic_int)), 0, NULL, calloc(64, 8), calloc(64, 8), 0 };
            for (int t = 0; t < 64; t++) c.lo[t] = c.hi[t] = -1;
            moty_par_for(n, 0, cov_fn, &c);
            CHECK(!c.bad);
            for (int64_t i = 0; i < n; i++) CHECK(atomic_load(&c.hits[i]) == 1);
            for (int t = 0; t < nt; t++) {
                int64_t i0, i1; moty_par_range(n, t, nt, &i0, &i1);
                if (i0 < i1) { CHECK(c.lo[t] == i0); CHECK(c.hi[t] == i1); }
                else CHECK(c.lo[t] == -1);
            }
            for (int t = nt; t < 64; t++) CHECK(c.lo[t] == -1);
            free(c.hits); free(c.lo); free(c.hi);
        }
    }
    return 0;
}

/* dynamic: chunks of `chunk` indices, each index once, only team tids */
int par_dynamic_chunks(void) {
    moty_par_set_threads(4);
    for (int chunk = 1; chunk <= 64; chunk *= 8) {
        int64_t n = 10007;
        Cov c = { calloc(n, sizeof(atomic_int)), 0, malloc(n * sizeof(int)), NULL, NULL, 0 };
        moty_par_for(n, chunk, cov_fn, &c);
        CHECK(!c.bad);
        CHECK(atomic_load(&c.calls) == (int)((n + chunk - 1) / chunk));
        for (int64_t i = 0; i < n; i++) { CHECK(atomic_load(&c.hits[i]) == 1); CHECK(c.tid_of[i] < 4); }
        /* a chunk runs on one thread */
        for (int64_t i = 0; i < n; i++) CHECK(c.tid_of[i] == c.tid_of[i / chunk * chunk]);
        free(c.hits); free(c.tid_of);
    }
    return 0;
}

/* THREADS_DECODE: a smaller team for a while, then the full team again */
int par_team_resize(void) {
    int sizes[] = { 4, 3, 1, 2, 4, 3, 4 };
    for (size_t k = 0; k < sizeof sizes / sizeof *sizes; k++) {
        moty_par_set_threads(sizes[k]);
        for (int rep = 0; rep < 200; rep++) {
            int64_t n = 257;
            Cov c = { calloc(n, sizeof(atomic_int)), 0, malloc(n * sizeof(int)), NULL, NULL, 0 };
            moty_par_for(n, rep & 1 ? 8 : 0, cov_fn, &c);
            CHECK(!c.bad);
            for (int64_t i = 0; i < n; i++) { CHECK(atomic_load(&c.hits[i]) == 1); CHECK(c.tid_of[i] < sizes[k]); }
            free(c.hits); free(c.tid_of);
        }
    }
    return 0;
}

/* a region started inside a region runs serially on the calling thread */
typedef struct { atomic_int *hits; int bad; } Nest;
static void inner_fn(void *c_, int64_t i0, int64_t i1, int tid) {
    Nest *c = c_;
    if (tid != moty_par_tid()) c->bad = 1;
    for (int64_t i = i0; i < i1; i++) atomic_fetch_add(&c->hits[i], 1);
}
static void outer_fn(void *c_, int64_t i0, int64_t i1, int tid) {
    Nest *c = c_;
    if (!moty_par_active()) c->bad = 1;
    for (int64_t i = i0; i < i1; i++) {
        Nest in = { c->hits + 100 + i * 50, 0 };
        moty_par_for(50, 0, inner_fn, &in);
        if (in.bad) c->bad = 1;
        atomic_fetch_add(&c->hits[i], 1);
    }
}
int par_nested(void) {
    moty_par_set_threads(4);
    atomic_int *hits = calloc(100 + 100 * 50, sizeof(atomic_int));
    Nest c = { hits, 0 };
    moty_par_for(100, 0, outer_fn, &c);
    CHECK(!c.bad);
    CHECK(!moty_par_active());
    for (int i = 0; i < 100 + 100 * 50; i++) CHECK(atomic_load(&hits[i]) == 1);
    free(hits);
    return 0;
}

/* many short regions with workers that sleep after one poll: every wakeup
 * must arrive (a lost one hangs the test) */
static void sum_fn(void *c_, int64_t i0, int64_t i1, int tid) {
    atomic_llong *s = c_;
    for (int64_t i = i0; i < i1; i++) atomic_fetch_add(s, i);
}
int par_sleep_wakeup(void) {
    setenv("MOTY_POOL_SPIN", "1", 1);          /* read when the pool starts (this process) */
    for (int rep = 0; rep < 20000; rep++) {
        moty_par_set_threads(rep % 7 == 0 ? 2 : 4);
        atomic_llong s = 0;
        moty_par_for(64, rep & 1 ? 4 : 0, sum_fn, &s);
        CHECK(atomic_load(&s) == 64 * 63 / 2);
    }
    return 0;
}

/* the Q4R4 matmul (decode and prefill tiles) gives the same bits at any
 * team size */
static uint64_t pr_rng = 777;
static float pr_frnd(void) {
    pr_rng ^= pr_rng << 13; pr_rng ^= pr_rng >> 7; pr_rng ^= pr_rng << 17;
    return (float)((pr_rng >> 11) * (1.0 / 9007199254740992.0)) - 0.5f;
}
int par_q4r4_threads_bitexact(void) {
    enum { O = 203, I = 256, SMAX = 37 };
    int nb = I / 32, O4 = (O + 3) / 4;
    float *W = malloc(sizeof(float) * O * I), *x = malloc(sizeof(float) * SMAX * I);
    uint8_t *q4 = calloc((size_t)O4 * nb * 64, 1); uint16_t *s16 = calloc((size_t)O4 * nb * 4, 2);
    for (int i = 0; i < O * I; i++) W[i] = pr_frnd();
    for (int i = 0; i < SMAX * I; i++) x[i] = pr_frnd() * 2.f;
    for (int b = 0; b < O4; b++)
        moty_pack_q4r4_block(W + (size_t)b * 4 * I, O - b * 4 < 4 ? O - b * 4 : 4, I, q4 + (size_t)b * nb * 64, s16 + (size_t)b * nb * 4);
    float *y1 = malloc(sizeof(float) * SMAX * O), *yn = malloc(sizeof(float) * SMAX * O);
    int Ss[] = { 1, 5, 8, SMAX };
    for (size_t k = 0; k < sizeof Ss / sizeof *Ss; k++) {
        moty_par_set_threads(1);
        moty_matmul_q4r4_s(y1, x, q4, s16, Ss[k], I, O);
        for (int nt = 2; nt <= 4; nt++) {
            moty_par_set_threads(nt);
            moty_matmul_q4r4_s(yn, x, q4, s16, Ss[k], I, O);
            CHECK(!memcmp(y1, yn, sizeof(float) * Ss[k] * O));
        }
    }
    free(W); free(x); free(q4); free(s16); free(y1); free(yn);
    return 0;
}
