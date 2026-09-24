/* moty_probe — two library properties a host depends on, measured:
 *   abort latency: moty_abort from another thread while moty_generate is in
 *     the prefill (and in the decode); time until moty_generate returns;
 *   idle CPU: process CPU time over 3 s with a model open and no call
 *     running, after a generate (pool workers spinning, then asleep).
 *
 *   moty_probe <model dir> <rows.f32 or -> <threads> <prefill_chunk> <pool_spin_us> <abort_after_ms>
 * The prompt is the LFM2.5-VL image prompt when rows are given (272 ids). */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include "moty.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double cpu_s(void) {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_stime.tv_sec + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6;
}
typedef struct { moty_model *h; int32_t *ids; int n; float *rows; int nrows; int ntok; moty_status rc; double t_end; moty_stats st; } Job;
static void *run(void *p) {
    Job *j = p;
    moty_sampling s; moty_sampling_init(&s); s.max_new_tokens = j->ntok; s.ignore_eos = 1;
    j->rc = moty_generate(j->h, j->ids, j->n, j->rows, j->nrows, moty_model_image_token(j->h), &s, NULL, NULL, &j->st);
    j->t_end = now();
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s <dir> <rows|-> <threads> <prefill_chunk> <spin_us> <abort_after_ms>\n", argv[0]); return 2; }
    moty_options o; moty_options_init(&o);
    o.threads = atoi(argv[3]); o.prefill_chunk = atoi(argv[4]); o.pool_spin_us = atoi(argv[5]); o.ctx = 1024;
    int after_ms = atoi(argv[6]);
    moty_model *h; char err[512];
    if (moty_model_open(argv[1], &o, &h, err, sizeof err)) { fprintf(stderr, "open: %s\n", err); return 1; }
    float *rows = NULL; int nrows = 0; int D = moty_model_hidden(h);
    if (strcmp(argv[2], "-")) {
        FILE *f = fopen(argv[2], "rb"); if (!f) { perror(argv[2]); return 1; }
        fseek(f, 0, SEEK_END); long nb = ftell(f); fseek(f, 0, SEEK_SET);
        nrows = (int)(nb / ((long)D * 4)); rows = malloc((size_t)nb);
        if (fread(rows, 1, (size_t)nb, f) != (size_t)nb) return 1;
        fclose(f);
    }
    static char text[65536]; int n = 0;
    if (rows) {
        n += snprintf(text + n, sizeof text - n, "<|image_start|>");
        for (int i = 0; i < nrows; i++) n += snprintf(text + n, sizeof text - n, "<image>");
        n += snprintf(text + n, sizeof text - n, "<|image_end|>Describe the image.");
    } else snprintf(text, sizeof text, "Describe a quiet street at night in great detail, with every sound and light.");
    int32_t ids[2048]; int k = moty_tokenize(h, text, 1, 1, ids, 2048);
    Job j = { h, ids, k, rows, nrows, 64, 0, 0, {0} };

    /* full run: reference timings */
    run(&j);
    printf("%s | prompt %d tok: prefill %.2f s, decode %d tok %.2f s\n", moty_version(), k, j.st.prefill_s, j.st.new_tokens, j.st.decode_s);
    /* idle CPU with the model open */
    double c0 = cpu_s(), w0 = now();
    struct timespec ts = {3, 0}; nanosleep(&ts, NULL);
    printf("idle 3 s after a call (spin %d us): process CPU %.3f s over %.2f s wall\n", o.pool_spin_us, cpu_s() - c0, now() - w0);
    /* abort during the prefill, then during the decode */
    const char *phase[2] = { "prefill", "decode" };
    double at[2] = { after_ms / 1000.0, j.st.prefill_s + 0.5 };
    for (int p = 0; p < 2; p++) {
        moty_reset(h);
        pthread_t th; double t0 = now();
        pthread_create(&th, NULL, run, &j);
        struct timespec d = { (time_t)at[p], (long)((at[p] - (time_t)at[p]) * 1e9) }; nanosleep(&d, NULL);
        double ta = now(); moty_abort(h);
        pthread_join(th, NULL);
        printf("abort in %s at %.2f s: returned %d after %.3f s\n", phase[p], ta - t0, j.rc, j.t_end - ta);
        moty_abort_clear(h);
    }
    moty_model_close(h);
    moty_release_scratch();
    free(rows);
    return 0;
}
