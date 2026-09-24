/* moty_cycle — the library the way a per-request host uses it: for each
 * cycle open -> generate -> close -> moty_release_scratch, printing speed,
 * RSS and the thread count, so memory and threads can be checked to come
 * back to the baseline (docs/api.md).
 *
 *   moty_cycle <model dir> <cycles> [rows.f32] [threads] [threads_decode] [new_tokens] [spin_us]
 *
 * With rows.f32 the prompt is the LFM2.5-VL image prompt (image_start +
 * one <image> per row + image_end + "Describe the image.", chat template);
 * without, "Describe a quiet street at night.". Greedy. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "moty.h"
#if defined(__GLIBC__)
#include <malloc.h>
#endif

static long status_kb(const char *key) {
    FILE *f = fopen("/proc/self/status", "r"); char l[256]; long v = -1; size_t kl = strlen(key);
    while (f && fgets(l, sizeof l, f)) if (!strncmp(l, key, kl)) { v = atol(l + kl + 1); break; }
    if (f) fclose(f);
    return v;
}
static long meminfo_kb(const char *key) {
    FILE *f = fopen("/proc/meminfo", "r"); char l[256]; long v = -1; size_t kl = strlen(key);
    while (f && fgets(l, sizeof l, f)) if (!strncmp(l, key, kl)) { v = atol(l + kl + 1); break; }
    if (f) fclose(f);
    return v;
}
static int on_token(void *u, int32_t t, const char *piece, int n) {
    (void)t;
    if (u && n > 0) fwrite(piece, 1, (size_t)n, stdout);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model dir> <cycles> [rows.f32] [threads] [threads_decode] [new_tokens] [spin_us]\n", argv[0]); return 2; }
    const char *dir = argv[1]; int cycles = atoi(argv[2]);
    const char *rp = argc > 3 && strcmp(argv[3], "-") ? argv[3] : NULL;
    moty_options o; moty_options_init(&o);
    o.threads = argc > 4 ? atoi(argv[4]) : 0;
    o.threads_decode = argc > 5 ? atoi(argv[5]) : 0;
    int ngen = argc > 6 ? atoi(argv[6]) : 64;
    if (argc > 7) o.pool_spin_us = atoi(argv[7]);
    o.ctx = 1024;
    printf("%s | base: RSS %ld kB, threads %ld, MemAvailable %ld kB\n", moty_version(),
           status_kb("VmRSS:"), status_kb("Threads:"), meminfo_kb("MemAvailable:"));
    for (int c = 0; c < cycles; c++) {
        moty_model *h; char err[512];
        moty_status rc = moty_model_open(dir, &o, &h, err, sizeof err);
        if (rc) { fprintf(stderr, "open: %d %s\n", rc, err); return 1; }
        float *rows = NULL; int nrows = 0, D = moty_model_hidden(h);
        if (rp) {
            FILE *f = fopen(rp, "rb"); if (!f) { perror(rp); return 1; }
            fseek(f, 0, SEEK_END); long nb = ftell(f); fseek(f, 0, SEEK_SET);
            nrows = (int)(nb / ((long)D * 4)); rows = malloc((size_t)nb);
            if (fread(rows, 1, (size_t)nb, f) != (size_t)nb) { perror(rp); return 1; }
            fclose(f);
        }
        static char text[65536]; int n = 0;
        if (rp) {
            n += snprintf(text + n, sizeof text - n, "<|image_start|>");
            for (int i = 0; i < nrows; i++) n += snprintf(text + n, sizeof text - n, "<image>");
            n += snprintf(text + n, sizeof text - n, "<|image_end|>Describe the image.");
        } else snprintf(text, sizeof text, "Describe a quiet street at night.");
        int32_t ids[2048];
        int k = moty_tokenize(h, text, 1, 1, ids, 2048);
        if (k < 0) { fprintf(stderr, "tokenize: %s\n", moty_last_error(h)); return 1; }
        moty_sampling s; moty_sampling_init(&s); s.max_new_tokens = ngen; s.ignore_eos = 1;
        moty_stats st;
        rc = moty_generate(h, ids, k, rows, nrows, moty_model_image_token(h), &s, on_token, c == 0 ? (void *)1 : NULL, &st);
        if (c == 0) printf("\n");
        long rss_open = status_kb("VmRSS:"), thr = status_kb("Threads:");
        if (rc) { fprintf(stderr, "generate: %d %s\n", rc, moty_last_error(h)); return 1; }
        double load = moty_model_load_s(h);
        moty_model_close(h);
        free(rows);
        long rss_close = status_kb("VmRSS:");
        moty_release_scratch();
#if defined(__GLIBC__)
        if (getenv("CYCLE_TRIM")) malloc_trim(0);          /* diagnosis: heap fragmentation vs growth */
#endif
#if defined(__GLIBC__) && (__GLIBC__ > 2 || __GLIBC_MINOR__ >= 33)
        { struct mallinfo2 mi = mallinfo2(); printf("heap in use %zu B, arena %zu B | ", mi.uordblks, mi.arena); }
#endif
        printf("cycle %d: load %.2f s | prefill %d tok %.2f s (%.1f tok/s) | decode %d tok %.2f s (%.2f tok/s) | "
               "RSS open %ld kB, closed %ld kB, released %ld kB | threads open %ld, released %ld | MemAvailable %ld kB\n",
               c, load, st.prompt_tokens, st.prefill_s, st.prompt_tokens / st.prefill_s, st.new_tokens, st.decode_s,
               st.new_tokens / st.decode_s, rss_open, rss_close, status_kb("VmRSS:"), thr, status_kb("Threads:"),
               meminfo_kb("MemAvailable:"));
        fflush(stdout);
    }
    return 0;
}
