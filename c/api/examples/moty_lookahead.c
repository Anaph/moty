/* moty_lookahead — lookahead decoding through the API: for each tile, greedy answer without a draft and with
 * the previous tile's answer as the draft (k = 2, 3, 4); prints whether the ids are identical, the decode
 * time and the verification steps (log level 2 line).
 *   moty_lookahead <container> <ids file> <image token> <threads> <threads_decode> <max_new> <tile.f32>... */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include "moty.h"
static double cpu_s(void) { struct rusage r; getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_stime.tv_sec + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6; }

typedef struct { int32_t ids[256]; int n; } Ans;
static int col(void *u, int32_t t, const char *p, int n) { Ans *a = u; (void)p; (void)n; if (a->n < 256) a->ids[a->n++] = t; return 0; }

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage\n"); return 2; }
    moty_options o; moty_options_init(&o); o.threads = atoi(argv[4]); o.threads_decode = atoi(argv[5]); o.ctx = 512; o.log_level = 2;
    moty_model *m; char err[256];
    if (moty_model_open(argv[1], &o, &m, err, sizeof err)) { puts(err); return 1; }
    int D = moty_model_hidden(m), itok = atoi(argv[3]), maxnew = atoi(argv[6]), nt = argc - 7;
    int32_t ids[256]; int n = 0; FILE *f = fopen(argv[2], "r"); while (n < 256 && fscanf(f, "%d", &ids[n]) == 1) n++; fclose(f);
    Ans *plain = calloc(nt, sizeof(Ans));
    int same[5] = {0}, cnt = 0;
    for (int i = 0; i < nt; i++) {
        float *rows = malloc((size_t)64 * D * 4); f = fopen(argv[7 + i], "rb");
        if (!f || fread(rows, 4, (size_t)64 * D, f) != (size_t)64 * D) return 1;
        fclose(f);
        moty_sampling s; moty_sampling_init(&s); s.max_new_tokens = maxnew; moty_stats st;
        moty_reset(m); double c0 = cpu_s();
        if (moty_generate(m, ids, n, rows, 64, itok, &s, col, &plain[i], &st)) { puts(moty_last_error(m)); return 1; }
        printf("%s: plain %d tokens, prefill %.3f s, decode %.3f s, cpu %.2f s\n", argv[7 + i], plain[i].n, st.prefill_s, st.decode_s, cpu_s() - c0);
        if (i > 0) for (int k = 2; k <= 4; k++) {
            Ans a = {0}; moty_sampling s2 = s; s2.draft = plain[i - 1].ids; s2.n_draft = plain[i - 1].n; s2.draft_k = k;
            moty_reset(m); double c1 = cpu_s();
            if (moty_generate(m, ids, n, rows, 64, itok, &s2, col, &a, &st)) { puts(moty_last_error(m)); return 1; }
            double cg = cpu_s() - c1;
            int eq = a.n == plain[i].n && !memcmp(a.ids, plain[i].ids, sizeof(int32_t) * (size_t)a.n);
            same[k] += eq;
            printf("  draft k=%d: %s, prefill %.3f s, decode %.3f s, cpu %.2f s\n", k, eq ? "identical" : "DIFFERENT", st.prefill_s, st.decode_s, cg);
        }
        if (i > 0) cnt++;
        free(rows);
    }
    for (int k = 2; k <= 4; k++) printf("k=%d: identical %d/%d\n", k, same[k], cnt);
    moty_model_close(m); moty_release_scratch();
    return 0;
}
