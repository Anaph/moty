/* moty_verify_cost — time of one forward over s+1 tokens (the pending token + s ids) after the VisionPsy
 * short-question prompt, i.e. what a lookahead verification step costs vs a single decode step.
 *   moty_verify_cost <container> <tile.f32> <ids file> <image token> <threads> <reps> */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "moty.h"

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
    moty_options o; moty_options_init(&o); o.threads = atoi(argv[5]); o.threads_decode = o.threads; o.ctx = 512;
    moty_model *m; char err[256];
    if (moty_model_open(argv[1], &o, &m, err, sizeof err)) { puts(err); return 1; }
    int D = moty_model_hidden(m); float *rows = malloc((size_t)64 * D * 4);
    FILE *f = fopen(argv[2], "rb"); if (!f || fread(rows, 4, (size_t)64 * D, f) != (size_t)64 * D) return 1; fclose(f);
    int32_t ids[256]; int n = 0; f = fopen(argv[3], "r"); while (n < 256 && fscanf(f, "%d", &ids[n]) == 1) n++; fclose(f);
    int itok = atoi(argv[4]), reps = atoi(argv[6]);
    moty_sampling s; moty_sampling_init(&s); s.max_new_tokens = 1; s.ignore_eos = 1;
    int32_t extra[8] = { 504, 2443, 314, 253, 2632, 284, 2537, 6945 };   /* any ids: the cost does not depend on them */
    for (int k = 0; k <= 8; k++) {
        double t = 0, tmin = 1e9;
        for (int r = 0; r < reps; r++) {
            moty_stats st; moty_reset(m);
            moty_sampling s2 = s; s2.max_new_tokens = k ? 1 : 2;       /* k = 0: time one plain decode step */
            if (moty_generate(m, ids, n, rows, 64, itok, &s2, NULL, NULL, &st)) { puts(moty_last_error(m)); return 1; }
            double x = st.decode_s;
            if (k) {   /* continuation: the pending token + k ids in one forward */
                if (moty_generate(m, extra, k, NULL, 0, itok, &s, NULL, NULL, &st)) { puts(moty_last_error(m)); return 1; }
                x = st.prefill_s;
            }
            t += x; if (x < tmin) tmin = x;
        }
        printf("forward of %d token(s): mean %.1f ms, min %.1f ms\n", k + 1, 1e3 * t / reps, 1e3 * tmin);
    }
    moty_model_close(m); moty_release_scratch(); free(rows);
    return 0;
}
