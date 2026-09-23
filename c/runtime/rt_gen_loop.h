/* Generazione: step_chunked, run_ref (ref.json), gen_turn.
 * Estratto da runtime.h (P2). Include DOPO Cfg/Layer/Model
 * e le dichiarazioni degli hook. Un'istanza per TU (tutto static). */
#ifndef RT_GEN_LOOP_H
#define RT_GEN_LOOP_H
#include "util/prof.h"    /* OP_T/OP_ACC per-op table (no-ops without MOTY_PROF) */

/* ---------- prefill a blocchi (PREFILL_CHUNK) ----------
 * Le attivazioni del prefill crescono con S (mlp: 2*S*inter f32 — 1.6 GB a
 * S=4096 su un 4B) e lo scratch statico di matmul_q_s resta a S*I per sempre:
 * spezzare il prompt in blocchi <= C limita entrambi a una costante. E' un
 * opt-in (default 0 = spento) perche' con MEM_GB ogni step() rilegge i layer
 * streamati dal disco: N blocchi = N riletture del prompt.
 * Bit-esattezza: ogni operazione per-token dipende solo dalla posizione
 * ASSOLUTA (RoPE, scrittura KV, ricorrenza deltanet, PLE) e l'attention legge
 * la KV scritta dalle posizioni precedenti, identica comunque si spezzi;
 * matmul_q_s quantizza le attivazioni PER RIGA, quindi e' batch-invariante.
 * g_skip_logits: sui blocchi intermedi il motore esce da step() PRIMA di
 * final-norm/lm_head (e dello stash TTA) e ritorna NULL — i logits (e lo
 * stash) esistono solo per l'ultimo token del prompt, come non-chunked. */
static int g_skip_logits = 0;

static float *step_chunked(Model *m, const int *ids, int S, int pos_base) {
    int C = g_prefill_chunk;
    if (C <= 0 || S <= C) return step(m, ids, S, pos_base);
    int done = 0;
    for (; S - done > C; done += C) {
        g_skip_logits = 1;
        float *lo = step(m, ids + done, C, pos_base + done);
        g_skip_logits = 0;
        if (lo) free(lo);              /* i motori ritornano NULL quando saltano */
    }
    return step(m, ids + done, S - done, pos_base + done);
}

/* ---------- ref.json (validazione) ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a) { fprintf(stderr, "ref.json: manca %s\n", key); exit(1); }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

static int run_ref(Model *m, const char *refpath) {
    char *buf = slurp_file(refpath, NULL);
    jval *ref = json_parse(buf);
    int np, nfull;
    int *prompt = read_int_array(ref,"prompt_ids",&np);
    int *full   = read_int_array(ref,"full_ids",&nfull);
    int n_new = nfull - np;
    if (n_new <= 0) { fprintf(stderr,"ref.json: full_ids non estende prompt_ids\n"); return 1; }
    kv_alloc(m, nfull + 1);
    int *out = malloc(nfull * sizeof(int));
    memcpy(out, prompt, np*sizeof(int));
    g_temp = 0;                                    /* la validazione e' greedy */
    double t0 = now_s();
    float *logit = step_chunked(m, prompt, np, 0);
    /* REF_LOGITS=<file>: raw f32 logits of the last prompt position, for a
     * logit-level comparison against the reference (tools/ref/cmp_logits.py) */
    const char *lpath = getenv("REF_LOGITS");
    if (lpath && *lpath) {
        FILE *lf = fopen(lpath, "wb");
        if (!lf || fwrite(logit, sizeof(float), m->c.vocab, lf) != (size_t)m->c.vocab) { perror(lpath); exit(1); }
        fclose(lf);
    }
    int len = np;
    for (int s = 0; s < n_new; s++) {
        int best = argmax_v(logit, m->c.vocab);
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        logit = step(m, &out[len-1], 1, len-1);
    }
    double dt = now_s() - t0;
    int match = 0;
    printf("Reference: "); for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : "); for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    printf("Speed: %.2f tok/s | PEAK RSS %.2f GB\n", n_new/dt, rss_gb());
    json_free(ref); free(buf); free(prompt); free(full); free(out);
    return match == n_new ? 0 : 2;
}

/* ---------- PPL=<file.json> {"ids":[...]}: teacher-forced quality probe ----------
 * Feeds ids one at a time (the decode path) and scores every next token:
 * prints perplexity over ids[1..n) and, with PPL_OUT=<file>, writes the
 * per-position argmax ids (one per line) so top-1 agreement against the
 * reference model can be computed offline. PPL_N caps the token count. */
static int run_ppl(Model *m, const char *path) {
    char *buf = slurp_file(path, NULL);
    jval *js = json_parse(buf);
    int n; int *ids = read_int_array(js, "ids", &n);
    const char *cap = getenv("PPL_N");
    if (cap && atoi(cap) > 1 && atoi(cap) < n) n = atoi(cap);
    if (n < 2) { fprintf(stderr, "PPL: need at least 2 ids\n"); return 1; }
    kv_alloc(m, n + 1);
    const char *opath = getenv("PPL_OUT");
    FILE *of = opath && *opath ? fopen(opath, "w") : NULL;
    int V = m->c.vocab;
    double nll = 0; double t0 = now_s();
    for (int t = 0; t + 1 < n; t++) {
        float *lo = step(m, &ids[t], 1, t);
        float mx = lo[0]; int am = 0;
        for (int v = 1; v < V; v++) if (lo[v] > mx) { mx = lo[v]; am = v; }
        double se = 0; for (int v = 0; v < V; v++) se += exp((double)lo[v] - mx);
        nll += -((double)lo[ids[t+1]] - mx - log(se));
        if (of) fprintf(of, "%d\n", am);
        free(lo);
    }
    if (of) fclose(of);
    printf("PPL: %.4f over %d tokens (%.1f s)\n", exp(nll / (n - 1)), n - 1, now_s() - t0);
    json_free(js); free(buf); free(ids);
    return 0;
}

/* ---------- generazione di un turno ----------
 * hist[len..len+k) = token nuovi da prefillare; genera fino a n_new token o stop.
 * Stampa il testo su stdout se echo!=0. Ritorna il numero di token generati
 * (stop incluso se emesso); *stopped=1 se l'ultimo token e' uno stop. */
/* per-op table (util/prof.h, MOTY_PROF builds): ms per phase, then reset */
static void prof_op_report(const char *what, int ntok, double wall) {
#ifdef MOTY_PROF
    static const char *nm[OP_N] = OP_NAMES;
    double tot = 0; for (int i = 0; i < OP_N; i++) tot += moty_prof_op[i];
    fprintf(stderr, "[prof-op] %s: %d tok, wall %.1f ms (%.2f ms/tok)\n", what, ntok, wall*1e3, wall*1e3/(ntok?ntok:1));
    for (int i = 0; i < OP_N; i++) if (moty_prof_op[i] > 0)
        fprintf(stderr, "[prof-op]   %-22s %9.2f ms %6.2f ms/tok %5.1f%%\n", nm[i], moty_prof_op[i]*1e3,
                moty_prof_op[i]*1e3/(ntok?ntok:1), 100*moty_prof_op[i]/wall);
    fprintf(stderr, "[prof-op]   %-22s %9.2f ms %6.2f ms/tok %5.1f%%\n", "(unattributed)", (wall-tot)*1e3,
            (wall-tot)*1e3/(ntok?ntok:1), 100*(wall-tot)/wall);
    for (int i = 0; i < OP_N; i++) moty_prof_op[i] = 0;
#else
    (void)what; (void)ntok; (void)wall;
#endif
}

static int gen_turn(Model *m, Tok *T, int *hist, int len, int k, int n_new, int echo, int *stopped) {
    int dump = g_tokens_dump;
    static int ignore_eos = -1;            /* IGNORE_EOS=1: fixed-length runs for benchmarks */
    if (ignore_eos < 0) ignore_eos = getenv("IGNORE_EOS") && atoi(getenv("IGNORE_EOS"));
#ifdef MOTY_PROF
    for (int i = 0; i < OP_N; i++) moty_prof_op[i] = 0;
#endif
    double t0 = now_s();
    float *logit = step_chunked(m, hist + len, k, len);  /* PREFILL (a blocchi se PREFILL_CHUNK) */
    ENGINE_LOGITS_HOOK(m, logit);
    double tpre = now_s() - t0;
    prof_op_report("prefill", k, tpre);
    int base = len + k, ng = 0; *stopped = 0;
    /* THREADS_DECODE=N: team size for the decode steps only. Prefill is
     * compute-bound and wants every core; single-token decode is
     * bandwidth-bound and runs ~100 short parallel regions per token, each
     * ending in a barrier that waits for the slowest thread — on a core
     * shared with another busy process one fewer thread can be faster. */
    static int td = -1;
    if (td < 0) td = getenv("THREADS_DECODE") ? atoi(getenv("THREADS_DECODE")) : 0;
    int th_prefill = omp_get_max_threads();
    if (td > 0) omp_set_num_threads(td);
    t0 = now_s();
    for (int s = 0; s < n_new; s++) {
        OP_T(t_s);
        int t = pick_tok(&m->base.scr, logit, m->c.vocab);
        OP_ACC(OP_SAMPLE, t_s);
        free(logit); logit = NULL;
        hist[base + ng++] = t;
        ENGINE_OBSERVE(m, t);
        if (dump) fprintf(stderr, "%d ", t);
        if (is_stop(t) && !ignore_eos) { *stopped = 1; break; }
        if (echo) {
            char buf[64]; int bn = tok_decode(T, &t, 1, buf, 63);
            fwrite(buf, 1, bn, stdout); fflush(stdout);
        }
        if (s == n_new - 1) break;
        logit = step(m, &hist[base + ng - 1], 1, base + ng - 1);
        ENGINE_LOGITS_HOOK(m, logit);
    }
    if (logit) free(logit);
    if (dump) fprintf(stderr, "\n");
    double tgen = now_s() - t0;
    if (td > 0) omp_set_num_threads(th_prefill);
    prof_op_report("decode", ng, tgen);
    fprintf(stderr, "\n[" ENGINE_TAG "] prefill %d tok in %.2fs (%.1f tok/s) | decode %d tok in %.2fs (%.2f tok/s) | RSS %.2f GB\n",
            k, tpre, k/(tpre>1e-9?tpre:1e-9), ng, tgen, ng/(tgen>1e-9?tgen:1e-9), rss_gb());
    return ng;
}

#endif /* RT_GEN_LOOP_H */
