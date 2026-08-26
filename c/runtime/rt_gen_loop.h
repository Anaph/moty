/* Generazione: step_chunked, run_ref (ref.json), gen_turn.
 * Estratto da runtime.h (P2). Include DOPO Cfg/Layer/Model
 * e le dichiarazioni degli hook. Un'istanza per TU (tutto static). */
#ifndef RT_GEN_LOOP_H
#define RT_GEN_LOOP_H

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
static int g_prefill_chunk = 0;
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

/* ---------- generazione di un turno ----------
 * hist[len..len+k) = token nuovi da prefillare; genera fino a n_new token o stop.
 * Stampa il testo su stdout se echo!=0. Ritorna il numero di token generati
 * (stop incluso se emesso); *stopped=1 se l'ultimo token e' uno stop. */
static int g_tokens_dump = 0;           /* TOKENS=1 (parse_env) */
static int gen_turn(Model *m, Tok *T, int *hist, int len, int k, int n_new, int echo, int *stopped) {
    int dump = g_tokens_dump;
    double t0 = now_s();
    float *logit = step_chunked(m, hist + len, k, len);  /* PREFILL (a blocchi se PREFILL_CHUNK) */
    ENGINE_LOGITS_HOOK(m, logit);
    double tpre = now_s() - t0;
    int base = len + k, ng = 0; *stopped = 0;
    t0 = now_s();
    for (int s = 0; s < n_new; s++) {
        int t = pick_tok(&m->scr, logit, m->c.vocab);
        free(logit); logit = NULL;
        hist[base + ng++] = t;
        ENGINE_OBSERVE(m, t);
        if (dump) fprintf(stderr, "%d ", t);
        if (is_stop(t)) { *stopped = 1; break; }
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
    fprintf(stderr, "\n[" ENGINE_TAG "] prefill %d tok in %.2fs (%.1f tok/s) | decode %d tok in %.2fs (%.2f tok/s) | RSS %.2f GB\n",
            k, tpre, k/(tpre>1e-9?tpre:1e-9), ng, tgen, ng/(tgen>1e-9?tgen:1e-9), rss_gb());
    return ng;
}

#endif /* RT_GEN_LOOP_H */
