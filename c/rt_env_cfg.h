/* Env: budget_from_env, omp_hot_tune, RunEnv/parse_env.
 * Estratto da runtime.h (P2). Include DOPO Cfg/Layer/Model
 * e le dichiarazioni degli hook. Un'istanza per TU (tutto static). */
#ifndef RT_ENV_CFG_H
#define RT_ENV_CFG_H

/* ---------- budget di memoria: MEM_GB (GiB) batte MEM_FRAC (frazione della
 * RAM fisica TOTALE, deterministico). 0 = tutto residente. ---------- */
static int64_t budget_from_env(const char *gb, const char *frac, int64_t total_ram) {
    if (gb && *gb) { double g = atof(gb); if (g > 0) return (int64_t)(g * 1073741824.0); }
    if (frac && *frac) { double f = atof(frac); if (f > 0 && f <= 1 && total_ram > 0) return (int64_t)(f * total_ram); }
    return 0;
}

/* ---------- tuning permanente dei thread OpenMP (portato da glm.c) ----------
 * Le regioni parallele dei motori densi sono piccole e back-to-back (centinaia
 * di fork/join per token); con la wait policy passiva di default libgomp
 * parcheggia il team tra una regione e l'altra e la latenza di risveglio
 * domina. Tenere i thread caldi (spin attivo) collassa quell'overhead: su glm
 * il tempo matmul e' passato da 66.9s a 20.9s sulla build Zen5, senza alcuna
 * variazione dell'output numerico.
 *
 * libgomp legge le variabili OMP_/GOMP_ in un COSTRUTTORE che gira prima di
 * main(): un setenv() qui seguito dall'esecuzione normale arriverebbe troppo
 * tardi. Quindi al primo ingresso si seminano i default vincenti — rispettando
 * qualunque valore l'utente abbia gia' impostato (overwrite=0) — e ci si
 * re-esegue una volta sola cosi' un costruttore libgomp fresco li raccoglie.
 * La sentinella MOTY_OMP_TUNED garantisce al massimo un re-exec; MOTY_NO_OMP_TUNE=1
 * e' il kill-switch documentato che disattiva tutto il percorso. Su piattaforme
 * senza /proc/self/exe (o se execv fallisce) si prosegue senza tuning. */
static void omp_hot_tune(char **argv) {
    if (!getenv("MOTY_OMP_TUNED") && !getenv("MOTY_NO_OMP_TUNE")) {
        setenv("OMP_WAIT_POLICY", "active", 0);  /* team caldo tra le regioni piccole e fitte */
        setenv("GOMP_SPINCOUNT", "200000", 0);   /* spin breve, poi yield: le attese lunghe non bruciano un core */
        setenv("OMP_PROC_BIND", "close", 0);     /* team impacchettato su core adiacenti (localita' di cache) */
        setenv("OMP_DYNAMIC", "FALSE", 0);       /* team a taglia fissa: niente churn per-regione */
        setenv("MOTY_OMP_TUNED", "1", 1);
#if defined(__linux__)
        fprintf(stderr, "[OMP] hot-thread tuning: re-exec once (MOTY_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/self/exe", argv);           /* ritorna solo in caso di errore -> si prosegue senza tuning */
        perror("[OMP] execv self-reexec failed, running untuned");
#elif defined(__FreeBSD__)
        fprintf(stderr, "[OMP] hot-thread tuning: re-exec once (MOTY_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/curproc/file", argv);       /* ritorna solo in caso di errore -> si prosegue senza tuning */
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
    }
}

/* ---------- lettura delle manopole d'ambiente, in un posto solo ----------
 * Riempie i globali g_* (che i test impostano direttamente, senza env) e i
 * parametri del run. Ritorna 0 su valore invalido (messaggio gia' stampato). */
typedef struct {
    const char *snap;
    int qbits, ngen, maxctx, templ;
    int64_t budget;
} RunEnv;

static int parse_env(RunEnv *e) {
    e->snap = getenv("SNAP");
    g_gguf = getenv("GGUF");
    if (g_gguf && !*g_gguf) g_gguf = NULL;
    if (!e->snap && !g_gguf) { fprintf(stderr, "set SNAP=<snapshot directory> oppure GGUF=<file.gguf>\n"); return 0; }
    e->qbits = getenv("QBITS") ? atoi(getenv("QBITS")) : 0;
    if (e->qbits != -1 && e->qbits != 0 && e->qbits != 2 && e->qbits != 4 && e->qbits != 8) { fprintf(stderr, "QBITS deve essere 0 (f32), 4 (int4) o 8 (int8)\n"); return 0; }
    if (getenv("QGROUP")) {
        g_qgroup = atoi(getenv("QGROUP"));
        if (g_qgroup < 0 || (g_qgroup > 0 && g_qgroup % 16)) {
            fprintf(stderr, "QGROUP deve essere 0 (scala per riga) o un multiplo di 16\n"); return 0; }
    }
    e->ngen = getenv("NGEN") ? atoi(getenv("NGEN")) : 256;
    if (getenv("PREFILL_CHUNK")) g_prefill_chunk = atoi(getenv("PREFILL_CHUNK"));
    if (getenv("KV_BITS")) {
        g_kv_bits = atoi(getenv("KV_BITS"));
        if (g_kv_bits != 0 && g_kv_bits != 8) { fprintf(stderr, "KV_BITS deve essere 0 (f32) o 8 (int8)\n"); return 0; }
    }
    /* MICRO=1: micro-RSS. La KV-cache resta l'unica voce grande -> il default
     * di contesto scende a 256 (CTX esplicito vince sempre). */
    const char *mi_ = getenv("MICRO");
    g_micro = mi_ && atoi(mi_) > 0;
    const char *md_ = getenv("MICRO_DROP");
    if (md_ && *md_) g_micro_drop = atoi(md_) != 0;
    e->maxctx = getenv("CTX") ? atoi(getenv("CTX")) : (g_micro ? 256 : 4096);
    if (getenv("TEMP"))    g_temp = (float)atof(getenv("TEMP"));
    if (getenv("NUCLEUS")) g_nuc  = (float)atof(getenv("NUCLEUS"));
    if (getenv("SEED"))    g_rng  = (uint64_t)strtoull(getenv("SEED"),NULL,10) | 1u;
    g_tokens_dump = getenv("TOKENS") && atoi(getenv("TOKENS"));
    e->templ = getenv("CHAT_TEMPLATE") ? atoi(getenv("CHAT_TEMPLATE")) : 1;
    e->budget = budget_from_env(getenv("MEM_GB"), getenv("MEM_FRAC"), compat_total_ram_bytes());
    return 1;
}

#endif /* RT_ENV_CFG_H */
