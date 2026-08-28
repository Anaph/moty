/* env.c — M4 libmoty-runtime: manopole ambiente + tuning OMP, una copia.
 * Corpo portato 1:1 da rt_env_cfg.h (era static-per-TU). */
#include "runtime/config.h"
#include "nn/nn_sample.h"      /* moty_g_temp/nuc/rng */
#include "util/compat.h"       /* compat_total_ram_bytes */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char *moty_rt_g_gguf = NULL;
int moty_rt_g_qgroup = 32;
int moty_rt_g_prefill_chunk = 0;
int moty_rt_g_kv_bits = 0;
int moty_rt_g_micro = 0;
int moty_rt_g_micro_drop = 1;
int moty_rt_g_tokens_dump = 0;

int64_t moty_rt_budget_from_env(const char *gb, const char *frac, int64_t total_ram) {
    if (gb && *gb) { double g = atof(gb); if (g > 0) return (int64_t)(g * 1073741824.0); }
    if (frac && *frac) { double f = atof(frac); if (f > 0 && f <= 1 && total_ram > 0) return (int64_t)(f * total_ram); }
    return 0;
}

void moty_rt_omp_hot_tune(char **argv) {
    if (!getenv("MOTY_OMP_TUNED") && !getenv("MOTY_NO_OMP_TUNE")) {
        setenv("OMP_WAIT_POLICY", "active", 0);
        setenv("GOMP_SPINCOUNT", "200000", 0);
        setenv("OMP_PROC_BIND", "close", 0);
        setenv("OMP_DYNAMIC", "FALSE", 0);
        setenv("MOTY_OMP_TUNED", "1", 1);
#if defined(__linux__)
        fprintf(stderr, "[OMP] hot-thread tuning: re-exec once (MOTY_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/self/exe", argv);
        perror("[OMP] execv self-reexec failed, running untuned");
#elif defined(__FreeBSD__)
        fprintf(stderr, "[OMP] hot-thread tuning: re-exec once (MOTY_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/curproc/file", argv);
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
    }
}

int moty_rt_parse_env(MotyRunConfig *e) {
    e->snap = getenv("SNAP");
    moty_rt_g_gguf = getenv("GGUF");
    if (moty_rt_g_gguf && !*moty_rt_g_gguf) moty_rt_g_gguf = NULL;
    if (!e->snap && !moty_rt_g_gguf) { fprintf(stderr, "set SNAP=<snapshot directory> oppure GGUF=<file.gguf>\n"); return 0; }
    e->qbits = getenv("QBITS") ? atoi(getenv("QBITS")) : 0;
    if (e->qbits != -1 && e->qbits != 0 && e->qbits != 2 && e->qbits != 4 && e->qbits != 8) { fprintf(stderr, "QBITS deve essere 0 (f32), 4 (int4) o 8 (int8)\n"); return 0; }
    if (getenv("QGROUP")) {
        moty_rt_g_qgroup = atoi(getenv("QGROUP"));
        if (moty_rt_g_qgroup < 0 || (moty_rt_g_qgroup > 0 && moty_rt_g_qgroup % 16)) {
            fprintf(stderr, "QGROUP deve essere 0 (scala per riga) o un multiplo di 16\n"); return 0; }
    }
    e->ngen = getenv("NGEN") ? atoi(getenv("NGEN")) : 256;
    if (getenv("PREFILL_CHUNK")) moty_rt_g_prefill_chunk = atoi(getenv("PREFILL_CHUNK"));
    if (getenv("KV_BITS")) {
        moty_rt_g_kv_bits = atoi(getenv("KV_BITS"));
        if (moty_rt_g_kv_bits != 0 && moty_rt_g_kv_bits != 8) { fprintf(stderr, "KV_BITS deve essere 0 (f32) o 8 (int8)\n"); return 0; }
    }
    const char *mi_ = getenv("MICRO");
    moty_rt_g_micro = mi_ && atoi(mi_) > 0;
    const char *md_ = getenv("MICRO_DROP");
    if (md_ && *md_) moty_rt_g_micro_drop = atoi(md_) != 0;
    e->maxctx = getenv("CTX") ? atoi(getenv("CTX")) : (moty_rt_g_micro ? 256 : 4096);
    if (getenv("TEMP"))    moty_g_temp = (float)atof(getenv("TEMP"));
    if (getenv("NUCLEUS")) moty_g_nuc  = (float)atof(getenv("NUCLEUS"));
    if (getenv("SEED"))    moty_g_rng  = (uint64_t)strtoull(getenv("SEED"),NULL,10) | 1u;
    moty_rt_g_tokens_dump = getenv("TOKENS") && atoi(getenv("TOKENS"));
    e->templ = getenv("CHAT_TEMPLATE") ? atoi(getenv("CHAT_TEMPLATE")) : 1;
    e->budget = moty_rt_budget_from_env(getenv("MEM_GB"), getenv("MEM_FRAC"), compat_total_ram_bytes());
    return 1;
}
