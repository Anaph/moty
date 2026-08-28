/* runtime/config.h — M4: lo stato di configurazione del runtime, UNA copia.
 * I globali g_* erano static-per-TU (uno per motore!): ora sono esportati
 * da libmoty-runtime e le macro legacy tengono i vecchi nomi nei motori.
 * parse_env li riempie DALL'AMBIENTE in un punto solo; i test li impostano
 * direttamente, senza env. */
#ifndef MOTY_RT_CONFIG_H
#define MOTY_RT_CONFIG_H

#include <stdint.h>

typedef struct {
    const char *snap;
    int qbits, ngen, maxctx, templ;
    int64_t budget;                 /* MEM_GB/MEM_FRAC in byte; 0 = tutto residente */
} MotyRunConfig;

/* stato globale del runtime (definito in runtime/env.c) */
extern const char *moty_rt_g_gguf;          /* GGUF=<file> (NULL: SNAP=) */
extern int moty_rt_g_qgroup;                /* QGROUP: 0 = scala per riga */
extern int moty_rt_g_prefill_chunk;         /* PREFILL_CHUNK */
extern int moty_rt_g_kv_bits;               /* KV_BITS: 0 f32, 8 int8 */
extern int moty_rt_g_micro;                 /* MICRO=1 */
extern int moty_rt_g_micro_drop;            /* MICRO_DROP=0 */
extern int moty_rt_g_tokens_dump;           /* TOKENS=1 */

int      moty_rt_parse_env(MotyRunConfig *e);   /* 0 = valore invalido (msg stampato) */
void     moty_rt_omp_hot_tune(char **argv);     /* re-exec una volta sola; MOTY_NO_OMP_TUNE spegne */
int64_t  moty_rt_budget_from_env(const char *gb, const char *frac, int64_t total_ram);

/* macro legacy: il codice runtime/engine continua a usare g_gguf ecc. */
#ifndef MOTY_CORE_NO_LEGACY
#define g_gguf          moty_rt_g_gguf
#define g_qgroup        moty_rt_g_qgroup
#define g_prefill_chunk moty_rt_g_prefill_chunk
#define g_kv_bits       moty_rt_g_kv_bits
#define g_micro         moty_rt_g_micro
#define g_micro_drop    moty_rt_g_micro_drop
#define g_tokens_dump   moty_rt_g_tokens_dump
#endif
#endif /* MOTY_RT_CONFIG_H */
