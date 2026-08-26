/* prof.h — profilazione a finestre gated da MOTY_PROF (P4).
 *
 * Uso: PROF_DECLARE() nello scope del TU (una volta); PROF_TICK(t0_var)
 * cattura now_s() in una variabile locale; le accumulazioni restano nel
 * chiamante (il pattern per-fase dipende dal motore). Con MOTY_PROF
 * assente tutto sparisce a compile time: zero overhead in produzione.
 *
 * La stampante a finestre (delta dall'ultima stampa >2s) e' il metodo che
 * ha trovato: sampler qsort-128k (11.9ms/tok), warmup expert (5ms/tok
 * mediati), regioni MoE 3×K. Da usare con NGEN>=256: le medie cumulative
 * su run corti sono domin dal warm-up della cache expert. */
#ifndef PROF_H
#define PROF_H

#ifdef MOTY_PROF
#define PROF_ON 1
#include "nn/nn_alloc.h"      /* now_s */
/* contatori per-fase di un TU; il motore li tocca dentro PROF_ACC(name,dt) */
#define PROF_DECL() static double pf_attn=0,pf_conv=0,pf_moe=0,pf_ffn=0,pf_log=0,pf_other=0,pf_wall0=0,pf_last=0; static int pf_n=0
#define PROF_TICK(v) double v = now_s()
#define PROF_ACC(field, t0) pf_##field += now_s() - (t0)
#define PROF_COUNT() (pf_n++)
/* stampa a finestre: chiama a fine step (S==1); "other" = tempo wall non
 * attribuito a nessuna fase — e' dove si nascondono i costi seriali. */
#define PROF_WINDOW(nom) do { \
    double nw = now_s(); \
    if (pf_wall0 == 0) { pf_wall0 = nw; pf_last = nw; } \
    if (pf_n > 0 && nw - pf_last > 2.0) { \
        double wall = (nw - pf_wall0); \
        fprintf(stderr, "[prof] win=%d attn=%.2f conv=%.2f moe=%.2f ffn=%.2f logit=%.2f other=%.2f (ms/tok)\n", \
            pf_n, pf_attn*1000/pf_n, pf_conv*1000/pf_n, pf_moe*1000/pf_n, \
            pf_ffn*1000/pf_n, pf_log*1000/pf_n, \
            (wall*1000/pf_n) - (pf_attn+pf_conv+pf_moe+pf_ffn+pf_log)*1000/pf_n); \
        pf_attn=pf_conv=pf_moe=pf_ffn=pf_log=pf_other=0; pf_n=0; pf_last=nw; pf_wall0=nw; \
    } \
} while (0)
#else
#define PROF_ON 0
#define PROF_DECL()
#define PROF_TICK(v)                 /* sparisce */
#define PROF_ACC(field, t0)          /* sparisce */
#define PROF_COUNT()                 /* sparisce */
#define PROF_WINDOW(nom)             /* sparisce */
#endif

#endif /* PROF_H */
