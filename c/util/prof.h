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

/* ---- per-op table (global, all TUs + libmoty-nn): where a token's time goes.
 * OP_T(v) / OP_ACC(id, v) wrap serial call sites (never inside an omp
 * region); gen_turn prints the table for the prefill and per decoded token.
 * The array always exists (nn/alloc.c) so libraries and engines can be built
 * with or without MOTY_PROF independently. */
enum { OP_EMBED, OP_NORM, OP_RESID, OP_QKV, OP_QKNORM_ROPE, OP_KV_STORE, OP_ATTN_CORE, OP_O_PROJ,
       OP_CONV_IN, OP_CONV_DW, OP_CONV_OUT, OP_FFN_GATE_UP, OP_FFN_SILU, OP_FFN_DOWN,
       OP_LM_HEAD, OP_SAMPLE, OP_N };
extern double moty_prof_op[OP_N];
#define OP_NAMES { "embed", "rmsnorm", "resid", "qkv_proj", "qknorm+rope", "kv_store", "attn(score+softmax+V)", \
    "o_proj", "conv_in_proj", "conv_depthwise", "conv_out_proj", "ffn_gate+up", "ffn_silu*up", "ffn_down", "lm_head", "sample" }
#ifdef MOTY_PROF
#define OP_T(v) double v = now_s()
#define OP_ACC(id, v) (moty_prof_op[id] += now_s() - (v))
#else
#define OP_T(v)
#define OP_ACC(id, v) ((void)0)
#endif

#endif /* PROF_H */
