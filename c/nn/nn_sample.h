/* nn_sample.h — sampling temperatura+nucleus + stop-set, portato da glm.c.
 * Estratto da nn.h: stato globale (g_temp/g_nuc/g_rng, g_stop) + pick_tok +
 * gestione stop-set. Dipende da nn_alloc.h (falloc per il buffer di sampling).
 * glm.c ha un sampler con parametri extra (ban per la speculative rejection):
 * per quello non include questo file; i motori densi e olmoe si'. */
#ifndef NN_SAMPLE_H
#define NN_SAMPLE_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn/nn_alloc.h"   /* falloc */

/* ---------- sampling (temperatura + nucleus) ---------- */
/* M3: stato sampler esportato (era static-per-TU): moty_g_* — assegnabile
 * dal gen-loop del runtime e dai test. Macro legacy: vecchi nomi. */
extern float moty_g_temp, moty_g_nuc;
extern uint64_t moty_g_rng;
#ifndef MOTY_CORE_NO_LEGACY
#define g_temp moty_g_temp
#define g_nuc  moty_g_nuc
#define g_rng  moty_g_rng
#endif
static inline double rndu(void){ g_rng^=g_rng<<13; g_rng^=g_rng>>7; g_rng^=g_rng<<17;
    return (double)(g_rng>>11)*(1.0/9007199254740992.0); }

/* argmax parallelo (V=128k: seriale ~0.4ms/tok). P5: scratch per-Model */

/* P5: buffer di sampling nell'arena per-Model. pbuf resta visibile al
 * comparator qsort via puntatore (niente qsort_r portabile): contratto
 * seriale del gen-loop. */
typedef struct { float *pbuf, *pbuf2; int *pidx, *pidx2; } SampBuf;





/* ---------- stop-set ---------- */
static int g_stop[9], g_nstop=0;
static inline int is_stop(int t){ for(int i=0;i<g_nstop;i++) if(t==g_stop[i]) return 1; return 0; }
static void stop_add(int t){ if(t>=0 && !is_stop(t) && g_nstop<9) g_stop[g_nstop++]=t; }


/* M3: implementazioni in nn/sample.c (libmoty-nn) */
int moty_argmax_v(const float *lo, int V);
int moty_argmax_v_par(Scratch *sc, const float *lo, int V);
void moty_dist_build(Scratch *sc, SampBuf *sb, const float *lo, int V);
int moty_dist_sample(const SampBuf *sb, int V);
int moty_pick_tok(Scratch *sc, const float *lo, int V);

#ifndef MOTY_CORE_NO_LEGACY
#define argmax_v moty_argmax_v
#define argmax_v_par moty_argmax_v_par
#define dist_build moty_dist_build
#define dist_sample moty_dist_sample
#define pick_tok moty_pick_tok
#endif

#endif /* NN_SAMPLE_H */
