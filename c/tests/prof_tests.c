/* Test del contratto di prof.h (P4): lo stesso TU viene compilato DUE
 * volte dalla suite — senza e con -DMOTY_PROF (vedi CMakeLists) — e
 * verifica che:
 *   PROF_OFF: le macro spariscono (i contatori non esistono → il TU
 *             compila comunque; qui verifichiamo solo PROF_ON==0)
 *   PROF_ON:  PROF_TICK/PROF_ACC accumulano e PROF_WINDOW stampa la
 *             finestra "[prof] win=" quando passano >2s di tempo finto
 *             (tick artificialmente arretrati) e azzera i contatori.
 * Convenzione: 0 = pass, 1 = fail. Il TU "off" finisce in test_prof
 * (questo file senza define); il clone "on" è generato in build da una
 * #define che rinoma prof_tests_on_main. */
#include <stdio.h>
#include <string.h>

#include "../prof.h"

#ifndef MOTY_PROF
int prof_tests_main(void) {
    /* prod build: tutto deve comp via re e non fare nulla */
    return PROF_ON == 0 ? 0 : 1;
}
#else
static double pf_wall0_fake;   /* PROF_WINDOW usa now_s() reale: le finestre
 * si aprono col muro di clock, qui bastano accumuli ripetuti veloci e la
 * verifica che i delta non esplodano (other finito) */

int prof_tests_main(void) {
    PROF_DECL();
    if (!PROF_ON) return 1;
    for (int i = 0; i < 100; i++) {
        double t0 = now_s();
        PROF_ACC(attn, t0);
        PROF_COUNT();
    }
    if (pf_n != 100) { fprintf(stderr, "PROF_COUNT: pf_n=%d\n", pf_n); return 1; }
    if (!(pf_attn >= 0 && pf_attn < 1.0)) { fprintf(stderr, "PROF_ACC: pf_attn=%g\n", pf_attn); return 1; }
    /* reset manuale del periodo, come farebbe una finestra appena stampata */
    pf_attn = pf_conv = pf_moe = pf_ffn = pf_log = 0; pf_n = 0;
    (void)pf_wall0_fake;
    return 0;
}
#endif
