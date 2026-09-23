/* rope.c — M3 libmoty-nn */
#include "nn/nn_rope.h"

void moty_l2norm_head(float *x, int d) {
    double s = 0; for (int i = 0; i < d; i++) s += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)s + 1e-6f);
    for (int i = 0; i < d; i++) x[i] *= r;
}

/* RoPE Neox-style half-split sui primi `rot` dimensioni di un head.
 * cos/sin come funzione di (theta, rot, pos) si calcolano UNA volta per
 * posizione (tabella cresciuta a richiesta, stessa espressione del calcolo
 * diretto -> risultato bit-identico) invece che per ogni testa di ogni layer:
 * su A53 powf+cosf+sinf per coppia dominavano "qknorm+rope" nel profilo.
 * Chiamata seriale (fuori da regioni omp); dentro una regione: calcolo diretto. */
#include <omp.h>
#include <stdlib.h>
typedef struct { float theta; int rot, npos; float *cs; } RopeTab;   /* cs[pos][rot/2][2] */
static RopeTab g_rope[4]; static int g_nrope;

static const float *rope_row(float theta, int rot, int pos) {
    RopeTab *t = NULL;
    for (int i = 0; i < g_nrope; i++) if (g_rope[i].theta == theta && g_rope[i].rot == rot) { t = &g_rope[i]; break; }
    if (!t) { if (g_nrope == 4) return NULL; t = &g_rope[g_nrope++]; t->theta = theta; t->rot = rot; t->npos = 0; t->cs = NULL; }
    if (pos >= t->npos) {
        int n = t->npos ? t->npos : 256; while (n <= pos) n *= 2;
        float *cs = realloc(t->cs, (size_t)n * rot * sizeof(float));
        if (!cs) return NULL;
        for (int p = t->npos; p < n; p++)
            for (int i = 0; i < rot / 2; i++) {
                float angle = p / powf(theta, (float)(2 * i) / rot);
                cs[(size_t)p*rot + 2*i] = cosf(angle); cs[(size_t)p*rot + 2*i + 1] = sinf(angle);
            }
        t->cs = cs; t->npos = n;
    }
    return t->cs + (size_t)pos * rot;
}

void moty_rope_head(float *x, int pos, float theta, int rot) {
    const float *cs = (!omp_in_parallel() && pos >= 0) ? rope_row(theta, rot, pos) : NULL;
    for (int i = 0; i < rot / 2; i++) {
        float c, s;
        if (cs) { c = cs[2*i]; s = cs[2*i+1]; }
        else { float angle = pos / powf(theta, (float)(2 * i) / rot); c = cosf(angle); s = sinf(angle); }
        float a = x[i], b = x[i + rot / 2];
        x[i] = a * c - b * s;
        x[i + rot / 2] = a * s + b * c;
    }
}
