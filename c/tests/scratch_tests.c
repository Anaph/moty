/* Test diretti dell'arena Scratch (P5, nn_alloc.h) — il contratto
 * reserve-before-take su cui si reggono tutti i kernel condivisi:
 *   1. take consegna chunk allineati a 64 byte (linea di cache)
 *   2. dopo una reserve, NESSUN take muove la base: i puntatori già
 *      consegnati restano validi (i kernel ne tengono parecchi in volo)
 *   3. la crescita (realloc) preserva il contenuto già scritto
 *   4. reset azzera used ma conserva cap; take riparte dalla base
 *   5. due arena nello stesso Model (scr/bscr) non si sovrappongono
 *   6. scr_free rilascia e l'arena resta riusabile
 * Convenzione: 0 = pass, 1 = fail. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../nn/nn_alloc.h"

static uint64_t sa_lcg = 0xC0FFEE42ULL;
static uint32_t sa_rnd(void) {
    sa_lcg = sa_lcg*6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(sa_lcg >> 33);
}

/* ---- 1. allineamento 64 di ogni take, anche da 1 byte ---- */
int sa_alignment(void) {
    Scratch s = {0};
    for (int i = 1; i <= 257; i += 8) {
        scr_reserve(&s, 4096);
        void *p = scr_take(&s, i);
        if (((uintptr_t)p & 63) != 0) { fprintf(stderr, "take(%d) non allineato: %p\n", i, p); return 1; }
        scr_reset(&s);
    }
    scr_free(&s);
    return 0;
}

/* ---- 2. stabilità dei puntatori dopo la reserve ---- */
int sa_stability(void) {
    Scratch s = {0};
    scr_reserve(&s, 1 << 20);
    void *ptrs[64];
    for (int i = 0; i < 64; i++) {
        ptrs[i] = scr_take(&s, 1 + (i % 17));
        memset(ptrs[i], 0xAB, 1 + (i % 17));
        if (ptrs[i] != ptrs[0] && (uintptr_t)ptrs[i] < (uintptr_t)ptrs[0]) { scr_free(&s); return 1; }
    }
    /* nessun take ha mosso la base: tutti distinti e monotoni */
    for (int i = 1; i < 64; i++)
        if (ptrs[i] == ptrs[i-1]) { fprintf(stderr, "take sovrapposto al precedente\n"); scr_free(&s); return 1; }
    scr_free(&s);
    return 0;
}

/* ---- 3. la crescita preserva il contenuto, i chunk non si calpestano ---- */
int sa_growth_preserves(void) {
    Scratch s = {0};
    scr_reserve(&s, 4096);
    unsigned char *a = scr_take(&s, 2048);
    for (int i = 0; i < 2048; i++) a[i] = (unsigned char)(i * 7 + 3);
    /* cresci oltre cap: la reserve può reallocare */
    scr_reserve(&s, 1 << 16);
    unsigned char *b = scr_take(&s, 8192);
    for (int i = 0; i < 8192; i++) b[i] = (unsigned char)(i ^ 0x5A);
    for (int i = 0; i < 2048; i++)
        if (a[i] != (unsigned char)(i * 7 + 3)) { fprintf(stderr, "growth ha corrotto a[%d]\n", i); scr_free(&s); return 1; }
    scr_free(&s);
    return 0;
}

/* ---- 4. reset: used torna a 0, cap resta, take riparte dalla base ---- */
int sa_reset(void) {
    Scratch s = {0};
    scr_reserve(&s, 8192);
    void *base = scr_take(&s, 64);
    memset(base, 0xCD, 64);
    scr_reset(&s);
    if (s.cap != 8192 && s.cap < 8192) { fprintf(stderr, "cap perso dopo reset\n"); scr_free(&s); return 1; }
    void *again = scr_take(&s, 64);
    if (again != base) { fprintf(stderr, "take dopo reset non riparte dalla base\n"); scr_free(&s); return 1; }
    scr_free(&s);
    return 0;
}

/* ---- 5. due arena indipendenti (il caso scr/bscr del Model) ---- */
int sa_two_arenas(void) {
    Scratch x = {0}, y = {0};
    scr_reserve(&x, 1 << 14); scr_reserve(&y, 1 << 14);
    float *xa = scr_take(&x, 4096), *ya = scr_take(&y, 4096);
    for (int i = 0; i < 1024; i++) xa[i] = 1.f * i;
    for (int i = 0; i < 1024; i++) ya[i] = 2.f * i;
    for (int i = 0; i < 1024; i++)
        if (xa[i] != 1.f*i || ya[i] != 2.f*i) { fprintf(stderr, "arena sovrapposte\n"); scr_free(&x); scr_free(&y); return 1; }
    scr_free(&x); scr_free(&y);
    return 0;
}

/* ---- 6. free e riuso; scr_al arrotonda a 64 ---- */
int sa_free_reuse(void) {
    Scratch s = {0};
    scr_reserve(&s, 1 << 12); scr_take(&s, 128);
    scr_free(&s);
    if (s.base != NULL || s.cap != 0 || s.used != 0) { fprintf(stderr, "scr_free non azzera\n"); return 1; }
    scr_reserve(&s, 1 << 12);
    void *p = scr_take(&s, 16);
    if (!p) { fprintf(stderr, "arena non riusabile dopo free\n"); return 1; }
    scr_free(&s);
    for (int64_t v = 0; v <= 256; v += 7)
        if (scr_al(v) % 64 != 0 || scr_al(v) < v) { fprintf(stderr, "scr_al rotto per %lld\n", (long long)v); return 1; }
    return 0;
}

/* ---- 7. contratto d'uso dei kernel: fasi reserve→take consecutive ----
 * Riproduce il pattern di nn_conv.h: una reserve che copre TUTTI i take
 * della fase, take in ordine, poi una seconda fase (nuovo reset+reserve). */
int sa_kernel_pattern(void) {
    Scratch s = {0};
    enum { S = 5, D = 128 };
    for (int rep = 0; rep < 3; rep++) {
        scr_reset(&s);
        scr_reserve(&s, scr_al((int64_t)S*3*D*4) + scr_al((int64_t)S*D*4) + 2*scr_al((int64_t)S*D));
        float *bcx = scr_take(&s, (int64_t)S*3*D*4);
        float *y   = scr_take(&s, (int64_t)S*D*4);
        int8_t *xi = scr_take(&s, scr_al((int64_t)S*D));
        int8_t *yq = scr_take(&s, scr_al((int64_t)S*D));
        for (int i = 0; i < S*3*D; i++) bcx[i] = (float)(sa_rnd() & 0xF);
        for (int i = 0; i < S*D; i++) { y[i] = 1.f; xi[i] = (int8_t)(i & 63); yq[i] = -(int8_t)(i & 63); }
        /* fase 2: reset e ricomincia — i vecchi puntatori non servono più */
        scr_reset(&s);
        scr_reserve(&s, scr_al((int64_t)S*D*4));
        float *z = scr_take(&s, (int64_t)S*D*4);
        for (int i = 0; i < S*D; i++) z[i] = 0.f;
        if (!z) return 1;
    }
    scr_free(&s);
    return 0;
}
