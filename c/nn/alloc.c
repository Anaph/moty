/* alloc.c — unica implementazione degli allocatori e dell'arena Scratch
 * (M2, libmoty-nn core). Le firme pubbliche sono moty_*; nn/nn_alloc.h
 * dichiara i prototipi e tiene le macro legacy (balloc -> moty_balloc). */
#include "nn/nn_alloc.h"

void *moty_grow(void **p, int64_t *cap, int64_t need, size_t esz, const char *what) {
    if (need <= *cap) return *p;
    *cap = need;
    *p = realloc(*p, (size_t)need * esz);
    if (!*p) { fprintf(stderr, "OOM %s (%lld x %lu byte)\n", what, (long long)need, (unsigned long)esz); exit(1); }
    return *p;
}
void *moty_balloc(int64_t n, const char *what) {
    void *p = malloc(n > 0 ? n : 1);
    if (!p) { fprintf(stderr, "OOM %s (%lld byte)\n", what, (long long)n); exit(1); }
    return p;
}
void *moty_bzalloc(int64_t n, const char *what) {
    void *p = calloc(1, n > 0 ? n : 1);
    if (!p) { fprintf(stderr, "OOM %s (%lld byte)\n", what, (long long)n); exit(1); }
    return p;
}
void moty_scr_reset(Scratch *s) { s->used = 0; }
void moty_scr_free(Scratch *s) { free(s->raw); s->raw = NULL; s->base = NULL; s->off = 0; s->cap = s->used = 0; }
void moty_scr_reserve(Scratch *s, int64_t bytes) {
    if (bytes <= s->cap) return;
    int64_t nc = s->cap ? s->cap : (int64_t)1 << 20;   /* 1MB iniziale, x2 */
    while (nc < bytes) nc *= 2;
    char *nr = realloc(s->raw, (size_t)nc + 63);
    if (!nr) { fprintf(stderr, "OOM scratch (%lld byte)\n", (long long)nc); exit(1); }
    int noff = (int)((64 - ((uintptr_t)nr & 63)) & 63);
    if (s->base && noff != s->off && s->used > 0) {
        int64_t live = s->used < s->cap ? s->used : s->cap;
        memmove(nr + noff, nr + s->off, (size_t)live);
    }
    s->raw = nr; s->off = noff; s->base = nr + noff; s->cap = nc;
}
void *moty_scr_take(Scratch *s, int64_t bytes) {
    bytes = (bytes + 63) & ~(int64_t)63;
    void *p = s->base + s->used;
    s->used += bytes;
    return p;
}
