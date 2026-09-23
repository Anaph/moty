/* nn/head.c two-stage lm_head (HEAD_TOPK). Plain C, 0 = ok / 1 = fail; gtest
 * glue in head_gtest.cc. Standalone runner for the target: -DHEAD_TEST_MAIN
 * (build line as in q4r4_tests.c). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nn/nn.h"

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

static uint64_t hd_rng = 424242;
static float hd_frnd(void) {
    hd_rng ^= hd_rng << 13; hd_rng ^= hd_rng >> 7; hd_rng ^= hd_rng << 17;
    return (float)((hd_rng >> 11) * (1.0 / 9007199254740992.0)) - 0.5f;
}

/* random Q4R4 head V x D (V not a multiple of 4: ragged last block) */
static void mk_head(Mat *h, int V, int D, float *W) {
    int nb = D / 32, O4 = (V + 3) / 4;
    memset(h, 0, sizeof *h);
    h->fmt = WF_Q4R4; h->O = V; h->I = D;
    h->q4 = calloc((size_t)O4*nb*64, 1); h->s16 = calloc((size_t)O4*nb*4, 2);
    for (int i = 0; i < V*D; i++) W[i] = hd_frnd();
    for (int b = 0; b < O4; b++)
        moty_pack_q4r4_block(W + (size_t)b*4*D, V - b*4 < 4 ? V - b*4 : 4, D, h->q4 + (size_t)b*nb*64, h->s16 + (size_t)b*nb*4);
}

/* K = V: every block is a candidate, the result is the full head bit for bit */
int hd_full_equivalence(void) {
    enum { V = 203, D = 96 };
    static float W[V*D]; float x[D], a[V], b[V];
    Mat h; mk_head(&h, V, D, W);
    for (int i = 0; i < D; i++) x[i] = hd_frnd() * 3.f;
    MotyHeadSL sl; CHECK(moty_nn_head_sl_build(&sl, &h, V));
    mat_apply(a, x, &h, 1);
    moty_nn_head_apply(b, x, &h, &sl);
    CHECK(!memcmp(a, b, sizeof a));
    moty_nn_head_sl_free(&sl); free(h.q4); free(h.s16);
    return 0;
}

/* K < V: the K best stage-1 scores (exact K-th threshold, the sampled path)
 * get their exact logits, every row outside their blocks gets -1e30 */
int hd_shortlist(void) {
    enum { V = 4099, D = 128, K = 40, HOT = 1234 };
    static float W[V*D], a[V], b[V], sc[V], srt[V]; float x[D];
    for (int i = 0; i < D; i++) x[i] = hd_frnd() * (i == 5 ? 30.f : 2.f);
    Mat h; mk_head(&h, V, D, W);
    {   /* one row aligned with x: the unambiguous argmax */
        int nb = D / 32; float blk[4*D];
        memcpy(blk, W + (size_t)(HOT & ~3)*D, sizeof blk);
        for (int i = 0; i < D; i++) blk[(HOT & 3)*D + i] = x[i] >= 0 ? 0.5f : -0.5f;
        moty_pack_q4r4_block(blk, 4, D, h.q4 + (size_t)(HOT/4)*nb*64, h.s16 + (size_t)(HOT/4)*nb*4);
    }
    MotyHeadSL sl; CHECK(moty_nn_head_sl_build(&sl, &h, K));
    mat_apply(a, x, &h, 1);
    moty_nn_head_apply(b, x, &h, &sl);
    moty_nn_head_sl_scores(&sl, x, sc);
    memcpy(srt, sc, sizeof sc);
    for (int i = 0; i < K; i++) {                     /* partial selection sort: K-th largest */
        int m = i; for (int j = i+1; j < V; j++) if (srt[j] > srt[m]) m = j;
        float t = srt[i]; srt[i] = srt[m]; srt[m] = t;
    }
    float thr = srt[K-1];
    int ncand = 0;
    for (int v = 0; v < V; v++) {
        int blk_in = 0;
        for (int r = 0; r < 4; r++) { int u = (v & ~3) + r; if (u < V && sc[u] >= thr) blk_in = 1; }
        if (sc[v] >= thr) ncand++;
        if (blk_in) CHECK(b[v] == a[v]);
        else CHECK(b[v] == -1e30f);
    }
    CHECK(ncand >= K);
    /* a row aligned with x is the argmax of the full head and of the shortlist */
    int am = 0, bm = 0;
    for (int v = 1; v < V; v++) { if (a[v] > a[am]) am = v; if (b[v] > b[bm]) bm = v; }
    CHECK(am == HOT && bm == HOT);
    moty_nn_head_sl_free(&sl); free(h.q4); free(h.s16);
    return 0;
}

/* the 1-bit rows: sign bits, popcounts and mean |w| match the packed codes */
int hd_build(void) {
    enum { V = 9, D = 64 };
    static float W[V*D];
    Mat h; mk_head(&h, V, D, W);
    MotyHeadSL sl; CHECK(moty_nn_head_sl_build(&sl, &h, 4));
    int nb = D / 32;
    for (int v = 0; v < V; v++) {
        int b = v / 4, r = v % 4, pc = 0; double sa = 0;
        for (int j = 0; j < D; j++) {
            const uint8_t *wb = h.q4 + ((size_t)b*nb + j/32)*64 + r*16;
            int jj = j % 32, q = jj < 16 ? (wb[jj] & 15) : (wb[jj-16] >> 4);
            float w = (float)(q - 8) * moty_hw_f16_to_f32(h.s16[((size_t)b*nb + j/32)*4 + r]);
            int bit = sl.bits[(size_t)v*(D/8) + j/8] >> (j & 7) & 1;
            CHECK(bit == (w >= 0)); pc += bit; sa += fabsf(w);
        }
        CHECK(pc == sl.pc[v] && fabsf(sl.scale[v] - (float)(sa / D)) <= 1e-6f * (float)(sa / D) + 1e-12f);
    }
    Mat f; memset(&f, 0, sizeof f); f.fmt = WF_F32; f.O = V; f.I = D;
    CHECK(moty_nn_head_sl_build(&sl, &f, 4) == 0);    /* only Q4R4 heads */
    free(h.q4); free(h.s16);
    return 0;
}

#ifdef HEAD_TEST_MAIN
int main(void) {
    struct { const char *n; int (*f)(void); } T[] = {
        {"build", hd_build}, {"full_equivalence", hd_full_equivalence}, {"shortlist", hd_shortlist} };
    int bad = 0;
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) { int r = T[i].f(); bad |= r; printf("[%s] %s\n", r ? "FAIL" : " OK ", T[i].n); }
    printf("tier: %s\n", HW_IDOT_KERNEL);
    return bad;
}
#endif
