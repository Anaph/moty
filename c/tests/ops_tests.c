/* hw/hw_ops.h row ops (+ the RoPE cos/sin table in nn/rope.c): kernel vs
 * scalar reference. Plain C, 0 = ok; gtest glue in ops_gtest.cc; standalone
 * runner for the target with -DOPS_TEST_MAIN (see q4r4_tests.c). On non-NEON
 * hosts the kernels are the references (exact equality); on aarch64 the
 * tolerances bound f32 summation order + the NEON exp polynomial. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nn/nn.h"
#include "nn/nn_rope.h"

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

static uint64_t op_rng = 99;
static float op_frnd(void) {
    op_rng ^= op_rng << 13; op_rng ^= op_rng >> 7; op_rng ^= op_rng << 17;
    return (float)((op_rng >> 11) * (1.0 / 9007199254740992.0)) - 0.5f;
}
static double relerr(const float *a, const float *b, int n) {
    double m = 0; for (int i = 0; i < n; i++) { double d = fabs(a[i]-b[i]) / (fabs(b[i]) + 1e-6); if (d > m) m = d; } return m;
}

int op_rmsnorm(void) {
    enum { N = 1030 };                                   /* not a multiple of 16: tail */
    float x[N], w[N], a[N], b[N];
    for (int i = 0; i < N; i++) { x[i] = op_frnd() * (i == 7 ? 200.f : 3.f); w[i] = 1.f + op_frnd(); }
    moty_hw_rmsnorm(a, x, w, N, 1e-5f); moty_hw_rmsnorm_ref(b, x, w, N, 1e-5f);
    CHECK(relerr(a, b, N) < 2e-6);
    return 0;
}

int op_silu_mul(void) {
    enum { N = 4099 };
    float g[N], u[N], a[N], b[N];
    for (int i = 0; i < N; i++) { g[i] = op_frnd() * 40.f; u[i] = op_frnd() * 4.f; }
    g[0] = -95.f; g[1] = 95.f; g[2] = 0.f;                 /* exp clamp range, zero */
    memcpy(a, g, sizeof g); memcpy(b, g, sizeof g);
    moty_hw_silu_mul(a, u, N); moty_hw_silu_mul_ref(b, u, N);
    double m = 0;
    for (int i = 0; i < N; i++) { double d = fabs(a[i]-b[i]) / (fabs(b[i]) + 1e-30); if (fabs(b[i]) > 1e-20 && d > m) m = d; }
    if (m > 1e-6) fprintf(stderr, "silu rel %.3g\n", m);
    CHECK(m < 1e-6 && isfinite(a[0]) && isfinite(a[1]));
    return 0;
}

int op_softmax(void) {
    enum { N = 777 };
    float a[N], b[N];
    for (int i = 0; i < N; i++) a[i] = b[i] = op_frnd() * 30.f;
    moty_hw_softmax(a, N); moty_hw_softmax_ref(b, N);
    double s = 0, m = 0; for (int i = 0; i < N; i++) { s += a[i]; m = fmax(m, fabs(a[i]-b[i])); }
    CHECK(fabs(s - 1.0) < 1e-5 && m < 1e-6);
    return 0;
}

int op_axpy_add(void) {
    enum { N = 131 };
    float x[N], y0[N], y1[N];
    for (int i = 0; i < N; i++) { x[i] = op_frnd(); y0[i] = y1[i] = op_frnd(); }
    moty_hw_axpy(y0, 0.37f, x, N); moty_hw_axpy_ref(y1, 0.37f, x, N);
    CHECK(relerr(y0, y1, N) < 1e-6);
    moty_hw_add(y0, x, N); moty_hw_add_ref(y1, x, N);
    CHECK(relerr(y0, y1, N) < 1e-6);
    return 0;
}

/* short conv: several tokens through the kernel and the reference, outputs
 * and the causal state must agree (K=3 NEON path + K=4 fallback), also on a
 * channel sub-range as the threaded conv layer uses */
int op_shortconv(void) {
    enum { D = 70, T = 5 };
    for (int K = 3; K <= 4; K++) {
        float w[D*4], s0[D*3], s1[D*3], b[D], c[D], x[D], y0[D], y1[D];
        for (int i = 0; i < D*K; i++) w[i] = op_frnd();
        for (int i = 0; i < D*(K-1); i++) s0[i] = s1[i] = op_frnd();
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < D; i++) { b[i] = op_frnd(); c[i] = op_frnd(); x[i] = op_frnd(); }
            moty_hw_shortconv_step(y0, b, c, x, w, s0, K, 0, 33);
            moty_hw_shortconv_step(y0, b, c, x, w, s0, K, 33, D);
            moty_hw_shortconv_step_ref(y1, b, c, x, w, s1, K, 0, D);
            for (int i = 0; i < D; i++) CHECK(fabsf(y0[i] - y1[i]) <= 1e-6f * (1 + fabsf(y1[i])));
            for (int i = 0; i < D*(K-1); i++) CHECK(fabsf(s0[i] - s1[i]) <= 1e-6f * (1 + fabsf(s1[i])));
        }
    }
    return 0;
}

/* RoPE: the per-position table gives bit-identical results to the direct
 * formula (same expression), across table growth and two (theta, rot) keys */
int op_rope_table(void) {
    int hd = 64;
    for (int k = 0; k < 2; k++) {
        float theta = k ? 5000000.f : 1000000.f; int rot = k ? 64 : 32;
        for (int pos = 0; pos < 700; pos += 37) {
            float a[64], b[64];
            for (int i = 0; i < hd; i++) a[i] = b[i] = op_frnd();
            rope_head(a, pos, theta, rot);
            for (int i = 0; i < rot / 2; i++) {
                float angle = pos / powf(theta, (float)(2 * i) / rot), c = cosf(angle), s = sinf(angle);
                float x0 = b[i], x1 = b[i + rot/2];
                b[i] = x0 * c - x1 * s; b[i + rot/2] = x0 * s + x1 * c;
            }
            CHECK(!memcmp(a, b, sizeof a));
        }
    }
    return 0;
}

/* attention rows: register-resident hd=64 kernels (and the generic hd path)
 * vs the per-row dot/axpy loops; odd row counts exercise the 2-row tails */
int op_attn_rows(void) {
    static const int hds[2] = {64, 48}, ns[3] = {1, 7, 130};
    for (int a = 0; a < 2; a++) for (int b = 0; b < 3; b++) {
        int hd = hds[a], n = ns[b];
        float *K = malloc(sizeof(float)*n*hd), *V = malloc(sizeof(float)*n*hd), q[64], s0[130], s1[130], c0[64], c1[64];
        for (int i = 0; i < n*hd; i++) { K[i] = op_frnd(); V[i] = op_frnd() * 2.f; }
        for (int i = 0; i < hd; i++) q[i] = op_frnd() * 3.f;
        moty_hw_attn_scores(s0, q, K, n, hd, 0.125f); moty_hw_attn_scores_ref(s1, q, K, n, hd, 0.125f);
        double m = 0; for (int t = 0; t < n; t++) m = fmax(m, fabs(s0[t] - s1[t]));
        CHECK(m < 1e-5);
        moty_hw_attn_accum(c0, s1, V, n, hd); moty_hw_attn_accum_ref(c1, s1, V, n, hd);
        m = 0; for (int d = 0; d < hd; d++) m = fmax(m, fabs(c0[d] - c1[d]));
        CHECK(m < 1e-4);
        free(K); free(V);
    }
    return 0;
}

/* bit-plane popcounts: kernel == reference exactly (integer), incl. more
 * than 16 chunks (the u8 lane flush) and a row stride != nbytes */
int op_popc4x3(void) {
    static const int nbs[3] = {16, 128, 272};
    for (int k = 0; k < 3; k++) {
        int nb = nbs[k]; int64_t bs = nb + 16;
        uint8_t *b = malloc(4*bs), *pl = malloc(3*nb); uint32_t c0[12], c1[12];
        for (int i = 0; i < 4*bs; i++) b[i] = (uint8_t)(op_frnd() * 512);
        for (int i = 0; i < 3*nb; i++) pl[i] = (uint8_t)(op_frnd() * 512);
        memset(b, 0xff, nb); memset(pl, 0xff, nb);            /* row 0 x plane 0: all bits */
        moty_hw_popc4x3(b, bs, pl, nb, c0); moty_hw_popc4x3_ref(b, bs, pl, nb, c1);
        CHECK(!memcmp(c0, c1, sizeof c0) && c0[0] == (uint32_t)nb*8);
        free(b); free(pl);
    }
    return 0;
}

#ifdef OPS_TEST_MAIN
int main(void) {
    struct { const char *n; int (*f)(void); } T[] = {
        {"rmsnorm", op_rmsnorm}, {"silu_mul", op_silu_mul}, {"softmax", op_softmax},
        {"axpy_add", op_axpy_add}, {"shortconv", op_shortconv}, {"rope_table", op_rope_table},
        {"attn_rows", op_attn_rows}, {"popc4x3", op_popc4x3} };
    int bad = 0;
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) { int r = T[i].f(); bad |= r; printf("[%s] %s\n", r ? "FAIL" : " OK ", T[i].n); }
    printf("tier: %s\n", HW_IDOT_KERNEL);
    return bad;
}
#endif
