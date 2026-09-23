/* ref_dense.h — independent double-precision reference pieces for engine
 * tests. Tensors are read by NAME straight from the safetensors shards (so
 * the engine's name mapping and loaders are part of what is checked), and
 * the math is a dumb serial reimplementation with no shared kernels
 * (docs/supporting-layers.md: "independent serial reference").
 * Include after the engine (needs io/st.h). */
#ifndef REF_DENSE_H
#define REF_DENSE_H
#include <math.h>
#include <stdlib.h>

static double *rd_load(shards *S, const char *name, int64_t n) {
    st_tensor *t = st_find(S, name);
    if (!t || t->numel != n) { fprintf(stderr, "ref_dense: %s missing or numel != %lld\n", name, (long long)n); exit(1); }
    float *f = malloc(n * sizeof(float));
    st_read_f32(S, name, f, 0);
    double *d = malloc(n * sizeof(double));
    for (int64_t i = 0; i < n; i++) d[i] = f[i];
    free(f);
    return d;
}

static void rd_rmsnorm(double *o, const double *x, const double *w, int n, double eps) {
    double ms = 0; for (int i = 0; i < n; i++) ms += x[i] * x[i];
    double r = 1.0 / sqrt(ms / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
}

/* y[O] = W[O,I] x[I] */
static void rd_matvec(double *y, const double *W, const double *x, int O, int I) {
    for (int o = 0; o < O; o++) { double a = 0; for (int i = 0; i < I; i++) a += W[(int64_t)o*I + i] * x[i]; y[o] = a; }
}

/* neox rotate_half RoPE on one head (transformers apply_rotary_pos_emb) */
static void rd_rope(double *x, int pos, double theta, int hd) {
    int h = hd / 2;
    for (int j = 0; j < h; j++) {
        double ang = pos * pow(theta, -2.0 * j / hd), c = cos(ang), s = sin(ang);
        double a = x[j], b = x[j + h];
        x[j] = a * c - b * s; x[j + h] = b * c + a * s;
    }
}

/* causal GQA attention over T positions (non-incremental: the full
 * sequence at once, unlike the engines' KV cache).
 * q[T][H*hd], k/v[T][KV*hd] -> ctx[T][H*hd] */
static void rd_attention(double *ctx, const double *q, const double *k, const double *v,
                         int T, int H, int KV, int hd) {
    int G = H / KV; double *sc = malloc(T * sizeof(double));
    for (int t = 0; t < T; t++) for (int h = 0; h < H; h++) {
        const double *qh = q + (int64_t)t*H*hd + h*hd;
        int kh = h / G; double mx = -1e300;
        for (int u = 0; u <= t; u++) {
            const double *ku = k + (int64_t)u*KV*hd + kh*hd;
            double a = 0; for (int d = 0; d < hd; d++) a += qh[d] * ku[d];
            sc[u] = a / sqrt((double)hd); if (sc[u] > mx) mx = sc[u];
        }
        double z = 0; for (int u = 0; u <= t; u++) { sc[u] = exp(sc[u] - mx); z += sc[u]; }
        double *c = ctx + (int64_t)t*H*hd + h*hd;
        for (int d = 0; d < hd; d++) c[d] = 0;
        for (int u = 0; u <= t; u++) {
            const double *vu = v + (int64_t)u*KV*hd + kh*hd;
            for (int d = 0; d < hd; d++) c[d] += sc[u] / z * vu[d];
        }
    }
    free(sc);
}

/* out[D] = W2 (silu(W1 x) * W3 x) */
static void rd_swiglu(double *out, const double *W1, const double *W3, const double *W2,
                      const double *x, int D, int I) {
    double *g = malloc(I * sizeof(double)), *u = malloc(I * sizeof(double));
    rd_matvec(g, W1, x, I, D); rd_matvec(u, W3, x, I, D);
    for (int i = 0; i < I; i++) g[i] = g[i] / (1.0 + exp(-g[i])) * u[i];
    rd_matvec(out, W2, g, D, I);
    free(g); free(u);
}

#endif /* REF_DENSE_H */
