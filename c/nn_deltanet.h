/* Gated DeltaNet (linear attention with delta rule) — shared kernel.
 * Estratto da qwenmoe.c / qwen.c per deduplicazione.
 *
 * Requisiti del Layer (definito dal caller):
 *   Mat aqkv, az, ab, aa, aout;
 *   float *conv_w, *conv_b, *dt_bias, *A_log, *dn_norm;
 *   float *conv_state, *Sstate;
 *
 * Requisiti del Cfg:
 *   int lin_hv, lin_hk, lin_dk, lin_dv, lin_conv; float eps;
 *
 * Compile-time switches:
 *   ENGINE_PRECOOKED_A  — A_log contiene gia' -exp(A_log) (qwen35moe GGUF)
 *
 * Richiede: nn_rope.h, nn_gemm.h (mat_apply), simd.h (dn_row_*).
 */
#ifndef NN_DELTANET_H
#define NN_DELTANET_H

#ifndef MAX_LIN_DV
#define MAX_LIN_DV 1024
#endif

/* un token: x[D] (gia' normato) -> out[D]; aggiorna conv_state + Sstate */
static void deltanet_token(Model *m, Layer *l, const float *x, float *out) {
    Cfg *c = &m->c;
    int Hv = c->lin_hv, Hk = c->lin_hk, dk = c->lin_dk, dv = c->lin_dv, K = c->lin_conv;
    int kd = Hk*dk, vd = Hv*dv, cd = 2*kd + vd, R = Hv/Hk;
    float *qkv = falloc(cd), *z = falloc(vd), *b = falloc(Hv), *a = falloc(Hv);
    mat_apply(qkv, x, &l->aqkv, 1);
    mat_apply(z,   x, &l->az,   1);
    mat_apply(b,   x, &l->ab,   1);
    mat_apply(a,   x, &l->aa,   1);
    float *q = qkv, *k = qkv + kd, *v = qkv + 2*kd;
    float qscale = 1.f / sqrtf((float)dk);
    float *o = falloc(vd);
    #pragma omp parallel
    {
    #pragma omp for schedule(static)
    for (int ch = 0; ch < cd; ch++) {
        float *cs = l->conv_state + (int64_t)ch*K;
        memmove(cs, cs+1, (K-1)*sizeof(float));
        cs[K-1] = qkv[ch];
        const float *w = l->conv_w + (int64_t)ch*K;
        float vv = 0; for (int t = 0; t < K; t++) vv += cs[t]*w[t];
        if (l->conv_b) vv += l->conv_b[ch];
        qkv[ch] = vv * sigmoidf(vv);
    }
    #pragma omp single
    for (int h = 0; h < Hk; h++) {
        l2norm_head(q + (int64_t)h*dk, dk);
        l2norm_head(k + (int64_t)h*dk, dk);
        for (int i = 0; i < dk; i++) q[(int64_t)h*dk + i] *= qscale;
    }
    #pragma omp for schedule(static)
    for (int hv = 0; hv < Hv; hv++) {
        int hk = hv / R;
        const float *qh = q + (int64_t)hk*dk, *kh = k + (int64_t)hk*dk, *vh = v + (int64_t)hv*dv;
        float *S = l->Sstate + (int64_t)hv*dk*dv;
#ifdef ENGINE_PRECOOKED_A
        float g = l->A_log[hv] * softplusf(a[hv] + l->dt_bias[hv]);
#else
        float g = -expf(l->A_log[hv]) * softplusf(a[hv] + l->dt_bias[hv]);
#endif
        float beta = sigmoidf(b[hv]), dec = expf(g);
        float kv[MAX_LIN_DV], delta[MAX_LIN_DV];
        for (int j = 0; j < dv; j++) kv[j] = 0;
        for (int i = 0; i < dk; i++) dn_row_decay_acc(S + (int64_t)i*dv, dec, kh[i], kv, dv);
        for (int j = 0; j < dv; j++) delta[j] = (vh[j] - kv[j]) * beta;
        float *oh = o + (int64_t)hv*dv;
        for (int j = 0; j < dv; j++) oh[j] = 0;
        for (int i = 0; i < dk; i++) dn_row_update_dot(S + (int64_t)i*dv, kh[i], delta, qh[i], oh, dv);
        double ms = 0; for (int j = 0; j < dv; j++) ms += (double)oh[j]*oh[j];
        float r = 1.f / sqrtf((float)(ms/dv) + c->eps);
        const float *zh = z + (int64_t)hv*dv;
        for (int j = 0; j < dv; j++) oh[j] = oh[j]*r*l->dn_norm[j]*(zh[j]*sigmoidf(zh[j]));
    }
    }
    mat_apply(out, o, &l->aout, 1);
    free(qkv); free(z); free(b); free(a); free(o);
}

/* deltanet su S token in sequenza (prefill = ricorrenza per token) */
static void deltanet(Model *m, Layer *l, float *x, int S, float *out) {
    int D = m->c.hidden;
    for (int s = 0; s < S; s++) deltanet_token(m, l, x + (int64_t)s*D, out + (int64_t)s*D);
}

#endif /* NN_DELTANET_H */
