/* mla.h — Multi-head Latent Attention (stile DeepSeek-V3 / GLM-5.2), estrazione
 * di MATEMATICA PURA: le 3 primitive MLA (encode del KV latente, scoring per
 * ABSORPTION, scoring per RECONSTRUCT) su array f32 piattissimi, SENZA legami a
 * QT/CUDA/disk/tier. Un engine MLA fornisce le proiezioni (q_a/q_b/kv_a/kv_b)
 * come Mat e chiama queste primitive; glm.c mantiene la sua versione ottimizzata
 * e legata al suo path (CUDA/PIPE/DSA), questa e' il RIFERIMENTO pulito per un
 * nuovo engine MLA (es. DeepSeek-V3) da copiare+adattare.
 *
 * Modello (DeepSeek-V3 / GLM-5.2 "kv_a_proj_with_mqa"):
 *   per token query:  qa = rmsnorm(q_a·x, q_a_ln) [q_lora];  q = q_b·qa [H*qkh]
 *                     split per testa qh = [qk_nope | qk_rope]; rope solo su qk_rope
 *   per token ctx t:  kva = kv_a·x_t [kv_lora+qk_rope];  kv_lat=kva[:kv_lora], k_rope=kva[kv_lora:]
 *                     Lc[t] = rmsnorm(kv_lat, kv_a_ln) [kv_lora]   (latente normato)
 *                     Rc[t] = rope(k_rope, t)         [qk_rope]    (key decoupled)
 *   score[h][t] = qk_nope_h · K_nope_h[t]  +  qk_rope_h · Rc[t]        (K_nope da kv_b)
 *
 * Due vie equivalenti per K_nope (il cuore di MLA):
 *   RECONSTRUCT:  K_nope_h[t] = kvb_nope_h · Lc[t]           (materializza K per tutti i t)
 *   ABSORPTION:   q_abs_h = qk_nope_h · kvb_nope_h^T [kv_lora]; score = q_abs_h · Lc[t]
 *                 (piega kv_b nel query: costo O(T·kv_lora) vs O(T·H·qk_nope) — vince in decode)
 * Assorption e' algebricamente IDENTICO a reconstruct (q·(W·c) = (q·W)·c): il test
 * mla_absorb_eq_reconstruct lo verifica. L'engine sceglie la via in base a S.
 *
 * Tutto static, solo <math.h>; niente SIMD qui (il calore e' nei matmul delle
 * proiezioni, che l'engine fa con mat_apply). Stesso pattern di simd.h/nn.h. */
#ifndef MLA_H
#define MLA_H
#include <math.h>

typedef struct {
    int H;            /* teste */
    int qk_nope;      /* dim query/key non-rotata */
    int qk_rope;      /* dim query/key rotata (decoupled) */
    int v_head;       /* dim value per testa */
    int q_lora;       /* rank latente query */
    int kv_lora;      /* rank latente KV (= dim di Lc) */
    float eps;        /* rms_norm eps */
} MlaCfg;

/* ---------- encode del KV latente per un token ----------
 * kva_out = output di kv_a_proj_with_mqa su x_t, lunghezza kv_lora+qk_rope
 *           (i primi kv_lora sono il latente, gli ultimi qk_rope sono k_rope).
 * kv_a_ln = peso rmsnorm [kv_lora] (self_attn.kv_a_layernorm).
 * Scrive Lc_dst[kv_lora] = rmsnorm(kva_out[:kv_lora], kv_a_ln) e
 *        Rc_dst[qk_rope] = rope_neox(kva_out[kv_lora:], pos, theta).
 * rope_neox half-split (coppie j, j+R/2), come qwen.c::rope_head / glm rope_interleave. */
static inline void mla_encode_kv(const MlaCfg *c, float *Lc_dst, float *Rc_dst,
                                 const float *kva_out, const float *kv_a_ln,
                                 int pos, float theta) {
    int R = c->kv_lora;
    /* rmsnorm sul latente */
    double ms = 0; for (int i = 0; i < R; i++) ms += (double)kva_out[i]*kva_out[i];
    float r = 1.f / sqrtf((float)(ms / R) + c->eps);
    for (int i = 0; i < R; i++) Lc_dst[i] = (float)((double)kva_out[i]*r*kv_a_ln[i]);
    /* rope half-split su k_rope */
    const float *kr = kva_out + R;
    int h = c->qk_rope / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(theta, -2.0f*j / c->qk_rope);
        float ang = pos*inv, cs = cosf(ang), sn = sinf(ang);
        float a = kr[j], b = kr[j+h];
        Rc_dst[j]   = a*cs - b*sn;
        Rc_dst[j+h] = b*cs + a*sn;
    }
}

/* ---------- scoring RECONSTRUCT per una testa ----------
 * qk_nope_h [qk_nope], qk_rope_h [qk_rope] = query gia' ruotata della testa h.
 * kvb_nope_h = righe di kv_b per la testa h, shape [qk_nope, kv_lora] row-major
 *             (O=qk_nope, I=kv_lora): K_nope_h[t] = kvb_nope_h · Lc[t].
 * Lc[t][kv_lora], Rc[t][qk_rope] per t in [t0,qpos]. scale = 1/sqrt(qk_nope+qk_rope).
 * Scrive sc[t-t0] = (qk_nope_h·K_nope_h[t] + qk_rope_h·Rc[t]) * scale. */
static inline void mla_reconstruct_scores(const MlaCfg *c, float *sc,
                                          const float *qk_nope_h, const float *qk_rope_h,
                                          const float *kvb_nope_h,
                                          const float *Lc, const float *Rc,
                                          int t0, int qpos, float scale) {
    int R = c->kv_lora;
    for (int t = t0; t <= qpos; t++) {
        const float *lat = Lc + (int64_t)t*R;
        double s = 0;
        for (int d = 0; d < c->qk_nope; d++) {
            const float *wrow = kvb_nope_h + (int64_t)d*R;
            double k = 0; for (int i = 0; i < R; i++) k += (double)wrow[i]*lat[i];
            s += (double)qk_nope_h[d]*k;
        }
        const float *kr = Rc + (int64_t)t*c->qk_rope;
        for (int d = 0; d < c->qk_rope; d++) s += (double)qk_rope_h[d]*kr[d];
        sc[t-t0] = (float)s * scale;
    }
}

/* ---------- scoring ABSORPTION per una testa (via equivalente, O(T·kv_lora)) ----------
 * q_abs_h [kv_lora] = qk_nope_h · kvb_nope_h^T  (il query "assorbito": piega kv_b
 *                     nel query invece di materializzare K). L'engine lo calcola
 *                     con un matmul (qk_nope_h [1,qk_nope] @ kvb_nope_h^T [qk_nope,kv_lora]).
 * score = (q_abs_h·Lc[t] + qk_rope_h·Rc[t]) * scale. Stesso risultato di reconstruct
 * a meno dell'ordine FP (q·(W·c) = (q·W)·c): il test verifica l'equivalenza. */
static inline void mla_absorb_scores(const MlaCfg *c, float *sc,
                                     const float *q_abs_h, const float *qk_rope_h,
                                     const float *Lc, const float *Rc,
                                     int t0, int qpos, float scale) {
    int R = c->kv_lora;
    for (int t = t0; t <= qpos; t++) {
        const float *lat = Lc + (int64_t)t*R;
        double s = 0; for (int i = 0; i < R; i++) s += (double)q_abs_h[i]*lat[i];
        const float *kr = Rc + (int64_t)t*c->qk_rope;
        for (int d = 0; d < c->qk_rope; d++) s += (double)qk_rope_h[d]*kr[d];
        sc[t-t0] = (float)s * scale;
    }
}

/* accumulo value (dopo softmax di sc): V_h[t] = kvb_v_h · Lc[t] [v_head], poi
 * out[d] += sc[t]*V_h[t][d]. kvb_v_h = righe value di kv_b per la testa h,
 * shape [v_head, kv_lora]. Serve sia ad absorb che a reconstruct (il value non
 * si assorbe: e' latente, va ricostruito). L'accumulo e' identico ai due path. */
static inline void mla_value_accum(const MlaCfg *c, float *out,
                                   const float *sc, const float *kvb_v_h,
                                   const float *Lc, int t0, int qpos) {
    int R = c->kv_lora, V = c->v_head;
    for (int d = 0; d < V; d++) out[d] = 0;
    for (int t = t0; t <= qpos; t++) {
        const float *lat = Lc + (int64_t)t*R;
        float a = sc[t-t0];
        for (int d = 0; d < V; d++) {
            const float *wrow = kvb_v_h + (int64_t)d*R;
            double v = 0; for (int i = 0; i < R; i++) v += (double)wrow[i]*lat[i];
            out[d] += a * (float)v;
        }
    }
}

#endif /* MLA_H */
