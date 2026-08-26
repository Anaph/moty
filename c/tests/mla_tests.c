/* Test di c/mla.h: l'assorption e' algebricamente equivalente al reconstruct
 * (q·(W·c) = (q·W)·c); verifichiamo che diano gli stessi score a meno dell'ordine
 * FP, e che l'encode (rmsnorm + rope half-split) matchi il riferimento scalare.
 * Convenzione 0=pass 1=fail 2=skip. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "mla.h"

static uint64_t mla_rng = 0x9E3779B97F4A7C15ULL;
static uint32_t mla_rnd(void){ mla_rng = mla_rng*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t)(mla_rng>>33); }
static float mla_fr(void){ return (float)((int)mla_rnd()%2000 - 1000) / 313.0f; }

/* assorption ≡ reconstruct sugli score: stesso q/kvb/Lc/Rc, due vie, devono
 * coincidere a meno dell'accumulo f32 (double qui dentro, tolleranza stretta). */
int mla_absorb_eq_reconstruct(void){
    for (int rep = 0; rep < 50; rep++) {
        MlaCfg c;
        c.H = 1 + mla_rnd()%8;
        c.qk_nope = 8 + mla_rnd()%48;          /* multipli/sottomultipli liberi */
        c.qk_rope = 8 + mla_rnd()%24; if (c.qk_rope & 1) c.qk_rope++;   /* pari per rope */
        c.v_head  = 8 + mla_rnd()%48;
        c.kv_lora = 16 + mla_rnd()%64;
        c.q_lora = c.kv_lora;
        c.eps = 1e-6f;
        int T = 1 + mla_rnd()%40;
        int qpos = T-1, t0 = 0;
        int R = c.kv_lora;
        float scale = 1.f/sqrtf((float)(c.qk_nope + c.qk_rope));

        float *qk_nope_h = malloc(c.qk_nope*sizeof(float));
        float *qk_rope_h = malloc(c.qk_rope*sizeof(float));
        float *kvb_nope_h = malloc((int64_t)c.qk_nope*R*sizeof(float));
        float *q_abs_h = malloc(R*sizeof(float));
        float *Lc = malloc((int64_t)T*R*sizeof(float));
        float *Rc = malloc((int64_t)T*c.qk_rope*sizeof(float));
        float *scr = malloc(T*sizeof(float)), *sca = malloc(T*sizeof(float));
        for (int i=0;i<c.qk_nope;i++) qk_nope_h[i]=mla_fr();
        for (int i=0;i<c.qk_rope;i++) qk_rope_h[i]=mla_fr();
        for (int i=0;i<(int)c.qk_nope*R;i++) kvb_nope_h[i]=mla_fr();
        for (int i=0;i<(int64_t)T*R;i++) Lc[i]=mla_fr();
        for (int i=0;i<(int64_t)T*c.qk_rope;i++) Rc[i]=mla_fr();

        /* q_abs_h = qk_nope_h · kvb_nope_h^T  (kvb_nope_h [qk_nope, kv_lora]) */
        for (int i = 0; i < R; i++) { double s=0;
            for (int d = 0; d < c.qk_nope; d++) s += (double)qk_nope_h[d]*kvb_nope_h[(int64_t)d*R+i];
            q_abs_h[i] = (float)s;
        }
        mla_reconstruct_scores(&c, scr, qk_nope_h, qk_rope_h, kvb_nope_h, Lc, Rc, t0, qpos, scale);
        mla_absorb_scores(&c, sca, q_abs_h, qk_rope_h, Lc, Rc, t0, qpos, scale);
        for (int t = t0; t <= qpos; t++) {
            double den = fabs(scr[t-t0]) > 1 ? fabs(scr[t-t0]) : 1;
            if (fabs((double)sca[t-t0] - scr[t-t0]) / den > 1e-4) {
                fprintf(stderr,"mla eq rep=%d t=%d: absorb=%.8f reconstruct=%.8f\n",rep,t,sca[t-t0],scr[t-t0]);
                return 1;
            }
        }
        free(qk_nope_h); free(qk_rope_h); free(kvb_nope_h); free(q_abs_h); free(Lc); free(Rc); free(scr); free(sca);
    }
    return 0;
}

/* encode: rmsnorm sul latente + rope half-split su k_rope, vs riferimento scalare. */
int mla_encode_correctness(void){
    MlaCfg c; c.kv_lora = 16; c.qk_rope = 16; c.eps = 1e-6f; c.q_lora = c.kv_lora;
    int R = c.kv_lora;
    float *kva = malloc((R+c.qk_rope)*sizeof(float));
    float *ln = malloc(R*sizeof(float));
    float *Lc = malloc(R*sizeof(float)), *Rc = malloc(c.qk_rope*sizeof(float));
    float theta = 10000.f; int pos = 7;
    for (int rep = 0; rep < 30; rep++) {
        for (int i=0;i<R+c.qk_rope;i++) kva[i]=mla_fr();
        for (int i=0;i<R;i++) ln[i]=mla_fr();
        mla_encode_kv(&c, Lc, Rc, kva, ln, pos, theta);
        /* rmsnorm riferimento */
        double ms=0; for (int i=0;i<R;i++) ms+=(double)kva[i]*kva[i];
        float r = 1.f/sqrtf((float)(ms/R)+c.eps);
        for (int i=0;i<R;i++) {
            float ref = (float)((double)kva[i]*r*ln[i]);
            if (fabsf(Lc[i]-ref) > 1e-6f) { fprintf(stderr,"mla encode rmsnorm i=%d %.7f/%.7f\n",i,Lc[i],ref); return 1; }
        }
        /* rope half-split riferimento */
        int h = c.qk_rope/2; const float *kr = kva+R;
        for (int j=0;j<h;j++){
            float inv=powf(theta,-2.0f*j/c.qk_rope), ang=pos*inv, cs=cosf(ang), sn=sinf(ang);
            float a=kr[j], b=kr[j+h], r0=a*cs-b*sn, r1=b*cs+a*sn;
            if (fabsf(Rc[j]-r0)>1e-6f || fabsf(Rc[j+h]-r1)>1e-6f){ fprintf(stderr,"mla encode rope j=%d\n",j); return 1; }
        }
    }
    free(kva); free(ln); free(Lc); free(Rc);
    return 0;
}

/* value_accum: out[d] += sc[t]*V_h[t][d] dove V_h = kvb_v_h·Lc. Verifica vs ref. */
int mla_value_accum_correctness(void){
    MlaCfg c; c.kv_lora=12; c.v_head=16; c.eps=1e-6f; c.q_lora=c.kv_lora;
    int R=c.kv_lora, V=c.v_head, T=5, t0=0, qpos=T-1;
    float *kvb_v_h=malloc((int64_t)V*R*sizeof(float));
    float *Lc=malloc((int64_t)T*R*sizeof(float)), *sc=malloc(T*sizeof(float));
    float out[64], ref[64];
    for (int i=0;i<V*R;i++) kvb_v_h[i]=mla_fr();
    for (int i=0;i<T*R;i++) Lc[i]=mla_fr();
    for (int i=0;i<T;i++) sc[i]=mla_fr();
    mla_value_accum(&c, out, sc, kvb_v_h, Lc, t0, qpos);
    for (int d=0;d<V;d++){ double s=0;
        for (int t=t0;t<=qpos;t++){ double v=0; const float *lat=Lc+(int64_t)t*R;
            const float *wrow=kvb_v_h+(int64_t)d*R;
            for (int i=0;i<R;i++) v+=(double)wrow[i]*lat[i]; s+=(double)sc[t-t0]*v; }
        ref[d]=(float)s;
        if (fabsf(out[d]-ref[d])>1e-5f){ fprintf(stderr,"mla value d=%d %.7f/%.7f\n",d,out[d],ref[d]); return 1; }
    }
    free(kvb_v_h); free(Lc); free(sc);
    return 0;
}
