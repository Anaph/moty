/* sample.c — M3 libmoty-nn: sampler (argmax/nucleus/pick). */
#include "nn/nn_sample.h"

float moty_g_temp = 0.7f, moty_g_nuc = 0.95f;
uint64_t moty_g_rng = 0x9E3779B97F4A7C15ULL;
static const float *g_cmp_p = NULL;
static int cmp_pdesc(const void *a,const void *b);

int moty_argmax_v(const float *lo, int V){
    int b=0; float bv=lo[0];
    for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;}
    return b;
}

/* per-thread partials of the vocabulary scans below */
typedef struct { const float *lo; float *p; int *bid; float *bvl; double *sum; int *cnt; int *off; int *pidx; float thr, mx, invt, inv; } VocJob;
static void argmax_part(void *c_, int64_t i0, int64_t i1, int t) {
    VocJob *c = c_; int nb = 0; float nv = -1e30f;
    for (int64_t i = i0; i < i1; i++) if (c->lo[i] > nv) { nv = c->lo[i]; nb = (int)i; }
    c->bid[t] = nb; c->bvl[t] = nv;
}
static void max_part(void *c_, int64_t i0, int64_t i1, int t) {      /* max of lo (or of p if lo is NULL) */
    VocJob *c = c_; const float *a = c->lo ? c->lo : c->p; float m = -1e30f;
    for (int64_t i = i0; i < i1; i++) if (a[i] > m) m = a[i];
    c->bvl[t] = m;
}
static void exp_part(void *c_, int64_t i0, int64_t i1, int t) {
    VocJob *c = c_; double s = 0;
    for (int64_t i = i0; i < i1; i++) { c->p[i] = expf((c->lo[i]-c->mx)*c->invt); s += c->p[i]; }
    c->sum[t] = s;
}
static void scale_part(void *c_, int64_t i0, int64_t i1, int t) {
    VocJob *c = c_;
    for (int64_t i = i0; i < i1; i++) c->p[i] *= c->inv;
}
static void count_part(void *c_, int64_t i0, int64_t i1, int t) {
    VocJob *c = c_; int n = 0;
    for (int64_t i = i0; i < i1; i++) if (c->p[i] > c->thr) n++;
    c->cnt[t] = n;
}
static void fill_part(void *c_, int64_t i0, int64_t i1, int t) {   /* same static split as count_part */
    VocJob *c = c_; int q = c->off[t];
    for (int64_t i = i0; i < i1; i++)
        if (c->p[i] > c->thr) { c->bvl[q] = c->p[i]; c->pidx[q] = (int)i; q++; }
}
/* partial max over threads (a thread with an empty range left -1e30) */
static float max_of(const float *v, int n) { float m = v[0]; for (int t = 1; t < n; t++) if (v[t] > m) m = v[t]; return m; }

int moty_argmax_v_par(Scratch *sc, const float *lo, int V){
    if (V < 8192) return argmax_v(lo, V);
    int bt = moty_par_threads();
    scr_reset(sc);
    scr_reserve(sc, scr_al((int64_t)bt*4) + scr_al((int64_t)bt*4));
    int *bid = scr_take(sc, scr_al((int64_t)bt*4));
    float *bvl = scr_take(sc, scr_al((int64_t)bt*4));
    for (int t = 0; t < bt; t++) { bid[t] = 0; bvl[t] = -1e30f; }
    VocJob c = { .lo = lo, .bid = bid, .bvl = bvl };
    moty_par_for(V, 0, argmax_part, &c);
    int b = bid[0]; float bv = bvl[0];
    for (int t = 1; t < bt; t++) if (bvl[t] > bv) { bv = bvl[t]; b = bid[t]; }
    return b;
}

void moty_dist_build(Scratch *sc, SampBuf *sb, const float *lo, int V){
    int nth = moty_par_threads();
    scr_reset(sc);
    scr_reserve(sc, 2*scr_al((int64_t)V*4) + 2*scr_al((int64_t)V*4)
                      + scr_al((int64_t)nth*4) + scr_al((int64_t)(nth+1)*4)
                      + scr_al((int64_t)nth*4) + scr_al((int64_t)nth*8));
    float *g_pbuf = scr_take(sc, (int64_t)V*4);   int *g_pidx = scr_take(sc, scr_al((int64_t)V*4));
    float *g_pbuf2 = scr_take(sc, (int64_t)V*4);  int *g_pidx2 = scr_take(sc, scr_al((int64_t)V*4));
    int *cnt = scr_take(sc, scr_al((int64_t)nth*4));
    int *off = scr_take(sc, scr_al((int64_t)(nth+1)*4));
    float *part = scr_take(sc, scr_al((int64_t)nth*4));
    double *psum = scr_take(sc, scr_al((int64_t)nth*8));
    g_cmp_p = g_pbuf;
    sb->pbuf = g_pbuf; sb->pidx = g_pidx; sb->pbuf2 = g_pbuf2; sb->pidx2 = g_pidx2;
    float invt=1.f/(g_temp>1e-4f?g_temp:1e-4f);
    if (V >= 16384) {
        /* V=128k: max+expf+sum+normalizza PARALLELI (seriali = ~1.5-2.5ms/tok) */
        for (int t = 0; t < nth; t++) { part[t] = -1e30f; psum[t] = 0; }
        VocJob c = { .lo = lo, .p = g_pbuf, .bvl = part, .sum = psum, .invt = invt };
        moty_par_for(V, 0, max_part, &c);
        c.mx = max_of(part, nth);
        moty_par_for(V, 0, exp_part, &c);
        double s = 0; for (int t = 0; t < nth; t++) s += psum[t];
        c.inv = (float)(1.0/s);
        moty_par_for(V, 0, scale_part, &c);
    } else {
        float mx=lo[0]; for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
        double s=0;
        for(int i=0;i<V;i++){ g_pbuf[i]=expf((lo[i]-mx)*invt); s+=g_pbuf[i]; }
        for(int i=0;i<V;i++) g_pbuf[i]/=(float)s;
    }
    if(g_nuc>0 && g_nuc<1.f){
        /* nucleus SENZA qsort completo (128k/tok = ~10ms seriali!): si
         * ordinano solo i candidati sopra soglia (tipicamente <1k); se la
         * loro massa non copre nuc, fallback sul qsort completo (raro). */
        float pmax;
        if (V >= 16384) {
            for (int t = 0; t < nth; t++) part[t] = -1e30f;
            VocJob c = { .p = g_pbuf, .bvl = part };
            moty_par_for(V, 0, max_part, &c);
            pmax = max_of(part, nth);
        } else { pmax = g_pbuf[0]; for(int i=1;i<V;i++) if(g_pbuf[i]>pmax) pmax=g_pbuf[i]; }
        float thr = pmax * 1e-5f;
        int nc = 0;
        if (V >= 16384) {
            /* scan+collect parallelo: conteggio per thread, offset, fill
             * (both passes on the same static split) */
            for (int t = 0; t < nth; t++) cnt[t] = 0;
            VocJob c = { .p = g_pbuf, .cnt = cnt, .off = off, .bvl = g_pbuf2, .pidx = g_pidx2, .thr = thr };
            moty_par_for(V, 0, count_part, &c);
            off[0] = 0;
            for (int t2 = 0; t2 < nth; t2++) off[t2+1] = off[t2] + cnt[t2];
            moty_par_for(V, 0, fill_part, &c);
            nc = off[nth];
        } else {
            for(int i=0;i<V;i++) if(g_pbuf[i] > thr) { g_pbuf2[nc] = g_pbuf[i]; g_pidx2[nc] = i; nc++; }
        }
        double cm = 0; for(int i=0;i<nc;i++) cm += g_pbuf2[i];
        if (cm >= (double)g_nuc && nc > 0 && nc < V/8) {
            /* ordina i soli candidati (crescente → poi invertito) */
            for(int i=1;i<nc;i++){ float p=g_pbuf2[i]; int ix=g_pidx2[i]; int j=i-1;
                while(j>=0 && g_pbuf2[j]<p){ g_pbuf2[j+1]=g_pbuf2[j]; g_pidx2[j+1]=g_pidx2[j]; j--; }
                g_pbuf2[j+1]=p; g_pidx2[j+1]=ix; }
            double cum=0; int keep=nc;
            for(int i=0;i<nc;i++){ cum+=g_pbuf2[i]; if(cum>=g_nuc){ keep=i+1; break; } }
            double s2=0; for(int i=0;i<keep;i++) s2+=g_pbuf2[i];
            float inv=1.f/(float)s2;
            for(int i=0;i<V;i++) g_pbuf[i]=0.f;
            for(int i=0;i<keep;i++) g_pbuf[g_pidx2[i]] = g_pbuf2[i]*inv;
        } else {
            for(int i=0;i<V;i++) g_pidx[i]=i;
            qsort(g_pidx,V,sizeof(int),cmp_pdesc);
            double cum=0; int keep=V;
            for(int i=0;i<V;i++){ cum+=g_pbuf[g_pidx[i]]; if(cum>=g_nuc){ keep=i+1; break; } }
            double s2=0; for(int i=keep;i<V;i++) g_pbuf[g_pidx[i]]=0;
            for(int i=0;i<keep;i++) s2+=g_pbuf[g_pidx[i]];
            for(int i=0;i<keep;i++) g_pbuf[g_pidx[i]]/=(float)s2;
        }
    }
}

int moty_dist_sample(const SampBuf *sb, int V){
    double u = rndu(), cum=0;
    for(int i=0;i<V;i++){ cum+=sb->pbuf[i]; if(cum>=u) return i; }
    for(int i=V-1;i>=0;i--) if(sb->pbuf[i]>0) return i;
    return 0;
}

int moty_pick_tok(Scratch *sc, const float *lo, int V){
    if(g_temp<=0) return argmax_v_par(sc,lo,V);
    SampBuf sb;
    dist_build(sc,&sb,lo,V);
    return dist_sample(&sb,V);
}

static int cmp_pdesc(const void *a,const void *b){
    float pa=g_cmp_p[*(const int*)a], pb=g_cmp_p[*(const int*)b];
    return pa<pb ? 1 : pa>pb ? -1 : 0; }
