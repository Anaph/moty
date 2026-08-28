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

int moty_argmax_v_par(Scratch *sc, const float *lo, int V){
    if (V < 8192) return argmax_v(lo, V);
    int bt = omp_get_max_threads();
    scr_reset(sc);
    scr_reserve(sc, scr_al((int64_t)bt*4) + scr_al((int64_t)bt*4));
    int *bid = scr_take(sc, scr_al((int64_t)bt*4));
    float *bvl = scr_take(sc, scr_al((int64_t)bt*4));
    #pragma omp parallel
    {
        int t = omp_get_thread_num(), nb = 0; float nv = -1e30f;
        #pragma omp for schedule(static)
        for (int i = 0; i < V; i++) if (lo[i] > nv) { nv = lo[i]; nb = i; }
        bid[t] = nb; bvl[t] = nv;
    }
    int b = bid[0]; float bv = bvl[0];
    for (int t = 1; t < bt; t++) if (bvl[t] > bv) { bv = bvl[t]; b = bid[t]; }
    return b;
}

void moty_dist_build(Scratch *sc, SampBuf *sb, const float *lo, int V){
    int nth = omp_get_max_threads();
    scr_reset(sc);
    scr_reserve(sc, 2*scr_al((int64_t)V*4) + 2*scr_al((int64_t)V*4)
                      + scr_al((int64_t)nth*4) + scr_al((int64_t)(nth+1)*4));
    float *g_pbuf = scr_take(sc, (int64_t)V*4);   int *g_pidx = scr_take(sc, scr_al((int64_t)V*4));
    float *g_pbuf2 = scr_take(sc, (int64_t)V*4);  int *g_pidx2 = scr_take(sc, scr_al((int64_t)V*4));
    int *cnt = scr_take(sc, scr_al((int64_t)nth*4));
    int *off = scr_take(sc, scr_al((int64_t)(nth+1)*4));
    g_cmp_p = g_pbuf;
    sb->pbuf = g_pbuf; sb->pidx = g_pidx; sb->pbuf2 = g_pbuf2; sb->pidx2 = g_pidx2;
    float invt=1.f/(g_temp>1e-4f?g_temp:1e-4f);
    if (V >= 16384) {
        /* V=128k: max+expf+sum+normalizza PARALLELI (seriali = ~1.5-2.5ms/tok) */
        float mx = -1e30f;
        #pragma omp parallel for reduction(max:mx) schedule(static)
        for (int i = 0; i < V; i++) if (lo[i] > mx) mx = lo[i];
        double s = 0;
        #pragma omp parallel for reduction(+:s) schedule(static)
        for (int i = 0; i < V; i++) { g_pbuf[i] = expf((lo[i]-mx)*invt); s += g_pbuf[i]; }
        float inv = (float)(1.0/s);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < V; i++) g_pbuf[i] *= inv;
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
        if (V >= 16384) { pmax = -1e30f;
            #pragma omp parallel for reduction(max:pmax) schedule(static)
            for (int i = 0; i < V; i++) if (g_pbuf[i] > pmax) pmax = g_pbuf[i];
        } else { pmax = g_pbuf[0]; for(int i=1;i<V;i++) if(g_pbuf[i]>pmax) pmax=g_pbuf[i]; }
        float thr = pmax * 1e-5f;
        int nc = 0;
        if (V >= 16384) {
            /* scan+collect parallelo: conteggio per thread, offset, fill */
            (void)0;
            #pragma omp parallel
            {
                int t = omp_get_thread_num();
                int c = 0;
                #pragma omp for schedule(static) nowait
                for (int i = 0; i < V; i++) if (g_pbuf[i] > thr) c++;
                cnt[t] = c;
            }
            off[0] = 0;
            for (int t2 = 0; t2 < nth; t2++) off[t2+1] = off[t2] + cnt[t2];
            #pragma omp parallel
            {
                int t = omp_get_thread_num();
                int p = off[t];
                int i0 = (int)((int64_t)V * t / nth), i1 = (int)((int64_t)V * (t+1) / nth);
                for (int i = i0; i < i1; i++)
                    if (g_pbuf[i] > thr) { g_pbuf2[p] = g_pbuf[i]; g_pidx2[p] = i; p++; }
            }
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
