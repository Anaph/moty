/* conv.c — M3 libmoty-nn: conv corta causale fusa (da nn_conv.h, 1:1). */
#include "util/prof.h"
#include "nn/conv.h"

/* the phases of the VNNI path, each one parallel region */
typedef struct {
    const MotyConvView *cv; float *bcx, *ybuf; const int8_t *cxi; const int32_t *cxg; const float *csx;
    int8_t *yqi; int32_t *yqg; float *ysx; float *out; int S, D, K, ngD, rb;
} ConvJob;
static void conv_in_part(void *c_, int64_t r0, int64_t r1, int tid) {
    const ConvJob *c = c_; const Mat *wi = c->cv->in_proj; int D = c->D, ngD = c->ngD, rb = c->rb;
    for (int64_t r = r0; r < r1; r++) {
        int s = (int)(r/(3*D)), o = (int)(r - (int64_t)s*3*D);
        c->bcx[r] = c->csx[s] * dot_i4g8p(wi->q4 + (int64_t)o*rb, wi->qs + (int64_t)o*ngD,
                                          c->cxi + (int64_t)s*D, c->cxg + (int64_t)s*ngD, D);
    }
}
static void conv_dw_part(void *c_, int64_t ch0, int64_t ch1, int tid) {
    const ConvJob *c = c_; int D = c->D;
    for (int s = 0; s < c->S; s++) {
        float *row = c->bcx + (int64_t)s*3*D;
        moty_hw_shortconv_step(c->ybuf + (int64_t)s*D, row, row+D, row+2*D,
                               c->cv->conv_w, c->cv->conv_state, c->K, (int)ch0, (int)ch1);
    }
}
static void conv_q_part(void *c_, int64_t s0, int64_t s1, int tid) {
    const ConvJob *c = c_; int D = c->D, ngD = c->ngD;
    for (int64_t s = s0; s < s1; s++) {
        c->ysx[s] = qrow_i8(c->ybuf + s*D, c->yqi + s*D, D);
        for (int g = 0; g < ngD; g++) {
            int32_t a = 0;
            for (int j = 0; j < 32; j++) a += c->yqi[s*D + g*32+j];
            c->yqg[s*ngD + g] = a;
        }
    }
}
static void conv_out_part(void *c_, int64_t r0, int64_t r1, int tid) {
    const ConvJob *c = c_; const Mat *wo = c->cv->out_proj; int D = c->D, ngD = c->ngD, rb = c->rb;
    for (int64_t r = r0; r < r1; r++) {
        int s = (int)(r/D), o = (int)(r - (int64_t)s*D);
        c->out[r] = c->ysx[s] * dot_i4g8p(wo->q4 + (int64_t)o*rb, wo->qs + (int64_t)o*ngD,
                                          c->yqi + (int64_t)s*D, c->yqg + (int64_t)s*ngD, D);
    }
}

void moty_nn_conv_layer(const MotyConvView *cv, const float *x, int S, float *out) {
    int D = cv->hidden, K = cv->conv_L;
    /* P5: arena per-Model, reserve unica per tutti i chunk (path VNNI completo) */
    {
        int ng_ = (D+31)/32;
        scr_reset(cv->scr);
        scr_reserve(cv->scr, scr_al((int64_t)S*3*D*4) + scr_al((int64_t)S*D*4)
                          + 2*scr_al((int64_t)S*D) + 2*scr_al((int64_t)S*ng_*4)
                          + 2*scr_al((int64_t)S*4));
    }
    float *bcx = scr_take(cv->scr, (int64_t)S*3*D*4);
    float *ybuf = scr_take(cv->scr, (int64_t)S*D*4);
    int8_t *cxi = scr_take(cv->scr, scr_al((int64_t)S*D));
    int32_t *cxg = scr_take(cv->scr, scr_al((int64_t)S*((D+31)/32)*4));
    float   *csx = scr_take(cv->scr, scr_al((int64_t)S*4));
    int8_t *yqi = scr_take(cv->scr, scr_al((int64_t)S*D));
    int32_t *yqg = scr_take(cv->scr, scr_al((int64_t)S*((D+31)/32)*4));
    float   *ysx = scr_take(cv->scr, scr_al((int64_t)S*4));
    const Mat *wi = cv->in_proj, *wo = cv->out_proj;
    int gs = wi->gs, ngD = gs > 0 ? (D+gs-1)/gs : 0;   /* gs=0 for f32/int8 weights */
    int vnni_in  = (wi->fmt == WF_I4G && gs == 32 && (D & 63) == 0);
    int vnni_out = (wo->fmt == WF_I4G && wo->gs == 32 && (D & 63) == 0);
    { static int conv_new = -1;
      if (conv_new < 0) { const char *e = getenv("CONV_VNNI"); conv_new = e ? atoi(e) : 1; }
      if (!conv_new) { vnni_in = 0; vnni_out = 0; } }
    if (!vnni_in || !vnni_out) {
        /* path legacy: proiezioni batched via mat_apply (2 fork/join) */
        OP_T(t_in);
        mat_apply(bcx, x, cv->in_proj, S);
        OP_ACC(OP_CONV_IN, t_in);
        OP_T(t_dw);
        for (int s = 0; s < S; s++) {
            float *row = bcx + (int64_t)s*3*D;
            moty_hw_shortconv_step(ybuf + (int64_t)s*D, row, row+D, row+2*D,
                                   cv->conv_w, cv->conv_state, K, 0, D);
        }
        OP_ACC(OP_CONV_DW, t_dw);
        OP_T(t_out);
        mat_apply(out, ybuf, cv->out_proj, S);
        OP_ACC(OP_CONV_OUT, t_out);
        return;
    }
    /* quant x per token (seriale: S=1 in decode; ~8us per S=27) */
    for (int s = 0; s < S; s++) {
        csx[s] = qrow_i8(x + (int64_t)s*D, cxi + (int64_t)s*D, D);
        for (int g = 0; g < ngD; g++) {
            int32_t a = 0;
            for (int j = 0; j < 32; j++) a += cxi[(int64_t)s*D + g*32+j];
            cxg[(int64_t)s*ngD + g] = a;
        }
    }
    ConvJob c = { cv, bcx, ybuf, cxi, cxg, csx, yqi, yqg, ysx, out, S, D, K, ngD, (D+1)/2 };
    moty_par_for((int64_t)S*3*D, 0, conv_in_part, &c);     /* fase 1: in_proj — righe [0, S*3D) */
    /* fase 2: conv depthwise — canali propri; per TOKEN: usa lo stato del
     * canale e AGGIORNA subito (shift), sequenziale su s (causale). Ogni
     * thread tocca solo i propri canali → niente race. The end of each
     * region is the barrier the next phase needs (ybuf complete before it
     * is quantized). */
    moty_par_for(D, 0, conv_dw_part, &c);
    moty_par_for(S, 0, conv_q_part, &c);                    /* quant ybuf per token */
    moty_par_for((int64_t)S*D, 0, conv_out_part, &c);       /* fase 3: out_proj */
}
