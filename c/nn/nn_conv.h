/* Shared short-conv (depthwise causal conv1d + gating). Include AFTER
 * Layer/Cfg/Model are defined.
 *
 * Conv weight layout: GGUF stores PyTorch Conv1d[D,K] with reversed ggml dims
 * → flat data is [D, K] C-order (channel-major). Access: conv_w[ch*K + t].
 * Conv state: [D, dc] same convention, dc = K-1.
 *
 * UNA regione parallela con barrier: in_proj (VNNI grouped) → conv depthwise
 * (canali partizionati, sequenziale nei token) → out_proj (VNNI grouped).
 * Richiede WF_I4G gs=32 e D%64==0 per il path VNNI; fallback: mat_apply
 * fuori regione (2 fork/join, come prima).
 *
 * Requires Layer fields: in_proj, out_proj (Mat), conv_w (float*), conv_state (float*)
 * Requires Cfg fields: hidden, conv_L
 */
#ifndef NN_CONV_H
#define NN_CONV_H

static void conv_layer(Model *m, Layer *l, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, K = c->conv_L;
    /* P5: arena per-Model, reserve unica per tutti i chunk (path VNNI completo) */
    {
        int ng_ = (D+31)/32;
        scr_reset(&m->scr);
        scr_reserve(&m->scr, scr_al((int64_t)S*3*D*4) + scr_al((int64_t)S*D*4)
                          + 2*scr_al((int64_t)S*D) + 2*scr_al((int64_t)S*ng_*4)
                          + 2*scr_al((int64_t)S*4));
    }
    float *bcx = scr_take(&m->scr, (int64_t)S*3*D*4);
    float *ybuf = scr_take(&m->scr, (int64_t)S*D*4);
    int8_t *cxi = scr_take(&m->scr, scr_al((int64_t)S*D));
    int32_t *cxg = scr_take(&m->scr, scr_al((int64_t)S*((D+31)/32)*4));
    float   *csx = scr_take(&m->scr, scr_al((int64_t)S*4));
    int8_t *yqi = scr_take(&m->scr, scr_al((int64_t)S*D));
    int32_t *yqg = scr_take(&m->scr, scr_al((int64_t)S*((D+31)/32)*4));
    float   *ysx = scr_take(&m->scr, scr_al((int64_t)S*4));
    Mat *wi = &l->in_proj, *wo = &l->out_proj;
    int gs = wi->gs, ngD = (D+gs-1)/gs;
    int vnni_in  = (wi->fmt == WF_I4G && gs == 32 && (D & 63) == 0);
    int vnni_out = (wo->fmt == WF_I4G && wo->gs == 32 && (D & 63) == 0);
    { static int conv_new = -1;
      if (conv_new < 0) { const char *e = getenv("CONV_VNNI"); conv_new = e ? atoi(e) : 1; }
      if (!conv_new) { vnni_in = 0; vnni_out = 0; } }
    if (!vnni_in || !vnni_out) {
        /* path legacy: proiezioni batched via mat_apply (2 fork/join) */
        mat_apply(bcx, x, &l->in_proj, S);
        int dc = K - 1;
        for (int s = 0; s < S; s++) {
            float *row = bcx + (int64_t)s*3*D;
            float *b = row, *co = row+D, *xx = row+2*D;
            float *ys = ybuf + (int64_t)s*D;
            for (int ch = 0; ch < D; ch++) {
                float bx = b[ch]*xx[ch], acc = 0;
                for (int tt = 0; tt < dc; tt++) acc += l->conv_state[ch*dc+tt] * l->conv_w[ch*K+tt];
                acc += bx * l->conv_w[ch*K+dc];
                ys[ch] = co[ch] * acc;
            }
            if (dc > 0)
                for (int ch = 0; ch < D; ch++) {
                    for (int tt = 0; tt < dc-1; tt++) l->conv_state[ch*dc+tt] = l->conv_state[ch*dc+tt+1];
                    l->conv_state[ch*dc+dc-1] = b[ch]*xx[ch];
                }
        }
        mat_apply(out, ybuf, &l->out_proj, S);
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
    int rb = (D+1)/2;
    int dc = K - 1;
    int nth = omp_get_max_threads();
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        /* fase 1: in_proj — righe [0, S*3D) */
        #pragma omp for schedule(static)
        for (int r = 0; r < S*3*D; r++) {
            int s = r/(3*D), o = r - s*3*D;
            bcx[r] = csx[s] * dot_i4g8p(wi->q4 + (int64_t)o*rb, wi->qs + (int64_t)o*ngD,
                                        cxi + (int64_t)s*D, cxg + (int64_t)s*ngD, D);
        }
        /* fase 2: conv depthwise — canali propri; per TOKEN: usa lo stato
         * del canale e AGGIORNA subito (shift), sequenziale su s (causale).
         * Ogni thread tocca solo i propri canali → niente race. */
        int ch0 = (int)((int64_t)D * t / nth), ch1 = (int)((int64_t)D * (t+1) / nth);
        for (int s = 0; s < S; s++) {
            float *row = bcx + (int64_t)s*3*D;
            float *b = row, *co = row+D, *xx = row+2*D;
            float *ys = ybuf + (int64_t)s*D;
            for (int ch = ch0; ch < ch1; ch++) {
                float bx = b[ch]*xx[ch], acc = 0;
                for (int tt = 0; tt < dc; tt++) acc += l->conv_state[ch*dc+tt] * l->conv_w[ch*K+tt];
                acc += bx * l->conv_w[ch*K+dc];
                ys[ch] = co[ch] * acc;
                if (dc > 0) {
                    for (int tt = 0; tt < dc-1; tt++) l->conv_state[ch*dc+tt] = l->conv_state[ch*dc+tt+1];
                    l->conv_state[ch*dc+dc-1] = bx;
                }
            }
        }
        /* quant ybuf per token — BARRIER prima: omp for sincronizza all'USCITA,
         * non all'ingresso; senza questa un thread quota ybuf[0] mentre altri
         * calcolano ancora i canali del token 0 (race) */
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (int s = 0; s < S; s++) {
            ysx[s] = qrow_i8(ybuf + (int64_t)s*D, yqi + (int64_t)s*D, D);
            for (int g = 0; g < ngD; g++) {
                int32_t a = 0;
                for (int j = 0; j < 32; j++) a += yqi[(int64_t)s*D + g*32+j];
                yqg[(int64_t)s*ngD + g] = a;
            }
        }
        /* fase 3: out_proj */
        #pragma omp for schedule(static)
        for (int r = 0; r < S*D; r++) {
            int s = r/D, o = r - s*D;
            out[r] = ysx[s] * dot_i4g8p(wo->q4 + (int64_t)o*rb, wo->qs + (int64_t)o*ngD,
                                        yqi + (int64_t)s*D, yqg + (int64_t)s*ngD, D);
        }
    }
}

#endif /* NN_CONV_H */
