/* nn/conv.h — short-conv (depthwise causal conv1d + gating), M3 public API.
 *
 * Conv weight layout: GGUF stores PyTorch Conv1d[D,K] with reversed ggml dims
 * → flat data is [D, K] C-order (channel-major). Access: conv_w[ch*K + t].
 * Conv state: [D, dc] same convention, dc = K-1.
 *
 * UNA regione parallela con barrier: in_proj (VNNI grouped) → conv depthwise
 * (canali partizionati, sequenziale nei token) → out_proj (VNNI grouped).
 * Richiede WF_I4G gs=32 e D%64==0 per il path VNNI; fallback: mat_apply
 * fuori regione (2 fork/join). CONV_VNNI=0 forza il fallback (A/B di test). */
#ifndef MOTY_NN_CONV_H
#define MOTY_NN_CONV_H

#include "nn/nn_mat.h"

typedef struct MotyConvView {
    const Mat *in_proj, *out_proj;   /* [3D,D], [D,D] */
    const float *conv_w;             /* [D*K] */
    float *conv_state;               /* [D*(K-1)] causale, aggiornata in-place */
    int hidden;                      /* D */
    int conv_L;                      /* K */
    Scratch *scr;
} MotyConvView;

void moty_nn_conv_layer(const MotyConvView *cv, const float *x, int S, float *out);

#endif /* MOTY_NN_CONV_H */
