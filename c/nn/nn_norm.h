/* nn_norm.h — norme attivazione (rmsnorm per riga) + softmax per riga. Estratto
 * da nn.h: primitive a granularita' di riga senza stato, dipendono solo da
 * <math.h>. Usate da ogni motore (attention, mlp, norme sandwich). */
#ifndef NN_NORM_H
#define NN_NORM_H
#include <math.h>

static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

#endif /* NN_NORM_H */
