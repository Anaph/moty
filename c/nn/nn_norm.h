/* nn_norm.h — norme attivazione (rmsnorm per riga) + softmax per riga. Estratto
 * da nn.h: primitive a granularita' di riga senza stato, dipendono solo da
 * <math.h>. Usate da ogni motore (attention, mlp, norme sandwich). */
#ifndef NN_NORM_H
#define NN_NORM_H
#include <math.h>





/* M3: implementazioni in nn/norm.c (libmoty-nn) */
void moty_rmsnorm_row(float *out, const float *x, const float *w, int D, float eps);
void moty_softmax_row(float *x, int n);
#ifndef MOTY_CORE_NO_LEGACY
#define rmsnorm_row moty_rmsnorm_row
#define softmax_row  moty_softmax_row
#endif

#endif /* NN_NORM_H */
