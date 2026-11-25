/* nn_rope.h — RoPE + activation primitives. Pure scalar C, no SIMD.
 * Include from engine .c files before nn_attn.h / nn_deltanet.h. */
#ifndef NN_ROPE_H
#define NN_ROPE_H

#include <math.h>

static inline float softplusf(float x) { return x > 20.f ? x : logf(1.f + expf(x)); }
static inline float sigmoidf(float x) { return x > 20.f ? 1.f : x < -20.f ? 0.f : 1.f / (1.f + expf(-x)); }



/* RoPE Neox-style half-split sui primi `rot` dimensioni di un head.
 * Modifica x[in] in-place. theta = base frequency, pos = position. */


/* M3: implementazioni in nn/rope.c (libmoty-nn) */
void moty_rope_head(float *x, int pos, float theta, int rot);
/* Prepare on the serial caller before sharing the borrowed rows with
 * workers. The rows remain valid until the table grows again. */
const float *moty_rope_rows(float theta, int rot, int first, int count);
void moty_rope_apply_row(float *x, int rot, const float *cs);
void moty_l2norm_head(float *x, int d);
#ifndef MOTY_CORE_NO_LEGACY
#define rope_head   moty_rope_head
#define l2norm_head moty_l2norm_head
#endif

#endif /* NN_ROPE_H */
