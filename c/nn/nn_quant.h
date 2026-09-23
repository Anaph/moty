/* nn_quant.h — weight quantization: int8, int4, int4-grouped, int2.
 *
 * Depends on: hw.h (qrow_i8), nn_alloc.h (grow), libc.
 * No model-specific types. No SIMD intrinsics directly (uses qrow_i8 from hw).
 */
#ifndef NN_QUANT_H
#define NN_QUANT_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* M2: le IMPLEMENTAZIONI vivono in nn/quant.c (libmoty-nn). Qui i prototipi
 * e le macro legacy che tengono i vecchi nomi nei motori finche' M3/M4 non
 * migrano i chiamanti (docs/symbol-map.md). */

void moty_quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits);
void moty_pack_int4(const float *w, uint8_t *q4, float *scale, int O, int I);
void moty_pack_int4_grouped(const float *w, uint8_t *q4, float *scale, int O, int I, int gs);
void moty_pack_int2(const float *w, uint8_t *q2, float *scale, int O, int I, int bits);
/* Q4R4 (hw/hw_q4r4.h): pack nr<=4 rows w[nr][I] (I%32==0) into one 4-row
 * block: blk[nb*64] nibbles + d[nb*4] f16 scales; rows >= nr are zero.
 * Per group: "no-clip Q4_0" scale (see nn/quant.c), stored signed in f16. */
void moty_pack_q4r4_block(const float *w, int nr, int I, uint8_t *blk, uint16_t *d);
uint16_t moty_f32_to_f16(float f);

/* macro legacy (M2 strangler) */
#ifndef MOTY_CORE_NO_LEGACY
#define quantize_rows     moty_quantize_rows
#define pack_int4         moty_pack_int4
#define pack_int4_grouped moty_pack_int4_grouped
#define pack_int2         moty_pack_int2
#endif

#endif /* NN_QUANT_H */
