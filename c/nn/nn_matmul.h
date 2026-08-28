/* nn_matmul.h — GEMV/GEMM kernels: f32, int8 (IDOT), int4 (IDOT4/f32), int4-grouped, int2.
 *
 * Depends on: hw.h (dot_i8i8, dot_i4i8, dot_f32), nn_alloc.h (grow, falloc).
 * nn_quant.h is NOT required (matmul reads already-packed weights).
 *
 * IDOT path (int8): quantize x once, then VPDPBUSD per weight row.
 * IDOT4 path (int4): same but with nibble unpack (opt-in via IDOT4=1).
 * f32 path: exact dequant-on-fly for int4/int2.
 */
#ifndef NN_MATMUL_H
#define NN_MATMUL_H

/* M1: hw.h non diffonde piu' le intrinsics — chi le usa se le include */
#if defined(__AVX2__)
  #include <immintrin.h>
  static inline float simd_hsum256_f32(__m256 v) {
      __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
      lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
      sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
  }
#endif

#define NN_QROW_MAX 16384

/* ---- f32 ---- */


/* ---- int8 + per-row scale (Q8_0 style) with IDOT ---- */




/* ---- int4 + per-row scale with IDOT4 ---- */


/* ---- int4 + per-group scale ---- */


/* ---- int2 + per-row scale ---- */


/* Q4_K/Q6_K native matmul: dequant-on-fly using gguf_dq_* (need gguf.h before nn.h) */
#ifdef GGUF_H



#endif /* GGUF_H */


/* M3: implementazioni in nn/matmul.c (libmoty-nn) */
void moty_matmul(float *y, const float *x, const float *W, int S, int I, int O);
void moty_matmul_q_s(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O);
void moty_matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O);
void moty_matmul_i4_s(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O);
void moty_matmul_i4_grouped_s(float *y, const float *x, const uint8_t *q4, const float *scale,
                                int S, int I, int O, int gs);
void moty_matmul_i2_s(float *y, const float *x, const uint8_t *q2, const float *scale, int S, int I, int O);
void moty_matmul_q4k_native(float *y, const float *x, const uint8_t *raw,
                              int S, int I, int O);
void moty_matmul_q6k_native(float *y, const float *x, const uint8_t *raw,
                              int S, int I, int O);

#ifndef MOTY_CORE_NO_LEGACY
#define matmul moty_matmul
#define matmul_q_s moty_matmul_q_s
#define matmul_q moty_matmul_q
#define matmul_i4_s moty_matmul_i4_s
#define matmul_i4_grouped_s moty_matmul_i4_grouped_s
#define matmul_i2_s moty_matmul_i2_s
#define matmul_q4k_native moty_matmul_q4k_native
#define matmul_q6k_native moty_matmul_q6k_native
#endif

#endif /* NN_MATMUL_H */
