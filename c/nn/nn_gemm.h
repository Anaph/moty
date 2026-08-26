/* nn_gemm.h — umbrella header for Mat, matmul, quant, attention kernels.
 *
 * This file is kept for backwards compatibility. New code should include
 * the individual headers directly:
 *   nn_mat.h          — Mat struct, MODEL_COMMON_FIELDS, mat_apply
 *   nn_matmul.h       — matmul_q_s, matmul_i4_s, matmul_i2_s, etc.
 *   nn_quant.h        — quantize_rows, pack_int4, pack_int4_grouped, pack_int2
 *   nn_attn_kernels.h — att_scores_f32/i8, att_accum_f32/i8
 */
#ifndef NN_GEMM_H
#define NN_GEMM_H
#include "nn/nn_quant.h"
#include "nn/nn_matmul.h"
#include "nn/nn_attn_kernels.h"
#include "nn/nn_mat.h"
#endif /* NN_GEMM_H */
