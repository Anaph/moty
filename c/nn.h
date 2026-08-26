/* nn.h — umbrella for all NN primitive headers.
 * Include this from engine .c files to get everything below L1. */
#ifndef NN_H
#define NN_H

#include "nn_alloc.h"
#include "hw.h"              /* L0: dot_*, qrow_i8, dn_row_*, WF_* enum */
#include "nn_quant.h"        /* L1: quantize_rows, pack_int4/2/grouped */
#include "nn_matmul.h"       /* L1: matmul, matmul_q_s, matmul_i4_s, matmul_i2_s */
#include "nn_attn_kernels.h" /* L1: att_scores_*, att_accum_* */
#include "nn_mat.h"          /* L1: Mat, MODEL_COMMON_FIELDS, mat_apply, kv_store_row */
#include "nn_norm.h"         /* L1: rmsnorm_row, softmax_row */
#include "nn_sample.h"       /* L1: pick_tok, nucleus, stop-set */

#endif /* NN_H */
