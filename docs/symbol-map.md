# symbol-map — public surface per library (M0)

Naming convention: `moty_<module>_<what>`. This is the reviewed contract
that phases M1–M4 implement; shims keep old spellings alive until the last
consumer migrates. `docs/modularization-plan.md` has the full context.

## libmoty-hw (M1) — compute kernels, one implementation per -march tier

| Current | Exported as |
|---|---|
| `dot_i8i8(w, x, I)` | `moty_hw_dot_i8i8` |
| `dot_i4i8(w4, x8, I)` | `moty_hw_dot_i4i8` |
| `dot_i4i8p(w4, xip, xsum, I)` | `moty_hw_dot_i4i8p` |
| `dot_i4g8p(w4, ws, x8, xg, I)` | `moty_hw_dot_i4g8p` |
| `dot_f32`, `dot_f32i8` | `moty_hw_dot_f32`, `moty_hw_dot_f32i8` |
| `dn_row_decay_acc`, `dn_row_update_dot` | `moty_hw_dn_*` |
| `qrow_i8`, `px_permute`, `px_sum` | `moty_hw_qrow_i8`, `moty_hw_px_*` |

Contract notes carried over from `hw/hw.h`: `dot_i8i8` activations in
[-127,127] (qrow_i8 output); `dot_i4g8p` group size power of two ≥16.

## libmoty-nn-core (M2) — types + quantization

| Current | Exported as |
|---|---|
| `Mat`, `mat_apply`, `kv_store_row`, `mat_reset_storage` | `MotyMat`, `moty_nn_mat_apply`, … |
| `MODEL_COMMON_FIELDS` (macro) | `MotyCommon` (real struct, embedded as `Model.base`) |
| `Scratch` API (`scr_reset/reserve/take/free`) | `MotyScratch`, `moty_scr_*` |
| `pack_int4[_grouped]`, `pack_int2`, `quantize_rows` | `moty_nn_pack_*`, `moty_nn_quantize_rows` |

## libmoty-nn (M3) — layers, explicit variants

| Current (macro-pasted) | Exported as |
|---|---|
| `attention()` / `ENGINE_GATED_ATTN` | `moty_nn_attention`, `moty_nn_attention_gated` |
| `ATTN_NORM` field macro | `MotyLayerView` descriptor |
| `moe_decode1` / `MOE_GATE_SIGMOID` | `moty_nn_moe_sigmoid_d1`, `moty_nn_moe_topk_d1` |
| `MOE_SHARED_EXPERT` | `MOTY_MOE_SHARED` flag in `MotyMoeDesc` |
| `MOE_LOAD_EXPERT` | `load_expert` fn-ptr in `MotyMoeDesc` |
| `conv_layer`, `dense_ffn` | `moty_nn_conv_layer`, `moty_nn_dense_ffn` |
| `pick_tok`, `dist_build` (+ `g_temp` globals) | `moty_nn_pick(MotySampler*, …)` state struct |

## libmoty-runtime (M4) — scaffolding + expert cache

| Current (static hook) | Exported as |
|---|---|
| 11 static hooks + `engine_main()` | `MotyEngineOps` vtable + `moty_rt_serve(&ops)` |
| `g_kv_bits`, `g_qgroup`, `g_prefill_chunk`, … | `MotyConfig` + `moty_rt_config_from_env()` |
| `ExpertCache` (`runtime/moe.h`) | `moty_rt_expertcache_*` |
| `load_cfg`/`cfg_common` JSON plumbing | `moty_rt_cfg_common` |
