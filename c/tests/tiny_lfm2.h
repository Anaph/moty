/* tiny_lfm2.h — a synthetic LFM2 snapshot (HF names, Lfm2ForCausalLM like
 * LFM2.5-350M): conv/attention alternating, QK-norm, FFN with
 * block_auto_adjust_ff_dim; wrap = 1 writes the LFM2-VL layout (config
 * under text_config, tensors under "model.language_model."). Used by the
 * lfm2 engine tests and the library API tests. Needs tiny_st.h. */
#ifndef TINY_LFM2_H
#define TINY_LFM2_H
#include "tiny_st.h"

enum { LD = 32, LL = 4, LH = 4, LKV = 2, LHD = 8, LI = 64, LV = 40, LK = 3 };
/* intermediate_size 100 -> int(2*100/3)=66 -> round up to 32 -> 96 */
#define LI_CFG 100
#define LI_EFF 96
static const int lt_is_attn[LL] = {0, 1, 0, 1};

static void lt_norm_w(const char *name, int n) {           /* 1 + small noise */
    char sh[32]; snprintf(sh, sizeof sh, "[%d]", n);
    float *w = tst_add(name, sh, n, 0.2f, 0);
    for (int i = 0; i < n; i++) w[i] += 1.f;
}

/* wrap = 1: the layout of a multimodal checkpoint (LFM2-VL): config under
 * text_config, tensors under "model.language_model." */
static void lt_write_dir_w(const char *dir, int wrap);
static void lt_write_dir(const char *dir) { lt_write_dir_w(dir, 0); }
static void lt_write_dir_w(const char *dir, int wrap) {
    char cfg[2048];
    snprintf(cfg, sizeof cfg, "%s%s%s", wrap ? "{\"model_type\":\"lfm2_vl\",\"image_token_id\":38,\"text_config\":" : "",
        "{\"architectures\":[\"Lfm2ForCausalLM\"],\"model_type\":\"lfm2\","
        "\"hidden_size\":32,\"num_hidden_layers\":4,\"num_attention_heads\":4,"
        "\"num_key_value_heads\":2,\"intermediate_size\":100,"
        "\"block_auto_adjust_ff_dim\":true,\"block_ffn_dim_multiplier\":1.0,\"block_multiple_of\":32,"
        "\"vocab_size\":40,\"norm_eps\":1e-05,\"conv_L_cache\":3,\"conv_bias\":false,"
        "\"rope_parameters\":{\"rope_theta\":10000.0,\"rope_type\":\"default\"},"
        "\"tie_embedding\":true,\"eos_token_id\":7,\"max_position_embeddings\":256,"
        "\"layer_types\":[\"conv\",\"full_attention\",\"conv\",\"full_attention\"]}", wrap ? "}" : "");
    tst_write_text(dir, "config.json", cfg);
    tst_reset(11);
    char nm[128];
    const char *pf = wrap ? "model.language_model." : "model.";
    snprintf(nm, sizeof nm, "%sembed_tokens.weight", pf); tst_add(nm, "[40,32]", LV*LD, 1.0f, 0);
    snprintf(nm, sizeof nm, "%sembedding_norm.weight", pf); lt_norm_w(nm, LD);
    for (int i = 0; i < LL; i++) {
        #define AT(suffix, shape, numel, sc) \
            do { snprintf(nm,sizeof(nm),"%slayers.%d." suffix,pf,i); tst_add(nm,shape,numel,sc,0); } while(0)
        #define ATN(suffix, n) do { snprintf(nm,sizeof(nm),"%slayers.%d." suffix,pf,i); lt_norm_w(nm,n); } while(0)
        ATN("operator_norm.weight", LD);
        ATN("ffn_norm.weight", LD);
        if (lt_is_attn[i]) {
            AT("self_attn.q_proj.weight", "[32,32]", LH*LHD*LD, 0.4f);
            AT("self_attn.k_proj.weight", "[16,32]", LKV*LHD*LD, 0.4f);
            AT("self_attn.v_proj.weight", "[16,32]", LKV*LHD*LD, 0.4f);
            AT("self_attn.out_proj.weight", "[32,32]", LD*LH*LHD, 0.4f);
            ATN("self_attn.q_layernorm.weight", LHD);
            ATN("self_attn.k_layernorm.weight", LHD);
        } else {
            AT("conv.in_proj.weight", "[96,32]", 3*LD*LD, 0.4f);
            AT("conv.out_proj.weight", "[32,32]", LD*LD, 0.4f);
            AT("conv.conv.weight", "[32,1,3]", LD*LK, 0.8f);
        }
        AT("feed_forward.w1.weight", "[96,32]", LI_EFF*LD, 0.4f);
        AT("feed_forward.w3.weight", "[96,32]", LI_EFF*LD, 0.4f);
        AT("feed_forward.w2.weight", "[32,96]", LD*LI_EFF, 0.4f);
        #undef AT
        #undef ATN
    }
    tst_write(dir);
}

#endif /* TINY_LFM2_H */
