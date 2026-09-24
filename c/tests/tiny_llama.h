/* tiny_llama.h — a synthetic Llama-family snapshot (model_type "llama", no
 * q/k norms, tied head, dims multiples of 32 so every packed format applies)
 * for the qwen/Llama engine through the public API (lookahead tests). */
#ifndef TINY_LLAMA_H
#define TINY_LLAMA_H
#include "tiny_st.h"

enum { TL_V = 48, TL_D = 64, TL_I = 128, TL_L = 2 };

static void tl_write_dir(const char *dir) {
    tst_write_text(dir, "config.json",
        "{\"architectures\":[\"LlamaForCausalLM\"],\"model_type\":\"llama\",\"hidden_size\":64,"
        "\"num_hidden_layers\":2,\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":16,"
        "\"intermediate_size\":128,\"vocab_size\":48,\"rope_theta\":10000.0,\"rms_norm_eps\":1e-05,"
        "\"tie_word_embeddings\":true,\"eos_token_id\":2,\"max_position_embeddings\":256}");
    tst_reset(21);
    tst_add("model.embed_tokens.weight", "[48,64]", TL_V*TL_D, 0.8f, 0);
    tst_add("model.norm.weight", "[64]", TL_D, 0, 1);
    char nm[128];
    for (int i = 0; i < TL_L; i++) {
        #define AT(suffix, shape, numel, sc, ones) \
            do { snprintf(nm, sizeof nm, "model.layers.%d." suffix, i); tst_add(nm, shape, numel, sc, ones); } while (0)
        AT("input_layernorm.weight", "[64]", TL_D, 0, 1);
        AT("post_attention_layernorm.weight", "[64]", TL_D, 0, 1);
        AT("self_attn.q_proj.weight", "[64,64]", 64*64, 0.3f, 0);
        AT("self_attn.k_proj.weight", "[32,64]", 32*64, 0.3f, 0);
        AT("self_attn.v_proj.weight", "[32,64]", 32*64, 0.3f, 0);
        AT("self_attn.o_proj.weight", "[64,64]", 64*64, 0.3f, 0);
        AT("mlp.gate_proj.weight", "[128,64]", TL_I*TL_D, 0.3f, 0);
        AT("mlp.up_proj.weight", "[128,64]", TL_I*TL_D, 0.3f, 0);
        AT("mlp.down_proj.weight", "[64,128]", TL_D*TL_I, 0.3f, 0);
        #undef AT
    }
    tst_write(dir);
}
#endif /* TINY_LLAMA_H */
