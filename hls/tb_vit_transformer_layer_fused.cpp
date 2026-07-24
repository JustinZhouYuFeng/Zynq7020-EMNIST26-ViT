#include "vit_transformer_layer_fused.h"

int main()
{
    static fused_float_t token[VIT_FUSED_TOKENS][VIT_FUSED_EMBED_DIM];
    static fused_float_t out[VIT_FUSED_TOKENS][VIT_FUSED_EMBED_DIM];
    static fused_float_t norm1_w[VIT_FUSED_EMBED_DIM];
    static fused_float_t norm1_b[VIT_FUSED_EMBED_DIM];
    static fused_float_t norm2_w[VIT_FUSED_EMBED_DIM];
    static fused_float_t norm2_b[VIT_FUSED_EMBED_DIM];
    static fused_float_t qkv_w[VIT_FUSED_EMBED_DIM][VIT_FUSED_QKV_DIM];
    static fused_float_t qkv_b[VIT_FUSED_QKV_DIM];
    static fused_float_t attn_w[VIT_FUSED_EMBED_DIM][VIT_FUSED_EMBED_DIM];
    static fused_float_t attn_b[VIT_FUSED_EMBED_DIM];
    static fused_float_t fc1_w[VIT_FUSED_MLP_DIM][VIT_FUSED_EMBED_DIM];
    static fused_float_t fc1_b[VIT_FUSED_MLP_DIM];
    static fused_float_t gelu_lut[VIT_FUSED_GELU_LUT_ENTRIES];
    static fused_float_t fc2_w[VIT_FUSED_EMBED_DIM][VIT_FUSED_MLP_DIM];
    static fused_float_t fc2_b[VIT_FUSED_EMBED_DIM];

    for (int e = 0; e < VIT_FUSED_EMBED_DIM; ++e) {
        norm1_w[e] = 1.0f;
        norm2_w[e] = 1.0f;
    }
    for (int i = 0; i < VIT_FUSED_GELU_LUT_ENTRIES; ++i) {
        gelu_lut[i] = 0.0f;
    }

    vit_transformer_layer_fused(
        token,
        norm1_w,
        norm1_b,
        qkv_w,
        qkv_b,
        attn_w,
        attn_b,
        norm2_w,
        norm2_b,
        fc1_w,
        fc1_b,
        gelu_lut,
        fc2_w,
        fc2_b,
        out,
        0.01f,
        0.01f,
        0.01f,
        0.01f,
        127.0f,
        -8.0f,
        15.9375f);

    return 0;
}
