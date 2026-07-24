#ifndef VIT_TRANSFORMER_LAYER_FUSED_H
#define VIT_TRANSFORMER_LAYER_FUSED_H

#include <stdint.h>

static const int VIT_FUSED_TOKENS = 17;
static const int VIT_FUSED_EMBED_DIM = 64;
static const int VIT_FUSED_HEADS = 4;
static const int VIT_FUSED_HEAD_DIM = 16;
static const int VIT_FUSED_QKV_DIM = 192;
#ifndef VIT_FUSED_MLP_DIM
#define VIT_FUSED_MLP_DIM 128
#endif
static const int VIT_FUSED_GELU_LUT_ENTRIES = 256;

#ifndef VIT_FUSED_MAC_UNROLL
#define VIT_FUSED_MAC_UNROLL 8
#endif
static const int VIT_FUSED_MAC_UNROLL_CONST = VIT_FUSED_MAC_UNROLL;

typedef float fused_float_t;
typedef int8_t fused_i8_t;
typedef int32_t fused_i32_t;

extern "C" {
void vit_transformer_layer_fused(
    const fused_float_t token_in[VIT_FUSED_TOKENS][VIT_FUSED_EMBED_DIM],
    const fused_float_t norm1_w[VIT_FUSED_EMBED_DIM],
    const fused_float_t norm1_b[VIT_FUSED_EMBED_DIM],
    const fused_float_t qkv_w[VIT_FUSED_EMBED_DIM][VIT_FUSED_QKV_DIM],
    const fused_float_t qkv_b[VIT_FUSED_QKV_DIM],
    const fused_float_t attn_proj_w[VIT_FUSED_EMBED_DIM][VIT_FUSED_EMBED_DIM],
    const fused_float_t attn_proj_b[VIT_FUSED_EMBED_DIM],
    const fused_float_t norm2_w[VIT_FUSED_EMBED_DIM],
    const fused_float_t norm2_b[VIT_FUSED_EMBED_DIM],
    const fused_float_t fc1_w[VIT_FUSED_MLP_DIM][VIT_FUSED_EMBED_DIM],
    const fused_float_t fc1_b[VIT_FUSED_MLP_DIM],
    const fused_float_t gelu_lut[VIT_FUSED_GELU_LUT_ENTRIES],
    const fused_float_t fc2_w[VIT_FUSED_EMBED_DIM][VIT_FUSED_MLP_DIM],
    const fused_float_t fc2_b[VIT_FUSED_EMBED_DIM],
    fused_float_t token_out[VIT_FUSED_TOKENS][VIT_FUSED_EMBED_DIM],
    fused_float_t qkv_w_scale,
    fused_float_t attn_proj_w_scale,
    fused_float_t fc1_w_scale,
    fused_float_t fc2_w_scale,
    fused_float_t hidden_inv_scale,
    fused_float_t lut_min,
    fused_float_t lut_index_scale);
}

#endif
