#include "vit_transformer_layer_fused.h"

#ifndef VIT_FUSED_DEBUG_STAGE
#define VIT_FUSED_DEBUG_STAGE 5
#endif

static fused_float_t abs_fused(fused_float_t x)
{
#pragma HLS INLINE
    return (x < (fused_float_t)0.0) ? -x : x;
}

static fused_float_t fast_rsqrt_fused(fused_float_t x)
{
#pragma HLS INLINE
    fused_float_t y;

    if (x > (fused_float_t)16.0) y = (fused_float_t)0.25;
    else if (x > (fused_float_t)9.0) y = (fused_float_t)0.33333334;
    else if (x > (fused_float_t)4.0) y = (fused_float_t)0.5;
    else if (x > (fused_float_t)2.25) y = (fused_float_t)0.66666669;
    else if (x > (fused_float_t)1.0) y = (fused_float_t)1.0;
    else if (x > (fused_float_t)0.5625) y = (fused_float_t)1.33333337;
    else if (x > (fused_float_t)0.25) y = (fused_float_t)1.33333337;
    else if (x > (fused_float_t)0.0625) y = (fused_float_t)2.0;
    else if (x > (fused_float_t)0.015625) y = (fused_float_t)4.0;
    else y = (fused_float_t)8.0;

    fused_float_t y2 = y * y;
    y = y * ((fused_float_t)1.5 - (fused_float_t)0.5 * x * y2);
    y2 = y * y;
    y = y * ((fused_float_t)1.5 - (fused_float_t)0.5 * x * y2);
    y2 = y * y;
    y = y * ((fused_float_t)1.5 - (fused_float_t)0.5 * x * y2);
    y2 = y * y;
    y = y * ((fused_float_t)1.5 - (fused_float_t)0.5 * x * y2);
    y2 = y * y;
    y = y * ((fused_float_t)1.5 - (fused_float_t)0.5 * x * y2);
    return y;
}

static fused_float_t fast_exp_fused(fused_float_t x)
{
    if (x < (fused_float_t)-10.0) {
        return (fused_float_t)0.0;
    }
    if (x > (fused_float_t)10.0) {
        x = (fused_float_t)10.0;
    }

    fused_float_t y = (fused_float_t)1.0 + x / (fused_float_t)256.0;
    for (int i = 0; i < 8; ++i) {
#pragma HLS PIPELINE II=1
        y *= y;
    }
    return y;
}

static fused_i8_t quantize_i8_fused(fused_float_t x, fused_float_t scale)
{
    fused_i32_t v;

    if (scale < (fused_float_t)1.0e-9) {
        scale = (fused_float_t)1.0e-9;
    }

    if (x >= (fused_float_t)0.0) {
        v = (fused_i32_t)(x / scale + (fused_float_t)0.5);
    } else {
        v = (fused_i32_t)(x / scale - (fused_float_t)0.5);
    }

    if (v > 127) {
        v = 127;
    }
    if (v < -128) {
        v = -128;
    }
    return (fused_i8_t)v;
}

static fused_i32_t round_i32_fused(fused_float_t x)
{
#pragma HLS INLINE
    if (x >= (fused_float_t)0.0) {
        return (fused_i32_t)(x + (fused_float_t)0.5);
    }
    return (fused_i32_t)(x - (fused_float_t)0.5);
}

static void layer_norm_fused(
    const fused_float_t in[VIT_FUSED_EMBED_DIM],
    const fused_float_t gamma[VIT_FUSED_EMBED_DIM],
    const fused_float_t beta[VIT_FUSED_EMBED_DIM],
    fused_float_t out[VIT_FUSED_EMBED_DIM])
{
    fused_float_t mean = (fused_float_t)0.0;
    fused_float_t var = (fused_float_t)0.0;

    for (int i = 0; i < VIT_FUSED_EMBED_DIM; ++i) {
#pragma HLS PIPELINE II=1
        mean += in[i];
    }
    mean *= (fused_float_t)(1.0 / VIT_FUSED_EMBED_DIM);

    for (int i = 0; i < VIT_FUSED_EMBED_DIM; ++i) {
#pragma HLS PIPELINE II=1
        fused_float_t d = in[i] - mean;
        var += d * d;
    }
    var *= (fused_float_t)(1.0 / VIT_FUSED_EMBED_DIM);

    fused_float_t inv = fast_rsqrt_fused(var + (fused_float_t)1.0e-5);
    for (int i = 0; i < VIT_FUSED_EMBED_DIM; ++i) {
#pragma HLS PIPELINE II=1
        out[i] = (in[i] - mean) * inv * gamma[i] + beta[i];
    }
}

static fused_float_t reduce_mac_fused(
    const fused_float_t partial[VIT_FUSED_MAC_UNROLL_CONST])
{
#pragma HLS INLINE
    fused_float_t pair[VIT_FUSED_MAC_UNROLL_CONST / 2];
#pragma HLS ARRAY_PARTITION variable=pair complete dim=1
    for (int i = 0; i < VIT_FUSED_MAC_UNROLL_CONST / 2; ++i) {
#pragma HLS UNROLL
        pair[i] = partial[2 * i] + partial[2 * i + 1];
    }

    fused_float_t quad[VIT_FUSED_MAC_UNROLL_CONST / 4];
#pragma HLS ARRAY_PARTITION variable=quad complete dim=1
    for (int i = 0; i < VIT_FUSED_MAC_UNROLL_CONST / 4; ++i) {
#pragma HLS UNROLL
        quad[i] = pair[2 * i] + pair[2 * i + 1];
    }
#if VIT_FUSED_MAC_UNROLL == 16
    return (quad[0] + quad[1]) + (quad[2] + quad[3]);
#elif VIT_FUSED_MAC_UNROLL == 8
    return quad[0] + quad[1];
#elif VIT_FUSED_MAC_UNROLL == 4
    return quad[0];
#else
#error "VIT_FUSED_MAC_UNROLL must be 4, 8, or 16"
#endif
}

static fused_float_t dot_qk_head_fused(
    fused_float_t qkv[VIT_FUSED_TOKENS][VIT_FUSED_QKV_DIM],
    int tq,
    int tk,
    int head)
{
#pragma HLS INLINE
    fused_float_t partial[VIT_FUSED_MAC_UNROLL_CONST];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
    for (int u = 0; u < VIT_FUSED_MAC_UNROLL_CONST; ++u) {
#pragma HLS UNROLL
        partial[u] = (fused_float_t)0.0;
    }

    for (int base = 0; base < VIT_FUSED_HEAD_DIM; base += VIT_FUSED_MAC_UNROLL_CONST) {
#pragma HLS PIPELINE II=1
        for (int u = 0; u < VIT_FUSED_MAC_UNROLL_CONST; ++u) {
#pragma HLS UNROLL
            int idx = head * VIT_FUSED_HEAD_DIM + base + u;
            partial[u] += qkv[tq][idx] * qkv[tk][VIT_FUSED_EMBED_DIM + idx];
        }
    }
    return reduce_mac_fused(partial);
}

static fused_float_t dot_attn_value_fused(
    const fused_float_t attn_row[VIT_FUSED_TOKENS],
    fused_float_t qkv[VIT_FUSED_TOKENS][VIT_FUSED_QKV_DIM],
    int value_idx)
{
#pragma HLS INLINE
    fused_float_t partial[VIT_FUSED_MAC_UNROLL_CONST];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
    for (int u = 0; u < VIT_FUSED_MAC_UNROLL_CONST; ++u) {
#pragma HLS UNROLL
        partial[u] = (fused_float_t)0.0;
    }

    for (int base = 0; base < VIT_FUSED_TOKENS; base += VIT_FUSED_MAC_UNROLL_CONST) {
#pragma HLS PIPELINE II=1
        for (int u = 0; u < VIT_FUSED_MAC_UNROLL_CONST; ++u) {
#pragma HLS UNROLL
            int tk = base + u;
            if (tk < VIT_FUSED_TOKENS) {
                partial[u] += attn_row[tk] * qkv[tk][2 * VIT_FUSED_EMBED_DIM + value_idx];
            }
        }
    }
    return reduce_mac_fused(partial);
}

static fused_float_t dot_attn_proj_fused(
    const fused_float_t attn_row[VIT_FUSED_EMBED_DIM],
    const fused_float_t weight_row[VIT_FUSED_EMBED_DIM])
{
#pragma HLS INLINE
    fused_float_t partial[VIT_FUSED_MAC_UNROLL_CONST];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
    for (int u = 0; u < VIT_FUSED_MAC_UNROLL_CONST; ++u) {
#pragma HLS UNROLL
        partial[u] = (fused_float_t)0.0;
    }

    for (int base = 0; base < VIT_FUSED_EMBED_DIM; base += VIT_FUSED_MAC_UNROLL_CONST) {
#pragma HLS PIPELINE II=1
        for (int u = 0; u < VIT_FUSED_MAC_UNROLL_CONST; ++u) {
#pragma HLS UNROLL
            int i = base + u;
            partial[u] += attn_row[i] * weight_row[i];
        }
    }
    return reduce_mac_fused(partial);
}

static fused_float_t dot_fc1_fused(
    const fused_float_t norm_row[VIT_FUSED_EMBED_DIM],
    const fused_float_t weight_row[VIT_FUSED_EMBED_DIM])
{
#pragma HLS INLINE
    fused_float_t partial[VIT_FUSED_MAC_UNROLL_CONST];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
    for (int u = 0; u < VIT_FUSED_MAC_UNROLL_CONST; ++u) {
#pragma HLS UNROLL
        partial[u] = (fused_float_t)0.0;
    }

    for (int base = 0; base < VIT_FUSED_EMBED_DIM; base += VIT_FUSED_MAC_UNROLL_CONST) {
#pragma HLS PIPELINE II=1
        for (int u = 0; u < VIT_FUSED_MAC_UNROLL_CONST; ++u) {
#pragma HLS UNROLL
            int e = base + u;
            partial[u] += norm_row[e] * weight_row[e];
        }
    }
    return reduce_mac_fused(partial);
}

static fused_float_t dot_fc2_fused(
    const fused_float_t hidden_row[VIT_FUSED_MLP_DIM],
    const fused_float_t weight_row[VIT_FUSED_MLP_DIM])
{
#pragma HLS INLINE
    fused_float_t partial[VIT_FUSED_MAC_UNROLL_CONST];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
    for (int u = 0; u < VIT_FUSED_MAC_UNROLL_CONST; ++u) {
#pragma HLS UNROLL
        partial[u] = (fused_float_t)0.0;
    }

    for (int base = 0; base < VIT_FUSED_MLP_DIM; base += VIT_FUSED_MAC_UNROLL_CONST) {
#pragma HLS PIPELINE II=1
        for (int u = 0; u < VIT_FUSED_MAC_UNROLL_CONST; ++u) {
#pragma HLS UNROLL
            int h = base + u;
            partial[u] += hidden_row[h] * weight_row[h];
        }
    }
    return reduce_mac_fused(partial);
}
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
    fused_float_t lut_index_scale)
{
#pragma HLS INTERFACE m_axi port=token_in offset=slave bundle=gmem_token depth=1088
#pragma HLS INTERFACE m_axi port=norm1_w offset=slave bundle=gmem_param depth=64
#pragma HLS INTERFACE m_axi port=norm1_b offset=slave bundle=gmem_param depth=64
#pragma HLS INTERFACE m_axi port=qkv_w offset=slave bundle=gmem_qkvw depth=12288
#pragma HLS INTERFACE m_axi port=qkv_b offset=slave bundle=gmem_param depth=192
#pragma HLS INTERFACE m_axi port=attn_proj_w offset=slave bundle=gmem_attnw depth=4096
#pragma HLS INTERFACE m_axi port=attn_proj_b offset=slave bundle=gmem_param depth=64
#pragma HLS INTERFACE m_axi port=norm2_w offset=slave bundle=gmem_param depth=64
#pragma HLS INTERFACE m_axi port=norm2_b offset=slave bundle=gmem_param depth=64
#if VIT_FUSED_MLP_DIM == 256
#pragma HLS INTERFACE m_axi port=fc1_w offset=slave bundle=gmem_mlpw depth=16384
#pragma HLS INTERFACE m_axi port=fc1_b offset=slave bundle=gmem_param depth=256
#pragma HLS INTERFACE m_axi port=fc2_w offset=slave bundle=gmem_mlpw depth=16384
#else
#pragma HLS INTERFACE m_axi port=fc1_w offset=slave bundle=gmem_mlpw depth=8192
#pragma HLS INTERFACE m_axi port=fc1_b offset=slave bundle=gmem_param depth=128
#pragma HLS INTERFACE m_axi port=fc2_w offset=slave bundle=gmem_mlpw depth=8192
#endif
#pragma HLS INTERFACE m_axi port=gelu_lut offset=slave bundle=gmem_param depth=256
#pragma HLS INTERFACE m_axi port=fc2_b offset=slave bundle=gmem_param depth=64
#pragma HLS INTERFACE m_axi port=token_out offset=slave bundle=gmem_out depth=1088
#pragma HLS INTERFACE s_axilite port=token_in bundle=control
#pragma HLS INTERFACE s_axilite port=norm1_w bundle=control
#pragma HLS INTERFACE s_axilite port=norm1_b bundle=control
#pragma HLS INTERFACE s_axilite port=qkv_w bundle=control
#pragma HLS INTERFACE s_axilite port=qkv_b bundle=control
#pragma HLS INTERFACE s_axilite port=attn_proj_w bundle=control
#pragma HLS INTERFACE s_axilite port=attn_proj_b bundle=control
#pragma HLS INTERFACE s_axilite port=norm2_w bundle=control
#pragma HLS INTERFACE s_axilite port=norm2_b bundle=control
#pragma HLS INTERFACE s_axilite port=fc1_w bundle=control
#pragma HLS INTERFACE s_axilite port=fc1_b bundle=control
#pragma HLS INTERFACE s_axilite port=gelu_lut bundle=control
#pragma HLS INTERFACE s_axilite port=fc2_w bundle=control
#pragma HLS INTERFACE s_axilite port=fc2_b bundle=control
#pragma HLS INTERFACE s_axilite port=token_out bundle=control
#pragma HLS INTERFACE s_axilite port=qkv_w_scale bundle=control
#pragma HLS INTERFACE s_axilite port=attn_proj_w_scale bundle=control
#pragma HLS INTERFACE s_axilite port=fc1_w_scale bundle=control
#pragma HLS INTERFACE s_axilite port=fc2_w_scale bundle=control
#pragma HLS INTERFACE s_axilite port=hidden_inv_scale bundle=control
#pragma HLS INTERFACE s_axilite port=lut_min bundle=control
#pragma HLS INTERFACE s_axilite port=lut_index_scale bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    fused_float_t token[VIT_FUSED_TOKENS][VIT_FUSED_EMBED_DIM];
    fused_float_t normed[VIT_FUSED_TOKENS][VIT_FUSED_EMBED_DIM];
    fused_float_t qkv[VIT_FUSED_TOKENS][VIT_FUSED_QKV_DIM];
    fused_float_t attn_out[VIT_FUSED_TOKENS][VIT_FUSED_EMBED_DIM];
    fused_float_t attn_prob[VIT_FUSED_TOKENS][VIT_FUSED_TOKENS];
    fused_float_t mlp_hidden[VIT_FUSED_TOKENS][VIT_FUSED_MLP_DIM];
    fused_float_t qkv_weight_cache[VIT_FUSED_EMBED_DIM][VIT_FUSED_QKV_DIM];
    fused_float_t proj_weight_row[VIT_FUSED_EMBED_DIM];
    fused_float_t fc1_weight_row[VIT_FUSED_EMBED_DIM];
    fused_float_t fc2_weight_row[VIT_FUSED_MLP_DIM];
#pragma HLS RESOURCE variable=token core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=normed core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=qkv core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=attn_out core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=attn_prob core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=mlp_hidden core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=qkv_weight_cache core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=qkv_weight_cache cyclic factor=VIT_FUSED_MAC_UNROLL_CONST dim=2
#pragma HLS ARRAY_PARTITION variable=proj_weight_row cyclic factor=VIT_FUSED_MAC_UNROLL_CONST dim=1
#pragma HLS ARRAY_PARTITION variable=fc1_weight_row cyclic factor=VIT_FUSED_MAC_UNROLL_CONST dim=1
#pragma HLS ARRAY_PARTITION variable=fc2_weight_row cyclic factor=VIT_FUSED_MAC_UNROLL_CONST dim=1
#pragma HLS ARRAY_PARTITION variable=token cyclic factor=VIT_FUSED_MAC_UNROLL_CONST dim=2
#pragma HLS ARRAY_PARTITION variable=normed cyclic factor=VIT_FUSED_MAC_UNROLL_CONST dim=2
#pragma HLS ARRAY_PARTITION variable=qkv cyclic factor=VIT_FUSED_MAC_UNROLL_CONST dim=2
#pragma HLS ARRAY_PARTITION variable=attn_out cyclic factor=VIT_FUSED_MAC_UNROLL_CONST dim=2
#pragma HLS ARRAY_PARTITION variable=mlp_hidden cyclic factor=VIT_FUSED_MAC_UNROLL_CONST dim=2

    for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
        for (int e = 0; e < VIT_FUSED_EMBED_DIM; ++e) {
#pragma HLS PIPELINE II=1
            token[t][e] = token_in[t][e];
            attn_out[t][e] = (fused_float_t)0.0;
        }
    }

    for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
        layer_norm_fused(token[t], norm1_w, norm1_b, normed[t]);
    }

#if VIT_FUSED_DEBUG_STAGE == 1
    {
        for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
            for (int e = 0; e < VIT_FUSED_EMBED_DIM; ++e) {
#pragma HLS PIPELINE II=1
                token_out[t][e] = normed[t][e];
            }
        }
        return;
    }
#endif

    for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
        for (int o = 0; o < VIT_FUSED_QKV_DIM; ++o) {
#pragma HLS PIPELINE II=1
            qkv[t][o] = qkv_b[o];
        }
    }

    for (int i = 0; i < VIT_FUSED_EMBED_DIM; ++i) {
        for (int o = 0; o < VIT_FUSED_QKV_DIM; ++o) {
#pragma HLS PIPELINE II=1
            qkv_weight_cache[i][o] = qkv_w[i][o];
        }
    }

    for (int i = 0; i < VIT_FUSED_EMBED_DIM; ++i) {
        for (int base = 0; base < VIT_FUSED_QKV_DIM; base += VIT_FUSED_MAC_UNROLL_CONST) {
            for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
#pragma HLS PIPELINE II=1
                fused_float_t x = normed[t][i];
                for (int u = 0; u < VIT_FUSED_MAC_UNROLL_CONST; ++u) {
#pragma HLS UNROLL
                    int o = base + u;
                    qkv[t][o] += x * qkv_weight_cache[i][o];
                }
            }
        }
    }

#if VIT_FUSED_DEBUG_STAGE == 2
    {
        for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
            for (int e = 0; e < VIT_FUSED_EMBED_DIM; ++e) {
#pragma HLS PIPELINE II=1
                token_out[t][e] = qkv[t][e];
            }
        }
        return;
    }
#endif

    for (int h = 0; h < VIT_FUSED_HEADS; ++h) {
        fused_float_t p_max = (fused_float_t)0.0;

        for (int tq = 0; tq < VIT_FUSED_TOKENS; ++tq) {
            fused_float_t max_score = (fused_float_t)-3.4e38;
            fused_float_t sum = (fused_float_t)0.0;

            for (int tk = 0; tk < VIT_FUSED_TOKENS; ++tk) {
                fused_float_t acc = dot_qk_head_fused(qkv, tq, tk, h);
                fused_float_t score = acc * (fused_float_t)0.25;
                attn_prob[tq][tk] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            for (int tk = 0; tk < VIT_FUSED_TOKENS; ++tk) {
#pragma HLS PIPELINE II=1
                fused_float_t prob = fast_exp_fused(attn_prob[tq][tk] - max_score);
                attn_prob[tq][tk] = prob;
                sum += prob;
            }
            for (int tk = 0; tk < VIT_FUSED_TOKENS; ++tk) {
#pragma HLS PIPELINE II=1
                fused_float_t prob = attn_prob[tq][tk] / sum;
                attn_prob[tq][tk] = prob;
                if (prob > p_max) {
                    p_max = prob;
                }
            }
        }

        for (int tq = 0; tq < VIT_FUSED_TOKENS; ++tq) {
            for (int d = 0; d < VIT_FUSED_HEAD_DIM; ++d) {
                int idx = h * VIT_FUSED_HEAD_DIM + d;
                attn_out[tq][idx] = dot_attn_value_fused(attn_prob[tq], qkv, idx);
            }
        }
    }

#if VIT_FUSED_DEBUG_STAGE == 3
    {
        for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
            for (int e = 0; e < VIT_FUSED_EMBED_DIM; ++e) {
#pragma HLS PIPELINE II=1
                token_out[t][e] = attn_out[t][e];
            }
        }
        return;
    }
#endif

    for (int o = 0; o < VIT_FUSED_EMBED_DIM; ++o) {
        for (int i = 0; i < VIT_FUSED_EMBED_DIM; ++i) {
#pragma HLS PIPELINE II=1
            proj_weight_row[i] = attn_proj_w[o][i];
        }
        for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
            fused_float_t acc = attn_proj_b[o] + dot_attn_proj_fused(attn_out[t], proj_weight_row);
            token[t][o] += acc;
        }
    }

#if VIT_FUSED_DEBUG_STAGE == 4
    {
        for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
            for (int e = 0; e < VIT_FUSED_EMBED_DIM; ++e) {
#pragma HLS PIPELINE II=1
                token_out[t][e] = token[t][e];
            }
        }
        return;
    }
#endif

    for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
        layer_norm_fused(token[t], norm2_w, norm2_b, normed[t]);
    }

    for (int h = 0; h < VIT_FUSED_MLP_DIM; ++h) {
        for (int e = 0; e < VIT_FUSED_EMBED_DIM; ++e) {
#pragma HLS PIPELINE II=1
            fc1_weight_row[e] = fc1_w[h][e];
        }
        for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
            fused_float_t acc = fc1_b[h] + dot_fc1_fused(normed[t], fc1_weight_row);

            fused_float_t scaled = (acc - lut_min) * lut_index_scale;
            int idx;
            if (scaled <= (fused_float_t)0.0) {
                idx = 0;
            } else if (scaled >= (fused_float_t)(VIT_FUSED_GELU_LUT_ENTRIES - 1)) {
                idx = VIT_FUSED_GELU_LUT_ENTRIES - 1;
            } else {
                idx = (int)(scaled + (fused_float_t)0.5);
            }
            mlp_hidden[t][h] = gelu_lut[idx];
        }
    }

    for (int e = 0; e < VIT_FUSED_EMBED_DIM; ++e) {
        for (int h = 0; h < VIT_FUSED_MLP_DIM; ++h) {
#pragma HLS PIPELINE II=1
            fc2_weight_row[h] = fc2_w[e][h];
        }
        for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
            fused_float_t acc = fc2_b[e] + dot_fc2_fused(mlp_hidden[t], fc2_weight_row);
            token[t][e] += acc;
        }
    }

    for (int t = 0; t < VIT_FUSED_TOKENS; ++t) {
        for (int e = 0; e < VIT_FUSED_EMBED_DIM; ++e) {
#pragma HLS PIPELINE II=1
            token_out[t][e] = token[t][e];
        }
    }
}
}
