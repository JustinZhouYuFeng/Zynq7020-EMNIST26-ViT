#ifndef VIT_TRANSFORMER_LAYER_FUSED_H
#define VIT_TRANSFORMER_LAYER_FUSED_H

#include <stdint.h>

// ------------------------------------------------------------------
// Vision Transformer (ViT) 模型架构超参数定义
// ------------------------------------------------------------------

// 序列长度 (Tokens数量)。
// 通常等于图像分块数量 (Patch数) + 1 个类别标记 (Class Token)。
// 例如：图像被分为 4x4=16 个 Patch，加上 1 个 CLS Token，总共 17 个 Token。
static const int VIT_FUSED_TOKENS = 17;

// 嵌入维度 (Embedding Dimension / D_model)。
// 每个 Token 特征向量的长度。
static const int VIT_FUSED_EMBED_DIM = 64;

// 多头注意力机制中的头数 (Number of Attention Heads)。
static const int VIT_FUSED_HEADS = 4;

// 每个注意力头的维度 (Head Dimension)。
// 通常满足: EMBED_DIM = HEADS * HEAD_DIM (此处 64 = 4 * 16)。
static const int VIT_FUSED_HEAD_DIM = 16;

// Q, K, V 矩阵合并后的总维度 (QKV Dimension)。
// 由于 Q, K, V 的维度通常与 EMBED_DIM 相同，因此合并后为 3 * EMBED_DIM (此处 3 * 64 = 192)。
static const int VIT_FUSED_QKV_DIM = 192;

// 前馈神经网络 (MLP/FFN) 隐藏层的维度。
// 允许通过编译宏从外部重定义，默认设置为 128 (通常是 EMBED_DIM 的 2 倍或 4 倍)。
#ifndef VIT_FUSED_MLP_DIM
#define VIT_FUSED_MLP_DIM 128
#endif

// GELU 激活函数硬件查找表 (Look-Up Table, LUT) 的条目数量。
// 在硬件中计算非线性激活函数成本较高，通常将其离散化存储为 256 个点的内存数组。
static const int VIT_FUSED_GELU_LUT_ENTRIES = 256;

// ------------------------------------------------------------------
// 硬件综合 (HLS) 优化参数
// ------------------------------------------------------------------

// 乘加运算 (MAC) 的循环展开因子。
// 决定了硬件底层并行计算树的宽度（例如：在一个时钟周期内并行执行 8 次乘法和加法）。
// 必须与 .cpp 文件中 reduce_mac_fused 函数支持的树状规约层级（4, 8 或 16）相匹配。
#ifndef VIT_FUSED_MAC_UNROLL
#define VIT_FUSED_MAC_UNROLL 8
#endif
static const int VIT_FUSED_MAC_UNROLL_CONST = VIT_FUSED_MAC_UNROLL;

// ------------------------------------------------------------------
// 数据类型定义
// ------------------------------------------------------------------
// 使用 typedef 隔离底层数据类型，便于在算法验证 (float) 和
// 硬件定点化部署 (如 ap_fixed 或 half) 之间无缝切换。
typedef float fused_float_t;
typedef int8_t fused_i8_t;
typedef int32_t fused_i32_t;

// ------------------------------------------------------------------
// 顶层硬件 IP 核 (Top-Level IP) 接口声明
// ------------------------------------------------------------------
extern "C" {
void vit_transformer_layer_fused(
    // --- 输入/输出数据流 ---
    const fused_float_t token_in[VIT_FUSED_TOKENS][VIT_FUSED_EMBED_DIM], // 输入特征张量 [17, 64]

    // --- Layer Norm 1 权重 ---
    const fused_float_t norm1_w[VIT_FUSED_EMBED_DIM],                    // Gamma 参数
    const fused_float_t norm1_b[VIT_FUSED_EMBED_DIM],                    // Beta 参数

    // --- Multi-Head Attention 权重 ---
    const fused_float_t qkv_w[VIT_FUSED_EMBED_DIM][VIT_FUSED_QKV_DIM],   // QKV 融合权重矩阵
    const fused_float_t qkv_b[VIT_FUSED_QKV_DIM],                        // QKV 偏置
    const fused_float_t attn_proj_w[VIT_FUSED_EMBED_DIM][VIT_FUSED_EMBED_DIM], // Attention后的投影层权重
    const fused_float_t attn_proj_b[VIT_FUSED_EMBED_DIM],                // Attention后的投影层偏置

    // --- Layer Norm 2 权重 ---
    const fused_float_t norm2_w[VIT_FUSED_EMBED_DIM],                    // Gamma 参数
    const fused_float_t norm2_b[VIT_FUSED_EMBED_DIM],                    // Beta 参数

    // --- MLP 权重与查找表 ---
    const fused_float_t fc1_w[VIT_FUSED_MLP_DIM][VIT_FUSED_EMBED_DIM],   // MLP 第一层权重
    const fused_float_t fc1_b[VIT_FUSED_MLP_DIM],                        // MLP 第一层偏置
    const fused_float_t gelu_lut[VIT_FUSED_GELU_LUT_ENTRIES],            // GELU 硬件查找表
    const fused_float_t fc2_w[VIT_FUSED_EMBED_DIM][VIT_FUSED_MLP_DIM],   // MLP 第二层权重
    const fused_float_t fc2_b[VIT_FUSED_EMBED_DIM],                      // MLP 第二层偏置

    // --- 输出数据流 ---
    fused_float_t token_out[VIT_FUSED_TOKENS][VIT_FUSED_EMBED_DIM],      // 输出特征张量 [17, 64]

    // --- 动态量化与缩放系数 (用于支持混合精度或动态量化校准) ---
    fused_float_t qkv_w_scale,        // QKV权重量化缩放系数
    fused_float_t attn_proj_w_scale,  // Attention投影层量化缩放系数
    fused_float_t fc1_w_scale,        // FC1量化缩放系数
    fused_float_t fc2_w_scale,        // FC2量化缩放系数
    fused_float_t hidden_inv_scale,   // 隐藏层反量化系数

    // --- LUT 索引映射参数 ---
    fused_float_t lut_min,            // GELU查找表对应的最小输入值
    fused_float_t lut_index_scale);   // 实际输入值转换为LUT索引的缩放因子
}

#endif
