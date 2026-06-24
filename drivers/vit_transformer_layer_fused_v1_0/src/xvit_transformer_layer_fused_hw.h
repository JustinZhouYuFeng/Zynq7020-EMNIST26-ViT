// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.1 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - enable ap_done interrupt (Read/Write)
//        bit 1  - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - ap_done (COR/TOW)
//        bit 1  - ap_ready (COR/TOW)
//        others - reserved
// 0x10 : Data signal of token_in
//        bit 31~0 - token_in[31:0] (Read/Write)
// 0x14 : Data signal of token_in
//        bit 31~0 - token_in[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of norm1_w
//        bit 31~0 - norm1_w[31:0] (Read/Write)
// 0x20 : Data signal of norm1_w
//        bit 31~0 - norm1_w[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of norm1_b
//        bit 31~0 - norm1_b[31:0] (Read/Write)
// 0x2c : Data signal of norm1_b
//        bit 31~0 - norm1_b[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of qkv_w
//        bit 31~0 - qkv_w[31:0] (Read/Write)
// 0x38 : Data signal of qkv_w
//        bit 31~0 - qkv_w[63:32] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of qkv_b
//        bit 31~0 - qkv_b[31:0] (Read/Write)
// 0x44 : Data signal of qkv_b
//        bit 31~0 - qkv_b[63:32] (Read/Write)
// 0x48 : reserved
// 0x4c : Data signal of attn_proj_w
//        bit 31~0 - attn_proj_w[31:0] (Read/Write)
// 0x50 : Data signal of attn_proj_w
//        bit 31~0 - attn_proj_w[63:32] (Read/Write)
// 0x54 : reserved
// 0x58 : Data signal of attn_proj_b
//        bit 31~0 - attn_proj_b[31:0] (Read/Write)
// 0x5c : Data signal of attn_proj_b
//        bit 31~0 - attn_proj_b[63:32] (Read/Write)
// 0x60 : reserved
// 0x64 : Data signal of norm2_w
//        bit 31~0 - norm2_w[31:0] (Read/Write)
// 0x68 : Data signal of norm2_w
//        bit 31~0 - norm2_w[63:32] (Read/Write)
// 0x6c : reserved
// 0x70 : Data signal of norm2_b
//        bit 31~0 - norm2_b[31:0] (Read/Write)
// 0x74 : Data signal of norm2_b
//        bit 31~0 - norm2_b[63:32] (Read/Write)
// 0x78 : reserved
// 0x7c : Data signal of fc1_w
//        bit 31~0 - fc1_w[31:0] (Read/Write)
// 0x80 : Data signal of fc1_w
//        bit 31~0 - fc1_w[63:32] (Read/Write)
// 0x84 : reserved
// 0x88 : Data signal of fc1_b
//        bit 31~0 - fc1_b[31:0] (Read/Write)
// 0x8c : Data signal of fc1_b
//        bit 31~0 - fc1_b[63:32] (Read/Write)
// 0x90 : reserved
// 0x94 : Data signal of gelu_lut
//        bit 31~0 - gelu_lut[31:0] (Read/Write)
// 0x98 : Data signal of gelu_lut
//        bit 31~0 - gelu_lut[63:32] (Read/Write)
// 0x9c : reserved
// 0xa0 : Data signal of fc2_w
//        bit 31~0 - fc2_w[31:0] (Read/Write)
// 0xa4 : Data signal of fc2_w
//        bit 31~0 - fc2_w[63:32] (Read/Write)
// 0xa8 : reserved
// 0xac : Data signal of fc2_b
//        bit 31~0 - fc2_b[31:0] (Read/Write)
// 0xb0 : Data signal of fc2_b
//        bit 31~0 - fc2_b[63:32] (Read/Write)
// 0xb4 : reserved
// 0xb8 : Data signal of token_out
//        bit 31~0 - token_out[31:0] (Read/Write)
// 0xbc : Data signal of token_out
//        bit 31~0 - token_out[63:32] (Read/Write)
// 0xc0 : reserved
// 0xc4 : Data signal of qkv_w_scale
//        bit 31~0 - qkv_w_scale[31:0] (Read/Write)
// 0xc8 : reserved
// 0xcc : Data signal of attn_proj_w_scale
//        bit 31~0 - attn_proj_w_scale[31:0] (Read/Write)
// 0xd0 : reserved
// 0xd4 : Data signal of fc1_w_scale
//        bit 31~0 - fc1_w_scale[31:0] (Read/Write)
// 0xd8 : reserved
// 0xdc : Data signal of fc2_w_scale
//        bit 31~0 - fc2_w_scale[31:0] (Read/Write)
// 0xe0 : reserved
// 0xe4 : Data signal of hidden_inv_scale
//        bit 31~0 - hidden_inv_scale[31:0] (Read/Write)
// 0xe8 : reserved
// 0xec : Data signal of lut_min
//        bit 31~0 - lut_min[31:0] (Read/Write)
// 0xf0 : reserved
// 0xf4 : Data signal of lut_index_scale
//        bit 31~0 - lut_index_scale[31:0] (Read/Write)
// 0xf8 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_AP_CTRL                0x00
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_GIE                    0x04
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_IER                    0x08
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ISR                    0x0c
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_TOKEN_IN_DATA          0x10
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_TOKEN_IN_DATA          64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM1_W_DATA           0x1c
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_NORM1_W_DATA           64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM1_B_DATA           0x28
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_NORM1_B_DATA           64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_W_DATA             0x34
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_QKV_W_DATA             64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_B_DATA             0x40
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_QKV_B_DATA             64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_W_DATA       0x4c
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_ATTN_PROJ_W_DATA       64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_B_DATA       0x58
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_ATTN_PROJ_B_DATA       64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM2_W_DATA           0x64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_NORM2_W_DATA           64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM2_B_DATA           0x70
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_NORM2_B_DATA           64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_W_DATA             0x7c
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_FC1_W_DATA             64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_B_DATA             0x88
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_FC1_B_DATA             64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_GELU_LUT_DATA          0x94
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_GELU_LUT_DATA          64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_W_DATA             0xa0
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_FC2_W_DATA             64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_B_DATA             0xac
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_FC2_B_DATA             64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_TOKEN_OUT_DATA         0xb8
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_TOKEN_OUT_DATA         64
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_W_SCALE_DATA       0xc4
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_QKV_W_SCALE_DATA       32
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_W_SCALE_DATA 0xcc
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_ATTN_PROJ_W_SCALE_DATA 32
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_W_SCALE_DATA       0xd4
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_FC1_W_SCALE_DATA       32
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_W_SCALE_DATA       0xdc
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_FC2_W_SCALE_DATA       32
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_HIDDEN_INV_SCALE_DATA  0xe4
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_HIDDEN_INV_SCALE_DATA  32
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_LUT_MIN_DATA           0xec
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_LUT_MIN_DATA           32
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_LUT_INDEX_SCALE_DATA   0xf4
#define XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_BITS_LUT_INDEX_SCALE_DATA   32

