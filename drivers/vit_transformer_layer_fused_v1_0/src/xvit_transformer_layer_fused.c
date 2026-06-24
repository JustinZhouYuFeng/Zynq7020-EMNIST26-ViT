// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.1 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xvit_transformer_layer_fused.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XVit_transformer_layer_fused_CfgInitialize(XVit_transformer_layer_fused *InstancePtr, XVit_transformer_layer_fused_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XVit_transformer_layer_fused_Start(XVit_transformer_layer_fused *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_AP_CTRL) & 0x80;
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XVit_transformer_layer_fused_IsDone(XVit_transformer_layer_fused *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XVit_transformer_layer_fused_IsIdle(XVit_transformer_layer_fused *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XVit_transformer_layer_fused_IsReady(XVit_transformer_layer_fused *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XVit_transformer_layer_fused_EnableAutoRestart(XVit_transformer_layer_fused *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XVit_transformer_layer_fused_DisableAutoRestart(XVit_transformer_layer_fused *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_AP_CTRL, 0);
}

void XVit_transformer_layer_fused_Set_token_in(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_TOKEN_IN_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_TOKEN_IN_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_token_in(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_TOKEN_IN_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_TOKEN_IN_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_norm1_w(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM1_W_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM1_W_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_norm1_w(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM1_W_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM1_W_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_norm1_b(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM1_B_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM1_B_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_norm1_b(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM1_B_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM1_B_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_qkv_w(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_W_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_W_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_qkv_w(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_W_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_W_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_qkv_b(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_B_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_B_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_qkv_b(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_B_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_B_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_attn_proj_w(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_W_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_W_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_attn_proj_w(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_W_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_W_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_attn_proj_b(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_B_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_B_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_attn_proj_b(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_B_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_B_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_norm2_w(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM2_W_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM2_W_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_norm2_w(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM2_W_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM2_W_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_norm2_b(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM2_B_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM2_B_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_norm2_b(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM2_B_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_NORM2_B_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_fc1_w(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_W_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_W_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_fc1_w(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_W_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_W_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_fc1_b(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_B_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_B_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_fc1_b(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_B_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_B_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_gelu_lut(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_GELU_LUT_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_GELU_LUT_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_gelu_lut(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_GELU_LUT_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_GELU_LUT_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_fc2_w(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_W_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_W_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_fc2_w(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_W_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_W_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_fc2_b(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_B_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_B_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_fc2_b(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_B_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_B_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_token_out(XVit_transformer_layer_fused *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_TOKEN_OUT_DATA, (u32)(Data));
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_TOKEN_OUT_DATA + 4, (u32)(Data >> 32));
}

u64 XVit_transformer_layer_fused_Get_token_out(XVit_transformer_layer_fused *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_TOKEN_OUT_DATA);
    Data += (u64)XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_TOKEN_OUT_DATA + 4) << 32;
    return Data;
}

void XVit_transformer_layer_fused_Set_qkv_w_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_W_SCALE_DATA, Data);
}

u32 XVit_transformer_layer_fused_Get_qkv_w_scale(XVit_transformer_layer_fused *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_QKV_W_SCALE_DATA);
    return Data;
}

void XVit_transformer_layer_fused_Set_attn_proj_w_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_W_SCALE_DATA, Data);
}

u32 XVit_transformer_layer_fused_Get_attn_proj_w_scale(XVit_transformer_layer_fused *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ATTN_PROJ_W_SCALE_DATA);
    return Data;
}

void XVit_transformer_layer_fused_Set_fc1_w_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_W_SCALE_DATA, Data);
}

u32 XVit_transformer_layer_fused_Get_fc1_w_scale(XVit_transformer_layer_fused *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC1_W_SCALE_DATA);
    return Data;
}

void XVit_transformer_layer_fused_Set_fc2_w_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_W_SCALE_DATA, Data);
}

u32 XVit_transformer_layer_fused_Get_fc2_w_scale(XVit_transformer_layer_fused *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_FC2_W_SCALE_DATA);
    return Data;
}

void XVit_transformer_layer_fused_Set_hidden_inv_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_HIDDEN_INV_SCALE_DATA, Data);
}

u32 XVit_transformer_layer_fused_Get_hidden_inv_scale(XVit_transformer_layer_fused *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_HIDDEN_INV_SCALE_DATA);
    return Data;
}

void XVit_transformer_layer_fused_Set_lut_min(XVit_transformer_layer_fused *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_LUT_MIN_DATA, Data);
}

u32 XVit_transformer_layer_fused_Get_lut_min(XVit_transformer_layer_fused *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_LUT_MIN_DATA);
    return Data;
}

void XVit_transformer_layer_fused_Set_lut_index_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_LUT_INDEX_SCALE_DATA, Data);
}

u32 XVit_transformer_layer_fused_Get_lut_index_scale(XVit_transformer_layer_fused *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_LUT_INDEX_SCALE_DATA);
    return Data;
}

void XVit_transformer_layer_fused_InterruptGlobalEnable(XVit_transformer_layer_fused *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_GIE, 1);
}

void XVit_transformer_layer_fused_InterruptGlobalDisable(XVit_transformer_layer_fused *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_GIE, 0);
}

void XVit_transformer_layer_fused_InterruptEnable(XVit_transformer_layer_fused *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_IER);
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_IER, Register | Mask);
}

void XVit_transformer_layer_fused_InterruptDisable(XVit_transformer_layer_fused *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_IER);
    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_IER, Register & (~Mask));
}

void XVit_transformer_layer_fused_InterruptClear(XVit_transformer_layer_fused *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XVit_transformer_layer_fused_WriteReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ISR, Mask);
}

u32 XVit_transformer_layer_fused_InterruptGetEnabled(XVit_transformer_layer_fused *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_IER);
}

u32 XVit_transformer_layer_fused_InterruptGetStatus(XVit_transformer_layer_fused *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XVit_transformer_layer_fused_ReadReg(InstancePtr->Control_BaseAddress, XVIT_TRANSFORMER_LAYER_FUSED_CONTROL_ADDR_ISR);
}

