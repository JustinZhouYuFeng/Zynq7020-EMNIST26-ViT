// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.1 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XVIT_TRANSFORMER_LAYER_FUSED_H
#define XVIT_TRANSFORMER_LAYER_FUSED_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xvit_transformer_layer_fused_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u32 Control_BaseAddress;
} XVit_transformer_layer_fused_Config;
#endif

typedef struct {
    u32 Control_BaseAddress;
    u32 IsReady;
} XVit_transformer_layer_fused;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XVit_transformer_layer_fused_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XVit_transformer_layer_fused_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XVit_transformer_layer_fused_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XVit_transformer_layer_fused_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XVit_transformer_layer_fused_Initialize(XVit_transformer_layer_fused *InstancePtr, u16 DeviceId);
XVit_transformer_layer_fused_Config* XVit_transformer_layer_fused_LookupConfig(u16 DeviceId);
int XVit_transformer_layer_fused_CfgInitialize(XVit_transformer_layer_fused *InstancePtr, XVit_transformer_layer_fused_Config *ConfigPtr);
#else
int XVit_transformer_layer_fused_Initialize(XVit_transformer_layer_fused *InstancePtr, const char* InstanceName);
int XVit_transformer_layer_fused_Release(XVit_transformer_layer_fused *InstancePtr);
#endif

void XVit_transformer_layer_fused_Start(XVit_transformer_layer_fused *InstancePtr);
u32 XVit_transformer_layer_fused_IsDone(XVit_transformer_layer_fused *InstancePtr);
u32 XVit_transformer_layer_fused_IsIdle(XVit_transformer_layer_fused *InstancePtr);
u32 XVit_transformer_layer_fused_IsReady(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_EnableAutoRestart(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_DisableAutoRestart(XVit_transformer_layer_fused *InstancePtr);

void XVit_transformer_layer_fused_Set_token_in(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_token_in(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_norm1_w(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_norm1_w(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_norm1_b(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_norm1_b(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_qkv_w(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_qkv_w(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_qkv_b(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_qkv_b(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_attn_proj_w(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_attn_proj_w(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_attn_proj_b(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_attn_proj_b(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_norm2_w(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_norm2_w(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_norm2_b(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_norm2_b(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_fc1_w(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_fc1_w(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_fc1_b(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_fc1_b(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_gelu_lut(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_gelu_lut(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_fc2_w(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_fc2_w(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_fc2_b(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_fc2_b(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_token_out(XVit_transformer_layer_fused *InstancePtr, u64 Data);
u64 XVit_transformer_layer_fused_Get_token_out(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_qkv_w_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data);
u32 XVit_transformer_layer_fused_Get_qkv_w_scale(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_attn_proj_w_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data);
u32 XVit_transformer_layer_fused_Get_attn_proj_w_scale(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_fc1_w_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data);
u32 XVit_transformer_layer_fused_Get_fc1_w_scale(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_fc2_w_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data);
u32 XVit_transformer_layer_fused_Get_fc2_w_scale(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_hidden_inv_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data);
u32 XVit_transformer_layer_fused_Get_hidden_inv_scale(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_lut_min(XVit_transformer_layer_fused *InstancePtr, u32 Data);
u32 XVit_transformer_layer_fused_Get_lut_min(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_Set_lut_index_scale(XVit_transformer_layer_fused *InstancePtr, u32 Data);
u32 XVit_transformer_layer_fused_Get_lut_index_scale(XVit_transformer_layer_fused *InstancePtr);

void XVit_transformer_layer_fused_InterruptGlobalEnable(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_InterruptGlobalDisable(XVit_transformer_layer_fused *InstancePtr);
void XVit_transformer_layer_fused_InterruptEnable(XVit_transformer_layer_fused *InstancePtr, u32 Mask);
void XVit_transformer_layer_fused_InterruptDisable(XVit_transformer_layer_fused *InstancePtr, u32 Mask);
void XVit_transformer_layer_fused_InterruptClear(XVit_transformer_layer_fused *InstancePtr, u32 Mask);
u32 XVit_transformer_layer_fused_InterruptGetEnabled(XVit_transformer_layer_fused *InstancePtr);
u32 XVit_transformer_layer_fused_InterruptGetStatus(XVit_transformer_layer_fused *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
