// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.1 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xvit_transformer_layer_fused.h"

extern XVit_transformer_layer_fused_Config XVit_transformer_layer_fused_ConfigTable[];

XVit_transformer_layer_fused_Config *XVit_transformer_layer_fused_LookupConfig(u16 DeviceId) {
	XVit_transformer_layer_fused_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XVIT_TRANSFORMER_LAYER_FUSED_NUM_INSTANCES; Index++) {
		if (XVit_transformer_layer_fused_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XVit_transformer_layer_fused_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XVit_transformer_layer_fused_Initialize(XVit_transformer_layer_fused *InstancePtr, u16 DeviceId) {
	XVit_transformer_layer_fused_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XVit_transformer_layer_fused_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XVit_transformer_layer_fused_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

