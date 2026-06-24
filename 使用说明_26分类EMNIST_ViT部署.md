# Zynq 7020 EMNIST 26 分类 ViT 部署使用说明

本文档说明如何使用已经完成的 26 分类 EMNIST Letters 版本，以及如果自己从 Vivado/Vitis 重新操作一遍，应该按什么顺序执行。

## 1. 项目用途

本项目实现的是一个基于 Zynq 7020 的 26 分类字母识别系统。

输入是 28x28 灰度字母图像，输出是 A-Z 共 26 类中的预测类别。系统采用 PS+PL 协同：

- PC 通过以太网 UDP 发送 784 字节图像。
- PS 端接收 UDP 数据、调度推理流程、控制 PL IP、通过 UART/UDP 输出结果。
- PL 端执行 fused transformer 加速计算。
- PC 端通过 PowerShell 脚本批量发送样本并接收 ACK。

数据流程：

```text
PC 端 28x28 图像样本
        ↓ UDP 5001
Zynq PS 以太网接收
        ↓
PS 写入 DDR / 配置 AXI-Lite
        ↓
PL fused transformer IP 推理
        ↓
PS 读取结果
        ↓ UART COM9 / UDP ACK 返回 PC
```

## 2. 最终工程位置

26 分类稳定版目录：

```text
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project
```

主开发工程目录：

```text
C:\Users\19571\Desktop\FPGA_ViT_Project
```

ViT-B RGB224 复杂版目录：

```text
C:\Users\19571\Desktop\FPGA_ViTB_RGB224_Project
```

本说明主要针对 26 分类 EMNIST 版本。

## 3. 关键文件说明

在 26 分类工程目录中：

```text
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project
```

关键文件如下：

| 文件或目录 | 作用 |
|---|---|
| `design_1_wrapper_mlp256_u16_rev25.bit` | 已验证的 FPGA bitstream |
| `design_1_wrapper_mlp256_u16_rev25.xsa` | Vivado 导出的硬件平台文件 |
| `vit_qvk_test_udp_fused_mlp256_u16_rev25_fixedphy.elf` | 已验证的 PS 端 ELF |
| `src\main.c` | PS 端主程序源码 |
| `src\tinyvit_samples_vitis.h` | 模型参数和样本相关头文件 |
| `src\emnist_pair_resolvers_vitis.h` | 字母混淆修正相关头文件 |
| `test_data\emnist_letters_udp` | 26 个测试样本 |
| `tools\verify_udp_classification_batch.ps1` | PC 端批量发送和验证脚本 |
| `verify_mlp256_u16_rev25_release_25of26_delay2500.log` | 已验证日志 |
| `zynq_vit_qkv_system` | Vivado 工程 |
| `vit_transformer` | Vitis platform 工程 |
| `vision_transformer_` | Vitis application 工程 |
| `vision_transformer__system` | Vitis system/package 工程 |
| `vision_transformer` | HLS/IP 相关工程 |

## 4. 硬件连接要求

需要连接：

```text
JTAG      用于 Program FPGA / 下载 ELF
网口      用于 PC 向板卡发送图像
串口 COM9 用于查看板卡输出
电源      保证板卡正常上电
```

默认通信参数：

| 项目 | 值 |
|---|---|
| 板卡 IP | `192.168.1.10` |
| PC IP | `192.168.1.100` |
| UDP 端口 | `5001` |
| UART | `COM9` |
| 波特率 | `115200` |
| 单张输入大小 | `784` 字节 |

PC 网口建议设置为静态 IP：

```text
IP:      192.168.1.100
Mask:    255.255.255.0
Gateway: 192.168.1.1
```

## 5. 最快使用现成版本

如果只是想直接验证已完成版本，不需要重新综合或重新编译。

### 5.1 Program FPGA

在 Vitis 中选择：

```text
Xilinx -> Program FPGA
```

推荐选择：

```text
Project: vit_transformer 或 vision_transformer_
Connection: Local
Device: 手动选择 xc7z020
Bitstream:
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project\design_1_wrapper_mlp256_u16_rev25.bit
```

下面的 `Software Configuration` 可以空着。这个窗口只负责下载 bitstream。

如果 GUI 自动检测失败，点 `Select...` 手动选择 JTAG 目标中的 `xc7z020`，不要选 ARM Cortex-A9。

### 5.2 运行 ELF

bitstream 下载完成后，右键 application 工程：

```text
vision_transformer_
```

选择：

```text
Run As -> Launch on Hardware
```

目标处理器选择：

```text
ps7_cortexa9_0
```

ELF 路径通常是：

```text
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project\vision_transformer_\Debug\vision_transformer_.elf
```

也可以使用已验证 ELF：

```text
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project\vit_qvk_test_udp_fused_mlp256_u16_rev25_fixedphy.elf
```

### 5.3 打开串口

串口参数：

```text
COM9
115200
8N1
```

程序运行后，串口会看到类似：

```text
WAIT_UDP_IMAGE_784_BYTES
```

表示板卡已经启动，正在等待 PC 通过 UDP 发送 784 字节图像。

### 5.4 批量发送 26 个样本

在 PowerShell 执行：

```powershell
cd C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project

powershell -ExecutionPolicy Bypass -File tools\verify_udp_classification_batch.ps1 -DataDir test_data\emnist_letters_udp -VitisHeader src\tinyvit_samples_vitis.h -Count 26 -DelayMs 5000 -SerialPort COM9
```

如果通信稳定，也可以把 `DelayMs` 改为：

```powershell
-DelayMs 2500
```

如果出现 `ack=False`，建议加大间隔：

```powershell
-DelayMs 8000
```

## 6. 输出日志怎么看

典型输出：

```text
[SEND] index=16 label=16 ack=True
[COM9] UDP_RX_IMAGE len=784 from=192.168.1.100:58023
UDP_INFER_BEGIN len=784
UDP_RESULT pred=0x00000010 base_pred=0x00000010 path=fused_transformer_pl depth=3 embed=64 mlp=256 classes=26 dataset=emnist-letters logit_pred_milli=0x00002E44 infer_ms=42 patch_ms=4 pl_stack_ms=37 head_ms=0
WAIT_UDP_IMAGE_784_BYTES
```

含义：

| 字段 | 含义 |
|---|---|
| `index=16` | 当前发送第 16 个测试样本 |
| `label=16` | 真实标签为第 16 类 |
| `ack=True` | PC 收到板卡回复 |
| `UDP_RX_IMAGE len=784` | 板卡收到 28x28 灰度图 |
| `pred=0x00000010` | 预测类别为十六进制 0x10，即十进制 16 |
| `path=fused_transformer_pl` | 推理路径走 PL fused transformer |
| `depth=3` | 模型深度 3 层 |
| `embed=64` | embedding 维度 64 |
| `mlp=256` | MLP 隐层维度 256 |
| `classes=26` | 26 分类 |
| `infer_ms=42` | 单张总推理约 42 ms |
| `patch_ms=4` | patch/预处理约 4 ms |
| `pl_stack_ms=37` | PL transformer 栈约 37 ms |
| `head_ms=0` | 分类头耗时接近 0 ms |

类别编号对应关系：

```text
0  = A
1  = B
2  = C
...
25 = Z
```

所以：

```text
pred=0x00000010
```

表示：

```text
0x10 = 16 = Q
```

## 7. GEM_STATS 里很多 0 是什么意思

日志中可能看到：

```text
GEM_STATS rx=0x00000000 tx=0x00000001 rx_fcs=0x00000000 rx_sym=0x00000000 rx_align=0x00000000 isr=0x00000000 rxsr=0x00000000 txsr=0x00000000
```

这是 Zynq 以太网 MAC 的状态寄存器调试信息。

常见字段：

| 字段 | 含义 |
|---|---|
| `rx` | 接收统计 |
| `tx` | 发送统计 |
| `rx_fcs` | FCS 校验错误 |
| `rx_sym` | 符号错误 |
| `rx_align` | 对齐错误 |
| `isr` | 中断状态 |
| `rxsr` | 接收状态 |
| `txsr` | 发送状态 |

很多字段为 0 是正常的，尤其是：

```text
rx_fcs=0
rx_sym=0
rx_align=0
```

这表示没有明显网口错误。

## 8. 自己重新从 Vivado/Vitis 做一遍

如果你要完整复现，不直接用现成 bit/ELF，按以下顺序。

### 8.1 Vivado 侧

1. 打开 Vivado 工程：

```text
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project\zynq_vit_qkv_system
```

2. 检查 Block Design。

关键连接应满足：

```text
PS M_AXI_GP0 -> AXI Interconnect -> HLS IP s_axi_control
HLS IP m_axi_gmem_* -> AXI Memory Interconnect -> PS S_AXI_HP0 / DDR
FCLK_CLK0 -> HLS IP ap_clk
Processor System Reset -> HLS IP ap_rst_n
```

3. Generate Bitstream。

4. 导出硬件：

```text
File -> Export -> Export Hardware
```

必须勾选：

```text
Include bitstream
```

导出 `.xsa`。

### 8.2 Vitis 侧

1. 新建或更新 Platform Project。

使用刚导出的 `.xsa`。

2. Processor 选择：

```text
ps7_cortexa9_0
```

3. OS 选择：

```text
standalone
```

4. BSP 里需要启用 lwIP：

```text
lwip211
```

否则会出现：

```text
fatal error: netif/xadapter.h: No such file or directory
```

5. 新建 Empty Application。

Application 源码至少需要：

```text
main.c
tinyvit_samples_vitis.h
emnist_pair_resolvers_vitis.h
qkv_params_vitis.h
platform.h
platform_config.h
platform_zynq.c
xemacpsif_physpeed_fixed.c
```

注意不要同时编译 `platform.c` 和 `platform_zynq.c`。如果两者都在工程里，会出现：

```text
multiple definition of `init_platform'
multiple definition of `cleanup_platform'
```

本工程保留：

```text
platform_zynq.c
```

把模板文件改名或排除：

```text
platform.c.disabled
```

6. 编译顺序：

```text
Build vit_transformer platform
Build vision_transformer_ application
Build vision_transformer__system package
```

不要先 build system。system 只是打包工程，需要 application 先生成 ELF。

## 9. 常见问题

### 9.1 `package Error 1`

这个通常不是根因。真正原因要看前面的第一条 `fatal error` 或 `undefined reference`。

常见原因：

```text
ELF does not exist
头文件缺失
链接失败
应用工程没有先 build
```

### 9.2 `ELF does not exist`

说明 system project 要打包 ELF，但 application 没生成 ELF。

先 build：

```text
vision_transformer_
```

确认生成：

```text
vision_transformer_\Debug\vision_transformer_.elf
```

再 build：

```text
vision_transformer__system
```

### 9.3 `netif/xadapter.h` 找不到

说明 BSP 没有启用 lwIP。

在 BSP Settings 中添加：

```text
lwip211
```

然后重新 build platform/BSP。

### 9.4 `platform.h` 找不到

说明 application 源码目录缺少平台封装文件。

需要补：

```text
platform.h
platform_config.h
platform_zynq.c
xemacpsif_physpeed_fixed.c
```

### 9.5 `multiple definition of init_platform`

说明 `platform.c` 和 `platform_zynq.c` 同时参与编译。

保留：

```text
platform_zynq.c
```

移除或改名：

```text
platform.c
```

### 9.6 `Program FPGA failed: could not find configuration request`

通常是 Vitis 没选中正确 JTAG FPGA 目标。

处理方法：

```text
Device 点 Select...
选择 xc7z020
不要选 ARM Cortex-A9
```

必要时勾选：

```text
Skip Revision Check
```

### 9.7 `ack=False`

说明 PC 没收到板卡 ACK。常见原因是 UDP 包或返回包丢失，或者发送太快。

处理：

```powershell
-DelayMs 5000
```

如果仍不稳定：

```powershell
-DelayMs 8000
```

如果板卡串口仍显示：

```text
WAIT_UDP_IMAGE_784_BYTES
```

说明程序还活着，可以直接重跑 PC 脚本。

## 10. 如何发送自己的图片

26 分类版本不能直接发送普通 JPG/PNG 彩色图。它要求输入格式是：

```text
28x28 灰度图
单通道
784 字节
像素范围 0~255
UDP 发送到 192.168.1.10:5001
```

项目中的标准测试样本位于：

```text
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project\test_data\emnist_letters_udp
```

如果要测试自己手写字母，需要先把图片转换成 28x28 灰度 raw/bin 格式，再发送。注意 EMNIST Letters 的字母方向和普通手写图片可能存在旋转/镜像差异，所以随手写的正向字母不一定有好效果。建议先用项目自带样本验证板端流程。

## 11. 推荐演示话术

可以这样介绍本项目：

```text
本项目在 Zynq 7020 上实现了 26 类 EMNIST 字母识别的 PS+PL 协同推理系统。PC 端通过 UDP 发送 28x28 灰度字母图像，PS 端完成网口接收、DDR 数据管理和 AXI-Lite 控制，PL 端通过 HLS 实现 fused transformer IP 执行主要推理计算，结果通过 UART/UDP 返回。最终验证流程可完成 26 个字母样本的板端分类，稳定版本达到 25/26 的验证结果。
```

重点强调：

```text
网口输入
PL fused transformer 推理
UART/UDP 输出
PS+PL 协同
完整上板闭环
```

