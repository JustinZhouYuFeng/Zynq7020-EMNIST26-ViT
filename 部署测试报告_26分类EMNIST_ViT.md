# Zynq 7020 上 EMNIST 26 分类 ViT 部署测试报告

## 1. 项目概述

本项目完成了一个面向 Zynq 7020 FPGA 板卡的 EMNIST Letters 26 分类 ViT 推理系统部署。系统输入为 `28 x 28` 单通道灰度字母图像，输出为 `A-Z` 共 26 类预测结果。

工程采用 PS+PL 协同架构。PC 端通过以太网 UDP 向板卡发送图像数据；Zynq PS 端负责网口接收、运行时调度、DDR 数据管理、AXI-Lite 控制和 UART/UDP 输出；Zynq PL 端通过 HLS 生成的 fused transformer IP 完成主要推理计算。

最终稳定版本实现了：

```text
PC UDP 输入图像
        ↓
Zynq PS 接收和调度
        ↓
PL fused transformer 推理
        ↓
PS 读取分类结果
        ↓
UART/UDP 返回 PC
```

该版本用于课题组考核时，可以作为“完成 26 分类轻量级 ViT 在 Zynq 7020 上部署”的证明材料。

## 2. 软硬件环境

### 2.1 硬件平台

| 项目 | 配置 |
|---|---|
| FPGA 平台 | Zynq 7020 |
| PS 处理器 | ARM Cortex-A9 |
| PL 资源 | LUT、FF、BRAM、DSP |
| PC 到板卡输入 | 以太网 UDP |
| 板卡到 PC 输出 | UART COM9 / UDP ACK |
| JTAG | 用于 Program FPGA 和下载 ELF |
| 串口 | COM9，115200 baud |

### 2.2 软件环境

| 工具 | 用途 |
|---|---|
| PyTorch | 模型训练、参数导出、golden 结果生成 |
| Vitis HLS 2020.1 | HLS IP 综合与导出 |
| Vivado 2020.1 | Block Design、综合、实现、bitstream 生成 |
| Vitis 2020.1 | Platform、BSP、Application、ELF 构建 |
| PowerShell | PC 端验证脚本运行 |

### 2.3 通信参数

| 项目 | 数值 |
|---|---|
| 板卡 IP | `192.168.1.10` |
| PC IP | `192.168.1.100` |
| UDP 端口 | `5001` |
| 单张输入大小 | `784` 字节 |
| 串口 | `COM9` |
| 波特率 | `115200` |

## 3. 工程文件位置

26 分类最终工程：

```text
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project
```

关键文件如下：

| 文件或目录 | 说明 |
|---|---|
| `design_1_wrapper_mlp256_u16_rev25.bit` | 已验证 FPGA bitstream |
| `design_1_wrapper_mlp256_u16_rev25.xsa` | Vivado 导出的硬件平台 |
| `vit_qvk_test_udp_fused_mlp256_u16_rev25_fixedphy.elf` | 已验证 PS 端 ELF |
| `src\main.c` | PS 端主程序源码 |
| `src\tinyvit_samples_vitis.h` | 模型参数、scale、测试样本相关头文件 |
| `src\emnist_pair_resolvers_vitis.h` | 字母混淆修正相关头文件 |
| `test_data\emnist_letters_udp` | 26 个 784 字节测试样本 |
| `tools\verify_udp_classification_batch.ps1` | PC 端批量验证脚本 |
| `reports\utilization_placed.rpt` | Vivado 资源利用率报告 |
| `reports\timing_summary.rpt` | Vivado 时序报告 |
| `verify_mlp256_u16_rev25_release_25of26_delay2500.log` | 最终验证日志 |

## 4. 模型与数据集说明

### 4.1 数据集

本工程使用 EMNIST Letters 数据集。该数据集包含英文字母 `A-Z`，共 26 类。原始 EMNIST Letters 标签通常为 `1-26`，工程内部统一转换为 `0-25`。

类别映射如下：

```text
0  = A
1  = B
2  = C
...
25 = Z
```

输入图像格式：

```text
28 x 28 grayscale
1 channel
784 bytes
pixel range: 0-255
```

### 4.2 模型结构

最终部署版本采用轻量级 ViT 结构，配置如下：

| 项目 | 数值 |
|---|---:|
| 输入尺寸 | `28 x 28` |
| patch size | `7 x 7` |
| patch token 数 | `16` |
| class token 数 | `1` |
| 总 token 数 | `17` |
| embedding 维度 | `64` |
| Transformer 深度 | `3` |
| attention heads | `4` |
| MLP hidden 维度 | `256` |
| 输出类别 | `26` |

张量流如下：

```text
28x28 image
        ↓
7x7 patch split
        ↓
16 patch tokens
        ↓
add class token
        ↓
17 x 64 token features
        ↓
3-layer transformer stack
        ↓
classification head
        ↓
26-class logits
```

## 5. 系统架构

系统由 PC 端、Zynq PS 端和 Zynq PL 端构成。

### 5.1 PC 端

PC 端负责：

```text
读取 test_data 中的 784 字节样本
通过 UDP 发送到 192.168.1.10:5001
监听板卡 ACK/结果
读取 COM9 串口日志
统计正确率和耗时
```

### 5.2 PS 端

PS 端负责：

```text
初始化 UART
初始化 Ethernet GEM
初始化 lwIP
绑定 UDP 5001 端口
接收 784 字节图像
准备推理输入
配置 PL fused transformer IP
等待 PL 计算完成
读取分类结果
通过 UART 和 UDP 返回结果
```

### 5.3 PL 端

PL 端负责主要计算：

```text
patch/token 相关计算
QKV/attention/projection/MLP 相关 fused transformer 计算
中间特征和输出写回 DDR
```

日志中出现：

```text
path=fused_transformer_pl
```

表示当前推理路径确实走 PL 端 fused transformer IP。

## 6. Vivado 部署流程

Vivado 负责硬件系统搭建和 bitstream 生成。

主要步骤：

```text
打开 Vivado 工程
        ↓
检查 Block Design
        ↓
确认 Zynq PS、AXI Interconnect、HLS IP、Reset、Clock 连接
        ↓
Validate Design
        ↓
Generate Output Products
        ↓
Synthesis
        ↓
Implementation
        ↓
Generate Bitstream
        ↓
Export Hardware，勾选 Include bitstream
```

关键连接：

```text
PS M_AXI_GP0 -> AXI Interconnect -> HLS IP s_axi_control
HLS IP m_axi_gmem_* -> AXI Memory Interconnect -> PS S_AXI_HP0 -> DDR
FCLK_CLK0 -> HLS IP ap_clk
Processor System Reset -> HLS IP ap_rst_n
```

其中：

| 接口 | 作用 |
|---|---|
| `s_axi_control` | PS 通过 AXI-Lite 控制 PL IP |
| `m_axi_gmem_*` | PL 访问 DDR 中的输入、权重、输出 |
| `ap_clk` | PL IP 时钟 |
| `ap_rst_n` | PL IP 复位 |

导出的 XSA 用于 Vitis 创建 platform。必须勾选 `Include bitstream`，否则 Vitis 后续 Program FPGA 时容易找不到正确 bit 文件。

## 7. Vitis 部署流程

Vitis 负责生成 BSP、编译应用程序并下载运行。

### 7.1 Platform Project

使用 Vivado 导出的 `.xsa` 创建 Platform Project：

```text
Processor: ps7_cortexa9_0
OS: standalone
```

因为工程使用 UDP，BSP 必须启用：

```text
lwip211
```

否则会出现：

```text
fatal error: netif/xadapter.h: No such file or directory
```

### 7.2 Application Project

Application 工程为：

```text
vision_transformer_
```

需要包含以下源码：

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

注意不要同时编译 `platform.c` 和 `platform_zynq.c`。本工程保留 `platform_zynq.c`，因为它包含 lwIP 所需的 Zynq 平台初始化逻辑。

### 7.3 构建顺序

正确构建顺序：

```text
Build vit_transformer platform
        ↓
Build vision_transformer_ application
        ↓
Build vision_transformer__system package
```

不能先 build system，因为 system 只是打包工程，必须等 application 先生成 ELF。

## 8. 上板运行流程

### 8.1 下载 bitstream

在 Vitis 中选择：

```text
Xilinx -> Program FPGA
```

配置：

```text
Connection: Local
Device: xc7z020
Bitstream: design_1_wrapper_mlp256_u16_rev25.bit
```

`Software Configuration` 可以为空。这个窗口只负责下载 bitstream，ELF 后面通过 `Launch on Hardware` 下载到 ARM。

### 8.2 运行 ELF

bitstream 下载后，右键 application：

```text
vision_transformer_
```

选择：

```text
Run As -> Launch on Hardware
```

目标处理器：

```text
ps7_cortexa9_0
```

### 8.3 板端启动状态

串口 COM9 正常启动后，会看到：

```text
WAIT_UDP_IMAGE_784_BYTES
```

表示程序已经运行，正在等待 PC 发送一张 784 字节图像。

## 9. 测试方法

测试使用固定 26 个 EMNIST Letters 样本，每个类别一个样本。

测试命令：

```powershell
cd C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project

powershell -ExecutionPolicy Bypass -File tools\verify_udp_classification_batch.ps1 -DataDir test_data\emnist_letters_udp -VitisHeader src\tinyvit_samples_vitis.h -Count 26 -DelayMs 2500 -SerialPort COM9
```

若 UDP ACK 不稳定，可使用：

```powershell
-DelayMs 5000
```

或：

```powershell
-DelayMs 8000
```

测试脚本会统计：

```text
UDP ACK 是否成功
板端预测类别
真实标签
是否识别正确
是否与模型 golden 匹配
单张推理耗时
平均推理耗时
```

## 10. 测试结果

最终验证日志：

```text
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project\verify_mlp256_u16_rev25_release_25of26_delay2500.log
```

关键结果：

| 指标 | 结果 |
|---|---:|
| 测试样本数 | 26 |
| UDP ACK | 26/26 |
| label match | 25/26 |
| model match | 25/26 |
| 平均推理时间 | 42.54 ms |
| 推理路径 | `fused_transformer_pl` |

其中 `label_match=25/26` 表示 26 个固定测试样本中有 25 个预测类别与真实标签一致。

日志摘要：

```text
label_match=25/26
model_match=25/26
infer_ms_avg=42.54 count=26
```

典型单样本输出：

```text
UDP_RESULT pred=0x00000010 base_pred=0x00000010 path=fused_transformer_pl depth=3 embed=64 mlp=256 classes=26 dataset=emnist-letters infer_ms=42 patch_ms=4 pl_stack_ms=37 head_ms=0
```

含义：

| 字段 | 说明 |
|---|---|
| `pred=0x00000010` | 预测类别为 16，即 Q |
| `path=fused_transformer_pl` | 使用 PL fused transformer |
| `depth=3` | 3 层 Transformer |
| `embed=64` | embedding 维度 64 |
| `mlp=256` | MLP hidden 维度 256 |
| `classes=26` | 26 分类 |
| `infer_ms=42` | 单张推理 42 ms |
| `patch_ms=4` | patch 预处理 4 ms |
| `pl_stack_ms=37` | PL transformer 栈 37 ms |
| `head_ms=0` | 分类头耗时约 0 ms |

## 11. 资源利用与时序结果

资源报告：

```text
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project\reports\utilization_placed.rpt
```

主要资源占用：

| 资源 | 使用量 | 总量 | 利用率 |
|---|---:|---:|---:|
| Slice LUTs | 29,073 | 53,200 | 54.65% |
| Slice Registers | 36,542 | 106,400 | 34.34% |
| Block RAM Tile | 68 | 140 | 48.57% |
| DSPs | 121 | 220 | 55.00% |

时序报告：

```text
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project\reports\timing_summary.rpt
```

时序结果：

| 指标 | 数值 |
|---|---:|
| WNS | +2.458 ns |
| TNS | 0.000 ns |
| Timing constraints | Met |

报告中显示：

```text
All user specified timing constraints are met.
```

说明实现后的硬件设计满足时序约束，可以作为稳定 bitstream 使用。

## 12. 过程问题与处理

### 12.1 HLS IP 导出问题

Vitis HLS 2020.1 在较新的日期环境下可能出现 IP packager `core_revision` 过大导致导出失败。处理方式是修改 IP packager 脚本中的 revision 为较小整数，再重新打包 IP。

### 12.2 Vitis BSP 编译问题

HLS 自动生成的驱动 Makefile 在 Windows 环境下可能出现：

```text
arm-none-eabi-ar: *.o: Invalid argument
```

原因是 `*.o` 没有被展开。处理方法是把 Makefile 中：

```text
OUTS = *.o
```

改成明确对象文件列表。

### 12.3 缺少 lwIP 头文件

如果出现：

```text
fatal error: netif/xadapter.h: No such file or directory
```

说明 BSP 没启用 lwIP，需要在 BSP Settings 中添加 `lwip211`。

### 12.4 缺少 platform 文件

如果出现：

```text
fatal error: platform.h: No such file or directory
```

说明 application 缺少平台支持源码，需要加入：

```text
platform.h
platform_config.h
platform_zynq.c
xemacpsif_physpeed_fixed.c
```

### 12.5 平台函数重复定义

如果出现：

```text
multiple definition of init_platform
multiple definition of cleanup_platform
```

说明 `platform.c` 和 `platform_zynq.c` 同时参与编译。保留 `platform_zynq.c`，移除或改名 `platform.c`。

### 12.6 Program FPGA 失败

如果 Vitis 显示：

```text
could not find configuration request
```

通常是没有选中 FPGA 配置目标。应在 Program FPGA 窗口中手动选择 `xc7z020`，不要选 ARM Cortex-A9。

## 13. 测试结论

本次部署测试完成了 EMNIST Letters 26 分类 ViT 在 Zynq 7020 板卡上的端到端验证。系统实现了 PC 端 UDP 输入、Zynq PS 端接收和调度、PL 端 fused transformer 加速推理，以及 UART/UDP 结果返回。

最终测试结果表明：

```text
26 个固定字母样本中 25 个分类正确
平均单张推理时间 42.54 ms
推理路径为 fused_transformer_pl
Vivado 实现满足时序约束
DSP、LUT、BRAM 资源利用处于可接受范围
```

因此，该系统已经完成从模型训练、参数导出、HLS IP、Vivado 硬件集成、Vitis 软件部署、网口输入、PL 推理到 UART/UDP 输出的完整闭环，可作为 26 分类轻量级 ViT 在 Zynq 7020 上部署成功的证明。

