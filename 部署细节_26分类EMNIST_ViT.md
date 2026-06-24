# EMNIST 26 分类 ViT FPGA 部署细节说明

本文档补充说明 26 分类 EMNIST Letters ViT 工程的部署细节，重点解释从模型到上板运行的完整链路。它适合作为部署测试报告中的“系统实现与部署流程”部分。

## 1. 部署目标

本工程目标是在 Zynq 7020 板卡上实现一个可实际运行的 26 类字母识别系统。

系统输入为单张 `28x28` 灰度字母图像，大小为 `784` 字节；输出为 `A-Z` 共 26 个类别中的预测结果。整体链路包含：

```text
PC 发送图像
        ↓
Zynq PS 网口接收
        ↓
PS 整理输入并调度推理
        ↓
PL fused transformer IP 加速计算
        ↓
PS 读取结果
        ↓
UART/UDP 返回 PC
```

该版本不是纯软件推理，也不是 PL 独立完成全部工作，而是典型的 Zynq PS+PL 协同部署。PS 负责通信、控制、调度和结果输出；PL 负责 Transformer 中主要计算部分。

## 2. 工程目录组织

最终可运行工程位于：

```text
C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project
```

主开发工程位于：

```text
C:\Users\19571\Desktop\FPGA_ViT_Project
```

关键目录说明如下：

| 目录 | 内容 |
|---|---|
| `src` | 26 分类板端应用源码和模型参数头文件 |
| `tools` | PC 端发送、验证、辅助脚本 |
| `test_data` | 26 个 EMNIST Letters UDP 测试样本 |
| `zynq_vit_qkv_system` | Vivado 硬件工程 |
| `vision_transformer` | HLS IP 工程 |
| `vit_transformer` | Vitis platform 工程 |
| `vision_transformer_` | Vitis application 工程 |
| `vision_transformer__system` | Vitis system/package 工程 |

最终已验证文件：

| 文件 | 作用 |
|---|---|
| `design_1_wrapper_mlp256_u16_rev25.bit` | FPGA bitstream |
| `design_1_wrapper_mlp256_u16_rev25.xsa` | Vivado 导出的硬件平台 |
| `vit_qvk_test_udp_fused_mlp256_u16_rev25_fixedphy.elf` | 已验证裸机 ELF |
| `verify_mlp256_u16_rev25_release_25of26_delay2500.log` | 25/26 验证日志 |

## 3. 模型设计细节

本版本采用轻量级 ViT 结构，适配 Zynq 7020 资源限制。

核心配置：

```text
dataset = EMNIST Letters
classes = 26
input = 28 x 28 grayscale
patch size = 7 x 7
patch tokens = 16
class token = 1
total tokens = 17
embed dim = 64
depth = 3
heads = 4
MLP hidden dim = 256
```

数据维度变化：

```text
28 x 28 image
        ↓ patch split
16 patches, each patch 7 x 7
        ↓ patch embedding
16 patch tokens
        ↓ add class token
17 tokens x 64 channels
        ↓ transformer stack
class token feature
        ↓ classifier head
26 logits
```

类别编号从 0 开始：

```text
0  = A
1  = B
2  = C
...
25 = Z
```

所以日志中的：

```text
pred=0x00000010
```

代表十六进制 `0x10`，即十进制 `16`，对应字母 `Q`。

## 4. 量化与参数导出

FPGA 不适合直接部署完整浮点 PyTorch 模型，因此工程采用离线导出与定点化思路。

基本过程：

```text
PyTorch 训练模型
        ↓
导出权重、bias、scale、测试样本
        ↓
转换为 Vitis/HLS 可直接包含的 C 头文件
        ↓
PS 程序编译时把参数打包进 ELF
        ↓
运行时 PS 将参数送入 DDR 或配置给 PL IP
```

主要头文件：

| 文件 | 作用 |
|---|---|
| `tinyvit_samples_vitis.h` | 样本、权重、scale、golden 结果 |
| `qkv_params_vitis.h` | QKV 相关参数 |
| `emnist_pair_resolvers_vitis.h` | 部分类别混淆修正逻辑 |

量化的意义是降低乘法器和存储压力，使 QKV、MLP 等计算可以映射到 DSP/LUT/BRAM 资源中。

## 5. HLS IP 部署细节

PL 端核心是 fused transformer 相关 IP。它通过 HLS 生成，导出后在 Vivado Block Design 中接入 Zynq PS 系统。

HLS IP 的典型接口包括：

```text
s_axi_control   AXI-Lite 控制口
m_axi_gmem_*    AXI master 数据访问口
ap_clk          时钟
ap_rst_n        低有效复位
interrupt       可选中断
```

各接口作用：

| 接口 | 作用 |
|---|---|
| `s_axi_control` | PS 写寄存器、启动 IP、查询 done |
| `m_axi_gmem_token` | 访问 token 数据 |
| `m_axi_gmem_param` | 访问运行参数 |
| `m_axi_gmem_qkvw` | 访问 QKV 权重 |
| `m_axi_gmem_attnw` | 访问 attention/projection 权重 |
| `m_axi_gmem_mlpw` | 访问 MLP 权重 |
| `m_axi_gmem_out` | 写回输出 |

PS 通过 AXI-Lite 方式控制 PL IP，流程一般是：

```text
写输入地址
写权重地址
写输出地址
写层数/维度/scale 等参数
写 ap_start
轮询 ap_done
读取输出
```

## 6. Vivado Block Design 连接细节

Vivado 中主要模块包括：

```text
processing_system7_0
ps7_0_axi_periph
axi_mem_intercon
rst_ps7_0_50M
vit_transformer_layer_fused_0
```

关键连接关系：

```text
PS M_AXI_GP0
        ↓
ps7_0_axi_periph
        ↓
HLS IP s_axi_control
```

这一路用于 PS 控制 PL IP。

```text
HLS IP m_axi_gmem_*
        ↓
axi_mem_intercon
        ↓
PS S_AXI_HP0
        ↓
DDR
```

这一路用于 PL 从 DDR 中读取输入、权重、参数并写回输出。

时钟和复位：

```text
processing_system7_0/FCLK_CLK0 -> HLS IP ap_clk
rst_ps7_0_50M/peripheral_aresetn -> HLS IP ap_rst_n
```

地址分配需要在 Address Editor 中完成。PS 程序最终通过 `xparameters.h` 中生成的 base address 访问 IP。

## 7. XSA 导出细节

bitstream 生成完成后，需要导出 XSA：

```text
File -> Export -> Export Hardware
```

必须勾选：

```text
Include bitstream
```

XSA 的作用是把以下信息传递给 Vitis：

```text
PS 配置
PL bitstream
AXI 地址映射
DDR 配置
外设信息
IP 驱动信息
```

如果 XSA 没包含 bitstream，Vitis 仍可能能编译程序，但 Program FPGA 时容易找不到正确 bit 文件。

## 8. Vitis Platform 与 BSP 细节

Vitis 中先创建 Platform Project，再创建 Application Project。

Platform 配置：

```text
Processor: ps7_cortexa9_0
OS: standalone
Architecture: 32-bit ARM Cortex-A9
```

因为工程使用 UDP 通信，BSP 必须启用 lwIP：

```text
lwip211
```

否则编译 `main.c` 时会出现：

```text
fatal error: netif/xadapter.h: No such file or directory
```

Application 中需要包含：

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

注意不要同时编译：

```text
platform.c
platform_zynq.c
```

否则会出现：

```text
multiple definition of init_platform
multiple definition of cleanup_platform
```

本工程保留 `platform_zynq.c`，因为它包含 lwIP 所需的 Zynq 定时器和中断初始化逻辑。

## 9. PS 端软件流程

PS 端主程序完成以下工作：

```text
初始化 cache
初始化 UART1
初始化 Ethernet GEM
初始化 lwIP
绑定 UDP 5001 端口
等待 784 字节图像
收到图像后开始推理
调用 PL fused transformer IP
读取预测结果
通过 UART 和 UDP 返回
继续等待下一张图像
```

板端典型状态：

```text
WAIT_UDP_IMAGE_784_BYTES
```

表示程序已经启动并正在等待 PC 发图。

收到图像后：

```text
UDP_RX_IMAGE len=784 from=192.168.1.100
UDP_INFER_BEGIN len=784
UDP_RESULT pred=...
```

表示图像接收、推理和结果输出都已经执行。

## 10. PC 端发送流程

PC 端通过 PowerShell 脚本批量发送测试样本。

命令：

```powershell
cd C:\Users\19571\Desktop\FPGA_26Class_EMNIST_Project

powershell -ExecutionPolicy Bypass -File tools\verify_udp_classification_batch.ps1 -DataDir test_data\emnist_letters_udp -VitisHeader src\tinyvit_samples_vitis.h -Count 26 -DelayMs 5000 -SerialPort COM9
```

参数说明：

| 参数 | 说明 |
|---|---|
| `DataDir` | 784 字节 raw 样本目录 |
| `VitisHeader` | 用于读取标签/golden 信息的头文件 |
| `Count` | 发送样本数量 |
| `DelayMs` | 样本之间延迟 |
| `SerialPort` | UART 输出端口 |

如果出现：

```text
ack=False
```

说明 PC 没收到板卡回复，通常是 UDP 回包丢失或发送太快。可将 `DelayMs` 增加到 `8000`。

## 11. 运行结果字段解释

示例：

```text
[SEND] index=16 label=16 ack=True
[COM9] UDP_RX_IMAGE len=784 from=192.168.1.100:58023
UDP_INFER_BEGIN len=784
UDP_RESULT pred=0x00000010 base_pred=0x00000010 path=fused_transformer_pl depth=3 embed=64 mlp=256 classes=26 dataset=emnist-letters logit_pred_milli=0x00002E44 infer_ms=42 patch_ms=4 pl_stack_ms=37 head_ms=0
WAIT_UDP_IMAGE_784_BYTES
```

字段含义：

| 字段 | 说明 |
|---|---|
| `index=16` | 第 16 个测试样本 |
| `label=16` | 真实标签为 16 |
| `ack=True` | PC 收到板卡回复 |
| `UDP_RX_IMAGE len=784` | 板卡收到完整图像 |
| `pred=0x00000010` | 预测类别为 16 |
| `base_pred=0x00000010` | 基准预测也为 16 |
| `path=fused_transformer_pl` | 使用 PL fused transformer 路径 |
| `depth=3` | Transformer 深度为 3 |
| `embed=64` | embedding 维度为 64 |
| `mlp=256` | MLP 隐层为 256 |
| `classes=26` | 输出类别数为 26 |
| `infer_ms=42` | 总推理约 42 ms |
| `patch_ms=4` | patch/输入预处理约 4 ms |
| `pl_stack_ms=37` | PL transformer 栈约 37 ms |
| `head_ms=0` | 分类 head 约 0 ms |

这说明单张图片从接收到完成推理大约需要 42 ms，其中主要时间在 PL transformer 栈。

## 12. 硬件资源与复杂度说明

该版本相较于原始手写数字识别更加复杂：

| 项目 | 早期 MNIST 10 类 | 当前 EMNIST 26 类 |
|---|---:|---:|
| 分类数 | 10 | 26 |
| 数据集 | MNIST | EMNIST Letters |
| 模型深度 | 更浅 | depth=3 |
| MLP hidden | 较小 | 256 |
| 推理路径 | 部分 PS/PL | fused_transformer_pl |
| 输入输出链路 | 验证型 | 网口输入 + UART/UDP 输出 |

资源利用目标不是越低越好，而是在 Zynq 7020 上保持可实现的前提下，让 DSP/LUT/BRAM 有合理利用，使 PL 加速具有实际意义。

## 13. 常见部署问题

### 13.1 Program FPGA 找不到目标

报错：

```text
could not find configuration request
```

处理：

```text
Device -> Select...
选择 xc7z020
不要选择 ARM Cortex-A9
```

### 13.2 System package 报错

报错：

```text
make: *** [makefile:39: package] Error 1
```

这通常只是后果。需要查看前面真正的错误，比如：

```text
ELF does not exist
fatal error: xxx.h: No such file or directory
undefined reference
multiple definition
```

修复 Application 后再 build system。

### 13.3 找不到 `netif/xadapter.h`

原因：BSP 没启用 lwIP。

处理：

```text
BSP Settings -> Libraries -> lwip211
```

### 13.4 找不到 `platform.h`

原因：Application 源码不完整。

需要补：

```text
platform.h
platform_config.h
platform_zynq.c
xemacpsif_physpeed_fixed.c
```

### 13.5 `init_platform` 重复定义

原因：`platform.c` 和 `platform_zynq.c` 同时参与编译。

处理：

```text
保留 platform_zynq.c
移除或改名 platform.c
```

## 14. 自定义图片输入要求

当前 26 分类版本不能直接输入普通 RGB JPG/PNG。它要求 PC 发给板卡的是：

```text
28x28
灰度
单通道
784 字节
raw/bin 格式
```

UDP 目标：

```text
IP:   192.168.1.10
Port: 5001
```

如果要用自己手写的字母，需要先转换成 28x28 灰度 raw 数据。还要注意 EMNIST Letters 的图像方向可能和普通手写字母存在旋转或镜像差异，因此推荐先用工程自带的 `test_data\emnist_letters_udp` 验证系统链路。

## 15. 报告中可使用的总结表述

可以在报告中写：

```text
本工程完成了基于 Zynq 7020 的 EMNIST Letters 26 分类 ViT 推理部署。系统采用 PC 端 UDP 输入、Zynq PS 端通信与调度、PL 端 fused transformer IP 加速计算、UART/UDP 返回结果的结构。模型输入为 28x28 灰度字母图像，输出为 A-Z 26 类预测结果。最终系统在板端完成完整闭环验证，日志显示推理路径为 fused_transformer_pl，单张样本推理约 42 ms，其中 PL transformer 栈约 37 ms，固定 26 类测试样本可达到 25/26 的验证结果。
```

