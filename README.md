# Zynq7020 EMNIST26 ViT Deployment

This repository contains a Zynq-7020 deployment package for a lightweight Vision Transformer targeting EMNIST Letters A-Z classification.

## Project Summary

- **Task:** EMNIST Letters, 26 classes (A-Z)
- **Board:** Zynq-7020, ARM Cortex-A9 + FPGA PL
- **Model:** Lightweight ViT, `depth=3`, `embed=64`, `heads=4`, `MLP=256`
- **Input:** 28x28 grayscale image, 784 bytes per sample
- **Communication:** PC sends image data by UDP, board returns ACK/result through UDP and COM9 UART
- **Acceleration:** PL-side fused transformer HLS IP
- **Accepted result:** 25/26 fixed EMNIST letter samples classified correctly

## Verified Result

| Metric | Result |
| --- | --- |
| Classes | 26 |
| UDP ACK | 26/26 |
| Fixed-sample accuracy | 25/26, about 96.15% |
| Average latency | 42.54 ms/image |
| Typical patch time | 4 ms |
| Typical PL transformer time | 37-38 ms |
| Inference path | `fused_transformer_pl` |
| WNS | +2.458 ns |

## Resource Usage

| Resource | Used | Total | Utilization |
| --- | ---: | ---: | ---: |
| Slice LUTs | 29,073 | 53,200 | 54.65% |
| Slice Registers | 36,542 | 106,400 | 34.34% |
| Block RAM Tile | 68 | 140 | 48.57% |
| DSP | 121 | 220 | 55.00% |

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/` | Vitis bare-metal application source and generated sample headers |
| `tools/` | Build, export, HLS, and UDP verification scripts |
| `drivers/` | HLS IP driver files for `vit_transformer_layer_fused` |
| `test_data/` | EMNIST Letters UDP test samples |
| `reports/` | Timing/utilization/IP status reports |
| `assets/` | Report figures |
| `Zynq7020_EMNIST26_ViT_部署测试报告_周玉峰.docx` | Final deployment test report |
| `使用说明_26分类EMNIST_ViT部署.md` | Usage guide in Chinese |
| `部署测试报告_26分类EMNIST_ViT.md` | Deployment test report in Chinese |
| `部署细节_26分类EMNIST_ViT.md` | Deployment details in Chinese |

## Hardware Artifacts

The repository includes the accepted hardware/software artifacts used for board verification:

- `design_1_wrapper_mlp256_u16_rev25.bit`
- `design_1_wrapper_mlp256_u16_rev25.xsa`
- `vit_qvk_test_udp_fused_mlp256_u16_rev25_fixedphy.elf`
- `verify_mlp256_u16_rev25_release_25of26_delay2500.log`

## Verification Flow

Board/network defaults:

- Board IP: `192.168.1.10`
- PC IP: `192.168.1.100`
- UDP port: `5001`
- UART: `COM9`, `115200 baud`

Run the batch verification from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools\verify_udp_classification_batch.ps1 `
  -DataDir test_data\emnist_letters_udp `
  -VitisHeader src\tinyvit_samples_vitis.h `
  -Count 26 `
  -DelayMs 2500 `
  -SerialPort COM9
```

Expected board-side markers include:

```text
WAIT_UDP_IMAGE_784_BYTES
UDP_RX_IMAGE len=784
UDP_INFER_BEGIN len=784
UDP_RESULT pred=... path=fused_transformer_pl
```

## Notes

This repository is the accepted 26-class EMNIST Letters deployment package. It is intentionally separated from the larger ViT-B/ImageNet exploration work. ViT-B/16 was also explored on Zynq-7020 for weight loading and board-side execution-chain validation, but its current accuracy and runtime are not suitable as the main result on this device.
