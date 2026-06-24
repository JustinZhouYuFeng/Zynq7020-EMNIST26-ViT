# EMNIST 26-Class Verification Package

This folder is the accepted 26-class EMNIST letters version.

Model/package:
- Task: EMNIST Letters, 26 classes
- Accepted result: 25/26 sample verification
- Bitstream: `design_1_wrapper_mlp256_u16_rev25.bit`
- ELF: `vit_qvk_test_udp_fused_mlp256_u16_rev25_fixedphy.elf`
- Verification log: `verify_mlp256_u16_rev25_release_25of26_delay2500.log`

Typical verification flow:

1. Program the board with:
   - bitstream: `design_1_wrapper_mlp256_u16_rev25.bit`
   - ELF: `vit_qvk_test_udp_fused_mlp256_u16_rev25_fixedphy.elf`

2. Keep the board network at:
   - board IP: `192.168.1.10`
   - PC/local IP: `192.168.1.100`
   - UDP port: `5001`
   - UART: `COM9`, `115200`

3. Run batch verification from this folder:

```powershell
powershell -ExecutionPolicy Bypass -File tools\verify_udp_classification_batch.ps1 -DataDir test_data\emnist_letters_udp -VitisHeader src\tinyvit_samples_vitis.h -Count 26 -DelayMs 2500 -SerialPort COM9
```

Notes:
- Inputs are 28x28 grayscale raw images, 784 bytes each.
- The model is for letters A-Z, not ImageNet pictures.
- This package is intentionally separated from the ViT-B ImageNet project.

