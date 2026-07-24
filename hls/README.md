# Final HLS Transformer Layer

This directory contains the source and Vitis HLS scripts for the final fused Transformer Encoder IP.

## Final Configuration

- Device: `xc7z020-clg400-1`
- MLP dimension: `256`
- MAC unroll factor: `16`
- Target period: `16 ns` (`62.5 MHz`)
- Solution: `solution_unroll16`
- Arithmetic path: floating point

The IP performs one complete Pre-LayerNorm Transformer Encoder layer. The PS invokes the same IP three times with different parameter addresses.

## Files

| File | Role |
| --- | --- |
| `vit_transformer_layer_fused.cpp/.h` | Final accepted HLS implementation |
| `tb_vit_transformer_layer_fused.cpp` | C-simulation smoke test |
| `run_transformer_layer_fused_mlp256_u16_csim_7020.tcl` | C simulation |
| `run_transformer_layer_fused_mlp256_u16_synth_7020.tcl` | Synthesis |
| `run_transformer_layer_fused_mlp256_u16_export_7020.tcl` | IP catalog export |
| `reports/vit_transformer_layer_fused_csynth.rpt` | Final synthesis report |

## Commands

Run from the repository root with Vitis HLS 2020.1 available on `PATH`:

```powershell
vitis_hls -f hls\run_transformer_layer_fused_mlp256_u16_csim_7020.tcl
vitis_hls -f hls\run_transformer_layer_fused_mlp256_u16_synth_7020.tcl
vitis_hls -f hls\run_transformer_layer_fused_mlp256_u16_export_7020.tcl
```

The C simulation is a structural smoke test. Numerical equivalence is established through exported PyTorch reference predictions and the 520-sample board comparison in `validation_results/board_520/`.
