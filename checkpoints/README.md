# Checkpoint Provenance

The committed checkpoints are small enough to keep the software-to-hardware flow directly reproducible.

| Checkpoint | Purpose | Recorded test accuracy |
| --- | --- | ---: |
| `tiny_vit_emnist_letters_depth3_mlp256.pt` | Base 3-layer TinyViT | 89.8846% |
| `emnist_cnn_teacher.pt` | CNN distillation teacher | 95.0144% |
| `tiny_vit_emnist_letters_depth3_mlp256_distilled.pt` | First distilled TinyViT | 90.1298% |
| `tiny_vit_emnist_letters_depth3_mlp256_accboost_v2.pt` | Distillation fine-tuning candidate | 90.3317% |
| `tiny_vit_emnist_letters_depth3_mlp256_soup_a086.pt` | Final deployed 14/86 parameter blend | 90.2933% GPU record; 90.2740% CPU rerun |

The final state dictionary is computed as:

```text
theta_final = 0.14 * theta_base + 0.86 * theta_accboost_v2
```

This is a parameter-level blend of two models with identical architecture. It does not run two models on the board and does not increase FPGA compute or storage requirements.

Rebuild and verify the exact deployed model tensors:

```powershell
python tools\blend_emnist_checkpoints.py `
  --reference checkpoints\tiny_vit_emnist_letters_depth3_mlp256_soup_a086.pt
```

The recorded `test_acc` fields reflect the original evaluation environment. Use `tools/evaluate_emnist_checkpoint.py` to recompute accuracy in the current environment.
