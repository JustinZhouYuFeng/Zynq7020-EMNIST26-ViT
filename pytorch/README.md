# PyTorch Model Pipeline

This directory contains the software model used by the deployed Zynq-7020 project.

## Files

| File | Role |
| --- | --- |
| `tiny_vit_mnist.py` | Dataset loading, TinyViT definition, base training and evaluation |
| `train_emnist_distill.py` | CNN teacher and TinyViT knowledge-distillation training |
| `infer_tiny_vit_mnist.py` | Single-sample inference |
| `train_emnist_pair_resolvers.py` | Optional pair-confusion postprocessor experiment; not part of the reported accelerator result |

## Deployed Model

```text
input                   1 x 28 x 28
patch embedding         Conv2d(1, 64, kernel=7, stride=7)
patch tokens            16
tokens with CLS         17
embedding dimension     64
attention heads         4
transformer depth       3
MLP hidden dimension    256
classes                 26
parameters              156,122
```

The `7x7` convolution is a non-overlapping patch projection. It is not an additional convolutional stem.

## Quick Check

From the repository root:

```powershell
python -m pip install -r requirements.txt
python pytorch\tiny_vit_mnist.py --dataset emnist-letters --shape-only `
  --embed-dim 64 --num-heads 4 --depth 3 --mlp-dim 256
python tools\evaluate_emnist_checkpoint.py --device cpu
```

The deployed checkpoint is:

```text
checkpoints/tiny_vit_emnist_letters_depth3_mlp256_soup_a086.pt
```

See [`docs/TRAINING_AND_EXPORT.md`](../docs/TRAINING_AND_EXPORT.md) for the full training, distillation, weight-soup, export and board-validation flow.
