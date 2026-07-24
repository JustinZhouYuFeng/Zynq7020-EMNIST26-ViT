# Training, Export and Board Reproduction

This document describes the exact mainline used by the deployed EMNIST Letters TinyViT. It intentionally excludes ViT-B/ImageNet, convolution-stem and INT8 experiments that were not used by the final `u16 rev25` board image.

## 1. Environment

The current reference rerun used:

```text
Python       3.12.13
PyTorch      2.12.0+cpu
torchvision  0.27.0+cpu
NumPy        2.4.4
Pillow       12.2.0
```

Create an environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

EMNIST is downloaded into `data/` on first use. The directory is intentionally ignored by Git.

## 2. Inspect the Model

```powershell
.\.venv\Scripts\python.exe pytorch\tiny_vit_mnist.py `
  --dataset emnist-letters `
  --shape-only `
  --embed-dim 64 `
  --num-heads 4 `
  --depth 3 `
  --mlp-dim 256
```

Expected shape flow:

```text
input                 [B, 1, 28, 28]
patch embedding       [B, 64, 4, 4]
patch tokens          [B, 16, 64]
CLS + position        [B, 17, 64]
3 encoder layers      [B, 17, 64]
CLS + final norm      [B, 64]
classifier logits     [B, 26]
```

## 3. Evaluate the Committed Final Checkpoint

```powershell
.\.venv\Scripts\python.exe tools\evaluate_emnist_checkpoint.py `
  --checkpoint checkpoints\tiny_vit_emnist_letters_depth3_mlp256_soup_a086.pt `
  --device cpu `
  --out-dir validation_results\reproduced_software
```

Reference CPU result:

```text
test_correct=18777/20800
test_accuracy=90.274038%
```

## 4. Base TinyViT Training

The committed base checkpoint was produced by three stages with decreasing learning rate. Each continuation loads model weights from the previous best checkpoint; optimizer state is not restored.

```powershell
.\.venv\Scripts\python.exe pytorch\tiny_vit_mnist.py `
  --dataset emnist-letters `
  --out checkpoints\tiny_vit_emnist_letters_depth3_mlp256_reproduced.pt `
  --epochs 12 --batch-size 256 --workers 0 `
  --embed-dim 64 --num-heads 4 --depth 3 --mlp-dim 256 `
  --lr 0.0003 --cosine-lr

.\.venv\Scripts\python.exe pytorch\tiny_vit_mnist.py `
  --dataset emnist-letters `
  --out checkpoints\tiny_vit_emnist_letters_depth3_mlp256_reproduced.pt `
  --resume checkpoints\tiny_vit_emnist_letters_depth3_mlp256_reproduced.pt `
  --epochs 8 --batch-size 256 --workers 0 `
  --embed-dim 64 --num-heads 4 --depth 3 --mlp-dim 256 `
  --lr 0.00008 --cosine-lr

.\.venv\Scripts\python.exe pytorch\tiny_vit_mnist.py `
  --dataset emnist-letters `
  --out checkpoints\tiny_vit_emnist_letters_depth3_mlp256_reproduced.pt `
  --resume checkpoints\tiny_vit_emnist_letters_depth3_mlp256_reproduced.pt `
  --epochs 5 --batch-size 256 --workers 0 `
  --embed-dim 64 --num-heads 4 --depth 3 --mlp-dim 256 `
  --lr 0.00003 --cosine-lr
```

The original base checkpoint recorded `89.8846%` on the official test split.

## 5. CNN Teacher and Knowledge Distillation

The teacher network is defined in `pytorch/train_emnist_distill.py`. The committed teacher recorded `95.0144%`.

Train a new teacher:

```powershell
.\.venv\Scripts\python.exe pytorch\train_emnist_distill.py `
  --teacher-only `
  --teacher-epochs 5 `
  --teacher-out checkpoints\emnist_cnn_teacher_reproduced.pt `
  --batch-size 512 --workers 0 --seed 20260620
```

The AccBoost V2 stage starts from the first distilled checkpoint and reuses the committed teacher:

```powershell
.\.venv\Scripts\python.exe pytorch\train_emnist_distill.py `
  --reuse-teacher `
  --teacher-in checkpoints\emnist_cnn_teacher.pt `
  --teacher-out checkpoints\emnist_cnn_teacher.pt `
  --student-in checkpoints\tiny_vit_emnist_letters_depth3_mlp256_distilled.pt `
  --student-out checkpoints\tiny_vit_emnist_letters_depth3_mlp256_accboost_reproduced.pt `
  --student-epochs 10 `
  --student-lr 0.00002 `
  --batch-size 512 --workers 0 `
  --alpha 0.9 `
  --temperature 2.5 `
  --label-smoothing 0.01 `
  --weight-decay 0.005 `
  --seed 20260620
```

For each batch, the student loss is:

```text
hard_loss = cross_entropy(student_logits, labels, label_smoothing=0.01)
soft_loss = KL(student/T, teacher/T) * T^2
loss      = 0.9 * hard_loss + 0.1 * soft_loss
T         = 2.5
```

The teacher runs in evaluation mode under `torch.no_grad()`; only the TinyViT student is updated.

Exact accuracy can vary across CPU/GPU versions because of floating-point reduction order and training nondeterminism. The committed checkpoints are the deployment references.

## 6. Rebuild the Final Weight Soup

The final model is a parameter-level blend, not a runtime ensemble:

```text
theta_final = 0.14 * theta_base + 0.86 * theta_accboost_v2
```

Rebuild it and require exact tensor equality with the deployed checkpoint:

```powershell
.\.venv\Scripts\python.exe tools\blend_emnist_checkpoints.py `
  --base checkpoints\tiny_vit_emnist_letters_depth3_mlp256.pt `
  --boost checkpoints\tiny_vit_emnist_letters_depth3_mlp256_accboost_v2.pt `
  --boost-alpha 0.86 `
  --output checkpoints\tiny_vit_emnist_letters_depth3_mlp256_soup_reproduced.pt `
  --reference checkpoints\tiny_vit_emnist_letters_depth3_mlp256_soup_a086.pt
```

Expected verification:

```text
reference_match=PASS tensors=44 max_abs_diff=0.000000000e+00
```

## 7. Export PyTorch Parameters to the Vitis Header

The final deployment uses floating-point arrays from the generated header. INT8 arrays are also emitted for historical experiments, but the accepted fused path reads `TINYVIT_*_W_ALL` float arrays.

Regenerate the header using the committed 26-sample smoke set:

```powershell
.\.venv\Scripts\python.exe tools\generate_vitis_tinyvit_samples.py `
  --dataset emnist-letters `
  --ckpt checkpoints\tiny_vit_emnist_letters_depth3_mlp256_soup_a086.pt `
  --raw-dir test_data\emnist_letters_udp `
  --count 26 `
  --out validation_results\reproduced_export\tinyvit_samples_vitis.h

.\.venv\Scripts\python.exe tools\compare_vitis_headers.py `
  src\tinyvit_samples_vitis.h `
  validation_results\reproduced_export\tinyvit_samples_vitis.h
```

The script exports:

- model configuration and normalization constants
- Patch Embedding, CLS token and position embedding
- all three layers of LayerNorm, QKV, attention projection and MLP parameters
- final LayerNorm and 64-to-26 classifier head
- PyTorch sample predictions and logits for board comparison

The comparison requires exact equality for all model/configuration arrays and
the deterministic input samples. Sample QKV scales and reference logits can
vary by a few floating-point ULPs across PyTorch/CPU versions, so those three
runtime-derived arrays are checked with explicit tolerances and reported
separately. In the reference rerun, all model arrays were exact, all 26 sample
predictions were unchanged, and the maximum logit difference was `6.676e-6`.

## 8. Recreate the 520-Sample Board Set

Only the deterministic manifest is committed; PNG and raw files are generated from the official dataset:

```powershell
.\.venv\Scripts\python.exe tools\prepare_emnist_stratified_udp.py `
  --checkpoint checkpoints\tiny_vit_emnist_letters_depth3_mlp256_soup_a086.pt `
  --per-class 20 `
  --seed 20260725 `
  --out test_data\emnist_letters_stratified_520
```

Expected PyTorch subset result:

```text
subset_pytorch_correct=472/520
subset_pytorch_accuracy=90.769231%
```

## 9. Program and Validate the Board

Required artifacts are committed at the repository root:

```text
design_1_wrapper_mlp256_u16_rev25.bit
vit_qvk_test_udp_fused_mlp256_u16_rev25_fixedphy.elf
```

After programming the board, run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\verify_udp_emnist_stratified.ps1 `
  -DataDir test_data\emnist_letters_stratified_520 `
  -OutDir validation_results\reproduced_board_520 `
  -BoardIp 192.168.1.10 `
  -LocalIp 192.168.1.100 `
  -SerialPort COM9
```

Committed evidence is under `validation_results/board_520/`.

## 10. Rebuild the HLS IP

With Vitis HLS 2020.1 available:

```powershell
vitis_hls -f hls\run_transformer_layer_fused_mlp256_u16_csim_7020.tcl
vitis_hls -f hls\run_transformer_layer_fused_mlp256_u16_synth_7020.tcl
vitis_hls -f hls\run_transformer_layer_fused_mlp256_u16_export_7020.tcl
```

The final synthesis report is committed as `hls/reports/vit_transformer_layer_fused_csynth.rpt`.

## 11. Evaluation Caveat

The original training scripts evaluated the official test split after each epoch and selected checkpoints using that result. The 14/86 blend ratio also referenced full-test behavior. Therefore 90.27% is a reproducible deployment evaluation on the official test split, not a claim of untouched-holdout generalization or SOTA performance.
