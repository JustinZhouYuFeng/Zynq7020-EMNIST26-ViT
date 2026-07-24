import argparse
import csv
import os
import sys

import numpy as np
import torch
from torchvision import transforms


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTORCH_DIR = os.path.join(PROJECT_ROOT, "pytorch")
OUT_PATH = os.path.join(
    PROJECT_ROOT, "src", "tinyvit_samples_vitis.h"
)

sys.path.insert(0, PYTORCH_DIR)
from tiny_vit_mnist import DATASET_CONFIGS, TinyViT, build_dataset  # noqa: E402


def symmetric_int8_quantize(array):
    max_abs = float(np.max(np.abs(array)))
    scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
    quantized = np.round(array / scale).clip(-128, 127).astype(np.int8)
    return quantized, scale


def write_1d_float(f, name, array):
    flat = array.reshape(-1)
    f.write(f"static const float {name}[{flat.shape[0]}] = {{\n")
    for start in range(0, flat.shape[0], 8):
        values = ", ".join(f"{float(v):.9e}f" for v in flat[start : start + 8])
        comma = "," if start + 8 < flat.shape[0] else ""
        f.write(f"    {values}{comma}\n")
    f.write("};\n\n")


def write_1d_i32(f, name, array):
    flat = array.reshape(-1)
    f.write(f"static const int {name}[{flat.shape[0]}] = {{\n")
    for start in range(0, flat.shape[0], 16):
        values = ", ".join(str(int(v)) for v in flat[start : start + 16])
        comma = "," if start + 16 < flat.shape[0] else ""
        f.write(f"    {values}{comma}\n")
    f.write("};\n\n")


def write_1d_i8(f, name, array):
    flat = array.reshape(-1).astype(np.int8)
    f.write(f"static const signed char {name}[{flat.shape[0]}] = {{\n")
    for start in range(0, flat.shape[0], 24):
        values = ", ".join(str(int(v)) for v in flat[start : start + 24])
        comma = "," if start + 24 < flat.shape[0] else ""
        f.write(f"    {values}{comma}\n")
    f.write("};\n\n")


def write_layer_float_array(f, base_name, layers, attr):
    values = [getattr(layer, attr).detach().cpu().numpy() for layer in layers]
    write_1d_float(f, f"{base_name}_ALL", np.stack(values).astype(np.float32))


def quantize_per_layer(arrays, transpose=None):
    q_arrays = []
    scales = []
    for array in arrays:
        value = array.detach().cpu().numpy()
        if transpose is not None:
            value = np.transpose(value, transpose).copy()
        q, scale = symmetric_int8_quantize(value)
        q_arrays.append(q)
        scales.append(scale)
    return np.stack(q_arrays), np.array(scales, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--raw-dir", default=None)
    return parser.parse_args()


def select_sample_indices(dataset, count, num_classes, balanced):
    if not balanced:
        return list(range(min(count, len(dataset))))

    selected = []
    seen = set()
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if label not in seen:
            selected.append(idx)
            seen.add(label)
            if len(selected) >= min(count, num_classes):
                break

    if len(selected) < count:
        used = set(selected)
        for idx in range(len(dataset)):
            if idx not in used:
                selected.append(idx)
                if len(selected) >= count:
                    break

    return selected[:count]


def load_raw_samples(raw_dir, count, dataset_cfg):
    manifest_path = os.path.join(raw_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"missing raw sample manifest: {manifest_path}")

    samples = []
    mean = float(dataset_cfg["mean"][0])
    std = float(dataset_cfg["std"][0])
    with open(manifest_path, newline="", encoding="ascii") as f:
        for row in csv.DictReader(f):
            if len(samples) >= count:
                break
            raw_path = os.path.join(raw_dir, row["raw"])
            payload = np.fromfile(raw_path, dtype=np.uint8)
            if payload.size != 28 * 28:
                raise ValueError(f"expected 784 bytes, got {payload.size}: {raw_path}")
            image = torch.from_numpy(payload.astype(np.float32).reshape(1, 28, 28) / 255.0)
            image = (image - mean) / std
            samples.append((image, int(row["label"])))
    return samples


@torch.no_grad()
def main():
    args = parse_args()
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    ckpt_path = args.ckpt or os.path.join(PROJECT_ROOT, dataset_cfg["checkpoint"])

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    dataset_name = checkpoint.get("dataset", args.dataset)
    dataset_cfg = DATASET_CONFIGS[dataset_name]
    model = TinyViT(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.raw_dir:
        test_samples = load_raw_samples(args.raw_dir, args.count, dataset_cfg)
        sample_indices = list(range(len(test_samples)))
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(dataset_cfg["mean"], dataset_cfg["std"]),
            ]
        )
        test_set = build_dataset(args.data_dir, dataset_name, False, transform)
        sample_indices = select_sample_indices(
            test_set,
            args.count,
            int(checkpoint["config"]["num_classes"]),
            args.balanced,
        )

    layers = list(model.encoder.layers)
    layer0 = layers[0]
    qkv_weight_hls = layer0.self_attn.in_proj_weight.detach().cpu().numpy().T.copy()
    _, weight_scale = symmetric_int8_quantize(qkv_weight_hls)

    images = []
    labels = []
    pytorch_preds = []
    qkv_x_scales = []
    qkv_bias_i32 = []
    pytorch_logits = []

    for sample_index in sample_indices:
        if args.raw_dir:
            image, label = test_samples[sample_index]
        else:
            image, label = test_set[sample_index]
        logits = model(image.unsqueeze(0))[0].detach().cpu().numpy()
        pred = int(np.argmax(logits))

        x = model.patch_embed(image.unsqueeze(0))
        x = x.flatten(2).transpose(1, 2)
        cls = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + model.pos_embed
        qkv_input = layer0.norm1(x).squeeze(0).cpu().numpy()
        _, x_scale = symmetric_int8_quantize(qkv_input)
        bias_i32 = np.round(
            layer0.self_attn.in_proj_bias.detach().cpu().numpy() / (x_scale * weight_scale)
        ).astype(np.int32)

        images.append(image.cpu().numpy().reshape(-1))
        labels.append(int(label))
        pytorch_preds.append(pred)
        qkv_x_scales.append(x_scale)
        qkv_bias_i32.append(bias_i32)
        pytorch_logits.append(logits)

    images = np.stack(images)
    qkv_bias_i32 = np.stack(qkv_bias_i32)
    pytorch_logits = np.stack(pytorch_logits)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("#ifndef TINYVIT_SAMPLES_VITIS_H\n")
        f.write("#define TINYVIT_SAMPLES_VITIS_H\n\n")
        f.write("// Generated by tools/generate_vitis_tinyvit_samples.py.\n")
        f.write(f"// Dataset: {dataset_name}\n")
        f.write(f"// Checkpoint: {os.path.relpath(ckpt_path, PROJECT_ROOT)}\n")
        f.write(f"#define TINYVIT_DATASET_NAME \"{dataset_name}\"\n")
        f.write(f"#define TINYVIT_NUM_CLASSES {int(checkpoint['config']['num_classes'])}\n")
        f.write(f"#define TINYVIT_EMBED_DIM {int(checkpoint['config']['embed_dim'])}\n")
        f.write(f"#define TINYVIT_NUM_HEADS {int(checkpoint['config']['num_heads'])}\n")
        f.write(f"#define TINYVIT_DEPTH {int(checkpoint['config']['depth'])}\n")
        f.write(f"#define TINYVIT_MLP_DIM {int(checkpoint['config']['mlp_dim'])}\n")
        f.write(f"#define TINYVIT_INPUT_MEAN {float(dataset_cfg['mean'][0]):.9e}f\n")
        f.write(f"#define TINYVIT_INPUT_STD {float(dataset_cfg['std'][0]):.9e}f\n")
        f.write(f"#define TINYVIT_SAMPLE_COUNT {len(sample_indices)}\n")
        f.write(f"#define TINYVIT_QKV_W_SCALE {weight_scale:.12e}f\n\n")
        write_1d_i32(f, "TINYVIT_SAMPLE_LABELS", np.array(labels, dtype=np.int32))
        write_1d_i32(f, "TINYVIT_SAMPLE_PYTORCH_PREDS", np.array(pytorch_preds, dtype=np.int32))
        write_1d_float(f, "TINYVIT_SAMPLE_QKV_X_SCALES", np.array(qkv_x_scales, dtype=np.float32))
        write_1d_float(f, "TINYVIT_SAMPLE_IMAGES", images.astype(np.float32))
        write_1d_i32(f, "TINYVIT_SAMPLE_QKV_BIAS_I32", qkv_bias_i32)
        write_1d_float(f, "TINYVIT_SAMPLE_PYTORCH_LOGITS", pytorch_logits.astype(np.float32))

        write_1d_float(f, "TINYVIT_PATCH_W", model.patch_embed.weight.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_PATCH_B", model.patch_embed.bias.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_CLS", model.cls_token.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_POS", model.pos_embed.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_QKV_W_ALL", np.stack([
            l.self_attn.in_proj_weight.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        write_1d_float(f, "TINYVIT_QKV_B_ALL", np.stack([
            l.self_attn.in_proj_bias.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        write_1d_float(f, "TINYVIT_NORM1_W_ALL", np.stack([
            l.norm1.weight.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        write_1d_float(f, "TINYVIT_NORM1_B_ALL", np.stack([
            l.norm1.bias.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        write_1d_float(f, "TINYVIT_ATTN_OUT_W_ALL", np.stack([
            l.self_attn.out_proj.weight.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        write_1d_float(f, "TINYVIT_ATTN_OUT_B_ALL", np.stack([
            l.self_attn.out_proj.bias.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        write_1d_float(f, "TINYVIT_NORM2_W_ALL", np.stack([
            l.norm2.weight.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        write_1d_float(f, "TINYVIT_NORM2_B_ALL", np.stack([
            l.norm2.bias.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        write_1d_float(f, "TINYVIT_FC1_W_ALL", np.stack([
            l.linear1.weight.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        write_1d_float(f, "TINYVIT_FC1_B_ALL", np.stack([
            l.linear1.bias.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        write_1d_float(f, "TINYVIT_FC2_W_ALL", np.stack([
            l.linear2.weight.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        write_1d_float(f, "TINYVIT_FC2_B_ALL", np.stack([
            l.linear2.bias.detach().cpu().numpy() for l in layers
        ]).astype(np.float32))
        qkv_i8_all, qkv_scale_all = quantize_per_layer(
            [l.self_attn.in_proj_weight for l in layers],
            transpose=(0, 1),
        )
        # QKV is stored as [layer][q][embed]; firmware indexes q * EMBED + e.
        write_1d_i8(f, "TINYVIT_QKV_W_I8_ALL", qkv_i8_all)
        write_1d_float(f, "TINYVIT_QKV_W_SCALE_ALL", qkv_scale_all)

        attn_i8_all, attn_scale_all = quantize_per_layer(
            [l.self_attn.out_proj.weight for l in layers]
        )
        write_1d_i8(f, "TINYVIT_ATTN_OUT_W_I8_ALL", attn_i8_all)
        write_1d_float(f, "TINYVIT_ATTN_OUT_W_SCALE_ALL", attn_scale_all)

        fc1_i8_all, fc1_scale_all = quantize_per_layer([l.linear1.weight for l in layers])
        write_1d_i8(f, "TINYVIT_FC1_W_I8_ALL", fc1_i8_all)
        write_1d_float(f, "TINYVIT_FC1_W_SCALE_ALL", fc1_scale_all)

        fc2_i8_all, fc2_scale_all = quantize_per_layer([l.linear2.weight for l in layers])
        write_1d_i8(f, "TINYVIT_FC2_W_I8_ALL", fc2_i8_all)
        write_1d_float(f, "TINYVIT_FC2_W_SCALE_ALL", fc2_scale_all)

        head_w = model.head.weight.detach().cpu().numpy()
        head_i8, head_scale = symmetric_int8_quantize(head_w)
        write_1d_i8(f, "TINYVIT_HEAD_W_I8", head_i8)
        write_1d_float(f, "TINYVIT_HEAD_W_SCALE", np.array([head_scale], dtype=np.float32))
        write_1d_float(f, "TINYVIT_NORM1_W", layer0.norm1.weight.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_NORM1_B", layer0.norm1.bias.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_ATTN_OUT_W", layer0.self_attn.out_proj.weight.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_ATTN_OUT_B", layer0.self_attn.out_proj.bias.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_NORM2_W", layer0.norm2.weight.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_NORM2_B", layer0.norm2.bias.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_FC1_W", layer0.linear1.weight.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_FC1_B", layer0.linear1.bias.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_FC2_W", layer0.linear2.weight.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_FC2_B", layer0.linear2.bias.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_FINAL_NORM_W", model.norm.weight.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_FINAL_NORM_B", model.norm.bias.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_HEAD_W", model.head.weight.detach().cpu().numpy())
        write_1d_float(f, "TINYVIT_HEAD_B", model.head.bias.detach().cpu().numpy())
        f.write("#endif\n")

    print(f"generated {args.out}")
    print(f"dataset: {dataset_name}")
    for i, (label, pred, scale) in enumerate(zip(labels, pytorch_preds, qkv_x_scales)):
        print(f"sample {i}: label={label} pytorch_pred={pred} qkv_x_scale={scale:.12e}")


if __name__ == "__main__":
    main()
