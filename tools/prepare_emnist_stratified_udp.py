import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch"))

from tiny_vit_mnist import DATASET_CONFIGS, TinyViT, build_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a deterministic class-balanced EMNIST Letters board-validation set."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "tiny_vit_emnist_letters_depth3_mlp256_soup_a086.pt",
    )
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "test_data" / "emnist_letters_stratified_520",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = TinyViT(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(str(args.data_dir), "emnist-letters", False, transform=None)
    by_class = [[] for _ in range(26)]
    for index in range(len(dataset)):
        _, label = dataset[index]
        by_class[int(label)].append(index)

    rng = random.Random(args.seed)
    selected = []
    for label, indices in enumerate(by_class):
        if len(indices) < args.per_class:
            raise ValueError(f"class {label} only has {len(indices)} samples")
        selected.extend((label, index) for index in rng.sample(indices, args.per_class))
    selected.sort(key=lambda item: (item[0], item[1]))

    cfg = DATASET_CONFIGS["emnist-letters"]
    mean = float(cfg["mean"][0])
    std = float(cfg["std"][0])
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample, (label, source_index) in enumerate(selected):
        image, actual_label = dataset[source_index]
        actual_label = int(actual_label)
        if actual_label != label:
            raise RuntimeError(f"label mismatch at source index {source_index}")
        image = image.convert("L").resize((28, 28))
        raw = np.asarray(image, dtype=np.uint8)
        tensor = torch.from_numpy(raw.astype(np.float32) / 255.0).reshape(1, 1, 28, 28)
        tensor = (tensor - mean) / std
        with torch.inference_mode():
            pytorch_pred = int(model(tensor).argmax(dim=1).item())

        stem = f"emnist_letters_{sample:04d}_src{source_index:05d}_label{label:02d}"
        png_name = stem + ".png"
        raw_name = stem + ".raw"
        image.save(args.out / png_name)
        (args.out / raw_name).write_bytes(raw.tobytes())
        rows.append(
            {
                "sample": sample,
                "source_index": source_index,
                "label": label,
                "pytorch_pred": pytorch_pred,
                "png": png_name,
                "raw": raw_name,
            }
        )

    with (args.out / "manifest.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    correct = sum(row["label"] == row["pytorch_pred"] for row in rows)
    print(f"samples={len(rows)} per_class={args.per_class} seed={args.seed}")
    print(f"subset_pytorch_correct={correct}/{len(rows)}")
    print(f"subset_pytorch_accuracy={correct / len(rows):.6%}")
    print(f"manifest={(args.out / 'manifest.csv').resolve()}")


if __name__ == "__main__":
    main()
