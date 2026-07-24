import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch"))

from tiny_vit_mnist import DATASET_CONFIGS, TinyViT, build_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the deployed EMNIST Letters checkpoint on the full test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "tiny_vit_emnist_letters_depth3_mlp256_soup_a086.pt",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation_results" / "software")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = TinyViT(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    cfg = DATASET_CONFIGS["emnist-letters"]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])]
    )
    dataset = build_dataset(str(args.data_dir), "emnist-letters", False, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = int(checkpoint["config"]["num_classes"])
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    rows = []
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            labels = labels.cpu()
            for index, (label, prediction) in enumerate(zip(labels.tolist(), predictions.tolist())):
                confusion[label, prediction] += 1
                rows.append(
                    {
                        "source_index": offset + index,
                        "label": label,
                        "prediction": prediction,
                        "correct": int(label == prediction),
                    }
                )
            offset += len(labels)

    correct = int(confusion.diag().sum())
    total = int(confusion.sum())
    accuracy = correct / total
    per_class = []
    for label in range(num_classes):
        class_total = int(confusion[label].sum())
        class_correct = int(confusion[label, label])
        per_class.append(
            {
                "class_index": label,
                "class_name": chr(ord("A") + label),
                "correct": class_correct,
                "total": class_total,
                "accuracy": class_correct / class_total if class_total else 0.0,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "predictions.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "confusion_matrix.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + [chr(ord("A") + i) for i in range(num_classes)])
        for label in range(num_classes):
            writer.writerow([chr(ord("A") + label)] + confusion[label].tolist())

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "checkpoint_recorded_test_acc": checkpoint.get("test_acc"),
        "config": checkpoint["config"],
        "per_class": per_class,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="ascii"
    )

    print(f"checkpoint={args.checkpoint}")
    print(f"device={device}")
    print(f"test_correct={correct}/{total}")
    print(f"test_accuracy={accuracy:.6%}")
    print("per_class=" + " ".join(f"{x['class_name']}:{x['accuracy']:.2%}" for x in per_class))
    print(f"results={args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
