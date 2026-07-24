import argparse
import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from tiny_vit_mnist import DATASET_CONFIGS, TinyViT, build_dataset


class PairResolver(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def collect_dataset(dataset):
    images = []
    labels = []
    for image, label in dataset:
        images.append(image.flatten())
        labels.append(int(label))
    return torch.stack(images), torch.tensor(labels)


def train_pair_mlp(images, labels, classes, device, seed, epochs):
    mask = (labels == classes[0]) | (labels == classes[1])
    x = images[mask].to(device)
    y = (labels[mask] == classes[1]).long().to(device)
    torch.manual_seed(seed)
    model = PairResolver().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    for _ in range(epochs):
        order = torch.randperm(len(x), device=device)
        for start in range(0, len(x), 512):
            index = order[start : start + 512]
            logits = model(x[index])
            loss = nn.functional.cross_entropy(logits, y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.cpu()


def kmeans_class(images, count, device, seed, iterations=25):
    x = images.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    centers = x[torch.randperm(len(x), generator=generator, device=device)[:count]].clone()
    for _ in range(iterations):
        assignment = torch.cdist(x, centers).argmin(1)
        sums = torch.zeros_like(centers).index_add_(0, assignment, x)
        counts = torch.bincount(assignment, minlength=count)
        updated = sums / counts.clamp_min(1).float().unsqueeze(1)
        empty = counts == 0
        updated[empty] = centers[empty]
        centers = updated
    return centers.cpu()


def apply_resolvers(images_norm, images_raw, predictions, ij_model, gy_model, uv_centers):
    result = predictions.clone()
    with torch.no_grad():
        for classes, model in [((8, 9), ij_model), ((6, 24), gy_model)]:
            mask = (predictions == classes[0]) | (predictions == classes[1])
            if mask.any():
                choice = model(images_norm[mask]).argmax(1)
                result[mask] = torch.tensor(classes)[choice]

        classes = (20, 21)
        mask = (predictions == classes[0]) | (predictions == classes[1])
        if mask.any():
            nearest = torch.cdist(images_raw[mask], uv_centers).argmin(1)
            result[mask] = (nearest >= (len(uv_centers) // 2)).long() + classes[0]
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--vit-checkpoint",
        default="checkpoints/tiny_vit_emnist_letters_depth3_mlp256_soup_a086.pt",
    )
    parser.add_argument("--out", default="checkpoints/emnist_pair_resolvers.pt")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--uv-prototypes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DATASET_CONFIGS["emnist-letters"]
    raw_transform = transforms.ToTensor()
    train_set = build_dataset(args.data_dir, "emnist-letters", True, raw_transform)
    test_set = build_dataset(args.data_dir, "emnist-letters", False, raw_transform)
    train_raw, train_labels = collect_dataset(train_set)
    test_raw, test_labels = collect_dataset(test_set)
    train_norm = (train_raw - cfg["mean"][0]) / cfg["std"][0]
    test_norm = (test_raw - cfg["mean"][0]) / cfg["std"][0]

    ij_model = train_pair_mlp(
        train_norm, train_labels, (8, 9), device, args.seed + 8, args.epochs
    )
    gy_model = train_pair_mlp(
        train_norm, train_labels, (6, 24), device, args.seed + 6, args.epochs
    )
    uv_centers = torch.cat(
        [
            kmeans_class(
                train_raw[train_labels == label],
                args.uv_prototypes,
                device,
                args.seed + 33,
            )
            for label in (20, 21)
        ]
    )

    checkpoint = torch.load(args.vit_checkpoint, map_location="cpu")
    vit = TinyViT(**checkpoint["config"]).to(device)
    vit.load_state_dict(checkpoint["model"])
    vit.eval()
    base_predictions = []
    with torch.no_grad():
        for start in range(0, len(test_norm), 512):
            batch = test_norm[start : start + 512].reshape(-1, 1, 28, 28).to(device)
            base_predictions.append(vit(batch).argmax(1).cpu())
    base_predictions = torch.cat(base_predictions)
    resolved = apply_resolvers(
        test_norm, test_raw, base_predictions, ij_model, gy_model, uv_centers
    )

    fixed_indices = torch.arange(26) * 800
    base_correct = int((base_predictions == test_labels).sum())
    resolved_correct = int((resolved == test_labels).sum())
    fixed_correct = int((resolved[fixed_indices] == test_labels[fixed_indices]).sum())
    print(f"base_full={base_correct}/{len(test_labels)}")
    print(f"resolved_full={resolved_correct}/{len(test_labels)}")
    print(f"resolved_acc={resolved_correct / len(test_labels):.9f}")
    print(f"resolved_fixed={fixed_correct}/26")
    print(f"fixed_predictions={resolved[fixed_indices].tolist()}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "ij_model": ij_model.state_dict(),
            "gy_model": gy_model.state_dict(),
            "uv_centers": uv_centers,
            "uv_prototypes_per_class": args.uv_prototypes,
            "vit_checkpoint": args.vit_checkpoint,
            "full_correct": resolved_correct,
            "full_count": len(test_labels),
            "fixed_correct": fixed_correct,
            "seed": args.seed,
        },
        args.out,
    )


if __name__ == "__main__":
    main()
