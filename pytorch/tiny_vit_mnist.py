import argparse
import os
import struct

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

DATASET_CONFIGS = {
    "mnist": {
        "dataset": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
        "num_classes": 10,
        "checkpoint": "checkpoints/tiny_vit_mnist.pt",
    },
    "fashion-mnist": {
        "dataset": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "num_classes": 10,
        "checkpoint": "checkpoints/tiny_vit_fashion_mnist.pt",
    },
    "emnist-letters": {
        "dataset": datasets.EMNIST,
        "dataset_kwargs": {"split": "letters", "target_transform": lambda y: y - 1},
        "mean": (0.1736,),
        "std": (0.3317,),
        "num_classes": 26,
        "checkpoint": "checkpoints/tiny_vit_emnist_letters.pt",
    },
}


class RawEMNISTLettersDataset(Dataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        split = "train" if train else "test"
        raw_dir = os.path.join(root, "EMNIST", "raw")
        image_path = os.path.join(raw_dir, f"emnist-letters-{split}-images-idx3-ubyte")
        label_path = os.path.join(raw_dir, f"emnist-letters-{split}-labels-idx1-ubyte")
        if not (os.path.exists(image_path) and os.path.exists(label_path)):
            raise FileNotFoundError("EMNIST Letters IDX files are missing")

        self.images = self._read_images(image_path)
        self.labels = self._read_labels(label_path)
        if len(self.images) != len(self.labels):
            raise ValueError("EMNIST image and label counts do not match")
        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def _read_images(path):
        with open(path, "rb") as f:
            magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"Unexpected image IDX magic {magic} in {path}")
            data = np.frombuffer(f.read(), dtype=np.uint8)
        expected = count * rows * cols
        if data.size != expected:
            raise ValueError(f"Unexpected image IDX length in {path}: {data.size} != {expected}")
        return data.reshape(count, rows, cols)

    @staticmethod
    def _read_labels(path):
        with open(path, "rb") as f:
            magic, count = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Unexpected label IDX magic {magic} in {path}")
            data = np.frombuffer(f.read(), dtype=np.uint8)
        if data.size != count:
            raise ValueError(f"Unexpected label IDX length in {path}: {data.size} != {count}")
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index], mode="L")
        target = int(self.labels[index])
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class TinyViT(nn.Module):
    def __init__(
        self,
        image_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        num_heads=4,
        depth=1,
        mlp_dim=128,
        dropout=0.0,
        pool="cls",
        patch_embed_type="linear",
        stem_dim=32,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        assert embed_dim % num_heads == 0
        if pool not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling mode: {pool}")
        if patch_embed_type not in {"linear", "conv", "conv_deep"}:
            raise ValueError(f"Unsupported patch embedding: {patch_embed_type}")

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.pool = pool
        self.patch_embed_type = patch_embed_type

        if patch_embed_type == "conv_deep":
            self.patch_embed = nn.Sequential(
                nn.Conv2d(in_channels, stem_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stem_dim),
                nn.GELU(),
                nn.Conv2d(
                    stem_dim,
                    embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.Conv2d(
                    embed_dim,
                    embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.Conv2d(
                    embed_dim,
                    embed_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(image_size // patch_size),
            )
        elif patch_embed_type == "conv":
            self.patch_embed = nn.Sequential(
                nn.Conv2d(in_channels, stem_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stem_dim),
                nn.GELU(),
                nn.Conv2d(
                    stem_dim,
                    embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                ),
            )
        else:
            self.patch_embed = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        if self.pool == "mean":
            return self.norm(x[:, 1:].mean(dim=1))
        return self.norm(x[:, 0])

    def forward(self, x):
        return self.head(self.forward_features(x))


def build_dataset(data_dir, dataset_name, train, transform):
    cfg = DATASET_CONFIGS[dataset_name]
    dataset_cls = cfg["dataset"]
    dataset_kwargs = dict(cfg.get("dataset_kwargs", {}))
    if dataset_name == "emnist-letters":
        raw_dir = os.path.join(data_dir, "EMNIST", "raw")
        split = "train" if train else "test"
        image_path = os.path.join(raw_dir, f"emnist-letters-{split}-images-idx3-ubyte")
        label_path = os.path.join(raw_dir, f"emnist-letters-{split}-labels-idx1-ubyte")
        if os.path.exists(image_path) and os.path.exists(label_path):
            return RawEMNISTLettersDataset(
                data_dir,
                train=train,
                transform=transform,
                target_transform=dataset_kwargs.get("target_transform"),
            )
    return dataset_cls(
        data_dir,
        train=train,
        download=True,
        transform=transform,
        **dataset_kwargs,
    )


def build_loaders(data_dir, batch_size, workers, dataset_name, augment=False):
    cfg = DATASET_CONFIGS[dataset_name]
    train_transforms = []
    if augment:
        if dataset_name == "emnist-letters":
            train_transforms.extend(
                [
                    transforms.RandomAffine(
                        degrees=8,
                        translate=(0.06, 0.06),
                        scale=(0.94, 1.06),
                        shear=4,
                    ),
                ]
            )
        else:
            train_transforms.extend(
                [
                    transforms.RandomCrop(28, padding=2),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ]
    )
    train_set = build_dataset(data_dir, dataset_name, True, transforms.Compose(train_transforms))
    test_set = build_dataset(data_dir, dataset_name, False, test_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def accuracy(logits, target):
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch = images.shape[0]
        total_samples += batch
        total_loss += loss.item() * batch
        total_acc += accuracy(logits.detach(), labels) * batch

        if step % 100 == 0:
            print(
                f"epoch {epoch} step {step:04d}: "
                f"loss={total_loss / total_samples:.4f}, "
                f"acc={total_acc / total_samples:.4f}"
            )

    return total_loss / total_samples, total_acc / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch = images.shape[0]
        total_samples += batch
        total_loss += loss.item() * batch
        total_acc += accuracy(logits, labels) * batch

    return total_loss / total_samples, total_acc / total_samples


@torch.no_grad()
def print_shape_trace(model, device):
    model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    print("shape trace")
    print(f"input                 : {tuple(x.shape)}")
    x = model.patch_embed(x)
    print(f"patch_embed           : {tuple(x.shape)}")
    x = x.flatten(2).transpose(1, 2)
    print(f"patch tokens          : {tuple(x.shape)}")
    cls = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    print(f"cls + tokens          : {tuple(x.shape)}")
    x = x + model.pos_embed
    print(f"with position         : {tuple(x.shape)}")
    print(f"qkv linear equivalent : (1, {x.shape[1]}, {3 * model.embed_dim})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default="mnist", choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--mlp-dim", type=int, default=128)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--cosine-lr", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--shape-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out is None:
        args.out = DATASET_CONFIGS[args.dataset]["checkpoint"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    model = TinyViT(
        num_classes=dataset_cfg["num_classes"],
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    ).to(device)
    resume_acc = 0.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        resume_acc = float(checkpoint.get("test_acc") or 0.0)
        print(f"resumed checkpoint    : {args.resume}")
        print(f"resume test_acc       : {resume_acc:.4f}")
    print_shape_trace(model, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters            : {params}")
    print(f"device                : {device}")

    if args.shape_only:
        return

    train_loader, test_loader = build_loaders(
        args.data_dir, args.batch_size, args.workers, args.dataset, args.augment
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.cosine_lr
        else None
    )

    best_acc = resume_acc
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        print(
            f"epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "image_size": 28,
                        "patch_size": 7,
                        "in_channels": 1,
                        "num_classes": dataset_cfg["num_classes"],
                        "embed_dim": args.embed_dim,
                        "num_heads": args.num_heads,
                        "depth": args.depth,
                        "mlp_dim": args.mlp_dim,
                        "dropout": args.dropout,
                    },
                    "test_acc": test_acc,
                    "dataset": args.dataset,
                    "mean": dataset_cfg["mean"],
                    "std": dataset_cfg["std"],
                },
                args.out,
            )
            print(f"saved best checkpoint to {args.out}")


if __name__ == "__main__":
    main()
