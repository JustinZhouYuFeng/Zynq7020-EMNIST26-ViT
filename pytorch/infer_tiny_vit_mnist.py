import argparse

import torch
from torchvision import transforms

from tiny_vit_mnist import DATASET_CONFIGS, TinyViT, build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.dataset is None and args.ckpt is None:
        args.dataset = "mnist"
    if args.dataset is not None and args.ckpt is None:
        args.ckpt = DATASET_CONFIGS[args.dataset]["checkpoint"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.ckpt, map_location=device)
    dataset_name = args.dataset or checkpoint.get("dataset", "mnist")
    dataset_cfg = DATASET_CONFIGS[dataset_name]
    model = TinyViT(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(dataset_cfg["mean"], dataset_cfg["std"]),
        ]
    )
    test_set = build_dataset(args.data_dir, dataset_name, False, transform)
    image, label = test_set[args.index]
    logits = model(image.unsqueeze(0).to(device))
    prob = torch.softmax(logits, dim=1)
    pred = int(prob.argmax(dim=1).item())

    print(f"sample index : {args.index}")
    print(f"dataset      : {dataset_name}")
    print(f"label        : {label}")
    print(f"prediction   : {pred}")
    print(f"confidence   : {float(prob[0, pred]):.4f}")


if __name__ == "__main__":
    main()
