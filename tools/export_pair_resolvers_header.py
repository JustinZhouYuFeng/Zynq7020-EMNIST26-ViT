import argparse
import os

import numpy as np
import torch


def write_float_array(handle, name, array):
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    handle.write(f"static const float {name}[{len(flat)}] = {{\n")
    for start in range(0, len(flat), 8):
        values = ", ".join(f"{float(v):.9e}f" for v in flat[start : start + 8])
        suffix = "," if start + 8 < len(flat) else ""
        handle.write(f"    {values}{suffix}\n")
    handle.write("};\n\n")


def write_mlp(handle, prefix, state):
    write_float_array(handle, f"{prefix}_W1", state["net.0.weight"].numpy())
    write_float_array(handle, f"{prefix}_B1", state["net.0.bias"].numpy())
    write_float_array(handle, f"{prefix}_W2", state["net.2.weight"].numpy())
    write_float_array(handle, f"{prefix}_B2", state["net.2.bias"].numpy())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/emnist_pair_resolvers_v2.pt")
    parser.add_argument(
        "--out", default="vitis_ws/vit_qvk_test/src/emnist_pair_resolvers_vitis.h"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    centers = checkpoint["uv_centers"].numpy()
    if centers.shape != (64, 784):
        raise ValueError(f"expected UV centers (64, 784), got {centers.shape}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="ascii") as handle:
        handle.write("#ifndef EMNIST_PAIR_RESOLVERS_VITIS_H\n")
        handle.write("#define EMNIST_PAIR_RESOLVERS_VITIS_H\n\n")
        handle.write("#define EMNIST_RESOLVER_INPUTS 784\n")
        handle.write("#define EMNIST_RESOLVER_HIDDEN 64\n")
        handle.write("#define EMNIST_UV_PROTOTYPES_PER_CLASS 32\n\n")
        write_mlp(handle, "EMNIST_IJ", checkpoint["ij_model"])
        write_mlp(handle, "EMNIST_GY", checkpoint["gy_model"])
        write_float_array(handle, "EMNIST_UV_CENTERS", centers)
        handle.write("#endif\n")

    print(f"generated {args.out}")
    print(f"fixed_correct={checkpoint['fixed_correct']}/26")
    print(f"full_correct={checkpoint['full_correct']}/{checkpoint['full_count']}")


if __name__ == "__main__":
    main()
