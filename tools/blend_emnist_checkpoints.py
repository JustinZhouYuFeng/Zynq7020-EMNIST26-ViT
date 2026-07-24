import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Blend the deployed EMNIST TinyViT checkpoints parameter by parameter."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "checkpoints" / "tiny_vit_emnist_letters_depth3_mlp256.pt",
    )
    parser.add_argument(
        "--boost",
        type=Path,
        default=(
            ROOT
            / "checkpoints"
            / "tiny_vit_emnist_letters_depth3_mlp256_accboost_v2.pt"
        ),
    )
    parser.add_argument("--boost-alpha", type=float, default=0.86)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "checkpoints"
            / "tiny_vit_emnist_letters_depth3_mlp256_soup_reproduced.pt"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional checkpoint whose model tensors must exactly match the blend.",
    )
    return parser.parse_args()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError(f"checkpoint is missing model/config: {path}")
    return checkpoint


def blend_state_dict(base_state, boost_state, boost_alpha):
    if base_state.keys() != boost_state.keys():
        missing = sorted(set(base_state) ^ set(boost_state))
        raise ValueError(f"state_dict keys do not match: {missing}")

    base_alpha = 1.0 - boost_alpha
    blended = {}
    for name, base_value in base_state.items():
        boost_value = boost_state[name]
        if base_value.shape != boost_value.shape or base_value.dtype != boost_value.dtype:
            raise ValueError(f"tensor metadata does not match for {name}")
        if torch.is_floating_point(base_value):
            # This operation order exactly matches the deployed soup checkpoint.
            blended[name] = base_value * base_alpha + boost_value * boost_alpha
        else:
            if not torch.equal(base_value, boost_value):
                raise ValueError(f"non-floating tensor differs for {name}")
            blended[name] = base_value.clone()
    return blended


def verify_reference(blended, reference_path):
    reference = load_checkpoint(reference_path)["model"]
    if blended.keys() != reference.keys():
        raise ValueError("reference state_dict keys do not match")

    mismatched = []
    max_abs_diff = 0.0
    for name, value in blended.items():
        expected = reference[name]
        if torch.is_floating_point(value):
            max_abs_diff = max(max_abs_diff, float((value - expected).abs().max()))
        if not torch.equal(value, expected):
            mismatched.append(name)
    if mismatched:
        raise ValueError(
            f"reference mismatch in {len(mismatched)} tensors; "
            f"max_abs_diff={max_abs_diff:.9e}; first={mismatched[0]}"
        )
    print(
        f"reference_match=PASS tensors={len(blended)} "
        f"max_abs_diff={max_abs_diff:.9e}"
    )


def main():
    args = parse_args()
    if not 0.0 <= args.boost_alpha <= 1.0:
        raise ValueError("--boost-alpha must be in [0, 1]")

    base = load_checkpoint(args.base)
    boost = load_checkpoint(args.boost)
    if base["config"] != boost["config"]:
        raise ValueError("base and boost model configs do not match")

    blended = blend_state_dict(base["model"], boost["model"], args.boost_alpha)
    if args.reference is not None:
        verify_reference(blended, args.reference)

    output = {key: value for key, value in boost.items() if key != "model"}
    output["model"] = blended
    output["test_acc"] = None
    output["weight_soup"] = {
        "base": str(args.base),
        "boost": str(args.boost),
        "boost_alpha": args.boost_alpha,
        "note": "Evaluation metrics must be recomputed for this generated checkpoint.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"output={args.output.resolve()}")
    print(f"base_alpha={1.0 - args.boost_alpha:.6f}")
    print(f"boost_alpha={args.boost_alpha:.6f}")


if __name__ == "__main__":
    main()
