#!/usr/bin/env python3
"""Compare generated TinyViT Vitis headers without hiding model changes."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


DEFINE_RE = re.compile(r"^#define\s+(TINYVIT_[A-Z0-9_]+)\s+(.+?)\s*$", re.MULTILINE)
ARRAY_RE = re.compile(
    r"static\s+const\s+(?P<ctype>signed char|float|int)\s+"
    r"(?P<name>TINYVIT_[A-Z0-9_]+)\[(?P<size>\d+)\]\s*=\s*\{"
    r"(?P<body>.*?)\};",
    re.DOTALL,
)

# These values depend on sample inference reductions in the installed PyTorch
# runtime. All model/configuration arrays and the source samples remain exact.
RUNTIME_DERIVED_ARRAYS = {
    "TINYVIT_SAMPLE_QKV_X_SCALES",
    "TINYVIT_SAMPLE_QKV_BIAS_I32",
    "TINYVIT_SAMPLE_PYTORCH_LOGITS",
}


@dataclass(frozen=True)
class ArrayData:
    ctype: str
    size: int
    tokens: tuple[str, ...]


def parse_header(path: Path) -> tuple[dict[str, str], dict[str, ArrayData]]:
    text = path.read_text(encoding="utf-8")
    defines = {name: value.strip() for name, value in DEFINE_RE.findall(text)}
    arrays: dict[str, ArrayData] = {}

    for match in ARRAY_RE.finditer(text):
        name = match.group("name")
        size = int(match.group("size"))
        tokens = tuple(
            token.strip()
            for token in match.group("body").split(",")
            if token.strip()
        )
        if len(tokens) != size:
            raise ValueError(
                f"{path}: {name} declares {size} values but contains {len(tokens)}"
            )
        if name in arrays:
            raise ValueError(f"{path}: duplicate array {name}")
        arrays[name] = ArrayData(match.group("ctype"), size, tokens)

    if not defines or not arrays:
        raise ValueError(f"{path}: no TinyViT definitions or arrays found")
    return defines, arrays


def as_number(token: str, ctype: str) -> float | int:
    if ctype == "float":
        return float(token[:-1] if token.endswith("f") else token)
    return int(token, 0)


def compare_runtime_array(
    name: str,
    expected: ArrayData,
    actual: ArrayData,
    float_atol: float,
    float_rtol: float,
    integer_atol: int,
) -> tuple[bool, str]:
    expected_values = [as_number(token, expected.ctype) for token in expected.tokens]
    actual_values = [as_number(token, actual.ctype) for token in actual.tokens]

    if expected.ctype == "float":
        differences = [abs(float(a) - float(b)) for a, b in zip(expected_values, actual_values)]
        relative = [
            difference / max(abs(float(a)), abs(float(b)), 1.0e-30)
            for a, b, difference in zip(expected_values, actual_values, differences)
        ]
        changed = sum(a != b for a, b in zip(expected.tokens, actual.tokens))
        close = all(
            math.isclose(float(a), float(b), rel_tol=float_rtol, abs_tol=float_atol)
            for a, b in zip(expected_values, actual_values)
        )
        return close, (
            f"changed={changed}/{expected.size} "
            f"max_abs_diff={max(differences, default=0.0):.9e} "
            f"max_rel_diff={max(relative, default=0.0):.9e}"
        )

    differences = [abs(int(a) - int(b)) for a, b in zip(expected_values, actual_values)]
    changed = sum(difference != 0 for difference in differences)
    maximum = max(differences, default=0)
    return maximum <= integer_atol, (
        f"changed={changed}/{expected.size} max_abs_diff={maximum}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Require exact TinyViT model/config/sample inputs while allowing "
            "bounded runtime variation in sample-derived reference values."
        )
    )
    parser.add_argument("expected", type=Path, help="Committed reference header")
    parser.add_argument("actual", type=Path, help="Newly generated header")
    parser.add_argument("--float-atol", type=float, default=1.0e-5)
    parser.add_argument("--float-rtol", type=float, default=1.0e-5)
    parser.add_argument("--integer-atol", type=int, default=1)
    args = parser.parse_args()

    expected_defines, expected_arrays = parse_header(args.expected)
    actual_defines, actual_arrays = parse_header(args.actual)
    failures: list[str] = []

    if expected_defines != actual_defines:
        names = sorted(set(expected_defines) | set(actual_defines))
        failures.extend(
            f"define mismatch: {name}"
            for name in names
            if expected_defines.get(name) != actual_defines.get(name)
        )

    if expected_arrays.keys() != actual_arrays.keys():
        missing = sorted(expected_arrays.keys() - actual_arrays.keys())
        extra = sorted(actual_arrays.keys() - expected_arrays.keys())
        failures.append(f"array set mismatch: missing={missing} extra={extra}")

    exact_count = 0
    print("runtime-derived arrays:")
    for name in sorted(expected_arrays.keys() & actual_arrays.keys()):
        expected = expected_arrays[name]
        actual = actual_arrays[name]
        if expected.ctype != actual.ctype or expected.size != actual.size:
            failures.append(
                f"array declaration mismatch: {name} "
                f"expected={expected.ctype}[{expected.size}] "
                f"actual={actual.ctype}[{actual.size}]"
            )
            continue

        if name in RUNTIME_DERIVED_ARRAYS:
            passed, details = compare_runtime_array(
                name,
                expected,
                actual,
                args.float_atol,
                args.float_rtol,
                args.integer_atol,
            )
            print(f"  {name}: {'PASS' if passed else 'FAIL'} {details}")
            if not passed:
                failures.append(f"runtime tolerance exceeded: {name} ({details})")
        elif expected.tokens != actual.tokens:
            changed = sum(a != b for a, b in zip(expected.tokens, actual.tokens))
            failures.append(f"exact array mismatch: {name} changed={changed}/{expected.size}")
        else:
            exact_count += 1

    print(f"exact_defines={len(expected_defines)}")
    print(f"exact_arrays={exact_count}")
    print(f"runtime_arrays={len(RUNTIME_DERIVED_ARRAYS)}")
    if failures:
        print("header_match=FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("header_match=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
