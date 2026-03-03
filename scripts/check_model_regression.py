#!/usr/bin/env python3
"""Fail CI when model metrics regress beyond configured thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check model regression thresholds")
    p.add_argument("--results", type=Path, required=True, help="Path to results.json")
    p.add_argument(
        "--required-model",
        action="append",
        default=[],
        help="Model key required in results.json (can be passed multiple times)",
    )
    p.add_argument(
        "--check-model",
        action="append",
        default=[],
        help="Only evaluate thresholds for these model keys (default: all keys)",
    )
    p.add_argument("--min-f1", type=float, default=0.8, help="Minimum allowed F1")
    p.add_argument("--max-fpr", type=float, default=0.2, help="Maximum allowed false-positive rate")
    p.add_argument(
        "--min-ood-f1",
        type=float,
        default=0.0,
        help="Minimum allowed OOD F1 when ood_f1 is present (0 disables)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.results.exists():
        raise SystemExit(f"results file not found: {args.results}")

    with args.results.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise SystemExit("results.json is empty or invalid")

    failures: list[str] = []

    for key in args.required_model:
        if key not in data:
            failures.append(f"missing required model: {key}")

    keys_to_check = args.check_model if args.check_model else list(data.keys())
    for key in keys_to_check:
        metrics = data.get(key)
        if metrics is None:
            failures.append(f"missing check-model: {key}")
            continue
        if not isinstance(metrics, dict):
            failures.append(f"{key}: metrics must be an object")
            continue

        f1 = metrics.get("f1")
        cm = metrics.get("confusion_matrix")
        if f1 is None:
            failures.append(f"{key}: missing f1")
            continue
        if float(f1) < args.min_f1:
            failures.append(f"{key}: f1={float(f1):.4f} < min_f1={args.min_f1:.4f}")

        if isinstance(cm, list) and len(cm) == 2 and all(isinstance(r, list) and len(r) == 2 for r in cm):
            tn, fp = cm[0]
            denom = tn + fp
            fpr = float(fp) / float(denom) if denom else 0.0
            if fpr > args.max_fpr:
                failures.append(f"{key}: fpr={fpr:.4f} > max_fpr={args.max_fpr:.4f}")

        ood_f1 = metrics.get("ood_f1")
        if ood_f1 is not None and args.min_ood_f1 > 0 and float(ood_f1) < args.min_ood_f1:
            failures.append(
                f"{key}: ood_f1={float(ood_f1):.4f} < min_ood_f1={args.min_ood_f1:.4f}"
            )

    if failures:
        print("Model regression check FAILED:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("Model regression check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
