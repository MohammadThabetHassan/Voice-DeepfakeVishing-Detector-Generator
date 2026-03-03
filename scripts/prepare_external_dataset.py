#!/usr/bin/env python3
"""
Download and prepare a broader real/fake audio training dataset.

Sources (Hugging Face datasets):
  1) garystafford/deepfake-audio-detection
  2) Bisher/ASVspoof_DF_2021
  3) UniDataPro/real-vs-fake-human-voice-deepfake-audio

Outputs:
  training/data_external/real/*.wav
  training/data_external/fake/*.wav
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass
class LabeledFile:
    src: Path
    label: str
    source: str


def _ffmpeg_convert_to_wav(src: Path, dst: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False


def _snapshot(repo_id: str, patterns: list[str]) -> Path:
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=patterns,
            resume_download=True,
        )
    )


def _collect_gary() -> list[LabeledFile]:
    root = _snapshot(
        "garystafford/deepfake-audio-detection",
        ["real/*.flac", "fake/*.flac", "README.md"],
    )
    out: list[LabeledFile] = []
    for p in (root / "real").glob("*.flac"):
        out.append(LabeledFile(src=p, label="real", source="gary"))
    for p in (root / "fake").glob("*.flac"):
        out.append(LabeledFile(src=p, label="fake", source="gary"))
    return out


def _collect_bisher() -> list[LabeledFile]:
    root = _snapshot(
        "Bisher/ASVspoof_DF_2021",
        ["validation/real/*.flac", "validation/fake/*.flac", "README.md"],
    )
    out: list[LabeledFile] = []
    for p in (root / "validation" / "real").glob("*.flac"):
        out.append(LabeledFile(src=p, label="real", source="bisher"))
    for p in (root / "validation" / "fake").glob("*.flac"):
        out.append(LabeledFile(src=p, label="fake", source="bisher"))
    return out


def _collect_unidatapro() -> list[LabeledFile]:
    root = _snapshot(
        "UniDataPro/real-vs-fake-human-voice-deepfake-audio",
        ["**/*.m4a", "**/*.mp3", "README.md"],
    )
    out: list[LabeledFile] = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in {".m4a", ".mp3"}:
            continue
        name = p.name.lower()
        if name.startswith("original"):
            out.append(LabeledFile(src=p, label="real", source="unidatapro"))
        elif "synthetic" in name:
            out.append(LabeledFile(src=p, label="fake", source="unidatapro"))
    return out


def _limit_per_label(items: list[LabeledFile], max_per_label: int, seed: int) -> list[LabeledFile]:
    random.seed(seed)
    by_label = {"real": [], "fake": []}
    for it in items:
        by_label[it.label].append(it)
    kept: list[LabeledFile] = []
    for label in ("real", "fake"):
        subset = by_label[label]
        random.shuffle(subset)
        if max_per_label > 0 and len(subset) > max_per_label:
            subset = subset[:max_per_label]
        kept.extend(subset)
    random.shuffle(kept)
    return kept


def _convert_all(items: list[LabeledFile], out_dir: Path) -> tuple[int, int]:
    ok = 0
    fail = 0
    counters = {"real": 0, "fake": 0}

    for it in items:
        label_dir = out_dir / it.label
        label_dir.mkdir(parents=True, exist_ok=True)
        counters[it.label] += 1
        idx = counters[it.label]
        dst = label_dir / f"{it.source}_{idx:05d}.wav"
        if _ffmpeg_convert_to_wav(it.src, dst):
            ok += 1
        else:
            fail += 1
            dst.unlink(missing_ok=True)

    return ok, fail


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare external real/fake audio training data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data_external"),
        help="Output directory with real/ and fake/ subfolders",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=1200,
        help="Maximum clips per label (0 = keep all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before writing",
    )
    args = parser.parse_args()

    out_dir = args.output.resolve()
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "fake").mkdir(parents=True, exist_ok=True)

    print("Collecting dataset file manifests...")
    all_items: list[LabeledFile] = []
    all_items.extend(_collect_gary())
    all_items.extend(_collect_bisher())
    all_items.extend(_collect_unidatapro())

    real_count = sum(1 for it in all_items if it.label == "real")
    fake_count = sum(1 for it in all_items if it.label == "fake")
    print(f"Found files before sampling: real={real_count}, fake={fake_count}, total={len(all_items)}")

    items = _limit_per_label(all_items, max_per_label=args.max_per_label, seed=args.seed)
    real_count = sum(1 for it in items if it.label == "real")
    fake_count = sum(1 for it in items if it.label == "fake")
    print(f"After sampling: real={real_count}, fake={fake_count}, total={len(items)}")

    print(f"Converting audio to WAV 16k mono -> {out_dir}")
    ok, fail = _convert_all(items, out_dir)
    print(f"Done: converted={ok}, failed={fail}")

    out_real = len(list((out_dir / "real").glob("*.wav")))
    out_fake = len(list((out_dir / "fake").glob("*.wav")))
    print(f"Final dataset: real={out_real}, fake={out_fake}")

    if out_real == 0 or out_fake == 0:
        print("ERROR: missing one class after conversion.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
