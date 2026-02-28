#!/usr/bin/env python3
"""
Training pipeline for Voice Deepfake Detection.

Trains and evaluates THREE model variants:
  1. MFCC-only     (13-dim MFCCs averaged over frames)
  2. FFT/Spectral  (6-dim: centroid, bandwidth, rolloff, band energies)
  3. Hybrid        (19-dim: MFCC + FFT features)

Usage:
    python training/train.py --data training/data/ --output models/
    python training/train.py --csv osr_features.csv --output models/

Input data options:
  A) --csv   : CSV with columns matching feature type + 'label' column
  B) --data  : Directory of WAV files:
                  data/real/   → real voice samples
                  data/fake/   → synthetic/deepfake samples

Outputs:
  models/deepfake_detector_mfcc.pkl
  models/deepfake_detector_fft.pkl
  models/deepfake_detector_hybrid.pkl
  models/deepfake_detector_best.pkl  (copy of best model)
  models/results.json                (metrics comparison table)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal as scipy_signal
import scipy.fftpack as fftpack
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─── feature extraction ───────────────────────────────────────────────────────

def load_audio_mono16k(path: Path):
    """Load WAV, convert to mono float32 @ 16 kHz."""
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    mx = np.max(np.abs(data))
    if mx > 0:
        data /= mx
    if sr != 16000:
        new_len = int(len(data) * 16000 / sr)
        data = scipy_signal.resample(data, new_len)
        sr = 16000
    return data, sr


def extract_mfcc(segment: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Return mean MFCC vector (n_mfcc dims)."""
    pre_emphasis = 0.97
    emph = np.append(segment[0], segment[1:] - pre_emphasis * segment[:-1])

    frame_len = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    n = len(emph)
    n_frames = max(1, int(np.ceil((n - frame_len) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_len
    padded = np.append(emph, np.zeros(max(0, pad_len - n)))

    idx = (
        np.tile(np.arange(frame_len), (n_frames, 1))
        + np.tile(np.arange(n_frames) * frame_step, (frame_len, 1)).T
    )
    frames = padded[idx.astype(np.int32)] * np.hamming(frame_len)

    NFFT = 512
    mag = np.abs(np.fft.rfft(frames, NFFT))
    power = (1.0 / NFFT) * mag ** 2

    nfilt = 26
    low_mel, high_mel = 0, 2595 * np.log10(1 + (sr / 2) / 700)
    mel_pts = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_f = np.floor((NFFT + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((nfilt, NFFT // 2 + 1))
    for m in range(1, nfilt + 1):
        lo, mid, hi = bin_f[m - 1], bin_f[m], bin_f[m + 1]
        for k in range(lo, mid):
            fbank[m - 1, k] = (k - lo) / (mid - lo + 1e-10)
        for k in range(mid, hi):
            fbank[m - 1, k] = (hi - k) / (hi - mid + 1e-10)

    fb = np.dot(power, fbank.T)
    fb = np.where(fb == 0, np.finfo(float).eps, fb)
    fb = 20 * np.log10(fb)
    mfcc = fftpack.dct(fb, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return np.mean(mfcc, axis=0)


def extract_fft(segment: np.ndarray, sr: int) -> np.ndarray:
    """Return 6-dim spectral feature vector."""
    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
    eps = 1e-10
    total = np.sum(spectrum) + eps
    centroid = np.sum(freqs * spectrum) / total
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total)
    cum = np.cumsum(spectrum)
    rolloff_idx = np.searchsorted(cum, 0.85 * cum[-1])
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    n = len(spectrum)
    low_e = np.log1p(np.mean(spectrum[: n // 3]))
    mid_e = np.log1p(np.mean(spectrum[n // 3 : 2 * n // 3]))
    high_e = np.log1p(np.mean(spectrum[2 * n // 3 :]))

    return np.array([centroid, bandwidth, rolloff, low_e, mid_e, high_e])


def extract_hybrid(segment: np.ndarray, sr: int) -> np.ndarray:
    """Return concatenated MFCC + FFT features (19 dims)."""
    return np.concatenate([extract_mfcc(segment, sr), extract_fft(segment, sr)])


FEATURE_COLS = {
    "mfcc": [f"MFCC{i+1}" for i in range(13)],
    "fft": ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"],
    "hybrid": (
        [f"MFCC{i+1}" for i in range(13)]
        + ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"]
    ),
}

EXTRACTORS = {
    "mfcc": extract_mfcc,
    "fft": extract_fft,
    "hybrid": extract_hybrid,
}


# ─── data loading ─────────────────────────────────────────────────────────────

def load_from_directory(data_dir: Path, seg_duration: float = 1.0):
    """
    Load WAV files from data_dir/real/ and data_dir/fake/.
    Returns (dict of feature_type → DataFrame).
    """
    records = {ft: [] for ft in ["mfcc", "fft", "hybrid"]}

    for label in ("real", "fake"):
        subdir = data_dir / label
        if not subdir.exists():
            print(f"  [WARN] directory not found: {subdir}")
            continue
        wavs = list(subdir.glob("*.wav")) + list(subdir.glob("*.WAV"))
        print(f"  {label}: {len(wavs)} files")
        for wav_path in wavs:
            try:
                audio, sr = load_audio_mono16k(wav_path)
                seg_len = int(seg_duration * sr)
                segment = (
                    audio[:seg_len]
                    if len(audio) >= seg_len
                    else np.pad(audio, (0, seg_len - len(audio)))
                )
                for ft, extractor in EXTRACTORS.items():
                    t0 = time.perf_counter()
                    feats = extractor(segment, sr)
                    elapsed = time.perf_counter() - t0
                    row = dict(zip(FEATURE_COLS[ft], feats))
                    row["label"] = label
                    row["_extraction_time"] = elapsed
                    records[ft].append(row)
            except Exception as e:
                print(f"  [SKIP] {wav_path.name}: {e}")

    dfs = {}
    for ft, rows in records.items():
        if rows:
            dfs[ft] = pd.DataFrame(rows)
    return dfs


def load_from_csv(csv_path: Path):
    """
    Load features CSV. Auto-detect feature type from columns.
    Returns (feature_type, DataFrame).
    """
    df = pd.read_csv(csv_path)
    # Normalise label column
    if "label" not in df.columns:
        raise ValueError(f"CSV must have a 'label' column. Got: {df.columns.tolist()}")

    # Detect feature type
    cols = set(df.columns) - {"label", "_extraction_time"}
    if any(c.startswith("MFCC") for c in cols) and any(c in cols for c in ("centroid", "bandwidth")):
        # May be hybrid or legacy 18-feature
        if "low_energy" in cols:
            ft = "hybrid"
        else:
            # Legacy: MFCC1-13 + centroid/bandwidth/rolloff/jitter/shimmer
            ft = "mfcc_legacy"
    elif any(c.startswith("MFCC") for c in cols):
        ft = "mfcc"
    elif "centroid" in cols:
        ft = "fft"
    else:
        # Try lowercase mfcc columns
        if any(c.startswith("mfcc_") for c in cols):
            # Rename: mfcc_0 → MFCC1 etc.
            renames = {f"mfcc_{i}": f"MFCC{i+1}" for i in range(13)}
            df = df.rename(columns=renames)
            ft = "mfcc" if "centroid" not in df.columns else "hybrid"
        else:
            ft = "unknown"
    return ft, df


# ─── training ─────────────────────────────────────────────────────────────────

def build_classifier():
    """Return a Gradient Boosting classifier pipeline (scaler + GBM)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )),
    ])


def evaluate_model(clf, X, y, cv_folds: int = 5):
    """Run stratified k-fold CV and return metrics dict."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    return {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, pos_label="fake", zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, pos_label="fake", zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, pos_label="fake", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "report": classification_report(y, y_pred, zero_division=0),
    }


def train_single(df: pd.DataFrame, feature_type: str, output_dir: Path):
    """Train one model variant on df. Returns (metrics_dict, model_path)."""
    # Identify feature columns
    feat_cols = [c for c in df.columns if c not in ("label", "_extraction_time")]
    X = df[feat_cols].values
    y = df["label"].values  # "real" or "fake"

    print(f"\n  [{feature_type.upper()}] Samples: {len(df)} | Features: {X.shape[1]}")
    print(f"  Label distribution: {pd.Series(y).value_counts().to_dict()}")

    # Measure feature extraction time if available
    avg_extract_ms = None
    if "_extraction_time" in df.columns:
        avg_extract_ms = round(df["_extraction_time"].mean() * 1000, 3)

    # Measure inference time on held-out slice
    t0 = time.perf_counter()
    clf = build_classifier()
    metrics = evaluate_model(clf, X, y)
    cv_time = time.perf_counter() - t0

    # Train final model on all data
    clf.fit(X, y)

    # Inference benchmark (1000 single-sample predictions)
    bench_feats = X[:1]
    bench_start = time.perf_counter()
    for _ in range(1000):
        clf.predict(bench_feats)
    bench_ms = round((time.perf_counter() - bench_start), 3)

    metrics["cv_time_s"] = round(cv_time, 2)
    metrics["inference_1k_ms"] = round(bench_ms * 1000, 1)
    if avg_extract_ms:
        metrics["avg_feature_extraction_ms"] = avg_extract_ms

    model_path = output_dir / f"deepfake_detector_{feature_type}.pkl"
    model_obj = {
        "model": clf,
        "model_name": f"deepfake_detector_{feature_type}",
        "feature_type": feature_type,
        "feature_columns": feat_cols,
        "metrics": metrics,
    }
    joblib.dump(model_obj, model_path)
    print(f"  Saved: {model_path}")
    print(f"  F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")

    return metrics, model_path


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train voice deepfake detection models")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data", type=Path, help="Directory with real/ and fake/ subdirs of WAV files")
    group.add_argument("--csv", type=Path, help="Pre-extracted features CSV")
    parser.add_argument("--output", type=Path, default=MODELS_DIR, help="Output directory for models")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--seg-duration", type=float, default=1.0, help="Segment length in seconds")
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Voice Deepfake Detector — Training Pipeline")
    print("=" * 60)

    all_results = {}

    if args.data:
        data_dir = args.data.resolve()
        print(f"\nLoading WAV files from: {data_dir}")
        dfs = load_from_directory(data_dir, seg_duration=args.seg_duration)

        if not dfs:
            print("\nERROR: No data found. Place WAV files in:")
            print("  training/data/real/*.wav")
            print("  training/data/fake/*.wav")
            sys.exit(1)

        best_f1 = -1
        best_path = None
        for ft in ["mfcc", "fft", "hybrid"]:
            if ft not in dfs:
                print(f"  [SKIP] {ft} — no data")
                continue
            metrics, model_path = train_single(dfs[ft], ft, output_dir)
            all_results[ft] = metrics
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_path = model_path

    elif args.csv:
        csv_path = args.csv.resolve()
        print(f"\nLoading CSV: {csv_path}")
        ft, df = load_from_csv(csv_path)
        print(f"  Detected feature type: {ft} | Rows: {len(df)}")

        # If the CSV has hybrid/mfcc_legacy features, train all three variants
        cols = set(df.columns) - {"label", "_extraction_time"}
        has_mfcc = any(c.startswith("MFCC") for c in cols)
        has_fft = "centroid" in cols and "bandwidth" in cols

        if has_mfcc and has_fft:
            print("  Training all three model variants (MFCC, FFT, Hybrid)...")
            best_f1 = -1
            best_path = None

            # MFCC-only (13 features)
            mfcc_cols = [c for c in df.columns if c.startswith("MFCC")][:13]
            df_mfcc = df[mfcc_cols + ["label"]].copy()
            metrics_mfcc, path_mfcc = train_single(df_mfcc, "mfcc", output_dir)
            all_results["mfcc"] = metrics_mfcc
            if metrics_mfcc["f1"] > best_f1:
                best_f1 = metrics_mfcc["f1"]
                best_path = path_mfcc

            # FFT-only: centroid, bandwidth, rolloff + derive band energies from MFCCs
            fft_cols = ["centroid", "bandwidth", "rolloff"]
            if "low_energy" in cols:
                fft_cols.extend(["low_energy", "mid_energy", "high_energy"])
            # For legacy CSVs without band energies, use centroid/bandwidth/rolloff + jitter/shimmer as proxy
            elif "jitter" in cols:
                fft_cols.extend(["jitter", "shimmer"])
                # Add 3 MFCC coefficients to reach 6 features
                fft_cols.extend(["MFCC1", "MFCC2", "MFCC3"])
            else:
                # Use first 3 MFCCs to reach 6 features
                fft_cols.extend(["MFCC1", "MFCC2", "MFCC3"])

            df_fft = df[fft_cols + ["label"]].copy()
            metrics_fft, path_fft = train_single(df_fft, "fft", output_dir)
            all_results["fft"] = metrics_fft
            if metrics_fft["f1"] > best_f1:
                best_f1 = metrics_fft["f1"]
                best_path = path_fft

            # Hybrid: all available features
            hybrid_cols = [c for c in df.columns if c not in ("label", "_extraction_time")]
            df_hybrid = df[hybrid_cols + ["label"]].copy()
            metrics_hybrid, path_hybrid = train_single(df_hybrid, "hybrid", output_dir)
            all_results["hybrid"] = metrics_hybrid
            if metrics_hybrid["f1"] > best_f1:
                best_f1 = metrics_hybrid["f1"]
                best_path = path_hybrid
        else:
            # Single feature type - train as before
            metrics, model_path = train_single(df, ft, output_dir)
            all_results[ft] = metrics
            best_path = model_path

    else:
        # Try defaults: look for combined osr_features.csv or training/data/
        defaults = [
            ROOT_DIR / "osr_features.csv",
            ROOT_DIR / "features.csv",
            ROOT_DIR / "training" / "data",
        ]
        found = None
        for d in defaults:
            if d.exists():
                found = d
                break
        if found is None:
            print("\nERROR: No data source specified and no defaults found.")
            print("  Use --data DIR  or  --csv FILE")
            print("\nExpected defaults:")
            for d in defaults:
                print(f"  {d}")
            sys.exit(1)

        if found.is_dir():
            print(f"\nLoading WAV files from default: {found}")
            dfs = load_from_directory(found)
            best_f1 = -1
            best_path = None
            for ft in ["mfcc", "fft", "hybrid"]:
                if ft in dfs:
                    metrics, model_path = train_single(dfs[ft], ft, output_dir)
                    all_results[ft] = metrics
                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        best_path = model_path
        else:
            print(f"\nLoading CSV from default: {found}")
            ft, df = load_from_csv(found)
            metrics, model_path = train_single(df, ft, output_dir)
            all_results[ft] = metrics
            best_path = model_path

    # Copy best model
    if best_path and best_path.exists():
        import shutil
        best_dest = output_dir / "deepfake_detector_best.pkl"
        shutil.copy2(best_path, best_dest)
        print(f"\nBest model ({best_path.name}) → {best_dest}")

    # Save results table
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    header = f"{'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for ft, m in all_results.items():
        print(
            f"{ft:<12} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>8.4f}"
        )

    if all_results:
        best_ft = max(all_results, key=lambda k: all_results[k]["f1"])
        print(f"\nBest model by F1: {best_ft.upper()} (F1={all_results[best_ft]['f1']:.4f})")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
