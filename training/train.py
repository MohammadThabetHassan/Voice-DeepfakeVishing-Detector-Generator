#!/usr/bin/env python3
"""
Training pipeline for Voice Deepfake Detection.

Trains and evaluates THREE model variants:
  1. MFCC-only     (13-dim MFCCs averaged over frames)
  2. FFT/Spectral  (6-dim: centroid, bandwidth, rolloff, band energies)
  3. Hybrid        (19-dim: MFCC + FFT features)
  4. Enhanced      (75-dim: MFCC + FFT + pitch + jitter + shimmer + delta MFCCs)

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
  models/deepfake_detector_enhanced.pkl
  models/deepfake_detector_best.pkl  (copy of best model)
  models/results.json                (metrics comparison table)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.fftpack as fftpack
from scipy import signal as scipy_signal
from scipy.io import wavfile
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# XGBoost classifier (optional - gracefully handles if not installed)
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

THRESHOLD_PROFILES = {
    "balanced": {"threshold": 0.8, "uncertain_margin": 0.08},
    "low_fp": {"threshold": 0.85, "uncertain_margin": 0.06},
    "high_recall": {"threshold": 0.7, "uncertain_margin": 0.1},
}

NON_FEATURE_COLUMNS = {
    "label",
    "_extraction_time",
    "_source_file",
    "_speaker_id",
    "_source_domain",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def hash_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_wav_directory_manifest(data_dir: Path) -> str:
    """
    Fast, reproducible directory fingerprint for training-data versioning.
    Hashes relative path + size + light content samples for all WAV files.
    """
    h = hashlib.sha256()
    wavs = sorted(list(data_dir.rglob("*.wav")) + list(data_dir.rglob("*.WAV")))
    for path in wavs:
        rel = path.relative_to(data_dir).as_posix()
        st = path.stat()
        h.update(rel.encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        # Include a small content sample to detect same-size file replacements.
        with path.open("rb") as f:
            head = f.read(65536)
            h.update(head)
            if st.st_size > 65536:
                f.seek(max(0, st.st_size - 65536))
                tail = f.read(65536)
                h.update(tail)
    return h.hexdigest()


def infer_source_domain(wav_path: Path, label: str, data_dir: Path) -> str:
    """Infer source domain (dataset/source bucket) from path and filename."""
    try:
        rel = wav_path.relative_to(data_dir)
        parts = rel.parts
        if len(parts) >= 3 and parts[0] == label:
            return parts[1].lower()
    except Exception:
        pass

    stem = wav_path.stem.lower()
    if "youtube" in stem or stem.startswith("yt"):
        return "youtube"
    if "podcast" in stem or "_pod_" in stem:
        return "podcast"
    if stem.startswith("osr"):
        return "osr"
    return "default"


def infer_speaker_id(wav_path: Path, label: str, data_dir: Path) -> str:
    """
    Infer a stable speaker id from folder/file naming.
    Falls back to filename stem if no clearer grouping is available.
    """
    try:
        rel = wav_path.relative_to(data_dir)
        parts = rel.parts
        # data/<label>/<speaker_or_domain>/<file>.wav
        if len(parts) >= 3 and parts[0] == label and parts[-2] != label:
            parent = parts[-2].strip().lower()
            if parent and parent not in ("real", "fake"):
                return parent
    except Exception:
        pass

    stem = wav_path.stem.lower()
    # Common dataset pattern: osr_us_000_0030_8k -> speaker "osr_us_000"
    m_osr = re.match(r"^(osr_[a-z]{2}_[0-9]{3})_", stem)
    if m_osr:
        return m_osr.group(1)

    # Generic speaker id tokens like speaker12 / spk_01 / id-004
    m = re.search(r"(?:spk|speaker|voice|id)[_-]?([a-z0-9]{1,8})", stem)
    if m:
        return f"spk_{m.group(1)}"

    tokens = [t for t in re.split(r"[_\-\s]+", stem) if t]
    if len(tokens) >= 3 and tokens[2].isdigit():
        return "_".join(tokens[:3])
    if len(tokens) >= 2:
        return "_".join(tokens[:2])
    return stem or "unknown_speaker"


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    speaker_disjoint: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return train/test split and split mode string."""
    from sklearn.model_selection import train_test_split

    if speaker_disjoint and "_speaker_id" in df.columns:
        groups = df["_speaker_id"].astype(str).fillna("unknown")
        unique_groups = groups.nunique()
        if unique_groups >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(df, y=df["label"], groups=groups))
            df_train = df.iloc[train_idx].copy()
            df_test = df.iloc[test_idx].copy()
            if (
                len(set(df_train["label"])) >= 2
                and len(set(df_test["label"])) >= 2
                and df_train["_speaker_id"].nunique() >= 1
                and df_test["_speaker_id"].nunique() >= 1
            ):
                overlap = set(df_train["_speaker_id"]).intersection(set(df_test["_speaker_id"]))
                if len(overlap) == 0:
                    return df_train, df_test, "speaker_disjoint"

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    return df_train.copy(), df_test.copy(), "stratified"


def evaluate_with_source_breakdown(
    clf,
    df_eval: pd.DataFrame,
    feat_cols: list[str],
    classifier_type: str,
    pos_label,
):
    """Evaluate on external/benchmark dataframe and return global + per-source metrics."""
    X_eval = df_eval[feat_cols].values
    y_eval = df_eval["label"].values
    if classifier_type == "xgboost":
        label_map = {"real": 0, "fake": 1}
        y_eval_encoded = np.array([label_map.get(label, label) for label in y_eval])
        global_metrics = evaluate_on_test_set(clf, X_eval, y_eval_encoded, pos_label=pos_label)
    else:
        global_metrics = evaluate_on_test_set(clf, X_eval, y_eval, pos_label=pos_label)

    source_metrics = {}
    if "_source_domain" in df_eval.columns:
        for source, df_src in df_eval.groupby("_source_domain"):
            if len(df_src) < 2:
                continue
            X_src = df_src[feat_cols].values
            y_src = df_src["label"].values
            if classifier_type == "xgboost":
                label_map = {"real": 0, "fake": 1}
                y_src_encoded = np.array([label_map.get(label, label) for label in y_src])
                src_metrics = evaluate_on_test_set(clf, X_src, y_src_encoded, pos_label=pos_label)
            else:
                src_metrics = evaluate_on_test_set(clf, X_src, y_src, pos_label=pos_label)
            src_metrics["samples"] = int(len(df_src))
            source_metrics[str(source)] = src_metrics

    return global_metrics, source_metrics


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
    power = (1.0 / NFFT) * mag**2

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


def extract_pitch(segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract fundamental frequency (F0) using autocorrelation method.
    Returns mean and std of pitch (2 dims).
    """
    eps = 1e-10
    frame_len = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    n = len(segment)
    n_frames = max(1, int(np.ceil((n - frame_len) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_len
    padded = np.append(segment, np.zeros(max(0, pad_len - n)))

    idx = (
        np.tile(np.arange(frame_len), (n_frames, 1))
        + np.tile(np.arange(n_frames) * frame_step, (frame_len, 1)).T
    )
    frames = padded[idx.astype(np.int32)]

    pitches = []
    for frame in frames:
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 1e-6:
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        corr = corr / (corr[0] + eps)

        min_lag = int(sr / 500)
        max_lag = int(sr / 50)
        if max_lag >= len(corr):
            max_lag = len(corr) - 1

        if max_lag > min_lag:
            peak_idx = min_lag + np.argmax(corr[min_lag:max_lag])
            if corr[peak_idx] > 0.5:
                f0 = sr / peak_idx
                pitches.append(f0)

    if len(pitches) < 2:
        return np.array([0.0, 0.0])

    pitches_arr = np.array(pitches)
    pitches_arr = pitches_arr[(pitches_arr > 50) & (pitches_arr < 600)]
    if len(pitches_arr) < 2:
        return np.array([0.0, 0.0])

    return np.array([np.mean(pitches_arr), np.std(pitches_arr)])


def extract_jitter(segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Calculate pitch period jitter (variation in pitch periods).
    Returns relative average perturbation (1 dim).
    """
    eps = 1e-10
    frame_len = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    n = len(segment)
    n_frames = max(1, int(np.ceil((n - frame_len) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_len
    padded = np.append(segment, np.zeros(max(0, pad_len - n)))

    idx = (
        np.tile(np.arange(frame_len), (n_frames, 1))
        + np.tile(np.arange(n_frames) * frame_step, (frame_len, 1)).T
    )
    frames = padded[idx.astype(np.int32)]

    periods = []
    for frame in frames:
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 1e-6:
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        corr = corr / (corr[0] + eps)

        min_lag = int(sr / 500)
        max_lag = int(sr / 50)
        if max_lag >= len(corr):
            max_lag = len(corr) - 1

        if max_lag > min_lag:
            peak_idx = min_lag + np.argmax(corr[min_lag:max_lag])
            if corr[peak_idx] > 0.5:
                periods.append(peak_idx)

    if len(periods) < 3:
        return np.array([0.0])

    periods_arr = np.array(periods, dtype=float)
    diffs = np.abs(np.diff(periods_arr))
    jitter = np.mean(diffs) / (np.mean(periods_arr) + eps)

    return np.array([jitter])


def extract_shimmer(segment: np.ndarray) -> np.ndarray:
    """
    Calculate amplitude shimmer (variation in amplitude).
    Returns dB variation (1 dim).
    """
    eps = 1e-10
    frame_len = int(0.025 * 16000)
    frame_step = int(0.010 * 16000)
    n = len(segment)
    n_frames = max(1, int(np.ceil((n - frame_len) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_len
    padded = np.append(segment, np.zeros(max(0, pad_len - n)))

    idx = (
        np.tile(np.arange(frame_len), (n_frames, 1))
        + np.tile(np.arange(n_frames) * frame_step, (frame_len, 1)).T
    )
    frames = padded[idx.astype(np.int32)]

    amplitudes = np.abs(np.mean(frames, axis=1))
    amplitudes = amplitudes[amplitudes > eps]

    if len(amplitudes) < 3:
        return np.array([0.0])

    amp_db = 20 * np.log10(amplitudes + eps)
    shimmer = np.mean(np.abs(np.diff(amp_db)))

    return np.array([shimmer])


def extract_delta_mfccs(mfcc_features: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """
    Calculate first and second derivatives of MFCCs.
    Returns flattened delta and delta-delta features.
    Assumes mfcc_features is 2D: (n_frames, n_mfcc)
    """
    eps = 1e-10

    if mfcc_features.ndim == 1:
        mfcc_features = mfcc_features.reshape(1, -1)

    n_frames = mfcc_features.shape[0]
    if n_frames < 3:
        delta = np.zeros((n_frames, n_mfcc))
        delta2 = np.zeros((n_frames, n_mfcc))
    else:
        delta = np.zeros_like(mfcc_features)
        for t in range(n_frames):
            if t == 0:
                delta[t] = mfcc_features[1] - mfcc_features[0]
            elif t == n_frames - 1:
                delta[t] = mfcc_features[-1] - mfcc_features[-2]
            else:
                delta[t] = (mfcc_features[t + 1] - mfcc_features[t - 1]) / 2

        delta2 = np.zeros_like(delta)
        for t in range(n_frames):
            if t == 0:
                delta2[t] = delta[1] - delta[0]
            elif t == n_frames - 1:
                delta2[t] = delta[-1] - delta[-2]
            else:
                delta2[t] = (delta[t + 1] - delta[t - 1]) / 2

    delta_mean = np.mean(delta, axis=0)
    delta_std = np.std(delta, axis=0)
    delta2_mean = np.mean(delta2, axis=0)
    delta2_std = np.std(delta2, axis=0)

    return np.concatenate([delta_mean, delta_std, delta2_mean, delta2_std])


def extract_enhanced_features(segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Combine all enhanced features into a single feature vector.
    Returns: MFCC (13) + FFT (6) + pitch (2) + jitter (1) + shimmer (1) + delta MFCCs (52)
    Total: 75 dimensions
    """
    mfcc = extract_mfcc(segment, sr)
    fft = extract_fft(segment, sr)
    pitch = extract_pitch(segment, sr)
    jitter = extract_jitter(segment, sr)
    shimmer = extract_shimmer(segment)

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
    power = (1.0 / NFFT) * mag**2

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
    mfcc_full = fftpack.dct(fb, type=2, axis=1, norm="ortho")[:, :13]

    delta_mfccs = extract_delta_mfccs(mfcc_full, n_mfcc=13)

    return np.concatenate([mfcc, fft, pitch, jitter, shimmer, delta_mfccs])


FEATURE_COLS = {
    "mfcc": [f"MFCC{i + 1}" for i in range(13)],
    "fft": ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"],
    "hybrid": (
        [f"MFCC{i + 1}" for i in range(13)]
        + ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"]
    ),
    "enhanced": (
        [f"MFCC{i + 1}" for i in range(13)]
        + ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"]
        + ["pitch_mean", "pitch_std", "jitter", "shimmer"]
        + [f"delta_mfcc{i + 1}_mean" for i in range(13)]
        + [f"delta_mfcc{i + 1}_std" for i in range(13)]
        + [f"delta2_mfcc{i + 1}_mean" for i in range(13)]
        + [f"delta2_mfcc{i + 1}_std" for i in range(13)]
    ),
}

EXTRACTORS = {
    "mfcc": extract_mfcc,
    "fft": extract_fft,
    "hybrid": extract_hybrid,
    "enhanced": extract_enhanced_features,
}


# ─── data loading ─────────────────────────────────────────────────────────────


def load_from_directory(data_dir: Path, seg_duration: float = 1.0):
    """
    Load WAV files from data_dir/real/ and data_dir/fake/.
    Returns (dict of feature_type → DataFrame).
    """
    records = {ft: [] for ft in ["mfcc", "fft", "hybrid", "enhanced"]}

    for label in ("real", "fake"):
        subdir = data_dir / label
        if not subdir.exists():
            print(f"  [WARN] directory not found: {subdir}")
            continue
        wavs = list(subdir.rglob("*.wav")) + list(subdir.rglob("*.WAV"))
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
                speaker_id = infer_speaker_id(wav_path, label=label, data_dir=data_dir)
                source_domain = infer_source_domain(wav_path, label=label, data_dir=data_dir)
                for ft, extractor in EXTRACTORS.items():
                    t0 = time.perf_counter()
                    feats = extractor(segment, sr)
                    elapsed = time.perf_counter() - t0
                    row = dict(zip(FEATURE_COLS[ft], feats))
                    row["label"] = label
                    row["_extraction_time"] = elapsed
                    row["_source_file"] = str(wav_path)
                    row["_speaker_id"] = speaker_id
                    row["_source_domain"] = source_domain
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

    # Ensure expected metadata columns exist for downstream split/evaluation logic.
    for meta_col in ("_speaker_id", "_source_domain"):
        if meta_col not in df.columns:
            df[meta_col] = "unknown"

    # Detect feature type
    cols = set(df.columns) - NON_FEATURE_COLUMNS
    if any(c.startswith("MFCC") for c in cols) and any(
        c in cols for c in ("centroid", "bandwidth")
    ):
        # May be hybrid, enhanced, or legacy 18-feature
        if "pitch_mean" in cols:
            ft = "enhanced"
        elif "low_energy" in cols:
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
            renames = {f"mfcc_{i}": f"MFCC{i + 1}" for i in range(13)}
            df = df.rename(columns=renames)
            ft = "mfcc" if "centroid" not in df.columns else "hybrid"
        else:
            ft = "unknown"
    return ft, df


# ─── training ─────────────────────────────────────────────────────────────────


def build_classifier():
    """Return a Gradient Boosting classifier pipeline (scaler + GBM)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                ),
            ),
        ]
    )


def build_xgboost_classifier():
    """Return an XGBoost classifier pipeline (scaler + XGB)."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def make_model_input(feats: np.ndarray, feat_cols: list[str] | None):
    """Build prediction input with optional pandas DataFrame column names."""
    arr = np.asarray(feats, dtype=np.float64).reshape(1, -1)
    if feat_cols:
        try:
            import pandas as pd

            return pd.DataFrame(arr, columns=feat_cols)
        except Exception:
            return arr
    return arr


class VotingEnsemble:
    """
    Voting ensemble classifier that combines predictions from multiple models.
    Supports both hard voting (majority vote) and soft voting (probability averaging).
    """

    def __init__(self, models: list, weights: list = None, voting: str = "soft"):
        """
        Args:
            models: List of model dicts with 'model', 'feature_type', 'feature_columns'
            weights: Optional list of weights for each model (must sum to 1.0)
            voting: 'soft' for probability averaging (recommended), 'hard' for majority vote
        """
        self.models = models
        self.voting = voting
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        self.label_map = {0: "real", 1: "fake"}
        self.inverse_label_map = {"real": 0, "fake": 1}

    def predict(self, X_dict: dict) -> str:
        """
        Predict class label for sample(s).
        X_dict: dict mapping feature_type -> features for that model
        Returns: predicted label ('real' or 'fake')
        """
        predictions = []
        for model_dict in self.models:
            feature_type = model_dict["feature_type"]
            model = model_dict["model"]
            feat_cols = model_dict.get("feature_columns", None)

            if feature_type not in X_dict:
                raise ValueError(f"Missing features for model type: {feature_type}")

            feats = X_dict[feature_type]
            df = make_model_input(feats, feat_cols)
            pred = model.predict(df)[0]
            # Normalize prediction to 0/1
            if str(pred) in ("1", "fake", "deepfake"):
                predictions.append(1)
            else:
                predictions.append(0)

        if self.voting == "hard":
            # Majority vote
            avg_pred = np.average(predictions, weights=self.weights)
            return self.label_map[int(round(avg_pred))]
        else:
            # Soft voting uses predict_proba, but we fall back to hard voting if not available
            return self.label_map[int(round(np.average(predictions, weights=self.weights)))]

    def predict_proba(self, X_dict: dict) -> np.ndarray:
        """
        Predict class probabilities for sample(s) using soft voting.
        X_dict: dict mapping feature_type -> features for that model
        Returns: array of probabilities [P(real), P(fake)]
        """
        probas = []
        valid_weights = []

        for model_dict, weight in zip(self.models, self.weights):
            feature_type = model_dict["feature_type"]
            model = model_dict["model"]
            feat_cols = model_dict.get("feature_columns", None)

            if feature_type not in X_dict:
                continue

            feats = X_dict[feature_type]
            df = make_model_input(feats, feat_cols)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df)[0]
                # Ensure consistent ordering: [P(real), P(fake)]
                classes = model.classes_
                if len(proba) == 2:
                    if str(classes[0]) in ("1", "fake", "deepfake"):
                        # classes[0] is fake, swap
                        proba = np.array([proba[1], proba[0]])
                    probas.append(proba * weight)
                    valid_weights.append(weight)

        if not probas:
            # Fallback to hard voting probabilities
            pred = self.predict(X_dict)
            if pred == "fake":
                return np.array([0.0, 1.0])
            else:
                return np.array([1.0, 0.0])

        # Normalize weights
        total_weight = sum(valid_weights)
        if total_weight > 0:
            avg_proba = np.sum(
                [p * (w / total_weight) for p, w in zip(probas, valid_weights)], axis=0
            )
        else:
            avg_proba = np.mean(probas, axis=0)

        return avg_proba


def create_ensemble(models: list, weights: list = None, voting: str = "soft"):
    """
    Create a voting ensemble from multiple trained models.

    Args:
        models: List of model dicts (loaded from .pkl files)
        weights: Optional list of weights for each model
        voting: 'soft' (default) for probability averaging, 'hard' for majority vote

    Returns:
        VotingEnsemble instance
    """
    return VotingEnsemble(models, weights=weights, voting=voting)


def train_ensemble(
    model_paths: list,
    output_dir: Path,
    dfs: dict = None,
    voting: str = "soft",
    custom_weights: list = None,
    training_data_hash: str | None = None,
    training_date_utc: str | None = None,
    training_data_source: str | None = None,
):
    """
    Load individual models and create an ensemble.

    Args:
        model_paths: List of paths to model .pkl files
        output_dir: Directory to save ensemble model
        dfs: Optional dict of feature_type -> DataFrame for evaluation
        voting: 'soft' or 'hard' voting
        custom_weights: Optional custom weights for models

    Returns:
        (ensemble_model_obj, model_path)
    """
    models = []
    for path in model_paths:
        if path.exists():
            obj = joblib.load(path)
            if isinstance(obj, dict) and "model" in obj:
                models.append(obj)
            else:
                # Legacy model format
                models.append(
                    {
                        "model": obj,
                        "model_name": path.stem,
                        "feature_type": "mfcc_hybrid",
                    }
                )

    if len(models) < 2:
        print("  [ENSEMBLE] Need at least 2 models for ensemble, skipping")
        return None, None

    print(f"\n  [ENSEMBLE] Creating ensemble from {len(models)} models ({voting} voting)")
    for m in models:
        print(f"    - {m.get('model_name', 'unknown')} ({m.get('feature_type', 'unknown')})")

    ensemble = create_ensemble(models, weights=custom_weights, voting=voting)

    # Evaluate ensemble if data provided
    metrics = {}
    if dfs:
        # Use hybrid dataframe if available, otherwise use first available
        eval_df = dfs.get("hybrid", dfs.get("mfcc", list(dfs.values())[0]))
        y_true = eval_df["label"].values

        predictions = []
        probabilities = []

        for _, row in eval_df.iterrows():
            # Build feature dict for this sample
            X_dict = {}
            for model_dict in models:
                ft = model_dict.get("feature_type", "mfcc")
                feat_cols = model_dict.get("feature_columns", None)

                if ft in dfs:
                    df_ft = dfs[ft]
                    if feat_cols:
                        feats = row[feat_cols].values if all(c in row for c in feat_cols) else None
                    else:
                        feat_cols_ft = [
                            c for c in df_ft.columns if c not in NON_FEATURE_COLUMNS
                        ]
                        feats = (
                            row[feat_cols_ft].values
                            if all(c in row for c in feat_cols_ft)
                            else None
                        )

                    if feats is not None:
                        X_dict[ft] = feats

            if len(X_dict) == len(models):
                pred = ensemble.predict(X_dict)
                predictions.append(pred)
                proba = ensemble.predict_proba(X_dict)
                probabilities.append(max(proba))

        if predictions:
            metrics["accuracy"] = round(accuracy_score(y_true[: len(predictions)], predictions), 4)
            metrics["precision"] = round(
                precision_score(
                    y_true[: len(predictions)], predictions, pos_label="fake", zero_division=0
                ),
                4,
            )
            metrics["recall"] = round(
                recall_score(
                    y_true[: len(predictions)], predictions, pos_label="fake", zero_division=0
                ),
                4,
            )
            metrics["f1"] = round(
                f1_score(
                    y_true[: len(predictions)], predictions, pos_label="fake", zero_division=0
                ),
                4,
            )
            metrics["confusion_matrix"] = confusion_matrix(
                y_true[: len(predictions)], predictions
            ).tolist()
            metrics["avg_confidence"] = round(np.mean(probabilities), 4) if probabilities else 0.5
            print(f"  Ensemble F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")

    model_name = "deepfake_detector_ensemble"
    model_path = output_dir / f"{model_name}.pkl"
    trained_at = training_date_utc or utc_now_iso()
    model_id = hashlib.sha1(
        f"{model_name}|{trained_at}|{training_data_hash or 'unknown'}".encode("utf-8")
    ).hexdigest()[:16]

    # Store ensemble and component models
    ensemble_obj = {
        "model": ensemble,
        "model_id": model_id,
        "models": models,
        "model_name": model_name,
        "feature_type": "ensemble",
        "classifier_type": "ensemble",
        "feature_columns": None,  # Multiple feature types
        "metrics": metrics,
        "voting": voting,
        "weights": custom_weights if custom_weights else [1.0 / len(models)] * len(models),
        "training_date_utc": trained_at,
        "training_data_hash": training_data_hash,
        "training_data_source": training_data_source,
        "recommended_threshold_profiles": THRESHOLD_PROFILES,
    }
    joblib.dump(ensemble_obj, model_path)
    print(f"  Saved: {model_path}")

    return ensemble_obj, model_path


def evaluate_model(clf, X, y, cv_folds: int = 5, pos_label="fake"):
    """Run stratified k-fold CV and return metrics dict."""
    class_counts = pd.Series(y).value_counts()
    max_folds = int(class_counts.min()) if not class_counts.empty else 0
    safe_folds = max(2, min(cv_folds, max_folds))
    if safe_folds != cv_folds:
        print(
            f"  [WARN] Reducing CV folds from {cv_folds} to {safe_folds} "
            f"(min samples per class: {max_folds})"
        )
    cv = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    return {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, pos_label=pos_label, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, pos_label=pos_label, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, pos_label=pos_label, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "report": classification_report(y, y_pred, zero_division=0),
    }


def evaluate_on_test_set(clf, X_test, y_test, pos_label="fake"):
    """Evaluate trained classifier on held-out test set."""
    y_pred = clf.predict(X_test)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(
            precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0), 4
        ),
        "recall": round(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_single(
    df_train: pd.DataFrame,
    feature_type: str,
    output_dir: Path,
    classifier_type: str = "gradient_boosting",
    df_test: pd.DataFrame = None,
    df_ood: pd.DataFrame = None,
    cv_folds: int = 5,
    calibration_method: str = "sigmoid",
    calibration_cv_folds: int = 3,
    training_data_hash: str | None = None,
    training_date_utc: str | None = None,
    training_data_source: str | None = None,
):
    """Train one model variant on df_train, evaluate on df_test. Returns (metrics_dict, model_path)."""
    # Identify feature columns
    feat_cols = [c for c in df_train.columns if c not in NON_FEATURE_COLUMNS]
    X_train = df_train[feat_cols].values
    y_train = df_train["label"].values  # "real" or "fake"

    # Validate classifier type
    if classifier_type not in ("gradient_boosting", "xgboost"):
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    # Skip XGBoost if not available
    if classifier_type == "xgboost" and not XGBOOST_AVAILABLE:
        print(f"\n  [{feature_type.upper()} + XGBoost] Skipped - XGBoost not installed")
        return None, None

    clf_name = "GBM" if classifier_type == "gradient_boosting" else "XGB"
    print(
        f"\n  [{feature_type.upper()} + {clf_name}] Train samples: {len(df_train)} | Features: {X_train.shape[1]}"
    )
    print(f"  Train label distribution: {pd.Series(y_train).value_counts().to_dict()}")

    if df_test is not None:
        print(f"  Test samples: {len(df_test)}")

    # Measure feature extraction time if available
    avg_extract_ms = None
    if "_extraction_time" in df_train.columns:
        avg_extract_ms = round(df_train["_extraction_time"].mean() * 1000, 3)

    # Build classifier based on type
    if classifier_type == "xgboost":
        clf = build_xgboost_classifier()
        # XGBoost requires numeric labels (0, 1), not strings ('fake', 'real')
        # Encode: 'fake' -> 1, 'real' -> 0
        label_map = {"real": 0, "fake": 1}
        y_train_encoded = np.array([label_map.get(label, label) for label in y_train])
        pos_label = 1  # Use numeric pos_label for XGBoost (1 = fake)
    else:
        clf = build_classifier()
        y_train_encoded = y_train
        pos_label = "fake"  # Use string pos_label for GradientBoosting

    # Cross-validation on training set
    t0 = time.perf_counter()
    cv_metrics = evaluate_model(
        clf, X_train, y_train_encoded, cv_folds=cv_folds, pos_label=pos_label
    )
    cv_time = time.perf_counter() - t0

    # Train final model on all training data (with optional probability calibration).
    model_for_inference = clf
    calibration_applied = False
    calibration_reason = "disabled"
    if calibration_method != "none":
        class_counts = pd.Series(y_train_encoded).value_counts()
        min_class = int(class_counts.min()) if not class_counts.empty else 0
        safe_cal_folds = max(2, min(int(calibration_cv_folds), min_class))
        if min_class >= 2 and safe_cal_folds >= 2:
            try:
                model_for_inference = CalibratedClassifierCV(
                    estimator=clf,
                    method=calibration_method,
                    cv=safe_cal_folds,
                )
                model_for_inference.fit(X_train, y_train_encoded)
                calibration_applied = True
                calibration_reason = f"{calibration_method}:{safe_cal_folds}fold"
                print(f"  Calibration applied ({calibration_reason})")
            except Exception as e:
                calibration_reason = f"failed:{e}"
                print(f"  [WARN] Calibration failed ({e}); using base classifier.")
                model_for_inference = clf
                model_for_inference.fit(X_train, y_train_encoded)
        else:
            calibration_reason = f"insufficient_data(min_class={min_class})"
            print("  [WARN] Skipping calibration: insufficient per-class samples.")
            model_for_inference = clf
            model_for_inference.fit(X_train, y_train_encoded)
    else:
        model_for_inference = clf
        model_for_inference.fit(X_train, y_train_encoded)

    # Evaluate on test set if provided
    if df_test is not None:
        X_test = df_test[feat_cols].values
        y_test = df_test["label"].values
        # Encode test labels for XGBoost
        if classifier_type == "xgboost":
            y_test_encoded = np.array([label_map.get(label, label) for label in y_test])
            test_metrics = evaluate_on_test_set(
                model_for_inference, X_test, y_test_encoded, pos_label=pos_label
            )
        else:
            test_metrics = evaluate_on_test_set(model_for_inference, X_test, y_test, pos_label=pos_label)
        print(f"  Test F1={test_metrics['f1']:.4f}  Test Acc={test_metrics['accuracy']:.4f}")
    else:
        test_metrics = None

    # Out-of-domain / benchmark evaluation (optional)
    if df_ood is not None and len(df_ood) > 0:
        try:
            ood_global, ood_by_source = evaluate_with_source_breakdown(
                clf=model_for_inference,
                df_eval=df_ood,
                feat_cols=feat_cols,
                classifier_type=classifier_type,
                pos_label=pos_label,
            )
            print(
                f"  OOD F1={ood_global['f1']:.4f}  OOD Acc={ood_global['accuracy']:.4f} "
                f"(sources={len(ood_by_source)})"
            )
        except Exception as e:
            print(f"  [WARN] OOD evaluation failed: {e}")
            ood_global = None
            ood_by_source = {}
    else:
        ood_global = None
        ood_by_source = {}

    # Inference benchmark (1000 single-sample predictions)
    bench_feats = X_train[:1]
    bench_start = time.perf_counter()
    for _ in range(1000):
        model_for_inference.predict(bench_feats)
    bench_ms = round((time.perf_counter() - bench_start), 3)

    # Combine metrics — use top-level keys for compatibility with results table
    metrics = {
        "accuracy": cv_metrics["accuracy"],
        "precision": cv_metrics["precision"],
        "recall": cv_metrics["recall"],
        "f1": cv_metrics["f1"],
        "confusion_matrix": cv_metrics["confusion_matrix"],
        "cv_time_s": round(cv_time, 2),
    }

    if test_metrics:
        metrics["test_accuracy"] = test_metrics["accuracy"]
        metrics["test_precision"] = test_metrics["precision"]
        metrics["test_recall"] = test_metrics["recall"]
        metrics["test_f1"] = test_metrics["f1"]
        metrics["test_confusion_matrix"] = test_metrics["confusion_matrix"]

    if ood_global:
        metrics["ood_accuracy"] = ood_global["accuracy"]
        metrics["ood_precision"] = ood_global["precision"]
        metrics["ood_recall"] = ood_global["recall"]
        metrics["ood_f1"] = ood_global["f1"]
        metrics["ood_confusion_matrix"] = ood_global["confusion_matrix"]
        metrics["ood_samples"] = int(len(df_ood))
        metrics["ood_by_source"] = ood_by_source

    metrics["calibration_applied"] = calibration_applied
    metrics["calibration_method"] = calibration_method if calibration_applied else "none"
    metrics["calibration_info"] = calibration_reason

    metrics["inference_1k_ms"] = round(bench_ms * 1000, 1)
    if avg_extract_ms:
        metrics["avg_feature_extraction_ms"] = avg_extract_ms

    # Model naming includes classifier type
    suffix = "" if classifier_type == "gradient_boosting" else "_xgboost"
    model_name = f"deepfake_detector_{feature_type}{suffix}"
    model_path = output_dir / f"{model_name}.pkl"
    trained_at = training_date_utc or utc_now_iso()
    model_id = hashlib.sha1(
        f"{model_name}|{trained_at}|{training_data_hash or 'unknown'}".encode("utf-8")
    ).hexdigest()[:16]

    model_obj = {
        "model": model_for_inference,
        "model_id": model_id,
        "model_name": model_name,
        "feature_type": feature_type,
        "classifier_type": classifier_type,
        "feature_columns": feat_cols,
        "metrics": metrics,
        "is_calibrated": calibration_applied,
        "calibration_method": calibration_method if calibration_applied else "none",
        "calibration_cv_folds": calibration_cv_folds if calibration_applied else 0,
        "training_date_utc": trained_at,
        "training_data_hash": training_data_hash,
        "training_data_source": training_data_source,
        "recommended_threshold_profiles": THRESHOLD_PROFILES,
        "feature_schema_version": 2 if feature_type == "enhanced" else 1,
        "train_samples": int(len(df_train)),
        "test_samples": int(len(df_test)) if df_test is not None else 0,
        "train_label_distribution": df_train["label"].value_counts().to_dict(),
    }
    joblib.dump(model_obj, model_path)
    print(f"  Saved: {model_path}")
    print(f"  CV F1={metrics['f1']:.4f}  CV Acc={metrics['accuracy']:.4f}")

    return metrics, model_path


def write_model_manifest(output_dir: Path, run_info: dict, best_model_name: str | None = None) -> Path:
    """Build and save a consolidated model manifest for traceable deployment."""
    manifest = {
        "manifest_version": 1,
        "generated_at_utc": utc_now_iso(),
        "run": run_info,
        "models": [],
        "best_model_name": best_model_name,
    }

    model_paths = sorted(output_dir.glob("deepfake_detector_*.pkl"))
    for model_path in model_paths:
        if model_path.name.endswith("_best.pkl"):
            # Alias copy of an already-listed model artifact.
            continue

        try:
            obj = joblib.load(model_path)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        metrics = obj.get("metrics", {}) or {}
        feature_columns = obj.get("feature_columns")
        feature_count = len(feature_columns) if isinstance(feature_columns, list) else None

        entry = {
            "model_name": obj.get("model_name", model_path.stem),
            "model_id": obj.get("model_id"),
            "path": model_path.name,
            "feature_type": obj.get("feature_type"),
            "classifier_type": obj.get("classifier_type"),
            "is_calibrated": obj.get("is_calibrated", False),
            "calibration_method": obj.get("calibration_method", "none"),
            "feature_count": feature_count,
            "training_date_utc": obj.get("training_date_utc"),
            "training_data_hash": obj.get("training_data_hash"),
            "metrics": {
                "f1": metrics.get("f1"),
                "test_f1": metrics.get("test_f1"),
                "accuracy": metrics.get("accuracy"),
                "test_accuracy": metrics.get("test_accuracy"),
            },
        }
        if obj.get("feature_type") == "ensemble":
            components = obj.get("models", [])
            entry["ensemble_components"] = [m.get("model_name", "unknown") for m in components]
            entry["voting"] = obj.get("voting")

        manifest["models"].append(entry)

    if manifest["models"] and not manifest["best_model_name"]:
        best = max(
            manifest["models"],
            key=lambda m: (
                -1 if m["metrics"].get("test_f1") is None else m["metrics"]["test_f1"],
                -1 if m["metrics"].get("f1") is None else m["metrics"]["f1"],
            ),
        )
        manifest["best_model_name"] = best["model_name"]

    manifest_path = output_dir / "model_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train voice deepfake detection models")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--data", type=Path, help="Directory with real/ and fake/ subdirs of WAV files"
    )
    group.add_argument("--csv", type=Path, help="Pre-extracted features CSV")
    parser.add_argument(
        "--output", type=Path, default=MODELS_DIR, help="Output directory for models"
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--seg-duration", type=float, default=1.0, help="Segment length in seconds")
    parser.add_argument(
        "--ood-data",
        type=Path,
        default=None,
        help="Optional out-of-domain benchmark dir with real/ and fake/ subdirs",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["none", "sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability calibration method for final model (default: sigmoid)",
    )
    parser.add_argument(
        "--calibration-cv-folds",
        type=int,
        default=3,
        help="Cross-validation folds for calibration model fitting",
    )
    parser.add_argument(
        "--speaker-disjoint",
        dest="speaker_disjoint",
        action="store_true",
        default=True,
        help="Use speaker-disjoint train/test split when speaker metadata exists (default: on)",
    )
    parser.add_argument(
        "--no-speaker-disjoint",
        dest="speaker_disjoint",
        action="store_false",
        help="Disable speaker-disjoint split and use stratified random split",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Voice Deepfake Detector — Training Pipeline")
    print("=" * 60)

    all_results = {}
    training_date_utc = utc_now_iso()
    training_data_source = None
    training_data_hash = None
    ood_dfs = None

    if args.ood_data:
        ood_dir = args.ood_data.resolve()
        print(f"\nLoading OOD benchmark data from: {ood_dir}")
        ood_dfs = load_from_directory(ood_dir, seg_duration=args.seg_duration)
        if not ood_dfs:
            print("  [WARN] OOD directory provided but no usable WAV files were found.")
            ood_dfs = None

    def train_both_classifiers(df: pd.DataFrame, ft: str, output_dir: Path, test_size: float = 0.2):
        """Train both GBM and XGBoost variants with train/test split, return results dict and best path."""
        results = {}
        best_local_f1 = -1
        best_local_path = None

        df_train, df_test, split_mode = split_train_test(
            df,
            test_size=test_size,
            random_state=42,
            speaker_disjoint=args.speaker_disjoint,
        )
        split_note = f"\n  [Data Split] Train: {len(df_train)} | Test: {len(df_test)} ({split_mode})"
        if "_speaker_id" in df.columns:
            split_note += (
                f" | speakers train/test: {df_train['_speaker_id'].nunique()}/"
                f"{df_test['_speaker_id'].nunique()}"
            )
        print(split_note)

        df_ood_ft = None
        if ood_dfs and ft in ood_dfs:
            df_ood_ft = ood_dfs[ft]
            print(f"  [OOD] {ft}: {len(df_ood_ft)} samples")

        # Train Gradient Boosting (GBM)
        metrics_gbm, path_gbm = train_single(
            df_train,
            ft,
            output_dir,
            classifier_type="gradient_boosting",
            df_test=df_test,
            df_ood=df_ood_ft,
            cv_folds=args.cv_folds,
            calibration_method=args.calibration_method,
            calibration_cv_folds=args.calibration_cv_folds,
            training_data_hash=training_data_hash,
            training_date_utc=training_date_utc,
            training_data_source=training_data_source,
        )
        if metrics_gbm:
            results[f"{ft}_gbm"] = metrics_gbm
            # Use test F1 if available, otherwise CV F1
            f1_score = metrics_gbm.get("test_f1", metrics_gbm.get("f1", 0))
            if f1_score > best_local_f1:
                best_local_f1 = f1_score
                best_local_path = path_gbm

        # Train XGBoost
        metrics_xgb, path_xgb = train_single(
            df_train,
            ft,
            output_dir,
            classifier_type="xgboost",
            df_test=df_test,
            df_ood=df_ood_ft,
            cv_folds=args.cv_folds,
            calibration_method=args.calibration_method,
            calibration_cv_folds=args.calibration_cv_folds,
            training_data_hash=training_data_hash,
            training_date_utc=training_date_utc,
            training_data_source=training_data_source,
        )
        if metrics_xgb:
            results[f"{ft}_xgboost"] = metrics_xgb
            f1_score = metrics_xgb.get("test_f1", metrics_xgb.get("f1", 0))
            if f1_score > best_local_f1:
                best_local_f1 = f1_score
                best_local_path = path_xgb

        return results, best_local_path, best_local_f1

    if args.data:
        data_dir = args.data.resolve()
        training_data_source = str(data_dir)
        training_data_hash = hash_wav_directory_manifest(data_dir)
        print(f"\nLoading WAV files from: {data_dir}")
        print(f"Data hash: {training_data_hash[:16]}...")
        dfs = load_from_directory(data_dir, seg_duration=args.seg_duration)

        if not dfs:
            print("\nERROR: No data found. Place WAV files in:")
            print("  training/data/real/*.wav")
            print("  training/data/fake/*.wav")
            sys.exit(1)

        best_f1 = -1
        best_path = None
        for ft in ["mfcc", "fft", "hybrid", "enhanced"]:
            if ft not in dfs:
                print(f"  [SKIP] {ft} — no data")
                continue
            ft_results, ft_best_path, ft_best_f1 = train_both_classifiers(dfs[ft], ft, output_dir)
            all_results.update(ft_results)
            if ft_best_f1 > best_f1:
                best_f1 = ft_best_f1
                best_path = ft_best_path

    elif args.csv:
        csv_path = args.csv.resolve()
        training_data_source = str(csv_path)
        training_data_hash = hash_file_sha256(csv_path)
        print(f"\nLoading CSV: {csv_path}")
        print(f"Data hash: {training_data_hash[:16]}...")
        ft, df = load_from_csv(csv_path)
        print(f"  Detected feature type: {ft} | Rows: {len(df)}")

        # If the CSV has hybrid/mfcc_legacy features, train all three variants
        cols = set(df.columns) - NON_FEATURE_COLUMNS
        has_mfcc = any(c.startswith("MFCC") for c in cols)
        has_fft = "centroid" in cols and "bandwidth" in cols

        if has_mfcc and has_fft:
            print(
                "  Training all three model variants (MFCC, FFT, Hybrid) with both classifiers..."
            )
            best_f1 = -1
            best_path = None

            # MFCC-only (13 features)
            mfcc_cols = [c for c in df.columns if c.startswith("MFCC")][:13]
            df_mfcc = df[mfcc_cols + ["label"]].copy()
            mfcc_results, mfcc_best_path, mfcc_best_f1 = train_both_classifiers(
                df_mfcc, "mfcc", output_dir
            )
            all_results.update(mfcc_results)
            if mfcc_best_f1 > best_f1:
                best_f1 = mfcc_best_f1
                best_path = mfcc_best_path

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
            fft_results, fft_best_path, fft_best_f1 = train_both_classifiers(
                df_fft, "fft", output_dir
            )
            all_results.update(fft_results)
            if fft_best_f1 > best_f1:
                best_f1 = fft_best_f1
                best_path = fft_best_path

            # Hybrid: all available features
            hybrid_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
            df_hybrid = df[hybrid_cols + ["label"]].copy()
            hybrid_results, hybrid_best_path, hybrid_best_f1 = train_both_classifiers(
                df_hybrid, "hybrid", output_dir
            )
            all_results.update(hybrid_results)
            if hybrid_best_f1 > best_f1:
                best_f1 = hybrid_best_f1
                best_path = hybrid_best_path
        else:
            # Single feature type - train both classifiers
            ft_results, ft_best_path, ft_best_f1 = train_both_classifiers(df, ft, output_dir)
            all_results.update(ft_results)
            best_f1 = ft_best_f1
            best_path = ft_best_path

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
            training_data_source = str(found.resolve())
            training_data_hash = hash_wav_directory_manifest(found.resolve())
            print(f"\nLoading WAV files from default: {found}")
            print(f"Data hash: {training_data_hash[:16]}...")
            dfs = load_from_directory(found)
            best_f1 = -1
            best_path = None
            for ft in ["mfcc", "fft", "hybrid", "enhanced"]:
                if ft in dfs:
                    ft_results, ft_best_path, ft_best_f1 = train_both_classifiers(
                        dfs[ft], ft, output_dir
                    )
                    all_results.update(ft_results)
                    if ft_best_f1 > best_f1:
                        best_f1 = ft_best_f1
                        best_path = ft_best_path
        else:
            training_data_source = str(found.resolve())
            training_data_hash = hash_file_sha256(found.resolve())
            print(f"\nLoading CSV from default: {found}")
            print(f"Data hash: {training_data_hash[:16]}...")
            ft, df = load_from_csv(found)
            ft_results, ft_best_path, ft_best_f1 = train_both_classifiers(df, ft, output_dir)
            all_results.update(ft_results)
            best_f1 = ft_best_f1
            best_path = ft_best_path

    # Copy best model
    if best_path and best_path.exists():
        import shutil

        best_dest = output_dir / "deepfake_detector_best.pkl"
        shutil.copy2(best_path, best_dest)
        print(f"\nBest model ({best_path.name}) → {best_dest}")

    # Train ensemble model
    if args.data:
        # For WAV data, we have dfs with all feature types
        ensemble_model_paths = []
        for ft in ["mfcc", "fft", "hybrid", "enhanced"]:
            path = output_dir / f"deepfake_detector_{ft}.pkl"
            if path.exists():
                ensemble_model_paths.append(path)

        if len(ensemble_model_paths) >= 2:
            train_ensemble(
                ensemble_model_paths,
                output_dir,
                dfs=dfs if "dfs" in locals() else None,
                voting="soft",
                training_data_hash=training_data_hash,
                training_date_utc=training_date_utc,
                training_data_source=training_data_source,
            )
    else:
        # For CSV data, look for available models
        ensemble_model_paths = []
        for ft in ["mfcc", "fft", "hybrid"]:
            path = output_dir / f"deepfake_detector_{ft}.pkl"
            if path.exists():
                ensemble_model_paths.append(path)

        if len(ensemble_model_paths) >= 2:
            train_ensemble(
                ensemble_model_paths,
                output_dir,
                dfs=dfs if "dfs" in locals() else None,
                voting="soft",
                training_data_hash=training_data_hash,
                training_date_utc=training_date_utc,
                training_data_source=training_data_source,
            )

    # Save results table
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    run_info = {
        "training_date_utc": training_date_utc,
        "training_data_source": training_data_source,
        "training_data_hash": training_data_hash,
        "args": {
            "data": str(args.data) if args.data else None,
            "csv": str(args.csv) if args.csv else None,
            "output": str(args.output),
            "cv_folds": args.cv_folds,
            "seg_duration": args.seg_duration,
            "ood_data": str(args.ood_data) if args.ood_data else None,
            "speaker_disjoint": args.speaker_disjoint,
            "calibration_method": args.calibration_method,
            "calibration_cv_folds": args.calibration_cv_folds,
        },
    }
    manifest_path = write_model_manifest(
        output_dir,
        run_info=run_info,
        best_model_name=best_path.stem if best_path is not None else None,
    )
    print(f"Model manifest saved: {manifest_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for ft, m in all_results.items():
        print(
            f"{ft:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>8.4f}"
        )

    if all_results:
        best_ft = max(all_results, key=lambda k: all_results[k]["f1"])
        print(f"\nBest model by F1: {best_ft.upper()} (F1={all_results[best_ft]['f1']:.4f})")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
