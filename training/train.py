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
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.fftpack as fftpack
from scipy import signal as scipy_signal
from scipy.io import wavfile
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
            import pandas as pd

            df = pd.DataFrame([feats], columns=feat_cols) if feat_cols else pd.DataFrame([feats])
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
            import pandas as pd

            df = pd.DataFrame([feats], columns=feat_cols) if feat_cols else pd.DataFrame([feats])

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
                            c for c in df_ft.columns if c not in ("label", "_extraction_time")
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

    # Store ensemble and component models
    ensemble_obj = {
        "model": ensemble,
        "models": models,
        "model_name": model_name,
        "feature_type": "ensemble",
        "classifier_type": "ensemble",
        "feature_columns": None,  # Multiple feature types
        "metrics": metrics,
        "voting": voting,
        "weights": custom_weights if custom_weights else [1.0 / len(models)] * len(models),
    }
    joblib.dump(ensemble_obj, model_path)
    print(f"  Saved: {model_path}")

    return ensemble_obj, model_path


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


def evaluate_on_test_set(clf, X_test, y_test):
    """Evaluate trained classifier on held-out test set."""
    y_pred = clf.predict(X_test)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label="fake", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, pos_label="fake", zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, pos_label="fake", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_single(
    df_train: pd.DataFrame,
    feature_type: str,
    output_dir: Path,
    classifier_type: str = "gradient_boosting",
    df_test: pd.DataFrame = None,
):
    """Train one model variant on df_train, evaluate on df_test. Returns (metrics_dict, model_path)."""
    # Identify feature columns
    feat_cols = [c for c in df_train.columns if c not in ("label", "_extraction_time")]
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
    else:
        clf = build_classifier()

    # Cross-validation on training set
    t0 = time.perf_counter()
    cv_metrics = evaluate_model(clf, X_train, y_train)
    cv_time = time.perf_counter() - t0

    # Train final model on all training data
    clf.fit(X_train, y_train)

    # Evaluate on test set if provided
    if df_test is not None:
        X_test = df_test[feat_cols].values
        y_test = df_test["label"].values
        test_metrics = evaluate_on_test_set(clf, X_test, y_test)
        print(f"  Test F1={test_metrics['f1']:.4f}  Test Acc={test_metrics['accuracy']:.4f}")
    else:
        test_metrics = None

    # Inference benchmark (1000 single-sample predictions)
    bench_feats = X_train[:1]
    bench_start = time.perf_counter()
    for _ in range(1000):
        clf.predict(bench_feats)
    bench_ms = round((time.perf_counter() - bench_start), 3)

    # Combine metrics
    metrics = {
        "cv_accuracy": cv_metrics["accuracy"],
        "cv_precision": cv_metrics["precision"],
        "cv_recall": cv_metrics["recall"],
        "cv_f1": cv_metrics["f1"],
        "cv_confusion_matrix": cv_metrics["confusion_matrix"],
        "cv_time_s": round(cv_time, 2),
    }

    if test_metrics:
        metrics["test_accuracy"] = test_metrics["accuracy"]
        metrics["test_precision"] = test_metrics["precision"]
        metrics["test_recall"] = test_metrics["recall"]
        metrics["test_f1"] = test_metrics["f1"]
        metrics["test_confusion_matrix"] = test_metrics["confusion_matrix"]

    metrics["inference_1k_ms"] = round(bench_ms * 1000, 1)
    if avg_extract_ms:
        metrics["avg_feature_extraction_ms"] = avg_extract_ms

    # Model naming includes classifier type
    suffix = "" if classifier_type == "gradient_boosting" else "_xgboost"
    model_name = f"deepfake_detector_{feature_type}{suffix}"
    model_path = output_dir / f"{model_name}.pkl"
    model_obj = {
        "model": clf,
        "model_name": model_name,
        "feature_type": feature_type,
        "classifier_type": classifier_type,
        "feature_columns": feat_cols,
        "metrics": metrics,
    }
    joblib.dump(model_obj, model_path)
    print(f"  Saved: {model_path}")
    print(f"  CV F1={metrics['cv_f1']:.4f}  CV Acc={metrics['cv_accuracy']:.4f}")

    return metrics, model_path


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
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Voice Deepfake Detector — Training Pipeline")
    print("=" * 60)

    all_results = {}

    def train_both_classifiers(df: pd.DataFrame, ft: str, output_dir: Path, test_size: float = 0.2):
        """Train both GBM and XGBoost variants with train/test split, return results dict and best path."""
        from sklearn.model_selection import train_test_split

        results = {}
        best_local_f1 = -1
        best_local_path = None

        # Split data into train/test (80/20 stratified)
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df["label"]
        )
        print(f"\n  [Data Split] Train: {len(df_train)} | Test: {len(df_test)}")

        # Train Gradient Boosting (GBM)
        metrics_gbm, path_gbm = train_single(
            df_train, ft, output_dir, classifier_type="gradient_boosting", df_test=df_test
        )
        if metrics_gbm:
            results[f"{ft}_gbm"] = metrics_gbm
            # Use test F1 if available, otherwise CV F1
            f1_score = metrics_gbm.get("test_f1", metrics_gbm.get("cv_f1", 0))
            if f1_score > best_local_f1:
                best_local_f1 = f1_score
                best_local_path = path_gbm

        # Train XGBoost
        metrics_xgb, path_xgb = train_single(
            df_train, ft, output_dir, classifier_type="xgboost", df_test=df_test
        )
        if metrics_xgb:
            results[f"{ft}_xgboost"] = metrics_xgb
            f1_score = metrics_xgb.get("test_f1", metrics_xgb.get("cv_f1", 0))
            if f1_score > best_local_f1:
                best_local_f1 = f1_score
                best_local_path = path_xgb

        return results, best_local_path, best_local_f1

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
        print(f"\nLoading CSV: {csv_path}")
        ft, df = load_from_csv(csv_path)
        print(f"  Detected feature type: {ft} | Rows: {len(df)}")

        # If the CSV has hybrid/mfcc_legacy features, train all three variants
        cols = set(df.columns) - {"label", "_extraction_time"}
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
            hybrid_cols = [c for c in df.columns if c not in ("label", "_extraction_time")]
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
            print(f"\nLoading WAV files from default: {found}")
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
            print(f"\nLoading CSV from default: {found}")
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
            )

    # Save results table
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

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
