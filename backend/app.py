"""
FastAPI backend for Voice Deepfake Vishing Detector & Generator.

Endpoints:
  GET  /health          → {"status": "ok", ...}
  POST /detect          → WAV upload → {prediction, confidence, model_used, notes}
  POST /generate        → speaker WAV + text → generated audio (base64 or file)

Audio format: WAV (mono, 16 kHz recommended).
WebM/OGG/MP3 are auto-converted via pydub/ffmpeg when available.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

# FastAPI & friends
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─── path setup ───────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.resolve()
ROOT_DIR = BACKEND_DIR.parent.resolve()
MODELS_DIR = ROOT_DIR / "models"
UPLOADS_DIR = BACKEND_DIR / "uploads"
GENERATED_DIR = BACKEND_DIR / "generated"
UPLOADS_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)

# Add root to path so we can import pipeline utilities
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("deepfake-api")

# ─── app ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Voice Deepfake API",
    description="Detect deepfake voices and generate voice clones for research purposes.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── lazy-loaded globals ──────────────────────────────────────────────────────
_detector = None          # loaded classifier dict
_tts_model = None         # Coqui TTS model
_tts_available = False    # whether Coqui loaded OK
_gtts_available = False   # whether gTTS is available

_START_TIME = time.time()


def _load_detector():
    """Load the best available detection model (lazy)."""
    global _detector
    if _detector is not None:
        return _detector

    # Preference order: hybrid > mfcc > osr > final > base
    candidates = [
        MODELS_DIR / "deepfake_detector_hybrid.pkl",
        MODELS_DIR / "deepfake_detector_mfcc.pkl",
        ROOT_DIR / "deepfake_detector_osr.pkl",
        ROOT_DIR / "deepfake_detector_final.pkl",
        ROOT_DIR / "deepfake_detector.pkl",
    ]
    try:
        import joblib
        for path in candidates:
            if path.exists():
                log.info(f"Loading detector: {path}")
                obj = joblib.load(path)
                # Support both raw classifier and wrapped dict
                if isinstance(obj, dict):
                    _detector = obj
                else:
                    _detector = {"model": obj, "model_name": path.stem, "feature_type": "mfcc_hybrid"}
                log.info(f"Detector loaded: {_detector.get('model_name', 'unknown')}")
                return _detector
    except Exception as e:
        log.error(f"Failed to load detector: {e}")
    _detector = None
    return None


def _load_tts():
    """Attempt to load Coqui TTS once."""
    global _tts_model, _tts_available, _gtts_available
    if _tts_available or _tts_model is not None:
        return _tts_available

    try:
        from TTS.api import TTS
        log.info("Loading Coqui TTS model (first call may download ~200 MB)…")
        _tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
        _tts_available = True
        log.info("Coqui TTS ready.")
    except Exception as e:
        log.warning(f"Coqui TTS not available: {e}. Will use gTTS fallback.")
        _tts_available = False

    try:
        from gtts import gTTS  # noqa: F401
        _gtts_available = True
    except ImportError:
        _gtts_available = False

    return _tts_available


# ─── audio utilities ──────────────────────────────────────────────────────────

def _convert_to_wav(src: Path, dst: Path) -> bool:
    """Try to convert any audio format to WAV mono 16 kHz. Returns True on success."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(src))
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(str(dst), format="wav")
        return True
    except Exception as e:
        log.warning(f"pydub conversion failed: {e}")
    return False


def _save_upload(upload: UploadFile) -> Path:
    """Save upload to temp dir, converting to WAV if needed. Returns WAV path."""
    uid = uuid.uuid4().hex
    raw_suffix = Path(upload.filename or "audio.wav").suffix.lower() or ".wav"
    raw_path = UPLOADS_DIR / f"{uid}_raw{raw_suffix}"
    wav_path = UPLOADS_DIR / f"{uid}.wav"

    data = upload.file.read()
    raw_path.write_bytes(data)

    if raw_suffix == ".wav":
        # Still normalise to mono/16k
        if not _convert_to_wav(raw_path, wav_path):
            wav_path = raw_path  # use as-is
    else:
        if not _convert_to_wav(raw_path, wav_path):
            raise HTTPException(
                status_code=415,
                detail=f"Could not convert {raw_suffix} to WAV. Install ffmpeg and pydub, or upload WAV directly.",
            )

    return wav_path


# ─── feature extraction ───────────────────────────────────────────────────────

def _extract_features_for_inference(wav_path: Path, feature_type: str = "mfcc_hybrid"):
    """Extract features matching what the model was trained on."""
    import numpy as np
    from scipy.io import wavfile
    from scipy import signal as scipy_signal

    sr, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    mx = np.max(np.abs(data))
    if mx > 0:
        data /= mx

    # Resample to 16 kHz
    target_sr = 16000
    if sr != target_sr:
        new_len = int(len(data) * target_sr / sr)
        data = scipy_signal.resample(data, new_len)
        sr = target_sr

    # Use 1 second window
    seg_len = sr
    segment = data[:seg_len] if len(data) >= seg_len else np.pad(data, (0, seg_len - len(data)))

    mfcc_feats = _mfcc_features(segment, sr)       # 13 dims
    fft_feats = _fft_features(segment, sr)          # 6 dims

    if feature_type == "mfcc":
        return mfcc_feats
    elif feature_type == "fft":
        return fft_feats
    else:  # hybrid
        return np.concatenate([mfcc_feats, fft_feats])


def _mfcc_features(segment, sr):
    """13-dim MFCC mean vector (matches training pipeline)."""
    import numpy as np
    import scipy.fftpack as fftpack

    pre_emphasis = 0.97
    emphasized = np.append(segment[0], segment[1:] - pre_emphasis * segment[:-1])

    frame_length = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    n = len(emphasized)
    n_frames = max(1, int(np.ceil((n - frame_length) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_length
    padded = np.append(emphasized, np.zeros(max(0, pad_len - n)))

    idx = (
        np.tile(np.arange(frame_length), (n_frames, 1))
        + np.tile(np.arange(n_frames) * frame_step, (frame_length, 1)).T
    )
    frames = padded[idx.astype(np.int32)] * np.hamming(frame_length)

    NFFT = 512
    mag = np.abs(np.fft.rfft(frames, NFFT))
    power = (1.0 / NFFT) * mag ** 2

    nfilt = 26
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_pts = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_f = np.floor((NFFT + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((nfilt, NFFT // 2 + 1))
    for m in range(1, nfilt + 1):
        for k in range(bin_f[m - 1], bin_f[m]):
            fbank[m - 1, k] = (k - bin_f[m - 1]) / (bin_f[m] - bin_f[m - 1] + 1e-10)
        for k in range(bin_f[m], bin_f[m + 1]):
            fbank[m - 1, k] = (bin_f[m + 1] - k) / (bin_f[m + 1] - bin_f[m] + 1e-10)

    fb = np.dot(power, fbank.T)
    fb = np.where(fb == 0, np.finfo(float).eps, fb)
    fb = 20 * np.log10(fb)
    mfcc = fftpack.dct(fb, type=2, axis=1, norm="ortho")[:, :13]
    return np.mean(mfcc, axis=0)


def _fft_features(segment, sr):
    """6-dim spectral feature vector (centroid, bandwidth, rolloff, band energies x3)."""
    import numpy as np

    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
    eps = 1e-10
    total = np.sum(spectrum) + eps
    centroid = np.sum(freqs * spectrum) / total
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total)
    cum = np.cumsum(spectrum)
    rolloff_idx = np.searchsorted(cum, 0.85 * cum[-1])
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    # Log band energies (low/mid/high)
    n = len(spectrum)
    low_e = np.log1p(np.mean(spectrum[: n // 3]))
    mid_e = np.log1p(np.mean(spectrum[n // 3 : 2 * n // 3]))
    high_e = np.log1p(np.mean(spectrum[2 * n // 3 :]))

    return np.array([centroid, bandwidth, rolloff, low_e, mid_e, high_e])


# ─── endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    detector = _load_detector()
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "detector_loaded": detector is not None,
        "model_name": detector.get("model_name", "none") if detector else "none",
        "tts_available": _tts_available,
        "gtts_available": _gtts_available,
    }


@app.post("/detect")
async def detect(audio: UploadFile = File(...)):
    """
    Classify an audio file as real or deepfake.

    Returns:
        prediction: "real" | "fake"
        confidence: float 0.0–1.0
        model_used: str
        notes: str
    """
    if not audio.filename:
        raise HTTPException(400, "No filename provided.")

    t0 = time.time()
    wav_path = None
    try:
        wav_path = _save_upload(audio)
        detector = _load_detector()
        if detector is None:
            raise HTTPException(503, "Detection model not loaded. Run training pipeline first.")

        model = detector["model"]
        feature_type = detector.get("feature_type", "mfcc_hybrid")

        feats = _extract_features_for_inference(wav_path, feature_type)

        # Build DataFrame with correct column names
        import pandas as pd
        import numpy as np
        col_map = {
            "mfcc": [f"MFCC{i+1}" for i in range(13)],
            "fft": ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"],
            "mfcc_hybrid": (
                [f"MFCC{i+1}" for i in range(13)]
                + ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"]
            ),
        }
        cols = col_map.get(feature_type, col_map["mfcc_hybrid"])
        # Handle legacy 18-feature models (MFCC x13 + centroid/bandwidth/rolloff/jitter/shimmer)
        if len(feats) != len(cols):
            feats_18 = _extract_legacy_18_features(wav_path)
            legacy_cols = [f"MFCC{i+1}" for i in range(13)] + ["centroid", "bandwidth", "rolloff", "jitter", "shimmer"]
            df = pd.DataFrame([feats_18], columns=legacy_cols)
        else:
            df = pd.DataFrame([feats], columns=cols)

        pred_raw = model.predict(df)[0]
        confidence = 0.5
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            confidence = float(max(proba))

        # Normalise label
        if str(pred_raw) in ("1", "fake", "deepfake"):
            prediction = "fake"
            label_str = "Deepfake voice detected"
        else:
            prediction = "real"
            label_str = "Real voice detected"

        elapsed = round(time.time() - t0, 3)
        return JSONResponse({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model_used": detector.get("model_name", "unknown"),
            "feature_type": feature_type,
            "inference_time_s": elapsed,
            "notes": label_str,
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Detection failed: {e}")
    finally:
        # Clean up uploaded file
        if wav_path and wav_path.exists():
            try:
                wav_path.unlink()
            except Exception:
                pass


def _extract_legacy_18_features(wav_path: Path):
    """Extract 18 features matching original pipeline (for old .pkl models)."""
    import numpy as np
    from scipy.io import wavfile
    from scipy import signal as scipy_signal

    sr, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    mx = np.max(np.abs(data))
    if mx > 0:
        data /= mx
    if sr != 16000:
        data = scipy_signal.resample(data, int(len(data) * 16000 / sr))
        sr = 16000
    seg = data[:sr] if len(data) >= sr else np.pad(data, (0, sr - len(data)))

    mfcc = _mfcc_features(seg, sr)   # 13

    spectrum = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    total = np.sum(spectrum) + 1e-10
    centroid = np.sum(freqs * spectrum) / total
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total)
    cum = np.cumsum(spectrum)
    rolloff_idx = np.searchsorted(cum, 0.85 * cum[-1])
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    diff = np.diff(seg)
    jitter = float(np.mean(np.abs(diff)))
    shimmer = float(np.std(np.diff(np.abs(seg))))

    return np.concatenate([mfcc, [centroid, bandwidth, rolloff, jitter, shimmer]])


@app.post("/generate")
async def generate(
    audio: UploadFile = File(...),
    text: str = Form(...),
):
    """
    Clone the voice in `audio` and synthesise `text`.

    Returns base64-encoded WAV + metadata.
    """
    if not audio.filename:
        raise HTTPException(400, "No speaker audio provided.")
    if not text or not text.strip():
        raise HTTPException(400, "text field is required and must not be empty.")
    if len(text) > 500:
        raise HTTPException(400, "text must be 500 characters or fewer.")

    t0 = time.time()
    speaker_wav = None
    out_path = None
    try:
        speaker_wav = _save_upload(audio)
        uid = uuid.uuid4().hex
        out_path = GENERATED_DIR / f"generated_{uid}.wav"

        _load_tts()  # ensure attempt was made

        method_used = "none"
        if _tts_available and _tts_model is not None:
            try:
                _tts_model.tts_to_file(
                    text=text.strip(),
                    speaker_wav=str(speaker_wav),
                    file_path=str(out_path),
                    language="en",
                )
                method_used = "coqui_your_tts"
            except Exception as e:
                log.warning(f"Coqui TTS failed: {e} — trying gTTS fallback")

        if method_used == "none" and _gtts_available:
            from gtts import gTTS
            tts = gTTS(text=text.strip(), lang="en")
            tts.save(str(out_path))
            method_used = "gtts_fallback"

        if method_used == "none":
            raise HTTPException(503, "No TTS engine available. Install TTS or gtts.")

        if not out_path.exists():
            raise HTTPException(500, "Generated file missing after synthesis.")

        audio_bytes = out_path.read_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        elapsed = round(time.time() - t0, 3)

        return JSONResponse({
            "success": True,
            "method_used": method_used,
            "audio_base64": audio_b64,
            "audio_mime": "audio/wav",
            "generation_time_s": elapsed,
            "notes": (
                "Voice cloned via Coqui YourTTS." if method_used == "coqui_your_tts"
                else "gTTS fallback used — voice is generic, not a clone of input speaker."
            ),
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Generation failed: {e}")
    finally:
        for p in [speaker_wav, out_path]:
            if p and p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass


# ─── run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
