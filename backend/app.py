"""
FastAPI backend — Voice Deepfake Vishing Detector & Generator.

Endpoints:
  GET  /health          → {"status": "ok", ...}
  POST /detect          → WAV upload → {prediction, confidence, model_used, notes}
  POST /generate        → speaker WAV + text → base64 WAV audio

TTS engine priority (generation):
  1. IndexTTS2  (index-tts/index-tts) — zero-shot voice cloning, emotion control
  2. gTTS       (fallback, no cloning) — requires internet, generic voice

Audio format: WAV (mono, 16 kHz recommended).
WebM/OGG/MP3 are auto-converted via pydub/ffmpeg when available.

Installation:
  pip install -r backend/requirements.txt

For IndexTTS2 voice cloning (optional, requires PyTorch + ~4 GB VRAM or CPU):
  git clone https://github.com/index-tts/index-tts.git /opt/index-tts
  cd /opt/index-tts && pip install uv && uv sync --all-extras
  # Download model weights:
  huggingface-cli download IndexTeam/IndexTTS-2 --local-dir /opt/index-tts/checkpoints
  # Then set env var so this backend finds it:
  export INDEXTTS_DIR=/opt/index-tts
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

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

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("deepfake-api")

# ─── app ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Voice Deepfake API",
    description=(
        "Detect deepfake voices (MFCC/FFT/Hybrid) and generate voice clones "
        "using IndexTTS2 zero-shot TTS. Research use only."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── globals (lazy-loaded) ────────────────────────────────────────────────────
_detector = None          # loaded classifier dict
_indextts2 = None         # IndexTTS2 model instance
_indextts2_available = False
_gtts_available = False
_START_TIME = time.time()

# Path where the user cloned index-tts (configurable via env var)
INDEXTTS_DIR = Path(os.environ.get("INDEXTTS_DIR", "/opt/index-tts"))
INDEXTTS_CHECKPOINTS = INDEXTTS_DIR / "checkpoints"


# ─── detector loading ─────────────────────────────────────────────────────────

def _load_detector():
    """Load the best available detection model (lazy, priority order)."""
    global _detector
    if _detector is not None:
        return _detector

    candidates = [
        MODELS_DIR / "deepfake_detector_hybrid.pkl",
        MODELS_DIR / "deepfake_detector_best.pkl",
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
                if isinstance(obj, dict):
                    _detector = obj
                else:
                    _detector = {
                        "model": obj,
                        "model_name": path.stem,
                        "feature_type": "mfcc_hybrid",
                    }
                log.info(f"Detector loaded: {_detector.get('model_name', 'unknown')}")
                return _detector
    except Exception as e:
        log.error(f"Failed to load detector: {e}")

    _detector = None
    return None


# ─── TTS loading ──────────────────────────────────────────────────────────────

def _load_indextts2():
    """
    Try to load IndexTTS2 from the cloned index-tts repository.

    The user must:
      1. Clone https://github.com/index-tts/index-tts  →  INDEXTTS_DIR
      2. Run:  cd INDEXTTS_DIR && uv sync --all-extras
      3. Download model weights to INDEXTTS_DIR/checkpoints/
      4. Set env var:  INDEXTTS_DIR=/path/to/index-tts

    The backend adds INDEXTTS_DIR to sys.path so `indextts` can be imported.
    """
    global _indextts2, _indextts2_available

    if _indextts2_available:
        return True
    if not INDEXTTS_DIR.exists():
        log.info(
            f"IndexTTS2 not found at {INDEXTTS_DIR}. "
            "Set INDEXTTS_DIR env var to enable voice cloning."
        )
        return False
    if not INDEXTTS_CHECKPOINTS.exists():
        log.warning(
            f"IndexTTS2 directory found at {INDEXTTS_DIR} but 'checkpoints/' is missing. "
            "Download model weights: "
            "huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints"
        )
        return False

    indextts_src = str(INDEXTTS_DIR)
    if indextts_src not in sys.path:
        sys.path.insert(0, indextts_src)

    try:
        from indextts.infer_v2 import IndexTTS2  # type: ignore[import]

        log.info("Loading IndexTTS2 model (first load may take 30–60 s)…")
        _indextts2 = IndexTTS2(
            cfg_path=str(INDEXTTS_CHECKPOINTS / "config.yaml"),
            model_dir=str(INDEXTTS_CHECKPOINTS),
            use_fp16=False,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
        _indextts2_available = True
        log.info("IndexTTS2 loaded successfully.")
        return True
    except ImportError:
        log.warning(
            "Could not import indextts.infer_v2. "
            "Ensure you ran: cd $INDEXTTS_DIR && uv sync --all-extras"
        )
    except Exception as e:
        log.warning(f"IndexTTS2 load failed: {e}")

    _indextts2_available = False
    return False


def _check_gtts():
    global _gtts_available
    try:
        from gtts import gTTS  # noqa: F401

        _gtts_available = True
    except ImportError:
        _gtts_available = False
    return _gtts_available


# ─── audio utilities ──────────────────────────────────────────────────────────

def _convert_to_wav(src: Path, dst: Path) -> bool:
    """Convert any audio format → mono 16 kHz WAV via pydub/ffmpeg."""
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
    """Persist upload to uploads/, normalise to mono WAV 16 kHz. Returns WAV path."""
    uid = uuid.uuid4().hex
    raw_suffix = Path(upload.filename or "audio.wav").suffix.lower() or ".wav"
    raw_path = UPLOADS_DIR / f"{uid}_raw{raw_suffix}"
    wav_path = UPLOADS_DIR / f"{uid}.wav"

    data = upload.file.read()
    raw_path.write_bytes(data)

    if raw_suffix == ".wav":
        if not _convert_to_wav(raw_path, wav_path):
            wav_path = raw_path
    else:
        if not _convert_to_wav(raw_path, wav_path):
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Cannot convert '{raw_suffix}' to WAV. "
                    "Install ffmpeg + pydub, or upload WAV directly."
                ),
            )
    return wav_path


def _cleanup(*paths: Path) -> None:
    for p in paths:
        if p and p.exists():
            try:
                p.unlink()
            except Exception:
                pass


# ─── feature extraction ───────────────────────────────────────────────────────

def _mfcc_features(segment, sr, n_mfcc: int = 13):
    """Mean MFCC vector (n_mfcc dims)."""
    import numpy as np
    import scipy.fftpack as fftpack

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


def _fft_features(segment, sr):
    """6-dim spectral feature vector."""
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

    n = len(spectrum)
    low_e = np.log1p(np.mean(spectrum[: n // 3]))
    mid_e = np.log1p(np.mean(spectrum[n // 3 : 2 * n // 3]))
    high_e = np.log1p(np.mean(spectrum[2 * n // 3 :]))

    return np.array([centroid, bandwidth, rolloff, low_e, mid_e, high_e])


def _load_wav_segment(wav_path: Path):
    """Load WAV → mono float32 @ 16 kHz, return (segment_1s, sr)."""
    import numpy as np
    from scipy import signal as scipy_signal
    from scipy.io import wavfile

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
    seg_len = sr
    segment = data[:seg_len] if len(data) >= seg_len else np.pad(data, (0, seg_len - len(data)))
    return segment, sr


def _extract_features_for_inference(wav_path: Path, feature_type: str = "mfcc_hybrid"):
    import numpy as np

    segment, sr = _load_wav_segment(wav_path)
    mfcc = _mfcc_features(segment, sr)
    fft = _fft_features(segment, sr)

    if feature_type == "mfcc":
        return mfcc
    elif feature_type == "fft":
        return fft
    else:
        return np.concatenate([mfcc, fft])


def _extract_legacy_18_features(wav_path: Path):
    """18-dim features for original .pkl models (MFCC×13 + centroid/bandwidth/rolloff/jitter/shimmer)."""
    import numpy as np

    segment, sr = _load_wav_segment(wav_path)
    mfcc = _mfcc_features(segment, sr)

    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
    total = np.sum(spectrum) + 1e-10
    centroid = np.sum(freqs * spectrum) / total
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total)
    cum = np.cumsum(spectrum)
    rolloff_idx = np.searchsorted(cum, 0.85 * cum[-1])
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
    jitter = float(np.mean(np.abs(np.diff(segment))))
    shimmer = float(np.std(np.diff(np.abs(segment))))

    return np.concatenate([mfcc, [centroid, bandwidth, rolloff, jitter, shimmer]])


# ─── endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Returns service health and capability status."""
    detector = _load_detector()
    _check_gtts()
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "detector_loaded": detector is not None,
        "model_name": detector.get("model_name", "none") if detector else "none",
        "tts_engine": (
            "indextts2" if _indextts2_available
            else "gtts_fallback" if _gtts_available
            else "none"
        ),
        "indextts2_available": _indextts2_available,
        "indextts2_dir": str(INDEXTTS_DIR),
        "gtts_available": _gtts_available,
    }


@app.post("/detect")
async def detect(audio: UploadFile = File(...)):
    """
    Classify an audio file as real or deepfake.

    Upload a WAV file (any sample rate/channels — auto-converted).
    Returns: prediction, confidence, model_used, feature_type, inference_time_s
    """
    if not audio.filename:
        raise HTTPException(400, "No filename provided.")

    t0 = time.time()
    wav_path = None
    try:
        wav_path = _save_upload(audio)
        detector = _load_detector()
        if detector is None:
            raise HTTPException(
                503,
                "Detection model not loaded. Run: python training/train.py --csv osr_features.csv",
            )

        model = detector["model"]
        feature_type = detector.get("feature_type", "mfcc_hybrid")
        feat_cols = detector.get("feature_columns", None)

        import numpy as np
        import pandas as pd

        col_map = {
            "mfcc": [f"MFCC{i+1}" for i in range(13)],
            "fft": ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"],
            "hybrid": (
                [f"MFCC{i+1}" for i in range(13)]
                + ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"]
            ),
            "mfcc_hybrid": (
                [f"MFCC{i+1}" for i in range(13)]
                + ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"]
            ),
        }

        expected_cols = feat_cols or col_map.get(feature_type, col_map["mfcc_hybrid"])
        feats = _extract_features_for_inference(wav_path, feature_type)

        # Handle legacy 18-feature models
        if len(feats) != len(expected_cols):
            feats = _extract_legacy_18_features(wav_path)
            expected_cols = (
                [f"MFCC{i+1}" for i in range(13)]
                + ["centroid", "bandwidth", "rolloff", "jitter", "shimmer"]
            )

        df = pd.DataFrame([feats], columns=expected_cols)
        pred_raw = model.predict(df)[0]

        confidence = 0.5
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            confidence = float(max(proba))

        if str(pred_raw) in ("1", "fake", "deepfake"):
            prediction = "fake"
            notes = "Deepfake voice detected"
        else:
            prediction = "real"
            notes = "Real voice detected"

        return JSONResponse({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model_used": detector.get("model_name", "unknown"),
            "feature_type": feature_type,
            "inference_time_s": round(time.time() - t0, 3),
            "notes": notes,
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Detection failed: {e}")
    finally:
        _cleanup(wav_path)


@app.post("/generate")
async def generate(
    audio: UploadFile = File(...),
    text: str = Form(...),
):
    """
    Clone the voice from `audio` and synthesise `text`.

    TTS engine priority:
      1. IndexTTS2  (zero-shot voice cloning — requires INDEXTTS_DIR env var)
      2. gTTS       (generic fallback, requires internet, NOT a voice clone)

    Returns: audio_base64 (WAV), method_used, generation_time_s, notes
    """
    if not audio.filename:
        raise HTTPException(400, "No speaker audio provided.")
    text = text.strip()
    if not text:
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

        method_used = "none"
        notes = ""

        # ── Attempt 1: IndexTTS2 ──────────────────────────────────────────────
        if _load_indextts2():
            try:
                log.info(f"IndexTTS2: synthesising '{text[:60]}…'")
                _indextts2.infer(
                    spk_audio_prompt=str(speaker_wav),
                    text=text,
                    output_path=str(out_path),
                    verbose=False,
                )
                if out_path.exists() and out_path.stat().st_size > 0:
                    method_used = "indextts2"
                    notes = (
                        "Voice cloned using IndexTTS2 zero-shot TTS "
                        "(Bilibili/index-tts). Emotion and timbre are "
                        "derived from the speaker reference audio."
                    )
                else:
                    log.warning("IndexTTS2 produced no output, trying fallback.")
            except Exception as e:
                log.warning(f"IndexTTS2 synthesis failed: {e} — trying gTTS fallback.")

        # ── Attempt 2: gTTS fallback ──────────────────────────────────────────
        if method_used == "none":
            _check_gtts()
            if _gtts_available:
                try:
                    from gtts import gTTS

                    log.info("Using gTTS fallback (generic voice, not a clone).")
                    tts = gTTS(text=text, lang="en")
                    tts.save(str(out_path))
                    if out_path.exists() and out_path.stat().st_size > 0:
                        method_used = "gtts_fallback"
                        notes = (
                            "⚠ gTTS fallback used — this is a generic Google TTS voice, "
                            "NOT a clone of the uploaded speaker. "
                            "To enable real voice cloning, install IndexTTS2 and set "
                            "the INDEXTTS_DIR environment variable."
                        )
                except Exception as e:
                    log.error(f"gTTS also failed: {e}")

        if method_used == "none":
            raise HTTPException(
                503,
                "No TTS engine available.\n"
                "• For voice cloning: install IndexTTS2 and set INDEXTTS_DIR env var.\n"
                "• For basic TTS fallback: pip install gtts",
            )

        audio_bytes = out_path.read_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        return JSONResponse({
            "success": True,
            "method_used": method_used,
            "audio_base64": audio_b64,
            "audio_mime": "audio/wav",
            "generation_time_s": round(time.time() - t0, 3),
            "notes": notes,
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Generation failed: {e}")
    finally:
        _cleanup(speaker_wav, out_path)


# ─── entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
