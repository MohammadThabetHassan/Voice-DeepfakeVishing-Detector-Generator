"""
FastAPI backend — Voice Deepfake Vishing Detector & Generator.

Endpoints:
  GET  /health          → {"status": "ok", ...}
  POST /detect          → WAV upload → {prediction, confidence, model_used, notes}
  POST /generate        → speaker WAV + text → base64 WAV audio
  POST /convert-voice   → source WAV + target WAV → base64 WAV (voice converted)

TTS engine priority (generation):
  1. XTTS v2    (coqui/TTS) — best voice cloning, pip installable
  2. IndexTTS2  (index-tts/index-tts) — zero-shot voice cloning, emotion control
  3. gTTS       (fallback, no cloning) — requires internet, generic voice

Voice conversion priority:
  1. Coqui TTS XTTS v2 (zero-shot voice conversion)
  2. Pitch/timbre matching algorithm (fallback)

Audio format: WAV (mono, 16 kHz recommended).
WebM/OGG/MP3 are auto-converted via pydub/ffmpeg when available.

Installation:
  pip install -r backend/requirements.txt

For XTTS v2 voice cloning (recommended, pip installable):
  pip install TTS>=0.22.0
  # First run will auto-download the model (~2 GB)

For IndexTTS2 voice cloning (optional, requires PyTorch + ~4 GB VRAM or CPU):
  git clone https://github.com/index-tts/index-tts.git /opt/index-tts
  cd /opt/index-tts && pip install uv && uv sync --all-extras
  # Download model weights:
  huggingface-cli download IndexTeam/IndexTTS-2 --local-dir /opt/index-tts/checkpoints
  # Then set env var so this backend finds it:
  export INDEXTTS_DIR=/opt/index-tts
"""

import base64
import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

# ─── security configuration ───────────────────────────────────────────────────
# File upload limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_AUDIO_DURATION_SECONDS = 60  # 60 seconds
MIN_AUDIO_DURATION_SECONDS = 0.5  # 500 ms

# CORS configuration - allow specific origins in production
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
if CORS_ORIGINS == ["*"]:
    log.warning("CORS configured to allow all origins. Set CORS_ORIGINS env var for production.")

# Rate limiting configuration
RATE_LIMIT_HEALTH = os.environ.get("RATE_LIMIT_HEALTH", "60/minute")
RATE_LIMIT_DETECT = os.environ.get("RATE_LIMIT_DETECT", "30/minute")
RATE_LIMIT_GENERATE = os.environ.get("RATE_LIMIT_GENERATE", "10/minute")
RATE_LIMIT_CONVERT = os.environ.get("RATE_LIMIT_CONVERT", "10/minute")

# ─── rate limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

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

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── globals (lazy-loaded) ────────────────────────────────────────────────────
_detector = None  # loaded classifier dict
_xtts = None  # XTTS v2 model instance
_xtts_available = False
_indextts2 = None  # IndexTTS2 model instance
_indextts2_available = False
_gtts_available = False
_coqui_tts_available = False  # Coqui TTS for voice conversion
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
        MODELS_DIR / "deepfake_detector_ensemble.pkl",
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
                    # Check if this is an ensemble model
                    if _detector.get("feature_type") == "ensemble":
                        log.info("Ensemble model loaded (soft voting)")
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


def _load_xtts():
    """
    Try to load Coqui XTTS v2 for voice cloning.

    XTTS v2 is pip-installable and provides high-quality voice cloning:
      pip install TTS>=0.22.0

    The model will auto-download on first use (~2 GB).
    """
    global _xtts, _xtts_available

    if _xtts_available:
        return True

    try:
        from TTS.api import TTS

        log.info("Loading XTTS v2 model (first load may take 30–60 s)…")
        _xtts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        _xtts_available = True
        log.info("XTTS v2 loaded successfully.")
        return True
    except ImportError:
        log.info("XTTS v2 not available. Install with: pip install TTS>=0.22.0")
    except Exception as e:
        log.warning(f"XTTS v2 load failed: {e}")

    _xtts_available = False
    return False


def _check_gtts():
    global _gtts_available
    try:
        from gtts import gTTS  # noqa: F401

        _gtts_available = True
    except ImportError:
        _gtts_available = False
    return _gtts_available


def _check_coqui_tts():
    """Check if Coqui TTS is available for voice conversion."""
    global _coqui_tts_available
    try:
        from TTS.api import TTS  # noqa: F401

        _coqui_tts_available = True
        log.info("Coqui TTS is available for voice conversion.")
    except ImportError:
        _coqui_tts_available = False
        log.info("Coqui TTS not available. Will use fallback methods.")
    return _coqui_tts_available


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


def _validate_file_size(upload: UploadFile) -> None:
    """Validate file size is within limits."""
    # Check content-length header if available
    content_length = upload.size
    if content_length and content_length > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.1f} MB.",
        )


def _validate_audio_duration(wav_path: Path) -> None:
    """Validate audio duration is within limits."""
    try:
        from scipy.io import wavfile

        sr, data = wavfile.read(str(wav_path))
        duration = len(data) / sr

        if duration < MIN_AUDIO_DURATION_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short. Minimum duration is {MIN_AUDIO_DURATION_SECONDS}s.",
            )
        if duration > MAX_AUDIO_DURATION_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long. Maximum duration is {MAX_AUDIO_DURATION_SECONDS}s.",
            )
    except HTTPException:
        raise
    except Exception as e:
        log.warning(f"Could not validate audio duration: {e}")
        # Don't fail if we can't validate duration


def _save_upload(
    upload: UploadFile,
    apply_noise_reduction: bool = False,
    apply_normalization: bool = False,
) -> Path:
    """Persist upload to uploads/, normalise to mono WAV 16 kHz. Returns WAV path."""
    # Validate file size
    _validate_file_size(upload)

    uid = uuid.uuid4().hex
    raw_suffix = Path(upload.filename or "audio.wav").suffix.lower() or ".wav"
    raw_path = UPLOADS_DIR / f"{uid}_raw{raw_suffix}"
    wav_path = UPLOADS_DIR / f"{uid}.wav"

    data = upload.file.read()

    # Check actual file size after reading
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.1f} MB.",
        )

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

    # Apply optional preprocessing
    if apply_noise_reduction:
        wav_path = _reduce_noise(wav_path)
    if apply_normalization:
        wav_path = _normalize_loudness(wav_path)

    # Validate audio duration after conversion
    _validate_audio_duration(wav_path)

    return wav_path


def _reduce_noise(wav_path: Path) -> Path:
    """Apply noise reduction using noisereduce library. Returns path to processed file."""
    try:
        import numpy as np
        import noisereduce as nr
        from scipy.io import wavfile

        log.info(f"Applying noise reduction: {wav_path.name}")
        sr, data = wavfile.read(str(wav_path))

        if data.ndim > 1:
            data = data.mean(axis=1)

        data = data.astype(np.float32)
        mx = np.max(np.abs(data))
        if mx > 0:
            data /= mx

        reduced = nr.reduce_noise(y=data, sr=sr, prop_decrease=0.75)

        reduced = np.clip(reduced, -1.0, 1.0)
        if mx > 0:
            reduced = (reduced * mx).astype(data.dtype)

        uid = uuid.uuid4().hex
        out_path = UPLOADS_DIR / f"{uid}_denoised.wav"
        wavfile.write(str(out_path), sr, (reduced * 32767).astype(np.int16))

        log.info(f"Noise reduction complete: {out_path.name}")
        return out_path
    except Exception as e:
        log.warning(f"Noise reduction failed: {e} — returning original file")
        return wav_path


def _normalize_loudness(wav_path: Path, target_lufs: float = -23.0) -> Path:
    """Normalize audio to target LUFS using pyloudnorm. Returns path to processed file."""
    try:
        import numpy as np
        import pyloudnorm as pyln
        from scipy.io import wavfile

        log.info(f"Normalizing loudness to {target_lufs} LUFS: {wav_path.name}")
        sr, data = wavfile.read(str(wav_path))

        if data.ndim > 1:
            data = data.mean(axis=1)

        data = data.astype(np.float32)
        mx = np.max(np.abs(data))
        if mx > 0:
            data /= mx

        meter = pyln.Meter(sr)
        current_loudness = meter.integrated_loudness(data)

        gain_db = target_lufs - current_loudness
        gain_linear = 10 ** (gain_db / 20)

        normalized = data * gain_linear
        normalized = np.clip(normalized, -1.0, 1.0)

        uid = uuid.uuid4().hex
        out_path = UPLOADS_DIR / f"{uid}_normalized.wav"
        wavfile.write(str(out_path), sr, (normalized * 32767).astype(np.int16))

        log.info(f"Loudness normalization complete: {out_path.name}")
        return out_path
    except Exception as e:
        log.warning(f"Loudness normalization failed: {e} — returning original file")
        return wav_path


def _cleanup(*paths: Path) -> None:
    for p in paths:
        if p and p.exists():
            try:
                p.unlink()
            except Exception:
                pass


# ─── voice conversion utilities ───────────────────────────────────────────────


def _convert_voice_xtts(source_wav: Path, target_wav: Path, output_wav: Path) -> bool:
    """
    Convert voice using Coqui TTS XTTS v2 voice conversion.
    Returns True if successful.
    """
    try:
        from TTS.api import TTS

        log.info("Loading XTTS v2 model for voice conversion...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        log.info(f"Converting voice: {source_wav.name} → {target_wav.name}")
        tts.voice_conversion_to_file(
            source_wav=str(source_wav),
            target_wav=str(target_wav),
            file_path=str(output_wav),
        )

        if output_wav.exists() and output_wav.stat().st_size > 0:
            log.info("XTTS voice conversion completed successfully.")
            return True
        else:
            log.warning("XTTS voice conversion produced no output.")
            return False

    except Exception as e:
        log.warning(f"XTTS voice conversion failed: {e}")
        return False


def _analyze_voice_characteristics(wav_path: Path) -> dict:
    """
    Analyze voice characteristics (pitch, formants, timbre) from audio.
    Returns dict with analysis results.
    """
    import numpy as np
    from scipy.io import wavfile

    sr, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    # Normalize
    mx = np.max(np.abs(data))
    if mx > 0:
        data /= mx

    # Resample to 16kHz if needed
    if sr != 16000:
        import scipy.signal as scipy_signal

        data = scipy_signal.resample(data, int(len(data) * 16000 / sr))
        sr = 16000

    # Extract pitch (F0)
    frame_len = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    min_freq, max_freq = 50, 500
    min_period = int(sr / max_freq)
    max_period = int(sr / min_freq)

    pitches = []
    n = len(data)
    n_frames = max(1, int(np.ceil((n - frame_len) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_len
    padded = np.append(data, np.zeros(max(0, pad_len - n)))

    for i in range(n_frames):
        frame = padded[i * frame_step : i * frame_step + frame_len]
        if len(frame) < frame_len:
            continue
        max_amp = np.max(np.abs(frame))
        if max_amp < 0.01:
            continue

        autocorr = np.correlate(frame, frame, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        if len(autocorr) > max_period:
            search_region = autocorr[min_period:max_period]
            if len(search_region) > 0:
                peak_idx = np.argmax(search_region)
                peak_val = search_region[peak_idx]
                if peak_val > 0.5:
                    pitch = sr / (min_period + peak_idx)
                    pitches.append(pitch)

    pitch_mean = np.mean(pitches) if pitches else 150.0
    pitch_std = np.std(pitches) if pitches else 20.0

    # Estimate formants using LPC
    formants = _estimate_formants(data, sr)

    # Spectral centroid
    spectrum = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), d=1.0 / sr)
    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "formants": formants,
        "spectral_centroid": centroid,
        "sample_rate": sr,
    }


def _estimate_formants(signal: np.ndarray, sr: int, num_formants: int = 4) -> list:
    """
    Estimate formant frequencies using LPC (Linear Predictive Coding).
    """
    import numpy as np
    from scipy.signal import lfilter

    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Frame the signal
    frame_len = int(0.025 * sr)
    frame_step = int(0.010 * sr)

    # Use middle frame for analysis
    mid = len(emphasized) // 2
    start = max(0, mid - frame_len // 2)
    frame = emphasized[start : start + frame_len]

    if len(frame) < frame_len:
        frame = np.pad(frame, (0, frame_len - len(frame)))

    # Apply Hamming window
    frame = frame * np.hamming(len(frame))

    # LPC order (typically 2*num_formants)
    lpc_order = 2 * num_formants

    try:
        # Autocorrelation method for LPC
        r = np.correlate(frame, frame, mode="full")
        r = r[len(r) // 2 : len(r) // 2 + lpc_order + 1]

        # Levinson-Durbin recursion
        a = np.zeros(lpc_order + 1)
        a[0] = 1.0
        e = r[0]

        for i in range(1, lpc_order + 1):
            k = r[i]
            for j in range(1, i):
                k -= a[j] * r[i - j]
            k /= e + 1e-10
            a[i] = k
            for j in range(1, i):
                a[j] -= k * a[i - j]
            e *= 1 - k * k

        # Find roots of LPC polynomial
        roots = np.roots(a)

        # Keep only roots inside unit circle (stable filters)
        roots = roots[np.abs(roots) < 1]

        # Convert to frequencies
        angles = np.angle(roots)
        freqs = angles * sr / (2 * np.pi)

        # Keep positive frequencies and sort
        formants = sorted([f for f in freqs if f > 50])

        return formants[:num_formants]
    except Exception:
        # Return default formant values if analysis fails
        return [500, 1500, 2500, 3500]


def _pitch_shift_audio(input_wav: Path, output_wav: Path, semitones: float) -> bool:
    """
    Pitch shift audio by specified semitones using librosa or pydub.
    Returns True if successful.
    """
    try:
        import librosa
        import soundfile as sf
        import numpy as np

        # Load audio
        y, sr = librosa.load(str(input_wav), sr=None)

        # Pitch shift
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

        # Save output
        sf.write(str(output_wav), y_shifted, sr)

        return True
    except ImportError:
        log.warning("librosa not available for pitch shifting.")
        return False
    except Exception as e:
        log.warning(f"Pitch shifting failed: {e}")
        return False


def _apply_timbre_matching(source_wav: Path, target_wav: Path, output_wav: Path) -> bool:
    """
    Apply pitch/timbre matching as fallback voice conversion.
    This uses spectral equalization and pitch shifting.
    """
    import numpy as np
    from scipy.io import wavfile

    try:
        # Analyze both voices
        source_chars = _analyze_voice_characteristics(source_wav)
        target_chars = _analyze_voice_characteristics(target_wav)

        # Load source audio
        sr, source_data = wavfile.read(str(source_wav))
        if source_data.ndim > 1:
            source_data = source_data.mean(axis=1)
        source_data = source_data.astype(np.float32)

        # Normalize
        mx = np.max(np.abs(source_data))
        if mx > 0:
            source_data /= mx

        # Resample to 16kHz if needed
        if sr != 16000:
            import scipy.signal as scipy_signal

            source_data = scipy_signal.resample(source_data, int(len(source_data) * 16000 / sr))
            sr = 16000

        # Calculate pitch shift needed
        pitch_shift_ratio = target_chars["pitch_mean"] / (source_chars["pitch_mean"] + 1e-10)
        semitones = 12 * np.log2(pitch_shift_ratio)
        semitones = np.clip(semitones, -12, 12)  # Limit to reasonable range

        log.info(f"Applying pitch shift: {semitones:.2f} semitones")

        # Try librosa pitch shift first
        temp_shifted = output_wav.parent / f"{output_wav.stem}_shifted.wav"
        if _pitch_shift_audio(source_wav, temp_shifted, semitones):
            # Apply spectral equalization
            sr_shifted, shifted_data = wavfile.read(str(temp_shifted))
            shifted_data = shifted_data.astype(np.float32)

            # Simple spectral envelope matching using FFT
            source_spectrum = np.abs(np.fft.rfft(source_data))
            target_spectrum = np.abs(
                np.fft.rfft(wavfile.read(str(target_wav))[1].astype(np.float32))
            )

            # Calculate spectral ratio for equalization
            min_len = min(len(source_spectrum), len(target_spectrum))
            spectral_ratio = target_spectrum[:min_len] / (source_spectrum[:min_len] + 1e-10)
            spectral_ratio = np.clip(spectral_ratio, 0.1, 10)  # Limit extremes

            # Apply to shifted audio
            shifted_fft = np.fft.rfft(shifted_data)
            min_len = min(len(shifted_fft), len(spectral_ratio))
            shifted_fft[:min_len] *= spectral_ratio[:min_len]
            result = np.fft.irfft(shifted_fft)

            # Normalize and save
            result = np.clip(result, -1.0, 1.0)
            result = (result * 32767).astype(np.int16)
            wavfile.write(str(output_wav), sr, result)

            _cleanup(temp_shifted)
            return True

        # Fallback: simple pitch shifting via resampling
        if abs(semitones) > 0.5:
            ratio = 2 ** (semitones / 12)
            new_len = int(len(source_data) / ratio)
            import scipy.signal as scipy_signal

            resampled = scipy_signal.resample(source_data, new_len)
            # Stretch back to original length
            stretched = scipy_signal.resample(resampled, len(source_data))
            source_data = stretched

        # Apply mild spectral filtering
        source_fft = np.fft.rfft(source_data)

        # Boost frequencies based on target spectral centroid
        centroid_ratio = target_chars["spectral_centroid"] / (
            source_chars["spectral_centroid"] + 1e-10
        )
        freqs = np.fft.rfftfreq(len(source_data), d=1.0 / sr)

        # Simple high-shelf filter based on centroid difference
        if centroid_ratio > 1.2:
            boost = 1 + 0.3 * (freqs / (sr / 2))
            source_fft *= boost
        elif centroid_ratio < 0.8:
            cut = 1 - 0.2 * (freqs / (sr / 2))
            source_fft *= cut

        result = np.fft.irfft(source_fft)
        result = np.clip(result, -1.0, 1.0)
        result = (result * 32767).astype(np.int16)
        wavfile.write(str(output_wav), sr, result)

        return True

    except Exception as e:
        log.warning(f"Timbre matching failed: {e}")
        return False


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


def _extract_pitch(segment: np.ndarray, sr: int) -> tuple:
    """
    Extract fundamental frequency (F0) using autocorrelation method.
    Returns (mean_pitch, std_pitch).
    """
    frame_len = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    min_freq = 50
    max_freq = 500
    min_period = int(sr / max_freq)
    max_period = int(sr / min_freq)

    n = len(segment)
    n_frames = max(1, int(np.ceil((n - frame_len) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_len
    padded = np.append(segment, np.zeros(max(0, pad_len - n)))

    pitches = []
    for i in range(n_frames):
        frame = padded[i * frame_step : i * frame_step + frame_len]
        if len(frame) < frame_len:
            continue
        max_amp = np.max(np.abs(frame))
        if max_amp < 0.01:
            continue
        clip_level = 0.3 * max_amp
        frame = np.where(np.abs(frame) < clip_level, 0, frame)

        autocorr = np.correlate(frame, frame, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        if len(autocorr) > max_period:
            search_region = autocorr[min_period:max_period]
            if len(search_region) > 0:
                peak_idx = np.argmax(search_region)
                peak_val = search_region[peak_idx]
                if peak_val > 0.5:
                    pitch = sr / (min_period + peak_idx)
                    pitches.append(pitch)

    if len(pitches) == 0:
        return 0.0, 0.0

    pitches = np.array(pitches)
    mean_p = np.mean(pitches)
    std_p = np.std(pitches)
    if std_p > 0:
        valid = np.abs(pitches - mean_p) < 3 * std_p
        pitches = pitches[valid]

    if len(pitches) == 0:
        return 0.0, 0.0

    return float(np.mean(pitches)), float(np.std(pitches))


def _extract_jitter(segment: np.ndarray, sr: int) -> float:
    """
    Calculate pitch period jitter (variation in pitch periods).
    Returns relative average perturbation (RAP).
    """
    frame_len = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    min_freq = 50
    max_freq = 500
    min_period = int(sr / max_freq)
    max_period = int(sr / min_freq)

    n = len(segment)
    n_frames = max(1, int(np.ceil((n - frame_len) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_len
    padded = np.append(segment, np.zeros(max(0, pad_len - n)))

    periods = []
    for i in range(n_frames):
        frame = padded[i * frame_step : i * frame_step + frame_len]
        if len(frame) < frame_len:
            continue
        max_amp = np.max(np.abs(frame))
        if max_amp < 0.01:
            continue
        clip_level = 0.3 * max_amp
        frame = np.where(np.abs(frame) < clip_level, 0, frame)

        autocorr = np.correlate(frame, frame, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        if len(autocorr) > max_period:
            search_region = autocorr[min_period:max_period]
            if len(search_region) > 0:
                peak_idx = np.argmax(search_region)
                peak_val = search_region[peak_idx]
                if peak_val > 0.5:
                    period = min_period + peak_idx
                    periods.append(period)

    if len(periods) < 3:
        return 0.0

    periods = np.array(periods, dtype=float)
    mean_p = np.mean(periods)
    std_p = np.std(periods)
    if std_p > 0:
        valid = np.abs(periods - mean_p) < 3 * std_p
        periods = periods[valid]

    if len(periods) < 3:
        return 0.0

    diffs = np.abs(np.diff(periods))
    rap = np.mean(diffs) / mean_p if mean_p > 0 else 0.0
    return float(rap)


def _extract_shimmer(segment: np.ndarray) -> float:
    """
    Calculate amplitude shimmer (variation in amplitude).
    Returns shimmer in dB.
    """
    frame_len = int(0.025 * 16000)
    frame_step = int(0.010 * 16000)

    n = len(segment)
    n_frames = max(1, int(np.ceil((n - frame_len) / frame_step)) + 1)
    pad_len = (n_frames - 1) * frame_step + frame_len
    padded = np.append(segment, np.zeros(max(0, pad_len - n)))

    amplitudes = []
    for i in range(n_frames):
        frame = padded[i * frame_step : i * frame_step + frame_len]
        if len(frame) < frame_len:
            continue
        amp = np.max(np.abs(frame))
        if amp > 0.01:
            amplitudes.append(amp)

    if len(amplitudes) < 2:
        return 0.0

    amplitudes = np.array(amplitudes)
    valid = amplitudes > 0.01
    amplitudes = amplitudes[valid]

    if len(amplitudes) < 2:
        return 0.0

    amplitudes_db = 20 * np.log10(amplitudes + 1e-10)
    shimmer_db = np.mean(np.abs(np.diff(amplitudes_db)))
    return float(shimmer_db)


def _extract_delta_mfccs(mfcc_features: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """
    Calculate first and second derivatives of MFCCs.
    Returns flattened delta and delta-delta features (2 * n_mfcc dims).
    """
    if mfcc_features.ndim == 1:
        return np.zeros(n_mfcc * 2)

    delta = np.zeros_like(mfcc_features)
    for t in range(len(mfcc_features)):
        if t == 0:
            delta[t] = mfcc_features[t + 1] - mfcc_features[t]
        elif t == len(mfcc_features) - 1:
            delta[t] = mfcc_features[t] - mfcc_features[t - 1]
        else:
            delta[t] = (mfcc_features[t + 1] - mfcc_features[t - 1]) / 2

    delta2 = np.zeros_like(delta)
    for t in range(len(delta)):
        if t == 0:
            delta2[t] = delta[t + 1] - delta[t]
        elif t == len(delta) - 1:
            delta2[t] = delta[t] - delta[t - 1]
        else:
            delta2[t] = (delta[t + 1] - delta[t - 1]) / 2

    return np.concatenate([np.mean(delta, axis=0), np.mean(delta2, axis=0)])


def _mfcc_features_frames(segment, sr, n_mfcc: int = 13):
    """Return frame-level MFCC matrix (n_frames x n_mfcc) for delta calculation."""
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
    return mfcc


def _extract_enhanced_features(segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Combine all enhanced features into a single feature vector.
    Returns: [pitch_mean, pitch_std, jitter, shimmer, delta_mfccs(26)]
    Total: 30 dimensions
    """
    mfcc_frames = _mfcc_features_frames(segment, sr)

    pitch_mean, pitch_std = _extract_pitch(segment, sr)
    jitter = _extract_jitter(segment, sr)
    shimmer = _extract_shimmer(segment)
    delta_features = _extract_delta_mfccs(mfcc_frames)

    features = np.concatenate([[pitch_mean, pitch_std, jitter, shimmer], delta_features])

    return features


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

    if feature_type == "ensemble":
        # Extract all feature types for ensemble
        mfcc = _mfcc_features(segment, sr)
        fft = _fft_features(segment, sr)
        hybrid = np.concatenate([mfcc, fft])
        return {
            "mfcc": mfcc,
            "fft": fft,
            "hybrid": hybrid,
        }

    if feature_type == "enhanced":
        return _extract_enhanced_features(segment, sr)

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


@app.get("/health", response_model=None)
@limiter.limit(RATE_LIMIT_HEALTH)
def health(request: Request):
    """Returns service health and capability status."""
    detector = _load_detector()
    _load_xtts()
    _check_gtts()
    model_info = {"name": "none", "type": "none", "ensemble": False}
    if detector:
        model_info["name"] = detector.get("model_name", "unknown")
        model_info["type"] = detector.get("feature_type", "unknown")
        model_info["ensemble"] = detector.get("feature_type") == "ensemble"
        if model_info["ensemble"]:
            models = detector.get("models", [])
            model_info["ensemble_models"] = [m.get("model_name", "unknown") for m in models]
            model_info["voting"] = detector.get("voting", "soft")

    # Determine which TTS engine is available (priority order)
    tts_engine = "none"
    if _xtts_available:
        tts_engine = "xtts_v2"
    elif _indextts2_available:
        tts_engine = "indextts2"
    elif _gtts_available:
        tts_engine = "gtts_fallback"

    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "detector_loaded": detector is not None,
        "model_name": model_info["name"],
        "model_type": model_info["type"],
        "ensemble_info": model_info if model_info["ensemble"] else None,
        "tts_engine": tts_engine,
        "xtts_v2_available": _xtts_available,
        "indextts2_available": _indextts2_available,
        "indextts2_dir": str(INDEXTTS_DIR),
        "gtts_available": _gtts_available,
    }


@app.post("/detect", response_model=None)
@limiter.limit(RATE_LIMIT_DETECT)
async def detect(
    request: Request,
    audio: UploadFile = File(...),
    apply_noise_reduction: bool = Query(False, description="Apply noise reduction preprocessing"),
    apply_normalization: bool = Query(
        False, description="Apply loudness normalization to target LUFS"
    ),
):
    """
    Classify an audio file as real or deepfake.

    Upload a WAV file (any sample rate/channels — auto-converted).
    Optional preprocessing: apply_noise_reduction, apply_normalization.
    Returns: prediction, confidence, model_used, feature_type, inference_time_s
    """
    if not audio.filename:
        raise HTTPException(400, "No filename provided.")

    t0 = time.time()
    wav_path = None
    try:
        wav_path = _save_upload(
            audio,
            apply_noise_reduction=apply_noise_reduction,
            apply_normalization=apply_normalization,
        )
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
            "mfcc": [f"MFCC{i + 1}" for i in range(13)],
            "fft": ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"],
            "hybrid": (
                [f"MFCC{i + 1}" for i in range(13)]
                + ["centroid", "bandwidth", "rolloff", "low_energy", "mid_energy", "high_energy"]
            ),
            "mfcc_hybrid": (
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

        expected_cols = feat_cols or col_map.get(feature_type, col_map["mfcc_hybrid"])
        feats = _extract_features_for_inference(wav_path, feature_type)

        # Handle ensemble model
        if feature_type == "ensemble":
            # feats is a dict with all feature types
            pred_raw = model.predict(feats)
            proba = model.predict_proba(feats)
            confidence = float(max(proba))
            prediction = pred_raw
            notes = "Deepfake voice detected" if prediction == "fake" else "Real voice detected"
        else:
            # Handle legacy 18-feature models
            if len(feats) != len(expected_cols):
                feats = _extract_legacy_18_features(wav_path)
                expected_cols = [f"MFCC{i + 1}" for i in range(13)] + [
                    "centroid",
                    "bandwidth",
                    "rolloff",
                    "jitter",
                    "shimmer",
                ]

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

        return JSONResponse(
            {
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "model_used": detector.get("model_name", "unknown"),
                "feature_type": feature_type,
                "inference_time_s": round(time.time() - t0, 3),
                "notes": notes,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Detection failed: {e}")
    finally:
        _cleanup(wav_path)


@app.post("/generate", response_model=None)
@limiter.limit(RATE_LIMIT_GENERATE)
async def generate(
    request: Request,
    audio: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form("en"),
):
    """
    Clone the voice from `audio` and synthesise `text`.

    TTS engine priority:
      1. XTTS v2    (best voice cloning — pip install TTS>=0.22.0)
      2. IndexTTS2  (zero-shot voice cloning — requires INDEXTTS_DIR env var)
      3. gTTS       (generic fallback, requires internet, NOT a voice clone)

    Args:
        audio: Speaker reference audio (WAV, MP3, etc.)
        text: Text to synthesize
        language: Language code for XTTS v2 (default: "en", supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko)

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

        # ── Attempt 1: XTTS v2 ────────────────────────────────────────────────
        if _load_xtts():
            try:
                log.info(f"XTTS v2: synthesising '{text[:60]}…' with language '{language}'")
                _xtts.tts_to_file(
                    text=text,
                    speaker_wav=str(speaker_wav),
                    language=language,
                    file_path=str(out_path),
                )
                if out_path.exists() and out_path.stat().st_size > 0:
                    method_used = "xtts_v2"
                    notes = (
                        "Voice cloned using Coqui XTTS v2 zero-shot TTS. "
                        "High-quality voice cloning with emotion and timbre "
                        "derived from the speaker reference audio."
                    )
                else:
                    log.warning("XTTS v2 produced no output, trying fallback.")
            except Exception as e:
                log.warning(f"XTTS v2 synthesis failed: {e} — trying IndexTTS2 fallback.")

        # ── Attempt 2: IndexTTS2 ──────────────────────────────────────────────
        if method_used == "none" and _load_indextts2():
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

        # ── Attempt 3: gTTS fallback ──────────────────────────────────────────
        if method_used == "none":
            _check_gtts()
            if _gtts_available:
                try:
                    from gtts import gTTS

                    log.info("Using gTTS fallback (generic voice, not a clone).")
                    tts = gTTS(text=text, lang=language[:2] if len(language) >= 2 else "en")
                    tts.save(str(out_path))
                    if out_path.exists() and out_path.stat().st_size > 0:
                        method_used = "gtts_fallback"
                        notes = (
                            "⚠ gTTS fallback used — this is a generic Google TTS voice, "
                            "NOT a clone of the uploaded speaker. "
                            "To enable real voice cloning, install XTTS v2 (pip install TTS) "
                            "or IndexTTS2 with INDEXTTS_DIR env var."
                        )
                except Exception as e:
                    log.error(f"gTTS also failed: {e}")

        if method_used == "none":
            raise HTTPException(
                503,
                "No TTS engine available.\n"
                "• For voice cloning: pip install TTS>=0.22.0 (XTTS v2, recommended)\n"
                "• Alternative: install IndexTTS2 and set INDEXTTS_DIR env var.\n"
                "• For basic TTS fallback: pip install gtts",
            )

        audio_bytes = out_path.read_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        return JSONResponse(
            {
                "success": True,
                "method_used": method_used,
                "audio_base64": audio_b64,
                "audio_mime": "audio/wav",
                "generation_time_s": round(time.time() - t0, 3),
                "notes": notes,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Generation failed: {e}")
    finally:
        _cleanup(speaker_wav, out_path)


@app.post("/convert-voice", response_model=None)
@limiter.limit(RATE_LIMIT_CONVERT)
async def convert_voice(
    request: Request,
    source_audio: UploadFile = File(..., description="The voice to convert FROM"),
    target_audio: UploadFile = File(
        ..., description="The voice to convert TO (reference for target voice characteristics)"
    ),
):
    """
    Convert voice from source audio to match target voice characteristics.

    This endpoint performs voice conversion by transforming the source voice
    to sound like the target voice while preserving the content/speech.

    Voice conversion methods (in priority order):
      1. XTTS v2 (Coqui TTS) — high-quality zero-shot voice conversion
      2. Pitch/timbre matching — spectral and pitch-based transformation

    Upload two WAV files:
      - source_audio: The voice/audio content to convert
      - target_audio: Reference voice for target characteristics (timbre, pitch, etc.)

    Returns: audio_base64 (WAV), method_used, conversion_time_s, notes
    """
    if not source_audio.filename:
        raise HTTPException(400, "No source audio provided.")
    if not target_audio.filename:
        raise HTTPException(400, "No target audio provided.")

    t0 = time.time()
    source_wav = None
    target_wav = None
    out_path = None

    try:
        # Save uploaded files
        source_wav = _save_upload(source_audio)
        target_wav = _save_upload(target_audio)

        uid = uuid.uuid4().hex
        out_path = GENERATED_DIR / f"converted_{uid}.wav"

        method_used = "none"
        notes = ""

        # ── Attempt 1: Coqui TTS XTTS v2 ──────────────────────────────────────
        if _check_coqui_tts():
            try:
                success = _convert_voice_xtts(source_wav, target_wav, out_path)
                if success:
                    method_used = "xtts_vc"
                    notes = (
                        "Voice converted using Coqui TTS XTTS v2. "
                        "High-quality zero-shot voice conversion applied."
                    )
            except Exception as e:
                log.warning(f"XTTS voice conversion failed: {e} — trying fallback.")

        # ── Attempt 2: Pitch/timbre matching fallback ─────────────────────────
        if method_used == "none":
            try:
                log.info("Using pitch/timbre matching fallback for voice conversion.")
                success = _apply_timbre_matching(source_wav, target_wav, out_path)
                if success:
                    method_used = "pitch_shift"
                    notes = (
                        "Voice converted using pitch/timbre matching algorithm. "
                        "This method applies spectral equalization and pitch shifting "
                        "to approximate target voice characteristics. Quality may vary."
                    )
                else:
                    log.warning("Pitch/timbre matching failed.")
            except Exception as e:
                log.warning(f"Pitch/timbre matching failed: {e}")

        # Check if any method succeeded
        if method_used == "none":
            raise HTTPException(
                503,
                "Voice conversion failed. No conversion method available.\n"
                "• For high-quality conversion: pip install TTS\n"
                "• Fallback requires: numpy, scipy, librosa (optional for better quality)",
            )

        # Verify output exists
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise HTTPException(500, "Voice conversion produced no output.")

        # Read and encode output
        audio_bytes = out_path.read_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        conversion_time = time.time() - t0

        return JSONResponse(
            {
                "success": True,
                "method_used": method_used,
                "audio_base64": audio_b64,
                "audio_mime": "audio/wav",
                "conversion_time_s": round(conversion_time, 3),
                "notes": notes,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Voice conversion failed: {e}")
    finally:
        _cleanup(source_wav, target_wav, out_path)


# ─── entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # Get reload mode from environment (default False for production)
    reload_mode = os.environ.get("UVICORN_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=reload_mode,
        access_log=True,
    )
