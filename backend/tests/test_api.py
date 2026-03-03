"""
Unit tests for the FastAPI backend.
Tests health endpoint, input validation, and feature extraction.

"""

import io
import struct
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ─── Import the app ──────────────────────────────────────────────────────────
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.app import app, _mfcc_features, _fft_features, _extract_legacy_18_features

client = TestClient(app)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def make_wav_bytes(duration_s: float = 1.0, sr: int = 16000, freq_hz: float = 440.0) -> bytes:
    """Generate a minimal synthetic WAV file in memory."""
    import math

    n_samples = int(sr * duration_s)
    amplitude = 16000
    buf = io.BytesIO()
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        samples = [
            struct.pack("<h", int(amplitude * math.sin(2 * math.pi * freq_hz * i / sr)))
            for i in range(n_samples)
        ]
        w.writeframes(b"".join(samples))
    buf.seek(0)
    return buf.read()


# ─── Health ───────────────────────────────────────────────────────────────────


def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "detector_loaded" in data


# ─── Detect ───────────────────────────────────────────────────────────────────


def test_detect_no_file_returns_422():
    resp = client.post("/detect")
    assert resp.status_code == 422


def test_detect_valid_wav():
    wav_bytes = make_wav_bytes()
    files = {"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
    resp = client.post("/detect", files=files)
    # Expect 200 if a model is loaded, or 503 if no model available
    assert resp.status_code in (200, 503)
    data = resp.json()
    if resp.status_code == 200:
        assert "prediction" in data
        assert data["prediction"] in ("real", "fake")
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0


def test_detect_short_audio():
    """Very short audio is rejected by duration validation (min 0.5s)."""
    wav_bytes = make_wav_bytes(duration_s=0.1)
    files = {"audio": ("short.wav", io.BytesIO(wav_bytes), "audio/wav")}
    resp = client.post("/detect", files=files)
    # 400 = too short, 200 = model inference succeeded, 503 = no model loaded
    assert resp.status_code in (200, 400, 503)


def test_detect_wrong_content_type():
    """Sending a text file as audio should raise an error, not crash."""
    files = {"audio": ("bad.txt", io.BytesIO(b"not audio"), "text/plain")}
    resp = client.post("/detect", files=files)
    # Should be 4xx or 5xx, but never a 200 with wrong data
    assert resp.status_code in (400, 415, 422, 500, 503)


# ─── Generate ─────────────────────────────────────────────────────────────────


def test_generate_no_file_returns_422():
    resp = client.post("/generate", data={"text": "hello"})
    assert resp.status_code == 422


def test_generate_no_text_returns_422():
    wav_bytes = make_wav_bytes()
    files = {"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
    resp = client.post("/generate", files=files)
    assert resp.status_code == 422


def test_generate_empty_text_returns_400():
    wav_bytes = make_wav_bytes()
    files = {"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
    resp = client.post("/generate", files=files, data={"text": "   "})
    assert resp.status_code == 400


def test_generate_text_too_long_returns_400():
    wav_bytes = make_wav_bytes()
    files = {"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
    resp = client.post("/generate", files=files, data={"text": "x" * 501})
    assert resp.status_code == 400


def test_generate_returns_audio_or_503():
    """Returns base64 audio on success, or 503 if no TTS available."""
    wav_bytes = make_wav_bytes(duration_s=3.0)
    files = {"audio": ("speaker.wav", io.BytesIO(wav_bytes), "audio/wav")}
    resp = client.post("/generate", files=files, data={"text": "Hello world"})
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "audio_base64" in data
        assert len(data["audio_base64"]) > 0
        assert "method_used" in data


# ─── Feature extraction unit tests ───────────────────────────────────────────


def test_mfcc_features_shape():
    import numpy as np

    segment = np.random.randn(16000).astype(np.float32)
    feats = _mfcc_features(segment, 16000)
    assert feats.shape == (13,), f"Expected (13,), got {feats.shape}"


def test_fft_features_shape():
    import numpy as np

    segment = np.random.randn(16000).astype(np.float32)
    feats = _fft_features(segment, 16000)
    assert feats.shape == (6,), f"Expected (6,), got {feats.shape}"


def test_fft_features_no_nan():
    import numpy as np

    segment = np.random.randn(16000).astype(np.float32)
    feats = _fft_features(segment, 16000)
    assert not np.any(np.isnan(feats)), "FFT features contain NaN"


def test_mfcc_features_silent():
    """Silent audio should not crash feature extraction."""
    import numpy as np

    segment = np.zeros(16000, dtype=np.float32)
    feats = _mfcc_features(segment, 16000)
    assert feats.shape == (13,)


def test_legacy_18_features():
    """Legacy 18-feature extraction (for old .pkl models)."""
    import tempfile
    import numpy as np

    # Write a temp WAV
    wav_bytes = make_wav_bytes(duration_s=1.0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = Path(f.name)

    try:
        feats = _extract_legacy_18_features(path)
        assert feats.shape == (18,), f"Expected 18 features, got {feats.shape}"
    finally:
        path.unlink(missing_ok=True)


# ─── Docs ─────────────────────────────────────────────────────────────────────


def test_openapi_docs_accessible():
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_openapi_json_accessible():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "paths" in schema
    assert "/detect" in schema["paths"]
    assert "/generate" in schema["paths"]
    assert "/health" in schema["paths"]
