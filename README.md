# Voice Deepfake Vishing Detector & Generator

[![CI](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/actions/workflows/ci.yml)
[![Pages](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/actions/workflows/pages.yml/badge.svg)](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/actions/workflows/pages.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Research & Educational Use Only.** Do not use this tool to clone voices without explicit consent, or for any malicious purpose.

A graduation-quality research project that:
- **Detects** AI-synthesised (deepfake) voices using three ML pipelines: MFCC-only, FFT/Spectral-only, and Hybrid.
- **Generates** voice clones using Coqui TTS YourTTS (with gTTS fallback).
- **Deploys** the UI to GitHub Pages (static, no backend required to load).
- **Runs** the backend locally or as a Docker container.

---

## Live Demo

**UI (GitHub Pages):** https://mohammadthabethassan.github.io/Voice-Deepfake-Vishing-Detector-Generator/

> The Pages site loads in Demo Mode by default. To run real detection/generation, start the backend locally and enter the URL via the ⚙ Config button.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ GitHub Pages (static)                                               │
│   frontend/index.html  +  js/app.js  +  css/main.css               │
│                                                                     │
│  Tabs: Detect | Generate | Model Results | About                    │
│  Demo Mode: fully functional UI, no backend needed                  │
│  Live Mode: calls configurable API URL stored in localStorage       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTPS POST /detect  /generate
                            │ GET  /health
┌───────────────────────────▼─────────────────────────────────────────┐
│ FastAPI Backend (local / Docker / GHCR)                             │
│   backend/app.py  running at http://localhost:8000                  │
│                                                                     │
│  POST /detect   → WAV upload → { prediction, confidence, ... }     │
│  POST /generate → speaker WAV + text → base64 WAV audio            │
│  GET  /health   → { status, model_name, uptime_seconds, ... }      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ joblib.load()
┌───────────────────────────▼─────────────────────────────────────────┐
│ Models  (models/)                                                   │
│   deepfake_detector_mfcc.pkl     — 13-dim MFCC                     │
│   deepfake_detector_fft.pkl      — 6-dim FFT/Spectral               │
│   deepfake_detector_hybrid.pkl   — 19-dim Hybrid  ← best           │
│   deepfake_detector_best.pkl     — copy of best model              │
│   results.json                   — metrics comparison table         │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ trained by
┌───────────────────────────▼─────────────────────────────────────────┐
│ Training Pipeline  (training/train.py)                              │
│   Input: WAV dirs OR pre-extracted CSV                              │
│   5-fold stratified cross-validation                                │
│   GradientBoosting + StandardScaler                                 │
│   Outputs: .pkl models + results.json                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── frontend/              ← Static site for GitHub Pages
│   ├── index.html
│   ├── css/main.css
│   └── js/app.js
│
├── backend/               ← FastAPI application
│   ├── app.py             ← Main API (detect, generate, health)
│   ├── requirements.txt
│   ├── uploads/           ← Temp (git-ignored)
│   ├── generated/         ← Temp (git-ignored)
│   └── tests/
│       └── test_api.py
│
├── training/              ← ML training scripts
│   ├── train.py           ← Main training pipeline
│   ├── requirements.txt
│   └── data/              ← Place WAV files here (git-ignored)
│       ├── real/
│       └── fake/
│
├── models/                ← Saved classifier models
│   ├── deepfake_detector_*.pkl
│   └── results.json
│
├── docs/                  ← Documentation
│   ├── threat-model.md
│   ├── evaluation.md
│   └── ethics.md
│
├── .github/workflows/
│   ├── ci.yml             ← Lint + tests on PR/push
│   ├── pages.yml          ← Deploy frontend to GitHub Pages
│   └── release.yml        ← Package models + build Docker on tag
│
├── Dockerfile             ← For GHCR container build
├── osr_features.csv       ← Pre-extracted features (475 rows)
├── features.csv           ← Additional features (400 rows)
├── pipeline.py            ← Legacy pipeline (kept for compatibility)
├── server.py              ← Legacy server (replaced by backend/app.py)
└── README.md
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator.git
cd Voice-Deepfake-Vishing-Detector-Generator
```

### 2. Train the Detection Models

```bash
pip install -r training/requirements.txt

# Option A: use pre-extracted CSV (fastest, no audio files needed)
python training/train.py --csv osr_features.csv --output models/

# Option B: train from WAV files
#   Place real WAV files in:  training/data/real/
#   Place fake WAV files in:  training/data/fake/
python training/train.py --data training/data/ --output models/
```

This creates `models/deepfake_detector_mfcc.pkl`, `*_fft.pkl`, `*_hybrid.pkl`, `*_best.pkl`, and `models/results.json`.

### 3. Start the Backend

```bash
pip install -r backend/requirements.txt

# Optional: install ffmpeg for audio format conversion
# Ubuntu: sudo apt-get install -y ffmpeg
# macOS:  brew install ffmpeg

cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API docs.

### 4. Use the Frontend

**Option A: GitHub Pages (recommended)**
1. Go to https://mohammadthabethassan.github.io/Voice-Deepfake-Vishing-Detector-Generator/
2. Click the ⚙ Config button
3. Enter `http://localhost:8000` as the API URL
4. Click "Save & Test"

**Option B: Open locally**
```bash
# Serve frontend locally (any static server)
cd frontend
python3 -m http.server 3000
# Then open http://localhost:3000
```

**Option C: Use the backend's built-in static serving (legacy)**
```bash
python server.py  # serves on port 8000 with legacy server
```

---

## API Reference

### `GET /health`

```json
{
  "status": "ok",
  "uptime_seconds": 42.1,
  "detector_loaded": true,
  "model_name": "deepfake_detector_hybrid",
  "tts_available": false,
  "gtts_available": true
}
```

### `POST /detect`

**Request:** `multipart/form-data` with field `audio` (WAV file, any sample rate/channels)

**Response:**
```json
{
  "prediction": "fake",
  "confidence": 0.9142,
  "model_used": "deepfake_detector_hybrid",
  "feature_type": "hybrid",
  "inference_time_s": 0.003,
  "notes": "Deepfake voice detected"
}
```

### `POST /generate`

**Request:** `multipart/form-data` with fields:
- `audio`: WAV speaker reference file (≥5 s, mono 16 kHz recommended)
- `text`: Text to synthesise (max 500 chars)

**Response:**
```json
{
  "success": true,
  "method_used": "coqui_your_tts",
  "audio_base64": "<base64 encoded WAV>",
  "audio_mime": "audio/wav",
  "generation_time_s": 12.4,
  "notes": "Voice cloned via Coqui YourTTS."
}
```

If Coqui TTS is not installed, the system falls back to gTTS (generic voice, not a speaker clone) and labels it clearly.

---

## ML Training Details

### Models Comparison

| Model | Features | Dims | Expected F1 |
|---|---|---|---|
| MFCC-only | Mel Frequency Cepstral Coefficients (mean) | 13 | ~0.89 |
| FFT/Spectral | Centroid, bandwidth, rolloff, band energies | 6 | ~0.84 |
| **Hybrid** | MFCC + FFT concatenated | **19** | **~0.92** |

Run `cat models/results.json` to see actual computed metrics.

### Training Pipeline Options

```bash
# Train on existing CSV (auto-detects feature type)
python training/train.py --csv osr_features.csv --output models/

# Train on WAV directory
python training/train.py --data training/data/ --output models/

# Specify output directory and CV folds
python training/train.py --csv osr_features.csv --output models/ --cv-folds 5

# Adjust segment duration (default 1 second)
python training/train.py --data training/data/ --seg-duration 2.0
```

---

## Voice Generation

### IndexTTS2 (real zero-shot voice cloning — recommended)

IndexTTS2 is developed by Bilibili and is the primary TTS engine. It performs zero-shot voice cloning from a short reference clip, with emotion-timbre separation and optional duration control.

```bash
# Step 1: Clone IndexTTS2 (must use their uv-based setup)
git clone https://github.com/index-tts/index-tts.git /opt/index-tts
cd /opt/index-tts

# Step 2: Install dependencies (uv is REQUIRED — pip is NOT supported)
pip install uv
uv sync --all-extras

# Step 3: Download model weights (~4 GB)
pip install "huggingface-hub[cli]"
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints

# Step 4: Tell this backend where to find IndexTTS2
export INDEXTTS_DIR=/opt/index-tts

# Step 5: Start the backend
cd /path/to/this/project
uvicorn backend.app:app --reload
```

Hardware requirements:
- **CPU**: Works but slow (30–120 s per synthesis on a modern CPU)
- **GPU**: Recommended — NVIDIA CUDA 12.8+, ~4 GB VRAM in FP16 mode

For GPU FP16, change `use_fp16=False` → `use_fp16=True` in `backend/app.py` line `_load_indextts2()`.

### gTTS fallback (no voice cloning)

When IndexTTS2 is not configured, the backend falls back to Google TTS (gTTS). This generates a **generic English voice** — it does NOT clone the uploaded speaker. The response clearly labels which engine was used.

---

## GitHub Actions Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Push / PR to `main`/`master` | Ruff lint, backend tests, training smoke test, HTML check |
| `pages.yml` | Push to `main`/`master` | Deploys `frontend/` to GitHub Pages |
| `release.yml` | Tag `v*.*.*` | Trains models, creates GitHub Release with artifacts, builds Docker image to GHCR |

### Enable GitHub Pages

1. Go to your repo **Settings → Pages**
2. Set **Source** to **GitHub Actions**
3. Push to `main` — the `pages.yml` workflow deploys automatically

### Create a Release

```bash
git tag v1.0.0
git push origin v1.0.0
# The release workflow runs automatically
```

### Docker / GHCR

```bash
# Pull the image (after a tagged release)
docker pull ghcr.io/mohammadthabethassan/voice-deepfake-vishing-detector-generator:latest

# Run
docker run -p 8000:8000 ghcr.io/mohammadthabethassan/voice-deepfake-vishing-detector-generator:latest
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio httpx
pip install -r backend/requirements.txt

pytest backend/tests/ -v
```

---

## Documentation

| Document | Path |
|---|---|
| Threat Model | `docs/threat-model.md` |
| Evaluation Methodology | `docs/evaluation.md` |
| Ethics Policy | `docs/ethics.md` |
| API Reference | http://localhost:8000/docs (live Swagger UI) |

---

## Limitations & Honest Caveats

1. **Small dataset**: Trained on ~475 samples. Do not use for production security systems.
2. **Distribution shift**: Models may not generalise to deepfakes from systems not represented in training data (ElevenLabs, Vall-E, etc.).
3. **GitHub Pages is static**: The backend cannot run on GitHub Pages. The Pages site is a UI shell; full functionality requires a local or containerised backend.
4. **gTTS ≠ voice cloning**: When Coqui is unavailable, gTTS generates a generic voice, not a clone of the input speaker.
5. **No real-time streaming**: The current pipeline processes uploaded files; real-time call interception requires additional VoIP integration (Asterisk/FreeSWITCH, out of scope).

---

## License

MIT License. See [LICENSE](LICENSE).

> **Disclaimer:** Research and educational use only. Do not use to impersonate others, conduct fraud, or violate any law. The authors accept no liability for misuse.

---

## References

- [ASVspoof Challenge](https://www.asvspoof.org/)
- [Coqui TTS YourTTS](https://docs.coqui.ai/en/latest/models/your_tts.html)
- [Open Speech Repository](http://www.voiptroubleshooter.com/open_speech/american.html)
- [Towards Audio Deepfake Detection](https://arxiv.org/abs/1910.11916)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
