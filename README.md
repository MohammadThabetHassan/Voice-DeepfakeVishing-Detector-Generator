# Voice Deepfake Vishing Detector & Generator

[![CI](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/actions/workflows/ci.yml)
[![Pages](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/actions/workflows/pages.yml/badge.svg)](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/actions/workflows/pages.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Research & Educational Use Only.** Do not use this tool to clone voices without explicit consent, or for any malicious purpose.

A graduation-quality research project that:
- **Detects** AI-synthesised (deepfake) voices using multiple ML pipelines: MFCC, FFT/Spectral, Hybrid, Enhanced, and Ensemble models
- **Generates** voice clones using Coqui XTTS v2, IndexTTS2, or gTTS fallback
- **Converts** voices from source to target characteristics using zero-shot voice conversion
- **Deploys** the UI to GitHub Pages (static, no backend required to load)
- **Runs** the backend locally or as a Docker container

---

## Live Demo

**UI (GitHub Pages):** https://mohammadthabethassan.github.io/Voice-Deepfake-Vishing-Detector-Generator/

> The Pages site loads in Demo Mode by default. To run real detection/generation, start the backend locally and enter the URL via the ⚙ Config button.

---

## New Features

### Detection
- **Audio Preprocessing** — Optional noise reduction and loudness normalization
- **Enhanced ML Features** — Pitch, jitter, shimmer, and delta MFCCs for improved accuracy
- **XGBoost & Ensemble Models** — Soft voting across multiple classifiers
- **Uncertain Decision Band** — Borderline scores are reported as `uncertain` instead of forced binary output
- **Multiple Model Types** — MFCC-only, FFT-only, Hybrid, Enhanced (75-dim), and Ensemble

### Generation
- **Coqui XTTS v2** — High-quality zero-shot voice cloning (pip installable)
- **IndexTTS2** — Zero-shot voice cloning with emotion control
- **gTTS Fallback** — Generic TTS when cloning unavailable

### Voice Conversion
- **Zero-shot Voice Conversion** — Transform source voice to match target characteristics
- **XTTS v2 Voice Conversion** — High-quality conversion using Coqui TTS
- **Pitch/Timbre Matching** — Fallback spectral transformation

### Frontend
- **Browser Recording** — Record audio directly in the browser
- **Waveform Visualization** — Visual audio preview with waveform display
- **Dark Mode** — Toggle between light and dark themes
- **Detection History** — Local storage of recent detection results

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ GitHub Pages (static)                                               │
│   frontend/index.html  +  js/app.js  +  css/main.css               │
│                                                                     │
│  Tabs: Detect | Generate | Convert | Model Results | About          │
│  Features: Dark mode, waveform viz, browser recording, history      │
│  Demo Mode: fully functional UI, no backend needed                  │
│  Live Mode: calls configurable API URL stored in localStorage       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTPS POST /detect  /generate  /convert-voice
                            │ GET /health
┌───────────────────────────▼─────────────────────────────────────────┐
│ FastAPI Backend (local / Docker / GHCR)                             │
│   backend/app.py  running at http://localhost:8000                  │
│                                                                     │
│  POST /detect        → WAV upload → { prediction, fake_probability, threshold, ... }│
│  POST /generate      → speaker WAV + text → base64 WAV audio       │
│  POST /convert-voice → source + target WAV → base64 converted      │
│  GET  /health        → { status, model_name, tts_engine, ... }     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ joblib.load()
┌───────────────────────────▼─────────────────────────────────────────┐
│ Models  (models/)                                                   │
│   deepfake_detector_mfcc.pkl      — 13-dim MFCC                    │
│   deepfake_detector_fft.pkl       — 6-dim FFT/Spectral              │
│   deepfake_detector_hybrid.pkl    — 19-dim Hybrid                   │
│   deepfake_detector_enhanced.pkl  — 75-dim Enhanced                 │
│   deepfake_detector_ensemble.pkl  — Soft voting ensemble            │
│   deepfake_detector_best.pkl      — Copy of best model              │
│   results.json                    — Metrics comparison table        │
│   model_manifest.json             — Model IDs + data/training hash  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ trained by
┌───────────────────────────▼─────────────────────────────────────────┐
│ Training Pipeline  (training/train.py)                              │
│   Input: WAV dirs OR pre-extracted CSV                              │
│   5-fold stratified cross-validation                                │
│   XGBoost + GradientBoosting + StandardScaler                      │
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
│   ├── app.py             ← Main API (detect, generate, convert)
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
│   ├── results.json
│   └── model_manifest.json
│
├── scripts/
│   ├── prepare_external_dataset.py
│   ├── collect_web_audio.py     ← YouTube/podcast data collector
│   └── web_sources.json         ← Source configuration template
│
├── docs/                  ← Documentation
│   ├── api.md             ← Complete API documentation
│   ├── architecture.md    ← System architecture
│   ├── threat-model.md
│   ├── evaluation.md
│   ├── ethics.md
│   └── CHANGELOG.md       ← Version history
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

# Option C: prepare and train on broader external real/fake data
python scripts/prepare_external_dataset.py --clean --max-per-label 0
python training/train.py --data training/data_external --output models_external/

# Option D: collect web audio (YouTube + podcast RSS), then train
python scripts/collect_web_audio.py --clean --max-youtube-per-query 8 --max-podcast-per-feed 8
python training/train.py --data training/web_data/processed --output models_web/
```

This creates `models/deepfake_detector_*.pkl`, `models/results.json`, and `models/model_manifest.json`.

See `docs/enhancements-and-threshold-tuning-2026-03-03.md` for the latest enhancement review and threshold calibration notes.

### 3. Start the Backend

```bash
pip install -r backend/requirements.txt

# Optional: install TTS for voice cloning
pip install TTS>=0.22.0

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

---

## API Reference

See [docs/api.md](docs/api.md) for complete API documentation.

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status, model info, TTS availability |
| `/detect` | POST | Upload WAV → `{prediction, fake_probability, threshold, ...}` |
| `/generate` | POST | Upload speaker WAV + text → base64 audio |
| `/convert-voice` | POST | Source + target WAV → converted voice |

---

## ML Training Details

### Models Comparison

| Model | Features | Dims | Expected F1 |
|---|---|---|---|
| MFCC-only | Mel Frequency Cepstral Coefficients (mean) | 13 | ~0.89 |
| FFT/Spectral | Centroid, bandwidth, rolloff, band energies | 6 | ~0.84 |
| Hybrid | MFCC + FFT concatenated | 19 | ~0.92 |
| **Enhanced** | MFCC + FFT + pitch + jitter + shimmer + delta MFCCs | **75** | **~0.94** |
| **Ensemble** | Soft voting across multiple classifiers | Varies | **~0.95** |

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

### XTTS v2 (recommended — pip installable)

Coqui XTTS v2 is the primary TTS engine for voice cloning:

```bash
pip install TTS>=0.22.0
# First run will auto-download the model (~2 GB)
```

Supports 17 languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko

### IndexTTS2 (zero-shot voice cloning)

IndexTTS2 is developed by Bilibili and provides zero-shot voice cloning:

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

### gTTS fallback (no voice cloning)

When XTTS v2 and IndexTTS2 are not configured, the backend falls back to Google TTS (gTTS). This generates a **generic voice** — it does NOT clone the uploaded speaker. The response clearly labels which engine was used.

---

## Voice Conversion

Convert voice from source to target characteristics:

```bash
curl -X POST "http://localhost:8000/convert-voice" \
  -F "source_audio=@source.wav" \
  -F "target_audio=@target.wav" \
  --output converted.wav
```

Methods (priority order):
1. **XTTS v2** — High-quality zero-shot voice conversion (requires `pip install TTS`)
2. **Pitch/Timbre Matching** — Spectral equalization and pitch shifting fallback

---

## Audio Preprocessing

Optional preprocessing for detection endpoint:

- **Noise Reduction** — Uses `noisereduce` library to reduce background noise
- **Loudness Normalization** — Normalizes to target LUFS (default: -23 LUFS)

Usage:
```bash
curl -X POST "http://localhost:8000/detect?apply_noise_reduction=true&apply_normalization=true" \
  -F "audio=@sample.wav"
```

### Detection Threshold Profiles

The backend supports threshold profiles and an uncertainty band:

- `DETECTION_THRESHOLD_PROFILE=balanced` (default): threshold `0.8`, uncertain margin `0.08`
- `DETECTION_THRESHOLD_PROFILE=low_fp`: threshold `0.85`, uncertain margin `0.06`
- `DETECTION_THRESHOLD_PROFILE=high_recall`: threshold `0.7`, uncertain margin `0.1`

You can still override values directly:

```bash
DETECTION_THRESHOLD_PROFILE=low_fp \
DETECTION_FAKE_THRESHOLD=0.85 \
DETECTION_UNCERTAIN_MARGIN=0.06 \
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

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
git tag v2.1.0
git push origin v2.1.0
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
| API Reference | [docs/api.md](docs/api.md) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Threat Model | [docs/threat-model.md](docs/threat-model.md) |
| Evaluation Methodology | [docs/evaluation.md](docs/evaluation.md) |
| Ethics Policy | [docs/ethics.md](docs/ethics.md) |
| Changelog | [docs/CHANGELOG.md](docs/CHANGELOG.md) |
| Interactive API Docs | http://localhost:8000/docs (live Swagger UI) |

---

## Limitations & Honest Caveats

1. **Small dataset**: Trained on ~475 samples. Do not use for production security systems.
2. **Distribution shift**: Models may not generalise to deepfakes from systems not represented in training data (ElevenLabs, Vall-E, etc.).
3. **GitHub Pages is static**: The backend cannot run on GitHub Pages. The Pages site is a UI shell; full functionality requires a local or containerised backend.
4. **gTTS ≠ voice cloning**: When XTTS/IndexTTS2 are unavailable, gTTS generates a generic voice, not a clone of the input speaker.
5. **No real-time streaming**: The current pipeline processes uploaded files; real-time call interception requires additional VoIP integration (Asterisk/FreeSWITCH, out of scope).
6. **Voice conversion quality**: Fallback pitch/timbre matching provides lower quality than XTTS v2 zero-shot conversion.

---

## License

MIT License. See [LICENSE](LICENSE).

> **Disclaimer:** Research and educational use only. Do not use to impersonate others, conduct fraud, or violate any law. The authors accept no liability for misuse.

---

## References

- [ASVspoof Challenge](https://www.asvspoof.org/)
- [Coqui TTS XTTS v2](https://github.com/coqui-ai/TTS)
- [IndexTTS2](https://github.com/index-tts/index-tts)
- [Open Speech Repository](http://www.voiptroubleshooter.com/open_speech/american.html)
- [Towards Audio Deepfake Detection](https://arxiv.org/abs/1910.11916)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
