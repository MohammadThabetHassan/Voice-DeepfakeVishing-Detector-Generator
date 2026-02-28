# System Architecture

## Overview

The Voice Deepfake Vishing Detector & Generator is a modular system designed for academic research and educational purposes. It consists of a static frontend (GitHub Pages), a Python FastAPI backend, and a training pipeline for ML models.

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Frontend["🌐 Frontend (GitHub Pages)"]
        A[Static HTML/CSS/JS]
        B[Tabs: Detect | Generate | Results | About]
        C[Demo Mode / Live Mode]
    end

    subgraph Backend["⚙️ Backend (FastAPI)"]
        D[UVicorn Server]
        E[API Routes]
        F[Feature Extraction]
        G[Model Inference]
        H[Voice Cloning]
    end

    subgraph ML["🧠 ML Pipeline"]
        I[MFCC Extractor]
        J[FFT/Spectral Extractor]
        K[Hybrid Extractor]
        L[Gradient Boosting Classifier]
        M[3 Model Variants]
    end

    subgraph Generation["🎙️ Voice Generation"]
        N[IndexTTS2 Clone]
        O[gTTS Fallback]
    end

    subgraph Data["💾 Data Layer"]
        P[WAV Uploads]
        Q[Models .pkl]
        R[Results JSON]
    end

    A -->|"HTTPS POST /detect"| E
    A -->|"HTTPS POST /generate"| E
    A -->|"HTTPS GET /health"| E

    E --> F
    F --> I
    F --> J
    F --> K
    I --> L
    J --> L
    K --> L
    L --> M
    M --> G
    G --> E

    E --> H
    H --> N
    H --> O
    N --> P
    O --> P

    M --> Q
    Q --> G
    L --> R

    P -->|"temp files"| P
```

## Component Breakdown

### 1. Frontend (GitHub Pages)

**Location:** [`frontend/`](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/tree/master/frontend)

- **Static hosting:** Works without any backend
- **Demo mode:** Fully functional UI with explanations when no backend connected
- **Live mode:** Calls configurable API URL stored in localStorage
- **Features:**
  - File upload with drag-and-drop
  - Audio playback
  - Real-time results display
  - Model comparison table
  - Ethical use warnings

**Key Files:**
- `index.html` — Main page structure
- `css/main.css` — Styling
- `js/app.js` — Application logic

### 2. Backend (FastAPI)

**Location:** [`backend/`](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/tree/master/backend)

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status, model info, TTS availability |
| `/detect` | POST | Upload WAV → `{prediction, confidence, model_used}` |
| `/generate` | POST | Upload speaker WAV + text → base64 audio |

**Features:**
- CORS enabled for GitHub Pages integration
- Automatic audio format conversion (via pydub/ffmpeg)
- Lazy model loading
- Temp file cleanup
- Multiple model support (MFCC, FFT, Hybrid)

**Key Files:**
- `app.py` — Main application
- `requirements.txt` — Dependencies
- `tests/test_api.py` — Unit tests

### 3. ML Training Pipeline

**Location:** [`training/`](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/tree/master/training)

**Three Model Variants:**

1. **MFCC-only** (13 dimensions)
   - Mel-Frequency Cepstral Coefficients
   - Standard speech processing features

2. **FFT/Spectral-only** (6 dimensions)
   - Spectral centroid, bandwidth, rolloff
   - Band energy ratios

3. **Hybrid** (19 dimensions)
   - Combination of MFCC + FFT features
   - Best overall performance

**Classifier:**
- Gradient Boosting Classifier
- StandardScaler preprocessing
- 5-fold stratified cross-validation

**Key Files:**
- `train.py` — Main training script
- `requirements.txt` — Dependencies

### 4. Voice Generation

**Primary Engine: IndexTTS2**
- Zero-shot voice cloning
- Requires speaker reference audio (5+ seconds)
- Optional: requires ~4GB VRAM for GPU acceleration
- Environment variable: `INDEXTTS_DIR=/opt/index-tts`

**Fallback Engine: gTTS**
- Google Text-to-Speech API
- Generic voice (NOT a clone)
- Requires internet connection
- Used when IndexTTS2 unavailable

## Data Flow

### Detection Flow

```
[User] → [Upload WAV] → [Frontend] → [POST /detect]
                                            ↓
[Backend] → [Convert to WAV 16kHz] → [Feature Extraction]
                                            ↓
[MFCC/FFT/Hybrid] → [Model Inference] → [JSON Response]
                                            ↓
[Frontend] → [Display Result with Confidence]
```

### Generation Flow

```
[User] → [Upload Speaker WAV + Text] → [Frontend] → [POST /generate]
                                                          ↓
[Backend] → [Try IndexTTS2] → [Success?] → [Return base64 audio]
                ↓ No
        [Fallback to gTTS] → [Return base64 audio]
```

## Security Considerations

1. **No audio storage** — All files processed in memory, deleted after response
2. **UUID filenames** — Prevents path traversal attacks
3. **File size limits** — 10MB max upload
4. **Consent checkbox** — Required before voice cloning
5. **CORS** — Configurable allowed origins

## Deployment Options

### Option A: GitHub Pages + Local Backend (Recommended for Development)

1. Frontend auto-deploys to GitHub Pages on push to master
2. Run backend locally: `cd backend && uvicorn app:app --reload`
3. Configure API URL in frontend UI

### Option B: Docker Container

```bash
# Build
docker build -t deepfake-api .

# Run
docker run -p 8000:8000 deepfake-api
```

### Option C: GitHub Container Registry (GHCR)

```bash
# Pull pre-built image
docker pull ghcr.io/mohammadthabethassan/voice-deepfake-vishing-detector-generator:latest

# Run
docker run -p 8000:8000 ghcr.io/mohammadthabethassan/voice-deepfake-vishing-detector-generator:latest
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Vanilla HTML5, CSS3, JavaScript (ES6+) |
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | scikit-learn, scipy, numpy, pandas |
| TTS | IndexTTS2 (optional), gTTS (fallback) |
| Audio | pydub, scipy.io.wavfile |
| CI/CD | GitHub Actions |
| Hosting | GitHub Pages (frontend), GHCR (backend container) |

## Scalability Notes

- **Frontend:** Static hosting scales infinitely via GitHub Pages CDN
- **Backend:** Stateless design allows horizontal scaling
- **ML Models:** Lightweight (< 2MB each), suitable for edge deployment
- **TTS:** IndexTTS2 is compute-intensive; consider queueing for production
