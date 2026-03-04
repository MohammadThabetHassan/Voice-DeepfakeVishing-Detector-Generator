# System Architecture

## Overview

The Voice Deepfake Vishing Detector & Generator is a modular system designed for academic research and educational purposes. It consists of a static frontend (GitHub Pages), a Python FastAPI backend, and a training pipeline for ML models.

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Frontend["🌐 Frontend (GitHub Pages)"]
        A[Static HTML/CSS/JS]
        B[Tabs: Detect | Generate | Convert | Results | About]
        C[Demo Mode / Live Mode]
        D[Dark Mode Toggle]
        E[Waveform Visualization]
        F[Browser Recording]
        G[Detection History]
    end

    subgraph Backend["⚙️ Backend (FastAPI)"]
        H[UVicorn Server]
        I[API Routes]
        J[Feature Extraction]
        K[Model Inference]
        L[Voice Cloning]
        M[Voice Conversion]
        N[Audio Preprocessing]
    end

    subgraph ML["🧠 ML Pipeline"]
        O[MFCC Extractor]
        P[FFT/Spectral Extractor]
        Q[Hybrid Extractor]
        R[Enhanced Extractor]
        S[Pitch/Jitter/Shimmer]
        T[Delta MFCCs]
        U[XGBoost Classifier]
        V[GradientBoosting Classifier]
        W[Ensemble Voting]
    end

    subgraph Generation["🎙️ Voice Generation"]
        X[XTTS v2 Engine]
        Y[IndexTTS2 Engine]
        Z[gTTS Fallback]
    end

    subgraph Conversion["🔄 Voice Conversion"]
        AA[XTTS v2 VC]
        AB[Pitch/Timbre Matching]
    end

    subgraph Preprocessing["🔧 Audio Preprocessing"]
        AC[Noise Reduction]
        AD[Loudness Normalization]
        AE[Format Conversion]
    end

    subgraph Data["💾 Data Layer"]
        AF[WAV Uploads]
        AG[Models .pkl]
        AH[Results JSON]
        AI[Generated Audio]
    end

    A -->|"HTTPS POST /detect"| I
    A -->|"HTTPS POST /batch-detect"| I
    A -->|"HTTPS POST /generate"| I
    A -->|"HTTPS POST /convert-voice"| I
    A -->|"HTTPS GET /health"| I

    I --> J
    I --> N
    I --> L
    I --> M

    J --> O
    J --> P
    J --> Q
    J --> R
    J --> S
    J --> T

    O --> U
    P --> V
    Q --> U
    R --> U
    S --> R
    T --> R
    U --> W
    V --> W

    N --> AC
    N --> AD
    N --> AE

    L --> X
    L --> Y
    L --> Z

    M --> AA
    M --> AB

    X --> AI
    Y --> AI
    Z --> AI
    AA --> AI
    AB --> AI

    W --> AG
    AG --> K
    K --> I

    AC --> AF
    AD --> AF
    AE --> AF
```

## Component Breakdown

### 1. Frontend (GitHub Pages)

**Location:** [`frontend/`](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/tree/master/frontend)

- **Static hosting:** Works without any backend
- **Demo mode:** Fully functional UI with explanations when no backend connected
- **Live mode:** Calls configurable API URL stored in localStorage
- **Features:**
  - File upload with drag-and-drop
  - Audio playback with waveform visualization
  - Real-time results display
  - Browser-based microphone recording
  - Model comparison table
  - Detection history (localStorage)
  - Dark/light mode toggle
  - Ethical use warnings

**Key Files:**
- `index.html` — Main page structure with tab navigation
- `css/main.css` — Styling with CSS variables for theming
- `js/app.js` — Application logic including:
  - Audio recording via MediaRecorder API
  - Waveform visualization via Canvas API
  - Theme management
  - History management
  - API communication

### 2. Backend (FastAPI)

**Location:** [`backend/`](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/tree/master/backend)

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service status, model info, TTS availability |
| `/detect` | POST | Upload WAV → `{prediction, confidence, model_used}` |
| `/batch-detect` | POST | Upload multiple WAVs → batch results |
| `/generate` | POST | Upload speaker WAV + text → base64 audio |
| `/convert-voice` | POST | Source + target WAV → converted voice |

**Features:**
- CORS enabled for GitHub Pages integration
- Automatic audio format conversion (via pydub/ffmpeg)
- Optional preprocessing (noise reduction, normalization)
- Lazy model loading
- Temp file cleanup
- Multiple model support (MFCC, FFT, Hybrid, Enhanced, Ensemble)
- Multiple TTS engines (XTTS v2, IndexTTS2, gTTS)
- Voice conversion (XTTS v2, pitch/timbre matching)

**Key Files:**
- `app.py` — Main application with all endpoints
- `requirements.txt` — Dependencies
- `tests/test_api.py` — Unit tests

### 3. ML Training Pipeline

**Location:** [`training/`](https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator/tree/master/training)

**Model Variants:**

1. **MFCC-only** (13 dimensions)
   - Mel-Frequency Cepstral Coefficients
   - Standard speech processing features

2. **FFT/Spectral-only** (6 dimensions)
   - Spectral centroid, bandwidth, rolloff
   - Band energy ratios

3. **Hybrid** (19 dimensions)
   - Combination of MFCC + FFT features
   - Best overall performance for basic models

4. **Enhanced** (30 dimensions)
   - MFCC + FFT + advanced features:
     - Pitch mean and standard deviation
     - Jitter (pitch period variation)
     - Shimmer (amplitude variation)
     - Delta and delta-delta MFCCs
   - Maximum accuracy

5. **Ensemble** (soft voting)
   - Combines predictions from multiple models
   - Weights based on cross-validation performance
   - Highest reliability

**Classifiers:**
- XGBoost Classifier (primary)
- Gradient Boosting Classifier (fallback)
- StandardScaler preprocessing
- 5-fold stratified cross-validation

**Key Files:**
- `train.py` — Main training script with all model types
- `requirements.txt` — Dependencies including xgboost

### 4. Voice Generation

**Primary Engine: XTTS v2**
- Pip-installable (`pip install TTS>=0.22.0`)
- High-quality zero-shot voice cloning
- Supports 17 languages
- ~2 GB model download on first use

**Secondary Engine: IndexTTS2**
- Zero-shot voice cloning with emotion control
- Requires manual setup with uv
- ~4 GB VRAM for GPU or CPU fallback
- Environment variable: `INDEXTTS_DIR=/opt/index-tts`

**Fallback Engine: gTTS**
- Google Text-to-Speech API
- Generic voice (NOT a clone)
- Requires internet connection
- Used when XTTS/IndexTTS2 unavailable

### 5. Voice Conversion

**Primary Method: XTTS v2 Voice Conversion**
- High-quality zero-shot voice conversion
- Same requirements as XTTS v2 generation
- Preserves content while transforming voice characteristics

**Fallback Method: Pitch/Timbre Matching**
- Spectral equalization
- Pitch shifting via librosa
- Always available (numpy/scipy only)
- Lower quality but no additional dependencies

### 6. Audio Preprocessing

**Noise Reduction**
- Uses `noisereduce` library
- Spectral gating algorithm
- Reduces background noise
- Configurable prop_decrease parameter

**Loudness Normalization**
- Uses `pyloudnorm` library
- Target: -23 LUFS (configurable)
- ITU-R BS.1770-4 compliant

**Format Conversion**
- Uses `pydub` with ffmpeg
- Input: Any format (WAV, MP3, OGG, WebM)
- Output: Mono 16 kHz WAV
- Automatic on upload

## Data Flow

### Detection Flow

```
[User] → [Upload WAV] → [Frontend]
                             ↓
                    [POST /detect]
                             ↓
[Backend] → [Format Conversion] → [Optional Preprocessing]
                             ↓
                    [Feature Extraction]
                             ↓
        [MFCC/FFT/Hybrid/Enhanced/Ensemble]
                             ↓
                    [Model Inference]
                             ↓
                    [JSON Response]
                             ↓
[Frontend] → [Display Result with Confidence & Waveform]
```

### Batch Detection Flow

```
[User] → [Upload Multiple WAVs] → [Frontend]
                                      ↓
                           [POST /batch-detect]
                                      ↓
[Backend] → [Process Each File Concurrently]
                                      ↓
                    [Individual Results Array]
                                      ↓
                    [Aggregate Statistics]
                                      ↓
                           [JSON Response]
```

### Generation Flow

```
[User] → [Upload Speaker WAV + Text] → [Frontend]
                                           ↓
                                [POST /generate]
                                           ↓
[Backend] → [Try XTTS v2] → [Success?] → [Return base64 audio]
                ↓ No
        [Try IndexTTS2] → [Success?] → [Return base64 audio]
                ↓ No
        [Fallback to gTTS] → [Return base64 audio]
```

### Voice Conversion Flow

```
[User] → [Upload Source + Target WAVs] → [Frontend]
                                             ↓
                                  [POST /convert-voice]
                                             ↓
[Backend] → [Try XTTS v2 VC] → [Success?] → [Return base64 audio]
                  ↓ No
        [Pitch/Timbre Matching] → [Return base64 audio]
```

## Security Considerations

1. **No audio storage** — All files processed in memory, deleted after response
2. **UUID filenames** — Prevents path traversal attacks
3. **File size limits** — Configurable via `MAX_UPLOAD_MB` (default 100MB per file)
4. **Consent checkbox** — Required before voice cloning
5. **CORS** — Configurable allowed origins (currently allows all for development)
6. **No authentication** — Add API keys for production use

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

### Option D: Full Production Deployment

For production environments:
1. Add reverse proxy (nginx/traefik) with SSL
2. Implement API key authentication
3. Add rate limiting
4. Configure CORS for specific origins only
5. Use persistent storage for model files
6. Add monitoring and logging

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Vanilla HTML5, CSS3, JavaScript (ES6+) |
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | scikit-learn, XGBoost, scipy, numpy, pandas |
| TTS | Coqui TTS XTTS v2, IndexTTS2, gTTS |
| Audio | pydub, noisereduce, pyloudnorm, librosa |
| CI/CD | GitHub Actions |
| Hosting | GitHub Pages (frontend), GHCR (backend container) |

## Scalability Notes

- **Frontend:** Static hosting scales infinitely via GitHub Pages CDN
- **Backend:** Stateless design allows horizontal scaling
- **ML Models:** Lightweight (< 2MB each), suitable for edge deployment
- **TTS:** XTTS/IndexTTS2 are compute-intensive; consider:
  - GPU acceleration for production
  - Request queueing (Redis/RabbitMQ)
  - Async processing with callbacks

## Feature Matrix

| Feature | Status | Dependencies |
|---------|--------|--------------|
| Single Detection | ✅ | numpy, scipy |
| Batch Detection | ✅ | numpy, scipy |
| MFCC Features | ✅ | numpy, scipy |
| FFT Features | ✅ | numpy, scipy |
| Hybrid Features | ✅ | numpy, scipy |
| Enhanced Features | ✅ | numpy, scipy |
| XGBoost Models | ✅ | xgboost |
| Ensemble Models | ✅ | scikit-learn |
| Noise Reduction | ✅ | noisereduce |
| Loudness Normalization | ✅ | pyloudnorm |
| XTTS v2 Generation | ⚠️ | TTS>=0.22.0 |
| IndexTTS2 Generation | ⚠️ | Manual setup |
| gTTS Fallback | ✅ | gtts |
| XTTS v2 Conversion | ⚠️ | TTS>=0.22.0 |
| Pitch/Timbre Conversion | ✅ | numpy, scipy |
| Browser Recording | ✅ | MediaRecorder API |
| Waveform Viz | ✅ | Canvas API |
| Dark Mode | ✅ | CSS Variables |
| Detection History | ✅ | localStorage |
