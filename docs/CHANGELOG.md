# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Detection
- **Batch Detection Endpoint** (`POST /batch-detect`) — Process multiple audio files in a single request with per-file error handling
- **Enhanced ML Features** — Added pitch, jitter, shimmer, and delta MFCC extraction for improved detection accuracy
  - Pitch mean and standard deviation
  - Jitter (relative average perturbation of pitch periods)
  - Shimmer (amplitude variation in dB)
  - Delta and delta-delta MFCCs (first and second derivatives)
- **XGBoost Classifier Support** — Alternative to GradientBoosting with better performance on enhanced features
- **Ensemble Models** — Soft voting across multiple classifiers (MFCC, FFT, Hybrid) with cross-validation weighted predictions

#### Audio Preprocessing
- **Noise Reduction** — Optional spectral gating via `noisereduce` library
- **Loudness Normalization** — ITU-R BS.1770-4 compliant normalization to target LUFS (default: -23)

#### Voice Generation
- **Coqui XTTS v2 Support** — Primary TTS engine for high-quality voice cloning
  - Pip-installable (no manual setup required)
  - 17 language support
  - ~2 GB model auto-download
- **Improved TTS Priority** — XTTS v2 → IndexTTS2 → gTTS fallback chain

#### Voice Conversion
- **Voice Conversion Endpoint** (`POST /convert-voice`) — Transform source voice to match target characteristics
  - XTTS v2 zero-shot voice conversion (primary method)
  - Pitch/timbre matching fallback (spectral equalization + pitch shifting)

#### Frontend
- **Browser Recording** — Record audio directly in the browser using MediaRecorder API
  - WebM/Opus codec support with automatic WAV conversion
  - Recording timer display
  - Visual recording status indicator
- **Waveform Visualization** — Canvas-based waveform display for uploaded and recorded audio
- **Dark Mode** — Toggle between light and dark themes with CSS variables
  - Persists preference in localStorage
  - Respects system preference on first visit
- **Detection History** — Local storage of recent detection results
  - Last 10 detections saved
  - Displays timestamp, filename, prediction, confidence, and model used
  - Clear history functionality

### Changed

- **API Version** — Bumped to 2.1.0
- **Health Endpoint** — Extended response with detailed model and TTS information
  - Added `ensemble_info` for ensemble models
  - Added `tts_engine` to indicate which engine is active
  - Added `xtts_v2_available` flag
- **Model Loading Priority** — Now checks for ensemble → hybrid → best → mfcc models
- **Documentation** — Complete rewrite with new features and API reference

### Fixed

- Audio format conversion edge cases with WebM/Opus browser recordings
- Proper cleanup of temporary files after batch processing
- Error handling in batch detection (partial failures don't abort entire batch)

---

## [2.0.0] - 2025-01-15

### Added

- **IndexTTS2 Support** — Zero-shot voice cloning with emotion control
  - Requires manual setup with uv
  - ~4 GB VRAM for GPU acceleration
  - Environment variable `INDEXTTS_DIR` configuration

### Changed

- **TTS Engine Priority** — IndexTTS2 is now primary, gTTS fallback
- **Model Format** — Models now saved as dictionaries with metadata
- **Training Pipeline** — Support for training from CSV or WAV directories

---

## [1.2.0] - 2024-12-10

### Added

- **Docker Support** — Containerized deployment with GHCR publishing
- **GitHub Actions Workflows**:
  - CI pipeline with linting and tests
  - Pages deployment
  - Release automation with model training

### Changed

- **Backend Structure** — Migrated from Flask to FastAPI
- **Frontend Hosting** — Moved to GitHub Pages with static export

---

## [1.1.0] - 2024-11-20

### Added

- **Hybrid Model** — Combined MFCC + FFT features (19 dimensions)
- **Model Comparison Table** — Results tab with metrics comparison
- **Demo Mode** — Full UI functionality without backend

### Changed

- Improved feature extraction performance with vectorized operations

---

## [1.0.0] - 2024-10-01

### Added

- Initial release
- **Detection**:
  - MFCC-only model (13 features)
  - FFT-only model (6 features)
  - GradientBoosting classifier
  - 5-fold stratified cross-validation
- **Generation**:
  - Coqui TTS YourTTS support
  - gTTS fallback
- **Frontend**:
  - Tab-based UI (Detect, Generate, Results, About)
  - File upload with drag-and-drop
  - Audio playback
  - Ethical use warnings
- **Documentation**:
  - Threat model
  - Evaluation methodology
  - Ethics policy

---

## Roadmap

### Planned for v2.2.0

- [ ] Real-time streaming detection via WebSocket
- [ ] VoIP integration (Asterisk/FreeSWITCH)
- [ ] Behavioral biometrics module
- [ ] Confidence calibration with temperature scaling
- [ ] Model versioning and A/B testing support

### Planned for v3.0.0

- [ ] Deep neural network models (CNN/RNN)
- [ ] Multi-modal detection (audio + metadata)
- [ ] API authentication and rate limiting
- [ ] Admin dashboard for model management
- [ ] Support for additional TTS engines (ElevenLabs, Azure)

---

## Migration Guides

### Upgrading to 2.1.0

1. Install new dependencies:
   ```bash
   pip install xgboost noisereduce pyloudnorm
   ```

2. For XTTS v2 support:
   ```bash
   pip install TTS>=0.22.0
   ```

3. Retrain models for ensemble support:
   ```bash
   python training/train.py --csv osr_features.csv --ensemble --output models/
   ```

4. Update frontend static files (dark mode and waveform visualization require new CSS/JS)

### Upgrading to 2.0.0

1. Set `INDEXTTS_DIR` environment variable if using IndexTTS2
2. Models from v1.x are compatible but will be loaded as legacy format
3. New health endpoint response format — update any monitoring scripts

---

## Deprecations

| Feature | Deprecated In | Removal In | Replacement |
|---------|---------------|------------|-------------|
| `server.py` | 1.2.0 | 3.0.0 | `backend/app.py` |
| `pipeline.py` | 1.2.0 | 3.0.0 | `training/train.py` |
| Legacy 18-feature models | 2.0.0 | 3.0.0 | 30-feature enhanced models |
| Coqui YourTTS | 2.0.0 | 2.2.0 | XTTS v2 |

---

## Security Updates

| Version | CVE/Advisory | Description | Fixed |
|---------|--------------|-------------|-------|
| 2.1.0 | — | Added file size limits (10MB) | ✅ |
| 2.0.0 | — | UUID filenames prevent path traversal | ✅ |
| 1.2.0 | — | CORS origin validation | ✅ |

---

## Performance Improvements

| Version | Feature | Improvement |
|---------|---------|-------------|
| 2.1.0 | Batch detection | 5x throughput for multiple files |
| 2.1.0 | Feature extraction | Vectorized numpy operations |
| 2.0.0 | Model loading | Lazy loading with caching |
| 1.2.0 | Audio conversion | pydub with ffmpeg vs pure Python |

---

## Acknowledgments

- Coqui TTS team for XTTS v2
- Bilibili for IndexTTS2
- scikit-learn team for ML infrastructure
- XGBoost contributors for the gradient boosting library
