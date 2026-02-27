# Threat Model — Voice Deepfake Vishing Detector

## System Overview

The system processes audio samples to classify them as real or synthesised (deepfake), and provides voice cloning for research/demonstration. It is deployed as a local or containerised backend with a static GitHub Pages frontend.

---

## Assets

| Asset | Sensitivity | Location |
|---|---|---|
| Trained ML models (`.pkl`) | Medium | `models/` directory |
| Uploaded audio samples | High | Backend memory only (not persisted) |
| Generated audio files | High | Temporary files, deleted after response |
| Feature CSV training data | Low–Medium | `osr_features.csv`, `features.csv` |
| API endpoint | Low | Public (designed for open research) |
| User localStorage (API URL) | Low | Browser only |

---

## Threat Actors

| Actor | Motivation | Capability |
|---|---|---|
| Attacker using cloning for vishing | Financial fraud | Medium–High |
| Researcher studying evasion | Academic | High |
| Script kiddie | Curiosity | Low |

---

## STRIDE Analysis

### Spoofing
- **Threat**: Attacker uploads a carefully crafted audio to fool the classifier.
- **Mitigation**: Use ensemble of feature types (MFCC + FFT); periodically retrain on adversarial examples; monitor confidence scores.

### Tampering
- **Threat**: Model files (`*.pkl`) replaced on the server.
- **Mitigation**: Run with least-privilege user; use file integrity monitoring; verify model SHA256 on load.

### Repudiation
- **Threat**: No audit log for who generated voice clones.
- **Mitigation**: Log metadata (file size, timestamp, consent flag) without storing audio; require explicit consent checkbox.

### Information Disclosure
- **Threat**: Uploaded audio exposed to other users.
- **Mitigation**: Audio is processed in memory; temp files deleted after each request; no database storage.

### Denial of Service
- **Threat**: Large audio uploads / Coqui TTS model load consuming all CPU.
- **Mitigation**: Limit upload size (10 MB max in backend); rate-limit API with a reverse proxy (nginx); use Uvicorn worker limits.

### Elevation of Privilege
- **Threat**: Path traversal in filename (e.g. `../../etc/passwd`).
- **Mitigation**: Use `uuid4().hex` for server-side filenames; `Path.name` strips directory components.

---

## Data Flow Diagram

```
[Browser]
   │ HTTPS POST /detect (WAV ≤10 MB)
   ▼
[FastAPI Backend]
   │ pydub → normalize to mono 16 kHz WAV (temp file)
   │ feature extraction (MFCC/FFT/Hybrid)
   │ load model → predict
   │ temp file deleted
   ▼
[JSON Response] → Browser (no audio stored)

[Browser]
   │ HTTPS POST /generate (speaker WAV + text)
   ▼
[FastAPI Backend]
   │ normalize WAV (temp file)
   │ Coqui TTS / gTTS synthesis (temp output file)
   │ base64 encode → JSON response
   │ both temp files deleted
   ▼
[JSON Response with base64 audio] → Browser
```

---

## Transport Security

- **TLS (HTTPS)** protects audio in transit between browser and backend.
- **TLS does NOT authenticate the voice**. A deepfake audio served over HTTPS is still a deepfake.
- For production deployment, place Nginx or Caddy as a TLS-terminating reverse proxy in front of Uvicorn.

---

## Consent & Legal

The system requires:
1. UI checkbox: _"I confirm I have explicit consent from the speaker"_ before any generation.
2. Prominent ethics banner on every page load.
3. Documentation clearly stating research/educational use only.

---

## Limitations

- The classifier is trained on limited data (~400–500 samples). Real-world deployment requires much larger, more diverse datasets.
- Adversarial audio crafted to look like real voice can fool MFCC/FFT classifiers.
- gTTS fallback does NOT clone the speaker's voice — it uses Google's TTS engine. This is clearly labelled.
- No speaker verification against a claimed identity (out of scope).
