# API Documentation

Complete reference for the Voice Deepfake Vishing Detector & Generator API.

## Base URL

```
http://localhost:8000
```

## Content Types

- **Request**: `multipart/form-data` for file uploads, `application/json` for other requests
- **Response**: `application/json`

## Authentication

The API currently does not require authentication. For production deployments, consider adding:

- API key authentication via `Authorization` header
- Rate limiting per IP/API key
- CORS origin restrictions (currently allows all origins)

---

## Endpoints

### Health Check

Check service status and available capabilities.

```http
GET /health
```

**Response** (200 OK):
```json
{
  "status": "ok",
  "uptime_seconds": 42.1,
  "detector_loaded": true,
  "model_name": "deepfake_detector_ensemble",
  "model_type": "ensemble",
  "ensemble_info": {
    "name": "deepfake_detector_ensemble",
    "type": "ensemble",
    "ensemble": true,
    "ensemble_models": ["deepfake_detector_mfcc", "deepfake_detector_fft", "deepfake_detector_hybrid"],
    "voting": "soft"
  },
  "tts_engine": "xtts_v2",
  "xtts_v2_available": true,
  "indextts2_available": false,
  "indextts2_dir": "/opt/index-tts",
  "gtts_available": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Service status (`ok` or `error`) |
| `uptime_seconds` | float | Service uptime in seconds |
| `detector_loaded` | boolean | Whether detection model is loaded |
| `model_name` | string | Name of loaded detection model |
| `model_type` | string | Feature type (`mfcc`, `fft`, `hybrid`, `enhanced`, `ensemble`) |
| `ensemble_info` | object\|null | Ensemble configuration if applicable |
| `tts_engine` | string | Available TTS engine (`xtts_v2`, `indextts2`, `gtts_fallback`, `none`) |
| `xtts_v2_available` | boolean | Whether XTTS v2 is available |
| `indextts2_available` | boolean | Whether IndexTTS2 is available |
| `gtts_available` | boolean | Whether gTTS fallback is available |

---

### Detect Deepfake

Analyze a single audio file for deepfake detection.

```http
POST /detect
```

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apply_noise_reduction` | boolean | `false` | Apply noise reduction preprocessing |
| `apply_normalization` | boolean | `false` | Apply loudness normalization to -23 LUFS |

**Request Body** (`multipart/form-data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | File | Yes | Audio file (WAV, MP3, OGG, WebM supported) |

**Response** (200 OK):
```json
{
  "prediction": "fake",
  "confidence": 0.9142,
  "fake_probability": 0.9142,
  "threshold": 0.8,
  "model_used": "deepfake_detector_ensemble",
  "feature_type": "ensemble",
  "windows_analyzed": 20,
  "voiced_windows": 18,
  "total_windows": 24,
  "inference_time_s": 0.003,
  "notes": "Deepfake voice detected"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | Classification result (`real` or `fake`) |
| `confidence` | float | Confidence score (0.0 to 1.0) |
| `fake_probability` | float | Estimated probability of fake class (0.0 to 1.0) |
| `threshold` | float | Decision threshold used for fake classification |
| `model_used` | string | Name of the model used for inference |
| `feature_type` | string | Feature extraction type used |
| `windows_analyzed` | integer | Number of windows used in aggregation |
| `voiced_windows` | integer | Number of voiced windows detected |
| `total_windows` | integer | Total candidate windows before selection |
| `inference_time_s` | float | Inference time in seconds |
| `notes` | string | Human-readable result description |

**Example**:
```bash
curl -X POST "http://localhost:8000/detect?apply_noise_reduction=true" \
  -F "audio=@sample.wav"
```

---

### Batch Detect

Analyze multiple audio files in a single request.

```http
POST /batch-detect
```

**Request Body** (`multipart/form-data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | File[] | Yes | Multiple audio files (array) |

**Response** (200 OK):
```json
{
  "results": [
    {
      "filename": "sample1.wav",
      "prediction": "fake",
      "confidence": 0.9142,
      "model_used": "deepfake_detector_ensemble",
      "feature_type": "ensemble",
      "inference_time_s": 0.003,
      "status": "success",
      "error_message": null
    },
    {
      "filename": "sample2.wav",
      "prediction": "real",
      "confidence": 0.8234,
      "model_used": "deepfake_detector_ensemble",
      "feature_type": "ensemble",
      "inference_time_s": 0.002,
      "status": "success",
      "error_message": null
    }
  ],
  "total_files": 2,
  "successful": 2,
  "failed": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | Array of detection results |
| `results[].filename` | string | Original filename |
| `results[].prediction` | string\|null | Classification result or null if failed |
| `results[].confidence` | float\|null | Confidence score or null if failed |
| `results[].status` | string | Processing status (`success` or `error`) |
| `results[].error_message` | string\|null | Error message if failed |
| `total_files` | integer | Total number of files submitted |
| `successful` | integer | Number of successfully processed files |
| `failed` | integer | Number of failed files |

**Example**:
```bash
curl -X POST "http://localhost:8000/batch-detect" \
  -F "audio=@sample1.wav" \
  -F "audio=@sample2.wav" \
  -F "audio=@sample3.wav"
```

---

### Generate Voice

Clone a voice from speaker reference audio and synthesize text.

```http
POST /generate
```

**Request Body** (`multipart/form-data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | File | Yes | Speaker reference audio (5+ seconds recommended) |
| `text` | string | Yes | Text to synthesize (max 500 characters) |
| `language` | string | No | Language code (default: `en`, supports 17 languages) |

**Supported Languages** (`language` parameter):
- `en` — English
- `es` — Spanish
- `fr` — French
- `de` — German
- `it` — Italian
- `pt` — Portuguese
- `pl` — Polish
- `tr` — Turkish
- `ru` — Russian
- `nl` — Dutch
- `cs` — Czech
- `ar` — Arabic
- `zh` — Chinese
- `ja` — Japanese
- `hu` — Hungarian
- `ko` — Korean

**Response** (200 OK):
```json
{
  "success": true,
  "method_used": "xtts_v2",
  "audio_base64": "<base64-encoded WAV data>",
  "audio_mime": "audio/wav",
  "generation_time_s": 12.4,
  "notes": "Voice cloned using Coqui XTTS v2 zero-shot TTS. High-quality voice cloning with emotion and timbre derived from the speaker reference audio."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether generation succeeded |
| `method_used` | string | TTS engine used (`xtts_v2`, `indextts2`, `gtts_fallback`) |
| `audio_base64` | string | Base64-encoded WAV audio |
| `audio_mime` | string | MIME type of audio |
| `generation_time_s` | float | Generation time in seconds |
| `notes` | string | Description of the method used |

**Example**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -F "audio=@speaker.wav" \
  -F "text=Hello, this is a cloned voice speaking." \
  -F "language=en"
```

**Decode base64 to file**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -F "audio=@speaker.wav" \
  -F "text=Hello world" | \
  jq -r '.audio_base64' | \
  base64 -d > output.wav
```

---

### Convert Voice

Convert voice characteristics from source to match target voice.

```http
POST /convert-voice
```

**Request Body** (`multipart/form-data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_audio` | File | Yes | Audio to convert (content to preserve) |
| `target_audio` | File | Yes | Target voice reference (characteristics to apply) |

**Response** (200 OK):
```json
{
  "success": true,
  "method_used": "xtts_vc",
  "audio_base64": "<base64-encoded WAV data>",
  "audio_mime": "audio/wav",
  "conversion_time_s": 8.5,
  "notes": "Voice converted using Coqui TTS XTTS v2. High-quality zero-shot voice conversion applied."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether conversion succeeded |
| `method_used` | string | Conversion method (`xtts_vc`, `pitch_shift`) |
| `audio_base64` | string | Base64-encoded WAV audio |
| `audio_mime` | string | MIME type of audio |
| `conversion_time_s` | float | Conversion time in seconds |
| `notes` | string | Description of the method used |

**Example**:
```bash
curl -X POST "http://localhost:8000/convert-voice" \
  -F "source_audio=@source.wav" \
  -F "target_audio=@target.wav" \
  --output converted.wav
```

---

## Error Handling

### Error Response Format

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| `200` | OK | Request successful |
| `400` | Bad Request | Missing required fields, invalid parameters |
| `415` | Unsupported Media Type | Cannot convert audio format (ffmpeg/pydub not installed) |
| `500` | Internal Server Error | Processing error, model inference failure |
| `503` | Service Unavailable | Model not loaded, TTS engine unavailable |

### Common Errors

**400 Bad Request**:
```json
{
  "detail": "No filename provided."
}
```

**400 Invalid Input**:
```json
{
  "detail": "text must be 500 characters or fewer."
}
```

**415 Unsupported Media Type**:
```json
{
  "detail": "Cannot convert '.mp3' to WAV. Install ffmpeg + pydub, or upload WAV directly."
}
```

**500 Detection Failed**:
```json
{
  "detail": "Detection failed: <error details>"
}
```

**503 Model Not Loaded**:
```json
{
  "detail": "Detection model not loaded. Run: python training/train.py --csv osr_features.csv"
}
```

**503 No TTS Available**:
```json
{
  "detail": "No TTS engine available.\n• For voice cloning: pip install TTS>=0.22.0 (XTTS v2, recommended)\n• Alternative: install IndexTTS2 and set INDEXTTS_DIR env var.\n• For basic TTS fallback: pip install gtts"
}
```

---

## Rate Limiting

The API currently does not implement rate limiting. For production deployments, recommended limits:

| Endpoint | Recommended Limit |
|----------|-------------------|
| `/health` | 60 requests/minute |
| `/detect` | 30 requests/minute |
| `/batch-detect` | 10 requests/minute (max 50 files/batch) |
| `/generate` | 10 requests/minute |
| `/convert-voice` | 10 requests/minute |

---

## File Size Limits

| Endpoint | Max File Size | Notes |
|----------|---------------|-------|
| `/detect` | 10 MB | Audio file |
| `/batch-detect` | 50 MB total | All files combined |
| `/generate` | 10 MB | Speaker reference audio |
| `/convert-voice` | 10 MB per file | Both source and target |

---

## Audio Format Support

### Input Formats

| Format | Extension | Requirements |
|--------|-----------|--------------|
| WAV | `.wav` | Native support |
| MP3 | `.mp3` | Requires ffmpeg + pydub |
| OGG | `.ogg` | Requires ffmpeg + pydub |
| WebM | `.webm` | Requires ffmpeg + pydub |
| WebM/Opus | `.webm` | Browser recording native |

### Recommended Format

- **Format**: WAV
- **Sample Rate**: 16 kHz (automatically resampled)
- **Channels**: Mono (automatically converted)
- **Bit Depth**: 16-bit

### Output Format

All generated/converted audio is returned as:
- **Format**: WAV
- **Sample Rate**: 16-24 kHz (depends on TTS engine)
- **Channels**: Mono
- **Encoding**: Base64 in JSON response

---

## Python Client Example

```python
import requests
import base64

API_URL = "http://localhost:8000"

def detect_audio(file_path: str, noise_reduction: bool = False) -> dict:
    """Detect if audio is real or fake."""
    with open(file_path, "rb") as f:
        files = {"audio": f}
        params = {
            "apply_noise_reduction": noise_reduction,
            "apply_normalization": False
        }
        response = requests.post(f"{API_URL}/detect", files=files, params=params)
        response.raise_for_status()
        return response.json()

def batch_detect(file_paths: list[str]) -> dict:
    """Detect multiple audio files."""
    files = [("audio", open(fp, "rb")) for fp in file_paths]
    response = requests.post(f"{API_URL}/batch-detect", files=files)
    for _, f in files:
        f.close()
    response.raise_for_status()
    return response.json()

def generate_voice(speaker_path: str, text: str, language: str = "en") -> bytes:
    """Generate voice clone and return audio bytes."""
    with open(speaker_path, "rb") as f:
        files = {"audio": f}
        data = {"text": text, "language": language}
        response = requests.post(f"{API_URL}/generate", files=files, data=data)
        response.raise_for_status()
        result = response.json()
        return base64.b64decode(result["audio_base64"])

def convert_voice(source_path: str, target_path: str) -> bytes:
    """Convert voice from source to target characteristics."""
    with open(source_path, "rb") as src, open(target_path, "rb") as tgt:
        files = {
            "source_audio": src,
            "target_audio": tgt
        }
        response = requests.post(f"{API_URL}/convert-voice", files=files)
        response.raise_for_status()
        result = response.json()
        return base64.b64decode(result["audio_base64"])

# Example usage
if __name__ == "__main__":
    # Check health
    health = requests.get(f"{API_URL}/health").json()
    print(f"API Status: {health['status']}")
    print(f"Model: {health['model_name']}")
    print(f"TTS Engine: {health['tts_engine']}")
    
    # Detect single file
    result = detect_audio("sample.wav", noise_reduction=True)
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
    
    # Generate voice
    audio_bytes = generate_voice("speaker.wav", "Hello, this is a test.")
    with open("generated.wav", "wb") as f:
        f.write(audio_bytes)
```

---

## JavaScript/TypeScript Client Example

```typescript
const API_URL = "http://localhost:8000";

interface DetectResponse {
  prediction: "real" | "fake";
  confidence: number;
  model_used: string;
  feature_type: string;
  inference_time_s: number;
  notes: string;
}

interface HealthResponse {
  status: string;
  model_name: string;
  tts_engine: string;
}

async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/health`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function detectAudio(
  file: File,
  options?: { noiseReduction?: boolean; normalization?: boolean }
): Promise<DetectResponse> {
  const formData = new FormData();
  formData.append("audio", file);
  
  const params = new URLSearchParams();
  if (options?.noiseReduction) params.append("apply_noise_reduction", "true");
  if (options?.normalization) params.append("apply_normalization", "true");
  
  const response = await fetch(`${API_URL}/detect?${params}`, {
    method: "POST",
    body: formData
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Detection failed");
  }
  
  return response.json();
}

async function batchDetect(files: File[]): Promise<{ results: DetectResponse[] }> {
  const formData = new FormData();
  files.forEach(file => formData.append("audio", file));
  
  const response = await fetch(`${API_URL}/batch-detect`, {
    method: "POST",
    body: formData
  });
  
  if (!response.ok) throw new Error("Batch detection failed");
  return response.json();
}

async function generateVoice(
  speakerAudio: File,
  text: string,
  language: string = "en"
): Promise<Blob> {
  const formData = new FormData();
  formData.append("audio", speakerAudio);
  formData.append("text", text);
  formData.append("language", language);
  
  const response = await fetch(`${API_URL}/generate`, {
    method: "POST",
    body: formData
  });
  
  if (!response.ok) throw new Error("Generation failed");
  const result = await response.json();
  
  // Decode base64 to blob
  const binary = atob(result.audio_base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  
  return new Blob([bytes], { type: result.audio_mime });
}

// Example usage
checkHealth().then(health => {
  console.log(`API ready: ${health.model_name}`);
});
```

---

## Model Information

### Detection Models

| Model | Features | Dimensions | Best For |
|-------|----------|------------|----------|
| MFCC | Mel-Frequency Cepstral Coefficients | 13 | Baseline detection |
| FFT | Spectral features (centroid, bandwidth, rolloff) | 6 | Fast inference |
| Hybrid | MFCC + FFT combined | 19 | Balanced accuracy/speed |
| Enhanced | MFCC + FFT + pitch + jitter + shimmer + delta MFCCs | 30 | Maximum accuracy |
| Ensemble | Soft voting across models | Varies | Highest reliability |

### TTS Engines

| Engine | Quality | Speed | Requirements |
|--------|---------|-------|--------------|
| XTTS v2 | Excellent | Medium | `pip install TTS`, ~2GB download |
| IndexTTS2 | Excellent | Slow (CPU) | Clone repo, ~4GB VRAM/CPU, uv setup |
| gTTS | Poor | Fast | Internet connection only |

### Voice Conversion Methods

| Method | Quality | Requirements |
|--------|---------|--------------|
| XTTS v2 | Excellent | `pip install TTS` |
| Pitch/Timbre | Moderate | numpy, scipy (always available) |

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and feature additions.
