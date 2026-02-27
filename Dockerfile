# ──────────────────────────────────────────────────────────────────────────────
# Voice Deepfake API — Docker image
# ──────────────────────────────────────────────────────────────────────────────
# Build:
#   docker build -t deepfake-api .
# Run:
#   docker run -p 8000:8000 deepfake-api
#   # With GPU (if available):
#   docker run --gpus all -p 8000:8000 deepfake-api
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Voice Deepfake API"
LABEL org.opencontainers.image.description="FastAPI backend for voice deepfake detection and generation"
LABEL org.opencontainers.image.source="https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator"

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python env ────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/backend/requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY backend/ /app/backend/
COPY models/  /app/models/
# Legacy model files at root (for compatibility)
COPY deepfake_detector*.pkl /app/ 2>/dev/null || true

# ── Create directories ────────────────────────────────────────────────────────
RUN mkdir -p /app/backend/uploads /app/backend/generated

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m -u 1001 apiuser && \
    chown -R apiuser:apiuser /app
USER apiuser

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

WORKDIR /app

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
