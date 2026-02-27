# ──────────────────────────────────────────────────────────────────────────────
# Voice Deepfake API — Docker image
#
# Build (basic, gTTS fallback only):
#   docker build -t deepfake-api .
#
# Build with IndexTTS2 voice cloning (requires ~10 GB disk, ~4 GB VRAM GPU):
#   docker build --build-arg INSTALL_INDEXTTS=1 -t deepfake-api .
#
# Run:
#   docker run -p 8000:8000 deepfake-api
#
# Run with IndexTTS2 (GPU):
#   docker run --gpus all -p 8000:8000 \
#     -e INDEXTTS_DIR=/opt/index-tts \
#     deepfake-api
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Voice Deepfake API"
LABEL org.opencontainers.image.description="FastAPI backend: deepfake detection + IndexTTS2 voice cloning"
LABEL org.opencontainers.image.source="https://github.com/MohammadThabetHassan/Voice-Deepfake-Vishing-Detector-Generator"

ARG INSTALL_INDEXTTS=0

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── Install core backend dependencies ────────────────────────────────────────
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/backend/requirements.txt

# ── Optionally install IndexTTS2 (zero-shot voice cloning) ───────────────────
# This step downloads ~4 GB of model weights — skip for basic deployments.
RUN if [ "$INSTALL_INDEXTTS" = "1" ]; then \
    echo "Installing IndexTTS2..." && \
    git lfs install && \
    git clone https://github.com/index-tts/index-tts.git /opt/index-tts && \
    pip install uv && \
    cd /opt/index-tts && uv sync --all-extras && \
    pip install "huggingface-hub[cli]" && \
    huggingface-cli download IndexTeam/IndexTTS-2 \
        --local-dir /opt/index-tts/checkpoints && \
    echo "IndexTTS2 installation complete."; \
fi

# ── Copy application source ───────────────────────────────────────────────────
COPY backend/ /app/backend/
COPY models/  /app/models/
# Legacy model files at root (for backwards compatibility)
COPY deepfake_detector*.pkl /app/ 2>/dev/null || true

RUN mkdir -p /app/backend/uploads /app/backend/generated

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1001 apiuser && chown -R apiuser:apiuser /app
USER apiuser

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

ENV INDEXTTS_DIR=/opt/index-tts

WORKDIR /app

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
