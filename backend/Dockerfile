# Production image for the VoiceMarkers API (also used by docker-compose for dev).
# Hugging Face Spaces builds this file and expects the app on port 7860.

# --- Stage 1: install Python dependencies into a virtualenv -------------------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY pyproject.toml ./
COPY app ./app
RUN pip install .

# --- Stage 2: slim runtime ----------------------------------------------------
FROM python:3.11-slim
# ffmpeg decodes the WebM/MP4 recordings browsers produce (in memory, via pipes).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# Run as an unprivileged user (uid 1000 is also what Hugging Face Spaces uses).
RUN useradd --create-home --uid 1000 appuser

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    # numba (used by shap) and matplotlib want writable cache directories.
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
# Only the app and the model: no dataset, research code or tests.
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser models ./models
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['PORT']}/health\", timeout=4)"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
