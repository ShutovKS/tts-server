# FILE: server/Dockerfile
# VERSION: 1.0.0
# START_MODULE_CONTRACT
#   PURPOSE: Define the container image for the HTTP server deployment lane.
#   SCOPE: Base image selection, runtime environment defaults, trust-store wiring, dependency installation, source copy, runtime directory creation, and default server command.
#   DEPENDS: M-SERVER, M-BOOTSTRAP, M-CONFIG
#   LINKS: M-DOCKER-ASSETS
#   ROLE: CONFIG
#   MAP_MODE: SUMMARY
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   base image and ENV - Establish container runtime defaults and mounted working directories.
#   certificate install - Inject the corporate proxy root certificate into the container trust store.
#   apt dependencies - Install ffmpeg and required system packages.
#   Python requirements prep/install - Filter incompatible packages and install runtime dependencies.
#   source copy - Copy the shared runtime and HTTP adapter sources into the image.
#   runtime directories - Create writable model, output, upload, and voice mount points.
#   default command - Start the FastAPI server through uvicorn.
# END_MODULE_MAP
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    QWEN_TTS_BACKEND=torch \
    QWEN_TTS_BACKEND_AUTOSELECT=true \
    QWEN_TTS_HOST=0.0.0.0 \
    QWEN_TTS_PORT=8000 \
    QWEN_TTS_MODELS_DIR=/app/.models \
    QWEN_TTS_OUTPUTS_DIR=/app/.outputs \
    QWEN_TTS_VOICES_DIR=/app/.voices \
    QWEN_TTS_UPLOAD_STAGING_DIR=/app/.uploads

WORKDIR /app

COPY docker/certs/infowatch-transparent-proxy-root.crt /usr/local/share/ca-certificates/infowatch-transparent-proxy-root.crt

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        sox \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY profiles ./profiles
RUN python - <<'PY'
from pathlib import Path

source = Path('requirements.txt')
target = Path('requirements.docker.txt')
lines = source.read_text(encoding='utf-8').splitlines()
filtered = [line for line in lines if not line.strip().startswith('flash-attn')]
target.write_text('\n'.join(filtered) + '\n', encoding='utf-8')
PY
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip setuptools wheel \
    && pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.docker.txt \
    && pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org qwen-tts==0.1.1

COPY cli ./cli
COPY core ./core
COPY server ./server
COPY .env.example ./.env.example

RUN mkdir -p /app/.models /app/.outputs /app/.uploads /app/.voices

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
