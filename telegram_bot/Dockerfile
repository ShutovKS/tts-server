FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QWEN_TTS_BACKEND=torch \
    QWEN_TTS_BACKEND_AUTOSELECT=true \
    QWEN_TTS_MODELS_DIR=/app/.models \
    QWEN_TTS_OUTPUTS_DIR=/app/.outputs \
    QWEN_TTS_VOICES_DIR=/app/.voices \
    QWEN_TTS_UPLOAD_STAGING_DIR=/app/.uploads \
    QWEN_TTS_TELEGRAM_DELIVERY_STORE_PATH=/app/.state/telegram_delivery_store.json

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python - <<'PY'
from pathlib import Path

source = Path('requirements.txt')
target = Path('requirements.docker.txt')
lines = source.read_text(encoding='utf-8').splitlines()
filtered = [line for line in lines if not line.strip().startswith('flash-attn')]
target.write_text('\n'.join(filtered) + '\n', encoding='utf-8')
PY
RUN pip install --upgrade pip setuptools wheel \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.docker.txt \
    && pip install qwen-tts==0.1.1

COPY core ./core
COPY telegram_bot ./telegram_bot
COPY .env.example ./.env.example

RUN mkdir -p /app/.models /app/.outputs /app/.uploads /app/.voices /app/.state

CMD ["python", "-m", "telegram_bot"]
