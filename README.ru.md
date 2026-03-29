# TTS Server

## Возможности

- Локальный text-to-speech inference с использованием `mlx_audio`
- Интерактивный CLI-пакет в [`cli/`](cli)
- Точка входа CLI-пакета через [`cli/__main__.py`](cli/__main__.py)
- Точка входа API-сервера через пакет [`server/__init__.py`](server/__init__.py)
- Совместимый с OpenAI-стилем `POST /v1/audio/speech`
- Расширенные эндпоинты для custom voice, voice design и voice cloning
- Единый JSON-формат ошибок с корреляцией по request id
- Углублённые readiness/liveness probes с диагностикой модели, runtime и конфигурации
- Структурированные логи запросов и сервиса с трассировкой запросов
- Необязательное сохранение выходных файлов в [`.outputs/`](.outputs)
- Многоуровневый набор тестов, разделённый на unit, integration, smoke и architecture

## Требования

- Python 3.11+
- `ffmpeg`, доступный в `PATH`
- Локальные папки моделей, размещённые в [`.models/`](.models)
- Для macOS Apple Silicon: MLX-совместимые зависимости и локальные MLX-конвертированные артефакты
- Для Linux и Windows: окружение, совместимое с PyTorch + Transformers

## Установка

### macOS Apple Silicon

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
brew install ffmpeg
```

### Linux или Windows

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Модели

Поместите загруженные папки моделей в [`.models/`](.models). Поддерживаемые локальные папки определяются в [`core/services/model_registry.py`](core/services/model_registry.py):

- `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-1.7B-Base-8bit`
- `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-0.6B-Base-8bit`

## Режим CLI

```bash
source .venv311/bin/activate
python -m cli
```

- [`cli/__main__.py`](cli/__main__.py) — это тонкая точка входа пакета;
- [`cli/main.py`](cli/main.py) остаётся явной модульной точкой входа;
- [`cli/runtime.py`](cli/runtime.py) управляет интерактивным потоком;
- [`cli/bootstrap.py`](cli/bootstrap.py) отвечает за связывание CLI-компонентов;
- [`cli/runtime_config.py`](cli/runtime_config.py) разрешает настройки CLI с использованием общих помощников парсинга из [`core/config.py`](core/config.py).

## Режим API-сервера

Запускайте API-сервер через Uvicorn из активированного виртуального окружения:

```bash
source .venv311/bin/activate
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### Переменные окружения

Общий парсинг переменных окружения core-слоя теперь находится в [`core/config.py`](core/config.py). [`server/bootstrap.py`](server/bootstrap.py) добавляет поверх него только server-специфичные adapter-настройки:

- `QWEN_TTS_MODELS_DIR`
- `QWEN_TTS_OUTPUTS_DIR`
- `QWEN_TTS_VOICES_DIR`
- `QWEN_TTS_BACKEND`
- `QWEN_TTS_BACKEND_AUTOSELECT`
- `QWEN_TTS_HOST`
- `QWEN_TTS_PORT`
- `QWEN_TTS_LOG_LEVEL`
- `QWEN_TTS_DEFAULT_SAVE_OUTPUT`
- `QWEN_TTS_ENABLE_STREAMING`
- `QWEN_TTS_MAX_UPLOAD_SIZE_BYTES`
- `QWEN_TTS_REQUEST_TIMEOUT_SECONDS`
- `QWEN_TTS_INFERENCE_BUSY_STATUS_CODE`
- `QWEN_TTS_SAMPLE_RATE`
- `QWEN_TTS_FILENAME_MAX_LEN`
- `QWEN_TTS_AUTO_PLAY_CLI`

Пример:

```bash
export QWEN_TTS_BACKEND=torch
export QWEN_TTS_BACKEND_AUTOSELECT=true
export QWEN_TTS_DEFAULT_SAVE_OUTPUT=false
export QWEN_TTS_MAX_UPLOAD_SIZE_BYTES=26214400
python -m uvicorn server --host 0.0.0.0 --port 8000
```

## API-эндпоинты

Публичные эндпоинты не изменились. [`server/app.py`](server/app.py) теперь является только composition root, а обработчики разделены по adapter-модулям:

- [`server/api/routes_health.py`](server/api/routes_health.py)
- [`server/api/routes_models.py`](server/api/routes_models.py)
- [`server/api/routes_tts.py`](server/api/routes_tts.py)
- [`server/api/responses.py`](server/api/responses.py)
- [`server/api/errors.py`](server/api/errors.py)

Эндпоинты:

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/models`
- `POST /v1/audio/speech`
- `POST /api/v1/tts/custom`
- `POST /api/v1/tts/design`
- `POST /api/v1/tts/clone`

## Примеры запросов

### Speech в OpenAI-стиле

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "input": "Hello from Qwen3-TTS",
    "voice": "Vivian",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

### Custom voice

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tts/custom \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "This is a custom voice request",
    "speaker": "Vivian",
    "emotion": "Calm and warm",
    "speed": 1.0,
    "save_output": true
  }' \
  --output custom.wav
```

### Voice design

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tts/design \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Design a new narrator voice",
    "voice_description": "deep calm documentary narrator"
  }' \
  --output design.wav
```

### Voice clone

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tts/clone \
  -F 'text=Clone this sentence' \
  -F 'ref_text=Clone this sentence' \
  -F 'ref_audio=@./sample.wav' \
  --output clone.wav
```

## Формат ошибок

Ошибки, не связанные с аудио, используют единую схему из [`server/schemas/errors.py`](server/schemas/errors.py):

```json
{
  "code": "model_not_available",
  "message": "Requested model is not available",
  "details": {},
  "request_id": "..."
}
```
