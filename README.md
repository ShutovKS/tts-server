# TTS Server

## Features

- Local text-to-speech inference using `mlx_audio`
- Interactive CLI package in [`cli/`](cli)
- CLI package entry point via [`cli/__main__.py`](cli/__main__.py)
- API server entry point via [`server/__init__.py`](server/__init__.py)
- OpenAI-style compatible `POST /v1/audio/speech`
- Extended endpoints for custom voice, voice design, and voice cloning
- Unified JSON error format with request id correlation
- Deep readiness/liveness probes with model, runtime, and configuration diagnostics
- Structured request and service logs with request tracing
- Optional output persistence in [`.outputs/`](.outputs)
- Multi-layer test suite split into unit, integration, smoke, and architecture tests

## Requirements

- Python 3.11+
- `ffmpeg` available in `PATH`
- Local model directories placed in [`.models/`](.models)
- For macOS Apple Silicon: MLX-compatible dependencies and local MLX-converted artifacts
- For Linux and Windows: an environment compatible with PyTorch + Transformers

## Installation

### macOS Apple Silicon

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
brew install ffmpeg
```

### Linux or Windows

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Models

Place downloaded model directories in [`.models/`](.models). Supported local directories are defined in [`core/services/model_registry.py`](core/services/model_registry.py):

- `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-1.7B-Base-8bit`
- `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-0.6B-Base-8bit`

## CLI Mode

```bash
source .venv311/bin/activate
python -m cli
```

- [`cli/__main__.py`](cli/__main__.py) is the thin package entry point;
- [`cli/main.py`](cli/main.py) remains the explicit module entry point;
- [`cli/runtime.py`](cli/runtime.py) manages the interactive runtime flow;
- [`cli/bootstrap.py`](cli/bootstrap.py) wires CLI components together;
- [`cli/runtime_config.py`](cli/runtime_config.py) resolves CLI settings using shared parsing helpers from [`core/config.py`](core/config.py).

## API Server Mode

Run the API server through Uvicorn from an activated virtual environment:

```bash
source .venv311/bin/activate
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### Environment Variables

Shared core-layer environment parsing now lives in [`core/config.py`](core/config.py). [`server/bootstrap.py`](server/bootstrap.py) only adds server-specific adapter settings on top of it:

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

Example:

```bash
export QWEN_TTS_BACKEND=torch
export QWEN_TTS_BACKEND_AUTOSELECT=true
export QWEN_TTS_DEFAULT_SAVE_OUTPUT=false
export QWEN_TTS_MAX_UPLOAD_SIZE_BYTES=26214400
python -m uvicorn server --host 0.0.0.0 --port 8000
```

## API Endpoints

The public endpoints are unchanged. [`server/app.py`](server/app.py) is now only the composition root, while handlers are split across adapter modules:

- [`server/api/routes_health.py`](server/api/routes_health.py)
- [`server/api/routes_models.py`](server/api/routes_models.py)
- [`server/api/routes_tts.py`](server/api/routes_tts.py)
- [`server/api/responses.py`](server/api/responses.py)
- [`server/api/errors.py`](server/api/errors.py)

Endpoints:

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/models`
- `POST /v1/audio/speech`
- `POST /api/v1/tts/custom`
- `POST /api/v1/tts/design`
- `POST /api/v1/tts/clone`

## Request Examples

### OpenAI-style Speech

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

### Custom Voice

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

### Voice Design

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tts/design \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Design a new narrator voice",
    "voice_description": "deep calm documentary narrator"
  }' \
  --output design.wav
```

### Voice Clone

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tts/clone \
  -F 'text=Clone this sentence' \
  -F 'ref_text=Clone this sentence' \
  -F 'ref_audio=@./sample.wav' \
  --output clone.wav
```

## Error Format

Non-audio errors use the unified schema from [`server/schemas/errors.py`](server/schemas/errors.py):

```json
{
  "code": "model_not_available",
  "message": "Requested model is not available",
  "details": {},
  "request_id": "..."
}
