# Модульный локальный TTS runtime

English version: [README.md](README.md)

## Обзор

Репозиторий содержит локальный стек синтеза речи с общим модульным runtime и тремя транспортными адаптерами:

- [server/](server/README.ru.md) — HTTP API на FastAPI
- [telegram_bot/](telegram_bot/README.ru.md) — Telegram-бот на long polling
- [cli/](cli/README.ru.md) — интерактивный локальный CLI
- [core/](core/README.ru.md) — общий runtime, реестр моделей, бэкенды, jobs и observability

После реорганизации Docker-структуры артефакты сборки находятся рядом с соответствующими компонентами:

- образ сервера: [server/Dockerfile](server/Dockerfile)
- образ Telegram-бота: [telegram_bot/Dockerfile](telegram_bot/Dockerfile)
- compose-сценарий сервера: [docker-compose.server.yaml](docker-compose.server.yaml)
- compose-сценарий Telegram-бота: [docker-compose.telegram-bot.yaml](docker-compose.telegram-bot.yaml)

Старые корневые Docker-артефакты, такие как удалённые `Dockerfile` и `compose.yaml`, больше не используются.

## Возможности

- Локальный Qwen3 TTS inference на общей платформе из [core/](core/README.ru.md)
- Локальный OmniVoice inference через общий Torch family-adapter lane
- Локальный Piper inference через ONNX runtime с использованием поддерживаемого Python API `piper-tts`
- OpenAI-совместимый endpoint `POST /v1/audio/speech`
- Расширенные HTTP endpoints для custom voice, voice design и voice cloning
- Команды Telegram-бота `/start`, `/help`, `/tts`, `/design`, `/clone`
- Интерактивный CLI для локальных сценариев синтеза
- Необязательный async job flow в HTTP-сервере
- Структурированные логи, request correlation и операционные метрики
- Изолированная staging-директория для clone-загрузок в [`.uploads/`](.uploads)
- Необязательное сохранение результатов в [`.outputs/`](.outputs)

## Требования

- Python 3.11+
- `ffmpeg`, доступный в `PATH`
- Локальные директории моделей в [`.models/`](.models)
- Для macOS Apple Silicon: MLX-совместимое окружение и MLX-подготовленные артефакты Qwen или ONNX-совместимое окружение для Piper
- Для Linux или Windows: окружение, совместимое с PyTorch/Transformers для Qwen и OmniVoice, либо ONNX runtime для Piper

## Установка

### Наборы зависимостей

- `requirements.txt` — дефолтная локальная operator-установка для стабильного общего окружения, собираемого из `profiles/packs/`
- `requirements-ci.txt` — облегчённый набор для CI/проверок репозитория без тяжёлых optional runtime-зависимостей
- `profiles/packs/base/common.txt` — общий базовый слой для компиляции family/module/platform pack-ов
- `profiles/packs/module/server.txt` — зависимости HTTP-адаптера
- `profiles/packs/module/telegram.txt` — зависимости Telegram-адаптера
- `profiles/packs/family/qwen.txt` — стандартный Qwen Torch lane для общего дефолтного окружения
- `profiles/packs/family/piper.txt` — Piper ONNX lane для общего дефолтного окружения
- `profiles/packs/family/qwen-fast-addon.txt` — optional ускоренный Qwen lane для совместимых CUDA-хостов
- `profiles/packs/family/omnivoice.txt` — отдельный family-pack для выделенного OmniVoice-окружения

Ключевое практическое изменение: OmniVoice больше не описывается как часть дефолтной общей установки. Он по-прежнему поддерживается runtime, но для живого запуска его лучше ставить в отдельное окружение, потому что upstream dependency stack может расходиться со стабильным Qwen-окружением.

### macOS Apple Silicon

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
brew install ffmpeg
```

### Linux

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Быстрый старт в Windows PowerShell

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
choco install ffmpeg -y
```

Если ваша машина разрешает неподписанные локальные PowerShell-скрипты, можно сначала выполнить `Set-ExecutionPolicy -Scope Process Bypass` в текущей сессии. На Windows-хостах, где `MachinePolicy` принудительно задаёт `AllSigned`, это не поможет, поэтому используйте CMD-entrypoint ниже.

### Интерактивный Windows launcher

Для управляемого запуска в Windows на хостах, где разрешён запуск `.ps1`, используйте интерактивный PowerShell-оркестратор:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch-windows.ps1
```

Скрипт переиспользует profile-aware пакет `launcher`, подбирает family/service contour, создаёт или переиспользует `.envs/<family>`, проверяет наличие выбранной модели в `.models/`, при необходимости предлагает скачать отсутствующие артефакты и затем запускает выбранный адаптер.

Если подпись PowerShell-скриптов принудительно включена через `MachinePolicy` (`AllSigned`), используйте Windows CMD wrapper:

```bat
.\scripts\launch-windows.cmd
```

Этот wrapper передаёт содержимое `scripts/launch-windows.ps1` в `powershell.exe -Command` как inline-текст вместо прямого запуска `.ps1`, поэтому file-signing gate не срабатывает, а сам интерактивный flow остаётся тем же.

Важные детали:

- Для Qwen и OmniVoice используется Hugging Face snapshot flow; repo ID запрашивается только если локальная папка модели отсутствует или неполная.
- HF token запрашивается только при необходимости и живёт только в текущем процессе — launcher его не сохраняет.
- Для Piper используется задокументированный `piper.download_voices`, после чего файлы приводятся к именам `model.onnx` и `model.onnx.json`.
- Для Telegram по-прежнему нужен реальный bot token; скрипт может запросить его только на текущий запуск.

### Интерактивный launcher для macOS

Для управляемого запуска на macOS с тем же profile-aware `launcher`, который используется в Windows flow, выполните:

```bash
bash ./scripts/launch-macos.sh
```

Скрипт сохраняет тот же верхнеуровневый сценарий, что и Windows launcher: выбор адаптера, выбор модели/family, создание или переиспользование `.envs/<family>`, проверка isolated environment, валидация локальных артефактов в `.models/`, при необходимости загрузка недостающих файлов и затем запуск выбранного адаптера.

Важные детали для macOS:

- Скрипт рассчитан только на macOS и ожидает `python3.11` и `ffmpeg` в `PATH`.
- Если `python3.11` или `ffmpeg` отсутствуют, скрипт может предложить opt-in установку через Homebrew командой `brew install python@3.11 ffmpeg`, но не делает этого молча.
- Для Qwen и OmniVoice используется тот же guided Hugging Face snapshot flow, а для Piper — тот же `piper.download_voices` с приведением имён к `model.onnx` и `model.onnx.json`.
- HF token и Telegram bot token остаются process-local и не сохраняются launcher’ом на диск.

### Интерактивный launcher для Linux

Для аналогичного управляемого запуска на Linux выполните:

```bash
bash ./scripts/launch-linux.sh
```

Этот launcher повторяет ту же схему выбора сервиса/модели и подготовки окружения через `launcher`, что и Windows/macOS, но оставляет установку системных пакетов полностью ручной.

Важные детали для Linux:

- Скрипт рассчитан только на Linux и ожидает `python3.11` и `ffmpeg` в `PATH`.
- Если системные зависимости отсутствуют, скрипт определяет распространённые package manager’ы (`apt`, `dnf`, `yum`, `pacman`, `zypper`) и печатает точные команды установки, но не выполняет их сам.
- Поведение по загрузке моделей остаётся тем же: Hugging Face snapshot guidance для Qwen и OmniVoice и задокументированный `piper.download_voices` path для Piper.
- HF token и Telegram bot token остаются process-local и не сохраняются launcher’ом на диск.

### Рекомендуемые схемы окружений

#### Дефолтное общее runtime-окружение

Используйте его для стабильного operator lane, который описывает `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Это окружение рассчитано на:

- стандартный Qwen Torch inference
- optional `qwen_fast` route diagnostics в self-check
- Piper ONNX inference

#### Отдельное окружение для OmniVoice

Когда нужен локальный запуск OmniVoice, используйте отдельное окружение:

```bash
python -m pip install --upgrade pip
pip install -r profiles/packs/family/omnivoice.txt
```

На текущем Windows-хосте OmniVoice сейчас требует более новый `transformers` surface, чем стабильное общее Qwen-окружение, поэтому отдельный venv — это безопасный operator path.

#### Optional ускоренное окружение для Qwen

Для CUDA-only ускоренного Qwen lane можно поверх подходящего хоста установить fast-pack:

```bash
python -m pip install --upgrade pip
pip install -r profiles/packs/family/qwen-fast-addon.txt
```

### Host prerequisites для Windows

- `ffmpeg` должен быть доступен в `PATH`
- `sox` настоятельно рекомендуется для host-side audio tooling и некоторых upstream runtime workflows

На текущем хосте `sox` установлен как standalone binary в:

```text
C:\Users\shutov.k.s\AppData\Local\Programs\sox
```

Если `sox --version` всё ещё не работает, добавьте эту директорию в пользовательский `PATH` и откройте новый shell.

### Важная оговорка по Qwen Torch lane

Немаковый Qwen lane зависит от официального Python-пакета `qwen-tts`, который используется в [`TorchBackend`](core/backends/torch_backend.py). Upstream-репозиторий Qwen3-TTS документирует `pip install -U qwen-tts` как стандартный путь установки и после этого использует `from qwen_tts import Qwen3TTSModel`. Поэтому и для Linux, и для Windows у стандартного Torch lane уже есть authoritative install path, но текущий support claim теперь зависит от платформы: для Linux поддержка остаётся частично подтверждённой, пока полный Torch lane не будет эмпирически прогнан на этом хосте, а Windows Torch Qwen support уже считается proven благодаря native host validation, зафиксированной в [docs/support-matrix.md](docs/support-matrix.md).

### Важная оговорка по ускоренному Qwen lane

В репозитории также появился дополнительный backend `qwen_fast` для ускоренного Qwen synthesis в режимах custom, design и clone на поддерживаемых CUDA-хостах. Этот lane optional, не подменяет стандартный `torch` backend key и при отсутствии нужных runtime prerequisites остаётся в состоянии rejected/unresolved route вместо автоматического переключения на Torch execution path.

Pinned README проекта faster-qwen3-tts документирует путь установки ускоренного runtime так:

```bash
pip install faster-qwen3-tts
```

Тот же upstream README указывает prerequisites для fast-lane: Python 3.10+, PyTorch 2.5.1+ и NVIDIA GPU с CUDA. В рамках этого репозитория такой install path нужно считать operator-managed optional dependency для поддерживаемых Linux/Windows CUDA-хостов, а не универсальной зависимостью для любого окружения.

### Примечание по Piper

В репозитории теперь есть поддерживаемый Piper lane через `piper-tts` + `onnxruntime`. На macOS wheel `piper-tts` уже включает `espeakbridge` и `espeak-ng-data`; на других платформах всё равно нужно отдельно проверить phonemization/runtime-совместимость в целевом окружении.

### Примечание по OmniVoice

OmniVoice встроен как **Torch-backed model family**, а не как отдельный backend key. Upstream-проект документирует Python install path вокруг `pip install omnivoice` и `from omnivoice import OmniVoice`. В этом репозитории OmniVoice нужно считать optional operator-managed зависимостью, которая использует уже существующий выбор backend `torch`, но для живого запуска на Linux/Windows её лучше ставить в **отдельное окружение**.

## Модели

Поместите загруженные директории моделей в [`.models/`](.models). Поддерживаемые локальные model ID регистрируются в [`ModelRegistry`](core/services/model_registry.py:20) и описаны в [core/models/manifest.v1.json](core/models/manifest.v1.json).

Обычно используются каталоги:

- `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-1.7B-Base-8bit`
- `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`
- `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit`
- `Qwen3-TTS-12Hz-0.6B-Base-8bit`
- `OmniVoice`
- `Piper-en_US-lessac-medium`

### Формат директории OmniVoice

Для встроенного семейства OmniVoice поместите загруженный Hugging Face snapshot или локально экспортированную модель в `.models/OmniVoice`. Текущий manifest проверяет минимальный root-набор config / weights / tokenizer, а опубликованный upstream model repo дополнительно содержит обязательное поддерево `audio_tokenizer/`. Поэтому практичная локальная раскладка должна включать как минимум:

- `.models/OmniVoice/config.json`
- `.models/OmniVoice/model.safetensors` или `.models/OmniVoice/model.safetensors.index.json`
- `.models/OmniVoice/tokenizer_config.json` или `.models/OmniVoice/tokenizer.json`
- `.models/OmniVoice/chat_template.jinja`
- `.models/OmniVoice/audio_tokenizer/config.json`
- `.models/OmniVoice/audio_tokenizer/model.safetensors`
- `.models/OmniVoice/audio_tokenizer/preprocessor_config.json`

Если модель загружена в виде Hugging Face cache snapshot layout, можно оставить вложенную структуру `snapshots/<revision>/...` — [`TorchBackend`](core/backends/torch_backend.py) уже умеет корректно её разрешать.

### Формат Piper voice directory

Для встроенного Piper lane поместите каждый Piper voice в отдельную директорию внутри `.models/`, например:

- `.models/Piper-en_US-lessac-medium/model.onnx`
- `.models/Piper-en_US-lessac-medium/model.onnx.json`

Скачать Piper voice можно встроенным downloader, например:

```bash
source .venv311/bin/activate
mkdir -p .models/Piper-en_US-lessac-medium
.venv311/bin/python -m piper.download_voices en_US-lessac-medium --download-dir .models/Piper-en_US-lessac-medium
mv .models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx .models/Piper-en_US-lessac-medium/model.onnx
mv .models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx.json .models/Piper-en_US-lessac-medium/model.onnx.json
```

Эквивалент для Windows PowerShell:

```powershell
.\.venv311\Scripts\Activate.ps1
New-Item -ItemType Directory -Force -Path ".models/Piper-en_US-lessac-medium" | Out-Null
python -m piper.download_voices en_US-lessac-medium --download-dir .models/Piper-en_US-lessac-medium
Move-Item ".models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx" ".models/Piper-en_US-lessac-medium/model.onnx" -Force
Move-Item ".models/Piper-en_US-lessac-medium/en_US-lessac-medium.onnx.json" ".models/Piper-en_US-lessac-medium/model.onnx.json" -Force
```

## Runtime self-check

Для проверки selected backend, per-model execution backend, missing artifacts и host/runtime-сигналов используйте встроенную утилиту self-check:

```bash
source .venv311/bin/activate
python scripts/runtime_self_check.py
```

В automation можно включать строгий режим, чтобы получать non-zero exit code при degraded runtime или missing assets:

```bash
python scripts/runtime_self_check.py --strict
```

Если `qwen_fast` включён или рассматривается для маршрутизации, self-check теперь дополнительно показывает `backend_support`, route candidates и явные причины отклонения, чтобы оператор видел, когда ускоренный lane был выбран, а когда нет.

Для OmniVoice тот же self-check теперь показывает его как **Torch-routed family entry**. Он не добавляет новый backend key; вместо этого появляется как model-family элемент, у которого `execution_backend` должен разрешаться в `torch`, если локальные артефакты и optional Python packages доступны.

Для воспроизводимых validation-сценариев используйте automation entry point, а не собирайте команды вручную:

```bash
python scripts/validate_runtime.py host-matrix
python scripts/validate_runtime.py smoke-server
python scripts/validate_runtime.py smoke-server --smoke-model-id Piper-en_US-lessac-medium --expected-backend onnx
python scripts/validate_runtime.py smoke-server --smoke-model-id OmniVoice-Custom --expected-backend torch
python scripts/validate_runtime.py telegram-live --bot-token "$QWEN_TTS_TELEGRAM_BOT_TOKEN"
python scripts/validate_runtime.py telegram-live --bot-token "$QWEN_TTS_TELEGRAM_BOT_TOKEN" --chat-id "$QWEN_TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-chat-id "$QWEN_TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-text "Qwen3-TTS validation ping."
```

- `host-matrix` проверяет текущий host snapshot и симулированные сценарии optional-lane для `qwen_fast`.
- `smoke-server` поднимает локальный HTTP-сервер, ждёт health probes, запускает smoke-suite и затем корректно останавливает runtime.
- `smoke-server --smoke-model-id Piper-en_US-lessac-medium --expected-backend onnx` явно валидирует Piper HTTP path через `POST /v1/audio/speech` и одновременно проверяет, что для этой модели используется ONNX routing.
- `smoke-server --smoke-model-id OmniVoice-Custom --expected-backend torch` валидирует OmniVoice HTTP path через общий Torch family lane.
- `telegram-live` проверяет доступность реального Telegram Bot API и при необходимости может отправить validation message, если передан `--chat-id`.
- Добавьте `--expect-update-chat-id` и при необходимости `--expect-update-text`, если нужен opt-in сценарий с выделенным чатом, который дополнительно подтверждает, что новый подходящий inbound update виден через `getUpdates` без запуска long-polling runtime.

## Optional GRACE CLI install

Для локального lint этого репозитория можно использовать optional `grace` CLI. Публичный GRACE packaging-репозиторий документирует Bun-based install path:

```bash
bun add -g @osovv/grace-cli
grace lint --path /path/to/grace-project
```

В CI этого репозитория `grace` всё равно остаётся optional, потому что мы не требуем Bun на каждом validation host.

## Запуск CLI

```bash
source .venv311/bin/activate
python -m cli
```

```powershell
.\.venv311\Scripts\Activate.ps1
python -m cli
```

Подробности по адаптеру — в [cli/README.ru.md](cli/README.ru.md).

## Запуск HTTP-сервера

### Локально

```bash
source .venv311/bin/activate
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

```powershell
.\.venv311\Scripts\Activate.ps1
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### Через Docker Compose

```bash
docker compose -f docker-compose.server.yaml up --build
```

Этот сценарий собирает образ из [server/Dockerfile](server/Dockerfile) с корневым build context репозитория, монтирует общие рабочие директории и по умолчанию публикует порт `8000`.

Подробности по endpoint'ам, async jobs и конфигурации — в [server/README.ru.md](server/README.ru.md).

## Запуск Telegram-бота

### Локально

```bash
source .venv311/bin/activate
export QWEN_TTS_TELEGRAM_BOT_TOKEN="ваш_токен_бота"
export QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS="123456789,987654321"
export QWEN_TTS_TELEGRAM_ADMIN_USER_IDS="123456789"
export QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED=true
export QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE=20
export QWEN_TTS_TELEGRAM_DELIVERY_STORE_PATH=.state/telegram_delivery_store.json
python -m telegram_bot
```

```powershell
.\.venv311\Scripts\Activate.ps1
$env:QWEN_TTS_TELEGRAM_BOT_TOKEN="ваш_токен_бота"
$env:QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS="123456789,987654321"
$env:QWEN_TTS_TELEGRAM_ADMIN_USER_IDS="123456789"
$env:QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED="true"
$env:QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE="20"
$env:QWEN_TTS_TELEGRAM_DELIVERY_STORE_PATH=".state/telegram_delivery_store.json"
python -m telegram_bot
```

### Через Docker Compose

```bash
docker compose -f docker-compose.telegram-bot.yaml up --build
```

Этот сценарий собирает образ из [telegram_bot/Dockerfile](telegram_bot/Dockerfile), монтирует общие директории моделей и результатов и сохраняет delivery metadata в именованном volume, описанном в [docker-compose.telegram-bot.yaml](docker-compose.telegram-bot.yaml).

### Важная оговорка про Telegram-токен

На текущем Windows-хосте с Docker Desktop Linux containers compose-развёртывание было проверено через реальный старт бота, проверку доступности Telegram API и переход в healthy polling loop. При этом полноценная end-to-end интеграция всё равно требует реального и корректного Telegram-токена и подходящего chat/user context.

Подробности по командам, эксплуатации и деплою — в [telegram_bot/README.ru.md](telegram_bot/README.ru.md).

## Ключевые переменные окружения

Общие настройки читаются через [`CoreSettings.from_env()`](core/config.py:112). Основные переменные:

- `QWEN_TTS_MODELS_DIR`
- `QWEN_TTS_OUTPUTS_DIR`
- `QWEN_TTS_VOICES_DIR`
- `QWEN_TTS_UPLOAD_STAGING_DIR`
- `QWEN_TTS_ACTIVE_FAMILY`
- `QWEN_TTS_DEFAULT_CUSTOM_MODEL`
- `QWEN_TTS_DEFAULT_DESIGN_MODEL`
- `QWEN_TTS_DEFAULT_CLONE_MODEL`
- `QWEN_TTS_BACKEND`
- `QWEN_TTS_BACKEND_AUTOSELECT`
- `QWEN_TTS_SAMPLE_RATE`
- `QWEN_TTS_MAX_INPUT_TEXT_CHARS`

Runtime теперь опирается на явный capability-binding contract. Активный процесс должен трактовать `family`, `custom_model`, `design_model` и `clone_model` как runtime-привязки, а не как вывод о том, какие каталоги случайно существуют на диске. Иными словами, это модели, привязанные к текущему запущенному контуру, а не синоним «скачано локально».

Поведение по умолчанию для этого контракта:

- capability binding отсутствует -> соответствующий режим недоступен для текущего процесса
- capability binding настроен -> запросы могут не передавать `model` и использовать runtime-bound модель

Эти переменные являются source-of-truth контрактом между launcher и runtime. Если capability binding отсутствует, соответствующий режим должен завершаться controlled unsupported-mode response, а не молча откатываться к какой-то другой локальной модели.

Поддерживаемые backend keys теперь включают:

- `mlx` — Qwen3 на Apple Silicon
- `qwen_fast` — optional ускоренный Qwen lane с явной диагностикой готовности и маршрутизации
- `torch` — Qwen3 и OmniVoice на Torch CPU/CUDA-совместимых runtime
- `onnx` — Piper local voice inference через ONNX runtime

Для release-facing platform claims используйте [docs/support-matrix.md](docs/support-matrix.md) как канонический источник.

Настройки HTTP-сервера описаны в [server/README.ru.md](server/README.ru.md), а настройки Telegram-бота — в [telegram_bot/README.ru.md](telegram_bot/README.ru.md).

## Карта документации

- [README.md](README.md) / [README.ru.md](README.ru.md) — корневой quick start
- [core/README.md](core/README.md) / [core/README.ru.md](core/README.ru.md) — общий runtime и архитектура
- [server/README.md](server/README.md) / [server/README.ru.md](server/README.ru.md) — HTTP API
- [telegram_bot/README.md](telegram_bot/README.md) / [telegram_bot/README.ru.md](telegram_bot/README.ru.md) — Telegram-адаптер
- [cli/README.md](cli/README.md) / [cli/README.ru.md](cli/README.ru.md) — интерактивный CLI
