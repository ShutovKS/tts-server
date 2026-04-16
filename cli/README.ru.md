# Модуль CLI — интерактивный локальный интерфейс

English version: [README.md](README.md)

## Назначение

[cli/](./) — локальный интерактивный адаптер для работы с общим multi-family TTS runtime из терминала. Он использует общий runtime из [../core/README.ru.md](../core/README.ru.md) и не поднимает отдельный сетевой сервис.

CLI удобен, когда нужно:

- быстро проверить локальный synthesis flow
- вручную пройти family-aware сценарии custom voice, voice design и voice cloning
- посмотреть доступность локальных моделей без запуска HTTP-сервера или Telegram-бота

## Точки входа

- [__main__.py](__main__.py) — package entry point для `python -m cli`
- [main.py](main.py) — явная модульная точка входа
- [`run_cli()`](runtime.py:420) и [`CliRuntime`](runtime.py:27) в [runtime.py](runtime.py)

## Запуск

```bash
source .venv311/bin/activate
python -m cli
```

```powershell
.\.venv311\Scripts\Activate.ps1
python -m cli
```

CLI работает в текущей shell-сессии и использует те же общие директории, что и остальные адаптеры.

## Сборка runtime

Runtime адаптера собирается через [`build_cli_runtime()`](bootstrap.py:14), а настройки CLI читаются через [`get_cli_settings()`](runtime_config.py:14).

```python
from cli.bootstrap import build_cli_runtime
from cli.runtime_config import get_cli_settings

runtime = build_cli_runtime(get_cli_settings())
```

## Поддерживаемые сценарии

### Выбор семейства

CLI теперь начинает с family-first меню и показывает только действия, которые реально поддерживает выбранное семейство:

- `Qwen3-TTS` — Custom Voice, Voice Design, Voice Clone
- `Piper` — Preset Speaker TTS
- `OmniVoice` — Custom Voice, Voice Design, Voice Clone

### Preset speaker synthesis

[`run_custom_session()`](runtime.py:454) проводит пользователя через:

1. выбор спикера
2. выбор эмоции или instruct
3. выбор скорости
4. ввод текста
5. сохранение и при необходимости воспроизведение результата

Для `Piper` этот же session используется как упрощённый preset-speaker flow: CLI выбирает runtime-ready Piper model, пропускает Qwen-style emotion prompts и сразу делает synthesis через ONNX lane.

### Voice design

[`run_design_session()`](runtime.py:518) принимает prompt для voice design, после чего позволяет многократно синтезировать текст с этим голосовым профилем.

Этот пункт показывается только для семейств с capability `voice_description_tts` (`Qwen3-TTS`, `OmniVoice`).

Для `OmniVoice` текущий runtime ожидает не свободное prose-описание, а structured style tokens. Практические примеры:

- `female`
- `female, whisper`
- `male, british accent`
- `young adult, moderate pitch`

### Voice cloning

[`run_clone_manager()`](runtime.py:542) поддерживает:

- выбор сохранённых voice profiles из [../.voices](../.voices)
- добавление нового voice profile из reference audio
- быстрый разовый clone flow

Этот пункт показывается только для семейств с capability `reference_voice_clone` (`Qwen3-TTS`, `OmniVoice`).

## Общие директории

CLI использует ту же схему хранения, что и остальные части репозитория:

- [../.models](../.models) — локальные директории моделей
- [../.outputs](../.outputs) — результаты генерации
- [../.voices](../.voices) — сохранённые профили голосов для clone flow
- [../.uploads](../.uploads) — временная staging-зона, если нужна нормализация аудио

## Конфигурация

Настройки CLI наследуют общий env-контракт из [`CoreSettings`](../core/config.py:27).

Полезные переменные:

- `QWEN_TTS_MODELS_DIR`
- `QWEN_TTS_OUTPUTS_DIR`
- `QWEN_TTS_VOICES_DIR`
- `QWEN_TTS_UPLOAD_STAGING_DIR`
- `QWEN_TTS_BACKEND`
- `QWEN_TTS_BACKEND_AUTOSELECT`
- `QWEN_TTS_QWEN_FAST_ENABLED`
- `QWEN_TTS_AUTO_PLAY_CLI`

## Эксплуатационные замечания

- CLI рассчитан на локальное интерактивное использование.
- Для него не предусмотрены отдельные Docker entrypoint'ы и compose-файлы.
- Воспроизведение аудио зависит от системных инструментов, которые вызывает [`maybe_play_audio()`](runtime.py:53); в Windows CLI теперь предпочитает нативное открытие файла через системную ассоциацию вместо хрупкого вызова `cmd /c start`.
- Из-за интерактивного интерфейса CLI лучше подходит для ручной проверки, а не для автоматизации.
- Если shared runtime выбирает `qwen_fast`, этот ускоренный lane применяется только к Qwen custom synthesis и при недоступности безопасно уходит в fallback на `torch`.
- `Piper` в CLI намеренно ограничен preset-speaker synthesis, потому что его family adapter поддерживает только capability `preset_speaker_tts`.
- `OmniVoice` переиспользует общий custom/design/clone interaction model, но его доступность по-прежнему зависит от runtime-ready OmniVoice family environment.

## Связанные документы

- [../README.ru.md](../README.ru.md) — корневой quick start
- [../core/README.ru.md](../core/README.ru.md) — общий runtime
- [../server/README.ru.md](../server/README.ru.md) — HTTP-адаптер
- [../telegram_bot/README.ru.md](../telegram_bot/README.ru.md) — Telegram-адаптер
