# Модуль CLI — интерактивный локальный интерфейс

English version: [README.md](README.md)

## Назначение

[cli/](./) — локальный интерактивный адаптер для работы с Qwen3 TTS из терминала. Он использует общий runtime из [../core/README.ru.md](../core/README.ru.md) и не поднимает отдельный сетевой сервис.

CLI удобен, когда нужно:

- быстро проверить локальный synthesis flow
- вручную пройти сценарии custom voice, voice design и voice cloning
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

CLI работает в текущей shell-сессии и использует те же общие директории, что и остальные адаптеры.

## Сборка runtime

Runtime адаптера собирается через [`build_cli_runtime()`](bootstrap.py:14), а настройки CLI читаются через [`get_cli_settings()`](runtime_config.py:14).

```python
from cli.bootstrap import build_cli_runtime
from cli.runtime_config import get_cli_settings

runtime = build_cli_runtime(get_cli_settings())
```

## Поддерживаемые сценарии

### Custom voice

[`run_custom_session()`](runtime.py:164) проводит пользователя через:

1. выбор спикера
2. выбор эмоции или instruct
3. выбор скорости
4. ввод текста
5. сохранение и при необходимости воспроизведение результата

### Voice design

[`run_design_session()`](runtime.py:217) принимает текстовое описание голоса, после чего позволяет многократно синтезировать текст с этим голосовым профилем.

### Voice cloning

[`run_clone_manager()`](runtime.py:247) поддерживает:

- выбор сохранённых voice profiles из [../.voices](../.voices)
- добавление нового voice profile из reference audio
- быстрый разовый clone flow

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
- `QWEN_TTS_AUTO_PLAY_CLI`

## Эксплуатационные замечания

- CLI рассчитан на локальное интерактивное использование.
- Для него не предусмотрены отдельные Docker entrypoint'ы и compose-файлы.
- Воспроизведение аудио зависит от системных инструментов, которые вызывает [`maybe_play_audio()`](runtime.py:53).
- Из-за интерактивного интерфейса CLI лучше подходит для ручной проверки, а не для автоматизации.

## Связанные документы

- [../README.ru.md](../README.ru.md) — корневой quick start
- [../core/README.ru.md](../core/README.ru.md) — общий runtime
- [../server/README.ru.md](../server/README.ru.md) — HTTP-адаптер
- [../telegram_bot/README.ru.md](../telegram_bot/README.ru.md) — Telegram-адаптер
