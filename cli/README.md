# CLI module — interactive local interface

Русская версия: [README.ru.md](README.ru.md)

## Purpose

[cli/](./) is the local interactive adapter for working with Qwen3 TTS from a terminal. It reuses the shared runtime from [../core/README.md](../core/README.md) and does not introduce a separate network service.

Use it when you want to:

- test local synthesis quickly
- run custom voice, voice design, or voice cloning flows manually
- inspect local model availability without starting the HTTP server or Telegram bot

## Entry points

- [__main__.py](__main__.py) — package entry point for `python -m cli`
- [main.py](main.py) — explicit module entry point
- [`run_cli()`](runtime.py:420) and [`CliRuntime`](runtime.py:27) in [runtime.py](runtime.py)

## Launch

```bash
source .venv311/bin/activate
python -m cli
```

The CLI runs locally in the current shell session and uses the same shared directories as the other adapters.

## Runtime wiring

The adapter runtime is assembled through [`build_cli_runtime()`](bootstrap.py:14), and CLI settings are resolved through [`get_cli_settings()`](runtime_config.py:14).

```python
from cli.bootstrap import build_cli_runtime
from cli.runtime_config import get_cli_settings

runtime = build_cli_runtime(get_cli_settings())
```

## Supported workflows

### Custom voice

[`run_custom_session()`](runtime.py:164) guides the user through:

1. selecting a speaker
2. selecting emotion or instruction
3. choosing speed
4. entering synthesis text
5. saving and optionally playing the result

### Voice design

[`run_design_session()`](runtime.py:217) collects a natural-language voice description and then repeatedly synthesizes text with that designed voice profile.

### Voice cloning

[`run_clone_manager()`](runtime.py:247) supports:

- selecting saved voice profiles from [../.voices](../.voices)
- enrolling a new voice profile from reference audio
- running a quick one-off clone flow

## Shared directories

The CLI uses the same storage conventions as the rest of the repository:

- [../.models](../.models) — local model directories
- [../.outputs](../.outputs) — generated outputs
- [../.voices](../.voices) — saved voice profiles for clone workflows
- [../.uploads](../.uploads) — temporary staging area when audio normalization is needed

## Configuration

CLI settings inherit the shared environment contract from [`CoreSettings`](../core/config.py:27).

Useful variables include:

- `QWEN_TTS_MODELS_DIR`
- `QWEN_TTS_OUTPUTS_DIR`
- `QWEN_TTS_VOICES_DIR`
- `QWEN_TTS_UPLOAD_STAGING_DIR`
- `QWEN_TTS_BACKEND`
- `QWEN_TTS_BACKEND_AUTOSELECT`
- `QWEN_TTS_AUTO_PLAY_CLI`

## Operational notes

- The CLI is intended for local interactive usage.
- It does not use Docker-specific entrypoints or compose files.
- Audio playback depends on OS tools invoked by [`maybe_play_audio()`](runtime.py:53).
- Because the interface is interactive, it is best suited for manual testing rather than automation.

## Related docs

- [../README.md](../README.md) — repository quick start
- [../core/README.md](../core/README.md) — shared runtime
- [../server/README.md](../server/README.md) — HTTP adapter
- [../telegram_bot/README.md](../telegram_bot/README.md) — Telegram adapter
