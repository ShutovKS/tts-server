# CLI module — interactive local interface

Русская версия: [README.ru.md](README.ru.md)

## Purpose

[cli/](./) is the local interactive adapter for working with the shared multi-family TTS runtime from a terminal. It reuses the shared runtime from [../core/README.md](../core/README.md) and does not introduce a separate network service.

Use it when you want to:

- test local synthesis quickly
- run family-aware custom voice, voice design, or voice cloning flows manually
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

```powershell
.\.venv311\Scripts\Activate.ps1
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

### Family selection

The CLI now opens with a family-first menu and only shows actions supported by the selected family:

- `Qwen3-TTS` — Custom Voice, Voice Design, Voice Clone
- `Piper` — Preset Speaker TTS
- `OmniVoice` — Custom Voice, Voice Design, Voice Clone

### Preset speaker synthesis

[`run_custom_session()`](runtime.py:454) guides the user through:

1. selecting a speaker
2. selecting emotion or instruction
3. choosing speed
4. choosing language (`auto` by default)
5. entering synthesis text
5. saving and optionally playing the result

For `Piper`, the same session is used as a simplified preset-speaker flow: the CLI selects a runtime-ready Piper model, skips Qwen-style emotion prompts, and synthesizes text directly through the ONNX runtime lane.

### Voice design

[`run_design_session()`](runtime.py:518) collects a voice-design prompt, prompts for language with default `auto`, and then repeatedly synthesizes text with that designed voice profile.

This action is shown only for families whose models expose `voice_description_tts` capability (`Qwen3-TTS`, `OmniVoice`).

For `OmniVoice`, the current runtime expects structured style tokens rather than a free-form prose description. Practical examples:

- `female`
- `female, whisper`
- `male, british accent`
- `young adult, moderate pitch`

### Voice cloning

[`run_clone_manager()`](runtime.py:542) supports:

- selecting saved voice profiles from [../.voices](../.voices)
- enrolling a new voice profile from reference audio
- running a quick one-off clone flow with language prompt defaulting to `auto`

This action is shown only for families whose models expose `reference_voice_clone` capability (`Qwen3-TTS`, `OmniVoice`).

For Qwen clone, treat the optional transcript as an exact transcript of the reference audio, not a rough hint. If the transcript is unknown, leave it empty rather than entering placeholder text.

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
- `QWEN_TTS_QWEN_FAST_ENABLED`
- `QWEN_TTS_AUTO_PLAY_CLI`

## Operational notes

- The CLI is intended for local interactive usage.
- It does not use Docker-specific entrypoints or compose files.
- Audio playback depends on OS tools invoked by [`maybe_play_audio()`](runtime.py:53); on Windows the CLI now prefers the native file association launcher instead of shelling through `cmd /c start`.
- Because the interface is interactive, it is best suited for manual testing rather than automation.
- If the shared runtime selects `qwen_fast`, that accelerated lane applies only to Qwen custom synthesis and still falls back safely to `torch` when unavailable.
- `Piper` is intentionally limited to preset-speaker synthesis in the CLI because its family adapter exposes only `preset_speaker_tts` capability.
- `OmniVoice` reuses the shared custom/design/clone interaction model, but its availability still depends on a runtime-ready OmniVoice family environment.

## V1 validation lane

CLI validation is intentionally hybrid in V1.

- The automated baseline is launchability only: run `python -m cli`, select a family, then feed stdin `q`, and retain stdout/stderr at `../.sisyphus/evidence/cli-launchability-transcript.txt`.
- Complex interactive paths stay transcript-based rather than fully automated: retain `../.sisyphus/evidence/cli-custom-voice-transcript.txt`, `../.sisyphus/evidence/cli-design-session-transcript.txt`, `../.sisyphus/evidence/cli-clone-manager-transcript.txt`, and `../.sisyphus/evidence/cli-playback-transcript.txt` for the corresponding manual sessions.
- When storing those transcripts, include enough context to reconnect the run to the exercised flow: timestamp, `python -m cli` command shape, selected model or voice profile, and any generated output path under `../.outputs/`.
- Docker is explicitly out of scope for the CLI in V1; do not reinterpret the server or Telegram compose lanes as CLI validation requirements.

## Related docs

- [../README.md](../README.md) — repository quick start
- [../core/README.md](../core/README.md) — shared runtime
- [../server/README.md](../server/README.md) — HTTP adapter
- [../telegram_bot/README.md](../telegram_bot/README.md) — Telegram adapter
