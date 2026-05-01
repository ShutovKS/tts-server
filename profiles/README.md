# Runtime Profiles

This directory introduces a first-class profile layer for the runtime architecture.

The goal is to stop relying on one shared implicit environment and move toward:

- host profiles
- model-family profiles
- module profiles
- resolved launch profiles

Existing `server`, `telegram_bot`, and `cli` entrypoints remain valid for development, but
profile-aware launcher commands are the canonical runtime path when preparing or executing a live
model family contour.

## Family isolation policy

The operating model is **one isolated environment per model family**.

- `qwen` resolves to `.envs/qwen` for standard Qwen execution.
- `piper` resolves to `.envs/piper` for local Piper ONNX voices.
- `omnivoice` resolves to `.envs/omnivoice` for its Torch-backed dependency contour.

The intent is to stop forcing incompatible upstream dependency stacks into one implicit environment.
Launcher JSON responses expose this contract through the `family_environment` payload, including
`environment_isolated: true`, `policy: one_family_one_environment`, the expected env root, the
expected interpreter path, and whether a legacy `.venv311` compatibility environment is present.

`.venv311` and `requirements.txt` remain useful for CI/dev compatibility checks, but they are not
the supported live runtime contour. Use `python -m launcher create-env --family <family> --module
<module> --apply` and `python -m launcher exec --family <family> --module <module>` for runtime
paths.
