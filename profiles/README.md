# Runtime Profiles

This directory introduces a first-class profile layer for the runtime architecture.

The goal is to stop relying on one shared implicit environment and move toward:

- host profiles
- model-family profiles
- module profiles
- resolved launch profiles

Current Phase 1 scope is non-breaking groundwork only. Existing `server`, `telegram_bot`, and `cli`
entrypoints remain valid while the profile resolver is introduced on top of the current runtime.

## Family isolation direction

The target operating model is **one isolated environment per model family**.

- `qwen` remains the default shared family contour for standard Qwen execution.
- `piper` remains the ONNX-based contour for local Piper voices.
- `omnivoice` is treated as a dedicated Torch family contour with its own dependency pack.

The intent is to stop forcing incompatible upstream dependency stacks into one implicit environment.
