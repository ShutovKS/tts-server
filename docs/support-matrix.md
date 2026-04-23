# Support Matrix

## Evidence levels

- **Proven** — validated by local execution in this repository session and/or CI workflow evidence
- **Partially proven** — validated architecturally and by targeted tests, but not fully exercised end-to-end on that platform/backend combination in this session
- **Best effort** — intended to work, guarded by tests or host checks, but external runtime dependencies may vary by host
- **Unsupported** — not declared as a supported operator path

## Platform × backend × family

| Platform | Backend | Model family | Evidence level | Notes |
|---|---|---|---|---|
| macOS Apple Silicon | MLX | Qwen3-TTS | Proven | Primary optimized path; local regression and live runtime validation exist |
| macOS Apple Silicon | ONNX | Piper | Proven | Local Piper voice synthesis validated through `piper-tts` |
| macOS Apple Silicon | Torch | Qwen3-TTS | Partially proven | Supported by code and tests, but MLX remains the preferred path |
| macOS Apple Silicon | Qwen Fast | Qwen3-TTS | Unsupported | Fast lane is CUDA-oriented and reports explicit rejection/fallback diagnostics on macOS |
| Linux | Torch | Qwen3-TTS | Partially proven | Official upstream install path exists via `pip install -U qwen-tts`; full host validation still remains outstanding |
| Linux | Qwen Fast | Qwen3-TTS | Partially proven | Additive accelerated lane exists in code/tests with explicit CUDA/runtime gating; end-to-end host evidence still depends on CUDA-capable validation |
| Linux | ONNX | Piper | Partially proven | Supported by code and CI-oriented setup guidance |
| Windows | Torch | Qwen3-TTS | Proven | Native Windows validation now covers strict self-check, Torch-only host-matrix checks with qwen_fast disabled by config, and smoke-server execution against the Torch backend on this host |
| Windows | Qwen Fast | Qwen3-TTS | Proven | Native Windows validation now covers strict self-check, host-matrix diagnostics, smoke-server execution, and live HTTP custom/design/clone execution on a CUDA-capable host; explicit Torch fallback remains in place when fast prerequisites are missing |
| Windows | ONNX | Piper | Proven | Native Windows validation now covers strict self-check with a real Piper runtime plus downloaded voice artifacts, direct synthesis evidence, and smoke-server HTTP validation via `--smoke-model-id Piper-en_US-lessac-medium --expected-backend onnx` on this host |
| Any platform | MLX | Piper | Unsupported | Piper routes through ONNX, not MLX |
| Any platform | ONNX | Qwen3-TTS | Unsupported | Qwen3 runtime is not wired to ONNX in this repository |

## Operator interpretation

- The **selected backend** can differ from the **execution backend** for a specific model.
- Mixed-family deployments are expected to report per-model routing explicitly.
- `qwen_fast` is an accelerated Qwen lane for custom, design, and clone on supported CUDA hosts.
- When fast prerequisites are missing, operators should expect explicit route candidates and fallback reasons rather than silent disablement.
- The accelerated runtime itself is operator-managed: use `pip install faster-qwen3-tts` on supported Linux/Windows CUDA hosts and keep the standard Torch lane available as fallback.
- Platform claims should be read together with CI status and runtime self-check output; Windows is no longer treated as a best-effort-only automation lane, and the CUDA-backed `qwen_fast` path now has native-host smoke and live endpoint evidence.
- On the current Windows host, Docker Desktop Linux-container validation now includes successful server image build plus live `/health/live`, `/health/ready`, and `/api/v1/models` probes, with retained evidence in `.sisyphus/evidence/server-docker-health-live.json`, `.sisyphus/evidence/server-docker-health-ready.json`, `.sisyphus/evidence/server-docker-models.json`, and `.sisyphus/evidence/server-docker-log.txt`. Telegram Docker remains a documented compose path, but this retained evidence set does not claim it as proven on this host.
- `grace` CLI is optional and currently documented upstream through the GRACE packaging repository as `bun add -g @osovv/grace-cli`.

## Canonical evidence commands

```bash
python scripts/runtime_self_check.py
python -m pytest -m "unit or integration"
python scripts/validate_runtime.py host-matrix
python scripts/validate_runtime.py docker-server
python scripts/validate_runtime.py representative-models --target piper
python scripts/validate_runtime.py representative-models --target omnivoice
TTS_QWEN_FAST_TEST_MODE=eligible python scripts/runtime_self_check.py
TTS_QWEN_FAST_TEST_MODE=cuda_missing python scripts/runtime_self_check.py
python scripts/validate_runtime.py smoke-server
python scripts/validate_runtime.py smoke-server --smoke-model-id Piper-en_US-lessac-medium --expected-backend onnx
python scripts/validate_runtime.py telegram-live --bot-token "$TTS_TELEGRAM_BOT_TOKEN"
python scripts/validate_runtime.py telegram-live --bot-token "$TTS_TELEGRAM_BOT_TOKEN" --chat-id "$TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-chat-id "$TTS_TELEGRAM_VALIDATION_CHAT_ID" --expect-update-text "Qwen3-TTS validation ping."
python scripts/validate_runtime.py artifact-review
```

## Automation notes

- `python scripts/validate_runtime.py host-matrix` runs the baseline runtime self-check plus simulated `qwen_fast` readiness scenarios (`eligible`, `cuda_missing`, `dependency_missing`) so optional-lane support claims stay tied to real routing evidence.
- `python scripts/validate_runtime.py docker-server` starts the server compose lane with a runtime-selected host port, waits for `/health/live`, `/health/ready`, and `/api/v1/models`, and retains the Docker JSON and log artifacts automatically.
- `python scripts/validate_runtime.py smoke-server` starts a temporary local HTTP server, waits for `/health/live` and `/health/ready`, runs the smoke pytest module, and stops the server automatically.
- Smoke model selection is model-aware: default behavior validates Qwen custom coverage, while `--smoke-model-id Piper-en_US-lessac-medium` (or `TTS_SMOKE_MODEL_ID=Piper-en_US-lessac-medium`) switches smoke requests to the Piper path through `POST /v1/audio/speech` and expects ONNX routing.
- `python scripts/validate_runtime.py telegram-live --bot-token "$TTS_TELEGRAM_BOT_TOKEN"` validates real Telegram Bot API reachability without entering the long-polling loop and records the configured remote-server boundary when `TTS_TELEGRAM_SERVER_BASE_URL` is present. It is not server-execution proof by itself. Add `--chat-id <id>` if you also want an automated `sendMessage` check.
- For Phase 1 operators, `telegram-live` should be read as boundary proof only. Pair it with server-side readiness evidence from the same deployment, usually `GET /health/ready` and `GET /api/v1/models`, before you declare the migration complete.
- Add `--expect-update-chat-id <id>` and optionally `--expect-update-text <substring>` for an opt-in dedicated validation chat where the script also confirms that a matching newer inbound update becomes visible through `getUpdates`.
- `python scripts/validate_runtime.py representative-models --target piper` is the optional representative-model success path on this host, while `--target omnivoice` remains an additive host-specific validation lane that depends on a runtime-ready OmniVoice environment and retained local assets.
- `python scripts/validate_runtime.py artifact-review` is advisory only and reads retained repository-local evidence such as the server Docker JSON and log artifacts plus the CLI launchability transcript.

## Canonical contract pointer

For Phase 1 remote clients and deployment notes, the canonical HTTP server contract is [docs/server-http-contract.md](server-http-contract.md). Use that document as the source of truth for server behavior, async states, readiness semantics, and remote error handling.
