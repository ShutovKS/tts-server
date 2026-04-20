# Frontend Demo Module

This module contains the standalone static one-page frontend demo for the TTS server.

It is intentionally separate from `server/` so the FastAPI adapter stays focused on server logic only.

## Purpose

The demo provides a minimal operator/demo UI for:

- selecting a local voice preset;
- entering text for synthesis;
- checking clone runtime readiness from the server;
- sending a clone request to the existing HTTP API;
- playing and downloading the generated WAV audio.

## Runtime model

This module is a static frontend only. It does not own backend logic, static server logic, or API routes.

It talks to the existing TTS API by configured base URL.

## Files

- `index.html` — page shell
- `styles.css` — Industrial Cyber-Noir styling
- `app.js` — browser logic for readiness, presets, clone submission, playback, and download
- `voice-presets.json` — local frontend voice preset config

## Running locally

You can serve the module with any static file server.

Because the preset config references repo-local audio files from `.outputs/` and `.models/`, the static server should use the **repository root** as its document root, not `frontend_demo/` itself.

### PowerShell example

```powershell
python -m http.server 8030 -d .
```

Then open:

```text
http://127.0.0.1:8030/frontend_demo/
```

By default the demo targets:

```text
http://127.0.0.1:8000
```

You can override the API base URL by opening the page with a query parameter:

```text
http://127.0.0.1:8030/frontend_demo/?apiBaseUrl=http://127.0.0.1:8020
```

## Important note

The demo checks `/health/ready` on startup and blocks clone submission when the active server runtime does not expose clone capability as both bound and runtime-ready.

The browser client no longer needs to send a model id for clone requests. It relies on the runtime capability bindings advertised by the server and submits only the clone inputs required by the endpoint.

For clone-capable live validation on this host, use the qwen runtime contour described in `server/README.md`.
