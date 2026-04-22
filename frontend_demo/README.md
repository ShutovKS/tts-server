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

By default the demo resolves the API base by deployment mode:

- local browser host (`127.0.0.1`, `localhost`, `0.0.0.0`) -> same host on port `8000`
- non-local HTTP host -> same host on port `8000`
- non-local HTTPS host -> same origin with no explicit port so reverse proxies can terminate TLS and route `/health`, `/api`, and `/v1` server-side

Examples:

```text
http://127.0.0.1:8000
http://185.186.142.205:8000
https://split-tts.drive-vr.ru
```

You can override the API base URL by opening the page with a query parameter:

```text
http://127.0.0.1:8030/frontend_demo/?apiBaseUrl=http://127.0.0.1:8020
```

This is the recommended escape hatch when your frontend and backend are forwarded through different ports or hosts.

## Browser access through SSH forwarding or a public host

When you open the static demo through a forwarded or public browser origin such as:

```text
http://185.186.142.205:8030/frontend_demo/
```

the server must explicitly allow that origin through CORS. Configure the backend with:

```text
TTS_CORS_ALLOWED_ORIGINS=http://127.0.0.1:8030,http://localhost:8030,http://0.0.0.0:8030,http://185.186.142.205:8030,https://split-tts.drive-vr.ru
```

If the backend is reachable through a different forwarded address or port than the page default, open the demo with an explicit API base URL:

```text
http://185.186.142.205:8030/frontend_demo/?apiBaseUrl=http://185.186.142.205:8000
```

Important: when the page itself is served over HTTPS, the browser will block direct requests to a plain HTTP API as mixed content. In that deployment shape, expose the API over HTTPS too, or publish it through the same HTTPS reverse proxy/domain as the frontend.

## Important note

The demo checks `/health/ready` on startup and blocks clone submission when the active server runtime does not expose clone capability as both bound and runtime-ready.

The browser client no longer needs to send a model id for clone requests. It relies on the runtime capability bindings advertised by the server and submits only the clone inputs required by the endpoint.

For clone-capable live validation on this host, use the qwen runtime contour described in `server/README.md`.
