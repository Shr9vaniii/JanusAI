# JanusAI — presentation web app

Vite + React + TypeScript UI for FastAPI onboarding engineers / interview demos.
Talks only to the existing API (`/sessions`, `/ask`, `/health`). No agent logic in the browser.

## Local development

Terminal 1 — API (repo root):

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Terminal 2 — UI:

```bash
cd web
npm install
npm run dev
```

Open http://127.0.0.1:5173 (Vite proxies API calls to port 8000).

## Production build (served by FastAPI)

```bash
cd web
npm install
npm run build
```

This writes `web/dist/`. Restart the API; `/` serves the SPA and `/assets/*` are mounted automatically.
Legacy minimal chat remains at `/legacy`.

## Features

- Demo scenario chips (grounded, abstain, follow-up, topic switch, multi, cache)
- Chat with markdown answers + cache/intent/latency pills
- Citations side panel
- Health strip (retrieval / Redis / generation / waking up)
- Demo mode → pipeline Trace drawer (rewrite, decompose, timings)
