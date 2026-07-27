# Public deployment checklist

## A. CPU application (API + presentation UI)

1. Build the presentation UI (or rely on the Docker multi-stage build):
   ```bash
   cd web && npm install && npm run build && cd ..
   ```
2. Build and push the Docker image from the repo root (`Dockerfile` builds `web/dist` then the API).
3. Provide env vars from `.env.example`:
   - `GROQ_API_KEY`
   - `INFERENCE_URL` (RunPod proxy URL — set **2–3 days before interviews**)
   - `INFERENCE_API_KEY` (shared secret; same value on RunPod)
   - `REDIS_URL` (optional but recommended)
   - `CORS_ORIGINS` (your public domain, or `*` for a short demo)
   - `RAG_BACKEND=remote`
4. Mount or bake retrieval artifacts:
   - `enterprise_data/chroma_db_v3`
   - `enterprise_data/bm25_index.pkl`
5. Health check: `GET /health`
6. Confirm UI at `/` (Vite app) and `POST /ask` from a clean browser.
   - Legacy chat: `/legacy`

Suggested hosts: Fly.io, Render, Railway, Google Cloud Run.

### Example Fly.io sketch
```bash
fly launch --name onboarding-rag --dockerfile Dockerfile
fly volumes create rag_data --size 5
# attach volume at /app/enterprise_data
fly secrets set GROQ_API_KEY=... INFERENCE_URL=... INFERENCE_API_KEY=... REDIS_URL=... RAG_BACKEND=remote
fly deploy
```

### Example Render / Railway
- Connect the GitHub repo, Dockerfile deploy
- Add a persistent disk mounted at `/app/enterprise_data`
- Set the same secrets as above
- Public URL → share with interviewers

## B. RunPod LoRA (GPU)

Follow [`deploy/runpod/README.md`](../deploy/runpod/README.md).

Minimum:
1. Pod with CUDA + exposed port 8000
2. Upload `onboarding_lora_v2` including `adapter_model.safetensors`
3. `INFERENCE_API_KEY=... ./start.sh`
4. Copy proxy URL into CPU app `INFERENCE_URL`

**Activate the pod 2–3 days before interviews**; stop it when idle to control cost.
If the pod is cold, the UI health strip shows **model waking up**.

## C. Validation before sharing the URL
```bash
# PowerShell
.\scripts\validate_deploy.ps1 https://YOUR_APP

# or bash
./scripts/validate_deploy.sh https://YOUR_APP
```

Also click the six scenario chips in the UI:
1. Grounded HTTPException args
2. Abstention (Redis / K8s)
3. Follow-up attributes (same session after grounded, or seed context)
4. Topic switch to UploadFile
5. Multi-query compound question
6. Cache hit on repeat

## D. Interview package
- Record a 2–3 minute walkthrough ([`DEMO_SCRIPT.md`](DEMO_SCRIPT.md)) as fallback if live GPU is down
- Keep [`INTERVIEW_BRIEF.md`](INTERVIEW_BRIEF.md) open
- Tag a release: `git tag -a v1.0-interview -m "Interview-ready RAG"`

## Known-good config template
```
RAG_BACKEND=remote
INFERENCE_URL=https://xxxxx-8000.proxy.runpod.net
INFERENCE_API_KEY=replace_me
GROQ_API_KEY=replace_me
REDIS_URL=redis://default:****@****:****
CACHE_VERSION=v2
MODEL_VERSION=lora_v2
CORS_ORIGINS=https://YOUR_APP_DOMAIN
```

## Local smoke (no Docker)
```bash
# API
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000

# UI (dev)
cd web && npm run dev
# or serve built assets from the API after npm run build
```
