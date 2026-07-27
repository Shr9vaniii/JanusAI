# RunPod GPU inference (testing)

Serve your fine-tuned LoRA on a RunPod GPU pod. Retrieval stays on your local machine; only generation runs remotely.

## Architecture

```
Local PC (CPU)                         RunPod Pod (GPU)
─────────────────                      ──────────────────
question → intent → retrieve  ──HTTP──►  LoRA Llama 3.1 8B
           format context      ◄────────  structured answer
```

## 1. Create a RunPod pod

1. Go to [runpod.io](https://www.runpod.io) → **Pods** → **Deploy**
2. **GPU:** T4 16GB, L4, or RTX 4090 (8B 4-bit needs ~6–8 GB VRAM)
3. **Template:** `RunPod PyTorch 2.x` or any CUDA 12 + Python 3.10+ image
4. **Volume:** 20 GB+ (stores base model cache + LoRA)
5. **Expose HTTP Ports:** add `8000` (RunPod gives a proxy URL like `https://xxxxx-8000.proxy.runpod.net`)

## 2. Upload LoRA weights to the pod

Your adapter must include `adapter_model.safetensors`. Copy the full `onboarding_lora_v2/` folder to the pod:

```bash
# From your PC (replace POD_IP and key path)
scp -r model/onboarding_lora_v2 root@POD_IP:/workspace/onboarding_lora_v2
```

Or use RunPod **Web Terminal** → upload via `wget` from Google Drive / Hugging Face.

Expected layout on the pod:

```
/workspace/onboarding_lora_v2/
  adapter_config.json
  adapter_model.safetensors   ← required
  tokenizer.json
  ...
```

## 3. Install and start the server (on the pod)

```bash
cd /workspace

# Clone your repo (or upload deploy/runpod only)
git clone https://github.com/YOUR_USER/Onboarding_rag_engine.git
cd Onboarding_rag_engine/deploy/runpod

# Install Unsloth (GPU + CUDA required — run on the pod, not locally)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements.txt

# Start inference server (first run downloads base model ~5 GB)
chmod +x start.sh
LORA_PATH=/workspace/onboarding_lora_v2 ./start.sh
```

First startup takes several minutes while the base model downloads and loads.

## 4. Verify the pod

Set a shared secret on the pod (recommended for public proxy URLs):

```bash
export INFERENCE_API_KEY=some-long-random-token
LORA_PATH=/workspace/onboarding_lora_v2 ./start.sh
```

```bash
curl https://YOUR-POD-ID-8000.proxy.runpod.net/health
```

Expected:

```json
{"status":"ok","lora_path":"/workspace/onboarding_lora_v2","model_loaded":true,"auth_required":true}
```

Test generation:

```bash
curl -X POST https://YOUR-POD-ID-8000.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer some-long-random-token" \
  -d '{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Say hello in one sentence."}]}'
```

## 5. Run the full pipeline from your local PC

Add to your project `.env`:

```
INFERENCE_URL=https://YOUR-POD-ID-8000.proxy.runpod.net
INFERENCE_API_KEY=some-long-random-token
```

Then:

```bash
python -m inference.rag_engine "what arguments does HTTPException take?" --backend remote -v
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Or pass the URL directly:

```bash
python -m inference.rag_engine "your question" \
  --backend remote \
  --inference-url https://YOUR-POD-ID-8000.proxy.runpod.net \
  -v
```

## Cost tip

Stop the pod when you are done testing. You only pay while the pod is running (~$0.20–0.50/hr for T4/4090).
Activate ~2–3 days before interviews; keep a recorded demo as fallback.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Model not loaded` / 503 | Wait for first-time base model download; check pod logs |
| `No adapter weights` | Upload `adapter_model.safetensors` to `LORA_PATH` |
| Connection timeout locally | Confirm port 8000 is exposed; use the RunPod proxy URL, not raw IP |
| OOM on T4 | Set `MAX_SEQ_LENGTH=1024` or use a 24GB GPU |
| 401/403 from `/generate` | Set matching `INFERENCE_API_KEY` on pod and CPU app |

## Environment variables (pod)

| Variable | Default | Description |
|----------|---------|-------------|
| `LORA_PATH` | `/workspace/onboarding_lora_v2` | Path to LoRA adapter directory |
| `PORT` | `8000` | HTTP port |
| `MAX_SEQ_LENGTH` | `2048` | Model context length |
| `INFERENCE_API_KEY` | _(empty)_ | Optional Bearer token required by `/generate` |
