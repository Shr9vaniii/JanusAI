# Colab GPU inference (free testing)

Serve your fine-tuned LoRA on a free Colab T4 GPU. Retrieval stays on your local PC; generation runs in Colab via ngrok tunnel.

## Quick start

1. Open `deploy/colab/inference_server.ipynb` in [Google Colab](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Restart runtime** (before first install cell)
4. Set paths in the config cell:
   - `LORA_PATH` — where `onboarding_lora_v2` lives on Drive (must include `adapter_model.safetensors`)
   - `NGROK_AUTHTOKEN` — free token from [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)
5. Run all cells — copy the printed `INFERENCE_URL`
6. On your local PC, add to `.env`:

```
INFERENCE_URL=https://xxxx.ngrok-free.app
```

7. Run the full pipeline:

```bash
python -m inference.rag_engine "what arguments does HTTPException take?" --backend remote -v
```

## LoRA on Drive

If you trained with `Fine-tuning/train_unsloth.ipynb`, your adapter is likely at:

```
/content/drive/MyDrive/onboarding_rag/lora_output/onboarding_lora_v2/
```

Or copy the folder from your local `model/onboarding_lora_v2/` to Drive (include `adapter_model.safetensors`).

## Notes

- **Free** — Colab T4 + ngrok free tier
- **Session limits** — Colab disconnects after ~12h or idle; re-run notebook to get a new ngrok URL
- **Keep notebook running** — closing the tab stops the server
- Same `/health` and `/generate` API as `deploy/runpod/`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No adapter weights` | Upload `adapter_model.safetensors` to `LORA_PATH` on Drive |
| ngrok error | Set `NGROK_AUTHTOKEN` in config cell |
| Connection refused locally | Re-copy `INFERENCE_URL` after re-running tunnel cell |
| `libnvJitLink.so.13` / bitsandbytes CUDA error | **Runtime → Restart runtime**, re-run install cell (v3 marker), then pre-flight cell. Upload latest `deploy/colab/` (serve.py + cuda_setup.py). |
| OOM on T4 | Set `MAX_SEQ_LENGTH=1024` in config cell |
| `does not appear to have a file named pytorch_model.bin or model.safetensors` | Upload latest `serve.py` (`drive-cache-4bit-v1`). It downloads the ~5GB 4bit base **once** to `hf_cache/` on Drive, then loads from that local path (avoids broken hub resolve). |
| Re-downloads base every Colab session | Set `HF_CACHE` on Drive (notebook config). Colab's `~/.cache` is wiped on new runtimes. |
| Stale serve.py error in start cell | Drive still has old file — copy `deploy/colab/` from your PC again |
