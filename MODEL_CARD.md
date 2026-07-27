# Model Card — onboarding_lora_v2

## Model details
- **Name:** `onboarding_lora_v2`
- **Base:** `unsloth/Llama-3.1-8B-Instruct`
- **Adaptation:** QLoRA (4-bit NF4) LoRA adapters
- **Task:** Grounded FastAPI onboarding Q&A over a hybrid-retrieved context
- **Output style:** Structured answers (`Direct Answer` / `What I found` / `Recommendation`) with abstention when context is insufficient

## Intended use
- Portfolio / interview demo of an AI/LLM engineering system
- Assist new engineers with FastAPI API usage questions grounded in the project corpus
- Not a general-purpose coding assistant and not a substitute for official FastAPI docs

## Training data (high level)
Synthetic + cleaned pairs derived from:
- API reference / code contracts
- Wiki / architecture docs
- Community Q&A style chunks
- Bug-history summaries
- Explicit negative / abstention examples

See `Fine-tuning/` for dataset builders. Training artifacts are research code; production inference uses the saved adapter weights.

## Evaluation
Run:
```bash
python -m evaluation.runner --retrieval-only
python -m evaluation.runner --base-url http://127.0.0.1:8000
```
Metrics tracked: Recall@5, MRR, citation correctness, abstention correctness, latency, cache hit rate.

## Limitations
- Knowledge is limited to the ingested FastAPI onboarding corpus
- Conversational rewrite/decompose depend on Groq availability
- Generation quality depends on remote GPU (Colab/RunPod) uptime and cold starts
- BM25 + dense hybrid retrieval can still miss rare entities without exact name matches
- Not evaluated for unsafe code generation or security advice

## Ethical considerations
- Do not put secrets into prompts or logs
- Answers may be incomplete; treat as onboarding assistance, not production authority

## Citation / reproduction
Adapter weights are stored outside git (`model/`). Mount or copy `adapter_model.safetensors` onto the GPU host before serving.
