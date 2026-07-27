"""
RunPod inference server for onboarding LoRA v2.

Exposes POST /generate and GET /health on port 8000 (configurable via PORT).
Local rag_engine calls this with --backend remote.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

LORA_PATH = os.environ.get("LORA_PATH", "/workspace/onboarding_lora_v2")
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Llama-3.1-8B-Instruct")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
INFERENCE_API_KEY = os.environ.get("INFERENCE_API_KEY", "").strip()
SERVE_REVISION = "base-plus-peft-v2"

_model = None
_tokenizer = None


def _extract_assistant_reply(decoded: str) -> str:
    marker = "assistant"
    if marker in decoded.lower():
        parts = decoded.split("assistant")
        tail = parts[-1].strip()
        if tail.startswith(":"):
            tail = tail[1:].strip()
        if tail:
            return tail
    if "**Direct Answer:**" in decoded:
        return decoded[decoded.index("**Direct Answer:**") :]
    return decoded.strip()


def _load_model_transformers() -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"[{SERVE_REVISION}] Loading base via transformers: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"[{SERVE_REVISION}] Attaching LoRA from: {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tokenizer


def _load_model_unsloth() -> tuple[Any, Any]:
    from peft import PeftModel
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    print(f"[{SERVE_REVISION}] Loading base via Unsloth (exact name): {BASE_MODEL}")
    kwargs: dict[str, Any] = {
        "model_name": BASE_MODEL,
        "max_seq_length": MAX_SEQ_LENGTH,
        "dtype": None,
        "load_in_4bit": True,
    }
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            **kwargs,
            use_exact_model_name=True,
        )
    except TypeError:
        model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)

    print(f"[{SERVE_REVISION}] Attaching LoRA from: {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _load_model() -> None:
    global _model, _tokenizer
    if not os.path.isdir(LORA_PATH):
        raise FileNotFoundError(
            f"LoRA directory not found: {LORA_PATH}. "
            "Upload onboarding_lora_v2 (with adapter_model.safetensors) to the pod."
        )
    weights = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
    )
    if not any(os.path.isfile(os.path.join(LORA_PATH, name)) for name in weights):
        raise FileNotFoundError(
            f"No adapter weights in {LORA_PATH}. "
            "Copy adapter_model.safetensors from Google Drive."
        )

    try:
        _model, _tokenizer = _load_model_transformers()
    except Exception as primary_exc:
        print(f"[{SERVE_REVISION}] transformers load failed: {primary_exc}")
        print(f"[{SERVE_REVISION}] Falling back to Unsloth…")
        _model, _tokenizer = _load_model_unsloth()
    print(f"[{SERVE_REVISION}] Model ready.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Onboarding RAG Inference", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    answer: str
    model: str = "onboarding_lora_v2"


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """If INFERENCE_API_KEY is set on the pod, require Bearer auth."""
    if not INFERENCE_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != INFERENCE_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _model is not None else "loading",
        "lora_path": LORA_PATH,
        "base_model": BASE_MODEL,
        "serve_revision": SERVE_REVISION,
        "model_loaded": _model is not None,
        "auth_required": bool(INFERENCE_API_KEY),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(
    req: GenerateRequest,
    _: None = Depends(require_api_key),
) -> GenerateResponse:
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import torch

    messages = [m.model_dump() for m in req.messages]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        encoded = _tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            input_ids = encoded.to(device)
            attention_mask = None

        gen_kwargs: dict = {
            "input_ids": input_ids,
            "max_new_tokens": req.max_tokens,
            "use_cache": True,
            "pad_token_id": _tokenizer.pad_token_id or _tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if req.temperature and req.temperature > 0:
            gen_kwargs["temperature"] = req.temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        outputs = _model.generate(**gen_kwargs)
        decoded = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = _extract_assistant_reply(decoded)
        return GenerateResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
