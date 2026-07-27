"""
Colab inference server for onboarding LoRA v2.

Same API as deploy/runpod/serve.py — POST /generate, GET /health.
Set LORA_PATH before starting uvicorn.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LORA_PATH = os.environ.get(
    "LORA_PATH",
    "/content/drive/MyDrive/onboarding_rag/lora_output/onboarding_lora_v2",
)
# Load this base (then attach LoRA). Avoids Unsloth remapping to the
# pre-quantized *-bnb-4bit hub id, which often fails with OSError on weights.
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Llama-3.1-8B-Instruct")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))

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


def _load_model() -> None:
    global _model, _tokenizer
    if not os.path.isdir(LORA_PATH):
        raise FileNotFoundError(
            f"LoRA directory not found: {LORA_PATH}. "
            "Update LORA_PATH in the notebook config cell."
        )
    weights = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
    )
    if not any(os.path.isfile(os.path.join(LORA_PATH, name)) for name in weights):
        raise FileNotFoundError(
            f"No adapter weights in {LORA_PATH}. "
            "Ensure adapter_model.safetensors is on Google Drive."
        )

    from peft import PeftModel
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    print(f"Loading base model: {BASE_MODEL}")
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Attaching LoRA from: {LORA_PATH}")
    _model = PeftModel.from_pretrained(_model, LORA_PATH)
    _tokenizer = get_chat_template(_tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(_model)
    print("Model ready.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Onboarding RAG Inference (Colab)", lifespan=lifespan)


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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _model is not None else "loading",
        "lora_path": LORA_PATH,
        "model_loaded": _model is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import torch

    messages = [m.model_dump() for m in req.messages]
    inputs = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = _model.generate(
        input_ids=inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        use_cache=True,
    )
    decoded = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = _extract_assistant_reply(decoded)
    return GenerateResponse(answer=answer)
