#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LORA_PATH="${LORA_PATH:-/workspace/onboarding_lora_v2}"
export PORT="${PORT:-8000}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"

echo "LoRA path: ${LORA_PATH}"
echo "Listening on 0.0.0.0:${PORT}"

python -m uvicorn serve:app --host 0.0.0.0 --port "${PORT}"
