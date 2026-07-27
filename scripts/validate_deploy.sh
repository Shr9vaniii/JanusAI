#!/usr/bin/env bash
# Pre-interview validation against a public or local API base URL.
# Usage: ./scripts/validate_deploy.sh https://your-app.example.com

set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"
BASE="${BASE%/}"

echo "== presentation UI (/) =="
if curl -fsS "$BASE/" | grep -Eq "JanusAI|root"; then
  echo "UI OK"
else
  echo "WARN: GET / did not look like the Vite SPA — run: cd web && npm run build" >&2
fi

echo "== health =="
curl -fsS "$BASE/health" | python -m json.tool

echo "== session =="
SID=$(curl -fsS -X POST "$BASE/sessions" | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "session_id=$SID"

echo "== ask (grounded) =="
curl -fsS -X POST "$BASE/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What arguments does HTTPException take?\",\"session_id\":\"$SID\"}" \
  | python -m json.tool | head -n 40

echo "== six demo scenarios (API) =="
run_ask() {
  local name="$1" q="$2" new="$3"
  if [[ "$new" == "1" ]]; then
    SID=$(curl -fsS -X POST "$BASE/sessions" | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
  fi
  curl -fsS -X POST "$BASE/ask" \
    -H "Content-Type: application/json" \
    -d "{\"question\":$(python -c "import json,sys; print(json.dumps(sys.argv[1]))" "$q"),\"session_id\":\"$SID\"}" \
    | python -c "import sys,json; r=json.load(sys.stdin); print(f\"  $name: cache_hit={r.get('cache_hit')} intent={r.get('intent')} chunks={r.get('num_chunks')} status={r.get('model_status')}\")"
}

run_ask grounded "What arguments does HTTPException take?" 0
run_ask abstain "How do I configure Redis connection pooling in FastAPI?" 1
run_ask followup "and what are its attributes?" 0
run_ask topic "How do I use UploadFile?" 0
run_ask multi "What args does HTTPException take and how do I use UploadFile?" 1
run_ask cache "What arguments does HTTPException take?" 0

echo "== eval runner =="
python -m evaluation.runner --base-url "$BASE" --output evaluation/results/deploy_latest.json

echo "OK: validated $BASE"
echo "Reminder: record docs/DEMO_SCRIPT.md walkthrough as interview fallback."
