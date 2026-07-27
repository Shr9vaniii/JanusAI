import json
from pathlib import Path

ABSTAIN = "I don't have enough information in the provided context"


def count(path: Path) -> tuple[int, int]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    abstain = 0
    for ex in rows:
        ans = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        direct = ans.split("**Details:**")[0] if "**Details:**" in ans else ans[:300]
        if ABSTAIN.lower() in direct.lower():
            abstain += 1
    return len(rows), abstain


root = Path(__file__).resolve().parent
for name in ("train.jsonl", "val.jsonl"):
    p = root / "jsonl" / name
    total, abst = count(p)
    print(f"{name}: {total} total, {abst} abstention answers")
