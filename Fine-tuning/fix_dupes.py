"""Remove exact duplicate questions from train and train/val leaks."""
import json
import re
from pathlib import Path

TRAIN = Path(__file__).parent / "jsonl" / "train.jsonl"
VAL = Path(__file__).parent / "jsonl" / "val.jsonl"


def norm_q(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"`[^`]+`", "<name>", q)
    return re.sub(r"\s+", " ", q)


def extract_q(ex: dict) -> str:
    user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    return user.split("Question:", 1)[1].strip()


def dedupe_rows(rows: list[dict]) -> tuple[list[dict], int]:
    seen = set()
    kept = []
    removed = 0
    for ex in rows:
        q = norm_q(extract_q(ex))
        if q in seen:
            removed += 1
            continue
        seen.add(q)
        kept.append(ex)
    return kept, removed


def main():
    train = [json.loads(l) for l in TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    val = [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines() if l.strip()]

    train, rm_train = dedupe_rows(train)
    val, rm_val = dedupe_rows(val)

    train_qs = {norm_q(extract_q(ex)) for ex in train}
    val_clean = []
    rm_leak = 0
    for ex in val:
        if norm_q(extract_q(ex)) in train_qs:
            rm_leak += 1
            continue
        val_clean.append(ex)
    val = val_clean

    TRAIN.write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in train) + "\n",
        encoding="utf-8",
    )
    VAL.write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in val) + "\n",
        encoding="utf-8",
    )
    print(f"Removed train exact dups: {rm_train}")
    print(f"Removed val exact dups:   {rm_val}")
    print(f"Removed val/train leaks:  {rm_leak}")
    print(f"Final train: {len(train)}, val: {len(val)}")


if __name__ == "__main__":
    main()
