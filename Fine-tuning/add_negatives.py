"""Merge generated/negatives.jsonl into train.jsonl and val.jsonl."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "jsonl" / "train.jsonl"
VAL_PATH = ROOT / "jsonl" / "val.jsonl"
NEG_PATH = ROOT / "generated" / "negatives.jsonl"
BACKUP_DIR = ROOT / "jsonl" / "backup_pre_negatives"
VAL_RATIO = 0.17


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip())


def extract_question(user: str) -> str:
    if "Question:" in user:
        return user.split("Question:", 1)[1].strip()
    return user.strip()


def split_key(ex: dict) -> str:
    user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    return norm_q(extract_question(user))


def is_val_split(key: str) -> bool:
    bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1000
    return bucket >= int((1 - VAL_RATIO) * 1000)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def is_abstention(ex: dict) -> bool:
    ans = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
    direct = ans.split("**Details:**")[0] if "**Details:**" in ans else ans[:400]
    return "don't have enough information in the provided context" in direct.lower()


def main() -> None:
    if not NEG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {NEG_PATH} — run: python generate_negatives.py"
        )

    negatives = load_jsonl(NEG_PATH)
    train = load_jsonl(TRAIN_PATH)
    val = load_jsonl(VAL_PATH)

    existing_keys = {split_key(ex) for ex in train + val}
    added_train, added_val, skipped = [], [], 0

    for ex in negatives:
        key = split_key(ex)
        if key in existing_keys:
            skipped += 1
            continue
        existing_keys.add(key)
        if is_val_split(key):
            added_val.append(ex)
        else:
            added_train.append(ex)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src in (TRAIN_PATH, VAL_PATH):
        if src.exists():
            (BACKUP_DIR / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    train_out = train + added_train
    val_out = val + added_val

    def write(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(ex, ensure_ascii=False) for ex in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    write(TRAIN_PATH, train_out)
    write(VAL_PATH, val_out)

    def count_abst(rows: list[dict]) -> int:
        return sum(1 for ex in rows if is_abstention(ex))

    print(f"Negatives file:     {len(negatives)}")
    print(f"Added to train:     {len(added_train)}")
    print(f"Added to val:       {len(added_val)}")
    print(f"Skipped (duplicate): {skipped}")
    print(f"Train total:        {len(train_out)} ({count_abst(train_out)} abstention)")
    print(f"Val total:          {len(val_out)} ({count_abst(val_out)} abstention)")
    print(f"Backup:             {BACKUP_DIR}")
    print("\nRe-upload train.jsonl + val.jsonl to Colab and run a short retrain (~1 epoch).")


if __name__ == "__main__":
    main()
