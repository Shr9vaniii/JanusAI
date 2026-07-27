"""Merge bug_history.jsonl into train/val with stratified 17% validation split."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "Fine-tuning" / "jsonl" / "train.jsonl"
VAL_PATH = ROOT / "Fine-tuning" / "jsonl" / "val.jsonl"
BUG_PATH = ROOT / "enterprise_data" / "bug_history.jsonl"
GENERATED_DIR = ROOT / "Fine-tuning" / "generated"
BACKUP_DIR = ROOT / "Fine-tuning" / "jsonl" / "backup_pre_merge"

VAL_RATIO = 0.17  # 17% validation


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip())


def extract_question(user: str) -> str:
    if "Question:" in user:
        return user.split("Question:", 1)[1].strip()
    return user.strip()


def extract_type(user: str) -> str:
    m = re.search(r"Context \(([^)]+)\):", user)
    return m.group(1).strip() if m else "unknown"


def example_key(ex: dict) -> str:
    user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    q = norm_q(extract_question(user))
    t = extract_type(user)
    src = ""
    sm = re.search(r"Source: ([^\n]+)", user)
    if sm:
        src = sm.group(1).strip()
    return f"{t}|{src}|{q}"


def split_key(ex: dict) -> str:
    """Question-only key so variants of the same question stay in one split."""
    user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    return norm_q(extract_question(user))


def is_val_split(key: str) -> bool:
    bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1000
    return bucket >= int((1 - VAL_RATIO) * 1000)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def row_score(ex: dict) -> tuple:
    ans = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
    user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    has_bold = 1 if "**Direct Answer:**" in ans else 0
    return (has_bold, len(ans))


def main():
    sources = [
        ("train", TRAIN_PATH),
        ("val", VAL_PATH),
        ("bug_history", BUG_PATH),
    ]
    for path in sorted(GENERATED_DIR.glob("*.jsonl")):
        if path.name.startswith("checkpoint"):
            continue
        sources.append((path.stem, path))

    merged: dict[str, dict] = {}
    dupes = 0
    for label, path in sources:
        for ex in load_jsonl(path):
            key = example_key(ex)
            if key not in merged:
                merged[key] = ex
            else:
                dupes += 1
                if row_score(ex) > row_score(merged[key]):
                    merged[key] = ex

    examples = list(merged.values())

    train_out, val_out = [], []
    for ex in examples:
        key = split_key(ex)
        if is_val_split(key):
            val_out.append(ex)
        else:
            train_out.append(ex)

    # Sort for reproducible file order
    def sort_key(ex):
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        return example_key(ex)

    train_out.sort(key=sort_key)
    val_out.sort(key=sort_key)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src_path in (TRAIN_PATH, VAL_PATH):
        if src_path.exists():
            dest = BACKUP_DIR / src_path.name
            dest.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")

    def write(path: Path, rows: list[dict]):
        path.write_text(
            "\n".join(json.dumps(ex, ensure_ascii=False) for ex in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

    write(TRAIN_PATH, train_out)
    write(VAL_PATH, val_out)

    def type_counts(rows):
        c = Counter()
        for ex in rows:
            user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            c[extract_type(user)] += 1
        return dict(c)

    total = len(examples)
    print(f"Merged unique examples: {total}")
    print(f"Duplicates removed:      {dupes}")
    print(f"Train: {len(train_out)} ({len(train_out)/total:.1%})")
    print(f"Val:   {len(val_out)} ({len(val_out)/total:.1%})")
    print(f"Train types: {type_counts(train_out)}")
    print(f"Val types:   {type_counts(val_out)}")
    print(f"Backup: {BACKUP_DIR}")
    print(f"Wrote:  {TRAIN_PATH}")
    print(f"        {VAL_PATH}")


if __name__ == "__main__":
    main()
