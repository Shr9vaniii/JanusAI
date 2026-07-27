"""Audit merged train/val JSONL quality."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "Fine-tuning" / "jsonl" / "train.jsonl"
VAL = ROOT / "Fine-tuning" / "jsonl" / "val.jsonl"


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def extract(ex):
    user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    ans = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
    q = user.split("Question:", 1)[1].strip() if "Question:" in user else ""
    ctx = re.search(r"Context \(([^)]+)\):", user)
    src = re.search(r"Source: ([^\n]+)", user)
    return {
        "q": q,
        "ans": ans,
        "type": ctx.group(1) if ctx else "unknown",
        "source": src.group(1).strip() if src else "",
    }


def norm_q(q):
    q = q.lower().strip()
    q = re.sub(r"`[^`]+`", "<name>", q)
    return re.sub(r"\s+", " ", q)


def jaccard(a, b):
    wa = set(norm_q(a).split())
    wb = set(norm_q(b).split())
    if not wa or not wb:
        return 0
    return len(wa & wb) / len(wa | wb)


def audit_file(name, rows):
    items = [extract(ex) for ex in rows]
    qs = [i["q"] for i in items]
    exact_dups = sum(c - 1 for c in Counter(qs).values() if c > 1)
    missing = sum(
        1 for i in items
        if not all(s in i["ans"] for s in ("**Direct Answer:**", "**Details:**", "**Source:**"))
    )
    generic = sum(1 for i in items if "specific issue and the change made" in i["ans"].lower())
    no_source = sum(1 for i in items if not i["source"])
    speculative = sum(
        1 for i in items
        if any(p in i["ans"].lower() for p in ("typically", "usually", "might ", "may ", "probably"))
    )
    types = Counter(i["type"] for i in items)
    return {
        "name": name,
        "total": len(rows),
        "types": dict(types),
        "exact_dup_rows": exact_dups,
        "missing_sections": missing,
        "generic_answers": generic,
        "missing_source": no_source,
        "speculative": speculative,
        "avg_ans_len": round(sum(len(i["ans"]) for i in items) / max(len(items), 1)),
    }


def main():
    train = load(TRAIN)
    val = load(VAL)
    train_qs = {norm_q(extract(ex)["q"]) for ex in train}
    val_qs = {norm_q(extract(ex)["q"]) for ex in val}
    overlap = train_qs & val_qs

    print("=" * 60)
    print("MERGED DATASET QUALITY REPORT")
    print("=" * 60)
    for stats in (audit_file("train", train), audit_file("val", val)):
        print(f"\n--- {stats['name'].upper()} ({stats['total']} rows) ---")
        print(f"  Types:             {stats['types']}")
        print(f"  Exact dup rows:    {stats['exact_dup_rows']}")
        print(f"  Missing sections:  {stats['missing_sections']}")
        print(f"  Generic answers:   {stats['generic_answers']}")
        print(f"  Missing source:    {stats['missing_source']}")
        print(f"  Speculative:       {stats['speculative']}")
        print(f"  Avg answer len:    {stats['avg_ans_len']}")

    print(f"\n--- TRAIN/VAL LEAKAGE ---")
    print(f"  Question overlap:  {len(overlap)}")
    print(f"  Total unique Qs:   {len(train_qs | val_qs)}")
    print(f"  Combined rows:     {len(train) + len(val)}")

    train_items = [extract(ex) for ex in train]

    print(f"\n--- PER-TYPE TRAIN COUNTS ---")
    for t, c in sorted(Counter(i["type"] for i in train_items).items()):
        print(f"  {t}: {c}")

    print(f"\n--- NEAR-DUP QUESTION PAIRS (train, jaccard>=0.72) ---")
    near = 0
    for t in set(i["type"] for i in train_items):
        subset = [i["q"] for i in train_items if i["type"] == t]
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                if jaccard(subset[i], subset[j]) >= 0.72:
                    near += 1
    print(f"  {near} pairs (template overlap is normal for code contracts)")

    ok = (
        len(overlap) == 0
        and audit_file("train", train)["exact_dup_rows"] == 0
        and audit_file("val", val)["exact_dup_rows"] == 0
        and audit_file("train", train)["missing_sections"] == 0
        and audit_file("val", val)["missing_sections"] == 0
    )
    print(f"\n{'=' * 60}")
    print(f"VERDICT: {'GOOD for QLoRA' if ok else 'NEEDS REVIEW'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
