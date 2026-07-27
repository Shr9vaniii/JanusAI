"""
Generate abstention (negative) fine-tuning examples.

Questions are related to a chunk but NOT fully answerable from that context.
Answers teach the model to say "I don't have enough information..." instead of hallucinating.

Usage:
  python generate_negatives.py --status
  python generate_negatives.py              # target 150
  python generate_negatives.py --count 50   # smaller batch for testing
  python generate_negatives.py --reset
  python generate_negatives.py --dry-run    # 5 examples only

Output:
  generated/negatives.jsonl
  generated/checkpoint_negatives.json

Then merge into train/val:
  python add_negatives.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, str(Path(__file__).parent))

from build_jsonl import (  # noqa: E402
    NEGATIVE_COUNT,
    build_training_context,
    format_as_jsonl,
    generate_negative_example,
    resolve_chunk_type,
    stable_chunk_id,
    validate_negative_example,
)

CHUNKS_DIR = Path(__file__).parent / "chunks"
OUTPUT_DIR = Path(__file__).parent / "generated"
OUTPUT_PATH = OUTPUT_DIR / "negatives.jsonl"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_negatives.json"

CHUNK_TYPES = [
    "code_contracts",
    "wikis_arch_docs",
    "bug_history",
    "community_qa",
    "reference",
]


def load_all_chunks() -> list[dict]:
    rows: list[dict] = []
    for chunk_type in CHUNK_TYPES:
        path = CHUNKS_DIR / f"chunks_{chunk_type}.jsonl"
        if not path.exists():
            print(f"  skip missing: {path.name}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_chunk_type"] = chunk_type
            rows.append(row)
    return rows


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {"completed": [], "accepted": 0, "rejected": {}, "target": NEGATIVE_COUNT}


def save_checkpoint(cp: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(cp, indent=2), encoding="utf-8")


def append_example(example: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")


def status() -> None:
    cp = load_checkpoint()
    n_out = (
        sum(1 for _ in OUTPUT_PATH.open(encoding="utf-8"))
        if OUTPUT_PATH.exists()
        else 0
    )
    print(f"Target:     {cp.get('target', NEGATIVE_COUNT)}")
    print(f"Accepted:   {cp.get('accepted', 0)}")
    print(f"On disk:    {n_out} rows in {OUTPUT_PATH.name}")
    print(f"Rejected:   {cp.get('rejected', {})}")
    print(f"Chunks dir: {CHUNKS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate abstention training examples")
    parser.add_argument("--count", type=int, default=NEGATIVE_COUNT)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        status()
        return

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set (check .env)")

    target = 5 if args.dry_run else args.count

    if args.reset:
        for path in (OUTPUT_PATH, CHECKPOINT_PATH):
            if path.exists():
                path.unlink()
                print(f"Removed {path}")

    cp = load_checkpoint()
    cp["target"] = target
    completed = set(cp.get("completed", []))
    rejected = defaultdict(int, cp.get("rejected", {}))
    accepted = cp.get("accepted", 0)

    chunks = load_all_chunks()
    if not chunks:
        raise FileNotFoundError(f"No chunk files found under {CHUNKS_DIR}")

    random.shuffle(chunks)
    per_type = max(1, target // len(CHUNK_TYPES))
    by_type: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_type[chunk["_chunk_type"]].append(chunk)

    pool: list[dict] = []
    for chunk_type in CHUNK_TYPES:
        sample = by_type.get(chunk_type, [])
        if sample:
            pool.extend(random.sample(sample, min(per_type * 3, len(sample))))
    random.shuffle(pool)

    print(f"\nGenerating {target} negative (abstention) examples")
    print(f"  Pool size: {len(pool)} chunks")
    print(f"  Output:    {OUTPUT_PATH}")
    print(f"  Already:   {accepted}\n")

    for i, chunk in enumerate(pool):
        if accepted >= target:
            break

        content = chunk["content"]
        metadata = chunk["metadata"]
        chunk_type = chunk["_chunk_type"]
        chunk_id = stable_chunk_id(
            f"negative_{chunk_type}",
            content,
            metadata.get("source", ""),
        )
        if chunk_id in completed:
            continue

        try:
            full_context = build_training_context(content, metadata, chunk_type)
            result = generate_negative_example(content, chunk_type, metadata)
            if not result:
                rejected["generation_failed"] += 1
                continue

            ok, reason = validate_negative_example(
                result["question"],
                result["answer"],
                full_context,
            )
            if not ok:
                print(f"  [{i + 1}] FILTERED — {reason}")
                rejected[reason] += 1
                completed.add(chunk_id)
                continue

            example = format_as_jsonl(
                question=result["question"],
                context=content,
                answer=result["answer"],
                chunk_type=chunk_type,
                metadata=metadata,
                example_id=chunk_id,
            )
            example["_meta"]["is_negative"] = True
            append_example(example)

            completed.add(chunk_id)
            accepted += 1
            cp["accepted"] = accepted
            cp["completed"] = list(completed)
            cp["rejected"] = dict(rejected)
            save_checkpoint(cp)

            print(f"  [{accepted}/{target}] Q: {result['question'][:72]}")

        except Exception as e:
            print(f"  [{i + 1}] ERROR: {e}")
            rejected["error"] += 1
            time.sleep(2)

    cp["accepted"] = accepted
    cp["rejected"] = dict(rejected)
    save_checkpoint(cp)

    print(f"\nDone: {accepted}/{target} negatives → {OUTPUT_PATH}")
    print(f"Rejected: {dict(rejected)}")
    print("\nNext: python add_negatives.py")


if __name__ == "__main__":
    main()
