"""
Process exported chunk JSONL files with Groq — one type at a time.

Done manually (skip): code_contracts, bug_history
Remaining:          wikis_arch_docs, community_qa, reference

Usage:
  python process_chunks_groq.py --status
  python process_chunks_groq.py --type wikis_arch_docs
  python process_chunks_groq.py --type wikis_arch_docs --dry-run
  python process_chunks_groq.py --type wikis_arch_docs --audit
  python process_chunks_groq.py --type wikis_arch_docs --reset

Each type writes to:
  generated/{type}.jsonl
  generated/checkpoint_{type}.json

After a run finishes, review the audit summary, then start the NEXT type yourself.
The script never auto-starts the following file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Load .env from repo root
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
    TARGET_COUNTS,
    TYPE_SOURCE_LABELS,
    build_training_context,
    format_as_jsonl,
    generate_answer,
    generate_question,
    stable_chunk_id,
    validate_positive_example,
)

CHUNKS_DIR = Path(__file__).parent / "chunks"
OUTPUT_DIR = Path(__file__).parent / "generated"

DONE_TYPES = frozenset({"code_contracts", "bug_history"})
REMAINING_ORDER = ["wikis_arch_docs", "community_qa", "reference"]


def chunks_path(chunk_type: str) -> Path:
    return CHUNKS_DIR / f"chunks_{chunk_type}.jsonl"


def output_path(chunk_type: str) -> Path:
    return OUTPUT_DIR / f"{chunk_type}.jsonl"


def checkpoint_path(chunk_type: str) -> Path:
    return OUTPUT_DIR / f"checkpoint_{chunk_type}.json"


def load_chunks(chunk_type: str) -> list[dict]:
    path = chunks_path(chunk_type)
    if not path.exists():
        raise FileNotFoundError(f"Missing chunk file: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append({"content": row["content"], "metadata": row.get("metadata", {})})
    return rows


def load_checkpoint(chunk_type: str) -> dict:
    path = checkpoint_path(chunk_type)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "completed": [],
        "failed": [],
        "rejected": {},
        "seen_questions": [],
        "accepted": 0,
        "target": TARGET_COUNTS[chunk_type],
    }


def save_checkpoint(chunk_type: str, cp: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path(chunk_type).write_text(
        json.dumps(cp, indent=2),
        encoding="utf-8",
    )


def append_example(chunk_type: str, example: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in example.items() if k != "_meta"}
    with open(output_path(chunk_type), "a", encoding="utf-8") as f:
        f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def extract_question_from_example(ex: dict) -> str:
    user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    return user.split("Question:", 1)[1].strip() if "Question:" in user else ""


def extract_type_from_example(ex: dict) -> str:
    user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    m = re.search(r"Context \(([^)]+)\):", user)
    return m.group(1) if m else "unknown"


def audit_output(chunk_type: str) -> dict:
    path = output_path(chunk_type)
    if not path.exists():
        return {"exists": False, "total": 0}

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = []
    bad_json = 0
    missing_sections = 0
    generic = 0
    wrong_ctx = 0

    for ex in rows:
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        ans = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        q = extract_question_from_example(ex)
        questions.append(q)
        if not all(s in ans for s in ("**Direct Answer:**", "**Details:**", "**Source:**")):
            missing_sections += 1
        if "specific issue and the change made" in ans.lower():
            generic += 1
        label = TYPE_SOURCE_LABELS.get(chunk_type, chunk_type)
        if f"Context ({label}):" not in user:
            wrong_ctx += 1

    exact_dups = sum(c - 1 for c in Counter(questions).values() if c > 1)
    return {
        "exists": True,
        "total": len(rows),
        "unique_questions": len(set(questions)),
        "exact_dup_rows": exact_dups,
        "missing_sections": missing_sections,
        "generic_answers": generic,
        "wrong_context_label": wrong_ctx,
        "target": TARGET_COUNTS.get(chunk_type, 0),
    }


def print_audit_report(chunk_type: str) -> None:
    stats = audit_output(chunk_type)
    print(f"\n{'=' * 55}")
    print(f"AUDIT: {chunk_type}")
    print(f"{'=' * 55}")
    if not stats["exists"]:
        print("  No output file yet.")
        return
    print(f"  Rows:              {stats['total']} / {stats['target']} target")
    print(f"  Unique questions:  {stats['unique_questions']}")
    print(f"  Exact dup rows:    {stats['exact_dup_rows']}")
    print(f"  Missing sections:  {stats['missing_sections']}")
    print(f"  Generic answers:   {stats['generic_answers']}")
    print(f"  Wrong ctx label:   {stats['wrong_context_label']}")
    ok = (
        stats["exact_dup_rows"] == 0
        and stats["missing_sections"] == 0
        and stats["generic_answers"] == 0
        and stats["wrong_context_label"] == 0
    )
    print(f"  Quality:           {'OK' if ok else 'NEEDS REVIEW'}")
    print(f"  File:              {output_path(chunk_type)}")


def print_status() -> None:
    print("\nChunk file status")
    print("-" * 55)
    for chunk_type in list(DONE_TYPES) + REMAINING_ORDER:
        cp = chunks_path(chunk_type)
        n_chunks = (
            len([l for l in cp.read_text(encoding="utf-8").splitlines() if l.strip()])
            if cp.exists()
            else 0
        )
        stats = audit_output(chunk_type)
        done_tag = "DONE (manual)" if chunk_type in DONE_TYPES else ""
        progress = (
            f"{stats['total']}/{stats['target']}"
            if stats["exists"]
            else f"0/{TARGET_COUNTS.get(chunk_type, 0)}"
        )
        print(
            f"  {chunk_type:18} chunks={n_chunks:4}  generated={progress:8}  {done_tag}"
        )
    print("\nRecommended order:")
    for i, t in enumerate(REMAINING_ORDER, 1):
        print(f"  {i}. python process_chunks_groq.py --type {t}")


def process_type(
    chunk_type: str,
    *,
    target: int | None = None,
    dry_run: bool = False,
    reset: bool = False,
) -> None:
    if chunk_type in DONE_TYPES:
        print(f"'{chunk_type}' is already handled separately. Pick a remaining type:")
        for t in REMAINING_ORDER:
            print(f"  python process_chunks_groq.py --type {t}")
        return

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set (check .env)")

    chunks = load_chunks(chunk_type)
    target = target or TARGET_COUNTS[chunk_type]
    if dry_run:
        target = min(3, target)

    if reset:
        for path in (output_path(chunk_type), checkpoint_path(chunk_type)):
            if path.exists():
                path.unlink()
                print(f"Removed {path}")

    cp = load_checkpoint(chunk_type)
    cp["target"] = target
    completed = set(cp.get("completed", []))
    rejected = defaultdict(int, cp.get("rejected", {}))
    seen_questions = list(cp.get("seen_questions", []))
    accepted = cp.get("accepted", 0)

    print(f"\nProcessing: {chunk_type}")
    print(f"  Chunks available: {len(chunks)}")
    print(f"  Target:           {target}")
    print(f"  Already accepted: {accepted}")
    print(f"  Output:           {output_path(chunk_type)}")
    print()

    try:
        for i, chunk in enumerate(chunks):
            if accepted >= target:
                break

            content = chunk["content"]
            metadata = chunk["metadata"]
            chunk_id = stable_chunk_id(
                chunk_type,
                content,
                metadata.get("source", ""),
            )
            if chunk_id in completed:
                continue

            try:
                full_context = build_training_context(content, metadata, chunk_type)
                question = generate_question(content, chunk_type, metadata)
                answer = generate_answer(question, content, chunk_type, metadata)

                ok, reason = validate_positive_example(
                    question,
                    answer,
                    full_context,
                    chunk_type,
                    seen_questions,
                    metadata,
                )
                if not ok:
                    print(f"  [{i + 1}] FILTERED — {reason}")
                    rejected[reason] += 1
                    continue

                seen_questions.append(question)
                example = format_as_jsonl(
                    question=question,
                    context=content,
                    answer=answer,
                    chunk_type=chunk_type,
                    metadata=metadata,
                    example_id=chunk_id,
                )
                append_example(chunk_type, example)
                completed.add(chunk_id)
                accepted += 1
                print(f"  [{accepted}/{target}] Q: {question[:75]}...")

                cp.update({
                    "completed": list(completed),
                    "failed": cp.get("failed", []),
                    "rejected": dict(rejected),
                    "seen_questions": seen_questions,
                    "accepted": accepted,
                })
                if accepted % 5 == 0:
                    save_checkpoint(chunk_type, cp)

                time.sleep(0.3)

            except KeyboardInterrupt:
                print("\n\nInterrupted — saving checkpoint...")
                cp.update({
                    "completed": list(completed),
                    "rejected": dict(rejected),
                    "seen_questions": seen_questions,
                    "accepted": accepted,
                })
                save_checkpoint(chunk_type, cp)
                print_audit_report(chunk_type)
                print_next_step(chunk_type, interrupted=True)
                return

            except Exception as e:
                print(f"  [{i + 1}] ERROR: {e}")
                failed = list(cp.get("failed", []))
                failed.append(chunk_id)
                cp["failed"] = failed
                continue

    finally:
        cp.update({
            "completed": list(completed),
            "rejected": dict(rejected),
            "seen_questions": seen_questions,
            "accepted": accepted,
        })
        save_checkpoint(chunk_type, cp)

    print_audit_report(chunk_type)
    print_next_step(chunk_type, interrupted=False)


def print_next_step(chunk_type: str, *, interrupted: bool) -> None:
    if chunk_type not in REMAINING_ORDER:
        return

    idx = REMAINING_ORDER.index(chunk_type)
    stats = audit_output(chunk_type)
    if interrupted:
        print("\nResume this type:")
        print(f"  python process_chunks_groq.py --type {chunk_type}")
        return

    if stats["total"] < stats["target"]:
        print("\nTarget not reached — run again to continue:")
        print(f"  python process_chunks_groq.py --type {chunk_type}")
        return

    if idx + 1 < len(REMAINING_ORDER):
        nxt = REMAINING_ORDER[idx + 1]
        print("\n" + "=" * 55)
        print("STOP — review the audit above before continuing.")
        print("When ready, start the NEXT file:")
        print(f"  python process_chunks_groq.py --type {nxt}")
        print("=" * 55)
    else:
        print("\n" + "=" * 55)
        print("All remaining chunk types processed.")
        print("Merge outputs with:")
        print("  python merge_datasets.py")
        print("=" * 55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Q&A JSONL from exported chunk files (one type per run)",
    )
    parser.add_argument(
        "--type",
        choices=REMAINING_ORDER,
        help="Chunk type to process (one file per invocation)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show progress for all chunk types",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Audit output for --type without generating",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate only 3 examples (smoke test)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear output + checkpoint for --type before running",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Override target count (default from TARGET_COUNTS)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.status:
        print_status()
        return

    if not args.type:
        print_status()
        print("\nPick a type to process, e.g.:")
        print("  python process_chunks_groq.py --type wikis_arch_docs")
        return

    if args.audit:
        print_audit_report(args.type)
        return

    process_type(
        args.type,
        target=args.target,
        dry_run=args.dry_run,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
