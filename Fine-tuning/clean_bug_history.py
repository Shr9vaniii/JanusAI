"""Clean enterprise_data/bug_history.jsonl for fine-tuning."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Fine-tuning"))

from build_jsonl import SYSTEM_PROMPT  # noqa: E402

INPUT_PATH = ROOT / "enterprise_data" / "bug_history.jsonl"
PRE_MERGE_BACKUP = ROOT / "enterprise_data" / "bug_history.jsonl.pre_merge.bak"
BUG_LABEL = "bug report"


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip())


def extract_question(user: str) -> str:
    if "Question:" in user:
        return user.split("Question:", 1)[1].strip()
    return user.strip()


def extract_source(user: str) -> str:
    m = re.search(r"Source: ([^\n]+)", user)
    return m.group(1).strip() if m else ""


def extract_subtype(user: str) -> str:
    m = re.search(r"Subtype: ([^\n]+)", user)
    return m.group(1).strip().lower() if m else ""


def normalize_answer(text: str) -> str:
    if "**Direct Answer:**" not in text:
        replacements = [
            (r"(?m)^Direct Answer:\s*", "**Direct Answer:**\n"),
            (r"(?m)^Details:\s*", "**Details:**\n"),
            (r"(?m)^Source:\s*", "**Source:**\n"),
            (r"(?m)^What I found:\s*", "**What I found:**\n"),
            (r"(?m)^Recommendation:\s*", "**Recommendation:**\n"),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
    text = re.sub(
        r"(?m)^\*\*Source:\*\*\s*\nbug_history\s+—",
        "**Source:**\nbug report —",
        text,
    )
    text = re.sub(
        r"(?m)^\*\*Source:\*\*\s*\nbug_history —",
        "**Source:**\nbug report —",
        text,
    )
    if "**Source:**" in text and "bug report —" not in text:
        text = text.replace("bug_history —", "bug report —")
    return text.strip()


def normalize_user(user: str) -> str:
    user = user.replace("Context (code contract):", f"Context ({BUG_LABEL}):")
    return user


def row_score(ex: dict) -> tuple:
    user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    ans = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
    subtype = extract_subtype(user)
    subtype_score = 1 if subtype == "solution" else 0
    return (subtype_score, len(ans))


def try_parse_line(line: str, line_no: int) -> dict | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass

    # Repair common failure: unescaped " inside JSON string values.
    # Only used for the handful of malformed export lines.
    if line_no in (23, 33):
        fixed = line
        if line_no == 23:
            fixed = fixed.replace(
                'generic type arguments as "subtypes." Is',
                'generic type arguments as \\"subtypes.\\" Is',
            )
            fixed = fixed.replace(
                'use the term "type parameters" in the documentation',
                'use the term \\"type parameters\\" in the documentation',
            )
            fixed = fixed.replace(
                'The term "subtype" was incorrectly',
                'The term \\"subtype\\" was incorrectly',
            )
            fixed = fixed.replace(
                'replaced with "type parameters"',
                'replaced with \\"type parameters\\"',
            )
        if line_no == 33:
            fixed = fixed.replace(
                'Depends scope="function" cleanup timing',
                'Depends scope=\\"function\\" cleanup timing',
            )
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            print(f"  Could not repair line {line_no}: {e}")
            return None
    return None


def clean_example(ex: dict) -> dict:
    for msg in ex["messages"]:
        if msg["role"] == "system":
            msg["content"] = SYSTEM_PROMPT
        elif msg["role"] == "user":
            msg["content"] = normalize_user(msg["content"])
        elif msg["role"] == "assistant":
            msg["content"] = normalize_answer(msg["content"])
    return ex


def main():
    raw_lines = [
        l for l in INPUT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()
    ]

    parsed = []
    skipped = 0
    repaired = 0
    for i, line in enumerate(raw_lines, 1):
        ex = try_parse_line(line, i)
        if ex is None:
            skipped += 1
            continue
        if i in (23, 33):
            repaired += 1
        parsed.append(clean_example(ex))

    # 1) exact question dedupe
    seen_q = set()
    after_q = []
    dup_q = 0
    for ex in parsed:
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        q = norm_q(extract_question(user))
        if q in seen_q:
            dup_q += 1
            continue
        seen_q.add(q)
        after_q.append(ex)

    # 2) one row per GitHub issue (keep best)
    by_source: dict[str, dict] = {}
    dup_src = 0
    for ex in after_q:
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        src = extract_source(user)
        if not src:
            key = f"__no_source__{len(by_source)}"
            by_source[key] = ex
            continue
        if src not in by_source:
            by_source[src] = ex
        else:
            dup_src += 1
            if row_score(ex) > row_score(by_source[src]):
                by_source[src] = ex

    final = list(by_source.values())

    # stable order: preserve original ordering by first appearance
    order = {}
    for idx, ex in enumerate(after_q):
        user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        src = extract_source(user) or f"__idx_{idx}"
        if src not in order:
            order[src] = idx
    final.sort(key=lambda ex: order.get(
        extract_source(next(m["content"] for m in ex["messages"] if m["role"] == "user")),
        9999,
    ))

    PRE_MERGE_BACKUP.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    out = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in final) + "\n"
    INPUT_PATH.write_text(out, encoding="utf-8")

    print(f"Input lines:        {len(raw_lines)}")
    print(f"Repaired JSON:      {repaired}")
    print(f"Skipped (bad JSON): {skipped}")
    print(f"After question dedupe: {len(after_q)} (removed {dup_q})")
    print(f"After source dedupe:   {len(final)} (removed {dup_src})")
    print(f"Pre-merge backup: {PRE_MERGE_BACKUP}")
    print(f"Wrote:  {INPUT_PATH}")


if __name__ == "__main__":
    main()
