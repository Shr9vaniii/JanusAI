"""
JSONL Dataset Generator for QLoRA Fine-tuning
===============================================
Generates 1200 high-quality Q&A training examples from ChromaDB chunks.

Pipeline per example:
  1. Sample chunk from ChromaDB (balanced across 5 types)
  2. Generate question using llama-3.3-70b (creative, natural)
  3. Generate structured answer using llama-3.1-8b-instant (fast, accurate)
  4. Format as chat JSONL for QLoRA

Also generates ~150 negative examples where the model learns to abstain
when context is insufficient.

Output:
  jsonl/train.jsonl       ← 1000 training examples
  jsonl/val.jsonl         ← 200 validation examples
  jsonl/checkpoint.json   ← resume state

Usage (local or Colab):
  python build_jsonl.py
  # Resumes automatically from checkpoint if interrupted
"""

import argparse
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
import os
from groq import Groq

# ── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_API_KEY =  os.environ.get("GROQ_API_KEY")
GROQ_API_KEY2 =  os.environ.get("GROQ_API_KEY2")

DB_PATH         = Path("../enterprise_data/chroma_db_v3")
OUTPUT_DIR      = Path("./jsonl")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"

QUESTION_MODEL = "llama-3.3-70b-versatile"   # better creativity
ANSWER_MODEL   = "llama-3.1-8b-instant"       # faster, accurate

# Target counts per type
TARGET_COUNTS = {
    "code_contracts":  250,
    "wikis_arch_docs": 250,
    "bug_history":     150,
    "community_qa":    200,
    "reference":       150,
}
NEGATIVE_COUNT = 150   # abstention examples
TOTAL_TARGET   = sum(TARGET_COUNTS.values())  # 1000 positive

TRAIN_RATIO    = 0.83  # 1000 train, 200 val
MAX_RETRIES    = 3
RETRY_DELAY    = 5

# Canonical labels used in answers and user context headers
TYPE_SOURCE_LABELS = {
    "code_contracts":  "code contract",
    "wikis_arch_docs": "documentation",
    "bug_history":     "bug report",
    "community_qa":    "community discussion",
    "reference":       "API reference",
}

DUPLICATE_THRESHOLDS = {
    "code_contracts": 0.65,
    "default":        0.72,
}

# Tokens that appear often in API reference signatures/parameters
REFERENCE_CONTEXT_TOKENS = {
    "depends", "default", "literal", "pathlike", "jsonresponse",
    "response", "incex", "enum", "any", "callable", "sequence",
    "websocket", "apirouter", "fastapi", "pydantic", "starlette",
}

ABSTENTION_PHRASE = "don't have enough information in the provided context"

# Metadata fields exposed to the model, per chunk type
METADATA_FIELDS = {
    "code_contracts":  ["source", "topic", "name", "kind", "subtype"],
    "wikis_arch_docs": ["source", "title", "header_path", "section",
                        "mentions_classes", "related_pages", "subtype"],
    "reference":       ["source", "qualified_name", "kind", "parent",
                        "section", "related_pages"],
    "bug_history":     ["source", "subtype"],
    "community_qa":    ["source", "subtype"],
}

METADATA_LABELS = {
    "source":           "Source",
    "topic":            "Topic",
    "name":             "Name",
    "kind":             "Kind",
    "subtype":          "Subtype",
    "title":            "Page",
    "header_path":      "Section",
    "section":          "Doc section",
    "mentions_classes": "Mentioned APIs",
    "related_pages":    "Related pages",
    "qualified_name":   "API",
    "parent":           "Parent",
}

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert onboarding assistant for a backend engineering team using FastAPI.

Your role is to help new engineers understand the codebase, APIs, bugs, and architectural decisions.

When answering:
- Base your answer STRICTLY on the provided context
- If the context is insufficient, say so explicitly
- Use this exact structure:

**Direct Answer:**
[One clear sentence answering the question directly]

**Details:**
[Technical explanation with specifics from the context]

**Source:**
[source type] — [exact Source path or URL from the metadata block when available]

If related pages are listed in the metadata and relevant, you may mention them in **Details** or **Recommendation**.
Do not invent file paths, URLs, or related pages that are not in the context.

If you cannot answer from the context:
**Direct Answer:**
I don't have enough information in the provided context to answer this accurately.

**What I found:**
[What the context does contain, if anything relevant]

**Recommendation:**
[Where the engineer should look instead]"""


# ── QUESTION TEMPLATES PER TYPE ───────────────────────────────────────────────
# Used as few-shot examples to guide the question generator

QUESTION_STYLE_GUIDE = {
    "code_contracts": """Generate a question a NEW backend engineer would ask when
encountering this code for the first time. Focus on:
- What does this function/class do?
- What arguments does it accept and what are their types?
- What does it return?
- When should I use this?
- How do I call this correctly?
Make it sound natural, like a real engineer asking in Slack.""",

    "wikis_arch_docs": """Generate a question a new engineer would ask while reading
FastAPI documentation. Focus on:
- How do I implement X?
- What is the purpose of Y?
- When should I use Z vs W?
- How does this work under the hood?
Make it practical and task-oriented.""",

    "bug_history": """Generate a question a new engineer would ask after hitting a bug
or unexpected behavior. Focus on:
- Why is X not working?
- Has anyone seen this error before?
- What causes this issue?
- How do I fix this?
Make it sound frustrated but specific, like a real bug report.""",

    "community_qa": """Generate a question a new engineer would ask in a community forum
or team chat. Focus on:
- Best practices for X
- Difference between A and B
- How to handle edge case Y
- What's the recommended way to do Z?
Make it conversational and specific.""",

    "reference": """Generate a question a new engineer would ask when looking up
API reference documentation. Focus on:
- What parameters does X accept?
- What does parameter Y do?
- What is the return type of Z?
- What's the default value of W?
Make it precise and technical.""",
}


# ── CONTEXT BUILDER ───────────────────────────────────────────────────────────

def format_metadata_header(metadata: dict, chunk_type: str) -> str:
    """Render selected metadata fields as a readable header for the model."""
    fields = METADATA_FIELDS.get(chunk_type, ["source", "subtype"])
    lines = []

    for key in fields:
        value = metadata.get(key, "")
        if not value or not str(value).strip():
            continue
        label = METADATA_LABELS.get(key, key.replace("_", " ").title())
        lines.append(f"{label}: {value}")

    if not lines:
        return ""

    return "--- Metadata ---\n" + "\n".join(lines)


def build_training_context(content: str, metadata: dict, chunk_type: str) -> str:
    """Combine metadata + chunk content — same structure used at train and inference time."""
    header = format_metadata_header(metadata, chunk_type)
    if header:
        return f"{header}\n\n--- Content ---\n{content}"
    return content


def resolve_chunk_type(metadata: dict, fallback: str = "wikis_arch_docs") -> str:
    """Map stored Chroma metadata to the training bucket / field schema."""
    if metadata.get("subtype") == "reference":
        return "reference"
    return metadata.get("type", fallback)


def source_citation(metadata: dict, chunk_type: str) -> str:
    """Build the expected Source line suffix from metadata."""
    label = TYPE_SOURCE_LABELS.get(chunk_type, "documentation")
    path = metadata.get("source", "").strip()
    if path:
        return f"{label} — `{path}`"
    return label


# ── GROQ CLIENT ───────────────────────────────────────────────────────────────

client1 = Groq(api_key=GROQ_API_KEY)
client2 = Groq(api_key=GROQ_API_KEY)


def call_groq(
    prompt: str,
    model: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 600,
    json_mode: bool = False
) -> str:
    """Call Groq API with retry logic."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = (
                client1.chat.completions.create(**kwargs)
                if model == QUESTION_MODEL
                else client2.chat.completions.create(**kwargs)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = RETRY_DELAY * attempt
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(RETRY_DELAY)

    raise RuntimeError(f"Groq failed after {MAX_RETRIES} attempts")


# ── QUESTION GENERATOR ────────────────────────────────────────────────────────

def generate_question(chunk_content: str, chunk_type: str, metadata: dict) -> str:
    """Generate a natural engineer question for this chunk."""
    style_guide = QUESTION_STYLE_GUIDE.get(chunk_type, QUESTION_STYLE_GUIDE["wikis_arch_docs"])
    full_context = build_training_context(chunk_content, metadata, chunk_type)

    prompt = f"""{style_guide}

Here is the technical content:
---
{full_context[:1800]}
---

Generate ONE specific question (1-2 sentences max).
Output ONLY the question, nothing else."""

    question = call_groq(
        prompt=prompt,
        model=QUESTION_MODEL,
        temperature=0.8,    # creative for questions
        max_tokens=100,
    )

    # Clean up
    question = question.strip().strip('"').strip("'")
    if not question.endswith("?"):
        question += "?"

    return question


# ── ANSWER GENERATOR ──────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    content: str,
    chunk_type: str,
    metadata: dict,
) -> str:
    """Generate a structured answer using the context."""
    full_context = build_training_context(content, metadata, chunk_type)
    source_line = source_citation(metadata, chunk_type)

    code_hint = ""
    if _context_has_code(content):
        code_hint = """
- The content includes code. In **Details**, include the most relevant excerpt
  inside a ```python block, copied exactly from the content.
- Do not write new code — only reuse code already shown in the content."""

    prompt = f"""{full_context[:2400]}

Engineer's Question: {question}

IMPORTANT:
- Use ONLY facts from the metadata and content above.
- Reference specific names, parameters, versions, or code from the content.
- In **Source**, use this exact citation: {source_line}
- If Related pages are listed in metadata and helpful, mention them in **Details**.
- Do not invent details, files, URLs, or recommendations not present above.{code_hint}

Use this exact structure:

**Direct Answer:**
[One clear sentence answering the question directly]

**Details:**
[Technical explanation with specifics from the context]

**Source:**
{source_line}"""

    answer = call_groq(
        prompt=prompt,
        model=ANSWER_MODEL,
        system=SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=500,
    )

    return answer


# ── NEGATIVE EXAMPLE GENERATOR ────────────────────────────────────────────────

def generate_negative_example(
    chunk_content: str,
    chunk_type: str,
    metadata: dict,
) -> dict | None:
    """
    Generate an example where the context is INSUFFICIENT to answer.
    The model learns to abstain rather than hallucinate.
    """
    full_context = build_training_context(chunk_content, metadata, chunk_type)
    related = metadata.get("related_pages", "").strip()
    related_hint = (
        f"\nIf recommending where to look next, prefer these related pages: {related}"
        if related else ""
    )

    prompt = f"""Here is a technical content chunk:
---
{full_context[:1000]}
---

Generate a question that is RELATED to this content area but CANNOT be
fully answered from this chunk alone. It should require additional context,
a different source, or more specific information.

Output ONLY the question."""

    try:
        question = call_groq(
            prompt=prompt,
            model=QUESTION_MODEL,
            temperature=0.9,
            max_tokens=80,
        )
        question = question.strip().strip('"').strip("'")
        if not question.endswith("?"):
            question += "?"

        # Generate abstention answer
        answer_prompt = f"""{full_context[:1000]}

Question: {question}

This context does NOT fully answer the question. Generate an honest response
that acknowledges the limitation and guides the engineer.
{related_hint}

Use this format:
**Direct Answer:**
I don't have enough information in the provided context to answer this accurately.

**What I found:**
[what the metadata and content do contain]

**Recommendation:**
[where to look instead — use Related pages from metadata if listed, otherwise official docs or team lead]"""

        answer = call_groq(
            prompt=answer_prompt,
            model=ANSWER_MODEL,
            system=SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=300,
        )

        return {"question": question, "answer": answer}

    except Exception as e:
        print(f"    Negative example generation failed: {e}")
        return None


# ── CHROMADB SAMPLER ──────────────────────────────────────────────────────────

def _chunk_where_filter(chunk_type: str) -> dict:
    """Map training bucket to the ChromaDB metadata filter."""
    if chunk_type == "reference":
        return {"$and": [{"type": "wikis_arch_docs"}, {"subtype": "reference"}]}
    if chunk_type == "wikis_arch_docs":
        return {"$and": [{"type": "wikis_arch_docs"}, {"subtype": {"$ne": "reference"}}]}
    return {"type": chunk_type}


def sample_chunks_from_db(collection, target_counts: dict) -> dict[str, list[dict]]:
    """
    Sample chunks from ChromaDB balanced across types.
    Returns dict: type → list of {content, metadata}
    """
    sampled = {}

    for chunk_type, count in target_counts.items():
        print(f"  Sampling {count} chunks of type '{chunk_type}'...")

        try:
            results = collection.get(
                where=_chunk_where_filter(chunk_type),
                limit=count * 3,
                include=["documents", "metadatas"]
            )

            docs   = results.get("documents", [])
            metas  = results.get("metadatas", [])

            if not docs:
                print(f"    WARNING: No chunks found for type '{chunk_type}'")
                sampled[chunk_type] = []
                continue

            # Filter out very short chunks — not enough content for Q&A
            valid = [
                {"content": doc, "metadata": meta}
                for doc, meta in zip(docs, metas)
                if len(doc.strip()) > 100
            ]

            # Random sample
            random.shuffle(valid)
            sampled[chunk_type] = valid[: count * 3]
            OUT_DIR=Path("./chunks")
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            file_name=f"chunks_{chunk_type}.jsonl"
            with open(OUT_DIR / file_name, "w") as f:
                for chunk in sampled[chunk_type]:
                    f.write(json.dumps(chunk) + "\n")
            print(f"Got {len(sampled[chunk_type])} valid chunks")

        except Exception as e:
            print(f"    ERROR sampling {chunk_type}: {e}")
            sampled[chunk_type] = []

    return sampled


# ── JSONL FORMATTER ───────────────────────────────────────────────────────────

def format_as_jsonl(
    question: str,
    context: str,
    answer: str,
    chunk_type: str,
    metadata: dict,
    example_id: str,
) -> dict:
    """Format one Q&A pair as a chat JSONL entry."""
    display_type = TYPE_SOURCE_LABELS.get(chunk_type, chunk_type)
    full_context = build_training_context(context, metadata, chunk_type)

    user_content = f"""Context ({display_type}):
{full_context}

Question: {question}"""

    return {
        "messages": [
            {
                "role":    "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role":    "user",
                "content": user_content
            },
            {
                "role":    "assistant",
                "content": answer
            }
        ],
        "_meta": {
            "example_id": example_id,
            "chunk_type": chunk_type,
            "source":     metadata.get("source", ""),
            "subtype":    metadata.get("subtype", ""),
        }
    }


# ── CHECKPOINT ────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {
        "completed":  [],
        "failed":     [],
        "counts":     defaultdict(int),
        "rejected":   defaultdict(int),
        "seen_questions": defaultdict(list),
        "total_done": 0,
    }


def save_checkpoint(cp: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Convert defaultdict to regular dict for JSON
    cp_save = dict(cp)
    cp_save["counts"] = dict(cp.get("counts", {}))
    cp_save["rejected"] = dict(cp.get("rejected", {}))
    cp_save["seen_questions"] = dict(cp.get("seen_questions", {}))
    CHECKPOINT_PATH.write_text(
        json.dumps(cp_save, indent=2),
        encoding="utf-8"
    )


# ── QUALITY FILTER ────────────────────────────────────────────────────────────

def _context_has_code(text: str) -> bool:
    """True when the chunk content is code-heavy."""
    if "```" in text:
        return True
    code_markers = (
        "Code:\n", "Signature:", "async def ", "def ", "@app.",
        "@router.", "class ", "import ",
    )
    return sum(1 for marker in code_markers if marker in text) >= 2


def _normalize_code_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _extract_grounding_terms(text: str) -> set[str]:
    """Pull technical tokens from context for grounding checks."""
    terms = set()
    patterns = [
        r"\bfastapi(?:\.\w+)+\b",
        r"\b[A-Z][a-zA-Z0-9_]+\b",
        r"\b[a-z_]{4,}\b",
        r"\bv?\d+\.\d+(?:\.\d+)?\b",
        r"`([^`]+)`",
        r"\bdef\s+([a-zA-Z_]\w*)",
        r"@(?:app|router)\.(get|post|put|delete|patch|websocket)\b",
    ]
    stopwords = {
        "this", "that", "with", "from", "have", "been", "will", "your",
        "when", "what", "where", "which", "their", "there", "about",
        "using", "used", "into", "only", "also", "should", "would",
        "could", "does", "doesn", "don", "answer", "details", "source",
        "context", "question", "information", "provided", "enough",
        "recommendation", "found", "direct", "technical", "example",
    }

    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            token = match.lower().strip()
            if len(token) >= 4 and token not in stopwords:
                terms.add(token)

    return terms


def is_grounded_answer(answer: str, context: str, min_overlap: int = 2) -> bool:
    """Reject answers that do not reuse specific details from the context."""
    context_terms = _extract_grounding_terms(context)
    if len(context_terms) < min_overlap:
        return True

    answer_terms = _extract_grounding_terms(answer)
    overlap = context_terms & answer_terms
    return len(overlap) >= min_overlap


def _extract_code_blocks(text: str) -> list[str]:
    return [m.strip() for m in re.findall(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)]


def _is_directory_tree_block(block: str) -> bool:
    """Detect ASCII directory-tree examples (not Python code)."""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return False
    tree_markers = ("├──", "└──", "│", "── ")
    tree_lines = sum(
        1 for line in lines
        if line == "." or any(marker in line for marker in tree_markers)
    )
    return tree_lines / len(lines) >= 0.4


def _allowed_identifier_terms(metadata: dict, chunk_type: str) -> set[str]:
    """Identifiers allowed because they come from chunk metadata, not invented."""
    allowed: set[str] = set()
    fields_by_type = {
        "reference":       ["qualified_name", "parent", "kind", "section"],
        "wikis_arch_docs": ["mentions_classes", "title", "header_path"],
        "code_contracts":  ["name", "topic", "kind"],
    }
    for field in fields_by_type.get(chunk_type, []):
        value = str(metadata.get(field, "")).strip()
        if not value:
            continue
        allowed.add(value.lower())
        for part in re.split(r"[,./\s]+", value):
            if len(part) >= 3:
                allowed.add(part.lower())
    if chunk_type == "reference":
        allowed |= REFERENCE_CONTEXT_TOKENS
    return allowed


def _identifier_is_allowed(
    identifier: str,
    context: str,
    allowed_extra: set[str],
) -> bool:
    key = identifier.lower().strip("`")
    if key in COMMON_TYPES:
        return True
    if key in allowed_extra:
        return True
    context_lower = context.lower()
    if key in context_lower:
        return True
    for part in key.split("."):
        if len(part) >= 3 and part in context_lower:
            return True
    for term in allowed_extra:
        if term in key or key in term:
            return True
    return False


def _code_block_grounded(
    block: str,
    context: str,
    min_line_ratio: float = 0.7,
) -> bool:
    """True when an answer code block is copied from context, not invented."""
    lines = [line for line in block.splitlines() if line.strip()]
    if not lines:
        return True

    context_lines = {
        _normalize_code_line(line)
        for line in context.splitlines()
        if line.strip()
    }
    matched = sum(1 for line in lines if _normalize_code_line(line) in context_lines)
    return (matched / len(lines)) >= min_line_ratio


def answer_has_invented_code(
    answer: str,
    context: str,
    chunk_type: str = "",
) -> bool:
    """Reject only when the answer introduces new code blocks."""
    blocks = _extract_code_blocks(answer)
    if not blocks:
        return False

    min_ratio = 0.5 if chunk_type == "reference" else 0.7

    for block in blocks:
        if chunk_type == "reference" and _is_directory_tree_block(block):
            continue
        if not _code_block_grounded(block, context, min_line_ratio=min_ratio):
            return True
    return False


def _extract_technical_identifiers(text: str) -> set[str]:
    ids: set[str] = set()
    for match in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text):
        ids.add(match)
    for match in re.findall(r"`([^`]+)`", text):
        ids.add(match.strip())
    for match in re.findall(
        r"\b(?:Request|Response|HTTPException|BackgroundTasks|APIRouter)\b", text
    ):
        ids.add(match)
    return ids


COMMON_TYPES = {
    "str", "int", "float", "bool", "none", "true", "false", "any",
    "dict", "list", "optional", "sequence", "callable", "type",
}


def answer_invents_identifiers(
    answer: str,
    context: str,
    metadata: dict | None = None,
    chunk_type: str = "",
) -> bool:
    """Reject answers that name APIs/constants absent from the context."""
    body = answer.split("**Source:**")[0]
    allowed_extra = _allowed_identifier_terms(metadata or {}, chunk_type)

    for identifier in _extract_technical_identifiers(body):
        if not _identifier_is_allowed(identifier, context, allowed_extra):
            return True
    return False


def _extract_urls(text: str) -> set[str]:
    return set(re.findall(r"https?://[^\s\)\]`\"']+", text))


def answer_has_invented_urls(answer: str, context: str) -> bool:
    """Reject abstention answers that cite URLs not present in the context."""
    answer_urls = _extract_urls(answer)
    if not answer_urls:
        return False
    allowed = _extract_urls(context)
    for url in answer_urls:
        if not any(url in known or known in url for known in allowed):
            return True
    return False


INTEGRATION_QUESTION = re.compile(
    r"\b(integrat(?:e|ion)|set\s*up|configure|implement|"
    r"production(?:\s+environment)?|deploy(?:ment)?|best practices)\b",
    re.IGNORECASE,
)

HOWTO_QUESTION = re.compile(
    r"\bhow (?:do|to|can|would|should)\b",
    re.IGNORECASE,
)

NAMED_TECH = re.compile(
    r"\b(celery|redis|kafka|rabbitmq|postgresql|postgres|mongodb|"
    r"docker|kubernetes|nginx|sqlalchemy|alembic)\b",
    re.IGNORECASE,
)


def _context_explains_topic(content: str, topic: str) -> bool:
    """True when context contains real how-to detail about topic, not a one-line mention."""
    topic_lower = topic.lower()
    content_lower = content.lower()
    if topic_lower not in content_lower:
        return False

    for match in re.finditer(re.escape(topic_lower), content_lower):
        start = max(0, match.start() - 200)
        end = min(len(content_lower), match.end() + 400)
        window = content_lower[start:end]

        if "```" in window:
            return True
        if window.count(".") >= 2 and len(window) > 150:
            return True
        if re.search(
            rf"{topic_lower}.{{0,100}}"
            r"(install|pip|broker|worker|queue|configure|setup|connect|deploy)",
            window,
        ):
            return True

    # One-line "Use Celery for..." style mentions are not explanatory
    return False


def community_qa_overreaches(
    question: str,
    answer: str,
    context: str,
    chunk_type: str,
) -> bool:
    """Reject thin community Q&A chunks answering deep integration how-tos."""
    if chunk_type != "community_qa":
        return False
    if ABSTENTION_PHRASE in answer.lower():
        return False

    needs_depth = bool(
        INTEGRATION_QUESTION.search(question) or HOWTO_QUESTION.search(question)
    )
    if not needs_depth:
        return False

    content = context.split("--- Content ---")[-1] if "--- Content ---" in context else context
    content = content.strip()

    # Question names a technology the context only name-drops (e.g. "use Celery")
    for tech in set(NAMED_TECH.findall(question)):
        if not _context_explains_topic(content, tech):
            return True

    has_substantive_code = "```" in content and len(content) > 300
    has_procedural_detail = len(content) > 500 and content.lower().count(".") >= 3

    return not (has_substantive_code or has_procedural_detail)


def _normalize_question(question: str) -> str:
    normalized = question.lower()
    normalized = re.sub(r"`[^`]+`", "<name>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def question_similarity(left: str, right: str) -> float:
    left_words = set(_normalize_question(left).split())
    right_words = set(_normalize_question(right).split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def is_duplicate_question(
    question: str,
    seen_questions: list[str],
    chunk_type: str = "",
    threshold: float | None = None,
) -> bool:
    if threshold is None:
        threshold = DUPLICATE_THRESHOLDS.get(
            chunk_type, DUPLICATE_THRESHOLDS["default"]
        )
    return any(
        question_similarity(question, prev) >= threshold
        for prev in seen_questions
    )


def validate_positive_example(
    question: str,
    answer: str,
    context: str,
    chunk_type: str,
    seen_questions: list[str],
    metadata: dict | None = None,
) -> tuple[bool, str]:
    if len(question) < 15:
        return False, "question_too_short"

    generic_questions = [
        "what is this", "what does this do", "explain this",
        "what is fastapi", "how does python work",
    ]
    if any(g in question.lower() for g in generic_questions):
        return False, "generic_question"

    if is_duplicate_question(question, seen_questions, chunk_type):
        return False, "duplicate_question"

    if len(answer) < 80:
        return False, "answer_too_short"

    required_sections = ("**Direct Answer:**", "**Details:**", "**Source:**")
    if not all(section in answer for section in required_sections):
        return False, "missing_sections"

    if ABSTENTION_PHRASE in answer.lower():
        return False, "unexpected_abstention"

    if answer_has_invented_code(answer, context, chunk_type):
        return False, "invented_code"

    if answer_invents_identifiers(answer, context, metadata, chunk_type):
        return False, "invented_identifiers"

    if not is_grounded_answer(answer, context):
        return False, "not_grounded"

    if community_qa_overreaches(question, answer, context, chunk_type):
        return False, "community_qa_overreach"

    return True, ""


def is_valid_positive_example(
    question: str,
    answer: str,
    context: str,
    chunk_type: str = "",
    seen_questions: list[str] | None = None,
    metadata: dict | None = None,
) -> bool:
    ok, _ = validate_positive_example(
        question, answer, context, chunk_type, seen_questions or [], metadata
    )
    return ok


def validate_negative_example(
    question: str,
    answer: str,
    context: str,
) -> tuple[bool, str]:
    if len(question) < 15:
        return False, "question_too_short"
    if len(answer) < 80:
        return False, "answer_too_short"

    required_sections = (
        "**Direct Answer:**",
        "**What I found:**",
        "**Recommendation:**",
    )
    if not all(section in answer for section in required_sections):
        return False, "missing_sections"

    if ABSTENTION_PHRASE not in answer.lower():
        return False, "missing_abstention"

    if answer_has_invented_urls(answer, context):
        return False, "invented_urls"

    return True, ""


def is_valid_negative_example(question: str, answer: str, context: str = "") -> bool:
    ok, _ = validate_negative_example(question, answer, context)
    return ok


def stable_chunk_id(chunk_type: str, content: str, source: str = "") -> str:
    """Stable ID for checkpoint resume across runs."""
    digest = hashlib.sha256(f"{chunk_type}|{source}|{content}".encode()).hexdigest()[:16]
    return f"{chunk_type}_{digest}"


def is_train_example(example_id: str) -> bool:
    """Deterministic train/val split so reruns stay consistent."""
    bucket = int(hashlib.sha256(example_id.encode()).hexdigest(), 16) % 100
    return bucket < int(TRAIN_RATIO * 100)


# ── MAIN GENERATOR ────────────────────────────────────────────────────────────

def build_dataset(target_counts: dict, negative_count: int):
    import chromadb

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ChromaDB
    print("Loading ChromaDB...")
    db_client  = chromadb.PersistentClient(path=str(DB_PATH))
    collection = db_client.get_collection(name="engineering_knowledge")
    print(f"Collection: {collection.count()} chunks\n")

    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_set = set(checkpoint.get("completed", []))
    counts = defaultdict(int, checkpoint.get("counts", {}))
    rejected = defaultdict(int, checkpoint.get("rejected", {}))
    seen_questions = defaultdict(
        list,
        {k: list(v) for k, v in checkpoint.get("seen_questions", {}).items()},
    )
    total_done = checkpoint.get("total_done", 0)

    # Sample chunks
    print("Sampling chunks from ChromaDB...")
    sampled = sample_chunks_from_db(collection, target_counts)

    # Open output files in append mode
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path   = OUTPUT_DIR / "val.jsonl"

    examples_buffer = []
    total_target = sum(target_counts.values())

    print(f"\nGenerating {total_target} positive examples...\n")

    for chunk_type, chunks in sampled.items():
        type_target = target_counts[chunk_type]
        type_done   = counts.get(chunk_type, 0)

        print(f"\n── {chunk_type} ({type_done}/{type_target}) ──")

        for i, chunk in enumerate(chunks):
            if type_done >= type_target:
                break

            content  = chunk["content"]
            metadata = chunk["metadata"]
            chunk_id = stable_chunk_id(
                chunk_type,
                content,
                metadata.get("source", ""),
            )
            if chunk_id in completed_set:
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
                    seen_questions[chunk_type],
                    metadata,
                )
                if not ok:
                    print(f"  [{i+1}] FILTERED — {reason}")
                    rejected[reason] += 1
                    continue

                seen_questions[chunk_type].append(question)

                example = format_as_jsonl(
                    question=question,
                    context=content,
                    answer=answer,
                    chunk_type=chunk_type,
                    metadata=metadata,
                    example_id=chunk_id,
                )

                examples_buffer.append(example)
                completed_set.add(chunk_id)
                counts[chunk_type] += 1
                type_done += 1
                total_done += 1

                print(f"  [{type_done}/{type_target}] Q: {question[:70]}")

                if total_done % 10 == 0:
                    checkpoint["completed"]  = list(completed_set)
                    checkpoint["counts"]     = dict(counts)
                    checkpoint["rejected"]   = dict(rejected)
                    checkpoint["seen_questions"] = dict(seen_questions)
                    checkpoint["total_done"] = total_done
                    save_checkpoint(checkpoint)
                    _flush_buffer(examples_buffer, train_path, val_path)
                    examples_buffer = []

                time.sleep(0.3)

            except KeyboardInterrupt:
                print("\n\nInterrupted — saving checkpoint...")
                checkpoint["completed"]  = list(completed_set)
                checkpoint["counts"]     = dict(counts)
                checkpoint["rejected"]   = dict(rejected)
                checkpoint["seen_questions"] = dict(seen_questions)
                checkpoint["total_done"] = total_done
                save_checkpoint(checkpoint)
                _flush_buffer(examples_buffer, train_path, val_path)
                return

            except Exception as e:
                print(f"  [{i+1}] ERROR: {e}")
                checkpoint.setdefault("failed", []).append(chunk_id)
                continue

    print(f"\n── Generating {negative_count} negative (abstention) examples ──")

    neg_chunks = []
    for chunk_type, chunks in sampled.items():
        sample_size = max(1, negative_count // max(len(sampled), 1))
        if chunks:
            neg_chunks.extend(random.sample(chunks, min(sample_size, len(chunks))))

    random.shuffle(neg_chunks)
    neg_done = counts.get("_negative", 0)

    for i, chunk in enumerate(neg_chunks):
        if neg_done >= negative_count:
            break

        chunk_type = resolve_chunk_type(chunk["metadata"])
        chunk_id = stable_chunk_id(
            f"negative_{chunk_type}",
            chunk["content"],
            chunk["metadata"].get("source", ""),
        )
        if chunk_id in completed_set:
            continue

        try:
            full_context = build_training_context(
                chunk["content"], chunk["metadata"], chunk_type
            )
            result = generate_negative_example(
                chunk["content"], chunk_type, chunk["metadata"]
            )
            if not result:
                continue

            ok, reason = validate_negative_example(
                result["question"],
                result["answer"],
                full_context,
            )
            if not ok:
                print(f"  [{i+1}] FILTERED — {reason}")
                rejected[reason] += 1
                continue

            example = format_as_jsonl(
                question=result["question"],
                context=chunk["content"],
                answer=result["answer"],
                chunk_type=chunk_type,
                metadata=chunk["metadata"],
                example_id=chunk_id,
            )
            example["_meta"]["is_negative"] = True
            examples_buffer.append(example)
            completed_set.add(chunk_id)
            neg_done += 1
            counts["_negative"] = neg_done
            total_done += 1
            print(f"  [{neg_done}/{negative_count}] Q: {result['question'][:70]}")

        except Exception as e:
            print(f"  [{i}] Negative ERROR: {e}")
            continue

    _flush_buffer(examples_buffer, train_path, val_path)

    checkpoint["completed"]  = list(completed_set)
    checkpoint["counts"]     = dict(counts)
    checkpoint["rejected"]   = dict(rejected)
    checkpoint["seen_questions"] = dict(seen_questions)
    checkpoint["total_done"] = total_done
    save_checkpoint(checkpoint)

    train_count = sum(1 for _ in open(train_path, encoding="utf-8"))
    val_count   = sum(1 for _ in open(val_path, encoding="utf-8"))

    print(f"\n{'='*55}")
    print("  Dataset generation complete")
    print(f"  Train examples:  {train_count}")
    print(f"  Val examples:    {val_count}")
    print(f"  Total:           {train_count + val_count}")
    print(f"  Rejected:        {dict(rejected)}")
    print(f"  Train path:      {train_path}")
    print(f"  Val path:        {val_path}")
    print("\n  Per type:")
    for t, c in counts.items():
        if not t.startswith("_"):
            print(f"    {t:20s}: {c}")
    print(f"{'='*55}\n")


def _flush_buffer(buffer: list, train_path: Path, val_path: Path):
    """Write buffered examples to train/val files with a stable split."""
    if not buffer:
        return

    train_ex = [ex for ex in buffer if is_train_example(ex["_meta"]["example_id"])]
    val_ex = [ex for ex in buffer if not is_train_example(ex["_meta"]["example_id"])]

    with open(train_path, "a", encoding="utf-8") as f:
        for ex in train_ex:
            ex_clean = {k: v for k, v in ex.items() if k != "_meta"}
            f.write(json.dumps(ex_clean, ensure_ascii=False) + "\n")

    with open(val_path, "a", encoding="utf-8") as f:
        for ex in val_ex:
            ex_clean = {k: v for k, v in ex.items() if k != "_meta"}
            f.write(json.dumps(ex_clean, ensure_ascii=False) + "\n")


# ── INSPECT SAMPLE ────────────────────────────────────────────────────────────

def inspect_sample(n: int = 3):
    """Print n sample examples from the generated dataset."""
    train_path = OUTPUT_DIR / "train.jsonl"
    if not train_path.exists():
        print("No dataset yet — run build_dataset() first")
        return

    examples = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    samples = random.sample(examples, min(n, len(examples)))

    for i, ex in enumerate(samples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}")
        print(f"{'='*60}")
        for msg in ex["messages"]:
            role = msg["role"].upper()
            content = msg["content"]
            if role == "SYSTEM":
                print(f"[SYSTEM] {content[:100]}...")
            elif role == "USER":
                print(f"\n[USER]\n{content[:400]}")
            else:
                print(f"\n[ASSISTANT]\n{content[:400]}")
        print()


def reset_outputs():
    """Clear generated dataset files and checkpoint."""
    for path in (CHECKPOINT_PATH, OUTPUT_DIR / "train.jsonl", OUTPUT_DIR / "val.jsonl"):
        if path.exists():
            path.unlink()
            print(f"Removed {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QLoRA JSONL from ChromaDB chunks")
    parser.add_argument(
        "--mode",
        choices=("test", "full"),
        default="test",
        help="test = 3 examples per type; full = production dataset",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing train.jsonl, val.jsonl, and checkpoint before generating",
    )
    parser.add_argument(
        "--inspect",
        type=int,
        default=0,
        metavar="N",
        help="Print N random samples from train.jsonl after generation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """args = parse_args()

    if args.reset:
        reset_outputs()

    target_counts = TARGET_COUNTS.copy()
    negative_count = NEGATIVE_COUNT

    if args.mode == "test":
        print("TEST MODE — generating 3 examples per type\n")
        target_counts = {k: 3 for k in TARGET_COUNTS}
        negative_count = 3

    build_dataset(target_counts, negative_count)

    if args.inspect > 0:
        inspect_sample(args.inspect)"""
    import chromadb
    db_client  = chromadb.PersistentClient(path=str(DB_PATH))
    collection = db_client.get_collection(name="engineering_knowledge")
    target_counts = TARGET_COUNTS.copy()

    sample_chunks_from_db(collection, target_counts)