"""Evaluation runner for retrieval + end-to-end RAG quality.

Usage:
  python -m evaluation.runner --retrieval-only
  python -m evaluation.runner --base-url http://127.0.0.1:8000
  python -m evaluation.runner --local
  python -m evaluation.runner --output evaluation/results/latest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.utils import load_dotenv

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class CaseResult:
    id: str
    category: str
    ok: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""


_ABSTAIN_PATTERNS = re.compile(
    r"i (don.t|do not|cannot|can.t) (have|find|provide|answer|give)|"
    r"not enough information|"
    r"(outside|beyond|not (in|within|part of)) (my|the) (corpus|context|scope|knowledge|documentation)|"
    r"(cannot|can.t|not able to) (answer|help|assist) (with )?this|"
    r"(no|insufficient) (information|context|data) (in|from|within) (the )?context|"
    r"the (provided )?context (does not|doesn.t) (contain|cover|address|include)|"
    r"(this topic|this question|that topic) (is|falls) (outside|beyond|not covered)",
    re.IGNORECASE,
)


def _is_abstention(answer: str) -> bool:
    """Return True if the LLM answer is a grounded refusal / abstention."""
    text = (answer or "").strip()
    if not text:
        return False
    # Multi-arm answers: ALL sub-arms must abstain for the whole response to count
    parts = re.split(r"### Question \d+:", text)
    if len(parts) > 1:
        return all(_ABSTAIN_PATTERNS.search(p) for p in parts if p.strip())
    return bool(_ABSTAIN_PATTERNS.search(text))


def _name_hit(hit: dict, needles: list[str]) -> bool:
    meta = hit.get("metadata") or {}
    blob = " ".join(
        str(meta.get(k, ""))
        for k in ("name", "qualified_name", "source", "parent")
    ).lower()
    blob += " " + str(hit.get("content", ""))[:400].lower()
    return any(n.lower() in blob for n in needles)


def precision_at_k(hits: list[dict], needles: list[str], k: int = 5) -> float:
    """Fraction of top-k hits that match any gold needle (0 if no needles)."""
    if not needles:
        return 0.0
    top = hits[:k]
    if not top:
        return 0.0
    relevant = sum(1 for h in top if _name_hit(h, needles))
    return relevant / len(top)


def recall_at_k(hits: list[dict], needles: list[str], k: int = 5) -> float:
    if not needles:
        return 0.0
    top = hits[:k]
    return 1.0 if any(_name_hit(h, needles) for h in top) else 0.0


def mrr(hits: list[dict], needles: list[str]) -> float:
    if not needles:
        return 0.0
    for i, h in enumerate(hits, 1):
        if _name_hit(h, needles):
            return 1.0 / i
    return 0.0


def citation_correct(citations: list[dict], needles: list[str]) -> float:
    if not needles:
        return 1.0
    if not citations:
        return 0.0
    for c in citations:
        blob = f"{c.get('name', '')} {c.get('source', '')} {c.get('snippet', '')}".lower()
        if any(n.lower() in blob for n in needles):
            return 1.0
    return 0.0


def load_dataset(path: Path = DATASET_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_retrieval_only(cases: list[dict]) -> list[CaseResult]:
    from inference.retriever import retrieve_for_query

    results: list[CaseResult] = []
    for case in cases:
        if case.get("session_seed"):
            continue
        # Abstention cases are generation-level — skip in retrieval-only scoring
        if case.get("expect_abstention") is True:
            results.append(
                CaseResult(
                    id=case["id"],
                    category=case["category"],
                    ok=True,
                    metrics={"skipped": True, "reason": "abstention evaluated in e2e only"},
                )
            )
            continue
        q = case["question"]
        needles = case.get("relevant_name_substrings") or []
        t0 = time.perf_counter()
        try:
            _, hits = retrieve_for_query(q, n_final=5, verbose=False)
            latency = (time.perf_counter() - t0) * 1000
            r_at_5 = recall_at_k(hits, needles, 5)
            p_at_5 = precision_at_k(hits, needles, 5)
            mrr_score = mrr(hits, needles)
            ok = True
            if needles:
                ok = r_at_5 >= 1.0
            results.append(
                CaseResult(
                    id=case["id"],
                    category=case["category"],
                    ok=ok,
                    metrics={
                        "recall_at_5": r_at_5,
                        "precision_at_5": round(p_at_5, 3),
                        "mrr": mrr_score,
                        "num_hits": len(hits),
                        "latency_ms": round(latency, 2),
                    },
                )
            )
        except Exception as exc:
            results.append(
                CaseResult(
                    id=case["id"],
                    category=case["category"],
                    ok=False,
                    error=str(exc),
                )
            )
    return results


def _http_json(method: str, url: str, payload: dict | None = None, timeout: int = 180) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def eval_via_api(cases: list[dict], base_url: str) -> list[CaseResult]:
    base = base_url.rstrip("/")
    results: list[CaseResult] = []

    for case in cases:
        try:
            if case.get("session_seed"):
                results.append(
                    CaseResult(
                        id=case["id"],
                        category=case["category"],
                        ok=True,
                        metrics={"skipped": True, "reason": "session cases use --local"},
                    )
                )
                continue

            session = _http_json("POST", f"{base}/sessions")
            session_id = session["session_id"]
            body = {
                "question": case["question"],
                "session_id": session_id,
                "bypass_cache": False,
            }
            t0 = time.perf_counter()
            first = _http_json("POST", f"{base}/ask", body)
            latency = (time.perf_counter() - t0) * 1000
            second = None
            if case.get("run_twice"):
                t1 = time.perf_counter()
                second = _http_json("POST", f"{base}/ask", body)
                latency = (time.perf_counter() - t1) * 1000

            answer = first.get("answer", "")
            abstained = _is_abstention(answer)
            needles = case.get("relevant_name_substrings") or []
            cite_ok = citation_correct(first.get("citations") or [], needles)
            metrics: dict[str, Any] = {
                "latency_ms": round(latency, 2),
                "cache_hit": first.get("cache_hit"),
                "abstained": abstained,
                "citation_correct": cite_ok,
                "num_chunks": first.get("num_chunks"),
                "is_multi": bool((first.get("decompose") or {}).get("is_multi")),
            }
            if second is not None:
                metrics["second_cache_hit"] = second.get("cache_hit")

            ok = True
            if case.get("expect_abstention") is True and not abstained:
                ok = False
            if case.get("expect_abstention") is False and abstained:
                ok = False
            if case.get("expect_multi") and not metrics["is_multi"]:
                ok = False
            if case.get("must_cite") and cite_ok < 1.0:
                ok = False

            results.append(
                CaseResult(
                    id=case["id"],
                    category=case["category"],
                    ok=ok,
                    metrics=metrics,
                )
            )
        except error.URLError as exc:
            results.append(
                CaseResult(id=case["id"], category=case["category"], ok=False, error=str(exc))
            )
        except Exception as exc:
            results.append(
                CaseResult(id=case["id"], category=case["category"], ok=False, error=str(exc))
            )
    return results


def eval_local_e2e(cases: list[dict]) -> list[CaseResult]:
    from inference.rag_engine import OnboardingRAGEngine
    from inference.session_store import SessionStore

    store = SessionStore(db_path=ROOT / "enterprise_data" / "eval_sessions.db")
    engine = OnboardingRAGEngine(session_store=store, use_cache=True)
    results: list[CaseResult] = []

    for case in cases:
        try:
            session_id = store.create_session()
            seed = case.get("session_seed") or []
            for i in range(0, len(seed) - 1, 2):
                if seed[i]["role"] == "user" and seed[i + 1]["role"] == "assistant":
                    store.add_turn(session_id, seed[i]["content"], seed[i + 1]["content"])

            t0 = time.perf_counter()
            resp = engine.generate(
                case["question"],
                session_id=session_id,
                use_cache=True,
                persist_session=True,
            )
            latency = (time.perf_counter() - t0) * 1000
            second = None
            if case.get("run_twice"):
                t1 = time.perf_counter()
                second = engine.generate(
                    case["question"],
                    session_id=session_id,
                    use_cache=True,
                )
                latency = (time.perf_counter() - t1) * 1000

            abstained = _is_abstention(resp.answer)
            needles = case.get("relevant_name_substrings") or []
            cite_ok = citation_correct(resp.citations, needles)
            metrics: dict[str, Any] = {
                "latency_ms": round(latency, 2),
                "cache_hit": resp.cache_hit,
                "abstained": abstained,
                "citation_correct": cite_ok,
                "num_chunks": resp.num_chunks,
                "is_multi": resp.is_multi,
                "dual_retrieval": resp.dual_retrieval,
                "timings": resp.timings.as_dict(),
            }
            if resp.rewrite:
                metrics["needs_rewrite"] = resp.rewrite.needs_rewrite
            if second is not None:
                metrics["second_cache_hit"] = second.cache_hit

            ok = True
            if case.get("expect_abstention") is True and not abstained:
                ok = False
            if case.get("expect_abstention") is False and abstained and needles:
                ok = False
            if case.get("expect_multi") and not resp.is_multi:
                ok = False
            if case.get("expect_rewrite") is True and not (resp.rewrite and resp.rewrite.needs_rewrite):
                ok = False
            if case.get("must_cite") and cite_ok < 1.0:
                ok = False

            results.append(
                CaseResult(
                    id=case["id"],
                    category=case["category"],
                    ok=ok,
                    metrics=metrics,
                )
            )
        except Exception as exc:
            results.append(
                CaseResult(id=case["id"], category=case["category"], ok=False, error=str(exc))
            )
    return results


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    scored = [r for r in results if not r.metrics.get("skipped")]
    total = len(scored)
    passed = sum(1 for r in scored if r.ok)
    recalls = [r.metrics.get("recall_at_5") for r in scored if "recall_at_5" in r.metrics]
    precs = [r.metrics.get("precision_at_5") for r in scored if "precision_at_5" in r.metrics]
    mrrs = [r.metrics.get("mrr") for r in scored if "mrr" in r.metrics]
    cites = [r.metrics.get("citation_correct") for r in scored if "citation_correct" in r.metrics]
    lats = [r.metrics.get("latency_ms") for r in scored if "latency_ms" in r.metrics]
    summary: dict[str, Any] = {
        "total_scored": total,
        "total_including_skipped": len(results),
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
    }
    if recalls:
        summary["avg_recall_at_5"] = round(sum(recalls) / len(recalls), 3)
    if precs:
        summary["avg_precision_at_5"] = round(sum(precs) / len(precs), 3)
    if mrrs:
        summary["avg_mrr"] = round(sum(mrrs) / len(mrrs), 3)
    if cites:
        summary["avg_citation_correct"] = round(sum(cites) / len(cites), 3)
    if lats:
        summary["avg_latency_ms"] = round(sum(lats) / len(lats), 2)
        summary["p50_latency_ms"] = round(sorted(lats)[len(lats) // 2], 2)
    return summary


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Onboarding RAG evaluation")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--local", action="store_true", help="Run local engine e2e")
    parser.add_argument("--base-url", default="", help="Evaluate against deployed API")
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "latest.json"),
        help="Where to write JSON results",
    )
    args = parser.parse_args()

    dataset = load_dataset()
    cases = dataset["cases"]

    if args.base_url:
        results = eval_via_api(cases, args.base_url)
    elif args.local:
        results = eval_local_e2e(cases)
    else:
        results = eval_retrieval_only(cases)

    summary = summarize(results)
    out = {
        "dataset_version": dataset.get("version"),
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
