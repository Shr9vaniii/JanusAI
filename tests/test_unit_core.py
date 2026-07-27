"""Unit tests for session store, cache keys, fusion, and orchestration helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.cache_store import (
    CACHE_VERSION,
    chunk_fingerprint,
    make_answer_key,
    normalize_query,
    should_cache_answer,
)
from inference.rag_engine import OnboardingRAGEngine
from inference.retriever import merge_results_rrf, should_dual_retrieve
from inference.session_store import SessionStore
from inference.utils import extract_assistant_reply
from retrieval.hybrid_retrieval import tokenize_query


@pytest.fixture()
def session_store(tmp_path: Path) -> SessionStore:
    return SessionStore(db_path=tmp_path / "sessions.db")


def test_session_create_and_turn(session_store: SessionStore):
    sid = session_store.create_session()
    assert session_store.session_exists(sid)
    session_store.add_turn(sid, "What is Depends?", "Depends injects dependencies.")
    turns = session_store.get_recent_exchanges(sid, max_exchanges=3)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"


def test_session_topic_update(session_store: SessionStore):
    sid = session_store.create_session()
    session_store.update_session_context(
        sid,
        active_topic="HTTPException",
        topic_summary="Discussing exception args",
        entities=["HTTPException"],
    )
    rec = session_store.get_session(sid)
    assert rec is not None
    assert rec.active_topic == "HTTPException"
    assert "HTTPException" in rec.entities


def test_normalize_and_cache_key_stable():
    a = make_answer_key("  What ARGS? ", [{"id": "c1"}, {"id": "c2"}])
    b = make_answer_key("what args?", [{"id": "c1"}, {"id": "c2"}])
    assert a == b
    assert CACHE_VERSION
    assert normalize_query("  A   B ") == "a b"


def test_cache_key_changes_with_chunks_or_version():
    k1 = make_answer_key("q", [{"id": "a"}])
    k2 = make_answer_key("q", [{"id": "b"}])
    k3 = make_answer_key("q", [{"id": "a"}], model_version="other")
    assert k1 != k2
    assert k1 != k3
    assert chunk_fingerprint([{"id": "x"}, {"metadata": {"source": "y"}}]) == "x,y"


def test_should_not_cache_abstention():
    assert should_cache_answer("I don't have enough information in the provided context") is False
    assert should_cache_answer("**Direct Answer:**\nHTTPException takes status_code") is True


def test_tokenize_query_strips_stopwords():
    tokens = tokenize_query("What arguments does HTTPException take?")
    assert "httpexception" in tokens
    assert "what" not in tokens
    assert "does" not in tokens


def test_merge_results_rrf_prefers_shared_hits():
    left = [
        {"id": "a", "final_score": 1.0, "metadata": {}},
        {"id": "b", "final_score": 0.5, "metadata": {}},
    ]
    right = [
        {"id": "b", "final_score": 0.9, "metadata": {}},
        {"id": "c", "final_score": 0.4, "metadata": {}},
    ]
    merged = merge_results_rrf([left, right], n_final=3)
    ids = [h["id"] for h in merged]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_should_dual_retrieve_rules():
    assert (
        should_dual_retrieve(
            needs_rewrite=True,
            confidence=0.4,
            topic_status="same",
            original="its args?",
            retrieval_query="What arguments does HTTPException take?",
        )
        is True
    )
    assert (
        should_dual_retrieve(
            needs_rewrite=False,
            confidence=0.4,
            topic_status="same",
            original="q",
            retrieval_query="q2",
        )
        is False
    )
    assert (
        should_dual_retrieve(
            needs_rewrite=True,
            confidence=0.9,
            topic_status="ambiguous",
            original="its args?",
            retrieval_query="What arguments does HTTPException take?",
        )
        is True
    )


def test_extract_assistant_reply_markers():
    raw = (
        "<|start_header_id|>user<|end_header_id|>\nQ\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "**Direct Answer:**\nHello"
    )
    assert extract_assistant_reply(raw).startswith("**Direct Answer:**")


def test_format_multi_answer_partial():
    from inference.query_decomposer import SubQueryAnswer

    text = OnboardingRAGEngine._format_multi_answer(
        [
            SubQueryAnswer(query="q1", answer="a1"),
            SubQueryAnswer(query="q2", answer="a2"),
        ],
        partial=True,
    )
    assert "could not be answered" in text
    assert "Question 1" in text
    assert "Question 2" in text
