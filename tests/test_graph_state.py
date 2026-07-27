"""Smoke checks for GraphState helpers (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.agents import GraphState, Timer, initial_state, merge_timing


def test_initial_state_defaults():
    state: GraphState = initial_state("What is Depends?")
    assert state["question"] == "What is Depends?"
    assert state["retrieval_query"] == "What is Depends?"
    assert state["is_multi"] is False
    assert state["intent"] is None
    assert state["hits"] == []
    assert state["sub_results"] == []
    assert "intent_ms" in state["timings"]
    assert "retrieve_ms" in state["timings"]
    assert state["request_id"]


def test_intent_and_hits_are_separate_fields():
    state = initial_state("q")
    # Agents will fill these independently — Intent then Hybrid.
    state["intent"] = {
        "intent_name": "API_REFERENCE",
        "retrieval_intent": "reference",
        "search_types": ["reference"],
        "confidence": "high",
        "where": {"subtype": "reference"},
    }
    state["hits"] = [{"id": "c1", "content": "..."}]
    assert state["intent"]["intent_name"] == "API_REFERENCE"
    assert state["hits"][0]["id"] == "c1"


def test_merge_timing_and_timer():
    state = initial_state("q")
    with Timer() as t:
        pass
    updated = merge_timing(state, rewrite_ms=t.ms, intent_ms=1.5)
    assert updated["intent_ms"] == 1.5
    assert "rewrite_ms" in updated
    # original state timings unchanged
    assert state["timings"]["intent_ms"] == 0.0
