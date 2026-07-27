"""Integration tests (network / model / corpus). Marked so unit CI can skip them.

Run:
  pytest -m integration
  pytest -m "not integration"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.integration


def _chroma_available() -> bool:
    db = ROOT / "enterprise_data" / "chroma_db_v3"
    return db.exists()


@pytest.mark.skipif(not _chroma_available(), reason="Chroma corpus not present")
def test_hybrid_retrieve_httpexception():
    from inference.retriever import retrieve_for_query

    classified, hits = retrieve_for_query(
        "What arguments does HTTPException take?",
        n_final=5,
        verbose=False,
    )
    assert classified is not None
    assert hits
    blob = " ".join(
        str((h.get("metadata") or {}).get("name", ""))
        + str((h.get("metadata") or {}).get("qualified_name", ""))
        for h in hits
    ).lower()
    assert "httpexception" in blob


@pytest.mark.skipif(not os.environ.get("INFERENCE_URL"), reason="INFERENCE_URL not set")
def test_remote_health_or_generate_smoke():
    from inference.utils import load_dotenv

    load_dotenv()
    from inference.rag_engine import OnboardingRAGEngine

    engine = OnboardingRAGEngine(backend="remote", load_model=False)
    health = engine.health()
    assert "generation" in health
