"""
Tests for src/rag.py — multi-skill retrieval with confidence classification.

Stubs:
- search_skills returns list of (name, description, body, score) tuples
- embed returns a dummy vector
- rerank returns sorted (idx, score) pairs from the raw input scores
  (bypass real cross-encoder)
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src/ to path so we can import rag
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers to build stub data
# ---------------------------------------------------------------------------

def _make_raw_results(scores: list[float]) -> list[tuple]:
    """Build fake search_skills rows: (name, description, body, score)."""
    return [
        (f"skill_{i}", f"Description {i}", f"Content of skill {i}", score)
        for i, score in enumerate(scores)
    ]


def _stub_rerank(query, passages, top_k=None):
    """
    Stub reranker: returns (idx, score) pairs in the same order they were
    passed in (no actual reranking), sorted by score descending.
    The scores come directly from the passages list length — we rely on the
    caller to pass scores-as-passages trick, but here we just return identity
    ranked tuples derived from the order (scores are already sorted desc by
    search_skills stub).

    Actually, rag.py passes texts derived from raw results, and we need to
    return (original_idx, score) pairs.  We fake it by returning indices in
    order with the same scores that search_skills returned.
    """
    # We receive len(passages) items; return them in order (already sorted desc)
    return [(i, 1.0 - i * 0.01) for i in range(len(passages))]


DUMMY_VEC = [0.0] * 1024


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoMatchReturnsNoneConfidence:
    def test_no_match_returns_none_confidence(self):
        """search_skills returns [] → confidence=='none', skills==[]"""
        with (
            patch("rag.search_skills", return_value=[]),
            patch("rag.embed", return_value=[DUMMY_VEC]),
            patch("rag.USE_RERANKER", False),
        ):
            import rag
            result = rag.get_skills_with_confidence("anything")

        assert result.confidence == "none"
        assert result.skills == []
        assert result.candidate_count == 0


class TestBelowAbsoluteFloor:
    def test_below_absolute_floor(self):
        """Single result with score 0.20 → below SKILL_ABSOLUTE_MIN_SCORE → 'none'"""
        raw = _make_raw_results([0.20])
        with (
            patch("rag.search_skills", return_value=raw),
            patch("rag.embed", return_value=[DUMMY_VEC]),
            patch("rag.USE_RERANKER", False),
            patch("rag.SKILL_ABSOLUTE_MIN_SCORE", 0.30),
        ):
            import rag
            result = rag.get_skills_with_confidence("some query")

        assert result.confidence == "none"
        assert result.skills == []
        assert result.top_score == 0.20


class TestHighConfidenceReturnsSingle:
    def test_high_confidence_returns_single(self):
        """Scores [0.85, 0.40, 0.35] → gap=0.45 >= HIGH_GAP → 'high', count=1"""
        raw = _make_raw_results([0.85, 0.40, 0.35])
        with (
            patch("rag.search_skills", return_value=raw),
            patch("rag.embed", return_value=[DUMMY_VEC]),
            patch("rag.USE_RERANKER", False),
            patch("rag.SKILL_ABSOLUTE_MIN_SCORE", 0.30),
            patch("rag.SKILL_RELATIVE_KEEP", 0.85),
            patch("rag.SKILL_HIGH_CONFIDENCE_GAP", 0.15),
            patch("rag.SKILL_MEDIUM_CONFIDENCE_GAP", 0.05),
            patch("rag.SKILL_MAX_RETURN", 3),
        ):
            import rag
            result = rag.get_skills_with_confidence("cloud run OOM")

        assert result.confidence == "high"
        assert len(result.skills) == 1
        assert result.skills[0].name == "skill_0"
        assert result.top_score == 0.85


class TestMediumConfidenceReturnsTwo:
    def test_medium_confidence_returns_two(self):
        """Scores [0.78, 0.70, 0.40] → gap=0.08 → 'medium', 2 skills"""
        raw = _make_raw_results([0.78, 0.70, 0.40])
        with (
            patch("rag.search_skills", return_value=raw),
            patch("rag.embed", return_value=[DUMMY_VEC]),
            patch("rag.USE_RERANKER", False),
            patch("rag.SKILL_ABSOLUTE_MIN_SCORE", 0.30),
            patch("rag.SKILL_RELATIVE_KEEP", 0.85),
            patch("rag.SKILL_HIGH_CONFIDENCE_GAP", 0.15),
            patch("rag.SKILL_MEDIUM_CONFIDENCE_GAP", 0.05),
            patch("rag.SKILL_MAX_RETURN", 3),
        ):
            import rag
            result = rag.get_skills_with_confidence("pubsub backlog")

        assert result.confidence == "medium"
        assert len(result.skills) == 2
        assert result.skills[0].name == "skill_0"
        assert result.skills[1].name == "skill_1"


class TestLowConfidenceReturnsAll:
    def test_low_confidence_returns_all(self):
        """Scores [0.65, 0.63, 0.62] → gap=0.02 < MEDIUM_GAP → 'low', SKILL_MAX_RETURN skills"""
        raw = _make_raw_results([0.65, 0.63, 0.62])
        with (
            patch("rag.search_skills", return_value=raw),
            patch("rag.embed", return_value=[DUMMY_VEC]),
            patch("rag.USE_RERANKER", False),
            patch("rag.SKILL_ABSOLUTE_MIN_SCORE", 0.30),
            patch("rag.SKILL_RELATIVE_KEEP", 0.85),
            patch("rag.SKILL_HIGH_CONFIDENCE_GAP", 0.15),
            patch("rag.SKILL_MEDIUM_CONFIDENCE_GAP", 0.05),
            patch("rag.SKILL_MAX_RETURN", 3),
        ):
            import rag
            result = rag.get_skills_with_confidence("incident report")

        assert result.confidence == "low"
        assert len(result.skills) == 3


class TestRelativeCutoffApplies:
    def test_relative_cutoff_applies(self):
        """
        Scores [0.80, 0.78, 0.50, 0.40] with SKILL_RELATIVE_KEEP=0.85
        → cutoff = max(0.80 * 0.85, 0.30) = 0.68
        → only 0.80 and 0.78 pass (0.50 < 0.68)
        → 2 skills returned
        """
        raw = _make_raw_results([0.80, 0.78, 0.50, 0.40])
        with (
            patch("rag.search_skills", return_value=raw),
            patch("rag.embed", return_value=[DUMMY_VEC]),
            patch("rag.USE_RERANKER", False),
            patch("rag.SKILL_ABSOLUTE_MIN_SCORE", 0.30),
            patch("rag.SKILL_RELATIVE_KEEP", 0.85),
            patch("rag.SKILL_HIGH_CONFIDENCE_GAP", 0.15),
            patch("rag.SKILL_MEDIUM_CONFIDENCE_GAP", 0.05),
            patch("rag.SKILL_MAX_RETURN", 3),
        ):
            import rag
            result = rag.get_skills_with_confidence("slo burn rate")

        # 0.80 and 0.78 survive the cutoff; gap = 0.02 < MEDIUM → low confidence
        assert len(result.skills) == 2
        assert result.skills[0].score == 0.80
        assert result.skills[1].score == 0.78


class TestBackwardCompatShim:
    def test_backward_compat_shim_with_match(self):
        """get_skill() returns '[skill: NAME]\n\nCONTENT' for a matching skill."""
        raw = _make_raw_results([0.85, 0.40])
        with (
            patch("rag.search_skills", return_value=raw),
            patch("rag.embed", return_value=[DUMMY_VEC]),
            patch("rag.USE_RERANKER", False),
            patch("rag.SKILL_ABSOLUTE_MIN_SCORE", 0.30),
            patch("rag.SKILL_RELATIVE_KEEP", 0.85),
            patch("rag.SKILL_HIGH_CONFIDENCE_GAP", 0.15),
            patch("rag.SKILL_MEDIUM_CONFIDENCE_GAP", 0.05),
            patch("rag.SKILL_MAX_RETURN", 3),
        ):
            import rag
            result = rag.get_skill("any query")

        assert result.startswith("[skill: skill_0]")
        assert "Content of skill 0" in result

    def test_backward_compat_shim_no_match(self):
        """get_skill() returns '' when no skills are retrieved."""
        with (
            patch("rag.search_skills", return_value=[]),
            patch("rag.embed", return_value=[DUMMY_VEC]),
            patch("rag.USE_RERANKER", False),
        ):
            import rag
            result = rag.get_skill("any query")

        assert result == ""
