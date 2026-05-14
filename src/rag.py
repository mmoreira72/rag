"""
RAG retrieval layer — multi-skill retrieval with confidence classification.

Public API:
    get_skills_with_confidence(query) -> SkillRetrievalResult
    format_skills_for_prompt(result)  -> str
    get_skill(query)                  -> str  # DEPRECATED shim
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Allow running from both repo root and src/ directly
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TOP_K_RETRIEVE_SKILLS,
    USE_RERANKER,
    SKILL_RETRIEVE_CANDIDATES,
    SKILL_MAX_RETURN,
    SKILL_ABSOLUTE_MIN_SCORE,
    SKILL_RELATIVE_KEEP,
    SKILL_HIGH_CONFIDENCE_GAP,
    SKILL_MEDIUM_CONFIDENCE_GAP,
)
from db import search_skills
from embeddings import embed
from reranker import rerank


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedSkill:
    name: str
    content: str
    score: float


@dataclass
class SkillRetrievalResult:
    skills: list[RetrievedSkill] = field(default_factory=list)
    confidence: Literal["high", "medium", "low", "none"] = "none"
    top_score: float = 0.0
    gap: float = 0.0
    candidate_count: int = 0


# ── Core retrieval ────────────────────────────────────────────────────────────

def get_skills_with_confidence(query: str) -> SkillRetrievalResult:
    """Retrieve skills and classify retrieval confidence.

    Algorithm:
    1. Embed query; fetch SKILL_RETRIEVE_CANDIDATES from pgvector.
    2. Rerank if USE_RERANKER, else use raw cosine scores.
    3. If top score < SKILL_ABSOLUTE_MIN_SCORE → confidence="none".
    4. Compute relative cutoff = max(top_score * SKILL_RELATIVE_KEEP,
                                     SKILL_ABSOLUTE_MIN_SCORE).
    5. Filter by cutoff; cap at SKILL_MAX_RETURN.
    6. Classify by gap between top-1 and top-2 scores:
       - gap >= SKILL_HIGH_CONFIDENCE_GAP   → "high"   (return 1 skill)
       - gap >= SKILL_MEDIUM_CONFIDENCE_GAP → "medium" (return ≤2 skills)
       - else                               → "low"    (return ≤SKILL_MAX_RETURN)
    7. Log to stderr.
    """
    # Step 1: embed + vector search
    qvec = embed([query])[0]
    raw = search_skills(qvec, SKILL_RETRIEVE_CANDIDATES)
    candidate_count = len(raw)

    if not raw:
        result = SkillRetrievalResult(
            skills=[], confidence="none", top_score=0.0,
            gap=0.0, candidate_count=0,
        )
        _log(query, 0, 0, "none", 0.0, 0.0)
        return result

    # Step 2: rerank or use raw scores
    # scored: list of (name, description, body, score) in descending score order
    if USE_RERANKER:
        texts = [f"{r[0]}\n{r[1]}\n{r[2][:1000]}" for r in raw]
        ranked_pairs = rerank(query, texts, top_k=None)
        # ranked_pairs: [(original_idx, score), ...] sorted by score desc
        scored = [(raw[idx][0], raw[idx][1], raw[idx][2], score)
                  for idx, score in ranked_pairs]
    else:
        # raw is already ordered by vector similarity desc
        scored = [(r[0], r[1], r[2], float(r[3])) for r in raw]

    if not scored:
        result = SkillRetrievalResult(
            skills=[], confidence="none", top_score=0.0,
            gap=0.0, candidate_count=candidate_count,
        )
        _log(query, candidate_count, 0, "none", 0.0, 0.0)
        return result

    top_score = scored[0][3]

    # Step 3: absolute floor
    if top_score < SKILL_ABSOLUTE_MIN_SCORE:
        result = SkillRetrievalResult(
            skills=[], confidence="none", top_score=top_score,
            gap=0.0, candidate_count=candidate_count,
        )
        _log(query, candidate_count, 0, "none", top_score, 0.0)
        return result

    # Step 4: relative cutoff
    cutoff = max(top_score * SKILL_RELATIVE_KEEP, SKILL_ABSOLUTE_MIN_SCORE)

    # Step 5: filter by cutoff, cap at SKILL_MAX_RETURN
    filtered = [s for s in scored if s[3] >= cutoff][:SKILL_MAX_RETURN]

    # Step 6: classify by gap
    second_score = filtered[1][3] if len(filtered) > 1 else 0.0
    gap = top_score - second_score

    if gap >= SKILL_HIGH_CONFIDENCE_GAP:
        confidence: Literal["high", "medium", "low", "none"] = "high"
        final = filtered[:1]
    elif gap >= SKILL_MEDIUM_CONFIDENCE_GAP:
        confidence = "medium"
        final = filtered[:2]
    else:
        confidence = "low"
        final = filtered[:SKILL_MAX_RETURN]

    skills = [RetrievedSkill(name=s[0], content=s[2], score=s[3]) for s in final]

    _log(query, candidate_count, len(skills), confidence, top_score, gap)

    return SkillRetrievalResult(
        skills=skills,
        confidence=confidence,
        top_score=top_score,
        gap=gap,
        candidate_count=candidate_count,
    )


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_skills_for_prompt(result: SkillRetrievalResult) -> str:
    """Format a SkillRetrievalResult as a string to inject into a system prompt.

    - confidence "high"  → single skill block, clean format.
    - confidence "medium"/"low" → warning header + multiple skills separated by ---.
    - confidence "none"  → empty string (caller should not inject anything).
    """
    if result.confidence == "none" or not result.skills:
        return ""

    if result.confidence == "high":
        sk = result.skills[0]
        return f"[skill: {sk.name}, relevance: {sk.score:.2f}]\n\n{sk.content}"

    # medium or low — multiple skills with warning
    header = (
        f"[WARNING: multiple potentially relevant skills retrieved "
        f"(confidence={result.confidence}). Review all before responding.]\n\n"
    )
    blocks = []
    for sk in result.skills:
        blocks.append(f"[skill: {sk.name}, relevance: {sk.score:.2f}]\n\n{sk.content}")
    return header + "\n\n---\n\n".join(blocks)


# ── Backward-compat shim ──────────────────────────────────────────────────────

def get_skill(query: str) -> str:
    """DEPRECATED: use get_skills_with_confidence() + format_skills_for_prompt().

    Returns the formatted top-skill string (legacy format) or "" if no match.
    Kept for backward compatibility with callers that have not yet been migrated.
    """
    result = get_skills_with_confidence(query)
    if not result.skills:
        return ""
    top = result.skills[0]
    return f"[skill: {top.name}]\n\n{top.content}"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log(query: str, candidates: int, returned: int,
         confidence: str, top_score: float, gap: float) -> None:
    short_query = query[:60].replace("\n", " ")
    print(
        f"  [skill_retrieval] query={short_query!r} candidates={candidates} "
        f"returned={returned} confidence={confidence} "
        f"top_score={top_score:.3f} gap={gap:.3f}",
        file=sys.stderr,
    )
