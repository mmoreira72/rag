"""
Retrieval baseline regression test — Se Liga (production `seliga` branch).

Calls the SAME function production uses, chat._retrieve_and_rerank, so the test
mirrors the live routing path (embed → search_skills → rerank → cut at
SKILL_MIN_SCORE_RERANK=0.65). For each query in dataset.yaml it checks:

  1. top_1 match  — skill returned at position 1 matches expected_top_1
  2. grounded     — a skill actually cleared the 0.65 threshold (not ungrounded)
  3. score drop   — top-1 rerank score didn't drop more than tolerance below baseline
  4. out-of-scope — for expected_none=true, NO skill may clear the threshold

This is a REGRESSION gate: run --update-baseline once against a known-good index
to snapshot current behavior, then run plain after each reindex to detect drift.

Usage (inside local-rag-worker):
    python run_baseline.py                 # check against snapshot, exit 1 on drift
    python run_baseline.py --update-baseline   # refresh observed_* snapshot
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# Locate the src/ dir that holds chat.py (RAG_SRC env, else common locations).
_CANDIDATES = [
    os.getenv("RAG_SRC", ""),
    str(Path(__file__).resolve().parent.parent.parent / "src"),  # repo: tests/rag_eval/ -> src/
    "/app/src",                                                   # worker container
]
for _p in _CANDIDATES:
    if _p and (Path(_p) / "chat.py").exists():
        sys.path.insert(0, _p)
        break

from chat import _retrieve_and_rerank  # noqa: E402

DATASET_PATH = Path(os.getenv("EVAL_DATASET", Path(__file__).parent / "dataset.yaml"))
DEFAULT_SCORE_DROP_TOLERANCE = 0.05


def _load() -> dict:
    return yaml.safe_load(DATASET_PATH.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    DATASET_PATH.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def run_case(case: dict) -> tuple[list[str], dict]:
    """Run one case. Returns (failures, observed)."""
    skills, _docs = _retrieve_and_rerank(case["query"], case.get("autenticado", True))
    top = skills[0] if skills else None
    observed = {
        "top_1_name":  top[0] if top else None,
        "top_1_score": float(top[3]) if top else None,
        "grounded":    bool(skills),
    }

    failures: list[str] = []

    if case.get("expected_none", False):
        if top is not None:
            failures.append(
                f"out-of-scope matched {observed['top_1_name']!r} "
                f"(score {observed['top_1_score']:.3f} >= 0.65)"
            )
        return failures, observed

    # in-scope: a skill must be returned
    if top is None:
        failures.append("expected a skill but retrieval was ungrounded (nothing cleared 0.65)")
        return failures, observed

    expected_top_1 = case.get("expected_top_1")
    if expected_top_1 and observed["top_1_name"] != expected_top_1:
        failures.append(
            f"top_1 mismatch: expected {expected_top_1!r}, got {observed['top_1_name']!r}"
        )

    baseline = case.get("observed_top1_score")
    actual = observed["top_1_score"]
    tol = case.get("tolerance_score_drop", DEFAULT_SCORE_DROP_TOLERANCE)
    if baseline is not None and actual is not None:
        drop = baseline - actual
        if drop > tol:
            failures.append(
                f"score drop {drop:.3f} exceeds tolerance {tol:.3f} "
                f"(baseline {baseline:.3f} -> observed {actual:.3f})"
            )

    return failures, observed


def main() -> None:
    parser = argparse.ArgumentParser(description="Se Liga retrieval baseline regression test")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Refresh observed_top1_score / observed_top1_name in dataset.yaml")
    args = parser.parse_args()

    data = _load()
    cases = data.get("retrieval_cases", [])
    passed = failed = 0
    disagreements: list[str] = []

    for case in cases:
        cid = case.get("id", "?")
        case_failures, observed = run_case(case)

        if args.update_baseline:
            if observed["top_1_score"] is not None:
                case["observed_top1_score"] = round(observed["top_1_score"], 3)
            case["observed_top1_name"] = observed["top_1_name"]
            # Flag where current routing disagrees with the intended label (review, not block)
            exp = case.get("expected_top_1")
            if exp and not case.get("expected_none") and observed["top_1_name"] != exp:
                disagreements.append(
                    f"  [{cid}] {case['query']!r}\n"
                    f"        intended {exp!r} but index routes to "
                    f"{observed['top_1_name']!r} ({observed['top_1_score']})"
                )

        if case_failures:
            failed += 1
            print(f"X [{cid}] {case['query']}")
            for f in case_failures:
                print(f"      -> {f}")
        else:
            passed += 1
            print(f"OK [{cid}] {case['query']}")

    print(f"\n{passed}/{len(cases)} passed" + (f"  ({failed} failed)" if failed else ""))

    if args.update_baseline:
        _save(data)
        print(f"\nBaseline updated: {DATASET_PATH}")
        if disagreements:
            print(f"\n{len(disagreements)} case(s) where index routing != intended label "
                  f"(review these — real mis-route or ambiguous label):")
            print("\n".join(disagreements))

    if failed and not args.update_baseline:
        sys.exit(1)


if __name__ == "__main__":
    main()
