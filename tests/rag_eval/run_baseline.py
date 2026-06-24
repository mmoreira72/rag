"""
Retrieval baseline regression test — Se Liga (production `seliga` branch).

Calls the SAME functions production uses:
  - skill path (default): seliga_graph._get_skill — embed -> search_skills ->
    rerank top-1 -> cut at SKILL_MIN_SCORE=0.65
  - doc path (`kind: doc`): seliga_graph._get_doc — mirrors buscar_documentos
    (embed -> search_documents -> rerank top -> cut at DOC_MIN_SCORE=0.55,
    authenticated path). For doc cases expected_top_1 is a SOURCE SUBSTRING.

For each query in dataset.yaml it checks:

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
    if _p and (Path(_p) / "seliga_graph.py").exists():
        sys.path.insert(0, _p)
        break

# Chama a MESMA função que o grafo de produção usa (seliga_ask -> seliga_graph.
# identificar_skill -> _get_skill), para o eval nunca divergir do roteamento real.
from seliga_graph import _get_skill, _get_doc  # noqa: E402

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
    """Run one case. Returns (failures, observed).

    `kind: doc` roda o caminho de documentos (_get_doc, corte 0.55); qualquer
    outro valor (default `skill`) roda o caminho de skill (_get_skill, corte 0.65).
    """
    is_doc = case.get("kind") == "doc"
    cut = 0.55 if is_doc else 0.65
    name, _body, score = (_get_doc if is_doc else _get_skill)(case["query"])
    grounded = name is not None
    observed = {
        "top_1_name":  name,
        "top_1_score": float(score) if grounded else None,
        "grounded":    grounded,
    }

    failures: list[str] = []

    if case.get("expected_none", False):
        if grounded:
            failures.append(
                f"out-of-scope matched {observed['top_1_name']!r} "
                f"(score {observed['top_1_score']:.3f} >= {cut})"
            )
        return failures, observed

    # in-scope: um skill/doc tem de ser retornado
    if not grounded:
        kind = "doc" if is_doc else "skill"
        failures.append(f"expected a {kind} but retrieval was ungrounded (nothing cleared {cut})")
        return failures, observed

    # Para doc, expected_top_1 é um trecho do source (ex: 'missoes-desafios');
    # para skill, é o nome exato da skill.
    expected_top_1 = case.get("expected_top_1")
    if expected_top_1:
        if is_doc:
            if expected_top_1.lower() not in (observed["top_1_name"] or "").lower():
                failures.append(
                    f"top_1 doc mismatch: expected source containing {expected_top_1!r}, "
                    f"got {observed['top_1_name']!r}"
                )
        elif observed["top_1_name"] != expected_top_1:
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
    passed = failed = gaps = 0
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
            got = observed["top_1_name"] or ""
            mismatch = (exp.lower() not in got.lower()) if case.get("kind") == "doc" \
                else (observed["top_1_name"] != exp)
            if exp and not case.get("expected_none") and mismatch:
                disagreements.append(
                    f"  [{cid}] {case['query']!r}\n"
                    f"        intended {exp!r} but index routes to "
                    f"{observed['top_1_name']!r} ({observed['top_1_score']})"
                )

        if case_failures:
            if case.get("known_gap"):
                # Lacuna conhecida: registra mas NÃO bloqueia o gate.
                gaps += 1
                print(f"~ [{cid}] {case['query']}  (lacuna conhecida)")
                for f in case_failures:
                    print(f"      -> {f}")
            else:
                failed += 1
                print(f"X [{cid}] {case['query']}")
                for f in case_failures:
                    print(f"      -> {f}")
        else:
            passed += 1
            print(f"OK [{cid}] {case['query']}")

    print(f"\n{passed}/{len(cases)} passed"
          + (f"  ({failed} failed)" if failed else "")
          + (f"  [{gaps} lacuna(s) conhecida(s)]" if gaps else ""))

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
