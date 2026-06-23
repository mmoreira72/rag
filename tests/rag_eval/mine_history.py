"""
Read-only miner: extract real (skill_name, mensagem_participante, skill_score)
triples from the LangGraph checkpoint history, to seed the retrieval eval dataset.

Runs inside the local-rag-worker container (has langgraph + db access).
Prints TSV to stdout: skill_name <TAB> score <TAB> mensagem
"""
from __future__ import annotations

import sys
from config import PG_DSN

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except Exception as e:  # pragma: no cover
    print(f"IMPORT_ERROR\t{e}", file=sys.stderr)
    raise

seen: set[tuple[str, str]] = set()
rows: list[tuple[str, float, str]] = []

with PostgresSaver.from_conn_string(PG_DSN) as saver:
    for tup in saver.list(None):
        cv = (tup.checkpoint or {}).get("channel_values", {}) or {}
        skill = (cv.get("skill_name") or "").strip()
        msg = (cv.get("mensagem_participante") or "").strip()
        score = cv.get("skill_score") or 0.0
        if not skill or not msg:
            continue
        msg_one = " ".join(msg.split())
        key = (skill, msg_one.lower())
        if key in seen:
            continue
        seen.add(key)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        rows.append((skill, score, msg_one))

rows.sort(key=lambda r: (r[0], -r[1]))
for skill, score, msg in rows:
    print(f"{skill}\t{score:.3f}\t{msg[:200]}")

print(f"\n# {len(rows)} unique (skill, message) pairs", file=sys.stderr)
