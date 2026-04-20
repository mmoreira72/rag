"""
Debug do retrieval em duas fases: vetorial + reranker.
Uso:
    python debug_retrieve.py "como configurar dms via ssh tunnel"
"""
import sys
from embeddings import embed
from reranker import rerank
from db import search_skills, search_documents
from config import (
    TOP_K_RETRIEVE_DOCS,
    TOP_K_RETRIEVE_SKILLS,
    USE_RERANKER,
)

if len(sys.argv) < 2:
    print("uso: python debug_retrieve.py 'sua pergunta'")
    sys.exit(1)

q = " ".join(sys.argv[1:])
print(f"query: {q}\n")

vec = embed([q])[0]
print(f"embedding dim: {len(vec)}\n")

# ─── SKILLS ───────────────────────────────────────────────
print(f"=== FASE 1: busca vetorial — TOP {TOP_K_RETRIEVE_SKILLS} SKILLS ===")
skills = search_skills(vec, TOP_K_RETRIEVE_SKILLS)
for name, desc, body, score in skills:
    print(f"  cos={score:.3f}  {name}")
    print(f"             {desc[:100]}")

if USE_RERANKER and skills:
    print(f"\n=== FASE 2: rerank — SKILLS reordenadas ===")
    skill_texts = [f"{s[0]}\n{s[1]}\n{s[2][:2000]}" for s in skills]
    reranked = rerank(q, skill_texts)
    for orig_idx, rscore in reranked:
        name, desc, body, cos = skills[orig_idx]
        print(f"  rerank={rscore:.3f}  (cos={cos:.3f})  {name}")

# ─── DOCS ─────────────────────────────────────────────────
print(f"\n=== FASE 1: busca vetorial — TOP {TOP_K_RETRIEVE_DOCS} DOCS ===")
docs = search_documents(vec, TOP_K_RETRIEVE_DOCS)
if not docs:
    print("  (nenhum doc indexado)")
for src, idx, content, score in docs[:10]:
    print(f"  cos={score:.3f}  {src} #chunk{idx}")

if USE_RERANKER and docs:
    print(f"\n=== FASE 2: rerank — DOCS reordenados (top 10) ===")
    doc_texts = [d[2] for d in docs]
    reranked = rerank(q, doc_texts, top_k=10)
    for orig_idx, rscore in reranked:
        src, idx, content, cos = docs[orig_idx]
        print(f"  rerank={rscore:.3f}  (cos={cos:.3f})  {src} #chunk{idx}")
        print(f"                     {content[:100].strip()}")
