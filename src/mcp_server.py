"""
Servidor MCP do local-rag com reranking.

Expõe o banco de skills/documents do projeto como tools MCP, pra que clientes
externos (Claude Desktop, Cursor, Cowork...) possam consultar o acervo
pessoal sem precisar rodar o chat.py local.

Fluxo interno: embed -> busca vetorial top-N -> rerank cross-encoder -> top-K.

Uso direto:
    uv run --with-requirements requirements.txt python src/mcp_server.py

Stdin/stdout falam o protocolo MCP via JSON-RPC. Não rode interativamente —
o cliente MCP que conecta nele é quem dispara as chamadas.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from db import search_skills, search_documents
from embeddings import embed
from reranker import rerank
from config import (
    TOP_K_RETRIEVE_DOCS,
    TOP_K_RETRIEVE_SKILLS,
    USE_RERANKER,
)

# Thresholds pós-rerank (escala sigmoid 0-1).
# CONFIDENT  >= 0.70 → retorna a skill directamente
# CANDIDATE  >= 0.45 → lista candidatos e pede ao utilizador para escolher
# REJECTED   <  0.45 → não encontrou nada relevante
SKILL_MIN_SCORE_RERANK = 0.70
SKILL_CANDIDATE_SCORE  = 0.45
DOC_MIN_SCORE_RERANK   = 0.30

# Fallback se USE_RERANKER=0.
SKILL_MIN_SCORE_COS = 0.72
SKILL_CANDIDATE_COS = 0.55
DOC_MIN_SCORE_COS   = 0.40

mcp = FastMCP("local-rag")


@mcp.tool()
def search_skills_tool(query: str, k: int = 3) -> str:
    """
    Busca skills (runbooks pessoais do Marcelo) relevantes para a query.

    Use esta tool SEMPRE que a pergunta envolver um procedimento técnico
    que ele possa ter documentado: git, MySQL, GCP DMS, Jira updates,
    ou qualquer fluxo recorrente de trabalho dele.

    Retorna até `k` skills com nome, descrição, corpo em markdown e score.
    O retrieval usa busca vetorial + reranking cross-encoder pra precisão.
    Se nada passar do threshold, retorna mensagem indicando que nada
    relevante foi encontrado — neste caso, NÃO invente um procedimento.
    """
    qvec = embed([query])[0]
    raw = search_skills(qvec, max(k, TOP_K_RETRIEVE_SKILLS))

    if USE_RERANKER and raw:
        skill_texts = [f"{s[0]}\n{s[1]}\n{s[2][:2000]}" for s in raw]
        ranked = rerank(query, skill_texts, top_k=k)
        confident, candidates = [], []
        for orig_idx, rscore in ranked:
            name, desc, body, _ = raw[orig_idx]
            if rscore >= SKILL_MIN_SCORE_RERANK:
                confident.append((name, desc, body, rscore))
            elif rscore >= SKILL_CANDIDATE_SCORE:
                candidates.append((name, desc, rscore))
    else:
        all_rows = raw[:k]
        confident  = [(n, d, b, s) for n, d, b, s in all_rows if s >= SKILL_MIN_SCORE_COS]
        candidates = [(n, d, s)    for n, d, b, s in all_rows if SKILL_CANDIDATE_COS <= s < SKILL_MIN_SCORE_COS]

    if confident:
        parts = []
        for name, description, body, score in confident:
            parts.append(
                f"### Skill: {name}  (score {score:.2f})\n"
                f"_{description}_\n\n"
                f"{body}"
            )
        return "\n\n---\n\n".join(parts)

    if candidates:
        lines = [
            "Confiança insuficiente para seleccionar a skill automaticamente.",
            "Encontrei estas candidatas — qual delas é a mais relevante para a tua pergunta?\n",
        ]
        for i, (name, desc, score) in enumerate(candidates, 1):
            lines.append(f"{i}. **{name}** (score {score:.2f}) — {desc}")
        lines.append(
            "\nResponde com o número ou o nome da skill e repetirei a busca com ela."
        )
        return "\n".join(lines)

    return (
        "Nenhuma skill relevante encontrada para esta query. "
        "Não invente um procedimento — pergunta ao utilizador se quer criar uma skill para este caso."
    )


@mcp.tool()
def search_documents_tool(query: str, k: int = 5) -> str:
    """
    Busca trechos relevantes nas notas e código pessoal do Marcelo.

    Use quando a pergunta envolver conteúdo específico do projeto, notas
    técnicas, decisões de arquitetura ou trechos de código que ele tenha
    indexado em ./data. Retorna até `k` chunks com nome do arquivo de
    origem, índice do chunk, conteúdo e score.

    Pipeline: busca vetorial top-20 -> reranker cross-encoder -> top-k.
    """
    qvec = embed([query])[0]
    raw = search_documents(qvec, max(k, TOP_K_RETRIEVE_DOCS))

    if USE_RERANKER and raw:
        doc_texts = [d[2] for d in raw]
        ranked = rerank(query, doc_texts, top_k=k)
        rows = []
        for orig_idx, rscore in ranked:
            if rscore < DOC_MIN_SCORE_RERANK:
                continue
            src, idx, content, _ = raw[orig_idx]
            rows.append((src, idx, content, rscore))
    else:
        rows = [r for r in raw[:k] if r[3] >= DOC_MIN_SCORE_COS]

    if not rows:
        return "Nenhum trecho relevante encontrado nas notas/código indexados."

    parts = []
    for source, chunk_index, content, score in rows:
        parts.append(
            f"### {source}  (chunk {chunk_index}, score {score:.2f})\n"
            f"```\n{content}\n```"
        )
    return "\n\n".join(parts)


@mcp.tool()
def list_skills() -> str:
    """
    Lista todas as skills disponíveis no banco do Marcelo, com nome e
    descrição. Útil quando o usuário pergunta 'o que você sabe fazer'
    ou 'quais skills estão disponíveis'.
    """
    from db import connect

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT name, description FROM skills ORDER BY name")
        rows = cur.fetchall()

    if not rows:
        return "Nenhuma skill indexada."

    lines = ["Skills disponíveis no banco do Marcelo:\n"]
    for name, description in rows:
        lines.append(f"- **{name}**: {description}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
