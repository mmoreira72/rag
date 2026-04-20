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
SKILL_MIN_SCORE_RERANK = 0.50
DOC_MIN_SCORE_RERANK = 0.30

# Fallback se USE_RERANKER=0.
SKILL_MIN_SCORE_COS = 0.65
DOC_MIN_SCORE_COS = 0.40

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
        rows = []
        for orig_idx, rscore in ranked:
            if rscore < SKILL_MIN_SCORE_RERANK:
                continue
            name, desc, body, _ = raw[orig_idx]
            rows.append((name, desc, body, rscore))
    else:
        rows = [r for r in raw[:k] if r[3] >= SKILL_MIN_SCORE_COS]

    if not rows:
        return (
            "Nenhuma skill relevante encontrada no banco do Marcelo "
            "para esta query. Não invente um procedimento — informe "
            "que não há skill registrada e pergunte se ele quer criar uma."
        )

    parts = []
    for name, description, body, score in rows:
        parts.append(
            f"### Skill: {name}  (score {score:.2f})\n"
            f"_{description}_\n\n"
            f"{body}"
        )
    return "\n\n---\n\n".join(parts)


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
