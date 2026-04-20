"""
CLI de chat com RAG + skills + reranker.
Uso: python chat.py

A função `ask(session_id, question)` é a interface pública usada pela API HTTP.
O histórico de cada sessão é persistido no PostgreSQL (tabela sessions).
"""
import time
from datetime import date
from embeddings import embed
from reranker import rerank
from llm_client import chat
from db import search_documents, search_skills, get_session_history, save_session_history
from config import (
    TOP_K_RETRIEVE_DOCS,
    TOP_K_RETRIEVE_SKILLS,
    TOP_K_DOCS,
    TOP_K_SKILLS,
    MAX_CONTEXT_CHARS,
    USE_RERANKER,
)

# Thresholds depois do rerank (escala sigmoid 0-1 do cross-encoder).
# IMPORTANTE: sigmoid(0) = 0.5 significa "no signal" — o reranker não viu
# nem match nem mismatch. Tudo em ~0.50 é ruído, não relevância.
# Match real começa em ~0.6+. Exemplo empírico: mysql-perf pontuou 0.684
# quando a pergunta era sobre queries pesadas; pontuou 0.500 quando era
# "capital do brasil" (ruído → falso positivo com threshold antigo).
SKILL_MIN_SCORE_RERANK = 0.65
DOC_MIN_SCORE_RERANK = 0.50

# Fallback: se reranker desligado, usa thresholds de cosine (escala diferente).
SKILL_MIN_SCORE_COS = 0.65
DOC_MIN_SCORE_COS = 0.55

def _system_grounded(context: str) -> str:
    today = date.today().strftime("%d/%m/%Y")
    return (
        f"Data de hoje: {today}. Use esta data como referência para qualquer cálculo de prazo.\n\n"
        "Você é um atendente virtual do programa Se Liga na SKY. "
        "Responda ao participante de forma cordial, direta e conversacional — como um atendente humano faria pelo WhatsApp.\n"
        "As SKILLS e o CONTEXTO abaixo foram recuperados automaticamente e já estão filtrados por relevância. Use-os como base de conhecimento.\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. Use a SKILL presente como guia do que fazer e dizer — mas escreva a resposta como uma mensagem natural para o participante, "
        "não liste os passos da skill nem repita o conteúdo interno dela.\n"
        "2. Use o CONTEXTO para complementar com informações do regulamento, FAQ ou outros documentos quando relevante.\n"
        "3. Seja objetivo: responda o que o participante perguntou, sem introduções longas.\n"
        "4. Tom: amigável, como a atendente Simone nos exemplos reais — próximo, mas profissional.\n"
        "5. Responda sempre em português do Brasil.\n\n"
        + context
    )


def _system_ungrounded() -> str:
    today = date.today().strftime("%d/%m/%Y")
    return (
        f"Data de hoje: {today}.\n\n"
        "Você é um atendente virtual do programa Se Liga na SKY. "
        "Não foi encontrada nenhuma skill ou contexto relevante para esta pergunta na base de conhecimento.\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. INICIE a resposta com: ⚠️ Não encontrei informações específicas sobre isso na minha base de conhecimento.\n"
        "2. Se conseguir ajudar com conhecimento geral, faça isso de forma breve.\n"
        "3. Oriente o participante a entrar em contato com a equipe de atendimento se a dúvida persistir.\n"
        "4. Responda sempre em português do Brasil."
    )


def _retrieve_and_rerank(question: str):
    """
    Fluxo de duas fases:
    1. Busca vetorial (rápida, recall alto) traz TOP_K_RETRIEVE candidatos.
    2. Reranker (cross-encoder) reordena e corta em TOP_K final.

    Retorna (skills, docs) — cada item é tupla com score final no último slot.
    Se USE_RERANKER=0, usa só cosine com thresholds maiores.
    """
    t0 = time.perf_counter()
    qvec = embed([question])[0]
    t_embed = time.perf_counter() - t0

    t1 = time.perf_counter()
    raw_skills = search_skills(qvec, TOP_K_RETRIEVE_SKILLS)
    raw_docs = search_documents(qvec, TOP_K_RETRIEVE_DOCS)
    t_search = time.perf_counter() - t1

    if not USE_RERANKER:
        skills = [s for s in raw_skills[:TOP_K_SKILLS] if s[3] >= SKILL_MIN_SCORE_COS]
        docs = [d for d in raw_docs[:TOP_K_DOCS] if d[3] >= DOC_MIN_SCORE_COS]
        print(f"  ⏱  embed={t_embed:.2f}s  search={t_search:.2f}s")
        return skills, docs

    t2 = time.perf_counter()
    # Rerank skills (texto = name + description + corpo truncado)
    skills = []
    if raw_skills:
        skill_texts = [f"{s[0]}\n{s[1]}\n{s[2][:2000]}" for s in raw_skills]
        ranked = rerank(question, skill_texts, top_k=TOP_K_SKILLS)
        for orig_idx, rscore in ranked:
            if rscore < SKILL_MIN_SCORE_RERANK:
                continue
            name, desc, body, _ = raw_skills[orig_idx]
            skills.append((name, desc, body, rscore))

    # Rerank docs
    docs = []
    if raw_docs:
        doc_texts = [d[2] for d in raw_docs]
        ranked = rerank(question, doc_texts, top_k=TOP_K_DOCS)
        for orig_idx, rscore in ranked:
            if rscore < DOC_MIN_SCORE_RERANK:
                continue
            src, idx, content, _ = raw_docs[orig_idx]
            docs.append((src, idx, content, rscore))
    t_rerank = time.perf_counter() - t2

    print(f"  ⏱  embed={t_embed:.2f}s  search={t_search:.2f}s  rerank={t_rerank:.2f}s")
    return skills, docs


def build_context(question: str):
    skills, docs = _retrieve_and_rerank(question)

    parts = []
    total_chars = 0

    if skills:
        parts.append("### SKILLS ATIVAS")
        for name, desc, body, score in skills:
            max_body = MAX_CONTEXT_CHARS // 2
            if len(body) > max_body:
                body = body[:max_body] + "\n\n[...skill truncada por limite de contexto]"
            entry = f"[{name}] (score {score:.2f}) — {desc}\n{body}"
            total_chars += len(entry)
            parts.append(entry)

    if docs:
        parts.append("\n### CONTEXTO (trechos relevantes)")
        for src, idx, content, score in docs:
            entry = f"--- {src} #chunk{idx} (score {score:.2f}) ---\n{content}"
            if total_chars + len(entry) > MAX_CONTEXT_CHARS:
                parts.append("[...chunks restantes omitidos por limite de contexto]")
                break
            total_chars += len(entry)
            parts.append(entry)

    return "\n\n".join(parts), docs, skills


def _build_messages(question: str, history: list) -> tuple[list, bool]:
    """Monta a lista de messages para a LLM, retorna (messages, grounded)."""
    context, docs, skills = build_context(question)
    grounded = bool(skills) or bool(docs)
    system = _system_grounded(context) if grounded else _system_ungrounded()
    messages = [{"role": "system", "content": system}] + history + [
        {"role": "user", "content": question}
    ]
    return messages, grounded


def ask(session_id: str, question: str) -> str:
    """
    Interface pública para a API HTTP.

    Busca o histórico da sessão no Postgres, chama a LLM com contexto RAG
    e persiste o histórico atualizado. Cada session_id é isolado — múltiplos
    usuários simultâneos não interferem entre si.

    TTL da sessão: SESSION_TTL_MINUTES (default 30 min de inatividade).
    """
    history = get_session_history(session_id)
    messages, _ = _build_messages(question, history)

    response = chat(messages, stream=False)
    answer = response.choices[0].message.content or ""

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    save_session_history(session_id, history[-4:])

    return answer


def main():
    history = []
    mode = "rerank" if USE_RERANKER else "cosine-only"
    print(f"Chat local ({mode}, Ctrl+C pra sair)\n")
    while True:
        try:
            q = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue

        context, docs, skills = build_context(q)
        grounded = bool(skills) or bool(docs)

        if skills:
            for name, desc, body, score in skills:
                print(f"  📎 skill: {name} (score {score:.3f})")
        if docs:
            for src, idx, content, score in docs:
                print(f"  📄 doc: {src}#chunk{idx} (score {score:.3f})")
        if not skills and not docs:
            print("  ⚠️  nada passou do threshold — resposta livre do modelo")
        print()

        if grounded:
            system = _system_grounded(context)
        else:
            system = _system_ungrounded()

        messages = [{"role": "system", "content": system}] + history + [
            {"role": "user", "content": q}
        ]

        print("assistente> ", end="", flush=True)
        answer = ""
        t_llm = time.perf_counter()
        stream = chat(messages, stream=True)
        first_token = None
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta and first_token is None:
                first_token = time.perf_counter() - t_llm
            print(delta, end="", flush=True)
            answer += delta
        t_total_llm = time.perf_counter() - t_llm
        print(f"\n\n  ⏱  llm: first_token={first_token:.2f}s  total={t_total_llm:.2f}s  ({len(answer)} chars)\n")

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        history[:] = history[-4:]


if __name__ == "__main__":
    main()
