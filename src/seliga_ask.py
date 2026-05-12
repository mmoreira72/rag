"""
Interface entre o worker e o seliga_graph LangGraph.

Substitui chat.ask() para o fluxo Se Liga — usa PostgresSaver como
checkpointer para que o estado do interrupt() sobreviva entre jobs do worker.

Lógica por turno:
  - Se há interrupt pendente para o session_id → resume com Command(resume=message)
  - Caso contrário → inicia nova invocação do grafo

Após cada turno, grava uma linha em `decisoes` (observabilidade — falha aqui
NUNCA pode quebrar o atendimento).
"""
from __future__ import annotations

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from config import PG_DSN
from db import insert_decisao, update_feedback_proximo
from seliga_graph import build_graph, SeligaState, _resolucao_da_skill

# Lazy init — não abre conexão Postgres no import, pra que módulos que só
# precisam dos helpers puros (ex: _derive_rota nos testes) consigam importar
# sem o DB no ar. A primeira chamada de ask() materializa grafo+checkpointer.
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        checkpointer = PostgresSaver(psycopg.connect(PG_DSN, autocommit=True))
        checkpointer.setup()  # cria tabelas de checkpoint se não existirem
        _graph = build_graph(checkpointer)
    return _graph


def _derive_rota(values: dict, interrupt_disparado: bool) -> str:
    """Resume em uma string o caminho que o grafo seguiu.

    Convenções:
      - 'autonomo'    skill autonomo respondeu direto
      - 'guiado'      skill guiado orientou (pode ter resolvido ou caído pra escalacao)
      - 'escalacao'   gravou escalada no banco (escalada_id > 0)
      - 'documentos'  sem skill, respondeu com chunks do corpus
      - 'nada'        nem skill nem docs aplicaram
      - 'interrupt'   pausou aguardando dado do participante (estado intermediário)
    """
    if interrupt_disparado:
        return "interrupt"
    if values.get("escalada_id"):
        return "escalacao"
    skill_body = values.get("skill_body") or ""
    if values.get("skill_name") and skill_body:
        resolucao = _resolucao_da_skill(skill_body)
        if resolucao == "autonomo":
            return "autonomo"
        if resolucao == "guiado":
            return "guiado"
        # frontmatter dizia escalacao mas não gravou — provavelmente faltou
        # interrupt resolver ainda. Reporta como guiado-ou-coleta intermediária.
        return "escalacao"
    if values.get("docs_contexto"):
        return "documentos"
    return "nada"


def _log_decisao(
    session_id: str,
    mensagem_original: str,
    autenticado: bool,
    values: dict,
    interrupt_disparado: bool,
    answer: str,
) -> None:
    """Wrapper try/except amplo — observabilidade nunca quebra atendimento."""
    try:
        rota = _derive_rota(values, interrupt_disparado)
        skill_name = values.get("skill_name") or None
        skill_score = values.get("skill_score")
        # skill_score chega como float ou 0.0; só grava se houver tentativa real
        if skill_score is not None:
            try:
                skill_score = float(skill_score)
            except (TypeError, ValueError):
                skill_score = None
        docs_usados = values.get("docs_usados") or None
        escalada_id = values.get("escalada_id") or None
        resolveu = values.get("guiado_resolvido")
        # guiado_resolvido só faz sentido quando passou por tentar_resolver
        if rota not in ("guiado", "escalacao"):
            resolveu = None
        # numero_atendimento default = session_id (worker.py usa session_id como thread_id)
        numero_atendimento = values.get("numero_atendimento") or session_id

        _, turno_idx = insert_decisao(
            numero_atendimento=numero_atendimento,
            session_id=session_id,
            mensagem=mensagem_original,
            autenticado=autenticado,
            skill_name=skill_name,
            skill_score=skill_score,
            rota=rota,
            docs_usados=docs_usados,
            interrupt_disparado=interrupt_disparado,
            resposta_final=answer or None,
            escalada_id=int(escalada_id) if escalada_id else None,
            resolveu=resolveu,
        )
        # Anota no turno anterior a mensagem subsequente do participante (sinal negativo / follow-up)
        update_feedback_proximo(numero_atendimento, turno_idx - 1, mensagem_original)
    except Exception as exc:  # noqa: BLE001 — observabilidade, engole tudo
        print(f"[seliga_ask] WARN: falha ao gravar decisão: {exc}", flush=True)


def ask(session_id: str, message: str, autenticado: bool = True) -> dict:
    config = {"configurable": {"thread_id": session_id}}

    graph = _get_graph()
    state = graph.get_state(config)
    if state.next:
        # Grafo pausado em interrupt — participante está respondendo a um campo
        result = graph.invoke(Command(resume=message), config)
    else:
        # Query augmentation apenas para retrieval — mensagem_participante fica original
        # para o LLM responder a pergunta certa, não a anterior.
        last_msg = state.values.get("mensagem_participante", "") if state.values else ""
        rag_query = f"{last_msg} {message}".strip() if last_msg else message

        initial: SeligaState = {
            "numero_atendimento":    session_id,
            "mensagem_participante": message,   # original — LLM responde a esta
            "rag_query":             rag_query, # augmentada — usada apenas no retrieval
            "autenticado":           autenticado,
            "skill_name":   "",
            "skill_body":   "",
            "skill_score":  0.0,
            "dados":             {},
            "campos_faltando":   [],
            "resposta_participante": "",
            "guiado_resolvido":  False,
            "escalada_id":       0,
            "docs_contexto":     "",
            "docs_usados":       [],
            "resposta_final":    "",
            "rota":              "coletar",
        }
        result = graph.invoke(initial, config)

    # Verifica se o grafo pausou em interrupt após o invoke
    state_after = graph.get_state(config)
    interrupt_disparado = bool(state_after.next and state_after.tasks)
    if interrupt_disparado:
        interrupts = state_after.tasks[0].interrupts
        answer = interrupts[0].value if interrupts else ""
    else:
        answer = (result or {}).get("resposta_final", "")

    # Observabilidade — try/except interno, falha aqui não propaga
    _log_decisao(
        session_id=session_id,
        mensagem_original=message,
        autenticado=autenticado,
        values=result or state_after.values or {},
        interrupt_disparado=interrupt_disparado,
        answer=answer,
    )

    skill_name  = (result or {}).get("skill_name", "")
    skill_score = (result or {}).get("skill_score", 0.0)
    sources = [{"type": "skill", "name": skill_name, "score": skill_score}] if skill_name else []
    return {"answer": answer, "sources": sources}
