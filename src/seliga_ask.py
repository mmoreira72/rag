"""
Interface entre o worker e o seliga_graph LangGraph.

Substitui chat.ask() para o fluxo Se Liga — usa PostgresSaver como
checkpointer para que o estado do interrupt() sobreviva entre jobs do worker.

Lógica por turno:
  - Se há interrupt pendente para o session_id → resume com Command(resume=message)
  - Caso contrário → inicia nova invocação do grafo
"""
from __future__ import annotations

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from config import PG_DSN
from seliga_graph import build_graph, SeligaState

# Grafo compilado uma vez, compartilhado entre jobs do worker.
# PostgresSaver é thread-safe para leitura/escrita concorrente.
_checkpointer = PostgresSaver(psycopg.connect(PG_DSN, autocommit=True))
_checkpointer.setup()  # cria tabelas de checkpoint se não existirem
_graph = build_graph(_checkpointer)


def ask(session_id: str, message: str, autenticado: bool = True) -> dict:
    config = {"configurable": {"thread_id": session_id}}

    state = _graph.get_state(config)
    if state.next:
        # Grafo pausado em interrupt — participante está respondendo a um campo
        result = _graph.invoke(Command(resume=message), config)
    else:
        # Nova conversa ou fluxo anterior já encerrou
        initial: SeligaState = {
            "numero_atendimento":    session_id,
            "mensagem_participante": message,
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
            "resposta_final":    "",
            "rota":              "coletar",
        }
        result = _graph.invoke(initial, config)

    answer = (result or {}).get("resposta_final", "")
    skill_name = (result or {}).get("skill_name", "")

    sources = [{"type": "skill", "name": skill_name}] if skill_name else []
    return {"answer": answer, "sources": sources}
