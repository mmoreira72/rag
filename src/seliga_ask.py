"""
Interface entre o worker e o seliga_graph LangGraph.

Substitui chat.ask() para o fluxo Se Liga — usa PostgresSaver como
checkpointer para que o estado do interrupt() sobreviva entre jobs do worker.

Lógica por turno:
  - Se há interrupt pendente para o session_id → resume com Command(resume=message)
  - Caso contrário → inicia nova invocação do grafo
"""
from __future__ import annotations

import os
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

# Langfuse callback — activo apenas se LANGFUSE_HOST estiver definido no ambiente.
_langfuse_handler = None
if os.getenv("LANGFUSE_HOST"):
    try:
        from langfuse.callback import CallbackHandler
        _langfuse_handler = CallbackHandler(
            host=os.environ["LANGFUSE_HOST"],
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        )
    except ImportError:
        pass


def ask(session_id: str, message: str, autenticado: bool = True) -> dict:
    callbacks = [_langfuse_handler] if _langfuse_handler else []
    config = {"configurable": {"thread_id": session_id}, "callbacks": callbacks}

    state = _graph.get_state({"configurable": {"thread_id": session_id}})
    is_new = not state.next  # nova pergunta (não é resposta a um interrupt)
    if state.next:
        # Grafo pausado em interrupt — participante está respondendo a um campo
        result = _graph.invoke(Command(resume=message), config)
    else:
        # Query augmentation apenas para retrieval — mensagem_participante fica original.
        # Não augmenta se a mensagem actual for muito similar à anterior (evita query duplicada).
        import difflib
        last_msg = state.values.get("mensagem_participante", "") if state.values else ""
        similarity = difflib.SequenceMatcher(None, last_msg.lower(), message.lower()).ratio()
        rag_query = f"{last_msg} {message}".strip() if last_msg and similarity < 0.7 else message

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
            "docs_sources":      [],
            "resposta_final":    "",
            "rota":              "coletar",
        }
        result = _graph.invoke(initial, config)

    # Verifica se o grafo pausou em interrupt após o invoke
    state_after = _graph.get_state(config)
    if state_after.next and state_after.tasks:
        interrupts = state_after.tasks[0].interrupts
        answer = interrupts[0].value if interrupts else ""
    else:
        answer = (result or {}).get("resposta_final", "")

    skill_name  = (result or {}).get("skill_name", "")
    skill_score = (result or {}).get("skill_score", 0.0)
    if skill_name:
        sources = [{"type": "skill", "name": skill_name, "score": skill_score}]
    else:
        sources = (result or {}).get("docs_sources", [])

    # Log da interação (só perguntas novas) para reavaliação semanal.
    # grounded considera AMBOS os caminhos do grafo: skill OU documento (FAQ).
    # Best-effort: nunca quebra o atendimento.
    if is_new:
        try:
            from db import insert_interacao
            docs = (result or {}).get("docs_sources") or []
            if skill_name:
                top1, sc, grounded = skill_name, float(skill_score), True
            elif docs:
                top1 = f"doc:{docs[0].get('source', '')}"
                sc = float(docs[0].get("score") or 0.0)
                grounded = True
            else:
                top1, sc, grounded = None, None, False
            insert_interacao(session_id, message, top1, sc, grounded)
        except Exception as e:  # noqa: BLE001
            print(f"[seliga_ask] falha ao logar interacao: {e}", flush=True)

    return {"answer": answer, "sources": sources}
