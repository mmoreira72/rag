"""
Worker de processamento de jobs do chat.

Roda como processo separado em container próprio.
Consome jobs da fila Redis, executa o LangGraph do Se Liga e grava o resultado.

O grafo usa PostgresSaver como checkpointer — o estado de cada sessão
(incluindo interrupts para coleta de ICARE/dados) sobrevive a reboots do worker.

Uso:
    python worker.py

Sinais:
    SIGTERM / SIGINT — encerra o loop após o job corrente terminar (graceful shutdown).
"""
import json
import signal
import sys
from datetime import datetime, timezone

from job_queue import dequeue, get_job, update_job
from config import PG_DSN

# Grafo e pool são inicializados dentro de main() para que erros
# de importação ou conexão apareçam nos logs antes de qualquer job.
_pool  = None
_graph = None


def _init_graph():
    global _pool, _graph
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver
    from seliga_graph import build_graph

    print("[worker] conectando ao Postgres...", flush=True)
    _pool = ConnectionPool(conninfo=PG_DSN, max_size=5)
    _pool.wait()   # aguarda conexões estarem prontas

    print("[worker] configurando checkpointer LangGraph...", flush=True)
    checkpointer = PostgresSaver(_pool)
    checkpointer.setup()   # cria tabelas do checkpointer se não existirem

    print("[worker] compilando grafo...", flush=True)
    _graph = build_graph(checkpointer)
    print("[worker] grafo pronto.", flush=True)


def _graph_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _is_interrupted(session_id: str) -> tuple[bool, str]:
    """Retorna (está_interrompido, pergunta_do_interrupt)."""
    state = _graph.get_state(_graph_config(session_id))
    if not state.next:
        return False, ""
    interrupts = state.tasks[0].interrupts if state.tasks else []
    pergunta = interrupts[0].value if interrupts else ""
    return True, pergunta


def process(job_id: str) -> None:
    from langgraph.types import Command
    from seliga_graph import SeligaState

    job = get_job(job_id)
    if not job:
        print(f"[worker] job {job_id} não encontrado (expirado?)", flush=True)
        return

    session_id = job["session_id"]
    message    = job["message"]
    print(f"[worker] processando job {job_id} | session={session_id}", flush=True)
    update_job(job_id, status="processing")

    try:
        config = _graph_config(session_id)
        interrupted, _ = _is_interrupted(session_id)

        if interrupted:
            # Sessão pausada — retoma com a resposta do participante
            _graph.invoke(Command(resume=message), config)
        else:
            # Nova conversa ou grafo finalizado — inicia do zero
            initial: SeligaState = {
                "numero_atendimento":    session_id,
                "mensagem_participante": message,
                "skill_name": "", "skill_body": "",
                "dados": {}, "campos_faltando": [],
                "resposta_participante": "",
                "escalada_id": 0,
                "resposta_final": "",
                "rota": "coletar",
            }
            _graph.invoke(initial, config)

        still_interrupted, pergunta = _is_interrupted(session_id)

        if still_interrupted:
            answer  = pergunta
            sources = []
        else:
            values  = _graph.get_state(config).values
            answer  = values.get("resposta_final", "")
            skill   = values.get("skill_name", "")
            sources = [{"type": "skill", "name": skill}] if skill else []

        update_job(
            job_id,
            status="done",
            answer=answer,
            sources=json.dumps(sources, ensure_ascii=False),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        print(f"[worker] job {job_id} concluído | interrupted={still_interrupted}", flush=True)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        update_job(
            job_id,
            status="error",
            error=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        print(f"[worker] job {job_id} ERRO: {exc}", flush=True)


def main() -> None:
    try:
        _init_graph()
    except Exception as exc:
        print(f"[worker] ERRO FATAL na inicialização: {exc}", flush=True)
        sys.exit(1)

    print("[worker] aguardando jobs...", flush=True)
    running = True

    def shutdown(sig, _frame):
        nonlocal running
        print(f"[worker] sinal {sig} — encerrando após job atual...", flush=True)
        running = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while running:
        job_id = dequeue(timeout=1)
        if job_id:
            process(job_id)

    _pool.close()
    print("[worker] encerrado.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
