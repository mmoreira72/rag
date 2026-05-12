"""
Worker de processamento de jobs do chat.

Roda como processo separado em container próprio.
Consome jobs da fila Redis, executa RAG + LLM e grava o resultado de volta.

Um único worker serializa as chamadas ao LLM — necessário porque a GPU
processa uma requisição por vez. Para aumentar throughput, suba mais workers
(só faz sentido se tiver múltiplas GPUs).

Uso:
    python worker.py

Sinais:
    SIGTERM / SIGINT — encerra o loop após o job corrente terminar (graceful shutdown).
"""
import json
import signal
import sys
from datetime import datetime, timezone

from seliga_ask import ask
from job_queue import dequeue, get_job, update_job
from llm_client import warmup_llm_models


def process(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        print(f"[worker] job {job_id} não encontrado (expirado?)", flush=True)
        return

    autenticado = job.get("autenticado", "0") == "1"
    print(f"[worker] processando job {job_id} | session={job['session_id']} | autenticado={autenticado}", flush=True)
    update_job(job_id, status="processing")

    try:
        result = ask(job["session_id"], job["message"], autenticado)
        answer = result["answer"]
        sources = json.dumps(result["sources"], ensure_ascii=False)
        update_job(
            job_id,
            status="done",
            answer=answer,
            sources=sources,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        print(f"[worker] job {job_id} concluído ({len(answer)} chars, {len(result['sources'])} fontes)", flush=True)
    except Exception as exc:
        update_job(
            job_id,
            status="error",
            error=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        print(f"[worker] job {job_id} ERRO: {exc}", flush=True)


def main() -> None:
    warmup_llm_models()
    print("[worker] iniciando — aguardando jobs...", flush=True)
    running = True

    def shutdown(sig, _frame):
        nonlocal running
        print(f"[worker] sinal {sig} recebido — encerrando após job atual...", flush=True)
        running = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while running:
        job_id = dequeue(timeout=1)
        if job_id:
            process(job_id)

    print("[worker] encerrado.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
