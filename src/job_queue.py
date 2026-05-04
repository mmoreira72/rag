"""
Fila de jobs via Redis.

Fluxo:
  1. API recebe mensagem → enqueue() → retorna job_id imediatamente
  2. Worker faz dequeue() em loop → processa RAG + LLM → update_job(done/error)
  3. Frontend faz polling em GET /chat/status/{job_id} até status=done

Estrutura do job no Redis (hash):
  chat:job:{job_id} → { job_id, session_id, message, status, answer, error, created_at, completed_at }

Fila (list Redis):
  chat:queue → lista de job_ids (LPUSH para enfileirar, BRPOP para consumir)
"""
import uuid
from datetime import datetime, timezone

import redis

from config import REDIS_URL

QUEUE_KEY = "chat:queue"
JOB_PREFIX = "chat:job:"
SESSION_ACTIVE_PREFIX = "chat:active:"   # session_id → job_id enquanto queued/processing
JOB_TTL_SECONDS = 3600  # jobs expiram após 1h
ACTIVE_TTL_SECONDS = 300  # trava de dedup expira em 5 min (evita deadlock se worker crashar)


def _client() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def get_active_job(session_id: str) -> str | None:
    """Retorna o job_id ativo (queued/processing) da sessão, ou None."""
    return _client().get(f"{SESSION_ACTIVE_PREFIX}{session_id}")


def enqueue(session_id: str, message: str) -> str:
    """Cria um job na fila e retorna o job_id.

    Se já existe um job queued/processing para esta sessão, devolve o job_id
    existente sem criar duplicata (idempotência contra retries do cliente).
    """
    r = _client()
    active_key = f"{SESSION_ACTIVE_PREFIX}{session_id}"

    # SET NX: só grava se não existir (atômico)
    job_id = str(uuid.uuid4())
    acquired = r.set(active_key, job_id, nx=True, ex=ACTIVE_TTL_SECONDS)
    if not acquired:
        # Já existe job ativo — devolve o existente
        existing = r.get(active_key)
        if existing:
            return existing
        # Chave expirou entre o get e agora — prossegue criando novo job
        r.set(active_key, job_id, ex=ACTIVE_TTL_SECONDS)

    job = {
        "job_id": job_id,
        "session_id": session_id,
        "message": message,
        "status": "queued",
        "answer": "",
        "sources": "",
        "error": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": "",
    }
    r.hset(f"{JOB_PREFIX}{job_id}", mapping=job)
    r.expire(f"{JOB_PREFIX}{job_id}", JOB_TTL_SECONDS)
    r.lpush(QUEUE_KEY, job_id)
    return job_id


def dequeue(timeout: int = 5) -> str | None:
    """Bloqueia até receber um job_id da fila (ou timeout). Retorna None se vazio."""
    r = _client()
    result = r.brpop(QUEUE_KEY, timeout=timeout)
    if result is None:
        return None
    _, job_id = result
    return job_id


def get_job(job_id: str) -> dict | None:
    """Retorna os dados do job ou None se não existir/expirado."""
    r = _client()
    data = r.hgetall(f"{JOB_PREFIX}{job_id}")
    return data if data else None


def update_job(job_id: str, **fields) -> None:
    """Atualiza campos do job e renova o TTL. Libera a trava de sessão quando finalizado."""
    r = _client()
    r.hset(f"{JOB_PREFIX}{job_id}", mapping=fields)
    r.expire(f"{JOB_PREFIX}{job_id}", JOB_TTL_SECONDS)

    # Libera a trava da sessão quando o job termina (done ou error)
    if fields.get("status") in ("done", "error"):
        job = r.hgetall(f"{JOB_PREFIX}{job_id}")
        if job.get("session_id"):
            active_key = f"{SESSION_ACTIVE_PREFIX}{job['session_id']}"
            # Só remove se ainda aponta para este job (evita remover job novo)
            current = r.get(active_key)
            if current == job_id:
                r.delete(active_key)
