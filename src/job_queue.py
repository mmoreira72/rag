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
JOB_TTL_SECONDS = 3600  # jobs expiram após 1h


def _client() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def enqueue(session_id: str, message: str) -> str:
    """Cria um job na fila e retorna o job_id."""
    job_id = str(uuid.uuid4())
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
    r = _client()
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
    """Atualiza campos do job e renova o TTL."""
    r = _client()
    r.hset(f"{JOB_PREFIX}{job_id}", mapping=fields)
    r.expire(f"{JOB_PREFIX}{job_id}", JOB_TTL_SECONDS)
