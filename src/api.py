"""
API HTTP do local-rag.

Fluxo assíncrono com fila Redis:
  POST /chat          → enfileira job, retorna job_id imediatamente
  GET  /chat/status/{job_id} → retorna status e resposta quando pronto
  DELETE /chat/{session_id}  → encerra/reseta uma sessão
  GET  /health        → liveness check

O processamento RAG + LLM acontece no worker (worker.py), não aqui.
Isso serializa as chamadas ao LLM (GPU processa 1 por vez) sem bloquear a API.

Uso:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from config import API_KEY
from db import delete_session
from embeddings import embed
from job_queue import enqueue, get_job
from llm_client import warmup_llm_models
from reranker import _get_reranker

_api_key_header = APIKeyHeader(name="X-API-Key")


def _require_api_key(key: str = Security(_api_key_header)) -> None:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY não configurada no servidor.")
    if key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key inválida.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pré-carrega embedding model e reranker durante o startup do container.
    # O worker também faz isso — aqui é para o caso de a API precisar de embeddings futuramente.
    print("[startup] pré-carregando embedding model...", flush=True)
    embed(["warmup"])
    print("[startup] pré-carregando reranker...", flush=True)
    _get_reranker()
    warmup_llm_models()
    print("[startup] modelos prontos.", flush=True)
    yield


app = FastAPI(title="local-rag chat API", lifespan=lifespan)


# ---------- schemas ----------

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="ID único da conversa")
    message: str = Field(..., min_length=1, description="Mensagem do usuário")
    autenticado: bool = Field(False, description="True se o participante está logado no portal")


class ChatQueued(BaseModel):
    job_id: str
    status: str = "queued"


class JobStatus(BaseModel):
    job_id: str
    status: str           # queued | processing | done | error
    answer: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""


# ---------- endpoints ----------

@app.post("/chat", response_model=ChatQueued, status_code=202,
          dependencies=[Security(_require_api_key)])
def chat_endpoint(req: ChatRequest):
    """
    Enfileira a mensagem e retorna imediatamente com o job_id.

    O frontend deve fazer polling em GET /chat/status/{job_id} até status=done.
    HTTP 202 Accepted indica que o job foi aceito mas ainda não processado.
    """
    job_id = enqueue(req.session_id, req.message, req.autenticado)
    return ChatQueued(job_id=job_id)


@app.get("/chat/status/{job_id}", response_model=JobStatus,
         dependencies=[Security(_require_api_key)])
def job_status(job_id: str):
    """
    Retorna o status do job e a resposta quando pronto.

    Polling recomendado: a cada 1-2s até status=done ou status=error.
    Jobs expiram após 1h — após isso retorna 404.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado ou expirado.")
    raw_sources = job.get("sources", "")
    sources = json.loads(raw_sources) if raw_sources else []
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        answer=job.get("answer", ""),
        sources=sources,
        error=job.get("error", ""),
    )


@app.delete("/chat/{session_id}", status_code=204,
            dependencies=[Security(_require_api_key)])
def reset_session(session_id: str):
    """Apaga o histórico da sessão. Chamar ao iniciar novo atendimento."""
    delete_session(session_id)


@app.get("/health")
def health():
    return {"status": "ok"}
