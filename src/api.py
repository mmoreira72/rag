"""
API HTTP do local-rag.

Expõe o chat com RAG + skills via REST, com histórico de sessão persistido
no PostgreSQL. Cada session_id é isolado — múltiplos usuários simultâneos
não interferem entre si.

Uso:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /chat          — envia mensagem, recebe resposta
    DELETE /chat/{id}   — encerra/reseta uma sessão
    GET  /health        — liveness check
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from chat import ask
from config import API_KEY
from db import delete_session
from embeddings import embed
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
    # Sem isso o primeiro request demora ~40s enquanto os modelos carregam.
    print("[startup] pré-carregando embedding model...", flush=True)
    embed(["warmup"])
    print("[startup] pré-carregando reranker...", flush=True)
    _get_reranker()
    print("[startup] modelos prontos.", flush=True)
    yield


app = FastAPI(title="local-rag chat API", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="ID único da conversa (ex: número de WhatsApp, user ID)")
    message: str = Field(..., min_length=1, description="Mensagem do usuário")


class ChatResponse(BaseModel):
    session_id: str
    answer: str


@app.post("/chat", response_model=ChatResponse, dependencies=[Security(_require_api_key)])
def chat_endpoint(req: ChatRequest):
    """
    Recebe uma mensagem do usuário e retorna a resposta do assistente.

    O histórico da sessão é buscado e persistido automaticamente no Postgres.
    Sessões expiram após 30 minutos de inatividade.
    """
    answer = ask(req.session_id, req.message)
    return ChatResponse(session_id=req.session_id, answer=answer)


@app.delete("/chat/{session_id}", status_code=204, dependencies=[Security(_require_api_key)])
def reset_session(session_id: str):
    """
    Encerra/reseta uma sessão, apagando o histórico.

    Útil quando o usuário digita 'reiniciar', 'novo atendimento', etc.
    O canal (WhatsApp, web) deve chamar este endpoint nesses casos.
    """
    delete_session(session_id)


@app.get("/health")
def health():
    return {"status": "ok"}
