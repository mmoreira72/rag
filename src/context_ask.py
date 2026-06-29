"""
Alternativa ao seliga_graph: context-stuffing + KV cache.

Em vez de RAG (embed → pgvector → reranker → LLM com contexto recuperado),
carrega TODAS as skills + docs como prefixo estático no system message.
O Ollama reutiliza o KV cache do prefixo em queries subsequentes → custo de
prefill pago UMA vez; queries seguintes = só o delta da pergunta.

Interface pública: ask(session_id, message, autenticado) → dict
  Idêntica à de seliga_ask.py — o worker chama sem saber qual backend usa.

Controlado por:
  USE_CONTEXT_STUFFING=1   → activa este módulo (setado no .env)
  CHAT_MODEL               → modelo Ollama (ex: gemma4:e4b)
  OLLAMA_NATIVE_URL        → endpoint nativo Ollama (default: host.docker.internal:11434)
  CONTENT_DIR              → raiz do conteúdo (default: /app)
"""
from __future__ import annotations

import glob
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict

logger = logging.getLogger(__name__)

_MODEL = os.getenv("CHAT_MODEL", "gemma4:e4b")
_OLLAMA_BASE = os.getenv("OLLAMA_NATIVE_URL", "http://host.docker.internal:11434")
_OLLAMA_CHAT = f"{_OLLAMA_BASE}/api/chat"
_CONTENT = os.getenv("CONTENT_DIR", "/app")

# System prompt — regras rígidas que o LLM deve seguir.
# Alinhado com o system prompt do RAG mas adaptado para context-stuffing
# (sem menção a SKILLS ATIVAS / CONTEXTO recuperados — tudo já está no prefixo).
_SYS = """\
Você é o assistente de atendimento do programa de incentivo "Se Liga na SKY".
ATENÇÃO: Se Liga na SKY é um programa de INCENTIVO E PONTUAÇÃO para colaboradores da SKY Brasil.
NÃO é um serviço de telecomunicações. Não mencione operadora, TV, linha telefónica, plano SKY,
106 10, suporte de TV ou internet. Nunca invente informações.

Abaixo estão TODAS as skills (fluxos de atendimento) e documentos do programa.
Responda USANDO APENAS este material.

REGRAS OBRIGATÓRIAS:
1. Se a pergunta NÃO estiver coberta pelo material abaixo, responda EXATAMENTE:
   "Não posso responder pois não está na minha base de dados."
2. Se o participante NÃO estiver logado:
   - Só responda temas PÚBLICOS: login bloqueado, esqueci/recuperar senha, primeiro acesso/cadastro, saudações.
   - Para QUALQUER outro tema responda APENAS: "Para isso, você precisa acessar a plataforma com seu login."
   - NÃO peça dados. NÃO ofereça mais ajuda.
3. Em skills de escalação que exigem campos (ex: código ICARE, e-mail, telefone),
   peça o campo que falta antes de encaminhar ao SAC. Um campo por vez.
4. Tom: amigável, direto, como WhatsApp. Máximo 3 frases.
5. Responda sempre em português do Brasil.\
"""

# Cache em memória do prefixo (processo único — o worker só tem 1 worker).
_prefix_cache: str | None = None

# Histórico de sessão em memória (simples — perde no restart do worker).
# Formato: session_id → [{"role": "user"|"assistant", "content": str}]
_history: dict[str, list[dict]] = defaultdict(list)
_MAX_HISTORY = 6  # turnos (3 user + 3 assistant)


def _build_prefix() -> str:
    """Monta o prefixo estático: regras + skills + docs."""
    parts = [_SYS, "\n\n===== SKILLS ====="]
    skill_files = sorted(glob.glob(f"{_CONTENT}/skills/*/SKILL.md"))
    if not skill_files:
        logger.warning("context_ask: nenhum SKILL.md encontrado em %s/skills/", _CONTENT)
    for f in skill_files:
        try:
            parts.append(open(f, encoding="utf-8").read())
        except OSError as e:
            logger.warning("context_ask: erro ao ler %s: %s", f, e)

    parts.append("\n===== DOCUMENTOS =====")
    doc_files = (
        sorted(glob.glob(f"{_CONTENT}/data/user/.converted/*.md"))
        + sorted(glob.glob(f"{_CONTENT}/data/user/*.txt"))
    )
    for f in doc_files:
        try:
            parts.append(open(f, encoding="utf-8").read())
        except OSError as e:
            logger.warning("context_ask: erro ao ler %s: %s", f, e)

    prefix = "\n".join(parts)
    n_skills = len(skill_files)
    n_docs = len(doc_files)
    approx_tokens = len(prefix) // 4
    logger.info("context_ask: prefixo montado — %d skills, %d docs, ~%d tokens",
                n_skills, n_docs, approx_tokens)
    return prefix


def _get_prefix() -> str:
    global _prefix_cache
    if _prefix_cache is None:
        _prefix_cache = _build_prefix()
    return _prefix_cache


def _call_ollama(messages: list[dict], timeout: int = 300) -> str:
    """Chama o Ollama nativo com think:false. Retorna o conteúdo da resposta."""
    body = json.dumps({
        "model": _MODEL,
        "think": False,
        "stream": False,
        "messages": messages,
        "options": {
            "num_predict": 200,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        },
    }).encode()
    req = urllib.request.Request(
        _OLLAMA_CHAT,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return (d.get("message", {}).get("content") or "").strip()


def warmup() -> None:
    """
    Aquece o KV cache do Ollama enviando o prefixo estático uma vez.
    Chamar no startup do worker — o custo de prefill é pago aqui (pode ser lento)
    mas queries seguintes reutilizam o cache.
    """
    logger.info("context_ask: a aquecer o KV cache (pode demorar 30-120s)...")
    t0 = time.time()
    prefix = _get_prefix()
    try:
        _call_ollama([
            {"role": "system", "content": prefix},
            {"role": "user", "content": "Estado: logado\nPergunta: oi"},
        ], timeout=300)
        logger.info("context_ask: cache aquecido em %.1fs", time.time() - t0)
    except Exception as e:
        logger.warning("context_ask: warmup falhou (não fatal): %s", e)


def ask(session_id: str, message: str, autenticado: bool = True) -> dict:
    """
    Interface pública — idêntica a seliga_ask.ask().

    Usa context-stuffing: prefixo estático (todas as skills+docs) no system
    message + KV cache do Ollama. Sem retrieval, sem reranker, sem pgvector.
    """
    prefix = _get_prefix()
    auth_label = "logado" if autenticado else "nao-logado"

    # Monta messages: system (prefixo estático, cacheado) + histórico + pergunta
    history = _history[session_id]
    messages = (
        [{"role": "system", "content": prefix}]
        + history
        + [{"role": "user", "content": f"Estado do participante: {auth_label}\nPergunta: {message}"}]
    )

    t0 = time.time()
    try:
        answer = _call_ollama(messages)
    except urllib.error.URLError as e:
        logger.error("context_ask: erro Ollama: %s", e)
        answer = "Ocorreu um erro no processamento. Tente novamente."
    except Exception as e:
        logger.error("context_ask: erro inesperado: %s", e)
        answer = "Ocorreu um erro no processamento. Tente novamente."

    elapsed = time.time() - t0
    logger.info("context_ask: session=%s auth=%s latency=%.1fs", session_id, auth_label, elapsed)

    # Guarda histórico (últimos _MAX_HISTORY mensagens user+assistant)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    _history[session_id] = history[-_MAX_HISTORY:]

    # Log da interação (best-effort — context stuffing não tem score de retrieval)
    try:
        from db import insert_interacao
        insert_interacao(session_id, message, None, None, True, autenticado, answer)
    except Exception as e:  # noqa: BLE001
        logger.debug("context_ask: falha ao logar interacao: %s", e)

    # sources vazio — não há retrieval (o LLM usa o prefixo inteiro)
    return {"answer": answer, "sources": []}
