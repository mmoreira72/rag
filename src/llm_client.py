"""
Cliente do modelo de chat via API compatível com OpenAI.

Funciona com Ollama (padrão) ou qualquer outro servidor com API OpenAI-compatible.
Configuração via variáveis de ambiente LLM_BASE_URL e CHAT_MODEL (ver config.py).
"""
from openai import OpenAI
from config import LLM_BASE_URL, CHAT_MODEL, WARMUP_MODELS

client = OpenAI(base_url=LLM_BASE_URL, api_key="ollama")


def chat(messages: list[dict], stream: bool = True):
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        stream=stream,
    )


def warmup_llm_models() -> None:
    """Pré-carrega os modelos LLM configurados em WARMUP_MODELS na VRAM do Ollama.

    Envia um request mínimo a cada modelo para forçar o carregamento antes do
    primeiro request real. Sem isso, o primeiro usuário espera o tempo de load
    (tipicamente 20-60s para modelos grandes).
    """
    for model in WARMUP_MODELS:
        print(f"[startup] pré-carregando LLM {model}...", flush=True)
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "oi"}],
                max_tokens=1,
                stream=False,
            )
            print(f"[startup] {model} pronto.", flush=True)
        except Exception as exc:
            print(f"[startup] aviso: falha ao pré-carregar {model}: {exc}", flush=True)
