"""
Cliente do modelo de chat via API compatível com OpenAI.

Funciona com Ollama (padrão) ou qualquer outro servidor com API OpenAI-compatible.
Configuração via variáveis de ambiente LLM_BASE_URL e CHAT_MODEL (ver config.py).
"""
from openai import OpenAI
from config import LLM_BASE_URL, CHAT_MODEL

client = OpenAI(base_url=LLM_BASE_URL, api_key="ollama")


def chat(messages: list[dict], stream: bool = True):
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        stream=stream,
    )
