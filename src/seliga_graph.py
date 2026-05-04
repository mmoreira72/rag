"""
Se Liga — Atendimento Graph (LangGraph).

Workflow:
  identificar_skill        (RAG: qual skill se aplica à mensagem do participante)
       ↓
  coletar_dados            (LLM extrai ICARE + campos específicos da skill)
       ├─ dados incompletos ──▶ solicitar_dados  ──▶ (interrupt: aguarda participante)
       └─ dados completos  ──────────────────────▶ escaldar_sac
                                                         ↓
                                                  exibir_mensagem    (resposta final)

Uso:
    python src/seliga_graph.py --atendimento "AT-12345" --mensagem "meu login está bloqueado"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict, Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import interrupt, Command

sys.path.insert(0, str(Path(__file__).parent))
from config import LLM_BASE_URL, CHAT_MODEL, TOP_K_RETRIEVE_SKILLS, USE_RERANKER
from db import search_skills, insert_escalada, mark_escalada_sent, mark_escalada_failed
from embeddings import embed
from reranker import rerank

SKILL_MIN_SCORE = 0.55

llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key="ollama",
    model=CHAT_MODEL,
    temperature=0.2,
    max_tokens=1024,
)

MENSAGEM_ENCAMINHAMENTO = (
    "Para te ajudar com esse assunto, preciso de algumas informações que só o nosso "
    "time de atendimento consegue acessar com segurança. Por isso vou encaminhar o "
    "que conversamos para o SAC, para que possamos te atender da melhor forma possível."
)

# Campos obrigatórios por skill — adicionar novas skills aqui conforme necessário.
CAMPOS_POR_SKILL: dict[str, list[str]] = {
    "login-bloqueado":        ["icare", "cpf"],
    "pontos-nao-creditados":  ["icare", "mes_referencia"],
    "antecipar-boleto":       ["icare"],
    "opçoes-resgate":         ["icare"],
    "pag-contas-processando": ["icare"],
    "premio-brinde-fisico":   ["icare"],
    "primeiro-acesso-cadastro": ["icare", "cpf"],
    "resgate-cartao-virtual": ["icare"],
    "resgate-pagamento-pendente": ["icare"],
    "troca-devolucao-resgate": ["icare"],
}


# ── State ────────────────────────────────────────────────────────────────────

class SeligaState(TypedDict):
    numero_atendimento: str
    mensagem_participante: str

    skill_name: str
    skill_body: str

    # Dados coletados (icare + campos específicos da skill)
    dados: dict

    # Campos ainda faltando para escalar
    campos_faltando: list[str]

    # Resposta do participante após interrupt (coleta de dados faltantes)
    resposta_participante: str

    # ID da escalada gravada no banco
    escalada_id: int

    # Mensagem final para o participante
    resposta_final: str

    # Decisão de roteamento interna
    rota: Literal["coletar", "escalar"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_skill(query: str) -> tuple[str, str] | tuple[None, None]:
    """Devolve (name, body) da skill mais relevante ou (None, None)."""
    qvec = embed([query])[0]
    raw = search_skills(qvec, TOP_K_RETRIEVE_SKILLS)
    if not raw:
        return None, None
    if USE_RERANKER:
        texts = [f"{s[0]}\n{s[1]}\n{s[2][:2000]}" for s in raw]
        ranked = rerank(query, texts, top_k=1)
        if not ranked:
            return None, None
        orig_idx, rscore = ranked[0]
        if rscore < SKILL_MIN_SCORE:
            return None, None
        return raw[orig_idx][0], raw[orig_idx][2]
    if raw[0][3] < SKILL_MIN_SCORE:
        return None, None
    return raw[0][0], raw[0][2]


def _call_llm(system: str, user: str) -> str:
    resp = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ])
    return resp.content.strip()


# ── Nodes ────────────────────────────────────────────────────────────────────

def identificar_skill(state: SeligaState) -> dict:
    name, body = _get_skill(state["mensagem_participante"])
    if not name:
        return {
            "skill_name": "",
            "skill_body": "",
            "resposta_final": (
                "Não consegui identificar o assunto da sua solicitação. "
                "Pode descrever melhor o que está acontecendo?"
            ),
            "rota": "escalar",   # vai direto para END sem escalar ao SAC
        }
    return {"skill_name": name, "skill_body": body}


def coletar_dados(state: SeligaState) -> dict:
    """Usa o LLM para extrair os dados necessários da conversa."""
    campos = CAMPOS_POR_SKILL.get(state["skill_name"], ["icare"])
    prompt_sys = (
        "Você é um extrator de informações. Analise a mensagem do participante "
        "e extraia os campos listados em JSON. Use null para campos não informados."
    )
    prompt_usr = (
        f"Campos a extrair: {campos}\n\n"
        f"Mensagem: {state['mensagem_participante']}\n\n"
        "Responda APENAS com JSON válido, sem markdown."
    )
    raw = _call_llm(prompt_sys, prompt_usr)
    try:
        dados = json.loads(raw)
    except json.JSONDecodeError:
        dados = {}

    faltando = [c for c in campos if not dados.get(c)]
    return {"dados": dados, "campos_faltando": faltando}


def solicitar_dados(state: SeligaState) -> dict:
    """Pede ao participante os campos faltantes via interrupt."""
    perguntas = {
        "icare":          "Pode me informar o seu código ICARE?",
        "cpf":            "Pode me informar o seu CPF?",
        "mes_referencia": "Os pontos de qual mês você está esperando?",
    }
    pergunta = " ".join(
        perguntas.get(c, f"Pode me informar: {c}?")
        for c in state["campos_faltando"]
    )
    resposta = interrupt(pergunta)
    dados_atualizados = {**state.get("dados", {})}

    # Tenta extrair os campos da resposta do participante
    campos = state["campos_faltando"]
    prompt_sys = (
        "Extrator de informações. Analise a resposta e extraia os campos em JSON. "
        "Use null para campos não informados."
    )
    prompt_usr = (
        f"Campos a extrair: {campos}\n\nResposta: {resposta}\n\n"
        "Responda APENAS com JSON válido, sem markdown."
    )
    raw = _call_llm(prompt_sys, prompt_usr)
    try:
        novos = json.loads(raw)
        dados_atualizados.update({k: v for k, v in novos.items() if v})
    except json.JSONDecodeError:
        pass

    faltando = [c for c in campos if not dados_atualizados.get(c)]
    return {
        "dados": dados_atualizados,
        "campos_faltando": faltando,
        "resposta_participante": resposta,
    }


def escalar_sac(state: SeligaState) -> dict:
    """Grava a escalada no banco e marca como sent (API a ser integrada)."""
    body = {
        "skill":    state["skill_name"],
        "mensagem": state["mensagem_participante"],
        **state.get("dados", {}),
    }
    escalada_id = insert_escalada(
        numero_atendimento=state["numero_atendimento"],
        skill_name=state["skill_name"],
        icare=state.get("dados", {}).get("icare"),
        body=body,
    )
    # TODO: chamar API do serviço terceiro aqui quando disponível.
    # Por ora marca como sent direto (worker externo vai processar via get_pending_escaladas).
    mark_escalada_sent(escalada_id)

    return {
        "escalada_id": escalada_id,
        "resposta_final": MENSAGEM_ENCAMINHAMENTO,
    }


def exibir_mensagem(state: SeligaState) -> dict:
    return {}   # resposta_final já está no state; nó terminal de saída


# ── Routing ──────────────────────────────────────────────────────────────────

def route_apos_coleta(state: SeligaState) -> str:
    if state.get("campos_faltando"):
        return "solicitar_dados"
    return "escalar_sac"


def route_apos_identificar(state: SeligaState) -> str:
    # Se skill não encontrada, resposta_final já está definida — vai direto ao fim
    if not state.get("skill_name"):
        return "exibir_mensagem"
    return "coletar_dados"


# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    g = StateGraph(SeligaState)

    g.add_node("identificar_skill", identificar_skill)
    g.add_node("coletar_dados",     coletar_dados)
    g.add_node("solicitar_dados",   solicitar_dados)
    g.add_node("escalar_sac",       escalar_sac)
    g.add_node("exibir_mensagem",   exibir_mensagem)

    g.set_entry_point("identificar_skill")

    g.add_conditional_edges("identificar_skill", route_apos_identificar, {
        "coletar_dados":   "coletar_dados",
        "exibir_mensagem": "exibir_mensagem",
    })
    g.add_conditional_edges("coletar_dados", route_apos_coleta, {
        "solicitar_dados": "solicitar_dados",
        "escalar_sac":     "escalar_sac",
    })
    g.add_conditional_edges("solicitar_dados", route_apos_coleta, {
        "solicitar_dados": "solicitar_dados",
        "escalar_sac":     "escalar_sac",
    })
    g.add_edge("escalar_sac",     "exibir_mensagem")
    g.add_edge("exibir_mensagem", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atendimento", required=True, help="Número do atendimento")
    parser.add_argument("--mensagem",    required=True, help="Mensagem do participante")
    args = parser.parse_args()

    graph = build_graph()
    config = {"configurable": {"thread_id": args.atendimento}}

    state_in: SeligaState = {
        "numero_atendimento":   args.atendimento,
        "mensagem_participante": args.mensagem,
        "skill_name": "", "skill_body": "",
        "dados": {}, "campos_faltando": [],
        "resposta_participante": "",
        "escalada_id": 0,
        "resposta_final": "",
        "rota": "coletar",
    }

    graph.invoke(state_in, config)

    while True:
        state = graph.get_state(config)
        if not state.next:
            break
        interrupts = state.tasks[0].interrupts if state.tasks else []
        pergunta = interrupts[0].value if interrupts else "Mais informações?"
        print(f"\nAssistente: {pergunta}")
        resposta = input("Participante: ").strip()
        graph.invoke(Command(resume=resposta), config)

    final = graph.get_state(config).values
    print(f"\nAssistente: {final.get('resposta_final', '')}")
    if final.get("escalada_id"):
        print(f"[escalada gravada: id={final['escalada_id']}]")
