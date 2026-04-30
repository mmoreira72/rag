"""
Incident Response Graph — LangGraph skeleton.

Workflow:
  parse_alert
      ↓
  fetch_errors          (gcloud logging read ±30min)
      ↓
  check_release         (Cloud Deploy / Artifact Registry)
      ├─ yes ──▶ fetch_release_logs ──▶ build_report
      └─ no  ──────────────────────────▶ build_report
                                              ↓
                                     notify_and_wait   ← interrupt() aguarda engenheiro
                                              ↓
                                           decide       (rollback / escalate / close)

Uso:
    python src/incident_graph.py --alert alert.json

Dependências adicionais (adicionar a requirements.txt):
    langgraph>=0.2
    langchain-openai>=0.1
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TypedDict, Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

# Imports locais (mesmo package)
sys.path.insert(0, str(Path(__file__).parent))
from db import search_skills
from embeddings import embed
from reranker import rerank
from config import LLM_BASE_URL, CHAT_MODEL, TOP_K_RETRIEVE_SKILLS, USE_RERANKER

# ── LLM ─────────────────────────────────────────────────────────────────────
# Aponta para o Ollama remoto (ou local) via API OpenAI-compatible
llm = ChatOpenAI(
    base_url=f"{LLM_BASE_URL}",
    api_key="ollama",               # Ollama não precisa de key real
    model=CHAT_MODEL,
    temperature=0.2,
    max_tokens=2048,
)

# ── State ────────────────────────────────────────────────────────────────────
class IncidentState(TypedDict):
    # Input
    alert_raw: str

    # Extraído do alerta
    incident_id: str
    project: str
    service: str
    region: str
    alert_ts: str           # ISO 8601
    condition_name: str
    alert_url: str

    # Logs de erro em torno do alerta
    error_logs: str         # JSON string com top erros

    # Release correlation
    release_found: bool
    release_id: str
    release_ts: str
    release_logs: str

    # Output
    report: str
    engineer_response: str  # preenchido após interrupt()
    decision: Literal["rollback", "escalate", "close", "monitor"]


# ── Skill retrieval ──────────────────────────────────────────────────────────
SKILL_MIN_SCORE = 0.70
SKILL_CANDIDATE_SCORE = 0.45


def get_skill(query: str) -> str:
    """Devolve o body da skill mais relevante ou string vazia."""
    qvec = embed([query])[0]
    raw = search_skills(qvec, TOP_K_RETRIEVE_SKILLS)
    if not raw:
        return ""
    if USE_RERANKER:
        texts = [f"{r[0]}\n{r[1]}\n{r[2][:1000]}" for r in raw]
        ranked = rerank(query, texts, top_k=1)
        if ranked:
            idx, score = ranked[0]
            if score >= SKILL_MIN_SCORE:
                return f"[skill: {raw[idx][0]}]\n\n{raw[idx][2]}"
    else:
        if raw[0][3] >= SKILL_MIN_SCORE:
            return f"[skill: {raw[0][0]}]\n\n{raw[0][2]}"
    return ""


def call_llm(system: str, user: str) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content


# ── Nodes ────────────────────────────────────────────────────────────────────

def parse_alert(state: IncidentState) -> dict:
    """Extrai campos estruturados do payload do alerta.

    Suporta dois formatos:
    - GCP Pub/Sub notification (campo 'incident' no root)
    - Formato simplificado (campos directos no root)
    """
    try:
        raw = json.loads(state["alert_raw"])
    except json.JSONDecodeError:
        raw = {}

    # Formato nativo GCP (Pub/Sub notification)
    inc = raw.get("incident", raw)

    # started_at pode ser Unix timestamp ou ISO string
    started_at = inc.get("started_at", "")
    if isinstance(started_at, (int, float)):
        alert_ts = datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat()
    elif started_at:
        alert_ts = started_at
    else:
        alert_ts = datetime.now(timezone.utc).isoformat()

    resource = inc.get("resource", {})
    labels   = resource.get("labels", {})

    # project_id: tenta resource.labels, fallback para monitoring_project
    project = (
        labels.get("project_id")
        or inc.get("project_id")
        or inc.get("monitoring_project")
        or ""
    )
    # Remove placeholder de teste
    if project.startswith("TODO"):
        project = inc.get("monitoring_project", "")

    return {
        "incident_id":    inc.get("incident_id") or inc.get("name", "unknown"),
        "project":        project,
        "service":        labels.get("service_name") or inc.get("service", ""),
        "region":         labels.get("location") or labels.get("region") or "europe-west1",
        "alert_ts":       alert_ts,
        "condition_name": (
            inc.get("condition_name")
            or inc.get("policy_display_name")
            or inc.get("condition", {}).get("displayName", "")
        ),
        "alert_url":      inc.get("url") or inc.get("alert_url", ""),
    }


def _run_gcloud_logging(filter_str: str, project: str, limit: int = 100, label: str = "") -> list:
    """Helper: corre gcloud logging read e devolve lista de entradas JSON."""
    cmd = [
        "gcloud", "logging", "read", filter_str,
        f"--project={project}",
        "--format=json",
        f"--limit={limit}",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        results = json.loads(out.stdout or "[]")
        print(f"  [logging:{label or 'query'}] {len(results)} entries | stderr: {out.stderr[:120] if out.stderr else 'none'}")
        return results
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [logging:{label or 'query'}] ERROR: {e}")
        return []


def fetch_errors(state: IncidentState) -> dict:
    """Corre gcloud logging read com duas queries:
    1. Erros gerais ±90 min (janela larga para capturar causa antes do sintoma)
    2. Erros de memória/OOM nas últimas 4 horas (padrões de memory leak)
    """
    skill = get_skill("gcloud logging read error logs Cloud Run service")

    alert_dt    = datetime.fromisoformat(state["alert_ts"].replace("Z", "+00:00"))
    window_start = (alert_dt - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
    window_end   = (alert_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    oom_start    = (alert_dt - timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
    project      = state["project"]
    service      = state["service"]

    # Query A — últimas 6h antes do alerta, sem filtro de severidade
    # (OOM do Node.js aparece como DEFAULT, e pode preceder o alerta por horas)
    six_hours_before = (alert_dt - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    all_logs = _run_gcloud_logging(
        (
            f'resource.type="cloud_run_revision"'
            f' AND resource.labels.service_name="{service}"'
            f' AND timestamp>="{six_hours_before}"'
            f' AND timestamp<="{window_end}"'
        ),
        project,
        limit=150,
        label="6h",
    )

    # Extrai texto de cada entrada para o LLM (textPayload ou jsonPayload.message)
    log_lines = []
    for entry in all_logs:
        ts  = entry.get("timestamp", "")
        sev = entry.get("severity", "DEFAULT")
        msg = (
            entry.get("textPayload")
            or entry.get("jsonPayload", {}).get("message")
            or entry.get("jsonPayload", {}).get("msg")
            or str(entry.get("jsonPayload", ""))
        )
        if msg:
            log_lines.append(f"[{ts}] [{sev}] {msg[:300]}")

    combined = "\n".join(log_lines[:100])

    system = f"""{skill}

Analyze these Cloud Run logs and identify ALL error patterns.
Pay special attention to:
- "FATAL ERROR" / "heap out of memory" / "Allocation failed" → JavaScript OOM (ROOT CAUSE)
- "LIVENESS probe failed" → instance killed after OOM (SYMPTOM)
- Connection errors / 503 → consequence of restarts (SYMPTOM)

If OOM errors predate the liveness failures, mark them as root_cause=true.

Return JSON: [{{"count": N, "message": "...", "first_seen": "ISO8601", "root_cause": true/false}}]"""

    summary = call_llm(system, combined[:8000])
    return {"error_logs": summary}


def check_release(state: IncidentState) -> dict:
    """Verifica se houve release no Cloud Deploy na hora anterior ao alerta."""
    skill = get_skill("Cloud Deploy release rollout history timestamp")

    alert_dt     = datetime.fromisoformat(state["alert_ts"].replace("Z", "+00:00"))
    window_start = (alert_dt - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    project      = state["project"]
    service      = state["service"]
    region       = state["region"]

    # Cloud Run revisions (mais universal que Cloud Deploy)
    cmd = [
        "gcloud", "run", "revisions", "list",
        f"--service={service}",
        f"--project={project}",
        f"--region={region}",
        "--format=json",
        "--limit=5",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        revisions = json.loads(out.stdout or "[]")
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        revisions = []

    for rev in revisions:
        created = rev.get("metadata", {}).get("creationTimestamp", "")
        if created and window_start <= created <= state["alert_ts"]:
            return {
                "release_found": True,
                "release_id":    rev.get("metadata", {}).get("name", ""),
                "release_ts":    created,
            }

    return {"release_found": False, "release_id": "", "release_ts": ""}


def fetch_release_logs(state: IncidentState) -> dict:
    """Repete fetch_errors mas centrado no timestamp da release."""
    if not state.get("release_ts"):
        return {"release_logs": "[]"}

    skill = get_skill("gcloud logging read error logs Cloud Run service")

    rel_dt = datetime.fromisoformat(state["release_ts"].replace("Z", "+00:00"))
    window_start = (rel_dt - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    window_end   = (rel_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    cmd = [
        "gcloud", "logging", "read",
        (
            f'resource.type="cloud_run_revision"'
            f' AND resource.labels.service_name="{state["service"]}"'
            f' AND severity>=ERROR'
            f' AND timestamp>="{window_start}"'
            f' AND timestamp<="{window_end}"'
        ),
        f"--project={state['project']}",
        "--format=json",
        "--limit=100",
    ]

    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        logs_raw = out.stdout or "[]"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logs_raw = "[]"

    system = f"""{skill}

Summarise errors that appeared AFTER the release deploy.
Return JSON: [{{"count": N, "message": "...", "first_seen": "ISO8601"}}]"""

    summary = call_llm(system, logs_raw[:6000])
    return {"release_logs": summary}


def build_report(state: IncidentState) -> dict:
    """Monta o relatório de incidente com toda a informação recolhida."""
    skill = get_skill("incident report template SRE post-mortem")

    context = json.dumps({
        "incident_id":    state.get("incident_id"),
        "service":        state.get("service"),
        "project":        state.get("project"),
        "alert_ts":       state.get("alert_ts"),
        "condition":      state.get("condition_name"),
        "error_logs":     state.get("error_logs"),
        "release_found":  state.get("release_found"),
        "release_id":     state.get("release_id"),
        "release_ts":     state.get("release_ts"),
        "release_logs":   state.get("release_logs"),
        "alert_url":      state.get("alert_url"),
    }, indent=2)

    system = f"""{skill}

Using the incident data below, generate a concise incident report in Markdown.
Include: Impact, Timeline, Error Pattern, Release Correlation, Recommended Action.
End with a RECOMMENDED ACTION: one of rollback / escalate / monitor / close"""

    report = call_llm(system, context)
    return {"report": report}


def notify_and_wait(state: IncidentState) -> dict:
    """Envia notificação ao engenheiro e aguarda resposta (LangGraph interrupt)."""
    # Envia para Slack/PagerDuty — substituir pelo webhook real
    print("\n" + "="*60)
    print("INCIDENT REPORT")
    print("="*60)
    print(state["report"])
    print("="*60)

    # TODO: substituir por chamada real ao Slack/PagerDuty
    # import httpx
    # httpx.post(SLACK_WEBHOOK, json={"text": state["report"]})

    # interrupt() pausa o grafo aqui e devolve controlo ao caller.
    # O caller pode chamar graph.invoke() novamente com o engineer_response.
    response = interrupt("Aguardando resposta do engenheiro (rollback/escalate/monitor/close):")

    return {"engineer_response": response}


def decide(state: IncidentState) -> dict:
    """Decide a acção final com base na resposta do engenheiro."""
    response = (state.get("engineer_response") or "").lower()

    if "rollback" in response:
        decision = "rollback"
    elif "escalate" in response:
        decision = "escalate"
    elif "close" in response:
        decision = "close"
    else:
        decision = "monitor"

    print(f"\n→ Decision: {decision}")

    # TODO: executar a acção (chamar terragrunt, gcloud deploy rollback, etc.)

    return {"decision": decision}


# ── Edges ────────────────────────────────────────────────────────────────────

def route_release(state: IncidentState) -> str:
    return "fetch_release_logs" if state.get("release_found") else "build_report"


# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(IncidentState)

    g.add_node("parse_alert",        parse_alert)
    g.add_node("fetch_errors",       fetch_errors)
    g.add_node("check_release",      check_release)
    g.add_node("fetch_release_logs", fetch_release_logs)
    g.add_node("build_report",       build_report)
    g.add_node("notify_and_wait",    notify_and_wait)
    g.add_node("decide",             decide)

    g.set_entry_point("parse_alert")
    g.add_edge("parse_alert",        "fetch_errors")
    g.add_edge("fetch_errors",       "check_release")
    g.add_conditional_edges("check_release", route_release)
    g.add_edge("fetch_release_logs", "build_report")
    g.add_edge("build_report",       "notify_and_wait")
    g.add_edge("notify_and_wait",    "decide")
    g.add_edge("decide",             END)

    return g.compile(checkpointer=MemorySaver())


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incident Response Graph")
    parser.add_argument("--alert", help="Path to alert JSON file")
    parser.add_argument("--thread", default="incident-1", help="Thread ID for checkpointing")
    parser.add_argument("--print-graph", action="store_true", help="Print graph structure and exit")
    args = parser.parse_args()

    if args.print_graph:
        graph = build_graph()
        print("\n── Mermaid ──────────────────────────────────────────────")
        print(graph.get_graph().draw_mermaid())
        print("\n── ASCII ────────────────────────────────────────────────")
        try:
            graph.get_graph().print_ascii()
        except ImportError:
            print("(install grandalf for ASCII: uv pip install grandalf)")
        sys.exit(0)

    if not args.alert:
        parser.error("--alert is required unless --print-graph is set")

    alert_raw = Path(args.alert).read_text()
    graph = build_graph()
    config = {"configurable": {"thread_id": args.thread}}

    # Primeira execução — corre até ao interrupt()
    print("▶ Starting incident response graph...")
    result = graph.invoke({"alert_raw": alert_raw}, config=config)

    # Se parou no interrupt, aguarda input do engenheiro
    if graph.get_state(config).next:
        print("\nGraph paused at notify_and_wait. Enter engineer response:")
        engineer_input = input("> ").strip()

        # Retoma o grafo com a resposta do engenheiro
        result = graph.invoke(
            {"engineer_response": engineer_input},
            config=config,
        )

    print(f"\n✓ Incident closed. Decision: {result.get('decision')}")
