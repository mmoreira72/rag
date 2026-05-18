"""
Incident Response Graph — LangGraph.

Workflow:
  parse_alert
      ↓
  investigate   ← tool-calling loop (LLM drives evidence collection)
      │             1. check_optimizely  — flag changes first (context for log search)
      │             2. fetch_logs / fetch_k8s_events / fetch_istio_logs  — evidence
      │             3. list_revisions / fetch_timeseries / list_slos  — as needed
      ↓
  build_report  ← structured markdown + deterministic evidence section
      ↓
  notify_and_wait  ← interrupt() aguarda engenheiro
      ↓
  decide           ← rollback / escalate / monitor / close

Uso:
    python src/incident_graph.py --alert alert.json

Dependências:
    langgraph>=0.2
    langchain-openai>=0.1
"""
from __future__ import annotations

import json
import logging
import os
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
from config import LLM_BASE_URL, CHAT_MODEL
from rag import get_skills_with_confidence, format_skills_for_prompt

# ── Logging ──────────────────────────────────────────────────────────────────
# LOG_FILE env var sobrescreve o default. Ex: LOG_FILE=/var/log/incident.log
_log_path = os.getenv(
    "INCIDENT_LOG_FILE",
    f"/tmp/incident-graph-{datetime.now(timezone.utc).strftime('%Y%m%d')}.log",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_path, encoding="utf-8"),
    ],
)
log = logging.getLogger("incident_graph")

# Silencia loggers verbosos de bibliotecas externas
for _noisy in ("httpx", "httpcore", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log.info("[startup] log → %s", _log_path)

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
    resource_type: str      # cloud_run_revision | pubsub_subscription | etc.
    metric_type: str        # pubsub.googleapis.com/... | run.googleapis.com/...
    condition_type: str     # COMPARISON_GT | COMPARISON_LT
    region: str
    alert_ts: str           # ISO 8601
    condition_name: str
    alert_url: str

    # Logs de erro (alertas log-based: Cloud Run, Cloud SQL, GCE)
    error_logs: str

    # Métricas (alertas metric-based: Pub/Sub, SLO, throughput)
    metrics_data: str

    # Release correlation
    release_found: bool
    release_id: str
    release_ts: str
    release_type: str       # "cloud_run" | "optimizely" | ""
    release_logs: str

    # Investigation traceability
    skill_retrievals: list   # [{node, patterns:[{name,score}], confidence}] per node
    scripts_invoked: list    # tool calls made by the LLM
    analysis_log_url: str    # GCP Logging URL for the main log query

    # Tool-calling investigation result (replaces error_logs + metrics_data in new flow)
    investigation_result: str

    # Output
    report: str
    engineer_response: str  # preenchido após interrupt()
    decision: Literal["rollback", "escalate", "close", "monitor"]


# ── Skill retrieval ──────────────────────────────────────────────────────────
# DEPRECATED: use SKILL_ABSOLUTE_MIN_SCORE from config.py (via rag.py)
SKILL_MIN_SCORE = 0.70
SKILL_CANDIDATE_SCORE = 0.45
# get_skill() has moved to rag.py as a backward-compat shim.
# Use get_skills_with_confidence() + format_skills_for_prompt() for new code.


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

    resource      = inc.get("resource", {})
    labels        = resource.get("labels", {})
    resource_type = resource.get("type", "")
    monitoring_project = inc.get("monitoring_project", "")

    # project_id: tenta resource.labels primeiro
    project = labels.get("project_id") or inc.get("project_id") or ""

    # Se não veio no payload (ou é placeholder), resolve via monitoring API
    if not project or project.startswith("TODO"):
        metric_type = inc.get("metric", {}).get("type", "")
        resource_id = (
            labels.get("subscription_id")
            or labels.get("service_name")
            or labels.get("instance_id")
            or ""
        )
        project = _resolve_project(
            monitoring_project, metric_type, resource_type, resource_id, alert_ts
        ) or monitoring_project

    service = (
        labels.get("service_name")
        or labels.get("subscription_id")
        or labels.get("instance_id")
        or inc.get("service", "")
    )

    metric_type    = inc.get("metric", {}).get("type", "")
    condition_type = inc.get("condition_type", "COMPARISON_GT")

    log.info("[parse_alert] resource=%s condition=%s project=%s",
             resource_type, condition_type, project)

    return {
        "incident_id":    inc.get("incident_id") or inc.get("name", "unknown"),
        "project":        project,
        "service":        service,
        "resource_type":  resource_type,
        "metric_type":    metric_type,
        "condition_type": condition_type,
        "region":         labels.get("location") or labels.get("region") or "europe-west1",
        "alert_ts":       alert_ts,
        "condition_name": (
            inc.get("condition_name")
            or inc.get("policy_display_name")
            or inc.get("condition", {}).get("displayName", "")
        ),
        "alert_url":           inc.get("url") or inc.get("alert_url", ""),
        # Initialise investigation fields
        "error_logs":          "",
        "metrics_data":        "",
        "release_found":       False,
        "release_id":          "",
        "release_ts":          "",
        "release_type":        "",
        "release_logs":        "",
        "investigation_result": "",
        "skill_retrievals":    [],
        "scripts_invoked":     [],
        "analysis_log_url":    "",
    }


def _resolve_project(monitoring_project: str, metric_type: str,
                     resource_type: str, resource_id: str, alert_ts: str) -> str:
    """Descobre o project_id do recurso consultando o monitoring project.

    O monitoring project tem acesso cross-project às métricas — a resposta
    do time series inclui resource.labels.project_id do recurso real.
    """
    if not monitoring_project or not metric_type:
        return ""

    alert_dt   = datetime.fromisoformat(alert_ts.replace("Z", "+00:00"))
    start_time = (alert_dt - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time   = alert_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        import urllib.request, urllib.parse
        token = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        params = urllib.parse.urlencode({
            "filter": f'metric.type="{metric_type}"',
            "interval.startTime": start_time,
            "interval.endTime":   end_time,
            "view": "HEADERS",
        })
        url = (f"https://monitoring.googleapis.com/v3/projects/"
               f"{monitoring_project}/timeSeries?{params}")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        series = data.get("timeSeries", [])
        if series:
            pid = series[0].get("resource", {}).get("labels", {}).get("project_id", "")
            if pid:
                log.info("[resolve_project] found via REST: %s", pid)
                return pid
    except Exception:
        pass
    return ""


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
        log.info("[logging:%s] %d entries | stderr: %s",
                 label or "query", len(results), out.stderr[:120] if out.stderr else "none")
        return results
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        log.warning("[logging:%s] ERROR: %s", label or "query", e)
        return []


def fetch_errors(state: IncidentState) -> dict:
    """Busca logs adaptando o resource_type e a skill ao tipo de alerta."""
    alert_dt      = datetime.fromisoformat(state["alert_ts"].replace("Z", "+00:00"))
    window_end    = (alert_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    six_hours_ago = (alert_dt - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    project       = state["project"]
    service       = state["service"]
    rtype         = state.get("resource_type") or "cloud_run_revision"

    # Constrói o resource label correcto conforme o tipo de recurso
    # K8s: usa regex no pod_name (pod muda a cada restart) + fallback em textPayload
    _is_k8s = rtype in ("k8s_container", "k8s_pod")
    resource_label_map = {
        "cloud_run_revision":   f'resource.labels.service_name="{service}"',
        "pubsub_subscription":  f'resource.labels.subscription_id="{service}"',
        "cloudsql_database":    f'resource.labels.database_id="{service}"',
        "gce_instance":         f'resource.labels.instance_id="{service}"',
        "k8s_container":        f'resource.labels.pod_name=~"{service}.*"',
        "k8s_pod":              f'labels.source_workload="{service}"',
    }
    resource_filter = resource_label_map.get(rtype, f'resource.labels.service_name="{service}"')

    skill_query = " ".join(filter(None, [
        state.get("condition_name", ""),
        service,
        rtype,
    ]))
    skill_result = get_skills_with_confidence(skill_query)
    skill_context = format_skills_for_prompt(skill_result)

    # Para K8s: usa o projecto GKE, não o projecto de monitoring
    log_project = "hw-prod-sre-mgmt-k8s-fc7f" if _is_k8s else project

    log_filter = (
        f'resource.type="{rtype}"'
        f' AND {resource_filter}'
        f' AND timestamp>="{six_hours_ago}"'
        f' AND timestamp<="{window_end}"'
    )
    import urllib.parse as _up
    analysis_log_url = (
        f"https://console.cloud.google.com/logs/query"
        f";query={_up.quote(log_filter)}?project={log_project}"
    )

    all_logs = _run_gcloud_logging(log_filter, log_project, limit=150, label=f"6h:{rtype}")

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

    skill_prefix = skill_context + "\n\n" if skill_context else ""
    system = f"""{skill_prefix}Analyze these Cloud Run logs and identify ALL error patterns.
Pay special attention to:
- "FATAL ERROR" / "heap out of memory" / "Allocation failed" → JavaScript OOM (ROOT CAUSE)
- "LIVENESS probe failed" → instance killed after OOM (SYMPTOM)
- Connection errors / 503 → consequence of restarts (SYMPTOM)

If OOM errors predate the liveness failures, mark them as root_cause=true.

Return JSON: [{{"count": N, "message": "...", "first_seen": "ISO8601", "root_cause": true/false}}]"""

    summary = call_llm(system, combined[:8000])

    retrievals = list(state.get("skill_retrievals") or [])
    retrievals.append({
        "node": "fetch_errors",
        "confidence": skill_result.confidence,
        "patterns": [{"name": s.name, "score": round(s.score, 3)} for s in skill_result.skills],
    })
    return {
        "error_logs": summary,
        "skill_retrievals": retrievals,
        "analysis_log_url": analysis_log_url,
        "scripts_invoked": list(state.get("scripts_invoked") or []) + [
            f"gcloud logging read (fetch_errors, {len(all_logs)} entries)"
        ],
    }


def fetch_metrics(state: IncidentState) -> dict:
    """Para alertas metric-based: consulta time series e descreve o recurso."""
    skill_query_map = {
        "pubsub_subscription": "Pub/Sub subscription backlog growing dropping NOT_FOUND",
        "cloudsql_database":   "Cloud SQL slow query lock contention missing index replication",
        "gce_instance":        "GCE instance CPU memory disk metrics",
    }
    skill_query = skill_query_map.get(
        state.get("resource_type", ""),
        "GCP service incident investigation root cause"
    )
    skill_result = get_skills_with_confidence(skill_query)
    skill_context = format_skills_for_prompt(skill_result)

    alert_dt   = datetime.fromisoformat(state["alert_ts"].replace("Z", "+00:00"))
    start_time = (alert_dt - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time   = (alert_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    project    = state["project"]
    service    = state["service"]
    rtype      = state.get("resource_type", "")
    mtype      = state.get("metric_type", "")

    results = {}

    # 1. Time series via REST API (gcloud monitoring time-series não existe em todas as versões)
    # Usa o monitoring project para acesso cross-project
    monitoring_project = "hw-prod-svc-sre-mon-a1f1"  # TODO: tornar configurável
    try:
        token_out = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, timeout=10)
        token = token_out.stdout.strip()

        import urllib.request, urllib.parse
        # Filtra pelo recurso específico para não trazer todas as subscriptions
        resource_label_key = {
            "pubsub_subscription": "subscription_id",
            "cloud_run_revision":  "service_name",
            "cloudsql_database":   "database_id",
        }.get(rtype, "service_name")
        metric_filter = (
            f'metric.type="{mtype}"'
            f' AND resource.labels.{resource_label_key}="{service}"'
        )
        params = urllib.parse.urlencode({
            "filter": metric_filter,
            "interval.startTime": start_time,
            "interval.endTime":   end_time,
            "view": "FULL",
        })
        url = (f"https://monitoring.googleapis.com/v3/projects/"
               f"{monitoring_project}/timeSeries?{params}")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            ts_json = json.loads(resp.read())

        ts_data = ts_json.get("timeSeries", [])
        log.info("[metrics:timeseries] %d series via REST", len(ts_data))

        # Extrai project_id do primeiro resultado se ainda não resolvido
        if ts_data and (not project or project == monitoring_project):
            resolved = ts_data[0].get("resource", {}).get("labels", {}).get("project_id", "")
            if resolved:
                project = resolved
                log.info("[resolve_project] resolved from timeseries: %s", project)

        results["timeseries"] = [
            {
                "resource_labels": s.get("resource", {}).get("labels", {}),
                "metric_labels":   s.get("metric", {}).get("labels", {}),
                "points": [
                    {"time": p["interval"]["endTime"],
                     "value": p["value"].get("int64Value") or p["value"].get("doubleValue")}
                    for p in s.get("points", [])[:10]
                ],
            }
            for s in ts_data[:3]
        ]
    except Exception as e:
        log.warning("[metrics:timeseries] ERROR: %s", e)
        results["timeseries"] = []

    # 2. Descreve o recurso (Pub/Sub subscription, etc.)
    if rtype == "pubsub_subscription":
        desc_cmd = [
            "gcloud", "pubsub", "subscriptions", "describe", service,
            f"--project={project}",
            "--format=json",
        ]
        try:
            out = subprocess.run(desc_cmd, capture_output=True, text=True, timeout=20)
            if out.returncode == 0:
                results["resource_describe"] = json.loads(out.stdout or "{}")
                log.info("[metrics:describe] OK project=%s", project)
            else:
                # NOT_FOUND é diagnóstico — pode significar subscription deletada
                err = out.stderr.strip()
                results["resource_describe"] = {"error": err}
                results["subscription_not_found"] = "NOT_FOUND" in err
                log.warning("[metrics:describe] %s", err[:120])
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            log.warning("[metrics:describe] ERROR: %s", e)
            results["resource_describe"] = {}

    # Pré-sumariza para não encher o contexto do LLM
    ts_summary = []
    for s in results.get("timeseries", [])[:2]:
        pts = s.get("points", [])
        ts_summary.append({
            "resource": s.get("resource_labels", {}),
            "latest_value": pts[0]["value"] if pts else None,
            "oldest_value": pts[-1]["value"] if pts else None,
            "point_count":  len(pts),
            "first_time":   pts[-1]["time"] if pts else None,
            "last_time":    pts[0]["time"] if pts else None,
        })

    desc = results.get("resource_describe", {})
    desc_summary = {
        "name": desc.get("name", ""),
        "topic": desc.get("topic", ""),
        "messageRetentionDuration": desc.get("messageRetentionDuration", ""),
        "state": desc.get("pushConfig", {}).get("pushEndpoint", "pull"),
    } if desc else {}

    condition_type = state.get("condition_type", "COMPARISON_GT")
    condition_direction = (
        "BELOW threshold (too few messages — publisher may have stopped)"
        if condition_type == "COMPARISON_LT"
        else "ABOVE threshold (too many messages — backlog growing)"
    )

    compact = json.dumps({
        "alert_condition": state.get("condition_name"),
        "condition_direction": condition_direction,
        "metric_type": mtype,
        "timeseries": ts_summary,
        "resource": desc_summary,
    })

    not_found = results.get("subscription_not_found", False)
    not_found_hint = (
        "\nCRITICAL: gcloud reports the subscription does NOT EXIST (NOT_FOUND). "
        "This is likely the root cause — a deleted or missing subscription receives 0 messages."
        if not_found else ""
    )

    skill_prefix = skill_context + "\n\n" if skill_context else ""
    system = f"""{skill_prefix}Analyze these GCP {rtype} metrics.
IMPORTANT: The alert fired because the metric is {condition_direction}.{not_found_hint}
Identify: trend, when it deviated from normal, likely cause.
Return JSON: [{{"metric": "...", "trend": "...", "likely_cause": "...", "root_cause": true}}]"""

    summary = call_llm(system, compact[:2000])

    retrievals = list(state.get("skill_retrievals") or [])
    retrievals.append({
        "node": "fetch_metrics",
        "confidence": skill_result.confidence,
        "patterns": [{"name": s.name, "score": round(s.score, 3)} for s in skill_result.skills],
    })
    scripts = list(state.get("scripts_invoked") or []) + [
        f"Monitoring REST API timeseries ({mtype})",
    ]
    if results.get("resource_describe"):
        scripts.append(f"gcloud pubsub subscriptions describe ({service})")
    return {
        "metrics_data": summary,
        "error_logs": "",
        "project": project,
        "skill_retrievals": retrievals,
        "scripts_invoked": scripts,
    }


def check_release(state: IncidentState) -> dict:
    """Verifica se houve release no Cloud Deploy na hora anterior ao alerta."""
    skill_result = get_skills_with_confidence("Cloud Run revision release rollout deploy history timestamp")
    skill_context = format_skills_for_prompt(skill_result)  # noqa: F841 — used if LLM call added later

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

    scripts = list(state.get("scripts_invoked") or []) + [
        f"gcloud run revisions list ({service}, {len(revisions)} found)"
    ]
    retrievals = list(state.get("skill_retrievals") or [])
    retrievals.append({
        "node": "check_release",
        "confidence": skill_result.confidence,
        "patterns": [{"name": s.name, "score": round(s.score, 3)} for s in skill_result.skills],
    })

    for rev in revisions:
        created = rev.get("metadata", {}).get("creationTimestamp", "")
        if created and window_start <= created <= state["alert_ts"]:
            return {
                "release_found": True,
                "release_id":    rev.get("metadata", {}).get("name", ""),
                "release_ts":    created,
                "release_type":  "cloud_run",
                "skill_retrievals": retrievals,
                "scripts_invoked": scripts,
            }

    # 2. Optimizely feature flag changes (Cloud Function logs in monitoring project)
    # Deployed in hw-prod-svc-sre-mon-a1f1 — logs regardless of service project
    monitoring_project = "hw-prod-svc-sre-mon-a1f1"
    opt_logs = _run_gcloud_logging(
        (
            f'resource.type="cloud_run_revision"'
            f' AND resource.labels.service_name="optimizely-webhook"'
            f' AND labels."source"="optimizely"'
            f' AND severity="NOTICE"'
            f' AND timestamp>="{window_start}"'
            f' AND timestamp<="{state["alert_ts"]}"'
        ),
        monitoring_project,
        limit=5,
        label="optimizely",
    )

    scripts.append(f"gcloud logging read optimizely-webhook ({len(opt_logs)} entries)")
    if opt_logs:
        first = opt_logs[0]
        payload = first.get("jsonPayload", {})
        event   = payload.get("event", first.get("labels", {}).get("event", ""))
        rev     = payload.get("revision", "")
        ts      = first.get("timestamp", "")
        log.info("[check_release] Optimizely event: %s rev=%s ts=%s", event, rev, ts)
        return {
            "release_found": True,
            "release_id":    f"optimizely-rev-{rev} ({event})",
            "release_ts":    ts,
            "release_type":  "optimizely",
            "skill_retrievals": retrievals,
            "scripts_invoked": scripts,
        }

    return {
        "release_found": False, "release_id": "", "release_ts": "", "release_type": "",
        "skill_retrievals": retrievals,
        "scripts_invoked": scripts,
    }


def fetch_release_logs(state: IncidentState) -> dict:
    """Repete fetch_errors mas centrado no timestamp da release."""
    if not state.get("release_ts"):
        return {"release_logs": "[]"}

    skill_result = get_skills_with_confidence("Cloud Run service HTTP errors OOM liveness probe health check noise")
    skill_context = format_skills_for_prompt(skill_result)

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

    skill_prefix = skill_context + "\n\n" if skill_context else ""
    system = f"""{skill_prefix}Summarise errors that appeared AFTER the release deploy.
Return JSON: [{{"count": N, "message": "...", "first_seen": "ISO8601"}}]"""

    summary = call_llm(system, logs_raw[:6000])

    retrievals = list(state.get("skill_retrievals") or [])
    retrievals.append({
        "node": "fetch_release_logs",
        "confidence": skill_result.confidence,
        "patterns": [{"name": s.name, "score": round(s.score, 3)} for s in skill_result.skills],
    })
    return {
        "release_logs": summary,
        "skill_retrievals": retrievals,
        "scripts_invoked": list(state.get("scripts_invoked") or []) + [
            f"gcloud logging read (fetch_release_logs, ±30min of release)"
        ],
    }


def investigate(state: IncidentState) -> dict:
    """Tool-calling investigation loop — LLM decides which commands to run."""
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    from tools import INVESTIGATION_TOOLS, TOOLS_BY_NAME

    rtype = state.get("resource_type", "cloud_run_revision")
    # Build skill query from actual alert data so RAG finds the most relevant pattern
    # for any of the 400+ alert types — not a hardcoded mapping
    skill_query = " ".join(filter(None, [
        state.get("condition_name", ""),
        state.get("service", ""),
        rtype,
    ]))
    skill_result = get_skills_with_confidence(skill_query)
    skill_context = format_skills_for_prompt(skill_result)

    alert_dt     = datetime.fromisoformat(state["alert_ts"].replace("Z", "+00:00"))
    window_start = (alert_dt - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    window_end   = (alert_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    project      = state["project"]
    service      = state["service"]

    import urllib.parse as _up
    _is_k8s = rtype in ("k8s_container", "k8s_pod")
    _log_project = "hw-prod-sre-mgmt-k8s-fc7f" if _is_k8s else project
    if _is_k8s:
        default_filter = (
            f'labels.source_workload="{service}"'
            f' AND timestamp>="{window_start}"'
            f' AND timestamp<="{window_end}"'
        )
    else:
        default_filter = (
            f'resource.type="{rtype}"'
            f' AND resource.labels.service_name="{service}"'
            f' AND timestamp>="{window_start}"'
            f' AND timestamp<="{window_end}"'
        )
    analysis_log_url = (
        f"https://console.cloud.google.com/logs/query"
        f";query={_up.quote(default_filter)}?project={_log_project}"
    )

    _is_k8s_inv = rtype in ("k8s_container", "k8s_pod")
    context = json.dumps({
        "service":       service,
        "project":       project,
        "resource_type": rtype,
        "condition":     state.get("condition_name"),
        "alert_ts":      state["alert_ts"],
        "window_start":  window_start,
        "window_end":    window_end,
        "region":        state.get("region", "europe-west1"),
        **({"k8s_project": "hw-prod-sre-mgmt-k8s-fc7f",
            "k8s_namespace": "social-platform"} if _is_k8s_inv else {}),
    }, indent=2)

    # Tool selection by resource_type
    _K8S_RTYPES = ("k8s_container", "k8s_pod")
    if rtype in _K8S_RTYPES:
        active_tool_names = {
            "fetch_logs", "check_optimizely",
            "fetch_k8s_events", "fetch_istio_logs",
            "fetch_container_logs", "fetch_pod_metrics", "fetch_dapr_logs",
        }
    elif rtype == "pubsub_subscription":
        active_tool_names = {"fetch_logs", "check_optimizely", "describe_subscription", "fetch_timeseries", "list_slos"}
    else:
        # cloud_run_revision, cloudsql_database, gce_instance, slo-based — always include list_slos
        active_tool_names = {"fetch_logs", "check_optimizely", "list_revisions", "list_slos"}
    active_tools = [t for t in INVESTIGATION_TOOLS if t.name in active_tool_names]

    skill_prefix = skill_context + "\n\n" if skill_context else ""
    _k8s_project_note = (
        "For ALL K8s tools: always use project=hw-prod-sre-mgmt-k8s-fc7f "
        "and namespace=social-platform — never use the monitoring project or namespace=default.\n"
    ) if rtype in _K8S_RTYPES else ""

    system = (
        f"{skill_prefix}"
        "You are an SRE investigating a GCP production incident.\n\n"
        "Investigation order:\n"
        "1) check_optimizely FIRST — flag changes give context for all other queries.\n"
        "2) Fetch evidence using the available tools (logs, events, metrics, SLOs).\n"
        "3) Correlate timestamps — root cause precedes symptoms.\n\n"
        f"{_k8s_project_note}"
        "Write a concise summary: errors found | changes detected | root cause | action.\n"
        "Do NOT invent data. If a tool returns nothing, say so explicitly."
    )

    llm_with_tools = llm.bind_tools(active_tools)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=context),
    ]

    scripts_used: list[str] = []
    called_tool_keys: set[str] = set()
    max_iterations = 6
    final_content  = ""

    try:
        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            final_content = getattr(response, "content", "") or ""

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                tool_key  = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                args_repr = ", ".join(f"{k}={str(v)[:40]}" for k, v in tool_args.items())

                if tool_key in called_tool_keys:
                    log.info("[tool_call] SKIP duplicate: %s(%s)", tool_name, args_repr)
                    messages.append(ToolMessage(
                        content="Duplicate — already called. Do not repeat.",
                        tool_call_id=tc["id"],
                    ))
                    continue

                called_tool_keys.add(tool_key)
                log.info("[tool_call] %s(%s)", tool_name, args_repr)
                scripts_used.append(f"{tool_name}({args_repr})")

                tool = TOOLS_BY_NAME.get(tool_name)
                result = tool.invoke(tool_args) if tool else f"Unknown tool: {tool_name}"
                messages.append(ToolMessage(
                    content=str(result)[:1500],   # reduced from 4000
                    tool_call_id=tc["id"],
                ))

    except Exception as e:
        log.warning("[investigate] tool-calling failed: %s — falling back to no-tool analysis", e)
        # Fallback: ask LLM to analyse with just the alert context (no tools)
        fallback_resp = llm.invoke([
            SystemMessage(content=skill_prefix + "Analyse this GCP alert and summarise what likely happened. "
                          "State clearly that automated log retrieval failed."),
            HumanMessage(content=context),
        ])
        final_content = getattr(fallback_resp, "content", str(fallback_resp))

    retrievals = [{
        "node": "investigate",
        "confidence": skill_result.confidence,
        "patterns": [{"name": s.name, "score": round(s.score, 3)} for s in skill_result.skills],
    }]

    return {
        "investigation_result": final_content,
        "skill_retrievals":     retrievals,
        "scripts_invoked":      scripts_used,
        "analysis_log_url":     analysis_log_url,
        # Keep fields for build_report compatibility
        "error_logs":    "",
        "metrics_data":  "",
        "release_found": False,
        "release_id":    "",
        "release_ts":    "",
        "release_type":  "",
        "release_logs":  "",
    }


def build_report(state: IncidentState) -> dict:
    """Monta o relatório de incidente com toda a informação recolhida."""
    skill_result = get_skills_with_confidence("GCP incident report template SRE post-mortem root cause")
    skill_context = format_skills_for_prompt(skill_result)

    project    = state.get("project", "")
    service    = state.get("service", "")
    alert_ts   = state.get("alert_ts", "")
    alert_url  = state.get("alert_url", "")
    # Gera URLs reais para os links do relatório
    try:
        alert_dt   = datetime.fromisoformat(alert_ts.replace("Z", "+00:00"))
        ts_start   = (alert_dt - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ts_end     = (alert_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        ts_start = ts_end = ""

    _is_slo = (
        "slo" in state.get("condition_name", "").lower()
        or "slo" in state.get("metric_type", "").lower()
    )
    if _is_slo:
        log_url = (
            f"https://console.cloud.google.com/monitoring/services"
            f"/{service}?project={project}"
        )
    else:
        import urllib.parse
        _is_k8s_br = state.get("resource_type", "") in ("k8s_container", "k8s_pod")
        _br_project = "hw-prod-sre-mgmt-k8s-fc7f" if _is_k8s_br else project
        if _is_k8s_br:
            _raw_filter = (
                f'labels.source_workload="{service}"'
                f' AND timestamp>="{ts_start}" AND timestamp<="{ts_end}"'
            )
        else:
            _raw_filter = (
                f'resource.labels.service_name="{service}"'
                f' AND timestamp>="{ts_start}" AND timestamp<="{ts_end}"'
            )
        log_filter = urllib.parse.quote(_raw_filter)
        log_url = (
            f"https://console.cloud.google.com/logs/query"
            f";query={log_filter}?project={_br_project}"
        )

    release_ts  = state.get("release_ts", "")
    if release_ts:
        try:
            rel_dt     = datetime.fromisoformat(release_ts.replace("Z", "+00:00"))
            rel_start  = (rel_dt - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
            rel_end    = (rel_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            import urllib.parse as _up
            _is_k8s_rel = state.get("resource_type", "") in ("k8s_container", "k8s_pod")
            _rel_svc_filter = (
                f'labels.source_workload="{service}"' if _is_k8s_rel
                else f'resource.labels.service_name="{service}"'
            )
            rel_filter = _up.quote(
                f'{_rel_svc_filter}'
                f' AND timestamp>="{rel_start}" AND timestamp<="{rel_end}"'
            )
            release_log_url = (
                f"https://console.cloud.google.com/logs/query"
                f";query={rel_filter}?project={project}"
            )
        except Exception:
            release_log_url = log_url
    else:
        release_log_url = log_url

    # URLs are injected deterministically in the evidence section — keep them OUT of LLM context
    investigation = state.get("investigation_result", "")
    context = json.dumps({
        "incident_id":          state.get("incident_id"),
        "service":              service,
        "project":              project,
        "alert_ts":             alert_ts,
        "condition":            state.get("condition_name"),
        "investigation_result": investigation,
    }, indent=2)

    skill_prefix = skill_context + "\n\n" if skill_context else ""
    system = f"""{skill_prefix}Using the investigation result below, generate a concise incident report in Markdown.
RULES:
- Base the report ONLY on evidence in investigation_result. Do not invent data.
- If investigation_result mentions an Optimizely event, label it FEATURE FLAG CHANGE (not a code deploy).
- If no release was found, write "No release identified in the investigation window."
- DO NOT generate any URLs, links, or placeholders — log links are added separately.
- DO NOT speculate about root causes not present in investigation_result.
  If no concrete errors found, write "No concrete evidence found — manual investigation required."
- Sections: ## Impact, ## Timeline, ## Error/Metric Pattern, ## Release Correlation, ## Recommended Action.
- End with: RECOMMENDED ACTION: rollback | escalate | monitor | close"""

    report = call_llm(system, context[:5000])

    # ── Deterministic evidence section (appended after LLM report) ──────────
    evidence_lines = ["\n\n---\n\n## Investigation Evidence\n"]

    # Patterns considered per node
    retrievals = state.get("skill_retrievals") or []
    if retrievals:
        evidence_lines.append("### Patterns Considered\n")
        evidence_lines.append("| Node | Pattern | Score | Confidence |")
        evidence_lines.append("|---|---|---|---|")
        for r in retrievals:
            for p in r.get("patterns", []):
                evidence_lines.append(
                    f"| `{r['node']}` | `{p['name']}` | {p['score']} | {r['confidence']} |"
                )
        evidence_lines.append("")

    # Scripts / commands invoked
    scripts = state.get("scripts_invoked") or []
    if scripts:
        evidence_lines.append("### Scripts / Commands Invoked\n")
        for s in scripts:
            evidence_lines.append(f"- `{s}`")
        evidence_lines.append("")

    # Log links
    analysis_log_url = state.get("analysis_log_url", "")
    if analysis_log_url or log_url or release_log_url:
        evidence_lines.append("### Log Links\n")
        if analysis_log_url:
            evidence_lines.append(f"- [Logs analyzed by LLM (fetch_errors query)]({analysis_log_url})")
        if log_url:
            evidence_lines.append(f"- [All service logs — alert window]({log_url})")
        if release_log_url and release_log_url != log_url:
            evidence_lines.append(f"- [Logs — around release/feature flag event]({release_log_url})")
        if alert_url:
            evidence_lines.append(f"- [Alert policy]({alert_url})")

    report = report + "\n".join(evidence_lines)
    return {"report": report}


def notify_and_wait(state: IncidentState) -> dict:
    """Envia notificação ao engenheiro e aguarda resposta (LangGraph interrupt)."""
    log.info("\n" + "="*60 + "\nINCIDENT REPORT\n" + "="*60 + "\n%s\n" + "="*60, state["report"])

    # Salva o report em markdown
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    inc_id   = state.get("incident_id", "unknown").replace("/", "-")
    service  = state.get("service", "service").replace("/", "-")
    md_path  = f"/tmp/incident-{service}-{inc_id}-{ts}.md"
    try:
        Path(md_path).write_text(state["report"], encoding="utf-8")
        log.info("[report] saved → %s", md_path)
        print(f"\n📄 Report saved: {md_path}")
    except Exception as e:
        log.warning("[report] failed to save: %s", e)

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

    log.info("→ Decision: %s", decision)

    # TODO: executar a acção (chamar terragrunt, gcloud deploy rollback, etc.)

    return {"decision": decision}


# ── Edges ────────────────────────────────────────────────────────────────────

def fetch_slo(state: IncidentState) -> dict:
    """Para alertas SLO burn rate: consulta compliance e error budget via REST API."""
    skill_result = get_skills_with_confidence("SLO burn rate error budget availability latency Cloud Run")
    skill_context = format_skills_for_prompt(skill_result)

    monitoring_project = state.get("project") or "hw-prod-svc-sre-mon-a1f1"
    service_name       = state.get("service") or "categories-service-api"
    alert_ts           = state.get("alert_ts", "")

    log.info("[slo] querying SLOs for service=%s project=%s", service_name, monitoring_project)

    try:
        token = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, timeout=10).stdout.strip()

        import urllib.request
        # 1. Lista os SLOs do serviço
        url = (f"https://monitoring.googleapis.com/v3/projects/"
               f"{monitoring_project}/services/{service_name}/serviceLevelObjectives")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            slos_data = json.loads(resp.read())
        slos = slos_data.get("serviceLevelObjectives", [])
        log.info("[slo] found %d SLOs", len(slos))

        slo_summary = []
        for slo in slos[:3]:
            slo_id  = slo.get("name", "").split("/")[-1]
            goal    = slo.get("goal", 0)
            slo_type = "availability" if "availability" in slo_id else "latency"

            # 2. Burn rate via MQL query API (select_slo_burn_rate é MQL, não filter)
            alert_dt   = datetime.fromisoformat(alert_ts.replace("Z", "+00:00"))
            start_time = (alert_dt - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_time   = alert_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            slo_path   = f"projects/{monitoring_project}/services/{service_name}/serviceLevelObjectives/{slo_id}"
            mql_body   = json.dumps({
                "query": (
                    f'fetch service::slo/window_burn_rate'
                    f' | filter slo_name = "{slo_id}"'
                    f' | within {start_time}, {end_time}'
                )
            }).encode()
            mql_url = (f"https://monitoring.googleapis.com/v3/projects/"
                       f"{monitoring_project}/timeSeries:query")
            mql_req = urllib.request.Request(
                mql_url,
                data=mql_body,
                headers={"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"},
                method="POST",
            )
            burn_rate = None
            try:
                with urllib.request.urlopen(mql_req, timeout=15) as r:
                    mql_data = json.loads(r.read())
                ts_list = mql_data.get("timeSeriesData", [])
                if ts_list:
                    pts = ts_list[0].get("pointData", [])
                    if pts:
                        burn_rate = pts[0].get("values", [{}])[0].get("doubleValue")
                log.info("[slo] %s: burn_rate=%s", slo_id, burn_rate)
            except Exception as e:
                log.warning("[slo] %s: burn_rate query failed: %s", slo_id, e)

            slo_summary.append({
                "slo_id":    slo_id,
                "type":      slo_type,
                "goal":      f"{goal*100:.1f}%",
                "burn_rate": burn_rate,
                "budget_exhausted_in_hours": round(720 / burn_rate, 1) if burn_rate and burn_rate > 0 else "unknown — query live data in console",
            })

    except Exception as e:
        log.warning("[slo] ERROR: %s", e)
        slo_summary = []

    # Tenta buscar logs de erro do serviço (via project do monitoring — cross-project)
    alert_dt      = datetime.fromisoformat(alert_ts.replace("Z", "+00:00"))
    six_hours_ago = (alert_dt - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    window_end    = (alert_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    error_logs = _run_gcloud_logging(
        (
            f'resource.type="cloud_run_revision"'
            f' AND resource.labels.service_name="{service_name}"'
            f' AND timestamp>="{six_hours_ago}"'
            f' AND timestamp<="{window_end}"'
        ),
        monitoring_project,
        limit=50,
        label="slo-errors",
    )
    log_lines = []
    for entry in error_logs:
        ts  = entry.get("timestamp", "")
        sev = entry.get("severity", "")
        msg = (entry.get("textPayload")
               or entry.get("jsonPayload", {}).get("message", "")
               or "")
        if msg:
            log_lines.append(f"[{ts}] [{sev}] {msg[:200]}")

    # Console link directo para o SLO dashboard
    console_url = (
        f"https://console.cloud.google.com/monitoring/services"
        f"/{service_name}?project={monitoring_project}"
    )

    compact = json.dumps({
        "service":     service_name,
        "alert":       state.get("condition_name"),
        "threshold":   "burn rate > 10x (error budget exhausts in ~3 days at this rate)",
        "slos":        slo_summary,
        "error_logs":  log_lines[:20] or ["No Cloud Run logs found — service may be in a different project"],
        "console":     console_url,
    })

    skill_prefix = skill_context + "\n\n" if skill_context else ""
    system = f"""{skill_prefix}Analyze this SLO burn rate incident for {service_name}.
Burn rate > 10x at 1h window = error budget exhausts in ~3 days.
Use error_logs to identify the underlying cause (errors, latency spikes).
Include the console URL in the report.
Return JSON: [{{"slo": "...", "goal": "...", "likely_cause": "...", "root_cause": true/false}}]"""

    summary = call_llm(system, compact[:3000])
    return {"metrics_data": summary, "error_logs": "\n".join(log_lines[:10])}


# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(IncidentState)

    g.add_node("parse_alert",     parse_alert)
    g.add_node("investigate",     investigate)   # tool-calling loop
    g.add_node("build_report",    build_report)
    g.add_node("notify_and_wait", notify_and_wait)
    g.add_node("decide",          decide)

    g.set_entry_point("parse_alert")
    g.add_edge("parse_alert",     "investigate")
    g.add_edge("investigate",     "build_report")
    g.add_edge("build_report",    "notify_and_wait")
    g.add_edge("notify_and_wait", "decide")
    g.add_edge("decide",          END)

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
