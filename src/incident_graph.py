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
from config import LLM_BASE_URL, CHAT_MODEL
from rag import get_skills_with_confidence, format_skills_for_prompt

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
    alert_type: str         # "log-based" | "metric-based"
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
    scripts_invoked: list    # ["gcloud run revisions list", "optimizely logs", ...]
    analysis_log_url: str    # exact GCP Logging URL used to fetch logs for LLM

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

    # Classificação do tipo de alerta
    LOG_BASED_RESOURCES = {"cloud_run_revision", "gae_app", "gce_instance",
                           "cloudsql_database", "k8s_container"}
    is_slo = (
        "select_slo_burn_rate" in metric_type
        or "slo" in (inc.get("policy_display_name", "") or "").lower()
        or bool(inc.get("slo_resources"))
    )
    if is_slo:
        alert_type = "slo-based"
    elif resource_type in LOG_BASED_RESOURCES and condition_type != "COMPARISON_LT":
        alert_type = "log-based"
    else:
        alert_type = "metric-based"

    print(f"  [parse_alert] resource={resource_type} condition={condition_type} "
          f"→ alert_type={alert_type} project={project}")

    return {
        "incident_id":    inc.get("incident_id") or inc.get("name", "unknown"),
        "project":        project,
        "service":        service,
        "resource_type":  resource_type,
        "metric_type":    metric_type,
        "condition_type": condition_type,
        "alert_type":     alert_type,
        "region":         labels.get("location") or labels.get("region") or "europe-west1",
        "alert_ts":       alert_ts,
        "condition_name": (
            inc.get("condition_name")
            or inc.get("policy_display_name")
            or inc.get("condition", {}).get("displayName", "")
        ),
        "alert_url":      inc.get("url") or inc.get("alert_url", ""),
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
                print(f"  [resolve_project] found via REST: {pid}")
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
        print(f"  [logging:{label or 'query'}] {len(results)} entries | stderr: {out.stderr[:120] if out.stderr else 'none'}")
        return results
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [logging:{label or 'query'}] ERROR: {e}")
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
    resource_label_map = {
        "cloud_run_revision":   f'resource.labels.service_name="{service}"',
        "pubsub_subscription":  f'resource.labels.subscription_id="{service}"',
        "cloudsql_database":    f'resource.labels.database_id="{service}"',
        "gce_instance":         f'resource.labels.instance_id="{service}"',
    }
    resource_filter = resource_label_map.get(rtype, f'resource.labels.service_name="{service}"')

    # skill retrieval dinâmico baseado no tipo de recurso
    skill_query = {
        "cloud_run_revision":  "Cloud Run service HTTP errors OOM liveness probe health check noise",
        "pubsub_subscription": "Pub/Sub subscription backlog growing dropping NOT_FOUND",
        "cloudsql_database":   "Cloud SQL slow query lock contention missing index replication",
    }.get(rtype, "GCP service incident investigation root cause")
    skill_result = get_skills_with_confidence(skill_query)
    skill_context = format_skills_for_prompt(skill_result)

    log_filter = (
        f'resource.type="{rtype}"'
        f' AND {resource_filter}'
        f' AND timestamp>="{six_hours_ago}"'
        f' AND timestamp<="{window_end}"'
    )
    import urllib.parse as _up
    analysis_log_url = (
        f"https://console.cloud.google.com/logs/query"
        f";query={_up.quote(log_filter)}?project={project}"
    )

    all_logs = _run_gcloud_logging(log_filter, project, limit=150, label=f"6h:{rtype}")

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
        print(f"  [metrics:timeseries] {len(ts_data)} series via REST")

        # Extrai project_id do primeiro resultado se ainda não resolvido
        if ts_data and (not project or project == monitoring_project):
            resolved = ts_data[0].get("resource", {}).get("labels", {}).get("project_id", "")
            if resolved:
                project = resolved
                print(f"  [resolve_project] resolved from timeseries: {project}")

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
        print(f"  [metrics:timeseries] ERROR: {e}")
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
                print(f"  [metrics:describe] OK project={project}")
            else:
                # NOT_FOUND é diagnóstico — pode significar subscription deletada
                err = out.stderr.strip()
                results["resource_describe"] = {"error": err}
                results["subscription_not_found"] = "NOT_FOUND" in err
                print(f"  [metrics:describe] {err[:120]}")
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  [metrics:describe] ERROR: {e}")
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
        print(f"  [check_release] Optimizely event: {event} rev={rev} ts={ts}")
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


def build_report(state: IncidentState) -> dict:
    """Monta o relatório de incidente com toda a informação recolhida."""
    skill_result = get_skills_with_confidence("GCP incident report template SRE post-mortem root cause")
    skill_context = format_skills_for_prompt(skill_result)

    project    = state.get("project", "")
    service    = state.get("service", "")
    alert_ts   = state.get("alert_ts", "")
    alert_url  = state.get("alert_url", "")
    alert_type = state.get("alert_type", "log-based")

    # Gera URLs reais para os links do relatório
    try:
        alert_dt   = datetime.fromisoformat(alert_ts.replace("Z", "+00:00"))
        ts_start   = (alert_dt - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ts_end     = (alert_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        ts_start = ts_end = ""

    if alert_type == "slo-based":
        log_url = (
            f"https://console.cloud.google.com/monitoring/services"
            f"/{service}?project={project}"
        )
    else:
        import urllib.parse
        log_filter = urllib.parse.quote(
            f'resource.labels.service_name="{service}"'
            f' AND timestamp>="{ts_start}" AND timestamp<="{ts_end}"'
        )
        log_url = (
            f"https://console.cloud.google.com/logs/query"
            f";query={log_filter}?project={project}"
        )

    release_ts  = state.get("release_ts", "")
    if release_ts:
        try:
            rel_dt     = datetime.fromisoformat(release_ts.replace("Z", "+00:00"))
            rel_start  = (rel_dt - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
            rel_end    = (rel_dt + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            import urllib.parse as _up
            rel_filter = _up.quote(
                f'resource.labels.service_name="{service}"'
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

    context = json.dumps({
        "incident_id":      state.get("incident_id"),
        "service":          service,
        "project":          project,
        "alert_ts":         alert_ts,
        "condition":        state.get("condition_name"),
        "alert_type":       alert_type,
        "error_logs":       state.get("error_logs"),
        "metrics_data":     state.get("metrics_data"),
        "release_found":    state.get("release_found"),
        "release_type":     state.get("release_type"),
        "release_id":       state.get("release_id"),
        "release_ts":       release_ts,
        "release_logs":     state.get("release_logs"),
        "alert_url":        alert_url or f"https://console.cloud.google.com/monitoring?project={project}",
        "log_url":          log_url,
        "release_log_url":  release_log_url,
    }, indent=2)

    skill_prefix = skill_context + "\n\n" if skill_context else ""
    system = f"""{skill_prefix}Using the incident data below, generate a concise incident report in Markdown.
For metric-based/slo-based alerts use metrics_data; for log-based use error_logs.
If release_type is "optimizely", highlight it as a FEATURE FLAG CHANGE, not a code deploy.
Use the exact URLs provided in log_url, release_log_url, and alert_url — do NOT use placeholder variables like ${{ALERT_URL}}.
Include: Impact, Timeline, Error/Metric Pattern, Release Correlation, Recommended Action.
End with a RECOMMENDED ACTION: one of rollback / escalate / monitor / close"""

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

def fetch_slo(state: IncidentState) -> dict:
    """Para alertas SLO burn rate: consulta compliance e error budget via REST API."""
    skill_result = get_skills_with_confidence("SLO burn rate error budget availability latency Cloud Run")
    skill_context = format_skills_for_prompt(skill_result)

    monitoring_project = state.get("project") or "hw-prod-svc-sre-mon-a1f1"
    service_name       = state.get("service") or "categories-service-api"
    alert_ts           = state.get("alert_ts", "")

    print(f"  [slo] querying SLOs for service={service_name} project={monitoring_project}")

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
        print(f"  [slo] found {len(slos)} SLOs")

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
                print(f"  [slo] {slo_id}: burn_rate={burn_rate}")
            except Exception as e:
                print(f"  [slo] {slo_id}: burn_rate query failed: {e}")

            slo_summary.append({
                "slo_id":    slo_id,
                "type":      slo_type,
                "goal":      f"{goal*100:.1f}%",
                "burn_rate": burn_rate,
                "budget_exhausted_in_hours": round(720 / burn_rate, 1) if burn_rate and burn_rate > 0 else "unknown — query live data in console",
            })

    except Exception as e:
        print(f"  [slo] ERROR: {e}")
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


def route_alert_type(state: IncidentState) -> str:
    atype = state.get("alert_type", "log-based")
    if atype == "slo-based":
        return "fetch_slo"
    if atype == "metric-based":
        return "fetch_metrics"
    return "fetch_errors"


def route_release(state: IncidentState) -> str:
    return "fetch_release_logs" if state.get("release_found") else "build_report"


# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(IncidentState)

    g.add_node("parse_alert",        parse_alert)
    g.add_node("fetch_errors",       fetch_errors)       # log-based alerts
    g.add_node("fetch_metrics",      fetch_metrics)      # metric-based alerts
    g.add_node("fetch_slo",          fetch_slo)          # SLO burn rate alerts
    g.add_node("check_release",      check_release)
    g.add_node("fetch_release_logs", fetch_release_logs)
    g.add_node("build_report",       build_report)
    g.add_node("notify_and_wait",    notify_and_wait)
    g.add_node("decide",             decide)

    g.set_entry_point("parse_alert")

    # Branch: log-based → fetch_errors | metric-based → fetch_metrics
    g.add_conditional_edges("parse_alert", route_alert_type)

    g.add_edge("fetch_errors",       "check_release")
    g.add_edge("fetch_metrics",      "check_release")
    g.add_edge("fetch_slo",          "check_release")
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
