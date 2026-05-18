"""LangChain tool wrappers for incident investigation scripts.

Each tool wraps one script from incident-investigation/scripts/ and
returns a JSON string so the LLM can read the result directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from skills_runtime import (
    check_optimizely,
    describe_subscription,
    fetch_logs,
    fetch_timeseries,
    fetch_k8s_events,
    fetch_istio_logs,
    fetch_container_logs,
    fetch_pod_metrics,
    fetch_dapr_logs,
    list_revisions,
    list_slos,
)


def _j(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


# ── Base schemas ──────────────────────────────────────────────────────────────

class _GCPArgs(BaseModel):
    project: str = Field(description="GCP project ID")


class _K8sArgs(BaseModel):
    """Base for all GKE workload tools — shares service, project, hours, namespace."""
    service:   str = Field(description="GKE workload name (pod_name prefix)")
    project:   str = Field(default="hw-prod-sre-mgmt-k8s-fc7f", description="GKE project ID")
    hours:     int = Field(default=6,                 description="Hours back to search")
    namespace: str = Field(default="social-platform",  description="Kubernetes namespace")


# ── Concrete schemas ───────────────────────────────────────────────────────────

class FetchLogsArgs(BaseModel):
    filter_str: str = Field(
        description=(
            "GCP Cloud Logging filter (gcloud logging read syntax). "
            "Always include a timestamp range. "
            "resource.type is optional — omit it for K8s workloads to avoid missing logs spread across multiple resource types (k8s_pod, k8s_cluster, k8s_container). "
            "Cloud Run example: "
            'resource.type="cloud_run_revision" AND resource.labels.service_name="payment-service-api" AND timestamp>="2026-05-15T10:00:00Z" AND timestamp<="2026-05-15T12:00:00Z" '
            "K8s example: "
            'labels.source_workload="og-event-service-event-sync-processor" AND timestamp>="2026-05-15T10:00:00Z" AND timestamp<="2026-05-15T12:00:00Z"'
        )
    )
    project: str = Field(description="GCP project ID where logs live")
    limit:   int = Field(default=150, description="Max entries to return")


class ListRevisionsArgs(BaseModel):
    service: str = Field(description="Cloud Run service name")
    project: str = Field(description="GCP project ID")
    region:  str = Field(default="europe-west1", description="Cloud Run region")
    limit:   int = Field(default=5)


class DescribeSubscriptionArgs(BaseModel):
    project:      str = Field(description="GCP project ID")
    subscription: str = Field(description="Pub/Sub subscription short name (not full path)")


class CheckOptimizelyArgs(BaseModel):
    project: str = Field(default="hw-prod-svc-sre-mon-a1f1", description="Monitoring project (Optimizely webhook runs here)")
    hours:   int = Field(default=2, description="Hours back to search for flag changes")


class FetchTimeseriesArgs(_GCPArgs):
    metric:         str = Field(description="Metric type, e.g. pubsub.googleapis.com/subscription/num_undelivered_messages")
    window_minutes: int = Field(default=90)


class ListSlosArgs(_GCPArgs):
    service: str = Field(description="Service name for SLO lookup")


# K8s schemas — extend _K8sArgs, add only fields that differ
class FetchK8sEventsArgs(_K8sArgs):  pass
class FetchIstioLogsArgs(_K8sArgs):  pass
class FetchPodMetricsArgs(_K8sArgs): pass
class FetchDaprLogsArgs(_K8sArgs):   pass

class FetchContainerLogsArgs(_K8sArgs):
    severity_filter: str = Field(default="ERROR", description="Minimum severity: ERROR, WARNING, or DEFAULT")


# ── Tool functions ────────────────────────────────────────────────────────────

def _fetch_logs(filter_str: str, project: str, limit: int = 150) -> str:
    results = fetch_logs(filter_str, project, limit)
    if not results:
        return "No log entries found matching the filter."
    return _j(results[:limit])


def _list_revisions(service: str, project: str,
                    region: str = "europe-west1", limit: int = 5) -> str:
    results = list_revisions(service, project, region, limit)
    if not results:
        return "No revisions found (service may not exist in this project/region)."
    # Return only essential fields to save context
    slim = [
        {
            "name": r.get("metadata", {}).get("name", ""),
            "creationTimestamp": r.get("metadata", {}).get("creationTimestamp", ""),
            "image": r.get("spec", {}).get("containers", [{}])[0].get("image", ""),
        }
        for r in results
    ]
    return _j(slim)


def _describe_subscription(project: str, subscription: str) -> str:
    result = describe_subscription(project, subscription)
    if result is None:
        return "NOT_FOUND — subscription does not exist or was deleted. This is a likely root cause."
    return _j(result)


def _check_optimizely(project: str = "hw-prod-svc-sre-mon-a1f1", hours: int = 2) -> str:
    results = check_optimizely(project, hours)
    if not results:
        return "No Optimizely feature flag changes found in the specified window."
    return _j(results)


def _fetch_timeseries(project: str, metric: str, window_minutes: int = 90) -> str:
    results = fetch_timeseries(project, metric, window_minutes)
    if not results:
        return "No time series data found for this metric."
    return _j(results)


def _list_slos(project: str, service: str) -> str:
    results = list_slos(project, service)
    if not results:
        return "No SLOs found for this service."
    return _j(results)


def _fetch_k8s_events(service: str, project: str = "hw-prod-sre-mgmt-k8s-fc7f",
                      hours: int = 6, namespace: str = "social-platform") -> str:
    results = fetch_k8s_events(service, project, hours, namespace)
    if not results:
        return "No Kubernetes events found for this workload."
    root_causes = [e for e in results if e.get("root_cause")]
    summary = {"total": len(results), "root_cause_events": root_causes, "all_events": results}
    return _j(summary)


def _fetch_istio_logs(service: str, project: str = "hw-prod-sre-mgmt-k8s-fc7f",
                      hours: int = 6) -> str:
    results = fetch_istio_logs(service, project, hours)
    if not results:
        return "No Istio access logs found for this workload."
    failures = [e for e in results if e.get("is_failure")]
    summary = {"total": len(results), "failures": len(failures), "failure_entries": failures[:10]}
    return _j(summary)


def _fetch_container_logs(service: str, project: str = "hw-prod-sre-mgmt-k8s-fc7f",
                          hours: int = 6, namespace: str = "social-platform",
                          severity_filter: str = "ERROR") -> str:
    results = fetch_container_logs(service, project, hours, namespace, severity_filter)
    if not results:
        return "No container logs found — try severity_filter=WARNING or DEFAULT."
    errors = [e for e in results if e.get("is_error")]
    return _j({"total": len(results), "errors": len(errors), "entries": results[:30]})


def _fetch_pod_metrics(service: str, project: str = "hw-prod-sre-mgmt-k8s-fc7f",
                       hours: int = 6, namespace: str = "social-platform") -> str:
    result = fetch_pod_metrics(service, project, project, hours, namespace)
    if "error" in result:
        return f"Metrics unavailable: {result['error']}"
    mem = result.get("memory_mb", [])
    cpu = result.get("cpu_percent", [])
    if not mem and not cpu:
        return "No metrics found — pod may not exist or metrics not yet available."
    # Summarise trend: last value, max value
    def _summary(pts: list[dict], unit: str) -> dict:
        if not pts:
            return {}
        vals = [p["value"] for p in pts]
        return {"last": vals[-1], "max": max(vals), "unit": unit, "points": len(pts)}
    return _j({"memory": _summary(mem, "MB"), "cpu": _summary(cpu, "%"), "raw": result})


def _fetch_dapr_logs(service: str, project: str = "hw-prod-sre-mgmt-k8s-fc7f",
                     hours: int = 6, namespace: str = "social-platform") -> str:
    results = fetch_dapr_logs(service, project, hours, namespace)
    if not results:
        return "No Dapr sidecar errors found — Dapr may be healthy or not present."
    return _j({"dapr_errors": len(results), "entries": results[:20]})


# ── Public tool list ──────────────────────────────────────────────────────────

INVESTIGATION_TOOLS = [
    StructuredTool.from_function(
        func=_fetch_logs,
        name="fetch_logs",
        description=(
            "Fetch GCP Cloud Logging entries matching a filter. "
            "Use for application logs, request logs, error logs, and Pub/Sub logs. "
            "Always filter by resource.type and timestamp range."
        ),
        args_schema=FetchLogsArgs,
    ),
    StructuredTool.from_function(
        func=_list_revisions,
        name="list_revisions",
        description=(
            "List recent Cloud Run revisions for a service. "
            "Use to detect recent code deployments that may correlate with the incident."
        ),
        args_schema=ListRevisionsArgs,
    ),
    StructuredTool.from_function(
        func=_describe_subscription,
        name="describe_subscription",
        description=(
            "Describe a Pub/Sub subscription. "
            "Returns NOT_FOUND if the subscription was deleted — likely root cause for backlog alerts."
        ),
        args_schema=DescribeSubscriptionArgs,
    ),
    StructuredTool.from_function(
        func=_check_optimizely,
        name="check_optimizely",
        description=(
            "Check for recent Optimizely feature flag changes (datafile_updated events). "
            "Use to detect feature flag rollouts that may have caused the incident."
        ),
        args_schema=CheckOptimizelyArgs,
    ),
    StructuredTool.from_function(
        func=_fetch_timeseries,
        name="fetch_timeseries",
        description="Fetch GCP Monitoring time series data for a specific metric.",
        args_schema=FetchTimeseriesArgs,
    ),
    StructuredTool.from_function(
        func=_list_slos,
        name="list_slos",
        description=(
            "List Service Level Objectives (SLOs) for a service with current burn rates. "
            "Use for SLO burn rate alerts."
        ),
        args_schema=ListSlosArgs,
    ),
    StructuredTool.from_function(
        func=_fetch_k8s_events,
        name="fetch_k8s_events",
        description=(
            "Fetch Kubernetes lifecycle events for a GKE workload — OOMKill, CrashLoopBackOff, "
            "Evicted, BackOff, NEG syncer shutdown. Use for GKE/k8s_container alerts. "
            "Returns root_cause=true on critical events (OOMKill, CrashLoop)."
        ),
        args_schema=FetchK8sEventsArgs,
    ),
    StructuredTool.from_function(
        func=_fetch_istio_logs,
        name="fetch_istio_logs",
        description=(
            "Fetch Istio sidecar access logs for a GKE workload — PassthroughCluster failures, "
            "UF/URX response flags indicating upstream connection failures. "
            "Use for GKE workloads with Istio service mesh."
        ),
        args_schema=FetchIstioLogsArgs,
    ),
    StructuredTool.from_function(
        func=_fetch_container_logs,
        name="fetch_container_logs",
        description=(
            "Fetch application container logs from a GKE pod — what the service itself writes. "
            "Use to find application errors, stack traces, and messages before a crash. "
            "Default severity=ERROR; use WARNING or DEFAULT for more context."
        ),
        args_schema=FetchContainerLogsArgs,
    ),
    StructuredTool.from_function(
        func=_fetch_pod_metrics,
        name="fetch_pod_metrics",
        description=(
            "Fetch CPU and memory usage metrics for a GKE pod via Cloud Monitoring. "
            "Use to detect memory leak (growing memory_mb) or CPU saturation before OOMKill."
        ),
        args_schema=FetchPodMetricsArgs,
    ),
    StructuredTool.from_function(
        func=_fetch_dapr_logs,
        name="fetch_dapr_logs",
        description=(
            "Fetch Dapr sidecar (daprd) error logs for a GKE workload — state store failures, "
            "pub/sub errors, gRPC connection issues. Use when the workload uses Dapr."
        ),
        args_schema=FetchDaprLogsArgs,
    ),
]

TOOLS_BY_NAME: dict[str, StructuredTool] = {t.name: t for t in INVESTIGATION_TOOLS}
