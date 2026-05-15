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
    list_revisions,
    list_slos,
)


def _j(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


# ── Arg schemas ───────────────────────────────────────────────────────────────

class FetchLogsArgs(BaseModel):
    filter_str: str = Field(
        description=(
            "GCP Cloud Logging filter (gcloud logging read syntax). "
            "Always include resource.type and timestamp range. "
            'Example: resource.type="cloud_run_revision" AND '
            'resource.labels.service_name="payment-service-api" AND '
            'timestamp>="2026-05-15T10:00:00Z" AND timestamp<="2026-05-15T12:00:00Z"'
        )
    )
    project: str = Field(description="GCP project ID where logs live")
    limit: int = Field(default=150, description="Max entries to return")


class ListRevisionsArgs(BaseModel):
    service: str = Field(description="Cloud Run service name")
    project: str = Field(description="GCP project ID")
    region: str = Field(default="europe-west1", description="Cloud Run region")
    limit: int = Field(default=5)


class DescribeSubscriptionArgs(BaseModel):
    project: str = Field(description="GCP project ID")
    subscription: str = Field(description="Pub/Sub subscription short name (not full path)")


class CheckOptimizelyArgs(BaseModel):
    project: str = Field(
        default="hw-prod-svc-sre-mon-a1f1",
        description="Monitoring project (Optimizely webhook runs here)",
    )
    hours: int = Field(default=2, description="Hours back to search for flag changes")


class FetchTimeseriesArgs(BaseModel):
    project: str = Field(description="GCP project ID")
    metric: str = Field(description="Metric type, e.g. pubsub.googleapis.com/subscription/num_undelivered_messages")
    window_minutes: int = Field(default=90)


class ListSlosArgs(BaseModel):
    project: str = Field(description="GCP monitoring project ID")
    service: str = Field(description="Service name for SLO lookup")


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
]

TOOLS_BY_NAME: dict[str, StructuredTool] = {t.name: t for t in INVESTIGATION_TOOLS}
