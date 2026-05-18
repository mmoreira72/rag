"""Stable import surface for incident-investigation skill scripts.

Adds the cloudengineerskills scripts directories to sys.path so callers can
import investigation functions without knowing the repo layout.

Usage:
    from skills_runtime import list_revisions, fetch_logs, fetch_k8s_events
"""
import sys
from pathlib import Path

_BASE = Path("/Users/marcelo.moreira/Library/CloudStorage/OneDrive-Hostelworld"
             "/Documents/personal/cloudengineerskills/investigation-skills")

for _scripts_dir in sorted(_BASE.glob("*/scripts")):
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))

from list_revisions import list_revisions
from fetch_logs import fetch_logs
from describe_subscription import describe_subscription
from check_optimizely import check_optimizely
from fetch_timeseries import fetch_timeseries
from list_slos import list_slos
from fetch_k8s_events import fetch_k8s_events, fetch_pod_restarts
from fetch_istio_logs import fetch_istio_logs
from fetch_container_logs import fetch_container_logs, fetch_container_logs_before_crash
from fetch_pod_metrics import fetch_pod_metrics
from fetch_dapr_logs import fetch_dapr_logs, fetch_dapr_metric_errors

__all__ = [
    "list_revisions",
    "fetch_logs",
    "describe_subscription",
    "check_optimizely",
    "fetch_timeseries",
    "list_slos",
    "fetch_k8s_events",
    "fetch_pod_restarts",
    "fetch_istio_logs",
    "fetch_container_logs",
    "fetch_container_logs_before_crash",
    "fetch_pod_metrics",
    "fetch_dapr_logs",
    "fetch_dapr_metric_errors",
]
