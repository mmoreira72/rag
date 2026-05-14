"""Stable import surface for incident-investigation skill scripts.

Adds the cloudengineerskills scripts directory to sys.path so callers can
import investigation functions without knowing the repo layout.

Usage:
    from skills_runtime import list_revisions, fetch_logs, describe_subscription
"""
import sys
from pathlib import Path

_SCRIPTS = (
    Path("/Users/marcelo.moreira/Library/CloudStorage/OneDrive-Hostelworld"
         "/Documents/personal/cloudengineerskills/investigation-skills"
         "/incident-investigation/scripts")
)
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from list_revisions import list_revisions
from fetch_logs import fetch_logs
from describe_subscription import describe_subscription
from check_optimizely import check_optimizely
from fetch_timeseries import fetch_timeseries
from list_slos import list_slos

__all__ = [
    "list_revisions",
    "fetch_logs",
    "describe_subscription",
    "check_optimizely",
    "fetch_timeseries",
    "list_slos",
]
