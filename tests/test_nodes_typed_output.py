"""Tests for typed node outputs (Sprint 2)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evidence import ErrorAnalysis, MetricsAnalysis, SLOAnalysis, ReleaseInfo


# ── fetch_errors ─────────────────────────────────────────────────────────────

def test_fetch_errors_returns_error_analysis():
    import incident_graph as ig

    fake_logs = [{"textPayload": "OOM crash", "severity": "ERROR",
                  "timestamp": "2025-01-15T10:00:00Z"}]
    llm_json = '{"patterns": [{"count": 1, "message": "OOM crash", "root_cause": true}]}'

    state = {
        "alert_ts": "2025-01-15T10:05:00Z",
        "project": "hw-prod",
        "service": "payments-api",
        "resource_type": "cloud_run_revision",
    }
    with patch.object(ig, "_run_gcloud_logging", return_value=fake_logs), \
         patch.object(ig, "call_llm", return_value=llm_json), \
         patch.object(ig, "get_skill", return_value=""):
        result = ig.fetch_errors(state)

    assert "error_logs" in result
    analysis = result["error_logs"]
    assert isinstance(analysis, ErrorAnalysis)
    assert len(analysis.patterns) == 1
    assert analysis.patterns[0].root_cause is True
    assert analysis.parse_failed is False


def test_fetch_errors_sets_parse_failed_on_bad_json():
    import incident_graph as ig

    state = {
        "alert_ts": "2025-01-15T10:05:00Z",
        "project": "hw-prod",
        "service": "payments-api",
        "resource_type": "cloud_run_revision",
    }
    with patch.object(ig, "_run_gcloud_logging", return_value=[]), \
         patch.object(ig, "call_llm", return_value="not json at all"), \
         patch.object(ig, "get_skill", return_value=""):
        result = ig.fetch_errors(state)

    assert isinstance(result["error_logs"], ErrorAnalysis)
    # parse_failed because "not json at all" → parse_llm_json returns {} → patterns=[]
    # empty patterns is valid, parse_failed stays False (no Pydantic error)
    assert result["error_logs"].parse_failed is False


def test_fetch_errors_sets_parse_failed_on_invalid_schema():
    import incident_graph as ig

    state = {
        "alert_ts": "2025-01-15T10:05:00Z",
        "project": "hw-prod",
        "service": "payments-api",
        "resource_type": "cloud_run_revision",
    }
    # Return patterns with count=-1 (violates ge=0 constraint)
    bad_json = '{"patterns": [{"count": -1, "message": "err"}]}'
    with patch.object(ig, "_run_gcloud_logging", return_value=[]), \
         patch.object(ig, "call_llm", return_value=bad_json), \
         patch.object(ig, "get_skill", return_value=""):
        result = ig.fetch_errors(state)

    assert isinstance(result["error_logs"], ErrorAnalysis)
    assert result["error_logs"].parse_failed is True


# ── check_release ─────────────────────────────────────────────────────────────

def test_check_release_returns_release_info_not_found():
    import incident_graph as ig
    import subprocess

    state = {
        "alert_ts": "2025-01-15T10:05:00Z",
        "project": "hw-prod",
        "service": "payments-api",
        "region": "europe-west1",
    }
    mock_result = MagicMock()
    mock_result.stdout = "[]"
    with patch("subprocess.run", return_value=mock_result), \
         patch.object(ig, "_run_gcloud_logging", return_value=[]), \
         patch.object(ig, "get_skill", return_value=""):
        result = ig.check_release(state)

    assert "release" in result
    assert isinstance(result["release"], ReleaseInfo)
    assert result["release"].found is False


def test_check_release_returns_release_info_found():
    import incident_graph as ig

    state = {
        "alert_ts": "2025-01-15T10:05:00Z",
        "project": "hw-prod",
        "service": "payments-api",
        "region": "europe-west1",
    }
    revisions = [{"metadata": {
        "name": "payments-api-rev-42",
        "creationTimestamp": "2025-01-15T10:00:00Z",
    }}]
    import json
    mock_result = MagicMock()
    mock_result.stdout = json.dumps(revisions)
    with patch("incident_graph.subprocess.run", return_value=mock_result), \
         patch.object(ig, "get_skill", return_value=""), \
         patch.object(ig, "_run_gcloud_logging", return_value=[]):
        result = ig.check_release(state)

    assert result["release"].found is True
    assert result["release"].release_type == "cloud_run"
    assert result["release"].release_id == "payments-api-rev-42"


# ── fetch_metrics ─────────────────────────────────────────────────────────────

def test_fetch_metrics_returns_metrics_analysis():
    import incident_graph as ig

    state = {
        "alert_ts": "2025-01-15T10:05:00Z",
        "project": "hw-prod",
        "service": "booking-subs",
        "resource_type": "pubsub_subscription",
        "metric_type": "pubsub.googleapis.com/subscription/num_undelivered_messages",
        "condition_type": "COMPARISON_GT",
        "condition_name": "backlog > 1000",
    }
    llm_json = '{"observations": [{"metric": "backlog", "trend": "rising", "likely_cause": "publisher stopped", "root_cause": true}]}'

    with patch.object(ig, "call_llm", return_value=llm_json), \
         patch.object(ig, "get_skill", return_value=""), \
         patch("subprocess.run", side_effect=FileNotFoundError), \
         patch("urllib.request.urlopen", side_effect=Exception("no network")):
        result = ig.fetch_metrics(state)

    assert isinstance(result["metrics_data"], MetricsAnalysis)
    assert len(result["metrics_data"].observations) == 1
    assert result["metrics_data"].observations[0].root_cause is True
