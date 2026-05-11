"""Tests for evidence.py Pydantic models (Sprint 2)."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evidence import (
    ErrorAnalysis, ErrorPattern,
    MetricsAnalysis, MetricObservation,
    SLOAnalysis, SLOFinding,
    ReleaseInfo, IncidentReport,
)
from incident_graph import IncidentState


def test_error_analysis_round_trip():
    analysis = ErrorAnalysis(
        patterns=[ErrorPattern(count=5, message="OOM crash", root_cause=True)],
        raw_log_count=42,
    )
    data = analysis.model_dump()
    restored = ErrorAnalysis.model_validate(data)
    assert restored.patterns[0].count == 5
    assert restored.patterns[0].root_cause is True
    assert restored.raw_log_count == 42
    assert restored.parse_failed is False


def test_error_analysis_partial_fields():
    analysis = ErrorAnalysis(
        patterns=[ErrorPattern(count=1, message="timeout")]
    )
    assert analysis.raw_log_count == 0
    assert analysis.parse_failed is False


def test_release_info_parses_iso_timestamps():
    info = ReleaseInfo(
        found=True,
        release_id="rev-001",
        release_ts="2025-01-15T10:30:00Z",
        release_type="cloud_run",
    )
    assert isinstance(info.release_ts, datetime)
    assert info.release_ts.year == 2025


def test_metrics_or_slo_union_in_state():
    """IncidentState can hold either MetricsAnalysis or SLOAnalysis in metrics_data."""
    metrics = MetricsAnalysis(
        observations=[MetricObservation(metric="pubsub/backlog", trend="rising")]
    )
    slo = SLOAnalysis(
        findings=[SLOFinding(slo_id="availability-slo", burn_rate=12.5)]
    )
    # Both should be valid TypedDict values for metrics_data
    state_m: dict = {"metrics_data": metrics}
    state_s: dict = {"metrics_data": slo}
    assert isinstance(state_m["metrics_data"], MetricsAnalysis)
    assert isinstance(state_s["metrics_data"], SLOAnalysis)


def test_incident_report_fields():
    report = IncidentReport(
        markdown="# Test\n\nSome content",
        recommended_action="monitor",
        sections_used=["error_logs"],
    )
    assert report.recommended_action == "monitor"
    assert "error_logs" in report.sections_used
    assert report.evidence_summary == {}
