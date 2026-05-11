"""Tests for report_builder.py (Sprint 2)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evidence import (
    ErrorAnalysis, ErrorPattern,
    MetricsAnalysis, MetricObservation,
    SLOAnalysis, SLOFinding,
    ReleaseInfo,
)
from report_builder import build_markdown_report, _extract_action


def _full_report(**overrides) -> str:
    kwargs = dict(
        incident_id="INC-001",
        service="payments-api",
        project="hw-prod",
        alert_ts=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
        alert_type="log-based",
        alert_url="https://console.cloud.google.com/monitoring/alerting/incidents/INC-001",
        log_url="https://console.cloud.google.com/logs",
        error_analysis=ErrorAnalysis(
            patterns=[
                ErrorPattern(count=12, message="OOM crash", root_cause=True,
                             first_seen=datetime(2025, 1, 15, 9, 55, tzinfo=timezone.utc)),
                ErrorPattern(count=5, message="LIVENESS probe failed", root_cause=False),
            ],
            raw_log_count=150,
        ),
        metrics_analysis=MetricsAnalysis(
            observations=[MetricObservation(metric="request_count", trend="dropping", root_cause=False)]
        ),
        release=ReleaseInfo(
            found=True,
            release_id="payments-api-rev-42",
            release_ts=datetime(2025, 1, 15, 9, 50, tzinfo=timezone.utc),
            release_type="cloud_run",
        ),
        release_log_url="https://console.cloud.google.com/logs/release",
        release_errors=ErrorAnalysis(
            patterns=[ErrorPattern(count=3, message="startup timeout")]
        ),
        impact_paragraph="The payments API experienced OOM crashes affecting checkout.",
        action_reasoning="rollback the recent deploy to restore stability.",
        recommended_action="rollback",
    )
    kwargs.update(overrides)
    return build_markdown_report(**kwargs)


def test_markdown_includes_all_sections_when_full_evidence():
    md = _full_report()
    for section in ("## Impact", "## Timeline", "## Evidence", "## Release Correlation",
                    "## Recommended Action", "## Links"):
        assert section in md, f"Missing section: {section}"


def test_markdown_omits_release_section_when_not_found():
    md = _full_report(
        release=ReleaseInfo(found=False),
        release_errors=None,
    )
    assert "## Release Correlation" not in md


def test_markdown_warning_on_parse_failed():
    md = _full_report(
        error_analysis=ErrorAnalysis(parse_failed=True, raw_log_count=10),
    )
    assert "⚠️" in md
    assert "evidence could not be parsed" in md.lower() or "Some evidence" in md


def test_extract_action_defaults_to_escalate_when_ambiguous():
    assert _extract_action("I'm not sure what to do here.") == "escalate"
    assert _extract_action("") == "escalate"


def test_extract_action_parses_known_actions():
    assert _extract_action("rollback the deploy immediately") == "rollback"
    assert _extract_action("monitor for 30 minutes then close") == "monitor"
    assert _extract_action("close the incident — no user impact") == "close"
    assert _extract_action("escalate to the on-call engineer") == "escalate"
