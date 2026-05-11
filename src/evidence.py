"""Typed evidence objects produced by each investigation node (Sprint 2)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ErrorPattern(BaseModel):
    """One identified error pattern from log analysis."""
    count: int = Field(ge=0)
    message: str
    first_seen: datetime | None = None
    root_cause: bool = False


class ErrorAnalysis(BaseModel):
    """Output of fetch_errors and fetch_release_logs."""
    patterns: list[ErrorPattern] = Field(default_factory=list)
    raw_log_count: int = 0
    parse_failed: bool = False  # True when LLM JSON was malformed


class MetricObservation(BaseModel):
    """One metric trend observation."""
    metric: str
    trend: str
    likely_cause: str = ""
    root_cause: bool = False


class MetricsAnalysis(BaseModel):
    """Output of fetch_metrics."""
    observations: list[MetricObservation] = Field(default_factory=list)
    condition_direction: str = ""
    subscription_not_found: bool = False
    parse_failed: bool = False


class SLOFinding(BaseModel):
    """One SLO burn rate finding."""
    slo_id: str
    slo_type: Literal["availability", "latency", "unknown"] = "unknown"
    goal: str = ""
    burn_rate: float | None = None
    budget_exhausted_in_hours: float | str | None = None
    likely_cause: str = ""
    root_cause: bool = False


class SLOAnalysis(BaseModel):
    """Output of fetch_slo."""
    findings: list[SLOFinding] = Field(default_factory=list)
    console_url: str = ""
    parse_failed: bool = False


class ReleaseInfo(BaseModel):
    """Output of check_release — pure deterministic, no LLM."""
    found: bool = False
    release_id: str = ""
    release_ts: datetime | None = None
    release_type: Literal["cloud_run", "optimizely", ""] = ""


class IncidentReport(BaseModel):
    """Final output of build_report."""
    markdown: str
    recommended_action: Literal["rollback", "escalate", "monitor", "close"]
    sections_used: list[str] = Field(default_factory=list)
    evidence_summary: dict = Field(default_factory=dict)
