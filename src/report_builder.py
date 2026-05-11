"""Deterministic markdown report builder (Sprint 2)."""
from __future__ import annotations

import re
from datetime import datetime, timezone

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)

from evidence import (
    ErrorAnalysis,
    MetricsAnalysis,
    SLOAnalysis,
    ReleaseInfo,
)

_ACTIONS = ("rollback", "escalate", "monitor", "close")

_ACTION_PATTERN = re.compile(
    r'^ACTION:\s*(rollback|escalate|monitor|close)\s*$',
    re.MULTILINE | re.IGNORECASE,
)


def _extract_action(text: str) -> str:
    """Extract recommended action from the LLM response.

    Priority:
    1. Explicit ``ACTION: <word>`` line (canonical path — model was instructed to emit this).
    2. Fallback: last 200 chars scanned for action word (conclusion usually at end).
    3. Default: ``escalate`` — safest when output is ambiguous.
    """
    matches = _ACTION_PATTERN.findall(text)
    if matches:
        return matches[-1].lower()   # last match wins if model emitted multiple

    # Fallback: scan only the LAST 200 chars to avoid negations like "warrant a rollback"
    tail = text[-200:].lower()
    found = []
    for action in _ACTIONS:
        if action in tail:
            found.append((tail.rindex(action), action))
    if found:
        found.sort(reverse=True)
        return found[0][1]

    return "escalate"


def _build_impact_prompt(state: dict) -> dict:
    error_analysis   = state.get("error_logs")
    metrics_analysis = state.get("metrics_data")

    facts = []
    if isinstance(error_analysis, ErrorAnalysis):
        root_causes = [p for p in error_analysis.patterns if p.root_cause]
        if root_causes:
            facts.append(f"Root-cause errors: {'; '.join(p.message[:80] for p in root_causes[:3])}")
        facts.append(f"Total log entries analysed: {error_analysis.raw_log_count}")

    if isinstance(metrics_analysis, MetricsAnalysis):
        obs = [o for o in metrics_analysis.observations if o.root_cause]
        if obs:
            facts.append(f"Metric root cause: {obs[0].likely_cause[:80]}")
        if metrics_analysis.subscription_not_found:
            facts.append("Subscription NOT FOUND — likely root cause.")

    if isinstance(metrics_analysis, SLOAnalysis):
        for f in metrics_analysis.findings[:2]:
            if f.burn_rate:
                facts.append(f"SLO {f.slo_id}: burn_rate={f.burn_rate:.1f}x, "
                              f"budget exhausts in {f.budget_exhausted_in_hours}h")

    release = state.get("release")
    if isinstance(release, ReleaseInfo) and release.found:
        facts.append(f"Release detected: {release.release_id} ({release.release_type})")

    verdict = error_analysis.verdict if isinstance(error_analysis, ErrorAnalysis) else ""
    raw_count = error_analysis.raw_log_count if isinstance(error_analysis, ErrorAnalysis) else 0
    # evidence_missing ONLY when LLM crashed (parse_failed).
    # patterns=[] with parse_failed=False = analysis completed, no errors found — not a failure.
    evidence_missing = (
        isinstance(error_analysis, ErrorAnalysis) and error_analysis.parse_failed
    )

    user = (
        f"Service: {state.get('service')} | Project: {state.get('project')}\n"
        f"Alert: {state.get('condition_name')}\n"
        f"Evidence missing: {evidence_missing} | Log entries fetched: {raw_count}\n"
        f"Facts:\n" + "\n".join(f"- {f}" for f in facts)
    ) if facts else (
        f"Service: {state.get('service')} | Evidence missing: {evidence_missing} "
        f"| Log entries fetched: {raw_count}"
    )

    missing_instruction = (
        f"\n\nIMPORTANT: Evidence indicates that error analysis failed or no error patterns "
        f"were identified despite {raw_count} log entries being available. "
        f"You MUST write exactly this paragraph:\n"
        f'"Automated investigation incomplete. {raw_count} log entries were fetched but '
        f'could not be analyzed by the model. Manual review required before drawing '
        f'conclusions about user impact."'
    ) if evidence_missing else ""

    return {
        "system": (
            "You are an SRE writing an incident impact summary. "
            "Write 2-3 concise sentences describing user/system impact based ONLY on the facts provided. "
            "Do NOT invent ticket numbers, severity levels, post-mortem dates, or status fields. "
            "No bullet points. Only use information from the facts listed."
            + missing_instruction
        ),
        "user": user,
    }


def _build_action_prompt(state: dict) -> dict:
    service      = state.get("service", "unknown")
    alert_type   = state.get("alert_type", "")
    error_analysis   = state.get("error_logs")
    metrics_analysis = state.get("metrics_data")
    release      = state.get("release")

    root_causes = []
    if isinstance(error_analysis, ErrorAnalysis):
        root_causes += [p.message[:80] for p in error_analysis.patterns if p.root_cause]
    if isinstance(metrics_analysis, MetricsAnalysis):
        root_causes += [o.likely_cause[:80] for o in metrics_analysis.observations if o.root_cause]
    if isinstance(metrics_analysis, SLOAnalysis):
        root_causes += [f.likely_cause[:80] for f in metrics_analysis.findings if f.root_cause]

    release_note = ""
    if isinstance(release, ReleaseInfo) and release.found:
        release_note = (
            f"\nA {release.release_type} change was deployed at {release.release_ts}. "
            "Rollback may be appropriate if it caused the incident."
        )

    evidence_missing = not root_causes and (
        isinstance(error_analysis, ErrorAnalysis) and
        (error_analysis.parse_failed or
         (error_analysis.raw_log_count > 0 and not error_analysis.patterns))
    )

    user = (
        f"Service: {service} | Alert type: {alert_type}{release_note}\n"
        f"Root causes identified: {'; '.join(root_causes) or 'NONE — no concrete errors found'}\n\n"
        "Recommend ONE action: rollback / escalate / monitor / close.\n"
        "RULES:\n"
        "- If root causes list is NONE or empty: do NOT recommend rollback. Use escalate.\n"
        "- If error analysis failed or no error patterns were identified despite logs being available, "
        "the appropriate action is ESCALATE — a human must investigate manually.\n"
        "- rollback requires concrete error evidence (non-empty root causes) in the alert window.\n"
        "- Do NOT start your response with the action word — start with your reasoning.\n\n"
        "You MUST end your response with exactly one line in this format, "
        "on its own line, with no trailing characters:\n\n"
        "ACTION: <rollback|escalate|monitor|close>\n\n"
        "Do not add anything after that line."
    )
    return {
        "system": (
            "You are an SRE on call making an incident decision. "
            "Be decisive and concise. "
            "When evidence is missing or analysis failed, bias toward ESCALATE."
        ),
        "user": user,
    }


def _parse_failed_any(state: dict) -> bool:
    for field in ("error_logs", "metrics_data", "release_logs"):
        obj = state.get(field)
        if obj is not None and getattr(obj, "parse_failed", False):
            return True
    return False


def build_markdown_report(
    incident_id: str,
    service: str,
    project: str,
    alert_ts: datetime | str,
    alert_type: str,
    alert_url: str,
    log_url: str,
    error_analysis: ErrorAnalysis | None,
    metrics_analysis: MetricsAnalysis | SLOAnalysis | None,
    release: ReleaseInfo | None,
    release_log_url: str,
    release_errors: ErrorAnalysis | None,
    impact_paragraph: str,
    action_reasoning: str,
    recommended_action: str,
    http_status_url: str = "",
    analysis_log_url: str = "",
) -> str:
    sections: list[str] = []

    # Build warning block — prepended right after the title
    warnings: list[str] = []

    _log_link = f"[Logs analyzed by LLM]({analysis_log_url})" if analysis_log_url else f"[Inspect logs]({log_url})"

    if error_analysis is not None:
        if error_analysis.parse_failed:
            warnings.append(
                f"⚠️ **Evidence parsing failed** — the LLM produced malformed output "
                f"for log analysis. Manual review required. {_log_link}"
            )
        elif error_analysis.verdict:
            # LLM made an explicit conclusion — show it prominently
            warnings.append(
                f"ℹ️ **LLM verdict:** {error_analysis.verdict} {_log_link}"
            )
        elif error_analysis.raw_log_count > 0 and not error_analysis.patterns:
            warnings.append(
                f"ℹ️ **{error_analysis.raw_log_count} log entries analyzed — no system-level "
                f"error patterns identified.** Failures may be at application level "
                f"(e.g. payment declines, validation errors) rather than infrastructure. "
                f"{_log_link}"
            )

    if metrics_analysis is not None and getattr(metrics_analysis, "parse_failed", False):
        warnings.append("⚠️ **Metrics evidence parsing failed.** Manual review required.")

    if release_errors is not None and release_errors.parse_failed:
        warnings.append("⚠️ **Release log evidence parsing failed.** Manual review required.")

    # 1. Header + warning block (immediately after title)
    warning_block = ("\n".join(warnings) + "\n\n---\n\n") if warnings else ""
    sections.append(
        f"# Incident {incident_id}\n\n"
        f"**Service:** `{service}`  |  **Project:** `{project}`  |  "
        f"**Alert time:** {alert_ts}\n\n"
        + warning_block
    )

    # 2. Impact
    sections.append(f"## Impact\n\n{impact_paragraph}")

    # 3. Timeline (deterministic)
    timeline_lines = [f"- **{alert_ts}** — Alert fired"]
    if isinstance(release, ReleaseInfo) and release.found and release.release_ts:
        timeline_lines.insert(0, f"- **{release.release_ts}** — {release.release_type} change deployed: `{release.release_id}`")
    if error_analysis and error_analysis.patterns:
        for p in sorted(error_analysis.patterns, key=lambda x: x.first_seen or _EPOCH)[:3]:
            ts = p.first_seen.isoformat() if p.first_seen else "?"
            label = " *(root cause)*" if p.root_cause else ""
            timeline_lines.append(f"- **{ts}** — Error: {p.message[:60]}{label}")
    sections.append(f"## Timeline\n\n" + "\n".join(sorted(set(timeline_lines))))

    # 4. Evidence
    evidence_parts: list[str] = []

    if error_analysis and not error_analysis.parse_failed and error_analysis.patterns:
        rows = [f"| {p.count} | `{p.message[:70]}` | {'✓ root cause' if p.root_cause else ''} |"
                for p in error_analysis.patterns[:10]]
        evidence_parts.append(
            "### Error Patterns\n\n"
            "| Count | Message | Root Cause |\n|---|---|---|\n" + "\n".join(rows)
        )
    elif error_analysis and error_analysis.parse_failed:
        evidence_parts.append("### Error Patterns\n\n*Evidence parsing failed — see log link.*")

    if isinstance(metrics_analysis, MetricsAnalysis):
        if not metrics_analysis.parse_failed and metrics_analysis.observations:
            rows = [f"| `{o.metric[:50]}` | {o.trend} | {o.likely_cause[:60]} | {'✓' if o.root_cause else ''} |"
                    for o in metrics_analysis.observations]
            evidence_parts.append(
                "### Metric Observations\n\n"
                f"Condition: *{metrics_analysis.condition_direction}*\n\n"
                "| Metric | Trend | Likely Cause | Root Cause |\n|---|---|---|---|\n"
                + "\n".join(rows)
            )
            if metrics_analysis.subscription_not_found:
                evidence_parts.append("> ⚠️ **Subscription NOT FOUND** — may be deleted/misconfigured.")
        elif metrics_analysis.parse_failed:
            evidence_parts.append("### Metric Observations\n\n*Evidence parsing failed — see log link.*")

    if isinstance(metrics_analysis, SLOAnalysis):
        if not metrics_analysis.parse_failed and metrics_analysis.findings:
            rows = [
                f"| `{f.slo_id}` | {f.slo_type} | {f.goal} | "
                f"{f.burn_rate:.1f}x" if f.burn_rate else f"| `{f.slo_id}` | {f.slo_type} | {f.goal} | N/A"
                + f" | {f.likely_cause[:60]} |"
                for f in metrics_analysis.findings
            ]
            evidence_parts.append(
                "### SLO Findings\n\n"
                "| SLO | Type | Goal | Burn Rate | Likely Cause |\n|---|---|---|---|---|\n"
                + "\n".join(rows)
            )
        elif metrics_analysis.parse_failed:
            evidence_parts.append("### SLO Findings\n\n*Evidence parsing failed — see log link.*")

    if evidence_parts:
        sections.append("## Evidence\n\n" + "\n\n".join(evidence_parts))

    # 5. Release Correlation
    if isinstance(release, ReleaseInfo) and release.found:
        release_label = (
            "**Feature flag change** (Optimizely) — NOT a code deploy."
            if release.release_type == "optimizely"
            else f"**Code deploy** (`{release.release_type}`)"
        )
        corr_parts = [f"{release_label}\n- Release ID: `{release.release_id}`\n- Time: {release.release_ts}"]

        if release_errors and not release_errors.parse_failed and release_errors.patterns:
            post_errors = [p.message[:60] for p in release_errors.patterns[:5]]
            corr_parts.append("Errors after deploy:\n" + "\n".join(f"- `{e}`" for e in post_errors))
        elif release_errors and release_errors.parse_failed:
            corr_parts.append("*Post-deploy error analysis failed — see release log link.*")

        sections.append("## Release Correlation\n\n" + "\n\n".join(corr_parts))

    # 6. Recommended Action
    sections.append(
        f"## Recommended Action: {recommended_action.upper()}\n\n{action_reasoning}"
    )

    # 7. Links
    links = [
        f"- [Alert]({alert_url})" if alert_url else "",
        f"- [Logs Analyzed by LLM]({analysis_log_url})" if analysis_log_url else "",
        f"- [Error Logs — HTTP {http_status_url.split('status=')[1].split('&')[0] if 'status=' in http_status_url else '5xx'}]({http_status_url})" if http_status_url else "",
        f"- [All Error Logs]({log_url})" if log_url else "",
        f"- [Release Logs]({release_log_url})" if release_log_url and isinstance(release, ReleaseInfo) and release.found else "",
    ]
    sections.append("## Links\n\n" + "\n".join(l for l in links if l))

    return "\n\n".join(sections)
