"""Lista alert policies do GCP Cloud Monitoring."""
import json
import os
import subprocess
import urllib.request
from datetime import datetime, timezone

project = os.environ.get("MON_PROJECT", "hw-prod-svc-sre-mon-a1f1")
token   = subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode().strip()
headers = {"Authorization": f"Bearer {token}"}


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=headers)
    try:
        return json.loads(urllib.request.urlopen(req, timeout=15).read())
    except Exception:
        return {}


def _fmt_ts(ts: str) -> str:
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts[:16]


def _condition_summary(policy: dict) -> str:
    """Extrai um resumo legível da condição principal."""
    conds = policy.get("conditions", [])
    if not conds:
        return "—"
    cond = conds[0]
    # Log-based
    if "conditionMatchedLog" in cond:
        f = cond["conditionMatchedLog"].get("filter", "")
        # extrair httpRequest.status ou severity
        for part in f.split(" AND "):
            if "status" in part or "severity" in part or "httpRequest" in part:
                return part.strip()[:50]
        return f"log-based"
    # Metric threshold
    if "conditionThreshold" in cond:
        ct = cond["conditionThreshold"]
        threshold = ct.get("thresholdValue", "")
        comparison = {"COMPARISON_GT": ">", "COMPARISON_LT": "<",
                      "COMPARISON_GE": "≥", "COMPARISON_LE": "≤"}.get(
            ct.get("comparison", ""), "?")
        metric = ct.get("filter", "").split('"')
        mtype = next((p for p in metric if "." in p and "/" in p), "")
        mshort = mtype.split("/")[-1][:30] if mtype else ""
        return f"{mshort} {comparison} {threshold}"[:50]
    # SLO / PromQL
    for key in ("conditionMonitoringQueryLanguage", "conditionPrometheusQueryLanguage"):
        if key in cond:
            return cond[key].get("query", "")[:50]
    return "—"


SEV_SYMBOL = {"CRITICAL": "🔴", "ERROR": "🟠", "WARNING": "🟡"}

data = _get(f"https://monitoring.googleapis.com/v3/projects/{project}/alertPolicies")
policies = data.get("alertPolicies", [])

print(f"\nAlert Policies — {project}\n")
print(f"{'#':<4} {'S':<2} {'LAST UPDATED (UTC)':<18} {'E':<2}  {'NOME / CONDIÇÃO'}")
print("─" * 100)

for i, p in enumerate(policies, 1):
    name    = p.get("displayName", "?")
    enabled = "✓" if p.get("enabled") else "✗"
    sev     = p.get("severity", "")
    sym     = SEV_SYMBOL.get(sev, "·")
    updated = _fmt_ts(p.get("mutationRecord", {}).get("mutateTime", ""))
    cond    = _condition_summary(p)
    print(f"{i:<4} {sym:<2} {updated:<18} {enabled:<2}  {name}")
    print(f"{'':4} {'':2} {'':18} {'':2}  └─ {cond}")

print(f"\n{len(policies)} policies  |  S=severity · E=enabled · LAST UPDATED=última alteração à policy")
print("Para investigar: make local-fetch-alert N=<número>")
