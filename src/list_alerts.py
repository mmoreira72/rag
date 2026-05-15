"""Lista alert policies do GCP Cloud Monitoring com severidade, data e contagem."""
import json
import os
import subprocess
import urllib.request
import urllib.parse
from collections import defaultdict
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


# ── 1. Alert policies ─────────────────────────────────────────────────────────
policies_data = _get(
    f"https://monitoring.googleapis.com/v3/projects/{project}/alertPolicies"
)
policies = policies_data.get("alertPolicies", [])

# ── 2. Recent incidents (last 7 days) ────────────────────────────────────────
# incidents API — groups by policyName to get last fired + open count
incidents_data = _get(
    f"https://monitoring.googleapis.com/v3/projects/{project}/incidents"
    f"?filter=state%3D%22open%22&pageSize=100"
)
incidents_all = incidents_data.get("incidents", [])

# Also fetch closed incidents for count
incidents_closed = _get(
    f"https://monitoring.googleapis.com/v3/projects/{project}/incidents"
    f"?pageSize=200"
).get("incidents", [])

# Build per-policy stats from combined list
policy_stats: dict[str, dict] = defaultdict(lambda: {
    "open": 0, "total": 0, "last_fired": ""
})
for inc in incidents_closed:
    pname = inc.get("policyName", "") or inc.get("policy", {}).get("name", "")
    # policyName is like "projects/PROJECT/alertPolicies/POLICY_ID"
    key = pname.split("/")[-1] if pname else ""
    if not key:
        continue
    policy_stats[key]["total"] += 1
    start = inc.get("startTime", "")
    if start > policy_stats[key]["last_fired"]:
        policy_stats[key]["last_fired"] = start
    if inc.get("state") == "OPEN":
        policy_stats[key]["open"] += 1

# ── 3. Print ──────────────────────────────────────────────────────────────────
SEV_SYMBOL = {"CRITICAL": "🔴", "ERROR": "🟠", "WARNING": "🟡", "": "⚪"}

print(f"\nAlert Policies — {project}\n")
print(f"{'#':<4} {'SEV':<4} {'LAST FIRED':<17} {'O':>2} {'TOT':>4}  {'NOME'}")
print("─" * 95)

for i, p in enumerate(policies, 1):
    name    = p.get("displayName", "?")
    enabled = p.get("enabled", False)
    sev     = p.get("severity", "")
    sym     = SEV_SYMBOL.get(sev, "⚪")

    # policy numeric ID from resource name "projects/.../alertPolicies/ID"
    policy_id = p.get("name", "").split("/")[-1]
    stats     = policy_stats.get(policy_id, {})
    last      = _fmt_ts(stats.get("last_fired", ""))
    open_c    = stats.get("open", 0)
    total_c   = stats.get("total", 0)

    flag = "" if enabled else " [disabled]"
    print(f"{i:<4} {sym:<4} {last:<17} {open_c:>2} {total_c:>4}  {name}{flag}")

print(f"\n{len(policies)} policies | columns: SEV · LAST FIRED (UTC) · OPEN · TOTAL (7d)")
print("Para investigar: make local-fetch-alert N=<número>")
