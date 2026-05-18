"""Converte uma alert policy GCP para o formato do incident_graph.py e roda a investigação."""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

n = int(os.environ.get("N", "1")) - 1
project = os.environ.get("MON_PROJECT", "hw-prod-svc-sre-mon-a1f1")

token = subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode().strip()
headers = {"Authorization": f"Bearer {token}"}

policies = []
page_token = ""
while True:
    url = f"https://monitoring.googleapis.com/v3/projects/{project}/alertPolicies?pageSize=100"
    if page_token:
        url += f"&pageToken={page_token}"
    req = urllib.request.Request(url, headers=headers)
    data = json.loads(urllib.request.urlopen(req).read())
    policies.extend(data.get("alertPolicies", []))
    page_token = data.get("nextPageToken", "")
    if not page_token:
        break

if n >= len(policies):
    print(f"Erro: só existem {len(policies)} policies. Use N entre 1 e {len(policies)}.")
    sys.exit(1)

policy = policies[n]
cond = policy.get("conditions", [{}])[0]

cond_threshold  = cond.get("conditionThreshold", {})
cond_log        = cond.get("conditionMatchedLog", {})
cond_promql     = cond.get("conditionPrometheusQueryLanguage", {})
cond_mql        = cond.get("conditionMonitoringQueryLanguage", {})

# metric_filter: preferência threshold/log; PromQL/MQL guardados separadamente
metric_filter   = cond_threshold.get("filter", "") or cond_log.get("filter", "")
promql_query    = cond_promql.get("query", "") or cond_mql.get("query", "")

resource_type = "cloud_run_revision"
service_name  = "unknown"
region        = "europe-west1"

# 0. Detecta resource_type a partir do prefixo do display name
_DISPLAY_RTYPE = {
    "[GKE]":      "k8s_container",
    "[Pubsub]":   "pubsub_subscription",
    "[CloudSQL]": "cloudsql_database",
    "[Cloudrun]": "cloud_run_revision",
    "[MIG]":      "gce_instance",
}
display_name = policy.get("displayName", "")
for prefix, rtype in _DISPLAY_RTYPE.items():
    if prefix.lower() in display_name.lower():
        resource_type = rtype
        break

# 1. Extrai resource_type e service_name do metric filter (threshold/log alerts)
_EXCLUDED = {"cloud_run_revision", "pubsub_subscription", "k8s_container",
             "gce_instance", "global", "europe-west1"}
for part2 in metric_filter.replace('"', ' ').split():
    if part2 in _EXCLUDED:
        resource_type = part2

for part in metric_filter.split('"'):
    part = part.strip()
    if part and re.match(r'^[\w][\w-]{2,}$', part) and part not in _EXCLUDED:
        service_name = part
        break

# 2. Para PromQL/MQL: procura service_name="..." ou resource.labels.service_name=...
if promql_query:
    m = re.search(r'service_name[=!~]+"([^"]+)"', promql_query)
    if m and service_name == "unknown":
        service_name = m.group(1)
    if "k8s_container" in promql_query or "kubernetes" in promql_query:
        resource_type = "k8s_container"

# 3. Fallback: extrai do display name usando o slug entre segundos colchetes
#    "[PromQL] - [og_event_service_event_sync_processor_critical_logs_high] - ..."
#    converte underscores→hífens e remove sufixos de alerta conhecidos
_ALERT_SUFFIXES = re.compile(
    r'[-_](critical[-_]logs[-_]high|error[-_]rate|response[-_]code|'
    r'slo[-_]burn[-_]rate|metrics[-_]absent|cpu[-_]utilisation|'
    r'restarting|health[-_]issues|could[-_]not[-_]resolve|'
    r'logs[-_]high|errors[-_]high|latency[-_]high|availability[-_]low)$',
    re.IGNORECASE,
)
if service_name == "unknown":
    display = policy.get("displayName", "")
    # Tenta slug entre colchetes: [PromQL] - [slug] - descrição
    m = re.search(r'\[PromQL\][^[]*\[([^\]]+)\]', display)
    if m:
        slug = m.group(1).replace("_", "-").lower()
        slug = _ALERT_SUFFIXES.sub("", slug)
        service_name = slug
    else:
        # Fallback: primeira palavra após ]
        m = re.search(r'\]\s*([\w][\w-]+)', display)
        if m:
            service_name = m.group(1).replace("_", "-")

# 4. Infere resource_type pelo service_name (depois de service_name estar definido)
if resource_type == "cloud_run_revision" and service_name != "unknown":
    if any(kw in service_name for kw in ("og-event", "og-", "k8s", "gke")):
        resource_type = "k8s_container"

alert = {
    "incident": {
        "incident_id": f"manual-{n + 1}",
        "resource_name": f"projects/{project}/services/{service_name}",
        "resource": {
            "type": resource_type,
            "labels": {
                "project_id": project,
                "service_name": service_name,
                "location": region,
            },
        },
        "metric": {
            "type": metric_filter,
            "displayName": policy.get("displayName", "?"),
        },
        "started_at": int(time.time()) - 3600,
        "ended_at": None,
        "state": "open",
        "condition_name": cond.get("displayName", policy.get("displayName", "?")),
        "url": f"https://console.cloud.google.com/monitoring/alerting?project={project}",
    }
}

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
json.dump(alert, tmp, indent=2)
tmp.close()

print(f"\nPolicy #{n + 1}: {policy.get('displayName')}")
print(f"Condition : {cond.get('displayName')}")
print(f"Resource  : {resource_type} / {service_name}")
print(f"Alert JSON: {tmp.name}\n")

os.execvp("python", ["python", "incident_graph.py", "--alert", tmp.name])
