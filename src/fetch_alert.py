"""Converte uma alert policy GCP para o formato do incident_graph.py e roda a investigação."""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

n = int(os.environ.get("N", "1")) - 1
project = os.environ.get("MON_PROJECT", "hw-prod-svc-sre-mon-a1f1")

token = subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode().strip()

url = f"https://monitoring.googleapis.com/v3/projects/{project}/alertPolicies"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
data = json.loads(urllib.request.urlopen(req).read())
policies = data.get("alertPolicies", [])

if n >= len(policies):
    print(f"Erro: só existem {len(policies)} policies. Use N entre 1 e {len(policies)}.")
    sys.exit(1)

policy = policies[n]
cond = policy.get("conditions", [{}])[0]
# Suporta conditionThreshold (metric-based) e conditionMatchedLog (log-based)
cond_threshold  = cond.get("conditionThreshold", {})
cond_log        = cond.get("conditionMatchedLog", {})
metric_filter   = cond_threshold.get("filter", "") or cond_log.get("filter", "")

resource_type = "cloud_run_revision"
service_name  = "unknown"
region        = "europe-west1"

for token_part in metric_filter.replace('"', ' ').split():
    if token_part in ("cloud_run_revision", "pubsub_subscription", "k8s_container", "gce_instance"):
        resource_type = token_part

for part in metric_filter.split('"'):
    if part and not part.startswith(" ") and "_service" in part:
        service_name = part
        break

# Fallback: extrair service do display name "[Cloudrun] payment-service-api response code"
if service_name == "unknown":
    import re
    display = policy.get("displayName", "")
    m = re.search(r'\]\s*([\w-]+(?:-api|-service|-worker|-job)?)', display)
    if m:
        service_name = m.group(1)

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
