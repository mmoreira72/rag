"""Lista alert policies do GCP Cloud Monitoring."""
import json
import os
import subprocess
import urllib.request

project = os.environ.get("MON_PROJECT", "hw-prod-svc-sre-mon-a1f1")
token = subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode().strip()

url = f"https://monitoring.googleapis.com/v3/projects/{project}/alertPolicies"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
data = json.loads(urllib.request.urlopen(req).read())
policies = data.get("alertPolicies", [])

print(f"\n{'#':<4} {'NOME':<60} {'ENABLED'}")
print("-" * 72)
for i, p in enumerate(policies, 1):
    name = p.get("displayName", "?")[:59]
    enabled = "✓" if p.get("enabled") else "✗"
    print(f"{i:<4} {name:<60} {enabled}")
print(f"\n{len(policies)} policies em {project}")
print("\nPara investigar: make local-fetch-alert N=<número>")
