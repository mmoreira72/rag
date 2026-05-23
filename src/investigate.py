"""
Investigate: extract distinct signals from a folder of JSON logs via LLM.

Usage:
    python investigate.py /tmp/myfolder

Reads *.json / *.jsonl from the folder, chunks the entries, asks the LLM to
extract distinct signals per chunk (per prompts/investigate_signals.txt),
deduplicates across chunks, and writes a markdown report into the same folder
as `signals-report-YYYYMMDD-HHMMSS.md`.

Env vars:
    CHAT_MODEL              chat model to use (see config.py — defaults apply)
    LLM_BASE_URL            OpenAI-compatible endpoint (see config.py)
    INVESTIGATE_CHUNK_CHARS max serialized chars per chunk (default 60000,
                            ~17k tokens — tuned for the 32k-ctx server. Lower
                            to ~20000 on 8k-ctx, ~8000 on 4k-ctx.)
    INVESTIGATE_OVERLAP     log lines of overlap between adjacent chunks (default 10)
    INVESTIGATE_MAX_FIELD_CHARS  per-field truncation cap (default 1500). Logs
                            from Apigee/SOAP/k8s often embed massive XML or
                            stack traces in single fields; truncating each
                            field individually preserves structure but stops
                            one verbose field from drowning the actual signal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from llm_client import client
from config import CHAT_MODEL

# Skill retrieval is optional — if PG_DSN points at a DB without the skills
# table (or pgvector isn't reachable), we want the extractor to still finish
# and produce a report. Surface the failure instead of crashing.
try:
    from rag import get_skills_with_confidence
    SKILL_RETRIEVAL_AVAILABLE = True
    SKILL_IMPORT_ERROR = ""
except Exception as _skill_exc:
    get_skills_with_confidence = None  # type: ignore[assignment]
    SKILL_RETRIEVAL_AVAILABLE = False
    SKILL_IMPORT_ERROR = str(_skill_exc)


CHUNK_CHARS = int(os.getenv("INVESTIGATE_CHUNK_CHARS", "60000"))
CHUNK_OVERLAP = int(os.getenv("INVESTIGATE_OVERLAP", "10"))
MAX_FIELD_CHARS = int(os.getenv("INVESTIGATE_MAX_FIELD_CHARS", "1500"))

PROMPT_PATH = Path(__file__).parent / "prompts" / "investigate_signals.txt"

REQUIRED_FIELDS = {"message", "severity"}


# ---------- load ----------

def load_logs_from_folder(folder: Path) -> tuple[list[dict], list[str]]:
    """Walk the folder, parse every *.json / *.jsonl, return (logs, files_read)."""
    logs: list[dict] = []
    files: list[str] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".json", ".jsonl"):
            continue
        # Never re-ingest our own output (alert payloads, prior reports, raw dumps).
        if path.name.startswith("signals-"):
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        loaded = _try_load_json(raw)
        if loaded is None:
            print(f"[warn] skipped {path.name}: not valid JSON or JSONL", flush=True)
            continue
        logs.extend(loaded)
        files.append(str(path.relative_to(folder)))
    return logs, files


def _try_load_json(raw: str) -> list[dict] | None:
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        return [data]
    out: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            out.append(obj)
    return out or None


# ---------- context extraction ----------

def extract_chunk_context(chunk: list[dict]) -> dict:
    """Mine GCP resource context (project, resource_type, service, region) from a
    chunk of logs. These fields come pre-structured from GCP — no need to ask
    the LLM to infer what's already in `resource.labels`.

    Returns the most common value per field. Empty string if not present.
    Region is derived from zone (europe-west1-c → europe-west1).
    """
    rtypes: dict[str, int] = {}
    projects: dict[str, int] = {}
    services: dict[str, int] = {}
    regions: dict[str, int] = {}

    for log in chunk:
        resource = log.get("resource", {}) if isinstance(log, dict) else {}
        rtype = resource.get("type", "") if isinstance(resource, dict) else ""
        if rtype:
            rtypes[rtype] = rtypes.get(rtype, 0) + 1
        labels = resource.get("labels", {}) if isinstance(resource, dict) else {}
        if not isinstance(labels, dict):
            labels = {}
        pid = labels.get("project_id", "")
        if pid:
            projects[pid] = projects.get(pid, 0) + 1
        # First non-empty: service_name (Cloud Run) | subscription_id (Pubsub)
        # | database_id (CloudSQL) | instance_id (GCE) | pod_name (k8s)
        for key in ("service_name", "subscription_id", "database_id",
                    "instance_id", "pod_name"):
            v = labels.get(key, "")
            if v:
                services[v] = services.get(v, 0) + 1
                break
        zone = labels.get("zone", "") or labels.get("location", "")
        if zone:
            region = zone.rsplit("-", 1)[0] if zone.count("-") >= 2 else zone
            regions[region] = regions.get(region, 0) + 1
        # GCE compute resource_name (e.g. "prd-wwn14") is more useful than
        # the bare instance_id (a number) — weight it higher.
        top_labels = log.get("labels", {}) if isinstance(log, dict) else {}
        if isinstance(top_labels, dict):
            cn = top_labels.get("compute.googleapis.com/resource_name", "")
            if cn:
                services[cn] = services.get(cn, 0) + 5

    def _top(d: dict) -> str:
        return max(d.items(), key=lambda x: x[1])[0] if d else ""

    return {
        "resource_type": _top(rtypes),
        "project":       _top(projects),
        "service":       _top(services),
        "region":        _top(regions) or "europe-west1",
    }


# ---------- preprocess ----------

def truncate_long_fields(obj, max_chars: int = MAX_FIELD_CHARS):
    """Recursively shorten any string > max_chars.

    GCP/Apigee logs often embed huge SOAP/XML bodies (originalRequestBody,
    errorContent, stack traces) that dwarf the useful fields. The model wastes
    its window on those and misses the actual signal. Keep head + tail so the
    LLM still sees what kind of field it was without choking on the full blob.
    """
    if isinstance(obj, str):
        if len(obj) > max_chars:
            head = obj[: max_chars * 2 // 3]
            tail = obj[-(max_chars // 3):]
            return f"{head}...[truncated {len(obj) - max_chars} chars]...{tail}"
        return obj
    if isinstance(obj, dict):
        return {k: truncate_long_fields(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [truncate_long_fields(v, max_chars) for v in obj]
    return obj


# ---------- chunk ----------

def chunk_logs(logs: list[dict], max_chars: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> Iterator[list[dict]]:
    """Yield chunks whose serialized size stays under max_chars.

    Char-based (not line-based) because GCP/k8s log lines vary 10x in size —
    a fixed line count blows the model's context on verbose payloads.
    """
    serialized_sizes = [
        len(json.dumps(line, ensure_ascii=False)) + 1  # +1 for the joining newline
        for line in logs
    ]
    n = len(logs)
    i = 0
    while i < n:
        total = 0
        j = i
        while j < n and (total + serialized_sizes[j] <= max_chars or j == i):
            total += serialized_sizes[j]
            j += 1
        yield logs[i:j]
        if j >= n:
            break
        i = max(j - overlap, i + 1)


# ---------- LLM ----------

def call_llm(prompt_with_logs: str) -> str:
    """Call the chat model. Try JSON mode first; fall back to plain if rejected."""
    messages = [
        {"role": "system",
         "content": "Respond with valid JSON only — no markdown, no prose."},
        {"role": "user", "content": prompt_with_logs},
    ]
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            stream=False,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            stream=False,
        )
    return resp.choices[0].message.content or ""


REWRITE_PROMPT = """You are rewriting a noisy raw log signal into a SHORT, alert-dialect summary suitable for embedding into an incident knowledge index.

RULES
- Output ONLY the rewritten message text. No JSON, no markdown, no quotes, no explanation, no preamble.
- Maximum 180 characters.
- Preserve: service/component name, error type, status code, error code, file basenames.
- Strip absolute file paths to basename only (e.g. `LanguageUtil.php`, not `/var/data/www/code/.../LanguageUtil.php`).
- Strip PIDs, TIDs, request IDs, client IPs, connection IDs, timestamps, URLs with query strings.
- Collapse repeated warnings into one summary with a count: `PHP Warning strpos() empty needle at LanguageUtil.php:31, UrlBuilder.php:173 (4x)`.
- DO NOT invent facts not in the input.
- If the input is already short, abstract, and clean, return it unchanged.

INPUT
{raw_message}

OUTPUT (one line, plain text, no quotes):"""


INFER_PROMPT = """You will receive a single alert/signal message. Extract the GCP context fields it implies and return them as JSON.

RULES
- Return ONLY a JSON object, no markdown, no prose.
- Fields to infer:
    - service:       the main service/component name (e.g. "hostelworld-web", "kong", "inventory-service-api-siteconnect-adapter"). Null if unclear.
    - project:       GCP project_id if mentioned in the message. Almost always null — leave it null unless explicitly present.
    - resource_type: one of "cloud_run_revision", "k8s_container", "k8s_pod", "pubsub_subscription", "cloudsql_database", "gce_instance", or null if you cannot tell.
      Hints:
        * apigee, kong, nginx ingress, PHP/FPM, web frontends → "cloud_run_revision"
        * k8s, kubernetes, istio, dapr, deployment, pod → "k8s_container"
        * pub/sub, subscription, topic → "pubsub_subscription"
        * mysql, postgres, cloudsql → "cloudsql_database"
    - region: GCP region if mentioned (e.g. "europe-west1", "us-central1"), else null.
- Do NOT invent values. When unsure, return null for that field.

INPUT
{message}

OUTPUT JSON (only this object, nothing else):"""


def infer_alert_fields(signal_message: str) -> dict:
    """LLM pass 3: best-effort extraction of {service, project, resource_type, region}.

    Used to build an alert payload that incident_graph.py can act on. Every
    field may be empty/null — incident_graph.py has defaults but works much
    better when at least service + resource_type are known.
    """
    prompt = INFER_PROMPT.format(message=signal_message)
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            stream=False,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "service":       (data.get("service") or "").strip(),
        "project":       (data.get("project") or "").strip(),
        "resource_type": (data.get("resource_type") or "").strip(),
        "region":        (data.get("region") or "europe-west1").strip(),
    }


def build_alert_json(signal: dict, fields: dict, idx: int) -> dict:
    """Build an alert payload in the shape incident_graph.py's parse_alert expects."""
    return {
        "incident": {
            "incident_id": f"investigate-{idx}",
            "resource": {
                "type": fields["resource_type"] or "cloud_run_revision",
                "labels": {
                    "project_id":   fields["project"],
                    "service_name": fields["service"],
                    "location":     fields["region"] or "europe-west1",
                },
            },
            "metric": {
                "type": "",
                "displayName": signal["message"],
            },
            "started_at":     int(time.time()) - 600,
            "ended_at":       None,
            "state":          "open",
            "condition_name": signal["message"],
            "url":            "",
        }
    }


def invoke_incident_graph(alert_path: Path, signal_idx: int,
                          out_folder: Path,
                          timeout_sec: int = 600) -> dict:
    """Run incident_graph.py --alert <alert_path>, return result dict.

    The incident graph pauses at notify_and_wait() expecting engineer input —
    we pipe "monitor" to stdin so it resumes and closes cleanly. The graph
    writes its report to /tmp; we copy it into out_folder as
    signals-incident-<idx>.md and return its new path.
    """
    cmd = [
        sys.executable, "incident_graph.py",
        "--alert", str(alert_path),
        "--thread", f"investigate-{signal_idx}",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input="monitor\n",
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=Path(__file__).parent,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "report_path": ""}
    except Exception as e:
        return {"status": f"error: {e}", "report_path": ""}

    # Parse the report path printed by notify_and_wait: "📄 Report saved: /tmp/..."
    saved_path = ""
    for line in proc.stdout.splitlines():
        if "Report saved:" in line:
            saved_path = line.split("Report saved:", 1)[1].strip()
            break

    if not saved_path or not Path(saved_path).exists():
        status = "ok-no-report" if proc.returncode == 0 else f"rc={proc.returncode}"
        return {"status": status, "report_path": "", "stderr_tail": proc.stderr[-500:]}

    # Copy the report into our folder so everything sits together.
    dst = out_folder / f"signals-incident-{signal_idx}.md"
    dst.write_text(Path(saved_path).read_text(encoding="utf-8"), encoding="utf-8")
    return {"status": "ok", "report_path": str(dst.relative_to(out_folder))}


def rewrite_message(raw: str) -> str:
    """Second LLM pass: turn a verbose / raw-copy signal message into alert dialect.

    Defensive: if the model returns empty / longer than original / something
    suspicious, fall back to the original so we never make things worse.
    """
    if len(raw) <= 180 and "/" not in raw:
        # Already short and no paths — skip the call.
        return raw
    prompt = REWRITE_PROMPT.format(raw_message=raw)
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=120,
            stream=False,
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception:
        return raw
    # Strip whatever wrapping the model added: quotes, backticks, leading "OUTPUT:" etc.
    out = out.strip("`'\"\n ")
    for prefix in ("OUTPUT:", "Output:", "output:"):
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
    # Reject garbage / no-op / longer-than-original
    if not out or len(out) >= len(raw) or "\n\n" in out:
        return raw
    return out


# ---------- parse ----------

def parse_signals(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return []
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
    signals = data.get("signals", []) if isinstance(data, dict) else []
    out: list[dict] = []
    for s in signals:
        if not isinstance(s, dict):
            continue
        if not REQUIRED_FIELDS.issubset(s.keys()):
            continue
        if not s.get("message"):
            continue
        out.append(s)
    return out


# ---------- dedup ----------

def signal_key(sig: dict) -> str:
    msg = sig["message"].lower().strip()
    return hashlib.sha1(msg.encode()).hexdigest()


# ---------- report ----------

def render_markdown(*, folder: Path, files: list[str], log_count: int,
                    chunk_count: int, signals: list[dict],
                    errors: list[dict], elapsed_sec: float,
                    raw_responses: list[tuple[int, str]] | None = None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []
    lines.append(f"# Investigation Signals — `{folder.name}`\n")
    lines.append(f"- Input: **{log_count}** log lines from **{len(files)}** file(s)")
    if files:
        lines.append("- Files: " + ", ".join(f"`{n}`" for n in files))
    lines.append(
        f"- Chunks processed: **{chunk_count}** "
        f"({chunk_count - len(errors)} ok, {len(errors)} failed)"
    )
    lines.append(f"- Signals extracted: **{len(signals)}** (after dedup)")
    lines.append(f"- Model: `{CHAT_MODEL}`")
    lines.append(f"- Chunk budget: {CHUNK_CHARS} chars (overlap {CHUNK_OVERLAP} lines)")
    lines.append(f"- Generated: {now} (took {elapsed_sec:.1f}s)\n")

    if signals:
        lines.append(f"## Signals ({len(signals)})\n")
        for s in signals:
            lines.append(f"### {s['message']}")
            lines.append(f"- severity: `{s.get('severity', '')}`")
            if s.get("service"):
                lines.append(f"- service: `{s['service']}`")
            if s.get("count") is not None:
                lines.append(f"- count: {s['count']}")
            if s.get("window"):
                lines.append(f"- window: {s['window']}")
            if s.get("evidence"):
                ev = str(s["evidence"]).replace("`", "'")
                lines.append(f"- evidence: `{ev}`")
            match = s.get("_skill_match")
            if match:
                if "error" in match:
                    lines.append(f"- matching skills: _retrieval failed: {match['error']}_")
                elif match.get("skills"):
                    skills_str = ", ".join(
                        f"`{sk['name']}` ({sk['score']})" for sk in match["skills"]
                    )
                    lines.append(
                        f"- matching skills ({match['confidence']}, "
                        f"top={match['top_score']}): {skills_str}"
                    )
                else:
                    lines.append(
                        f"- matching skills: _none above threshold "
                        f"(top={match.get('top_score', 0)})_"
                    )
            inferred = s.get("_inferred_fields")
            if inferred:
                lines.append(
                    f"- inferred: service=`{inferred['service'] or '?'}`, "
                    f"resource_type=`{inferred['resource_type'] or '?'}`, "
                    f"project=`{inferred['project'] or '?'}`"
                )
            alert_path = s.get("_alert_path")
            if alert_path:
                lines.append(f"- alert payload: [`{alert_path}`]({alert_path})")
            incident = s.get("_incident")
            if incident:
                if incident.get("report_path"):
                    rel = incident["report_path"]
                    lines.append(
                        f"- incident report: [`{rel}`]({rel}) ({incident['status']})"
                    )
                    # Inline the report content collapsed so users can read it
                    # without leaving the file. Wrapped in <details> so the
                    # report's own ## headings don't shuffle the toc.
                    report_full = folder / rel
                    if report_full.exists():
                        try:
                            inline = report_full.read_text(encoding="utf-8")
                            lines.append("")
                            lines.append("<details>")
                            lines.append(f"<summary>incident report — {rel}</summary>\n")
                            lines.append(inline)
                            lines.append("\n</details>")
                        except Exception as e:
                            lines.append(f"  - _failed to inline report: {e}_")
                else:
                    lines.append(f"- incident: _{incident['status']}_")
                    if alert_path:
                        lines.append(
                            f"  - manual rerun: `make local-incident "
                            f"ALERT={folder}/{alert_path}`"
                        )
            lines.append("")
    else:
        lines.append("## Signals\n\nNo signals extracted.\n")

    if errors:
        lines.append("## Errors\n")
        for e in errors:
            lines.append(f"- chunk {e['chunk']} / {e['stage']}: {e['err']}")
        lines.append("")

    if raw_responses:
        lines.append("<details>")
        lines.append("<summary>Raw LLM responses (debug)</summary>\n")
        for idx, raw in raw_responses:
            lines.append(f"### chunk {idx}\n")
            lines.append("```")
            lines.append(raw)
            lines.append("```\n")
        lines.append("</details>")
    return "\n".join(lines)


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract signals from a folder of JSON logs via LLM."
    )
    parser.add_argument("folder", help="Folder containing *.json or *.jsonl log files")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"error: {folder} is not a directory", file=sys.stderr)
        return 2

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    print(f"[load] reading logs from {folder}", flush=True)
    logs, files = load_logs_from_folder(folder)
    if not logs:
        print("[load] no logs found — nothing to do", flush=True)
        return 1

    raw_size = sum(len(json.dumps(l, ensure_ascii=False)) for l in logs)
    logs = [truncate_long_fields(l) for l in logs]
    trunc_size = sum(len(json.dumps(l, ensure_ascii=False)) for l in logs)
    if trunc_size < raw_size:
        pct = 100 * (raw_size - trunc_size) / raw_size
        print(
            f"[load] truncated long fields: {raw_size} → {trunc_size} chars "
            f"({pct:.0f}% reduction, cap {MAX_FIELD_CHARS} chars/field)",
            flush=True,
        )

    chunks = list(chunk_logs(logs))
    print(
        f"[load] {len(logs)} log lines from {len(files)} file(s) → "
        f"{len(chunks)} chunk(s), budget {CHUNK_CHARS} chars/chunk "
        f"(overlap {CHUNK_OVERLAP} lines)",
        flush=True,
    )

    seen: set[str] = set()
    signals: list[dict] = []
    errors: list[dict] = []
    raw_responses: list[tuple[int, str]] = []

    start = time.time()
    for idx, chunk in enumerate(chunks):
        logs_text = "\n".join(json.dumps(line, ensure_ascii=False) for line in chunk)
        filled_prompt = system_prompt.replace("{logs_here}", logs_text)

        print(
            f"[llm] chunk {idx + 1}/{len(chunks)} "
            f"({len(chunk)} lines, ~{len(filled_prompt)} chars)...",
            flush=True,
        )
        try:
            raw = call_llm(filled_prompt)
        except Exception as e:
            errors.append({"chunk": idx, "stage": "llm", "err": str(e)})
            msg = str(e)
            print(f"  [error] LLM failed: {msg}", flush=True)
            if "n_ctx" in msg or "context length" in msg.lower() or "context_length" in msg:
                print(
                    "  [hint] chunk exceeds the model's context window. Either:\n"
                    "         - raise the model's n_ctx (LM Studio: model settings → context length)\n"
                    f"         - or shrink chunks: INVESTIGATE_CHUNK_CHARS=<smaller than {CHUNK_CHARS}> make investigate FOLDER=...",
                    flush=True,
                )
            continue

        raw_responses.append((idx, raw))
        chunk_signals = parse_signals(raw)

        # Attach the chunk's resource context to every signal it produced.
        # These structured fields (project_id, resource.type, etc.) come
        # straight from GCP — way more reliable than asking the LLM to infer.
        chunk_ctx = extract_chunk_context(chunk)
        for sig in chunk_signals:
            sig["_chunk_context"] = chunk_ctx

        # Second pass: rewrite each message into alert dialect. Done BEFORE
        # dedup so near-identical raw signals collapse to the same hash.
        for sig in chunk_signals:
            original = sig["message"]
            rewritten = rewrite_message(original)
            if rewritten != original:
                print(
                    f"  [rewrite] {original[:60]}{'...' if len(original) > 60 else ''}",
                    flush=True,
                )
                print(f"         → {rewritten}", flush=True)
                sig["_original_message"] = original
                sig["message"] = rewritten

        added = 0
        for sig in chunk_signals:
            key = signal_key(sig)
            if key in seen:
                continue
            seen.add(key)
            signals.append(sig)
            added += 1
        print(f"  → {len(chunk_signals)} signals ({added} new after dedup)", flush=True)

    elapsed = time.time() - start

    # ---------- per-signal skill retrieval ----------
    # Each signal's `message` is run through the same skills index used by the
    # alert pipeline (rag.get_skills_with_confidence). The match is attached to
    # the signal as `_skill_match` so the report can show "this signal would
    # find skill X" without us having to invoke the incident graph.
    if SKILL_RETRIEVAL_AVAILABLE and signals:
        print(f"\n[skills] retrieving matches for {len(signals)} signal(s)...", flush=True)
        for sig in signals:
            try:
                res = get_skills_with_confidence(sig["message"])
                sig["_skill_match"] = {
                    "confidence": res.confidence,
                    "top_score": round(res.top_score, 3),
                    "skills": [
                        {"name": s.name, "score": round(s.score, 3)}
                        for s in res.skills
                    ],
                }
            except Exception as e:
                sig["_skill_match"] = {"error": str(e)}
                print(f"  [warn] skill retrieval failed for one signal: {e}", flush=True)
    elif signals:
        print(
            f"\n[skills] skill retrieval disabled — rag.py import failed "
            f"({SKILL_IMPORT_ERROR or 'unknown'}). Set PG_DSN to the skills "
            "DB and re-run if you want matches in the report.",
            flush=True,
        )

    # ---------- per-signal alert JSON + auto-invoke ----------
    # For each signal: infer {service, project, resource_type} (LLM pass 3),
    # write a signals-alert-<N>.json next to the report. If the skill match was
    # high-confidence AND we know enough to build a real query, also auto-invoke
    # incident_graph.py to produce a full incident report. Set
    # INVESTIGATE_NO_INVOKE=1 to disable auto-invocation.
    no_invoke = os.getenv("INVESTIGATE_NO_INVOKE", "") == "1"
    if signals:
        print(f"\n[handoff] building alert payloads for {len(signals)} signal(s)...", flush=True)
        for idx, sig in enumerate(signals, start=1):
            # Prefer the chunk's structured context (from resource.labels) over
            # LLM inference. Call the LLM only to fill gaps the chunk didn't have.
            ctx = sig.get("_chunk_context") or {}
            need_llm = not (ctx.get("service") and ctx.get("resource_type"))
            inferred_via_llm = infer_alert_fields(sig["message"]) if need_llm else {}
            fields = {
                "service":       ctx.get("service")       or inferred_via_llm.get("service", ""),
                "project":       ctx.get("project")       or inferred_via_llm.get("project", ""),
                "resource_type": ctx.get("resource_type") or inferred_via_llm.get("resource_type", ""),
                "region":        ctx.get("region")        or inferred_via_llm.get("region", "europe-west1"),
            }
            sig["_inferred_fields"] = fields
            alert_json = build_alert_json(sig, fields, idx)
            alert_path = folder / f"signals-alert-{idx}.json"
            alert_path.write_text(json.dumps(alert_json, indent=2), encoding="utf-8")
            sig["_alert_path"] = alert_path.name
            print(
                f"  [{idx}] fields: service={fields['service'] or '?'} "
                f"resource_type={fields['resource_type'] or '?'} "
                f"project={fields['project'] or '?'}  → {alert_path.name}",
                flush=True,
            )

            match = sig.get("_skill_match") or {}
            high_conf  = match.get("confidence") == "high"
            actionable = bool(fields["service"]) and bool(fields["resource_type"])
            if no_invoke:
                sig["_incident"] = {"status": "skipped (INVESTIGATE_NO_INVOKE=1)", "report_path": ""}
            elif high_conf and actionable:
                print(f"      [invoke] running incident_graph.py for signal {idx}...", flush=True)
                t0 = time.time()
                result = invoke_incident_graph(alert_path, idx, folder)
                dt = time.time() - t0
                sig["_incident"] = result
                if result.get("report_path"):
                    print(f"      [invoke] {result['status']} ({dt:.0f}s) → {result['report_path']}", flush=True)
                else:
                    print(f"      [invoke] {result['status']} ({dt:.0f}s, no report)", flush=True)
            else:
                reason = []
                if not high_conf:  reason.append(f"skill confidence={match.get('confidence', 'n/a')}")
                if not actionable: reason.append("missing service/resource_type")
                sig["_incident"] = {"status": f"skipped ({', '.join(reason)})", "report_path": ""}

    report = render_markdown(
        folder=folder,
        files=files,
        log_count=len(logs),
        chunk_count=len(chunks),
        signals=signals,
        errors=errors,
        elapsed_sec=elapsed,
        raw_responses=raw_responses,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = folder / f"signals-report-{ts}.md"
    out_path.write_text(report, encoding="utf-8")

    raw_path = folder / f"signals-raw-{ts}.txt"
    raw_path.write_text(
        "\n\n".join(
            f"=== chunk {i} ===\n{r}" for i, r in raw_responses
        ),
        encoding="utf-8",
    )

    print(f"\n[done] {len(signals)} signals, {len(errors)} errors", flush=True)
    print(f"[done] report → {out_path}", flush=True)
    print(f"[done] raw LLM responses → {raw_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
