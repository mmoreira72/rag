from __future__ import annotations

import json
import sys
from enum import Enum


class LLMRole(str, Enum):
    EXTRACTIVE = "extractive"  # log summarization, simple extraction
    ANALYTICAL = "analytical"  # root cause vs symptom, causal reasoning
    NARRATIVE  = "narrative"   # final report composition, prose synthesis


def parse_llm_json(raw: str, default=None):
    """
    Parse JSON from an LLM response. Returns `default` on failure,
    logging the failure to stderr. Never raises.
    """
    if not isinstance(raw, str):
        return default
    text = raw.strip()

    # Strip thinking blocks emitted by reasoning models (ministral, etc.)
    # Formats: [THINK]...[/THINK] or <think>...</think>
    import re as _re
    text = _re.sub(r'\[THINK\].*?\[/THINK\]', '', text, flags=_re.DOTALL).strip()
    text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[parse_llm_json] failed to parse JSON: {exc}", file=sys.stderr)
        return default
