"""Tests for the configurable LLM layer (Sprint 1)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make src importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_roles import LLMRole, parse_llm_json


def _make_fake_chatai(captured: dict):
    """Return a ChatOpenAI replacement that records constructor kwargs."""
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def invoke(self, messages):
            msg = MagicMock()
            msg.content = '{"ok": true}'
            return msg

    return FakeChatOpenAI


def test_call_llm_uses_role_defaults():
    """NARRATIVE role must pass temperature=0.3 to ChatOpenAI."""
    captured = {}
    with patch("incident_graph.ChatOpenAI", _make_fake_chatai(captured)):
        from incident_graph import call_llm
        call_llm("sys", "user", role=LLMRole.NARRATIVE)

    assert captured["temperature"] == 0.3, (
        f"Expected temperature=0.3 for NARRATIVE, got {captured['temperature']}"
    )


def test_call_llm_temperature_override():
    """Explicit temperature must beat the role default."""
    captured = {}
    with patch("incident_graph.ChatOpenAI", _make_fake_chatai(captured)):
        from incident_graph import call_llm
        call_llm("sys", "user", role=LLMRole.NARRATIVE, temperature=0.0)

    assert captured["temperature"] == 0.0, (
        f"Expected override temperature=0.0, got {captured['temperature']}"
    )


def test_call_llm_json_mode_sets_response_format():
    """json_mode=True + LLM_JSON_FORMAT=True must pass model_kwargs with response_format=json_object."""
    import incident_graph as ig
    import incident_graph as _config_ref  # same module object used at call time
    captured = {}
    with patch("incident_graph.ChatOpenAI", _make_fake_chatai(captured)), \
         patch("incident_graph._config.LLM_JSON_FORMAT", True):
        ig.call_llm("sys", "user", role=LLMRole.EXTRACTIVE, json_mode=True)

    assert "model_kwargs" in captured, "model_kwargs not passed when LLM_JSON_FORMAT=True and json_mode=True"
    fmt = captured["model_kwargs"].get("response_format", {})
    assert fmt.get("type") == "json_object", (
        f"Expected response_format.type=json_object, got {fmt}"
    )


def test_call_llm_json_mode_no_response_format_without_flag():
    """json_mode=True without LLM_JSON_FORMAT must NOT pass model_kwargs (LM Studio compat)."""
    import incident_graph as ig
    captured = {}
    with patch("incident_graph.ChatOpenAI", _make_fake_chatai(captured)), \
         patch("incident_graph._config.LLM_JSON_FORMAT", False):
        ig.call_llm("sys", "user", role=LLMRole.EXTRACTIVE, json_mode=True)

    assert "model_kwargs" not in captured, (
        "model_kwargs must NOT be passed when LLM_JSON_FORMAT=False (LM Studio compat)"
    )


def test_parse_llm_json_handles_garbage():
    """Malformed JSON must return default and not raise."""
    result = parse_llm_json("this is not json", default="FALLBACK")
    assert result == "FALLBACK"


def test_parse_llm_json_valid():
    """Valid JSON string must be parsed correctly."""
    result = parse_llm_json('[{"count": 1, "message": "err"}]')
    assert result == [{"count": 1, "message": "err"}]


def test_parse_llm_json_strips_markdown_fences():
    """JSON wrapped in markdown code fences must still parse."""
    raw = "```json\n[{\"ok\": true}]\n```"
    result = parse_llm_json(raw)
    assert result == [{"ok": True}]
