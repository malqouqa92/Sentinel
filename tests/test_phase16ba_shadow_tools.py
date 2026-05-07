"""Phase 16 Batch A -- tools-enabled shadow planner.

The Phase 15c shadow plan was a blind one-shot text completion.
That capped agreement scores around 0.75 because Qwen had to guess
what the file looked like; on weaker problem classes (test, skill)
it dropped to 0.30 and produced confabulations (telegram-bot
tutorial in `SEN-fec42f17`'s shadow recipe).

Batch A gives Qwen a READ-ONLY tool surface (read_file, list_dir)
during shadow planning. NO write_file, NO edit_file, NO run_bash --
shadow is measurement, not execution. Hard cap of 5 tool calls
plus a final recipe-output turn; 90s overall timeout.

Coverage:
  Read-only tool surface:
    A01 -- SHADOW_TOOLS_SCHEMA contains read_file + list_dir
    A02 -- SHADOW_TOOLS_SCHEMA does NOT contain write_file
    A03 -- SHADOW_TOOLS_SCHEMA does NOT contain edit_file
    A04 -- SHADOW_TOOLS_SCHEMA does NOT contain run_bash
    A05 -- SHADOW_TOOLS_SCHEMA does NOT contain done

  run_shadow_planner behavior (with stubbed _ollama_chat):
    A11 -- recipe-only response: returns recipe immediately, 0 tool calls
    A12 -- one read_file then recipe: returns recipe, 1 tool call
    A13 -- max_tool_calls cap stops exploration
    A14 -- write_file attempt is REFUSED (logged + tool error fed back)
    A15 -- run_bash attempt is REFUSED (logged + tool error fed back)
    A16 -- ollama_chat exception returns empty recipe + error string
    A17 -- response with no tool calls AND no STEP-N content returns empty recipe

  Wiring + config:
    A21 -- SHADOW_PLAN_USE_TOOLS exists, default True
    A22 -- QWEN_SHADOW_TIMEOUT_S is 90 (was 30 in Phase 15c)
    A23 -- _qwen_shadow_plan dispatches to run_shadow_planner when flag is True
    A24 -- _qwen_shadow_plan falls back to one-shot when flag is False
    A25 -- timeout path returns None (best-effort contract preserved)

  Recipe output shape:
    A31 -- emitted recipe has STEP N: prefix on each line
    A32 -- recipe parses via _parse_recipe_steps to >= 1 step
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import qwen_agent
from core.qwen_agent import (
    SHADOW_TOOLS_SCHEMA, SHADOW_MAX_TOOL_CALLS,
    SHADOW_PLANNER_SYSTEM, run_shadow_planner,
    _parse_recipe_steps,
)
import skills.code_assist as ca


# ─────────────────────────────────────────────────────────────────
# Read-only tool surface
# ─────────────────────────────────────────────────────────────────


def _tool_names(schema):
    return {t["function"]["name"] for t in schema}


def test_a01_shadow_schema_has_read_file_and_list_dir():
    names = _tool_names(SHADOW_TOOLS_SCHEMA)
    assert "read_file" in names
    assert "list_dir" in names


def test_a02_shadow_schema_no_write_file():
    assert "write_file" not in _tool_names(SHADOW_TOOLS_SCHEMA)


def test_a03_shadow_schema_no_edit_file():
    assert "edit_file" not in _tool_names(SHADOW_TOOLS_SCHEMA)


def test_a04_shadow_schema_no_run_bash():
    assert "run_bash" not in _tool_names(SHADOW_TOOLS_SCHEMA)


def test_a05_shadow_schema_no_done_tool():
    """done() is for execution, not planning. Shadow output is the
    recipe text itself; no tool needed."""
    assert "done" not in _tool_names(SHADOW_TOOLS_SCHEMA)


# ─────────────────────────────────────────────────────────────────
# run_shadow_planner behavior (stubbed _ollama_chat)
# ─────────────────────────────────────────────────────────────────


def _make_response(content="", tool_calls=None):
    """Build the dict shape that _ollama_chat returns."""
    return {
        "message": {
            "content": content,
            "tool_calls": tool_calls or [],
        },
    }


def _stub_ollama(monkeypatch, sequence):
    """Replace _ollama_chat with one that yields from the supplied
    response sequence in order. Returns a list of (model, messages)
    pairs the function was called with for inspection."""
    calls = []

    def fake_chat(model, messages, tools=None, **kwargs):
        calls.append((model, messages))
        if calls.__len__() > len(sequence):
            return sequence[-1]  # stick on the last response
        return sequence[calls.__len__() - 1]

    monkeypatch.setattr(qwen_agent, "_ollama_chat", fake_chat)
    return calls


def test_a11_recipe_only_response_no_tool_calls(monkeypatch):
    """If Qwen emits the recipe in the first turn with no tool
    calls, run_shadow_planner returns it immediately."""
    seq = [_make_response(content=(
        'STEP 1: read_file path="x.py"\n'
        'STEP 2: done summary="ok"'
    ))]
    _stub_ollama(monkeypatch, seq)
    result = run_shadow_planner(
        problem="test", kb_context_block="", project_map="",
        trace_id="SEN-test", model="m",
    )
    assert result["error"] is None
    assert result["tool_calls"] == 0
    # Recipe round-trips through _parse_recipe_steps so it's not the
    # raw input -- but it should still have >=2 STEP blocks.
    steps = _parse_recipe_steps(result["recipe"])
    assert len(steps) >= 2


def test_a12_one_read_file_then_recipe(monkeypatch):
    """Qwen reads one file, then outputs a recipe."""
    seq = [
        _make_response(content="", tool_calls=[{
            "id": "c1",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "core/util.py"}),
            },
        }]),
        _make_response(content=(
            'STEP 1: edit_file path="core/util.py" old="x" new="y"\n'
            'STEP 2: done summary="ok"'
        )),
    ]
    _stub_ollama(monkeypatch, seq)
    result = run_shadow_planner(
        problem="test", kb_context_block="", project_map="",
        trace_id="SEN-test", model="m",
    )
    assert result["tool_calls"] == 1
    steps = _parse_recipe_steps(result["recipe"])
    assert len(steps) >= 1


def test_a13_max_tool_calls_cap(monkeypatch):
    """If Qwen keeps calling tools past the cap, the loop forces
    a final recipe turn."""
    # 6 read_file calls then a recipe -- cap is 5, so we should
    # only execute 5 tools then push for the recipe.
    read_call = {
        "id": "c", "function": {
            "name": "read_file",
            "arguments": json.dumps({"path": "math_utils.py"}),
        },
    }
    seq = [_make_response(content="", tool_calls=[read_call])
           for _ in range(SHADOW_MAX_TOOL_CALLS + 1)]
    seq.append(_make_response(content=(
        'STEP 1: done summary="hit cap"'
    )))
    _stub_ollama(monkeypatch, seq)
    result = run_shadow_planner(
        problem="test", kb_context_block="", project_map="",
        trace_id="SEN-test", model="m",
        max_tool_calls=SHADOW_MAX_TOOL_CALLS,
    )
    assert result["tool_calls"] <= SHADOW_MAX_TOOL_CALLS


def test_a14_write_file_attempt_is_refused(monkeypatch):
    """If Qwen tries to call write_file, the loop refuses + feeds
    back an error tool message. The write does NOT execute."""
    seq = [
        _make_response(content="", tool_calls=[{
            "id": "c1",
            "function": {
                "name": "write_file",
                "arguments": json.dumps({
                    "path": "core/danger.py",
                    "content": "import os; os.remove('/etc/passwd')",
                }),
            },
        }]),
        _make_response(content=(
            'STEP 1: done summary="bailed"'
        )),
    ]
    calls = _stub_ollama(monkeypatch, seq)
    result = run_shadow_planner(
        problem="test", kb_context_block="", project_map="",
        trace_id="SEN-test", model="m",
    )
    # The 2nd ollama call has a tool message in its history
    # explaining the refusal.
    second_call_messages = calls[1][1]
    tool_msgs = [m for m in second_call_messages if m["role"] == "tool"]
    assert tool_msgs, "no tool result fed back after refused write_file"
    err_blob = tool_msgs[-1]["content"]
    assert "not allowed" in err_blob or "read-only" in err_blob


def test_a15_run_bash_attempt_is_refused(monkeypatch):
    """run_bash is the most dangerous tool. Shadow MUST refuse it."""
    seq = [
        _make_response(content="", tool_calls=[{
            "id": "c1",
            "function": {
                "name": "run_bash",
                "arguments": json.dumps({"command": "rm -rf core"}),
            },
        }]),
        _make_response(content=(
            'STEP 1: done summary="x"'
        )),
    ]
    calls = _stub_ollama(monkeypatch, seq)
    run_shadow_planner(
        problem="test", kb_context_block="", project_map="",
        trace_id="SEN-test", model="m",
    )
    second_call_messages = calls[1][1]
    tool_msgs = [m for m in second_call_messages if m["role"] == "tool"]
    assert tool_msgs
    err_blob = tool_msgs[-1]["content"]
    assert "not allowed" in err_blob or "read-only" in err_blob


def test_a16_ollama_exception_returns_empty(monkeypatch):
    def raise_chat(*a, **kw):
        raise TimeoutError("simulated")
    monkeypatch.setattr(qwen_agent, "_ollama_chat", raise_chat)
    result = run_shadow_planner(
        problem="test", kb_context_block="", project_map="",
        trace_id="SEN-test", model="m",
    )
    assert result["recipe"] == ""
    assert result["error"] is not None
    assert "TimeoutError" in result["error"]


def test_a17_no_recipe_no_tools_returns_empty(monkeypatch):
    """Qwen emits prose with no STEP blocks and no tool calls --
    we get back an empty recipe rather than the prose."""
    seq = [_make_response(content=(
        "I think the answer is to do something interesting"
    ))]
    _stub_ollama(monkeypatch, seq)
    result = run_shadow_planner(
        problem="test", kb_context_block="", project_map="",
        trace_id="SEN-test", model="m",
    )
    assert result["recipe"] == ""


# ─────────────────────────────────────────────────────────────────
# Wiring + config
# ─────────────────────────────────────────────────────────────────


def test_a21_shadow_use_tools_flag_exists():
    """The flag exists and is a bool. Default flipped True -> False
    in the Batch A revert (stress test surfaced 180s+ timeouts on
    this hardware). The agentic path is preserved behind the flag;
    flip back to True once inference-time controls land or on
    bigger GPU."""
    assert hasattr(ca, "SHADOW_PLAN_USE_TOOLS")
    assert isinstance(ca.SHADOW_PLAN_USE_TOOLS, bool)


def test_a22_shadow_timeout_accommodates_cold_load():
    """Phase 15c shipped at 30s. Phase 16 Batch A initially picked
    90s (multi-turn agentic budget). First production trace
    (SEN-bbef3a4f) timed out at 90s on a cold qwen2.5-coder:3b
    load -- bumped to 180s. Test allows 90-300s so future
    re-tunings don't break the suite."""
    assert 90 <= ca.QWEN_SHADOW_TIMEOUT_S <= 300


def test_a23_qwen_shadow_plan_dispatches_to_run_shadow_planner(monkeypatch):
    """When SHADOW_PLAN_USE_TOOLS=True, _qwen_shadow_plan calls
    run_shadow_planner instead of the one-shot _qwen_generate."""
    monkeypatch.setattr(ca, "SHADOW_PLAN_USE_TOOLS", True)
    called = {"shadow_planner": 0, "qwen_generate": 0}

    def fake_planner(*a, **kw):
        called["shadow_planner"] += 1
        return {"recipe": "STEP 1: done summary=\"x\"",
                "tool_calls": 1, "error": None}

    def fake_generate(*a, **kw):
        called["qwen_generate"] += 1
        return ""

    monkeypatch.setattr(qwen_agent, "run_shadow_planner", fake_planner)
    monkeypatch.setattr(ca, "_qwen_generate", fake_generate)
    result = asyncio.run(ca._qwen_shadow_plan(
        problem="x", code_context=None,
        kb_patterns_block="", project_map="",
        backend_model="m", trace_id="SEN-test",
    ))
    assert called["shadow_planner"] == 1
    assert called["qwen_generate"] == 0
    assert result is not None


def test_a24_qwen_shadow_plan_falls_back_to_one_shot(monkeypatch):
    """When SHADOW_PLAN_USE_TOOLS=False, the legacy one-shot path
    runs."""
    monkeypatch.setattr(ca, "SHADOW_PLAN_USE_TOOLS", False)
    called = {"shadow_planner": 0, "qwen_generate": 0}

    def fake_planner(*a, **kw):
        called["shadow_planner"] += 1
        return {"recipe": "x", "tool_calls": 0, "error": None}

    def fake_generate(*a, **kw):
        called["qwen_generate"] += 1
        return 'STEP 1: done summary="x"'

    monkeypatch.setattr(qwen_agent, "run_shadow_planner", fake_planner)
    monkeypatch.setattr(ca, "_qwen_generate", fake_generate)
    asyncio.run(ca._qwen_shadow_plan(
        problem="x", code_context=None,
        kb_patterns_block="", project_map="",
        backend_model="m", trace_id="SEN-test",
    ))
    assert called["shadow_planner"] == 0
    assert called["qwen_generate"] == 1


def test_a25_timeout_returns_none(monkeypatch):
    """If the agentic shadow times out, _qwen_shadow_plan returns
    None (best-effort: caller continues with Claude's recipe)."""
    monkeypatch.setattr(ca, "SHADOW_PLAN_USE_TOOLS", True)
    monkeypatch.setattr(ca, "QWEN_SHADOW_TIMEOUT_S", 0.05)

    def slow_planner(*a, **kw):
        import time
        time.sleep(2.0)
        return {"recipe": "x", "tool_calls": 0, "error": None}

    monkeypatch.setattr(qwen_agent, "run_shadow_planner", slow_planner)
    result = asyncio.run(ca._qwen_shadow_plan(
        problem="x", code_context=None,
        kb_patterns_block="", project_map="",
        backend_model="m", trace_id="SEN-test",
    ))
    assert result is None


# ─────────────────────────────────────────────────────────────────
# Recipe output shape
# ─────────────────────────────────────────────────────────────────


def test_a31_emitted_recipe_has_step_prefix(monkeypatch):
    """run_shadow_planner re-emits the recipe with STEP N: prefixes
    so callers (plan_agreement scorer) can parse it consistently."""
    seq = [_make_response(content=(
        'STEP 1: edit_file path="x.py" old="a" new="b"\n'
        'STEP 2: done summary="ok"'
    ))]
    _stub_ollama(monkeypatch, seq)
    result = run_shadow_planner(
        problem="test", kb_context_block="", project_map="",
        trace_id="SEN-test", model="m",
    )
    assert "STEP 1:" in result["recipe"]
    assert "STEP 2:" in result["recipe"]


def test_a40_max_tool_calls_tightened_to_3():
    """Phase 16 Batch A polish: SHADOW_MAX_TOOL_CALLS dropped from
    5 to 3 after SEN-6a8a6539 showed Qwen burning all 5 slots on
    file reads without emitting the recipe. Tighter budget forces
    the model to transition to recipe output sooner."""
    assert SHADOW_MAX_TOOL_CALLS == 3, (
        f"SHADOW_MAX_TOOL_CALLS = {SHADOW_MAX_TOOL_CALLS} "
        f"-- Batch A polish was supposed to set this to 3"
    )


def test_a41_planner_prompt_forbids_chained_reads():
    """Phase 16 Batch A polish: the system prompt now explicitly
    forbids chains of read_file calls. The 3B coder model needs
    explicit instruction to STOP exploring and start planning."""
    # The strengthened prompt MUST contain language that makes the
    # transition-to-recipe directive clear.
    must_have = [
        "ONE OR TWO reads at MOST",  # explicit budget statement
        "next response MUST",          # transition directive
        "do NOT chain",                # negative directive
    ]
    for phrase in must_have:
        assert phrase in SHADOW_PLANNER_SYSTEM, (
            f"prompt missing the directive '{phrase}'"
        )


def test_a42_planner_prompt_mentions_3_call_budget():
    """Sanity: the prompt names the 3-call budget explicitly so
    Qwen knows the constraint, not just feels it via the cap."""
    assert "3 tool calls TOTAL" in SHADOW_PLANNER_SYSTEM


def test_a32_recipe_parses_through_canonical_parser(monkeypatch):
    seq = [_make_response(content=(
        'STEP 1: read_file path="a.py"\n'
        'STEP 2: edit_file path="a.py" old="x" new="y"\n'
        'STEP 3: done summary="changed"'
    ))]
    _stub_ollama(monkeypatch, seq)
    result = run_shadow_planner(
        problem="test", kb_context_block="", project_map="",
        trace_id="SEN-test", model="m",
    )
    steps = _parse_recipe_steps(result["recipe"])
    assert len(steps) >= 3
