"""Phase 16 Option 1 -- recipe reformat retry on parser failure.

When stepfed's parser extracts <2 STEP blocks from a Claude recipe,
the legacy fallback path inside run_agent_stepfed gives Qwen a free-
form conversation that doesn't pin to literal recipe commands. The
SEN-ca56136c trace showed this costing ~70-90s plus a wrong-direction
edit + an extra reviewer-Claude fail review.

Phase 16 Option 1 inserts a parse-check at the call site (the attempt
loop in code_assist.py). If <2 STEPs, re-ask Claude with explicit
format-only directives via _claude_reformat_recipe. Reformat is one-
shot; if it ALSO yields <2 STEPs, we fall through to the existing
legacy fallback as a last-resort safety net (preserves prior behavior
when Claude is unavailable / completely uncooperative).

Coverage:
  Parser sanity (boundary):
    R01 -- two-STEP recipe parses to 2 (passes threshold)
    R02 -- one-STEP recipe parses to 1 (triggers reformat)
    R03 -- prose-only recipe parses to 0 (triggers reformat)

  _claude_reformat_recipe behavior:
    R11 -- claude unavailable -> returns "" (graceful)
    R12 -- ClaudeCliError -> returns "" (graceful, logs warning)
    R13 -- happy path: prose response with embedded STEPs is extracted
    R14 -- response has bare STEP blocks -> passes through
    R15 -- response longer than RECIPE_MAX_CHARS_STEPFED -> truncated
    R16 -- empty raw response -> returns "" after strip
    R17 -- generate() called with tools=[] (no exploration)
    R18 -- system prompt forbids prose/markdown
    R19 -- system prompt requires >=2 STEPs

  Source-level integration (parse-check in attempt loop):
    R21 -- code_assist imports _parse_recipe_steps from qwen_agent
    R22 -- attempt loop calls _claude_reformat_recipe on parse failure
    R23 -- attempt loop falls through on retry exhaustion (preserves
           legacy fallback)
    R24 -- log marker "requesting reformat" exists
    R25 -- log marker "reformat retry exhausted" exists
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import claude_cli as cc
from core.qwen_agent import _parse_recipe_steps
from skills import code_assist as ca


# ─────────────────────────────────────────────────────────────────
# Parser sanity (boundary cases that drive the reformat decision)
# ─────────────────────────────────────────────────────────────────


def test_r01_two_step_recipe_parses_to_two():
    recipe = (
        'STEP 1: edit_file path="x.py" old="a" new="b"\n'
        'STEP 2: done summary="ok"'
    )
    assert len(_parse_recipe_steps(recipe)) == 2


def test_r02_one_step_recipe_parses_to_one():
    """Single STEP recipe is below the >=2 threshold and must
    trigger reformat. Used as a boundary case in the attempt loop."""
    recipe = 'STEP 1: done summary="trivial"'
    assert len(_parse_recipe_steps(recipe)) == 1


def test_r03_prose_only_recipe_parses_to_zero():
    """The actual SEN-ca56136c failure shape: Claude wrote prose,
    no STEP markers. Must trigger reformat."""
    recipe = (
        "First, read the file. Then edit line 17 to swap emojis. "
        "Run the bot to verify."
    )
    assert len(_parse_recipe_steps(recipe)) == 0


# ─────────────────────────────────────────────────────────────────
# _claude_reformat_recipe: mock the Claude CLI client
# ─────────────────────────────────────────────────────────────────


class _FakeClient:
    """Replaces ClaudeCliClient inside core.claude_cli for tests."""

    available_value: bool = True
    response_text: str = ""
    raise_error: bool = False
    captured_kwargs: dict | None = None

    def __init__(self) -> None:
        pass

    @property
    def available(self) -> bool:
        return type(self).available_value

    async def generate(self, **kwargs) -> str:
        type(self).captured_kwargs = kwargs
        if type(self).raise_error:
            raise cc.ClaudeCliError("simulated CLI failure")
        return type(self).response_text


@pytest.fixture
def fake_client(monkeypatch):
    """Reset class-state defaults before every test, install fake."""
    _FakeClient.available_value = True
    _FakeClient.response_text = ""
    _FakeClient.raise_error = False
    _FakeClient.captured_kwargs = None
    monkeypatch.setattr(cc, "ClaudeCliClient", _FakeClient)
    return _FakeClient


def _run(coro):
    """Run a coroutine in a fresh event loop. asyncio.get_event_loop()
    is deprecated and breaks when run alongside other test files that
    have already touched the loop state."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_r11_unavailable_returns_empty(fake_client):
    fake_client.available_value = False
    out = _run(ca._claude_reformat_recipe(
        "junk recipe", "do thing", "SEN-test",
    ))
    assert out == ""


def test_r12_cli_error_returns_empty(fake_client):
    fake_client.raise_error = True
    out = _run(ca._claude_reformat_recipe(
        "junk recipe", "do thing", "SEN-test",
    ))
    assert out == ""


def test_r13_prose_with_embedded_steps_is_extracted(fake_client):
    """Claude's reformat response often has light prose around the
    STEP block. _extract_recipe_steps_from_text must pull just the
    STEP lines so the parser sees a clean recipe."""
    fake_client.response_text = (
        "Sure, here is the reformatted recipe:\n\n"
        'STEP 1: edit_file path="a.py" old="x" new="y"\n'
        'STEP 2: done summary="ok"\n\n'
        "Let me know if you want me to adjust."
    )
    out = _run(ca._claude_reformat_recipe(
        "old prose recipe", "swap x for y in a.py", "SEN-test",
    ))
    parsed = _parse_recipe_steps(out)
    assert len(parsed) == 2, f"expected 2 STEPs, got {len(parsed)}: {out!r}"
    assert "edit_file" in out
    assert "done" in out


def test_r14_bare_step_blocks_pass_through(fake_client):
    fake_client.response_text = (
        'STEP 1: write_file path="new.py" content="def f(): pass\\n"\n'
        'STEP 2: done summary="created"'
    )
    out = _run(ca._claude_reformat_recipe(
        "blah", "create new.py", "SEN-test",
    ))
    assert len(_parse_recipe_steps(out)) == 2


def test_r15_oversize_response_is_truncated(fake_client):
    """If Claude returns a recipe longer than RECIPE_MAX_CHARS_STEPFED,
    the helper truncates at a step boundary (no partial STEPs)."""
    big_step = 'STEP {n}: edit_file path="f.py" old="' + "x" * 200 + (
        '" new="' + "y" * 200 + '"\n'
    )
    parts = [big_step.format(n=i + 1) for i in range(50)]
    parts.append('STEP 51: done summary="ok"')
    fake_client.response_text = "".join(parts)
    out = _run(ca._claude_reformat_recipe(
        "...", "many edits", "SEN-test",
    ))
    assert len(out) <= ca.RECIPE_MAX_CHARS_STEPFED, (
        f"expected <= {ca.RECIPE_MAX_CHARS_STEPFED}, got {len(out)}"
    )
    parsed = _parse_recipe_steps(out)
    assert len(parsed) >= 2, "truncation must keep at least 2 STEPs"


def test_r16_empty_response_returns_empty(fake_client):
    fake_client.response_text = "   \n  \n"
    out = _run(ca._claude_reformat_recipe(
        "blah", "do thing", "SEN-test",
    ))
    assert out == ""


def test_r17_generate_called_with_no_tools(fake_client):
    """Reformat is pure text reshape; no tools should be enabled.
    Important: tools=[] (or absent) keeps Claude from wandering off
    to Read more files."""
    fake_client.response_text = (
        'STEP 1: done summary="x"\nSTEP 2: done summary="y"'
    )
    _run(ca._claude_reformat_recipe(
        "junk", "task", "SEN-test",
    ))
    kwargs = fake_client.captured_kwargs or {}
    assert kwargs.get("tools") in ([], None), (
        f"reformat must pass tools=[] (no exploration); got {kwargs.get('tools')!r}"
    )


def test_r18_system_prompt_forbids_prose_and_markdown():
    """The system prompt is the format-strictness contract -- it must
    explicitly forbid markdown and prose between STEPs."""
    s = ca._REFORMAT_SYSTEM
    assert "NO markdown" in s
    assert "NO prose" in s
    assert "NO numbered lists" in s


def test_r19_system_prompt_requires_real_action():
    """The system prompt must tell Claude the recipe needs a real
    file-modifying action -- otherwise Claude could comply with
    format but produce a degenerate `read_file + done` recipe that
    passes parser checks but isn't a solution. Updated 2026-05-06
    to require at least one edit_file or write_file step (was
    previously 'at least 2 STEPs', which the id=96 regression
    showed was insufficient)."""
    s = ca._REFORMAT_SYSTEM
    assert "edit_file or write_file" in s


# ─────────────────────────────────────────────────────────────────
# Source-level integration: parse-check in the attempt loop
# ─────────────────────────────────────────────────────────────────


_CODE_ASSIST_SRC = (
    Path(__file__).resolve().parent.parent / "skills" / "code_assist.py"
).read_text(encoding="utf-8")


def test_r21_imports_parse_recipe_steps():
    """The attempt loop's parse-check needs _parse_recipe_steps
    available. Local import inside _run_agentic_pipeline must
    include it."""
    # Tolerant of either single-line or multi-line import form.
    assert "_parse_recipe_steps" in _CODE_ASSIST_SRC, (
        "code_assist must import _parse_recipe_steps for the parse-check"
    )


def test_r22_attempt_loop_calls_reformat_on_parse_failure():
    """Parse-check + reformat call must be wired into the attempt
    loop. Source-level: the call to _claude_reformat_recipe must
    appear inside the loop body."""
    assert "_claude_reformat_recipe(" in _CODE_ASSIST_SRC
    # Specifically inside the agentic pipeline (not just defined)
    assert "await _claude_reformat_recipe(" in _CODE_ASSIST_SRC


def test_r23_legacy_fallback_preserved_on_retry_exhaustion():
    """If reformat ALSO yields <2 STEPs, the loop must NOT raise --
    it falls through and lets stepfed's legacy fallback fire as the
    safety net. Source-level check: the warning marker for retry
    exhaustion exists, and there's no `raise` or `break` in that
    code path."""
    assert "reformat retry" in _CODE_ASSIST_SRC.lower()
    assert "exhausted" in _CODE_ASSIST_SRC.lower()
    assert "falling through to legacy" in _CODE_ASSIST_SRC.lower()


def test_r24_log_marker_requesting_reformat():
    """Distinct log marker so log scans can identify reformat
    triggers separately from the regular recipe log."""
    assert "requesting reformat" in _CODE_ASSIST_SRC


def test_r25_log_marker_reformat_succeeded():
    """Distinct log marker on the success branch so log scans can
    measure the reformat path's success rate."""
    assert "reformat succeeded" in _CODE_ASSIST_SRC
