"""Phase 17h -- size-forced DECOMPOSE on pre-teach truncation.

Trigger 2026-05-07 ~02:23Z: Claude wrote a 9077-char STEP-N recipe
for /code 'add a /qcode command...'. Parser truncated to 6160 chars
at step boundary, dropping the trailing `done` step. Stepfed ran
5 partial steps with done=False; reviewer correctly failed it.

Root cause: complexity classifier scored the prompt as tier=standard
(0.55), CODE_TIERS Tier 2 said 'STEP-N is the default', Claude
one-shotted, blew past the 8000-char cap.

Fix:
  (a) PRE_TEACH_SYSTEM gains a 'SIZE-FORCED DECOMPOSE' rule:
      pre-recipe checklist (file count + step count + new-lane?)
      that mandates DECOMPOSE regardless of tier.
  (b) `_claude_pre_teach` signals truncation via a module-level flag.
  (c) New `_claude_force_decompose(problem, prior_recipe, trace_id)`
      helper re-asks Claude with DECOMPOSE-or-bail framing.
  (d) `_run_agentic_pipeline` checks the flag after pre-teach; on
      truncation, calls force_decompose; if a valid DECOMPOSE returns,
      replaces the truncated recipe; otherwise falls back to the
      original (existing behavior).

Three groups:
  P -- prompt edits (PRE_TEACH_SYSTEM + _FORCE_DECOMPOSE_SYSTEM)
  H -- helper behavior (_claude_force_decompose)
  W -- pipeline wiring (truncation flag + retry call)
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent


# ============================================================
# Group P: prompt content
# ============================================================


def test_p01_pre_teach_has_size_forced_decompose_rule():
    """PRE_TEACH_SYSTEM must include the new mandatory-DECOMPOSE
    rule for multi-component tasks (regardless of tier)."""
    from skills.code_assist import PRE_TEACH_SYSTEM
    assert "SIZE-FORCED DECOMPOSE" in PRE_TEACH_SYSTEM


def test_p02_pre_teach_mentions_specific_thresholds():
    """The rule must give Claude concrete numbers, not 'big' or
    'small' -- vague guidance was the problem."""
    from skills.code_assist import PRE_TEACH_SYSTEM
    # >2 files, >7 STEPs, new command lane
    assert "2 files" in PRE_TEACH_SYSTEM
    assert "7 STEPs" in PRE_TEACH_SYSTEM
    assert "command lane" in PRE_TEACH_SYSTEM


def test_p03_pre_teach_explains_why_decompose_wins():
    """Should mention the 8000-char cap + silent step drop so Claude
    understands the failure mode it's avoiding."""
    from skills.code_assist import PRE_TEACH_SYSTEM
    assert "8000-char" in PRE_TEACH_SYSTEM
    assert "DROPPED" in PRE_TEACH_SYSTEM or "dropped" in PRE_TEACH_SYSTEM
    # Live failure citation gives credibility
    assert "2026-05-07" in PRE_TEACH_SYSTEM


def test_p04_force_decompose_system_exists_and_demands_decompose_only():
    from skills.code_assist import _FORCE_DECOMPOSE_SYSTEM
    assert "DECOMPOSE" in _FORCE_DECOMPOSE_SYSTEM
    assert "NO STEP-N" in _FORCE_DECOMPOSE_SYSTEM


def test_p05_force_decompose_system_explains_truncation_cause():
    """Tells Claude WHY this retry is happening, not just what to do."""
    from skills.code_assist import _FORCE_DECOMPOSE_SYSTEM
    assert "truncated" in _FORCE_DECOMPOSE_SYSTEM.lower()
    assert "8000" in _FORCE_DECOMPOSE_SYSTEM


def test_p06_force_decompose_system_shows_format_template():
    from skills.code_assist import _FORCE_DECOMPOSE_SYSTEM
    assert "DECOMPOSE\n- /code" in _FORCE_DECOMPOSE_SYSTEM


def test_p07_force_decompose_warns_against_just_repeating():
    """Anti-pattern: Claude could 'decompose' by just listing the
    same recipe steps as bullets. Must explicitly forbid."""
    from skills.code_assist import _FORCE_DECOMPOSE_SYSTEM
    # 'Do NOT just repeat your previous recipe in DECOMPOSE shape'
    assert "repeat" in _FORCE_DECOMPOSE_SYSTEM.lower()


# ============================================================
# Group H: helper signature + behavior
# ============================================================


def test_h01_force_decompose_helper_exists_and_async():
    """Must be an async coroutine (pipeline awaits it)."""
    import inspect
    from skills.code_assist import _claude_force_decompose
    assert inspect.iscoroutinefunction(_claude_force_decompose)


def test_h02_force_decompose_signature():
    """(problem, prior_recipe, trace_id) -> str"""
    import inspect
    from skills.code_assist import _claude_force_decompose
    sig = inspect.signature(_claude_force_decompose)
    assert "problem" in sig.parameters
    assert "prior_recipe" in sig.parameters
    assert "trace_id" in sig.parameters


def test_h03_force_decompose_no_tools_passed():
    """Source-level: helper should call client.generate with tools=None
    (Claude already explored the codebase in the truncated pre-teach
    -- no need to re-explore for the format conversion)."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = src.find("async def _claude_force_decompose")
    assert idx > 0
    body = src[idx:idx + 3000]
    assert "tools=None" in body


def test_h04_force_decompose_caps_prior_recipe_inclusion():
    """Prior recipe in the prompt must be capped (not raw 9000+
    chars) -- otherwise it'd blow the new pre-teach prompt budget."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = src.find("async def _claude_force_decompose")
    body = src[idx:idx + 3000]
    # prior_recipe[:N] form
    assert "prior_recipe[:" in body


def test_h05_force_decompose_unavailable_returns_empty():
    """If Claude CLI isn't installed, helper returns "" without
    crashing the pipeline."""
    import asyncio
    import unittest.mock as mock

    with mock.patch("core.claude_cli.ClaudeCliClient") as fake:
        fake.return_value.available = False
        from skills.code_assist import _claude_force_decompose
        result = asyncio.run(_claude_force_decompose(
            "anything", "prior recipe text", "SEN-test-h05",
        ))
    assert result == ""


def test_h06_force_decompose_cli_error_returns_empty():
    """If Claude CLI raises mid-call, helper returns "" not raise."""
    import asyncio
    import unittest.mock as mock
    from core.claude_cli import ClaudeCliError

    async def fake_generate(*a, **kw):
        raise ClaudeCliError("simulated")

    with mock.patch("core.claude_cli.ClaudeCliClient") as fake:
        fake.return_value.available = True
        fake.return_value.generate = fake_generate
        from skills.code_assist import _claude_force_decompose
        result = asyncio.run(_claude_force_decompose(
            "anything", "prior", "SEN-test-h06",
        ))
    assert result == ""


# ============================================================
# Group W: pipeline wiring
# ============================================================


def test_w01_truncation_flag_module_level():
    """_LAST_PRE_TEACH_TRUNCATED is module-level so test mocks of
    _claude_pre_teach (which return only str) don't break the new
    truncation-detection path."""
    from skills import code_assist
    assert hasattr(code_assist, "_LAST_PRE_TEACH_TRUNCATED")
    assert isinstance(code_assist._LAST_PRE_TEACH_TRUNCATED, bool)


def test_w02_pre_teach_resets_flag_before_each_call():
    """Source-level: _claude_pre_teach must set the flag at the
    start (False), then update to True only on truncation. Without
    reset, a previous truncation would leak into the next /code."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    pre_teach_idx = src.find("async def _claude_pre_teach(")
    assert pre_teach_idx > 0
    # Find the end of _claude_pre_teach (next def or class or end)
    next_def = src.find("\nasync def ", pre_teach_idx + 100)
    body = src[pre_teach_idx:next_def] if next_def > 0 else src[pre_teach_idx:pre_teach_idx + 5000]
    assert "_LAST_PRE_TEACH_TRUNCATED = False" in body
    assert "_LAST_PRE_TEACH_TRUNCATED = True" in body


def test_w03_pipeline_checks_flag_after_pre_teach():
    """_run_agentic_pipeline must check the flag after attempt 1's
    pre-teach call AND before the existing decomposition check."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    pipeline_idx = src.find("async def _run_agentic_pipeline")
    body = src[pipeline_idx:pipeline_idx + 30000]
    pre_teach_call = body.find("recipe = await _claude_pre_teach(")
    flag_check = body.find("_LAST_PRE_TEACH_TRUNCATED")
    decomp_check = body.find("_extract_decomposition(recipe)")
    assert pre_teach_call > 0
    assert flag_check > pre_teach_call, (
        "flag check must come AFTER pre-teach call"
    )
    assert flag_check < decomp_check, (
        "flag check must come BEFORE decomposition check (so the "
        "force-decompose recipe is what gets checked)"
    )


def test_w04_pipeline_calls_force_decompose_on_truncation():
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    pipeline_idx = src.find("async def _run_agentic_pipeline")
    body = src[pipeline_idx:pipeline_idx + 30000]
    assert "_claude_force_decompose(" in body


def test_w05_pipeline_validates_force_decompose_result():
    """Force-decompose result must be checked via _extract_decomposition
    BEFORE replacing the original recipe -- otherwise an invalid response
    (no DECOMPOSE first line, no bullets) would replace the truncated
    recipe with garbage."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    pipeline_idx = src.find("async def _run_agentic_pipeline")
    body = src[pipeline_idx:pipeline_idx + 30000]
    fd_idx = body.find("_claude_force_decompose(")
    # Within ~600 chars after the call, must verify with
    # _extract_decomposition before assigning to `recipe`.
    after = body[fd_idx:fd_idx + 1200]
    assert "_extract_decomposition(forced)" in after
    assert "recipe = forced" in after


def test_w06_pipeline_falls_back_on_force_decompose_failure():
    """If force_decompose returns "" or invalid, pipeline must
    keep the original (truncated) recipe and continue. No crash."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    pipeline_idx = src.find("async def _run_agentic_pipeline")
    body = src[pipeline_idx:pipeline_idx + 30000]
    # Look for the warning log about falling back
    assert "falling back" in body.lower()


def test_w07_pipeline_truncation_path_is_try_excepted():
    """The whole truncation-retry block must be inside try/except
    so a Claude CLI failure during force-decompose can't crash /code."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    pipeline_idx = src.find("async def _run_agentic_pipeline")
    body = src[pipeline_idx:pipeline_idx + 30000]
    flag_idx = body.find("if _LAST_PRE_TEACH_TRUNCATED:")
    # Within ~150 chars BEFORE the flag check, there must be a try:
    window_before = body[max(0, flag_idx - 200):flag_idx]
    assert "try:" in window_before


# ============================================================
# Group I: import + cross-batch sanity
# ============================================================


def test_i01_imports_clean():
    from skills import code_assist  # noqa
    # Specifically the new symbols
    from skills.code_assist import (  # noqa
        _LAST_PRE_TEACH_TRUNCATED,
        _claude_force_decompose,
        _FORCE_DECOMPOSE_SYSTEM,
    )


def test_i02_pre_teach_signature_unchanged():
    """Backwards-compat: _claude_pre_teach still returns a str.
    The truncation flag is signaled via module-level state, NOT
    via signature change. This protects existing test mocks."""
    import inspect
    from skills.code_assist import _claude_pre_teach
    sig = inspect.signature(_claude_pre_teach)
    # Signature unchanged from prior phases
    assert "problem" in sig.parameters
    assert "trace_id" in sig.parameters
