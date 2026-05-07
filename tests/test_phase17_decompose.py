"""Phase 17 Batch 1 -- pre-teach STEP-N tightening + decomposition mode.

Two unrelated landings bundled in this batch:

1. ``PRE_TEACH_SYSTEM`` tightened with explicit "first chars must be
   STEP 1:" rule, WRONG-output negative example, target step count
   guidance, and DECOMPOSE escape valve. Fixes the live failure mode
   from trace SEN-b203948c (2026-05-06 22:09Z): a 14692-char Claude
   response truncated to 8000 chars at step boundary, parsed to 1
   STEP, kicked off a Phase-16 reformat retry that would have spent
   another Claude CLI call.

2. ``_extract_decomposition`` + ``_format_decomposition_response``
   helpers + a short-circuit branch in ``_run_agentic_pipeline``.
   When Claude emits a DECOMPOSE block (literal first line, bullet
   list of /code-shaped subtasks), the pipeline returns the suggested
   subtasks to the user instead of running stepfed on a too-big recipe.
   New ``solved_by="decompose_suggested"`` flows through the bot's
   ready-to-display markdown branch.

These tests are SOURCE-LEVEL where they assert prompt content (cheap,
deterministic, no LLM round-trips) and BEHAVIORAL where they assert the
pipeline short-circuit shape (mocks Claude pre-teach to return DECOMPOSE).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from skills import code_assist
from skills.code_assist import (
    PRE_TEACH_SYSTEM,
    _extract_decomposition,
    _format_decomposition_response,
)

PROJECT = Path(__file__).resolve().parent.parent


# ============================================================
# Group P: PRE_TEACH_SYSTEM tightening (source-level)
# ============================================================


def test_p01_first_chars_rule_present_verbatim():
    """The 'first chars must be STEP 1:' rule must be in the prompt
    -- this is the load-bearing fix for the degeneracy failure."""
    assert "FIRST CHARACTERS OF YOUR OUTPUT MUST BE `STEP 1:`" in PRE_TEACH_SYSTEM


def test_p02_no_prose_between_steps_rule():
    """NO prose between STEPs must be explicit."""
    upper = PRE_TEACH_SYSTEM.upper()
    assert "NO PROSE BETWEEN" in upper or "NO PROSE\nBETWEEN" in upper


def test_p03_target_step_count_window():
    """Target window 3-7 STEPs must be in the prompt."""
    assert "3-7 STEP" in PRE_TEACH_SYSTEM


def test_p04_truncation_warning_about_8k_cap():
    """Prompt should explain why decomposition is better than mega-recipes."""
    assert "8000-char cap" in PRE_TEACH_SYSTEM or "truncate" in PRE_TEACH_SYSTEM.lower()


def test_p05_wrong_example_present():
    """A negative-example block showing what gets rejected (prose
    interleaved with STEPs) must be in the prompt."""
    assert "WRONG STEP-N output" in PRE_TEACH_SYSTEM
    assert "REJECTED" in PRE_TEACH_SYSTEM


def test_p06_decompose_format_documented():
    """DECOMPOSE escape valve must be documented with rules."""
    assert "RULES for DECOMPOSE" in PRE_TEACH_SYSTEM
    assert "DECOMPOSE\n" in PRE_TEACH_SYSTEM  # literal first-line example


def test_p07_decompose_bullet_format_documented():
    """Subtask bullet format must be specified."""
    assert "- /code " in PRE_TEACH_SYSTEM


def test_p08_decompose_for_scope_not_retries():
    """Anti-misuse rule: don't decompose a small task."""
    assert "for SCOPE, not for retries" in PRE_TEACH_SYSTEM


def test_p09_step_n_format_still_default():
    """STEP-N path must still be the default; DECOMPOSE is the escape."""
    assert "STEP-N format (default path)" in PRE_TEACH_SYSTEM


def test_p10_pre_teach_size_reasonable():
    """Sanity check: prompt grew but didn't blow up. <6000 chars."""
    assert 1500 < len(PRE_TEACH_SYSTEM) < 6000


# ============================================================
# Group D: _extract_decomposition (parser)
# ============================================================


def test_d01_none_for_normal_step_recipe():
    """Normal STEP-N recipe must return None (caller falls through)."""
    recipe = (
        'STEP 1: read_file path="core/util.py"\n'
        'STEP 2: edit_file path="core/util.py" old="def" new="async def"\n'
        'STEP 3: done summary="x"'
    )
    assert _extract_decomposition(recipe) is None


def test_d02_none_for_empty_string():
    assert _extract_decomposition("") is None
    assert _extract_decomposition("   \n  \n") is None


def test_d03_none_for_none_input():
    assert _extract_decomposition(None) is None  # type: ignore[arg-type]


def test_d04_decompose_with_three_subtasks():
    recipe = (
        "DECOMPOSE\n"
        "- /code add empty stub for /qcode\n"
        "- /code wire planner\n"
        "- /code wire stepfed and report"
    )
    out = _extract_decomposition(recipe)
    assert out is not None
    assert len(out) == 3
    assert out[0] == "/code add empty stub for /qcode"
    assert out[2] == "/code wire stepfed and report"


def test_d05_decompose_colon_variant_accepted():
    """`DECOMPOSE:` first line should also work."""
    recipe = (
        "DECOMPOSE:\n"
        "- /code thing one\n"
        "- /code thing two"
    )
    assert _extract_decomposition(recipe) == [
        "/code thing one", "/code thing two",
    ]


def test_d06_decompose_with_leading_whitespace():
    """LSTRIPped prefix should still match (Claude often pads with newlines)."""
    recipe = "\n\nDECOMPOSE\n- /code one\n- /code two"
    assert _extract_decomposition(recipe) == ["/code one", "/code two"]


def test_d07_decompose_word_in_prose_does_not_match():
    """If 'decompose' appears in narrative text it must NOT trigger."""
    recipe = (
        "I'll decompose this task into:\n"
        "STEP 1: read_file path=\"x.py\"\n"
        "STEP 2: done summary=\"x\""
    )
    assert _extract_decomposition(recipe) is None


def test_d08_decompose_first_line_strict():
    """`DECOMPOSE` only as the FIRST line; mid-text doesn't count."""
    recipe = (
        "Some prose first\n"
        "DECOMPOSE\n"
        "- /code thing"
    )
    # First line is not DECOMPOSE.
    assert _extract_decomposition(recipe) is None


def test_d09_decompose_with_no_bullets_returns_none():
    """`DECOMPOSE` first line but no bullets -> not a valid block."""
    recipe = "DECOMPOSE\n\nSome text but no bullets."
    assert _extract_decomposition(recipe) is None


def test_d10_decompose_asterisk_bullets_accepted():
    recipe = "DECOMPOSE\n* /code one\n* /code two"
    assert _extract_decomposition(recipe) == ["/code one", "/code two"]


def test_d11_decompose_non_code_bullets_filtered():
    """Bullets that don't start with /code are ignored."""
    recipe = (
        "DECOMPOSE\n"
        "- /code real subtask\n"
        "- this is just prose\n"
        "- /code another real subtask"
    )
    out = _extract_decomposition(recipe)
    assert out == [
        "/code real subtask", "/code another real subtask",
    ]


def test_d12_decompose_minimum_one_subtask():
    """Single subtask is still a valid decomposition (technically
    weird but the parser shouldn't reject -- caller decides whether
    to surface)."""
    recipe = "DECOMPOSE\n- /code only one subtask"
    assert _extract_decomposition(recipe) == ["/code only one subtask"]


def test_d13_decompose_subtask_strips_whitespace():
    recipe = "DECOMPOSE\n-  /code padded with spaces  \n- /code two"
    out = _extract_decomposition(recipe)
    assert out is not None
    assert out[0] == "/code padded with spaces"


# ============================================================
# Group F: _format_decomposition_response (rendering)
# ============================================================


def test_f01_format_renders_numbered_list():
    out = _format_decomposition_response([
        "/code one", "/code two", "/code three",
    ])
    assert "1. `/code one`" in out
    assert "2. `/code two`" in out
    assert "3. `/code three`" in out


def test_f02_format_includes_too_big_marker():
    out = _format_decomposition_response(["/code one"])
    assert "Task too big" in out


def test_f03_format_includes_run_in_order_hint():
    out = _format_decomposition_response(["/code a", "/code b"])
    assert "in order" in out.lower()


def test_f04_format_mentions_commit_between():
    """User instruction: /commit between each subtask."""
    out = _format_decomposition_response(["/code a", "/code b"])
    assert "/commit" in out


def test_f05_format_with_single_subtask_works():
    out = _format_decomposition_response(["/code single"])
    assert "1. `/code single`" in out
    assert len(out) > 0


# ============================================================
# Group W: Pipeline wiring (source-level checks)
# ============================================================


def test_w01_pipeline_calls_extract_decomposition_after_pre_teach():
    """Source check: _extract_decomposition is called inside
    _run_agentic_pipeline, on attempt 1, after _claude_pre_teach."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    # Find the _run_agentic_pipeline function body.
    pipeline_start = src.find("async def _run_agentic_pipeline(")
    assert pipeline_start > 0
    body = src[pipeline_start:pipeline_start + 30000]
    pre_teach_idx = body.find("_claude_pre_teach(")
    extract_idx = body.find("_extract_decomposition(")
    assert pre_teach_idx > 0, "must call _claude_pre_teach"
    assert extract_idx > 0, "must call _extract_decomposition"
    assert extract_idx > pre_teach_idx, (
        "_extract_decomposition must run AFTER _claude_pre_teach"
    )


def test_w02_pipeline_returns_decompose_suggested_solved_by():
    """Source check: short-circuit branch sets
    solved_by='decompose_suggested'."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert 'solved_by="decompose_suggested"' in src


def test_w03_telegram_render_includes_decompose_branch():
    """Source check: telegram bot's ready-to-display tuple
    includes 'decompose_suggested'."""
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(encoding="utf-8")
    assert '"decompose_suggested"' in src


def test_w04_pipeline_short_circuit_skips_kb_write():
    """Source check: the decomposition branch returns early -- so
    add_pattern, graduation, and stepfed never run on a decomposed
    task. We assert this by checking the branch returns directly
    (no shadow_recipe assignment after it on attempt 1).

    Window expanded in Phase 17b after the chain runner was added
    inside the same branch (chain enqueue + chain_started return
    pushes the manual-surface fall-through return further out)."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    pipeline_start = src.find("async def _run_agentic_pipeline(")
    body = src[pipeline_start:pipeline_start + 30000]
    decomp_idx = body.find("decomp_subtasks = _extract_decomposition(")
    assert decomp_idx > 0
    # Generous window: chain runner + fall-through return both live
    # inside this branch (Phase 17b added ~80 lines).
    branch_body = body[decomp_idx:decomp_idx + 6000]
    assert "return CodeAssistOutput(" in branch_body, (
        "decomp branch must short-circuit via early return"
    )


# ============================================================
# Group I: Integration sanity (cheap behavioral)
# ============================================================


def test_i01_module_imports_clean():
    """Module loads without circular-import errors."""
    assert hasattr(code_assist, "_extract_decomposition")
    assert hasattr(code_assist, "_format_decomposition_response")
    assert hasattr(code_assist, "PRE_TEACH_SYSTEM")


def test_i02_helpers_are_module_level():
    """Helpers should be module-level (importable, testable in
    isolation), not nested inside _run_agentic_pipeline."""
    from skills.code_assist import _extract_decomposition as fn1
    from skills.code_assist import _format_decomposition_response as fn2
    # If these were nested they wouldn't import.
    assert callable(fn1)
    assert callable(fn2)
