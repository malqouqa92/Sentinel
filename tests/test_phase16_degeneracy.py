"""Phase 16 hot-fix -- degenerate recipe rejection (SEN-b46a27cf).

Trace SEN-b46a27cf (pattern id=96, 2026-05-06): Claude pre-teach
returned a clarification question with no STEP blocks. Option 1's
reformat coerced it into a 2-STEP recipe (read_file + done where
the done summary was the question). Stepfed executed it, reviewer-
Claude grep'd line 17, saw existing emojis, and PASSED -- producing
a fake 1/1 graduated pattern with no actual edit.

Fix:
  - Reformat prompt now explicitly tells Claude that legitimate
    refusal beats fabricating a recipe.
  - `_recipe_has_edit_step()` helper checks if a recipe contains
    at least one edit_file or write_file step.
  - Reformat result with no edit step -> bail attempt without
    storing pattern.
  - Pre-stepfed check: even if pre-teach didn't go through reformat,
    a recipe with no edit step is rejected.
  - Recipe linter L8: same check for skip-eligibility.

Coverage:
  Helper (_recipe_has_edit_step):
    G01 -- recipe with edit_file -> True
    G02 -- recipe with write_file -> True
    G03 -- recipe with both edit_file and write_file -> True
    G04 -- recipe with read_file + done only -> False (the bug)
    G05 -- recipe with read_file + run_bash + done -> False
    G06 -- empty recipe -> False
    G07 -- malformed recipe -> False (no crash)
    G08 -- mixed: read + edit + done -> True

  Linter L8 (recipe_linter.lint_recipe_for_skip):
    G11 -- 2-step read+done recipe -> L8 fail
    G12 -- 3-step read+run_bash+done recipe -> L8 fail
    G13 -- 3-step edit+run_bash+done recipe -> L8 pass + overall safe
    G14 -- 3-step write+run_bash+done recipe -> L8 pass + overall safe
    G15 -- L8 in failed_checks list when triggered
    G16 -- safe-recipe success message mentions edit/write count

  Reformat prompt content:
    G21 -- _REFORMAT_SYSTEM tells Claude refusal is preferred
    G22 -- _REFORMAT_SYSTEM forbids edit-less recipes explicitly

  Source-level integration:
    G31 -- code_assist defines _recipe_has_edit_step
    G32 -- attempt loop calls _recipe_has_edit_step on reformat result
    G33 -- attempt loop calls _recipe_has_edit_step pre-stepfed too
    G34 -- bail messages reference 'no edit_file/write_file'
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.recipe_linter import lint_recipe_for_skip
from skills import code_assist as ca


# ─────────────────────────────────────────────────────────────────
# _recipe_has_edit_step helper
# ─────────────────────────────────────────────────────────────────


def test_g01_edit_file_recipe_returns_true():
    recipe = (
        'STEP 1: edit_file path="x.py" old="a" new="b"\n'
        'STEP 2: done summary="ok"'
    )
    assert ca._recipe_has_edit_step(recipe) is True


def test_g02_write_file_recipe_returns_true():
    recipe = (
        'STEP 1: write_file path="new.py" content="def f(): pass"\n'
        'STEP 2: done summary="ok"'
    )
    assert ca._recipe_has_edit_step(recipe) is True


def test_g03_both_edit_and_write_returns_true():
    recipe = (
        'STEP 1: edit_file path="a.py" old="x" new="y"\n'
        'STEP 2: write_file path="b.py" content="z"\n'
        'STEP 3: done summary="ok"'
    )
    assert ca._recipe_has_edit_step(recipe) is True


def test_g04_read_done_only_returns_false_THE_BUG():
    """The exact recipe shape that caused id=96."""
    recipe = (
        'STEP 1: read_file path="interfaces/telegram_bot.py"\n'
        'STEP 2: done summary="provide the emoji pair you want"'
    )
    assert ca._recipe_has_edit_step(recipe) is False


def test_g05_read_runbash_done_returns_false():
    """No edit, just verification + done -> not a solution."""
    recipe = (
        'STEP 1: read_file path="x.py"\n'
        'STEP 2: run_bash command="python -c \\"print(1)\\""\n'
        'STEP 3: done summary="checked"'
    )
    assert ca._recipe_has_edit_step(recipe) is False


def test_g06_empty_recipe_returns_false():
    assert ca._recipe_has_edit_step("") is False


def test_g07_malformed_recipe_no_crash():
    """Garbage strings must return False without raising."""
    for garbage in ["", "not a recipe", "STEP", "@@@", None]:
        try:
            r = ca._recipe_has_edit_step(garbage or "")
            assert r is False
        except Exception as e:
            pytest.fail(f"raised on {garbage!r}: {e}")


def test_g08_mixed_read_edit_done_returns_true():
    recipe = (
        'STEP 1: read_file path="x.py"\n'
        'STEP 2: edit_file path="x.py" old="a" new="b"\n'
        'STEP 3: done summary="ok"'
    )
    assert ca._recipe_has_edit_step(recipe) is True


# ─────────────────────────────────────────────────────────────────
# Recipe linter L8
# ─────────────────────────────────────────────────────────────────


def test_g11_two_step_read_done_l8_fail():
    """The id=96 shape rejected at lint time."""
    steps = [
        {"tool": "read_file", "args": {"path": "x.py"}},
        {"tool": "done", "args": {"summary": "ask user"}},
    ]
    r = lint_recipe_for_skip("STEP 1: read_file ...", parsed_steps=steps)
    assert r.safe is False
    assert "L8" in r.failed_checks


def test_g12_three_step_no_edit_l8_fail():
    """Recipe with run_bash but no edit -> L8 fail."""
    steps = [
        {"tool": "read_file", "args": {"path": "x.py"}},
        {"tool": "run_bash", "args": {"command": 'python -c "print(1)"'}},
        {"tool": "done", "args": {"summary": "verified"}},
    ]
    r = lint_recipe_for_skip("STEP ...", parsed_steps=steps)
    assert r.safe is False
    assert "L8" in r.failed_checks


def test_g13_edit_runbash_done_lint_safe():
    """Canonical valid recipe."""
    steps = [
        {"tool": "edit_file", "args": {
            "path": "core/util.py", "old": "x = 1", "new": "x = 2",
        }},
        {"tool": "run_bash", "args": {
            "command": 'python -c "import core.util"',
        }},
        {"tool": "done", "args": {"summary": "x = 2"}},
    ]
    r = lint_recipe_for_skip("STEP ...", parsed_steps=steps)
    assert r.safe is True, f"expected safe, got: {r.reason}"


def test_g14_write_runbash_done_lint_safe():
    steps = [
        {"tool": "write_file", "args": {
            "path": "core/new_mod.py", "content": "def f(): pass\n",
        }},
        {"tool": "run_bash", "args": {
            "command": 'python -c "import core.new_mod"',
        }},
        {"tool": "done", "args": {"summary": "created"}},
    ]
    r = lint_recipe_for_skip("STEP ...", parsed_steps=steps)
    assert r.safe is True


def test_g15_l8_in_failed_checks_list():
    """Verify L8 specifically appears in failed_checks (so log
    scanning can identify L8 events)."""
    steps = [
        {"tool": "read_file", "args": {"path": "x.py"}},
        {"tool": "done", "args": {"summary": "ask"}},
    ]
    r = lint_recipe_for_skip("...", parsed_steps=steps)
    assert "L8" in r.failed_checks


def test_g16_safe_message_mentions_edit_count():
    """The success reason should describe edit/write count + bash
    verify count, so logs are informative."""
    steps = [
        {"tool": "edit_file", "args": {"path": "x.py"}},
        {"tool": "run_bash", "args": {"command": "pytest"}},
        {"tool": "done", "args": {"summary": "ok"}},
    ]
    r = lint_recipe_for_skip("...", parsed_steps=steps)
    assert r.safe is True
    assert "edit/write" in r.reason or "edit_file" in r.reason \
        or "edit" in r.reason.lower()


# ─────────────────────────────────────────────────────────────────
# Reformat prompt content
# ─────────────────────────────────────────────────────────────────


def test_g21_reformat_prompt_allows_legitimate_refusal():
    s = ca._REFORMAT_SYSTEM
    # Must explicitly tell Claude refusal beats fake recipe.
    assert "refusal" in s.lower()
    assert "DO NOT INVENT" in s or "do not invent" in s.lower() \
        or "do not fabricate" in s.lower()


def test_g22_reformat_prompt_forbids_editless_recipes():
    s = ca._REFORMAT_SYSTEM
    assert "edit_file or write_file" in s
    assert "degenerate" in s.lower() or "rejected" in s.lower()


# ─────────────────────────────────────────────────────────────────
# Source-level integration
# ─────────────────────────────────────────────────────────────────


_CODE_ASSIST_SRC = (
    Path(__file__).resolve().parent.parent / "skills" / "code_assist.py"
).read_text(encoding="utf-8")


def test_g31_helper_defined_in_code_assist():
    assert "def _recipe_has_edit_step(" in _CODE_ASSIST_SRC


def test_g32_attempt_loop_uses_helper_on_reformat_result():
    """The reformat result must be checked for an edit step before
    being accepted as the new recipe."""
    # A reasonable proxy: the helper is called somewhere AFTER
    # the reformat call. We check both that it's called and that
    # one occurrence is in the same neighborhood as 'reformat'.
    assert "_recipe_has_edit_step(reformatted)" in _CODE_ASSIST_SRC


def test_g33_attempt_loop_pre_stepfed_check():
    """Even if pre-teach didn't go through reformat, the loop
    must check the recipe for an edit step before invoking stepfed."""
    # The check needs to be on `recipe` (the bound name used after
    # the reformat branch), AND it must come before
    # `run_agent_stepfed` is called.
    assert "_recipe_has_edit_step(recipe)" in _CODE_ASSIST_SRC
    # Check ordering: helper invocation appears before run_agent_stepfed
    helper_pos = _CODE_ASSIST_SRC.rfind("_recipe_has_edit_step(recipe)")
    stepfed_pos = _CODE_ASSIST_SRC.rfind("run_agent_stepfed,")
    assert 0 < helper_pos < stepfed_pos, (
        "edit-step check must precede run_agent_stepfed call"
    )


def test_g34_bail_message_references_edit_check():
    """The WARNING log when bailing should say WHY the recipe was
    rejected (no edit_file/write_file). Important for log scans."""
    assert "no edit_file/write_file" in _CODE_ASSIST_SRC
