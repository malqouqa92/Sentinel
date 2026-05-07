"""Phase 16 Batch C drift-detect (bail-no-penalty) ECC."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.skip_drift import detect_recipe_drift


def test_d01_empty_recipe(tmp_path):
    drift, reason = detect_recipe_drift("", tmp_path)
    assert drift is False
    assert reason == ""


def test_d02_no_edit_file_steps_means_no_drift(tmp_path):
    """write_file recipes are idempotent; drift cannot apply."""
    recipe = (
        'STEP 1: write_file path="x.py" content="hello"\n'
        'STEP 2: run_bash command="echo ok"\n'
        'STEP 3: done summary="x"'
    )
    drift, reason = detect_recipe_drift(recipe, tmp_path)
    assert drift is False
    assert reason == ""


def test_d10_edit_file_old_substring_present_returns_no_drift(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("alpha bravo charlie", encoding="utf-8")
    recipe = (
        'STEP 1: edit_file path="a.py" old="bravo" new="zulu"\n'
        'STEP 2: done summary="x"'
    )
    drift, reason = detect_recipe_drift(recipe, tmp_path)
    assert drift is False


def test_d11_old_not_in_file_returns_drift(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("alpha NOT_BRAVO charlie", encoding="utf-8")
    recipe = (
        'STEP 1: edit_file path="a.py" old="bravo" new="zulu"\n'
        'STEP 2: done summary="x"'
    )
    drift, reason = detect_recipe_drift(recipe, tmp_path)
    assert drift is True
    assert reason.startswith("old_not_found:")
    assert "a.py" in reason


def test_d12_target_missing_returns_drift(tmp_path):
    """Recipe references a file that doesn't exist -- skip-replay
    would create it via the recipe IF it's a write_file (no drift),
    but for edit_file with non-existent target, that's drift."""
    recipe = (
        'STEP 1: edit_file path="missing.py" old="x" new="y"\n'
        'STEP 2: done summary="x"'
    )
    drift, reason = detect_recipe_drift(recipe, tmp_path)
    assert drift is True
    assert reason.startswith("target_missing:")


def test_d20_unescapes_newline_in_old(tmp_path):
    """Recipe stores `\\n` in old= literal; on disk file has actual
    newline; drift-detect must unescape to compare correctly."""
    target = tmp_path / "a.py"
    target.write_text("def foo():\n    return 1\n", encoding="utf-8")
    recipe = (
        'STEP 1: edit_file path="a.py" '
        'old="def foo():\\n    return 1" new="def foo():\\n    return 42"\n'
        'STEP 2: done summary="x"'
    )
    drift, reason = detect_recipe_drift(recipe, tmp_path)
    assert drift is False, f"expected no drift, got {reason}"


def test_d21_unescapes_quote_in_old(tmp_path):
    target = tmp_path / "a.py"
    target.write_text('return "hello"', encoding="utf-8")
    recipe = (
        'STEP 1: edit_file path="a.py" '
        'old="return \\"hello\\"" new="return \\"world\\""\n'
        'STEP 2: done summary="x"'
    )
    drift, reason = detect_recipe_drift(recipe, tmp_path)
    assert drift is False, f"expected no drift, got {reason}"


def test_d30_emoji_bar_state_drift_scenario(tmp_path):
    """The headline scenario: emoji-bar recipe stored from previous
    run says old="⚡/⬛" but file is now 🟦/⬜ from a more recent
    successful run. drift-detect catches this."""
    target = tmp_path / "interfaces"
    target.mkdir()
    bot_file = target / "telegram_bot.py"
    bot_file.write_text(
        '    return "🟦" * f + "⬜" * (w - f) + " " + str(pct) + "%"\n',
        encoding="utf-8",
    )
    recipe = (
        'STEP 1: edit_file path="interfaces/telegram_bot.py" '
        'old="    return \\"⚡\\" * f + \\"⬛\\" * (w - f)" '
        'new="    return \\"\U0001f7e9\\" * f + \\"⬛\\" * (w - f)"\n'
        'STEP 2: done summary="x"'
    )
    drift, reason = detect_recipe_drift(recipe, tmp_path)
    assert drift is True
    assert reason.startswith("old_not_found:")


def test_d40_mixed_steps_only_edit_file_checked(tmp_path):
    """Recipe has read_file + edit_file + run_bash. Only edit_file
    is checked for drift."""
    target = tmp_path / "x.py"
    target.write_text("present_substring", encoding="utf-8")
    recipe = (
        'STEP 1: read_file path="x.py"\n'
        'STEP 2: edit_file path="x.py" old="present_substring" '
        'new="new_content"\n'
        'STEP 3: run_bash command="python -c \\"pass\\""\n'
        'STEP 4: done summary="x"'
    )
    drift, _r = detect_recipe_drift(recipe, tmp_path)
    assert drift is False


def test_d41_all_edit_steps_must_pass(tmp_path):
    """Two edit_file steps; first matches, second doesn't. drift."""
    target = tmp_path / "x.py"
    target.write_text("first_token middle", encoding="utf-8")
    recipe = (
        'STEP 1: edit_file path="x.py" old="first_token" new="zz"\n'
        'STEP 2: edit_file path="x.py" old="MISSING_TOKEN" new="zz"\n'
        'STEP 3: done summary="x"'
    )
    drift, reason = detect_recipe_drift(recipe, tmp_path)
    assert drift is True
    assert "x.py" in reason
