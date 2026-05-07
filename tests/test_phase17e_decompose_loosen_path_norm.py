"""Phase 17e -- loosen DECOMPOSE matcher + normalize path separators.

Two bugs surfaced live 2026-05-07 ~01:14-01:15Z during the
post-17d chain re-run:

  1. ``_extract_decomposition`` first-line match was too strict
     (required exactly ``DECOMPOSE`` or ``DECOMPOSE:``). Claude's
     2nd run emitted a slightly different format (likely with
     markdown decoration), got rejected, fell through to STEP-N
     parser, parser found 0 STEPs, triggered Phase 16 reformat
     retry. Chain runner never fired.

  2. Recipe parser ate ``path="interfaces\telegram_bot.py"`` as
     a Python string literal -> ``\t`` decoded to literal TAB
     character -> path became ``interfaces<TAB>elegram_bot.py``
     -> 4 of 11 STEPs failed with "file not found". Ironic since
     the PRE_TEACH_SYSTEM tells Claude to use POSIX paths.

Two groups:
  D -- _extract_decomposition matcher acceptance/rejection
  P -- _safe_resolve path normalization
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent


# ============================================================
# Group D: matcher acceptance + rejection
# ============================================================


@pytest.mark.parametrize("first_line", [
    "DECOMPOSE",
    "DECOMPOSE:",
    "**DECOMPOSE**",
    "**DECOMPOSE**:",
    "# DECOMPOSE",
    "## DECOMPOSE",
    "### DECOMPOSE",
    "### DECOMPOSE:",
    "- DECOMPOSE",
    "* DECOMPOSE",
    "decompose",
    "decompose:",
    "Decompose:",
    "DeCoMpOsE",
])
def test_d01_accepted_first_line_variants(first_line):
    from skills.code_assist import _extract_decomposition
    recipe = f"{first_line}\n- /code one\n- /code two"
    out = _extract_decomposition(recipe)
    assert out is not None, f"first line {first_line!r} should be accepted"
    assert out == ["/code one", "/code two"]


@pytest.mark.parametrize("first_line", [
    "STEP 1: read_file path=\"x.py\"",  # STEP-N recipe, not DECOMPOSE
    "DECOMPOSED",  # word containing DECOMPOSE -- \b boundary rejects
    "DECOMPOSITION",  # likewise
    "Plan:",  # no DECOMPOSE word
])
def test_d02_rejected_first_line_variants(first_line):
    """Phase 17e v2: matcher loosened to accept ANY first line
    containing the word 'DECOMPOSE' (case-insensitive, word boundary).
    The bullet list (>=1 '- /code ...' line) is the real gating
    signal -- bulletless prose never reaches the chain runner.

    These cases must STILL reject because:
    - STEP-N recipes start with 'STEP N:'
    - 'decomposed' / 'decomposition' don't match \\bDECOMPOSE\\b
    - Plain prose without the word entirely
    """
    from skills.code_assist import _extract_decomposition
    recipe = f"{first_line}\n- /code one\n- /code two"
    out = _extract_decomposition(recipe)
    assert out is None, f"first line {first_line!r} should be rejected"


@pytest.mark.parametrize("first_line", [
    "I will DECOMPOSE this task into steps",
    "Here's the plan: DECOMPOSE",
    "DECOMPOSE THIS",
    "DECOMPOSE NOW:",
    "Looking at this, I should DECOMPOSE the task",
])
def test_d02b_loosened_now_accepts_narrative_with_bullets(first_line):
    """Phase 17e v2: prose-formatted DECOMPOSE intent + bullet list
    is now ACCEPTED. Previously rejected for false-positive paranoia,
    but live experience showed Claude legitimately writes these and
    means decomposition. The /code bullet requirement is the safety
    net -- if no bullets, returns None regardless of first line."""
    from skills.code_assist import _extract_decomposition
    recipe = f"{first_line}\n- /code one\n- /code two"
    out = _extract_decomposition(recipe)
    assert out == ["/code one", "/code two"], (
        f"first line {first_line!r} + bullets should be accepted"
    )


def test_d02c_narrative_word_without_bullets_still_rejected():
    """Critical: narrative mentioning 'decompose' but with NO
    /code bullets must still reject. The bullet requirement is
    what prevents matcher false-positives on anything Claude says."""
    from skills.code_assist import _extract_decomposition
    cases = [
        "I will DECOMPOSE this task into steps\n\nLet me think...",
        "DECOMPOSE THIS\n\nOn second thought, here's a STEP-N recipe:",
        "Here's the plan: DECOMPOSE\nI changed my mind, doing it inline",
    ]
    for recipe in cases:
        assert _extract_decomposition(recipe) is None, (
            f"narrative without /code bullets must reject: {recipe[:50]!r}"
        )


def test_d03_no_bullets_returns_none_even_if_first_line_matches():
    from skills.code_assist import _extract_decomposition
    recipe = "**DECOMPOSE**\n\nNo bullets here, just narrative text."
    assert _extract_decomposition(recipe) is None


def test_d04_first_line_must_be_first_nonblank():
    """If DECOMPOSE appears mid-text, doesn't count."""
    from skills.code_assist import _extract_decomposition
    recipe = "Some prose first.\nDECOMPOSE\n- /code x"
    assert _extract_decomposition(recipe) is None


def test_d05_leading_blank_lines_tolerated():
    """LSTRIPed before matching -- Claude often pads with newlines."""
    from skills.code_assist import _extract_decomposition
    recipe = "\n\n\n## DECOMPOSE\n- /code one\n- /code two"
    out = _extract_decomposition(recipe)
    assert out == ["/code one", "/code two"]


def test_d06_normal_step_n_still_returns_none():
    """Critical regression-guard: the loosened matcher must NOT
    accept a regular STEP-N recipe as a decomposition."""
    from skills.code_assist import _extract_decomposition
    recipe = (
        'STEP 1: read_file path="x.py"\n'
        'STEP 2: edit_file path="x.py" old="a" new="b"\n'
        'STEP 3: done summary="ok"'
    )
    assert _extract_decomposition(recipe) is None


def test_d07_empty_input():
    from skills.code_assist import _extract_decomposition
    assert _extract_decomposition("") is None
    assert _extract_decomposition("   \n\n  ") is None
    assert _extract_decomposition(None) is None  # type: ignore[arg-type]


# ============================================================
# Group P: _safe_resolve path normalization
# ============================================================


def test_p01_backslash_normalized(tmp_path, monkeypatch):
    """The headline fix: path with backslash should NOT decode \t
    as TAB. The recipe parser hands us 'interfaces\\telegram_bot.py'
    (already escape-decoded to 'interfaces<TAB>elegram_bot.py'
    by the time it reaches our tool); _safe_resolve must
    normalize the SLASH form so the file is found."""
    # Set up a real project structure under tmp_path.
    monkeypatch.setattr("core.qwen_agent.PROJECT_ROOT", tmp_path)
    (tmp_path / "interfaces").mkdir()
    (tmp_path / "interfaces" / "telegram_bot.py").write_text(
        "ok", encoding="utf-8",
    )
    from core.qwen_agent import _safe_resolve
    # Backslash form (POSIX-style not used here):
    target = _safe_resolve("interfaces\\telegram_bot.py")
    assert target.exists()
    assert target.name == "telegram_bot.py"


def test_p02_forward_slash_unaffected(tmp_path, monkeypatch):
    """Posix paths must still work (most common, was already correct)."""
    monkeypatch.setattr("core.qwen_agent.PROJECT_ROOT", tmp_path)
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "config.py").write_text("ok", encoding="utf-8")
    from core.qwen_agent import _safe_resolve
    target = _safe_resolve("core/config.py")
    assert target.exists()


def test_p03_mixed_separators(tmp_path, monkeypatch):
    monkeypatch.setattr("core.qwen_agent.PROJECT_ROOT", tmp_path)
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c.py").write_text("ok", encoding="utf-8")
    from core.qwen_agent import _safe_resolve
    # Mixed: a\b/c.py
    target = _safe_resolve("a\\b/c.py")
    assert target.exists()


def test_p04_escapes_sandbox_still_blocked(tmp_path, monkeypatch):
    """Normalization must NOT weaken sandbox protection. Path
    escape via .. should still raise ValueError."""
    monkeypatch.setattr("core.qwen_agent.PROJECT_ROOT", tmp_path)
    from core.qwen_agent import _safe_resolve
    with pytest.raises(ValueError):
        _safe_resolve("..\\..\\etc\\passwd")
    with pytest.raises(ValueError):
        _safe_resolve("../../etc/passwd")


def test_p05_tool_read_file_handles_backslash(tmp_path, monkeypatch):
    """Integration: tool_read_file gets a backslash path, normalizes
    via _safe_resolve, finds the file."""
    monkeypatch.setattr("core.qwen_agent.PROJECT_ROOT", tmp_path)
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "foo.py").write_text("hello", encoding="utf-8")
    from core.qwen_agent import tool_read_file
    result = tool_read_file("skills\\foo.py")
    assert result.get("ok") is True
    assert result.get("content") == "hello"


def test_p06_tool_edit_file_handles_backslash(tmp_path, monkeypatch):
    """Integration: tool_edit_file with backslash path successfully
    edits. This is the EXACT failure mode from 2026-05-07 ~01:15Z."""
    monkeypatch.setattr("core.qwen_agent.PROJECT_ROOT", tmp_path)
    (tmp_path / "interfaces").mkdir()
    target = tmp_path / "interfaces" / "telegram_bot.py"
    target.write_text("OLD\n", encoding="utf-8")
    from core.qwen_agent import tool_edit_file
    # The backslash path that failed live:
    result = tool_edit_file("interfaces\\telegram_bot.py", "OLD", "NEW")
    assert result.get("ok") is True, (
        f"backslash path edit must succeed; got {result}"
    )
    assert target.read_text(encoding="utf-8") == "NEW\n"


def test_p07_tool_write_file_handles_backslash(tmp_path, monkeypatch):
    monkeypatch.setattr("core.qwen_agent.PROJECT_ROOT", tmp_path)
    (tmp_path / "skills").mkdir()
    from core.qwen_agent import tool_write_file
    result = tool_write_file("skills\\new.py", "x = 1\n")
    assert result.get("ok") is True
    assert (tmp_path / "skills" / "new.py").exists()
