"""Phase 17c -- CODE_TIERS.md playbook injection into Claude pre-teach.

Three groups:
  T -- CODE_TIERS.md content + structure (file exists, has all 4
       tiers, mentions DECOMPOSE on tier-3)
  L -- _load_code_tiers_memo + _classify_complexity_tier helpers
  W -- wiring source-checks (file_guard list, config caps, pre-teach
       loads + appends, complexity wired)

Why a Claude-side playbook (vs QWENCODER.md which is Qwen-side):
  Claude is the pre-teach author. When Claude judges "is this task
  too big for one recipe?", it needs an explicit complexity rubric.
  Phase 17 Batch 1's DECOMPOSE escape works only when Claude USES it.
  Tonight's failure: Claude was given the option but chose to one-shot
  a multi-component task anyway. The tier playbook adds deterministic
  nudges + a complexity verdict from `core.complexity.classify_complexity`
  so Claude has structured guidance instead of pure judgment.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent


# ============================================================
# Group T: CODE_TIERS.md content
# ============================================================


def test_t01_file_exists():
    p = PROJECT / "workspace" / "persona" / "CODE_TIERS.md"
    assert p.exists(), "CODE_TIERS.md must live in workspace/persona/"


def test_t02_file_size_reasonable():
    """File should be substantive but stay under the persona cap
    so Claude's pre-teach budget is preserved."""
    from core import config
    p = PROJECT / "workspace" / "persona" / "CODE_TIERS.md"
    text = p.read_text(encoding="utf-8")
    cap = config.PERSONA_INJECT_MAX_CHARS["CODE_TIERS.md"]
    assert 500 < len(text) < cap


def test_t03_all_four_tiers_present():
    p = PROJECT / "workspace" / "persona" / "CODE_TIERS.md"
    text = p.read_text(encoding="utf-8")
    assert "## Tier 1" in text
    assert "## Tier 2" in text
    assert "## Tier 3" in text
    assert "## Tier 4" in text


def test_t04_tier_3_mandates_decompose():
    """The load-bearing rule for Phase 17c: tier-3 tasks MUST emit
    DECOMPOSE, not one-shot STEP-N."""
    p = PROJECT / "workspace" / "persona" / "CODE_TIERS.md"
    text = p.read_text(encoding="utf-8")
    # Find the Tier 3 section.
    idx = text.find("## Tier 3")
    assert idx > 0
    end_idx = text.find("## Tier 4", idx)
    assert end_idx > idx
    section = text[idx:end_idx]
    assert "DECOMPOSE" in section
    assert "DO NOT one-shot" in section or "do not one-shot" in section.lower()


def test_t05_anti_patterns_documented():
    """Each tier should call out an anti-pattern so Claude knows
    what NOT to do, not just what TO do."""
    p = PROJECT / "workspace" / "persona" / "CODE_TIERS.md"
    text = p.read_text(encoding="utf-8")
    assert text.count("**Anti-pattern:**") >= 3


def test_t06_includes_decompose_example():
    """A worked example helps Claude pattern-match. Allow either
    flush-left or indented (inside a fenced code block)."""
    p = PROJECT / "workspace" / "persona" / "CODE_TIERS.md"
    text = p.read_text(encoding="utf-8")
    # Strip leading whitespace per line, then check for the example.
    normalized = "\n".join(line.strip() for line in text.splitlines())
    assert "DECOMPOSE\n- /code" in normalized


def test_t07_qcode_example_mentioned():
    """Tonight's exact failure scenario should be in the example so
    Claude recognizes the pattern next time."""
    p = PROJECT / "workspace" / "persona" / "CODE_TIERS.md"
    text = p.read_text(encoding="utf-8")
    assert "/qcode" in text


# ============================================================
# Group L: helpers behavior
# ============================================================


def test_l01_load_code_tiers_memo_returns_text():
    from skills.code_assist import _load_code_tiers_memo
    text = _load_code_tiers_memo()
    assert isinstance(text, str)
    assert "Tier 1" in text


def test_l02_load_capped_to_persona_max(monkeypatch, tmp_path):
    """Returns text capped to PERSONA_INJECT_MAX_CHARS['CODE_TIERS.md'].
    Patch in a long file, verify cap."""
    from core import config
    fake_persona = tmp_path / "persona"
    fake_persona.mkdir()
    big = "X" * 10000
    (fake_persona / "CODE_TIERS.md").write_text(big, encoding="utf-8")
    monkeypatch.setattr(config, "PERSONA_DIR", fake_persona)
    from skills.code_assist import _load_code_tiers_memo
    text = _load_code_tiers_memo()
    cap = config.PERSONA_INJECT_MAX_CHARS["CODE_TIERS.md"]
    assert len(text) == cap


def test_l03_load_returns_empty_when_missing(monkeypatch, tmp_path):
    from core import config
    empty = tmp_path / "no_persona_here"
    empty.mkdir()
    monkeypatch.setattr(config, "PERSONA_DIR", empty)
    from skills.code_assist import _load_code_tiers_memo
    assert _load_code_tiers_memo() == ""


def test_l04_load_swallows_io_errors(monkeypatch, tmp_path):
    """If something exotic happens (encoding error etc.) the loader
    must return "" not raise. Covered by the broad try/except."""
    from core import config
    bad = tmp_path / "bad_persona"
    bad.mkdir()
    # Simulate unreadable: monkeypatch read_text to raise.
    from skills import code_assist
    target = config.PERSONA_DIR / "CODE_TIERS.md"
    if not target.exists():
        # Make sure the path *does* exist so the broad except path
        # is the one actually exercised (not the not-exists return).
        target.write_text("x", encoding="utf-8")
    import pathlib
    orig_read = pathlib.Path.read_text

    def boom(self, *args, **kwargs):
        raise OSError("simulated")

    monkeypatch.setattr(pathlib.Path, "read_text", boom)
    try:
        result = code_assist._load_code_tiers_memo()
        assert result == ""
    finally:
        monkeypatch.setattr(pathlib.Path, "read_text", orig_read)


def test_l05_classify_complexity_tier_returns_valid_tier():
    from skills.code_assist import _classify_complexity_tier
    tier, score = _classify_complexity_tier("add a constant to config.py")
    assert tier in ("basic", "standard", "advanced")
    assert 0.0 <= score <= 1.0


def test_l06_classify_complexity_tier_handles_empty(monkeypatch):
    """Empty input shouldn't crash."""
    from skills.code_assist import _classify_complexity_tier
    tier, score = _classify_complexity_tier("")
    assert tier in ("basic", "standard", "advanced")


def test_l07_classify_falls_back_on_exception(monkeypatch):
    """If complexity classifier raises, returns ('standard', 0.5)."""
    import core.complexity as cc
    monkeypatch.setattr(
        cc, "classify_complexity",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    from skills.code_assist import _classify_complexity_tier
    tier, score = _classify_complexity_tier("anything")
    assert tier == "standard"
    assert score == 0.5


# ============================================================
# Group W: wiring source-checks
# ============================================================


def test_w01_code_tiers_in_protected_files():
    from core import config
    assert "CODE_TIERS.md" in config.PROTECTED_FILES


def test_w02_code_tiers_has_persona_inject_cap():
    from core import config
    assert "CODE_TIERS.md" in config.PERSONA_INJECT_MAX_CHARS
    assert config.PERSONA_INJECT_MAX_CHARS["CODE_TIERS.md"] >= 1500


def test_w03_pre_teach_loads_code_tiers_memo():
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert "_load_code_tiers_memo()" in src


def test_w04_pre_teach_runs_complexity_classifier():
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert "_classify_complexity_tier(" in src


def test_w05_pre_teach_appends_memo_after_pre_teach_system():
    """Memo must be APPENDED to PRE_TEACH_SYSTEM, not prepended,
    so the strict format rules read first."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = src.find("CODE_TIERS playbook")
    assert idx > 0
    # The composition must include PRE_TEACH_SYSTEM before the memo.
    window = src[max(0, idx - 600):idx]
    assert "PRE_TEACH_SYSTEM" in window


def test_w06_pre_teach_logs_tier_decision():
    """Log line should surface tier + score for telemetry / debugging."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = src.find("claude pre-teach starting")
    assert idx > 0
    nearby = src[idx:idx + 600]
    assert "tier=" in nearby
    assert "score=" in nearby


def test_w07_complexity_verdict_block_appended():
    """The 'complexity verdict' label tells Claude THIS specific
    task's tier, not just the playbook abstract rules."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert "complexity verdict" in src


def test_w08_pre_teach_falls_through_when_memo_missing():
    """If CODE_TIERS.md is absent, pre-teach uses bare PRE_TEACH_SYSTEM
    -- no crash, no empty injection block."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    # The fall-through pattern: `if tier_memo:` with else to the bare prompt.
    idx = src.find("if tier_memo:")
    assert idx > 0
    after = src[idx:idx + 800]
    assert "system_prompt = PRE_TEACH_SYSTEM" in after


def test_w09_imports_clean():
    """Module loads without errors."""
    from skills import code_assist  # noqa: F401
    from core import config  # noqa: F401
    from core import complexity  # noqa: F401
