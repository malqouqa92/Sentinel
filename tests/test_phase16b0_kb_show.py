"""Phase 16 Batch 0 -- /kb show <id> transparency command.

Source-level + behavioral tests for the new /kb show subcommand
that prints the full record of a pattern: problem, both recipes,
agreement breakdown (file Jaccard / tool Jaccard / step proximity),
graduation stats, lifecycle state.

Coverage:
  Wiring:
    K01 -- handle_kb dispatches `show <int>` correctly
    K02 -- show without arg or with non-int falls through to usage
    K03 -- usage line + docstring mention `/kb show <id>`

  Display content:
    K11 -- output includes the pattern's problem_summary
    K12 -- output includes both Claude recipe and Qwen recipe
    K13 -- output includes the stored agreement score
    K14 -- output includes per-component agreement breakdown
    K15 -- output includes graduation solo stats
    K16 -- output includes lifecycle state + pinned + origin
    K17 -- show on a missing id returns "No pattern with id="
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_bot_src() -> str:
    return (
        PROJECT_ROOT / "interfaces" / "telegram_bot.py"
    ).read_text(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────
# Source-level wiring
# ─────────────────────────────────────────────────────────────────


def test_k01_show_dispatch_present():
    src = _read_bot_src()
    # The dispatch must check sub == "show" AND that args[1] is a
    # digit (i.e. /kb show 42, not /kb show foo).
    assert (
        'if sub == "show" and len(args) >= 2 and args[1].isdigit():'
        in src
    ), "show dispatch missing or malformed"


def test_k02_show_falls_through_when_no_arg():
    """Without a numeric arg, the show branch should NOT match
    (so we fall through to the usage-line catch-all)."""
    src = _read_bot_src()
    # Find the show branch + the usage catch-all
    show_idx = src.find('if sub == "show"')
    usage_idx = src.find(
        '"Usage: /kb  •  /kb verify <id>'
    )
    assert show_idx > 0 and usage_idx > 0
    assert show_idx < usage_idx, (
        "usage line must come AFTER the show branch as fallthrough"
    )


def test_k03_usage_and_docstring_mention_show():
    src = _read_bot_src()
    assert "/kb show <id>" in src
    # The docstring on handle_kb mentions /kb show too
    handle_kb_idx = src.find("async def handle_kb")
    body = src[handle_kb_idx:handle_kb_idx + 2000]
    assert "/kb show <id>" in body, (
        "handle_kb docstring missing /kb show"
    )


# ─────────────────────────────────────────────────────────────────
# Display content (source-level: assert the literal labels exist)
# ─────────────────────────────────────────────────────────────────


def test_k11_output_includes_problem_summary():
    src = _read_bot_src()
    assert '_trim(entry.problem_summary' in src, (
        "show handler doesn't render entry.problem_summary"
    )


def test_k12_output_includes_both_recipes():
    src = _read_bot_src()
    # Both labels present
    assert "Claude recipe" in src
    assert "Qwen shadow recipe" in src
    # And both fields are read from the entry
    assert "entry.solution_pattern" in src
    assert "entry.qwen_plan_recipe" in src


def test_k13_output_includes_stored_agreement():
    src = _read_bot_src()
    assert "Agreement (stored)" in src
    assert "entry.qwen_plan_agreement" in src


def test_k14_output_includes_component_breakdown():
    """File Jaccard + tool Jaccard + step proximity each shown
    with their weight, so the user can see why the score is
    whatever it is."""
    src = _read_bot_src()
    assert "file Jaccard" in src
    assert "tool Jaccard" in src
    assert "step proximity" in src
    # Weights from plan_agreement are imported and displayed
    assert "W_FILES" in src
    assert "W_TOOLS" in src
    assert "W_STEPS" in src


def test_k15_output_includes_graduation_stats():
    src = _read_bot_src()
    assert "Graduation:" in src
    assert "entry.solo_passes" in src
    assert "entry.solo_attempts" in src
    assert "needs_reteach" in src
    assert "last_verified" in src


def test_k16_output_includes_lifecycle_state():
    src = _read_bot_src()
    # state, pinned, created_by_origin, base_sha all displayed
    assert "entry.state" in src
    assert "entry.pinned" in src
    assert "entry.created_by_origin" in src
    assert "entry.base_sha" in src


def test_k17_missing_id_returns_clear_message():
    src = _read_bot_src()
    # The handler checks `if entry is None` and replies with
    # "No pattern with id=<id>"
    show_idx = src.find('if sub == "show"')
    body = src[show_idx:show_idx + 1000]
    assert "if entry is None:" in body
    assert "No pattern with id=" in body


# ─────────────────────────────────────────────────────────────────
# Sanity: the imports the handler needs are reachable
# ─────────────────────────────────────────────────────────────────


def test_k21_plan_agreement_exports_helpers():
    """The handler imports _files_and_tools, _jaccard,
    _step_count_proximity, W_FILES, W_TOOLS, W_STEPS from
    core.plan_agreement -- they all need to exist."""
    from core.plan_agreement import (
        _files_and_tools, _jaccard, _step_count_proximity,
        W_FILES, W_TOOLS, W_STEPS,
    )
    # Smoke: helpers are callable / weights are floats
    assert callable(_files_and_tools)
    assert callable(_jaccard)
    assert callable(_step_count_proximity)
    assert isinstance(W_FILES, float)
    assert isinstance(W_TOOLS, float)
    assert isinstance(W_STEPS, float)
    # Weights sum to 1.0 (canonical scoring contract)
    assert abs(W_FILES + W_TOOLS + W_STEPS - 1.0) < 1e-6


def test_k22_kb_get_pattern_returns_full_entry():
    """The handler relies on KnowledgeBase.get_pattern(pid) returning
    a KnowledgeEntry with all fields populated. Sanity check."""
    from core.knowledge_base import KnowledgeBase, KnowledgeEntry
    import inspect
    sig = inspect.signature(KnowledgeBase.get_pattern)
    assert "pattern_id" in sig.parameters
    # KnowledgeEntry has the fields the show handler reads
    fields = set(KnowledgeEntry.model_fields)
    needed = {
        "id", "category", "tags", "problem_summary",
        "solution_code", "solution_pattern", "explanation",
        "source_trace_id", "created_at", "usage_count",
        "solo_attempts", "solo_passes", "last_verified_at",
        "needs_reteach", "base_sha", "state", "pinned",
        "archived_at", "created_by_origin",
        "qwen_plan_recipe", "qwen_plan_agreement",
    }
    missing = needed - fields
    assert not missing, f"KnowledgeEntry missing fields: {missing}"
