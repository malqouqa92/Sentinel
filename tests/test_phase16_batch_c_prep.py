"""Phase 16 Batch C prep -- config flags + commit-scope hygiene.

Two unrelated landings bundled in this commit:

1. `_COMMIT_INCLUDE` extended to include `tools/` so migration /
   stress / operations scripts (preload_kb, sanitize_kb_secrets,
   stress_test_runner) get versioned alongside the code that uses
   them. Closes the narrow-flavor of the Phase 17 snapshot-scope bug
   for tooling.

2. Batch C feature flags wired in `core/config.py`. Default OFF so
   existing /code behavior is unchanged; flags are read-only sentinels
   that the Batch C implementation will consume when it lands.

Source-level + behavioral checks only -- no live LLM calls.
"""
from __future__ import annotations

from pathlib import Path

from core import config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# /commit scope hygiene
# ─────────────────────────────────────────────────────────────────────


def test_h01_commit_scope_includes_tools():
    """tools/ MUST be in _COMMIT_INCLUDE so operations scripts get
    versioned. Without this, tools/preload_kb.py + sibling files
    stay untracked and only exist on whoever ran them last."""
    src = (_PROJECT_ROOT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8"
    )
    # Find the _COMMIT_INCLUDE tuple and assert "tools" is in it.
    idx = src.find("_COMMIT_INCLUDE = (")
    assert idx > 0, "_COMMIT_INCLUDE definition not found"
    end = src.find(")", idx)
    block = src[idx:end + 1]
    assert '"tools"' in block, (
        f"tools/ missing from _COMMIT_INCLUDE; current:\n{block}"
    )


def test_h02_commit_scope_keeps_existing_paths():
    """Adding tools/ must NOT have removed existing scope entries."""
    src = (_PROJECT_ROOT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8"
    )
    idx = src.find("_COMMIT_INCLUDE = (")
    end = src.find(")", idx)
    block = src[idx:end + 1]
    must_contain = [
        '"core"', '"skills"', '"agents"', '"tests"', '"interfaces"',
        '"workspace/persona"', '"PHASES.md"', '"CLAUDE.md"',
        '"main.py"',
    ]
    for needle in must_contain:
        assert needle in block, (
            f"existing scope entry {needle} disappeared after tools/ add"
        )


def test_h03_commit_scope_does_not_include_workspace_root():
    """workspace/ as a whole MUST NOT be in scope -- it's a runtime
    directory containing backups, output artifacts, etc. Only
    workspace/persona is tracked. The stress-test files in
    workspace/stress_test/ are intentionally transient."""
    src = (_PROJECT_ROOT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8"
    )
    idx = src.find("_COMMIT_INCLUDE = (")
    end = src.find(")", idx)
    block = src[idx:end + 1]
    # `"workspace"` as a bare entry would sweep workspace/backups/ etc.
    # `"workspace/persona"` is fine.
    forbidden = [
        '"workspace",',
        '"workspace"\n',
        '"workspace" ',
    ]
    for f in forbidden:
        assert f not in block, (
            f"workspace/ as a bare scope entry would sweep runtime "
            f"artifacts; saw {f!r}"
        )


# ─────────────────────────────────────────────────────────────────────
# Batch C feature flags
# ─────────────────────────────────────────────────────────────────────


def test_c01_skip_path_enabled_exists():
    """Master switch must exist + be a bool. Default was False on
    first ship (telemetry-only); flipped to True by owner once
    skip-path was validated. Test asserts existence + type only."""
    assert hasattr(config, "SKIP_PATH_ENABLED")
    assert isinstance(config.SKIP_PATH_ENABLED, bool), (
        "Batch C ships in telemetry-only mode; SKIP_PATH_ENABLED "
        "must default to False."
    )


def test_c02_skip_path_threshold_constants_present():
    """The four threshold constants Batch C will read at runtime."""
    for name in (
        "SKIP_PATH_MIN_PASSES",
        "SKIP_PATH_FRESHNESS_DAYS",
        "SKIP_PATH_AGREEMENT_FLOOR",
        "SKIP_PATH_DIFF_MATCH_THRESHOLD",
    ):
        assert hasattr(config, name), f"missing {name}"


def test_c03_min_passes_strict_enough():
    """SKIP_PATH_MIN_PASSES is the lower-bar threshold used when a
    pattern is NOT pinned. Below 3 would let a single lucky run
    qualify a pattern for skip; above ~5 would never beat auto-pin
    (which is the foundation tier). 3 is the documented sweet spot."""
    assert 2 <= config.SKIP_PATH_MIN_PASSES <= 5, (
        f"SKIP_PATH_MIN_PASSES={config.SKIP_PATH_MIN_PASSES} "
        f"outside the sane band [2, 5]"
    )


def test_c04_freshness_window_reasonable():
    """Patterns more than ~30 days unverified may target stale code
    state. A range of 7-90 days is the sane band; <7 invalidates
    every pattern after a week (defeats trust accumulation), >90
    risks replays against drifted code."""
    days = config.SKIP_PATH_FRESHNESS_DAYS
    assert 7 <= days <= 90, (
        f"SKIP_PATH_FRESHNESS_DAYS={days} outside sane band [7, 90]"
    )


def test_c05_agreement_floor_is_meaningful():
    """A pattern with shadow_agreement <0.5 means Qwen's plan
    diverged structurally from Claude's; risky to skip Claude on it.
    NULL is allowed (pre-15c rows have no shadow data)."""
    floor = config.SKIP_PATH_AGREEMENT_FLOOR
    assert 0.0 < floor <= 1.0, (
        f"SKIP_PATH_AGREEMENT_FLOOR={floor} must be in (0.0, 1.0]"
    )


def test_c06_diff_match_threshold_in_sane_band():
    """Diff-match acceptance: 0.7 is the recommended balance of
    'accept reasonable variations' and 'reject wrong-direction
    edits'. <0.5 would accept almost anything; >0.9 would reject
    even minor line-number drift."""
    t = config.SKIP_PATH_DIFF_MATCH_THRESHOLD
    assert 0.5 <= t <= 0.95, (
        f"SKIP_PATH_DIFF_MATCH_THRESHOLD={t} outside sane band [0.5, 0.95]"
    )


# ─────────────────────────────────────────────────────────────────────
# PHASES.md docs marker
# ─────────────────────────────────────────────────────────────────────


def test_d01_phases_md_has_batch_c_readiness_update():
    """PHASES.md should document that Batch C prereqs are in place
    so a future engineer reading the file can confirm 'why is this
    ship plan now real'."""
    src = (_PROJECT_ROOT / "PHASES.md").read_text(encoding="utf-8")
    assert "Readiness update" in src
    assert "AUTO-PIN" in src, (
        "PHASES.md should reference the observed AUTO-PIN event "
        "as concrete evidence Batch D works"
    )
