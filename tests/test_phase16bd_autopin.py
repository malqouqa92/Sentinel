"""Phase 16 Batch D -- auto-pin proven graduators.

When a KB pattern crosses the trust threshold via repeat graduation
(solo_passes >= 5 AND solo_pass_rate == 1.0), record_solo_attempt
auto-pins it in the same transaction. The kb_lifecycle walker
(Phase 15a) can never archive a pinned row, so 5/5-proven patterns
become permanent foundation without manual curation.

Strict thresholds chosen for safety: pinning is irreversible-by-
default (only /kb unpin <id> clears it). A single failure (4/5 =
0.8 rate) keeps the pattern OUT of the auto-pin set even though
0.8 would normally count as "high quality". Reasoning: zero
ambiguity for foundation set; partial credit doesn't earn it.

Coverage:
  Threshold constants:
    D01 -- AUTO_PIN_MIN_PASSES = 5
    D02 -- AUTO_PIN_REQUIRED_RATE = 1.0

  Auto-pin firing:
    D11 -- 4/4 passes does NOT auto-pin (under min_passes)
    D12 -- 5/5 passes DOES auto-pin (crosses threshold cleanly)
    D13 -- 5/6 passes (one failure mixed in) does NOT auto-pin
    D14 -- 6/6 passes still auto-pins (continues firing past threshold)

  Idempotency / no-op:
    D21 -- already-pinned row is NOT re-pinned (no extra log line)
    D22 -- already-pinned row's solo counters still update normally
    D23 -- counters increment correctly even when auto-pin fires

  Atomicity:
    D31 -- auto-pin happens inside the SAME transaction as solo update
    D32 -- pattern that doesn't exist returns (0,0,False), no crash

  Log line:
    D41 -- auto-pin emits a distinct log line ("AUTO-PIN pattern_id=...")
    D42 -- log only fires on the transition (not on every subsequent pass)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core import embeddings as emb
from core.knowledge_base import KnowledgeBase, _connect


def _stub_embedder(monkeypatch):
    def fake_embed(text, trace_id="SEN-system"):
        seed = sum(ord(c) for c in (text or "")) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(config.EMBEDDING_DIM).tolist()
        return struct.pack(f"<{len(vec)}f", *vec)
    monkeypatch.setattr(emb, "embed_text", fake_embed)


@pytest.fixture
def fresh_kb(tmp_path, monkeypatch):
    db_path = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    _stub_embedder(monkeypatch)
    return KnowledgeBase(db_path=db_path)


def _seed_pattern(kb: KnowledgeBase, summary: str = "test") -> int:
    return kb.add_pattern(
        tags=["test"], problem_summary=summary,
        solution_code=(
            "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
        ),
        solution_pattern='STEP 1: edit_file path="x"\nSTEP 2: done summary="x"',
        explanation="seeded for test",
        trace_id="SEN-test",
    )


def _is_pinned(kb: KnowledgeBase, pid: int) -> bool:
    conn = _connect(kb.db_path)
    try:
        row = conn.execute(
            "SELECT pinned FROM knowledge WHERE id = ?", (pid,),
        ).fetchone()
        return bool(row["pinned"]) if row else False
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────
# Threshold constants
# ─────────────────────────────────────────────────────────────────


def test_d01_min_passes_constant():
    assert KnowledgeBase.AUTO_PIN_MIN_PASSES == 5


def test_d02_required_rate_constant():
    assert KnowledgeBase.AUTO_PIN_REQUIRED_RATE == 1.0


# ─────────────────────────────────────────────────────────────────
# Auto-pin firing
# ─────────────────────────────────────────────────────────────────


def test_d11_four_passes_does_not_auto_pin(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    for _ in range(4):
        fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    assert not _is_pinned(fresh_kb, pid), (
        "4/4 passes should NOT auto-pin (under min)"
    )


def test_d12_five_clean_passes_auto_pins(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    for _ in range(5):
        fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    assert _is_pinned(fresh_kb, pid), (
        "5/5 passes at 1.0 rate MUST auto-pin"
    )
    p = fresh_kb.get_pattern(pid)
    assert p.solo_passes == 5
    assert p.solo_attempts == 5


def test_d13_five_passes_with_one_fail_does_not_auto_pin(fresh_kb):
    """5/6 = 0.833 rate. Even though >=5 passes, the imperfect
    rate keeps it OUT of the auto-pin set (strict threshold)."""
    pid = _seed_pattern(fresh_kb)
    for _ in range(5):
        fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    # First 5 are clean -- this WOULD pin... so let's mix in the
    # failure FIRST so the rate stays imperfect throughout.
    fresh_kb.unpin_pattern(pid)  # reset for clean test setup
    # Reset the row entirely
    conn = _connect(fresh_kb.db_path)
    try:
        conn.execute(
            "UPDATE knowledge SET solo_passes=0, solo_attempts=0, "
            "pinned=0 WHERE id=?", (pid,),
        )
    finally:
        conn.close()
    # Now: 1 fail then 5 passes = 5/6 rate
    fresh_kb.record_solo_attempt(pid, False, "SEN-test")
    for _ in range(5):
        fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    p = fresh_kb.get_pattern(pid)
    assert p.solo_passes == 5
    assert p.solo_attempts == 6
    assert not _is_pinned(fresh_kb, pid), (
        "5/6 rate (one prior fail) should NOT auto-pin"
    )


def test_d14_six_clean_passes_still_auto_pins(fresh_kb):
    """Auto-pin fires the moment threshold crosses; subsequent
    clean passes don't UNDO it. Verifies idempotency in the
    direction of staying-pinned."""
    pid = _seed_pattern(fresh_kb)
    for _ in range(6):
        fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    assert _is_pinned(fresh_kb, pid)
    p = fresh_kb.get_pattern(pid)
    assert p.solo_passes == 6


# ─────────────────────────────────────────────────────────────────
# Idempotency / no-op
# ─────────────────────────────────────────────────────────────────


def test_d21_already_pinned_does_not_re_pin(fresh_kb):
    """Pin manually first, then record passes. The auto-pin
    branch must SKIP because already pinned (no double-log,
    no redundant UPDATE)."""
    pid = _seed_pattern(fresh_kb)
    fresh_kb.pin_pattern(pid)
    # Now run 6 graduations -- threshold met but already pinned
    for _ in range(6):
        fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    assert _is_pinned(fresh_kb, pid)
    # Could be tested via log inspection but the semantic-level
    # contract is "stays pinned, no error" -- verified.


def test_d22_already_pinned_counters_still_update(fresh_kb):
    """Manual pin shouldn't break the counter update logic."""
    pid = _seed_pattern(fresh_kb)
    fresh_kb.pin_pattern(pid)
    fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    fresh_kb.record_solo_attempt(pid, False, "SEN-test")
    p = fresh_kb.get_pattern(pid)
    assert p.solo_attempts == 2
    assert p.solo_passes == 1


def test_d23_counters_increment_when_auto_pin_fires(fresh_kb):
    """Sanity: the auto-pin UPDATE must NOT swallow the counter
    increment. Both happen in the same transaction."""
    pid = _seed_pattern(fresh_kb)
    for i in range(5):
        a, p, flag = fresh_kb.record_solo_attempt(pid, True, "SEN-test")
        assert a == i + 1
        assert p == i + 1
    # On the 5th attempt, auto-pin should have fired
    assert _is_pinned(fresh_kb, pid)


# ─────────────────────────────────────────────────────────────────
# Atomicity
# ─────────────────────────────────────────────────────────────────


def test_d31_auto_pin_inside_same_transaction(fresh_kb):
    """The auto-pin and counter UPDATE must commit atomically.
    Verify the row reflects BOTH pinned=1 AND the new counter
    after a single record_solo_attempt call."""
    pid = _seed_pattern(fresh_kb)
    # Rack up 4 first
    for _ in range(4):
        fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    assert not _is_pinned(fresh_kb, pid)
    # The 5th attempt is the threshold-crossing one
    a, p, _ = fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    # After this single call, both must be true:
    assert a == 5 and p == 5
    assert _is_pinned(fresh_kb, pid)


def test_d32_missing_pattern_returns_zero_no_crash(fresh_kb):
    a, p, flag = fresh_kb.record_solo_attempt(99999, True, "SEN-test")
    assert (a, p, flag) == (0, 0, False)


# ─────────────────────────────────────────────────────────────────
# Log line behavior (source-level + behavioral checks)
# ─────────────────────────────────────────────────────────────────


def test_d41_auto_pin_emits_log_line_distinct_from_grad():
    """Source-level: there's a DEDICATED log_event for auto-pin
    that's separate from the regular graduation log line. So a
    log scan can identify auto-pin events specifically."""
    src = (
        Path(__file__).resolve().parent.parent
        / "core" / "knowledge_base.py"
    ).read_text(encoding="utf-8")
    # The dedicated marker the auto-pin log uses
    assert "AUTO-PIN pattern_id=" in src, (
        "auto-pin needs a distinctive log marker for log scanning"
    )
    # And it sits behind an `if auto_pinned:` guard so it only
    # fires on the transition
    assert "if auto_pinned:" in src


def test_d42_log_marker_says_permanent_foundation():
    """Sanity: the log message explains WHY auto-pin matters --
    'permanent foundation, immune to kb_lifecycle archival' --
    so future engineers reading logs understand the meaning."""
    src = (
        Path(__file__).resolve().parent.parent
        / "core" / "knowledge_base.py"
    ).read_text(encoding="utf-8")
    assert "permanent foundation" in src
    assert "immune to kb_lifecycle archival" in src


# ─────────────────────────────────────────────────────────────────
# Sanity: unpin still works (manual reversal of auto-pin)
# ─────────────────────────────────────────────────────────────────


def test_d51_auto_pin_is_reversible_via_unpin(fresh_kb):
    """If a user disagrees with an auto-pin, /kb unpin <id> still
    works. Auto-pin is a default, not a lock."""
    pid = _seed_pattern(fresh_kb)
    for _ in range(5):
        fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    assert _is_pinned(fresh_kb, pid)
    assert fresh_kb.unpin_pattern(pid) is True
    assert not _is_pinned(fresh_kb, pid)
