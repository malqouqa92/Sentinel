"""Phase 16 Option A -- dedup add_pattern by problem_summary.

Without this fix, every successful /code on a recurring prompt
inserts a brand-new KB row with solo_attempts/solo_passes=1, making
the Batch D auto-pin threshold (>=5 passes) and future Batch C
skip-eligibility mathematically unreachable.

This test file proves Option A's contract:
  - First add: inserts new row
  - Repeat same prompt: UPDATEs existing in place, returns same id
  - Different prompt: still inserts new (no false dedup)
  - Archived existing: treated as no-match (revival is explicit /kb restore)
  - Limitation existing: treated as no-match (different category)
  - Pinned existing: still dedups (idempotent on auto-pin)
  - Recipe/diff/embedding refreshed; counters/lifecycle untouched
  - End-to-end: after 5 dedup'd attempts + graduation, auto-pin fires

Coverage:
  Lookup helper:
    A01 -- find_active_pattern_by_problem returns None on miss
    A02 -- returns id on exact match
    A03 -- ignores limitations
    A04 -- ignores archived rows
    A05 -- returns most recent on multiple matches (ORDER BY DESC)
    A06 -- case-sensitive (different case = no match)
    A07 -- credential-shaped problem matches scrubbed-form

  Dedup behavior in add_pattern:
    A10 -- first add inserts row, returns new id
    A11 -- second same prompt does NOT insert; returns same id
    A12 -- different prompt inserts separately
    A13 -- archived existing does NOT block a new insert
    A14 -- limitation with same problem_summary doesn't block insert

  Field updates on dedup:
    A20 -- solution_pattern replaced
    A21 -- solution_code replaced
    A22 -- explanation replaced
    A23 -- tags replaced
    A24 -- base_sha refreshed (None preserves existing via COALESCE)
    A25 -- qwen_plan_recipe + agreement refreshed
    A26 -- embedding regenerated (changed)

  Fields preserved (NOT touched on dedup):
    A30 -- created_at unchanged (first-creation time stable)
    A31 -- created_by_origin unchanged
    A32 -- pinned status unchanged
    A33 -- state unchanged
    A34 -- solo_attempts / solo_passes NOT incremented by add_pattern
            (graduation owns those; add_pattern is the dedup site only)
    A35 -- usage_count unchanged
    A36 -- needs_reteach unchanged

  Telemetry:
    A40 -- log marker 'DEDUP pattern_id=' fires on dedup hit

  End-to-end (the headline behavior):
    A50 -- after 5 successful dedup'd attempts + graduation,
            same pattern auto-pins (Batch D threshold reached)
    A51 -- new prompt after dedup chain: separate counter starts
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from core import config, embeddings as emb
from core.knowledge_base import KnowledgeBase
from core.write_origin import (
    BACKGROUND, FOREGROUND,
    reset_current_write_origin, set_current_write_origin,
)


def _stub_embedder(monkeypatch):
    """Deterministic fake embedding -- no Ollama calls in tests."""
    def fake_embed(text, trace_id="SEN-system"):
        seed = sum(ord(c) for c in (text or "")) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(config.EMBEDDING_DIM).tolist()
        return struct.pack(f"<{len(vec)}f", *vec)
    monkeypatch.setattr(emb, "embed_text", fake_embed)


@pytest.fixture
def fresh_kb(tmp_path: Path, monkeypatch) -> KnowledgeBase:
    db_path = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    _stub_embedder(monkeypatch)
    return KnowledgeBase(db_path=db_path)


def _row(kb: KnowledgeBase, pid: int) -> sqlite3.Row | None:
    conn = sqlite3.connect(kb.db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM knowledge WHERE id = ?", (pid,)
        ).fetchone()
    finally:
        conn.close()


def _add_simple(kb: KnowledgeBase, problem: str, recipe: str = "STEP 1: done",
                code: str = "(diff)", trace_id: str = "SEN-test") -> int:
    return kb.add_pattern(
        tags=["test"],
        problem_summary=problem,
        solution_code=code,
        solution_pattern=recipe,
        explanation="test",
        trace_id=trace_id,
    )


# ─────────────────────────────────────────────────────────────────────
# Lookup helper -- find_active_pattern_by_problem
# ─────────────────────────────────────────────────────────────────────


def test_a01_find_returns_none_on_miss(fresh_kb):
    assert fresh_kb.find_active_pattern_by_problem("never seen") is None


def test_a02_find_returns_id_on_exact_match(fresh_kb):
    pid = _add_simple(fresh_kb, "change the bar")
    assert fresh_kb.find_active_pattern_by_problem("change the bar") == pid


def test_a03_find_ignores_limitations(fresh_kb):
    fresh_kb.add_limitation(
        tags=["x"], problem_summary="hopeless", explanation="failed",
        trace_id="SEN-t",
    )
    assert fresh_kb.find_active_pattern_by_problem("hopeless") is None


def test_a04_find_ignores_archived_rows(fresh_kb):
    pid = _add_simple(fresh_kb, "old job")
    # Manually archive
    conn = sqlite3.connect(fresh_kb.db_path)
    conn.execute(
        "UPDATE knowledge SET state = 'archived' WHERE id = ?", (pid,)
    )
    conn.commit()
    conn.close()
    assert fresh_kb.find_active_pattern_by_problem("old job") is None


def test_a05_find_returns_most_recent_on_multiple(fresh_kb):
    """Two active patterns shouldn't normally exist post-Option-A
    (dedup prevents it), but if they do via direct SQL, return the
    most recent so add_pattern updates the freshest row."""
    p1 = _add_simple(fresh_kb, "x", recipe="OLD")
    # Force a second active row via direct INSERT bypassing add_pattern.
    # Match the canonical ISO timestamp format used by _utcnow_iso so
    # the ORDER BY string comparison sorts correctly.
    conn = sqlite3.connect(fresh_kb.db_path)
    conn.execute(
        "INSERT INTO knowledge (category, tags, problem_summary, "
        "explanation, source_trace_id, created_at, usage_count, state) "
        "VALUES ('pattern', '', 'x', '', 'SEN-t', "
        "strftime('%Y-%m-%dT%H:%M:%S+00:00','now','+1 hour'), "
        "0, 'active')"
    )
    conn.commit()
    conn.close()
    found = fresh_kb.find_active_pattern_by_problem("x")
    assert found is not None and found != p1, (
        "should have returned the newer (later created_at) row"
    )


def test_a06_find_is_case_sensitive(fresh_kb):
    pid = _add_simple(fresh_kb, "CamelCase Prompt")
    # Same prompt in lowercase should NOT match -- prompts are
    # byte-stable per /code submit, case differences signal a
    # different prompt class.
    assert fresh_kb.find_active_pattern_by_problem("camelcase prompt") is None
    assert fresh_kb.find_active_pattern_by_problem("CamelCase Prompt") == pid


def test_a07_find_handles_credential_shaped_prompt(fresh_kb):
    """Edge case: prompt itself contains an ENV-style credential.
    Storage scrubs it; lookup must scrub the query the same way."""
    raw = "set API_TOKEN=my_secret_xyz in env"
    _add_simple(fresh_kb, raw)
    found = fresh_kb.find_active_pattern_by_problem(raw)
    assert found is not None, (
        "lookup must scrub the query so it matches the stored "
        "(scrubbed) problem_summary"
    )


# ─────────────────────────────────────────────────────────────────────
# Dedup behavior in add_pattern
# ─────────────────────────────────────────────────────────────────────


def test_a10_first_add_inserts(fresh_kb):
    pid = _add_simple(fresh_kb, "first time")
    assert pid is not None
    assert _row(fresh_kb, pid) is not None


def test_a11_second_same_prompt_returns_existing_id(fresh_kb):
    p1 = _add_simple(fresh_kb, "repeat me", recipe="STEP 1: done")
    p2 = _add_simple(fresh_kb, "repeat me", recipe="STEP 1: done")
    assert p1 == p2, "second call must return the existing pattern id"
    # No new row inserted
    conn = sqlite3.connect(fresh_kb.db_path)
    n = conn.execute(
        "SELECT COUNT(*) FROM knowledge WHERE category='pattern'"
    ).fetchone()[0]
    conn.close()
    assert n == 1, f"expected 1 pattern row, got {n}"


def test_a12_different_prompt_inserts_separately(fresh_kb):
    p1 = _add_simple(fresh_kb, "alpha")
    p2 = _add_simple(fresh_kb, "beta")
    assert p1 != p2


def test_a13_archived_existing_does_not_block_insert(fresh_kb):
    p1 = _add_simple(fresh_kb, "to_archive")
    conn = sqlite3.connect(fresh_kb.db_path)
    conn.execute(
        "UPDATE knowledge SET state='archived' WHERE id=?", (p1,)
    )
    conn.commit()
    conn.close()
    p2 = _add_simple(fresh_kb, "to_archive")
    assert p2 != p1, (
        "archived existing should not be revived by add_pattern; "
        "explicit /kb restore is the only revival path"
    )


def test_a14_limitation_with_same_summary_does_not_block(fresh_kb):
    fresh_kb.add_limitation(
        tags=["x"], problem_summary="overlapping",
        explanation="failed", trace_id="SEN-t",
    )
    pid = _add_simple(fresh_kb, "overlapping")
    assert pid is not None
    # Both rows exist independently
    conn = sqlite3.connect(fresh_kb.db_path)
    n = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
    conn.close()
    assert n == 2


# ─────────────────────────────────────────────────────────────────────
# Field updates on dedup
# ─────────────────────────────────────────────────────────────────────


def test_a20_solution_pattern_replaced(fresh_kb):
    pid = _add_simple(fresh_kb, "p", recipe="STEP 1: original")
    _add_simple(fresh_kb, "p", recipe="STEP 1: NEWER")
    assert _row(fresh_kb, pid)["solution_pattern"] == "STEP 1: NEWER"


def test_a21_solution_code_replaced(fresh_kb):
    pid = _add_simple(fresh_kb, "p", code="(old diff)")
    _add_simple(fresh_kb, "p", code="(new diff)")
    assert _row(fresh_kb, pid)["solution_code"] == "(new diff)"


def test_a22_explanation_replaced(fresh_kb):
    fresh_kb.add_pattern(
        tags=["x"], problem_summary="p", solution_code="d",
        solution_pattern="STEP 1: done",
        explanation="original explanation", trace_id="SEN-t",
    )
    fresh_kb.add_pattern(
        tags=["x"], problem_summary="p", solution_code="d",
        solution_pattern="STEP 1: done",
        explanation="NEW explanation", trace_id="SEN-t",
    )
    pid = fresh_kb.find_active_pattern_by_problem("p")
    assert _row(fresh_kb, pid)["explanation"] == "NEW explanation"


def test_a23_tags_replaced(fresh_kb):
    fresh_kb.add_pattern(
        tags=["alpha", "beta"], problem_summary="p", solution_code="d",
        solution_pattern="STEP 1: done", explanation="x", trace_id="SEN-t",
    )
    fresh_kb.add_pattern(
        tags=["gamma"], problem_summary="p", solution_code="d",
        solution_pattern="STEP 1: done", explanation="x", trace_id="SEN-t",
    )
    pid = fresh_kb.find_active_pattern_by_problem("p")
    assert "gamma" in _row(fresh_kb, pid)["tags"]
    assert "alpha" not in _row(fresh_kb, pid)["tags"]


def test_a24_base_sha_refreshed_with_coalesce(fresh_kb):
    """If the new attempt provides a base_sha, replace. If new is
    None, COALESCE keeps the old (don't blow away history)."""
    fresh_kb.add_pattern(
        tags=["x"], problem_summary="p", solution_code="d",
        solution_pattern="STEP 1: done", explanation="x", trace_id="SEN-t",
        base_sha="abcd1234",
    )
    fresh_kb.add_pattern(
        tags=["x"], problem_summary="p", solution_code="d",
        solution_pattern="STEP 1: done", explanation="x", trace_id="SEN-t",
        base_sha="ef567890",
    )
    pid = fresh_kb.find_active_pattern_by_problem("p")
    assert _row(fresh_kb, pid)["base_sha"] == "ef567890"
    # Now attempt 3 with base_sha=None -- should keep ef567890
    fresh_kb.add_pattern(
        tags=["x"], problem_summary="p", solution_code="d",
        solution_pattern="STEP 1: done", explanation="x", trace_id="SEN-t",
        base_sha=None,
    )
    assert _row(fresh_kb, pid)["base_sha"] == "ef567890"


def test_a25_qwen_plan_fields_refreshed(fresh_kb):
    fresh_kb.add_pattern(
        tags=["x"], problem_summary="p", solution_code="d",
        solution_pattern="STEP 1: done", explanation="x", trace_id="SEN-t",
        qwen_plan_recipe="OLD shadow",
        qwen_plan_agreement=0.5,
    )
    fresh_kb.add_pattern(
        tags=["x"], problem_summary="p", solution_code="d",
        solution_pattern="STEP 1: done", explanation="x", trace_id="SEN-t",
        qwen_plan_recipe="NEW shadow",
        qwen_plan_agreement=0.875,
    )
    pid = fresh_kb.find_active_pattern_by_problem("p")
    r = _row(fresh_kb, pid)
    assert r["qwen_plan_recipe"] == "NEW shadow"
    assert abs(r["qwen_plan_agreement"] - 0.875) < 0.0001


def test_a26_embedding_regenerated(fresh_kb):
    """Embedding is computed from tags + problem + solution_pattern.
    A new solution_pattern means a new embedding."""
    fresh_kb.add_pattern(
        tags=["x"], problem_summary="p", solution_code="d",
        solution_pattern="STEP 1: original_recipe_text",
        explanation="x", trace_id="SEN-t",
    )
    pid = fresh_kb.find_active_pattern_by_problem("p")
    e1 = _row(fresh_kb, pid)["embedding"]
    fresh_kb.add_pattern(
        tags=["x"], problem_summary="p", solution_code="d",
        solution_pattern="STEP 1: completely_different_shape",
        explanation="x", trace_id="SEN-t",
    )
    e2 = _row(fresh_kb, pid)["embedding"]
    assert e1 != e2, "embedding should change when solution_pattern changes"


# ─────────────────────────────────────────────────────────────────────
# Fields preserved on dedup
# ─────────────────────────────────────────────────────────────────────


def test_a30_created_at_preserved(fresh_kb):
    pid = _add_simple(fresh_kb, "p")
    t1 = _row(fresh_kb, pid)["created_at"]
    _add_simple(fresh_kb, "p")  # dedup
    t2 = _row(fresh_kb, pid)["created_at"]
    assert t1 == t2, "first-creation timestamp must be stable across dedup"


def test_a31_created_by_origin_preserved(fresh_kb):
    """First writer's origin tag is the audit signal -- don't
    overwrite when a later same-prompt submit dedups."""
    tok = set_current_write_origin(BACKGROUND)
    try:
        pid = _add_simple(fresh_kb, "p")
    finally:
        reset_current_write_origin(tok)
    assert _row(fresh_kb, pid)["created_by_origin"] == BACKGROUND
    # Re-add as foreground -- should NOT change origin
    tok = set_current_write_origin(FOREGROUND)
    try:
        _add_simple(fresh_kb, "p")
    finally:
        reset_current_write_origin(tok)
    assert _row(fresh_kb, pid)["created_by_origin"] == BACKGROUND


def test_a32_pinned_status_preserved(fresh_kb):
    pid = _add_simple(fresh_kb, "p")
    fresh_kb.pin_pattern(pid)
    _add_simple(fresh_kb, "p")  # dedup
    assert _row(fresh_kb, pid)["pinned"] == 1


def test_a33_state_preserved(fresh_kb):
    pid = _add_simple(fresh_kb, "p")
    assert _row(fresh_kb, pid)["state"] == "active"
    _add_simple(fresh_kb, "p")
    assert _row(fresh_kb, pid)["state"] == "active"


def test_a34_solo_counters_not_changed_by_add_pattern(fresh_kb):
    """add_pattern does NOT bump solo_attempts/solo_passes -- those
    are owned by record_solo_attempt (called from graduation). Dedup
    site only refreshes recipe/diff/embedding."""
    pid = _add_simple(fresh_kb, "p")
    r1 = _row(fresh_kb, pid)
    assert r1["solo_attempts"] == 0 and r1["solo_passes"] == 0
    _add_simple(fresh_kb, "p")  # dedup
    r2 = _row(fresh_kb, pid)
    assert r2["solo_attempts"] == 0 and r2["solo_passes"] == 0


def test_a35_usage_count_preserved(fresh_kb):
    """usage_count is incremented by retrieval (search), not write
    paths. Dedup must not touch it."""
    pid = _add_simple(fresh_kb, "p")
    conn = sqlite3.connect(fresh_kb.db_path)
    conn.execute("UPDATE knowledge SET usage_count = 7 WHERE id = ?", (pid,))
    conn.commit()
    conn.close()
    _add_simple(fresh_kb, "p")  # dedup
    assert _row(fresh_kb, pid)["usage_count"] == 7


def test_a36_needs_reteach_preserved(fresh_kb):
    """needs_reteach is owned by record_solo_attempt's failure-rate
    logic. add_pattern dedup must not flip it back to 0."""
    pid = _add_simple(fresh_kb, "p")
    conn = sqlite3.connect(fresh_kb.db_path)
    conn.execute("UPDATE knowledge SET needs_reteach = 1 WHERE id = ?", (pid,))
    conn.commit()
    conn.close()
    _add_simple(fresh_kb, "p")  # dedup
    assert _row(fresh_kb, pid)["needs_reteach"] == 1


# ─────────────────────────────────────────────────────────────────────
# Telemetry
# ─────────────────────────────────────────────────────────────────────


def test_a40_dedup_log_marker_fires(fresh_kb, caplog):
    """A grep-able 'DEDUP pattern_id=' marker fires on dedup hit so
    log scans can see the difference between insert and dedup."""
    import logging
    caplog.set_level(logging.DEBUG)
    # Switch the logger module to use stdlib logging so caplog catches.
    # The project's log_event emits to a JSONL file plus stdlib root
    # logger; assert via a side-channel: re-read the trace via a
    # custom log capture.
    captured = []
    from core import logger as core_logger
    real = core_logger.log_event

    def hook(trace_id, level, component, message):
        captured.append((level, component, message))
        return real(trace_id, level, component, message)

    import core.knowledge_base as kb_mod
    original = kb_mod.log_event
    kb_mod.log_event = hook
    try:
        _add_simple(fresh_kb, "p")
        _add_simple(fresh_kb, "p")
    finally:
        kb_mod.log_event = original

    assert any("DEDUP pattern_id=" in m for (_, _, m) in captured), (
        f"expected 'DEDUP pattern_id=' marker, got: {captured}"
    )


# ─────────────────────────────────────────────────────────────────────
# End-to-end: 5 successful dedup'd attempts -> auto-pin fires
# ─────────────────────────────────────────────────────────────────────


def test_a50_auto_pin_fires_after_5_dedup_then_graduate(fresh_kb):
    """The headline test. With Option A landed, repeated successful
    /code on the same prompt accumulates solo_passes on a SINGLE
    pattern row. After 5 successful graduations, the Batch D
    auto-pin threshold fires and pinned=1.

    This is the test that actually validates 'Qwen earns trust'
    architecturally. Without Option A, this test is unreachable
    (each add_pattern would create a new row and the counter would
    never exceed 1)."""
    pid = None
    for i in range(5):
        new_pid = _add_simple(
            fresh_kb, "the same recurring prompt",
            recipe=f"STEP 1: done summary=\"attempt {i+1}\"",
        )
        if pid is None:
            pid = new_pid
        else:
            assert new_pid == pid, f"attempt {i+1} should dedup"
        # Graduation runs immediately after add_pattern in production;
        # simulate the resulting record_solo_attempt(passed=True).
        fresh_kb.record_solo_attempt(pid, passed=True, trace_id="SEN-grad")

    r = _row(fresh_kb, pid)
    assert r["solo_attempts"] == 5
    assert r["solo_passes"] == 5
    # Batch D auto-pin (>=5 passes AND 100% rate) -- must have fired.
    assert r["pinned"] == 1, (
        f"Expected auto-pin after 5/5 solo passes; pinned={r['pinned']}"
    )


def test_a51_new_prompt_after_dedup_chain_starts_fresh(fresh_kb):
    """Verify dedup is keyed on prompt; a different prompt gets its
    own fresh counter (doesn't free-ride on someone else's pinning)."""
    p1 = _add_simple(fresh_kb, "prompt A")
    for _ in range(5):
        fresh_kb.record_solo_attempt(p1, passed=True, trace_id="SEN-t")
    assert _row(fresh_kb, p1)["pinned"] == 1

    p2 = _add_simple(fresh_kb, "prompt B")
    assert p2 != p1
    r = _row(fresh_kb, p2)
    assert r["solo_attempts"] == 0 and r["solo_passes"] == 0
    assert r["pinned"] == 0
