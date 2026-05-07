"""Phase 15a -- archive-not-delete lifecycle for KB + memory.

Three subsystems used to silently DELETE old data: knowledge_base.prune,
memory.prune_episodes, memory._prune_semantic. Phase 15a swapped each
DELETE for an UPDATE state='archived'; pin protects rows from every
automatic transition; auto_transition_lifecycle ages active->stale and
stale->archived with the same pin guard. This file ECCs all of that.

Coverage:
  Schema migration:
    L01 -- KB fresh DB has state/pinned/archived_at + partial indexes
    L02 -- KB pre-Phase-15a DB gets ALTERed cleanly on init
    L03 -- KB double-init is idempotent (no duplicate-column error)
    L04 -- memory fresh DB has lifecycle cols + indexes (both tables)
    L05 -- memory pre-Phase-15a DB gets ALTERed cleanly
    L06 -- memory double-init is idempotent

  KB search filtering:
    L11 -- search() default excludes archived rows
    L12 -- search(include_archived=True) returns archived rows
    L13 -- archived row still readable via get_pattern (direct lookup)

  KB pin/unpin/restore:
    L21 -- pin_pattern flips flag, idempotent
    L22 -- unpin_pattern clears flag, doesn't change state
    L23 -- restore_pattern: archived -> active, archived_at -> NULL
    L24 -- pin/unpin/restore on missing id return False

  KB prune (archive-not-delete):
    L31 -- prune ARCHIVES instead of deleting (rows still on disk)
    L32 -- prune skips pinned rows
    L33 -- prune respects existing archived rows (doesn't re-archive)

  KB auto_transition_lifecycle:
    L41 -- active row older than stale_after with low usage -> stale
    L42 -- stale row older than archive_after -> archived
    L43 -- pinned active row is NEVER transitioned even when ancient
    L44 -- high-usage active row is NOT marked stale (still earning)
    L45 -- recent active row not affected

  Memory episodic + semantic mirrors:
    L51 -- episodic prune archives instead of deletes
    L52 -- episodic prune skips pinned rows
    L53 -- episodic search excludes archived by default
    L54 -- episodic pin/unpin/restore round-trip
    L55 -- semantic prune archives instead of deletes
    L56 -- semantic prune skips pinned rows
    L57 -- list_facts excludes archived by default
    L58 -- semantic pin/unpin/restore round-trip
    L59 -- store_fact upsert revives archived row to active

  Memory auto_transition_lifecycle:
    L61 -- episodic active(low rel + old) -> stale -> archived
    L62 -- semantic active(low conf + old) -> stale -> archived
    L63 -- pinned memory rows are NEVER transitioned
"""
from __future__ import annotations

import struct
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core import embeddings as emb
from core.knowledge_base import KnowledgeBase, _connect as _kb_connect
from core.memory import MemoryManager, _connect as _mem_connect


# ─────────────────────────────────────────────────────────────────
# fixtures + stubs (match the Phase 14a / pre-15 patterns)
# ─────────────────────────────────────────────────────────────────


def _stub_embedder(monkeypatch):
    """Deterministic per-text vector so equal texts -> equal vectors.
    Lifted directly from test_pre15_hybrid_retrieval.py."""
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


@pytest.fixture
def fresh_mem(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(config, "MEMORY_DB_PATH", db_path)
    return MemoryManager(db_path=db_path)


def _seed_pattern(kb: KnowledgeBase, summary: str = "test problem") -> int:
    return kb.add_pattern(
        tags=["test"],
        problem_summary=summary,
        solution_code="print(2+2)",
        solution_pattern="just print",
        explanation="trivial",
        trace_id="SEN-test",
    )


def _force_age(db_path: Path, table: str, row_id: int, days_old: int):
    """Backdate a row's created_at so age-based queries hit it."""
    ts = (
        datetime.now(timezone.utc) - timedelta(days=days_old)
    ).isoformat()
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute(
            f"UPDATE {table} SET created_at = ? WHERE id = ?",
            (ts, row_id),
        )
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────
# Schema migration
# ─────────────────────────────────────────────────────────────────


def test_l01_fresh_kb_has_lifecycle_cols(fresh_kb):
    conn = _kb_connect(fresh_kb.db_path)
    try:
        cols = {r["name"] for r in conn.execute(
            "PRAGMA table_info(knowledge)"
        ).fetchall()}
        idx = {r["name"] for r in conn.execute(
            "PRAGMA index_list(knowledge)"
        ).fetchall()}
    finally:
        conn.close()
    for c in ("state", "pinned", "archived_at"):
        assert c in cols, f"missing column: {c}"
    assert "idx_knowledge_pinned" in idx
    assert "idx_knowledge_archived" in idx


def test_l02_pre_phase15_kb_db_gets_migrated(tmp_path, monkeypatch):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            tags TEXT NOT NULL,
            problem_summary TEXT NOT NULL,
            solution_code TEXT,
            solution_pattern TEXT,
            explanation TEXT NOT NULL,
            source_trace_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            usage_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO knowledge (category, tags, problem_summary, "
        "solution_code, solution_pattern, explanation, source_trace_id, "
        "created_at) VALUES ('pattern', 'x', 'old problem', 'old code', "
        "'old pattern', 'old expl', 'SEN-old', '2026-01-01T00:00:00')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    _stub_embedder(monkeypatch)
    kb = KnowledgeBase(db_path=db_path)
    p = kb.get_pattern(1)
    assert p is not None
    # Defaults backfilled by the NOT NULL DEFAULT clauses.
    assert p.state == "active"
    assert p.pinned is False
    assert p.archived_at is None


def test_l03_kb_double_init_idempotent(fresh_kb):
    KnowledgeBase(db_path=fresh_kb.db_path)
    KnowledgeBase(db_path=fresh_kb.db_path)


def test_l04_fresh_mem_has_lifecycle_cols(fresh_mem):
    conn = _mem_connect(fresh_mem.db_path)
    try:
        for tbl in ("episodic_memory", "semantic_memory"):
            cols = {r["name"] for r in conn.execute(
                f"PRAGMA table_info({tbl})"
            ).fetchall()}
            idx = {r["name"] for r in conn.execute(
                f"PRAGMA index_list({tbl})"
            ).fetchall()}
            for c in ("state", "pinned", "archived_at"):
                assert c in cols, f"{tbl} missing column: {c}"
            assert f"idx_{tbl}_pinned" in idx
            assert f"idx_{tbl}_archived" in idx
    finally:
        conn.close()


def test_l05_pre_phase15_mem_db_gets_migrated(tmp_path, monkeypatch):
    db_path = tmp_path / "old_mem.db"
    conn = sqlite3.connect(db_path)
    # Old shape (Phase 10).
    conn.execute(
        """
        CREATE TABLE episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            relevance_score REAL NOT NULL DEFAULT 1.0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE semantic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL DEFAULT 'fact',
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO episodic_memory (scope, trace_id, event_type, "
        "summary, created_at) VALUES "
        "('global', 'SEN-old', 'note', 'old summary', "
        "'2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO semantic_memory (key, value, source, "
        "created_at, updated_at) VALUES "
        "('k1', 'v1', 'user_explicit', '2026-01-01T00:00:00', "
        "'2026-01-01T00:00:00')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "MEMORY_DB_PATH", db_path)
    mem = MemoryManager(db_path=db_path)
    # Old episodic row should fall through to defaults.
    eps = mem.get_recent_episodes(scope="global")
    assert len(eps) == 1
    assert eps[0].state == "active"
    assert eps[0].pinned is False
    facts = mem.list_facts()
    assert len(facts) == 1
    assert facts[0].state == "active"


def test_l06_mem_double_init_idempotent(fresh_mem):
    MemoryManager(db_path=fresh_mem.db_path)
    MemoryManager(db_path=fresh_mem.db_path)


# ─────────────────────────────────────────────────────────────────
# KB search filtering
# ─────────────────────────────────────────────────────────────────


def test_l11_search_excludes_archived_by_default(fresh_kb):
    pid_a = _seed_pattern(fresh_kb, "kept active")
    pid_b = _seed_pattern(fresh_kb, "going archived")
    # Manually archive B.
    conn = _kb_connect(fresh_kb.db_path)
    try:
        conn.execute(
            "UPDATE knowledge SET state = 'archived', "
            "archived_at = '2026-05-05T00:00:00' WHERE id = ?",
            (pid_b,),
        )
    finally:
        conn.close()
    results = fresh_kb.search("test", hybrid=False)
    ids = {r.id for r in results}
    assert pid_a in ids
    assert pid_b not in ids


def test_l12_search_includes_archived_when_flagged(fresh_kb):
    pid = _seed_pattern(fresh_kb, "going archived")
    conn = _kb_connect(fresh_kb.db_path)
    try:
        conn.execute(
            "UPDATE knowledge SET state = 'archived', "
            "archived_at = '2026-05-05T00:00:00' WHERE id = ?",
            (pid,),
        )
    finally:
        conn.close()
    results = fresh_kb.search("test", hybrid=False, include_archived=True)
    assert any(r.id == pid for r in results)


def test_l13_get_pattern_returns_archived(fresh_kb):
    """Direct id lookup should NOT filter -- needed by /kb verify
    + /kb restore on an archived pattern."""
    pid = _seed_pattern(fresh_kb)
    conn = _kb_connect(fresh_kb.db_path)
    try:
        conn.execute(
            "UPDATE knowledge SET state = 'archived' WHERE id = ?",
            (pid,),
        )
    finally:
        conn.close()
    p = fresh_kb.get_pattern(pid)
    assert p is not None
    assert p.state == "archived"


# ─────────────────────────────────────────────────────────────────
# KB pin/unpin/restore
# ─────────────────────────────────────────────────────────────────


def test_l21_pin_pattern_flips_flag(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    assert fresh_kb.pin_pattern(pid) is True
    p = fresh_kb.get_pattern(pid)
    assert p.pinned is True
    # Idempotent.
    assert fresh_kb.pin_pattern(pid) is True
    assert fresh_kb.get_pattern(pid).pinned is True


def test_l22_unpin_does_not_touch_state(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    fresh_kb.pin_pattern(pid)
    # Move it to archived state too.
    conn = _kb_connect(fresh_kb.db_path)
    try:
        conn.execute(
            "UPDATE knowledge SET state = 'archived' WHERE id = ?",
            (pid,),
        )
    finally:
        conn.close()
    fresh_kb.unpin_pattern(pid)
    p = fresh_kb.get_pattern(pid)
    assert p.pinned is False
    assert p.state == "archived"  # untouched


def test_l23_restore_pattern_clears_archived_state(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    conn = _kb_connect(fresh_kb.db_path)
    try:
        conn.execute(
            "UPDATE knowledge SET state = 'archived', "
            "archived_at = '2026-05-05T00:00:00' WHERE id = ?",
            (pid,),
        )
    finally:
        conn.close()
    assert fresh_kb.restore_pattern(pid) is True
    p = fresh_kb.get_pattern(pid)
    assert p.state == "active"
    assert p.archived_at is None


def test_l24_pin_unpin_restore_missing_id(fresh_kb):
    assert fresh_kb.pin_pattern(99999) is False
    assert fresh_kb.unpin_pattern(99999) is False
    assert fresh_kb.restore_pattern(99999) is False


# ─────────────────────────────────────────────────────────────────
# KB prune
# ─────────────────────────────────────────────────────────────────


def test_l31_prune_archives_instead_of_deleting(fresh_kb):
    ids = []
    for i in range(10):
        ids.append(_seed_pattern(fresh_kb, f"summary {i}"))
    # Set usage_count so the prune-order is deterministic.
    conn = _kb_connect(fresh_kb.db_path)
    try:
        for i, pid in enumerate(ids):
            conn.execute(
                "UPDATE knowledge SET usage_count = ? WHERE id = ?",
                (i, pid),
            )
    finally:
        conn.close()
    archived = fresh_kb.prune(max_entries=5)
    assert archived == 5
    # All 10 rows still exist on disk -- nothing was deleted.
    conn = _kb_connect(fresh_kb.db_path)
    try:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM knowledge"
        ).fetchone()["n"]
        archived_count = conn.execute(
            "SELECT COUNT(*) AS n FROM knowledge "
            "WHERE state = 'archived'"
        ).fetchone()["n"]
    finally:
        conn.close()
    assert total == 10
    assert archived_count == 5
    # The 5 lowest usage_count rows (i=0..4) are the ones archived;
    # all of them have archived_at populated.
    for i in range(5):
        p = fresh_kb.get_pattern(ids[i])
        assert p.state == "archived"
        assert p.archived_at is not None
    # And i=5..9 are still active.
    for i in range(5, 10):
        assert fresh_kb.get_pattern(ids[i]).state == "active"


def test_l32_prune_skips_pinned(fresh_kb):
    ids = [_seed_pattern(fresh_kb, f"s{i}") for i in range(6)]
    # Set usage_count and pin the lowest-usage one.
    conn = _kb_connect(fresh_kb.db_path)
    try:
        for i, pid in enumerate(ids):
            conn.execute(
                "UPDATE knowledge SET usage_count = ? WHERE id = ?",
                (i, pid),
            )
    finally:
        conn.close()
    fresh_kb.pin_pattern(ids[0])  # would normally be archived first
    fresh_kb.prune(max_entries=3)
    # Pinned row stayed active despite being the lowest-usage.
    assert fresh_kb.get_pattern(ids[0]).state == "active"
    assert fresh_kb.get_pattern(ids[0]).pinned is True


def test_l33_prune_does_not_re_archive(fresh_kb):
    """Already-archived rows shouldn't count toward the cap, and
    shouldn't be re-archived on a second prune call."""
    ids = [_seed_pattern(fresh_kb, f"s{i}") for i in range(4)]
    conn = _kb_connect(fresh_kb.db_path)
    try:
        # Pre-archive id[0] manually.
        conn.execute(
            "UPDATE knowledge SET state = 'archived', "
            "archived_at = '2026-05-05T00:00:00' WHERE id = ?",
            (ids[0],),
        )
        # Set usage_count so id[1] would be next-lowest.
        for i, pid in enumerate(ids):
            conn.execute(
                "UPDATE knowledge SET usage_count = ? WHERE id = ?",
                (i, pid),
            )
    finally:
        conn.close()
    # Active count is 3 (id[1..3]); cap at 2 -> archive 1.
    archived = fresh_kb.prune(max_entries=2)
    assert archived == 1
    # The pre-archived row's archived_at should not have been
    # overwritten.
    p = fresh_kb.get_pattern(ids[0])
    assert p.state == "archived"
    assert p.archived_at == "2026-05-05T00:00:00"


# ─────────────────────────────────────────────────────────────────
# KB auto_transition_lifecycle
# ─────────────────────────────────────────────────────────────────


def test_l41_active_to_stale(fresh_kb):
    pid = _seed_pattern(fresh_kb, "old low-usage")
    _force_age(fresh_kb.db_path, "knowledge", pid, days_old=45)
    out = fresh_kb.auto_transition_lifecycle(
        stale_after_days=30, archive_after_days=90,
    )
    assert out["stale"] >= 1
    assert fresh_kb.get_pattern(pid).state == "stale"


def test_l42_stale_to_archived(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    _force_age(fresh_kb.db_path, "knowledge", pid, days_old=120)
    # A row that's already past BOTH cutoffs (30d + 90d) collapses
    # active -> stale -> archived in a single sweep -- the two
    # UPDATEs run sequentially in one call and the second sees the
    # row in its newly-flipped 'stale' state. That's intentional:
    # very-old rows should not need a second nightly cron tick to
    # finally archive.
    out = fresh_kb.auto_transition_lifecycle(
        stale_after_days=30, archive_after_days=90,
    )
    assert out["archived"] >= 1
    p = fresh_kb.get_pattern(pid)
    assert p.state == "archived"
    assert p.archived_at is not None


def test_l43_pinned_never_transitions(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    fresh_kb.pin_pattern(pid)
    _force_age(fresh_kb.db_path, "knowledge", pid, days_old=365)
    out = fresh_kb.auto_transition_lifecycle(
        stale_after_days=30, archive_after_days=90,
    )
    assert out == {"stale": 0, "archived": 0}
    assert fresh_kb.get_pattern(pid).state == "active"


def test_l44_high_usage_active_not_marked_stale(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    _force_age(fresh_kb.db_path, "knowledge", pid, days_old=45)
    # Bump usage_count above the low threshold (<=1).
    conn = _kb_connect(fresh_kb.db_path)
    try:
        conn.execute(
            "UPDATE knowledge SET usage_count = 50 WHERE id = ?",
            (pid,),
        )
    finally:
        conn.close()
    fresh_kb.auto_transition_lifecycle(
        stale_after_days=30, archive_after_days=90,
    )
    assert fresh_kb.get_pattern(pid).state == "active"


def test_l45_recent_active_unaffected(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    # Don't backdate -- stays at "now".
    out = fresh_kb.auto_transition_lifecycle(
        stale_after_days=30, archive_after_days=90,
    )
    assert out == {"stale": 0, "archived": 0}
    assert fresh_kb.get_pattern(pid).state == "active"


# ─────────────────────────────────────────────────────────────────
# Memory episodic mirror
# ─────────────────────────────────────────────────────────────────


def test_l51_episodic_prune_archives(fresh_mem, monkeypatch):
    monkeypatch.setattr(config, "EPISODIC_MAX_PER_SCOPE", 3)
    for i in range(6):
        fresh_mem.store_episode(
            scope="agentX", trace_id="SEN-t",
            event_type="note", summary=f"ep{i}",
            relevance_score=(i + 1) / 10.0,
        )
    # store_episode calls prune internally; auto-archive should have
    # already trimmed the lowest-relevance ones.
    conn = _mem_connect(fresh_mem.db_path)
    try:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM episodic_memory"
        ).fetchone()["n"]
        archived = conn.execute(
            "SELECT COUNT(*) AS n FROM episodic_memory "
            "WHERE state = 'archived'"
        ).fetchone()["n"]
    finally:
        conn.close()
    # Nothing actually deleted.
    assert total == 6
    # Some rows archived to keep active count at the cap.
    assert archived >= 1


def test_l52_episodic_prune_skips_pinned(fresh_mem, monkeypatch):
    # Seed with a cap high enough that store_episode's auto-prune
    # is a no-op; pin AFTER seeding so the pinned id is known to be
    # active at prune time.
    monkeypatch.setattr(config, "EPISODIC_MAX_PER_SCOPE", 100)
    ids = []
    for i in range(5):
        eid = fresh_mem.store_episode(
            scope="agentY", trace_id="SEN-t",
            event_type="note", summary=f"ep{i}",
            relevance_score=0.1,  # all low-relevance, ties by created_at
        )
        ids.append(eid)
    fresh_mem.pin_episode(ids[0])  # the would-be-first-archived row
    # Now lower the cap and call prune explicitly.
    monkeypatch.setattr(config, "EPISODIC_MAX_PER_SCOPE", 2)
    fresh_mem.prune_episodes(scope="agentY")
    eps = fresh_mem.get_recent_episodes(
        scope="agentY", include_archived=True, limit=10,
    )
    by_id = {e.id: e for e in eps}
    assert by_id[ids[0]].state == "active"
    assert by_id[ids[0]].pinned is True


def test_l53_episodic_search_excludes_archived(fresh_mem):
    eid = fresh_mem.store_episode(
        scope="g", trace_id="SEN-t", event_type="note",
        summary="findme uniquetoken",
    )
    # Archive it manually.
    conn = _mem_connect(fresh_mem.db_path)
    try:
        conn.execute(
            "UPDATE episodic_memory SET state = 'archived' WHERE id = ?",
            (eid,),
        )
    finally:
        conn.close()
    assert fresh_mem.search_episodes("uniquetoken") == []
    matched = fresh_mem.search_episodes(
        "uniquetoken", include_archived=True,
    )
    assert any(m.id == eid for m in matched)


def test_l54_episodic_pin_unpin_restore_round_trip(fresh_mem):
    eid = fresh_mem.store_episode(
        scope="g", trace_id="SEN-t", event_type="note",
        summary="ep",
    )
    assert fresh_mem.pin_episode(eid) is True
    conn = _mem_connect(fresh_mem.db_path)
    try:
        # Force archive while pinned -- should still be allowed by
        # direct UPDATE (prune wouldn't but a manual move can).
        conn.execute(
            "UPDATE episodic_memory SET state = 'archived' WHERE id = ?",
            (eid,),
        )
    finally:
        conn.close()
    assert fresh_mem.unpin_episode(eid) is True
    assert fresh_mem.restore_episode(eid) is True
    eps = fresh_mem.get_recent_episodes(scope="g")
    assert any(e.id == eid for e in eps)


# ─────────────────────────────────────────────────────────────────
# Memory semantic mirror
# ─────────────────────────────────────────────────────────────────


def test_l55_semantic_prune_archives(fresh_mem, monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_MAX_ENTRIES", 2)
    fresh_mem.store_fact("k1", "v1", source="user_explicit")
    fresh_mem.store_fact("k2", "v2", source="user_explicit")
    # 3rd insert triggers prune -> archive.
    fresh_mem.store_fact("k3", "v3", source="auto_extracted")
    conn = _mem_connect(fresh_mem.db_path)
    try:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM semantic_memory"
        ).fetchone()["n"]
        archived = conn.execute(
            "SELECT COUNT(*) AS n FROM semantic_memory "
            "WHERE state = 'archived'"
        ).fetchone()["n"]
    finally:
        conn.close()
    assert total == 3  # nothing deleted
    assert archived >= 1


def test_l56_semantic_prune_skips_pinned(fresh_mem, monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_MAX_ENTRIES", 2)
    fresh_mem.store_fact(
        "low_conf", "v", source="auto_extracted", confidence=0.4,
    )
    fresh_mem.pin_fact("low_conf")
    fresh_mem.store_fact("ok1", "v", source="user_explicit")
    fresh_mem.store_fact("ok2", "v", source="user_explicit")
    # _prune_semantic should pick something OTHER than the pinned
    # low_conf row.
    fact = fresh_mem.get_fact("low_conf")
    assert fact is not None
    assert fact.state == "active"
    assert fact.pinned is True


def test_l57_list_facts_excludes_archived(fresh_mem):
    fresh_mem.store_fact("kA", "vA", source="user_explicit")
    fresh_mem.store_fact("kB", "vB", source="user_explicit")
    conn = _mem_connect(fresh_mem.db_path)
    try:
        conn.execute(
            "UPDATE semantic_memory SET state = 'archived' WHERE key = 'kB'"
        )
    finally:
        conn.close()
    keys = {f.key for f in fresh_mem.list_facts()}
    assert "kA" in keys
    assert "kB" not in keys
    keys_with = {
        f.key for f in fresh_mem.list_facts(include_archived=True)
    }
    assert "kB" in keys_with


def test_l58_semantic_pin_unpin_restore(fresh_mem):
    fresh_mem.store_fact("k", "v", source="user_explicit")
    assert fresh_mem.pin_fact("k") is True
    conn = _mem_connect(fresh_mem.db_path)
    try:
        conn.execute(
            "UPDATE semantic_memory SET state = 'archived' WHERE key = 'k'"
        )
    finally:
        conn.close()
    assert fresh_mem.unpin_fact("k") is True
    assert fresh_mem.restore_fact("k") is True
    f = fresh_mem.get_fact("k")
    assert f is not None
    assert f.state == "active"


def test_l59_store_fact_revives_archived(fresh_mem):
    """An explicit re-store of an archived key must revive it back
    to active state -- otherwise auto-archival could silently swallow
    a /remember command."""
    fresh_mem.store_fact("location", "Detroit", source="user_explicit")
    # Auto-walker archived it.
    conn = _mem_connect(fresh_mem.db_path)
    try:
        conn.execute(
            "UPDATE semantic_memory SET state = 'archived', "
            "archived_at = '2026-04-01T00:00:00' WHERE key = 'location'"
        )
    finally:
        conn.close()
    fresh_mem.store_fact("location", "Dearborn", source="user_explicit")
    f = fresh_mem.get_fact("location")
    assert f is not None
    assert f.state == "active"
    assert f.value == "Dearborn"


# ─────────────────────────────────────────────────────────────────
# Memory auto_transition_lifecycle
# ─────────────────────────────────────────────────────────────────


def test_l61_episodic_active_low_rel_old_to_stale(fresh_mem):
    eid = fresh_mem.store_episode(
        scope="g", trace_id="SEN-t", event_type="note",
        summary="old", relevance_score=0.3,
    )
    _force_age(fresh_mem.db_path, "episodic_memory", eid, days_old=45)
    out = fresh_mem.auto_transition_lifecycle(
        stale_after_days=30, archive_after_days=90,
    )
    assert out["episodic"]["stale"] >= 1
    eps = fresh_mem.get_recent_episodes(
        scope="g", include_archived=True, limit=10,
    )
    rec = next((e for e in eps if e.id == eid), None)
    assert rec is not None and rec.state == "stale"
    # Push past archive cutoff.
    _force_age(fresh_mem.db_path, "episodic_memory", eid, days_old=120)
    out2 = fresh_mem.auto_transition_lifecycle(
        stale_after_days=30, archive_after_days=90,
    )
    assert out2["episodic"]["archived"] >= 1


def test_l62_semantic_low_conf_old_to_stale(fresh_mem):
    fresh_mem.store_fact(
        "k", "v", source="auto_extracted", confidence=0.5,
    )
    fid = fresh_mem.get_fact("k").id
    _force_age(fresh_mem.db_path, "semantic_memory", fid, days_old=45)
    out = fresh_mem.auto_transition_lifecycle(
        stale_after_days=30, archive_after_days=90,
    )
    assert out["semantic"]["stale"] >= 1


def test_l63_pinned_memory_never_transitions(fresh_mem):
    eid = fresh_mem.store_episode(
        scope="g", trace_id="SEN-t", event_type="note",
        summary="forever", relevance_score=0.1,
    )
    fresh_mem.pin_episode(eid)
    _force_age(fresh_mem.db_path, "episodic_memory", eid, days_old=999)
    out = fresh_mem.auto_transition_lifecycle(
        stale_after_days=30, archive_after_days=90,
    )
    assert out["episodic"] == {"stale": 0, "archived": 0}
    fresh_mem.store_fact(
        "pinned_key", "v",
        source="auto_extracted", confidence=0.4,
    )
    fresh_mem.pin_fact("pinned_key")
    fid = fresh_mem.get_fact("pinned_key").id
    _force_age(fresh_mem.db_path, "semantic_memory", fid, days_old=999)
    out2 = fresh_mem.auto_transition_lifecycle(
        stale_after_days=30, archive_after_days=90,
    )
    assert out2["semantic"] == {"stale": 0, "archived": 0}
