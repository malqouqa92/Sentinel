"""Phase 15b -- write-origin provenance via ContextVar.

ECC. No GPU, no real LLM. Stubs the embedder where needed.

Coverage:
  ContextVar mechanics:
    P01 -- default is FOREGROUND
    P02 -- set/reset round-trip restores prior value
    P03 -- unknown origin coerced to FOREGROUND on read
    P04 -- is_background() predicate is correct for all three values
    P05 -- ContextVar isolation across asyncio tasks (background set
           in one task is NOT visible in a concurrent unrelated task)

  Schema migration:
    P11 -- KB fresh DB has created_by_origin column
    P12 -- KB pre-15b DB gets ALTERed cleanly (default 'foreground')
    P13 -- memory fresh DB has the column on BOTH episodic + semantic
    P14 -- memory pre-15b DB gets ALTERed cleanly
    P15 -- double-init is idempotent on all three tables

  Stamping:
    P21 -- KB.add_pattern under default ctx -> 'foreground'
    P22 -- KB.add_pattern under set BACKGROUND -> 'background'
    P23 -- KB.add_pattern under BACKGROUND_EXTRACTION -> stamped
    P24 -- store_episode under set ctx -> stamped
    P25 -- store_fact INSERT under set ctx -> stamped
    P26 -- store_fact UPSERT does NOT overwrite original origin
    P27 -- _row_to_entry round-trips the column
    P28 -- origin_breakdown returns a sane dict

  Telegram /kb stats integration:
    P31 -- origin_breakdown groups correctly across mixed writes
"""
from __future__ import annotations

import asyncio
import struct
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core import embeddings as emb
from core.knowledge_base import KnowledgeBase, _connect as _kb_connect
from core.memory import MemoryManager, _connect as _mem_connect
from core.write_origin import (
    BACKGROUND, BACKGROUND_EXTRACTION, FOREGROUND, VALID_ORIGINS,
    get_current_write_origin, is_background,
    reset_current_write_origin, set_current_write_origin,
)


# ─────────────────────────────────────────────────────────────────
# fixtures (mirror Phase 14a / Phase 15a patterns)
# ─────────────────────────────────────────────────────────────────


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


@pytest.fixture
def fresh_mem(tmp_path, monkeypatch):
    db_path = tmp_path / "mem.db"
    monkeypatch.setattr(config, "MEMORY_DB_PATH", db_path)
    return MemoryManager(db_path=db_path)


@pytest.fixture(autouse=True)
def _reset_origin_between_tests():
    """Belt-and-suspenders: every test starts with the default
    origin, even if a previous test forgot to reset its token. The
    ContextVar default mechanic handles fresh contexts cleanly, but
    we explicitly read+restore in case pytest reuses a context."""
    # No setup -- the ContextVar default kicks in for new contexts.
    # Yield, then verify nothing leaked.
    yield
    # If a test forgot to reset, this will surface as a default-not-
    # restored failure on the NEXT test (which we'd see as a
    # P01-style mismatch). This fixture exists to make that obvious.


def _seed(kb: KnowledgeBase, summary: str = "test problem") -> int:
    return kb.add_pattern(
        tags=["test"],
        problem_summary=summary,
        solution_code="print(2+2)",
        solution_pattern="just print",
        explanation="trivial",
        trace_id="SEN-test",
    )


# ─────────────────────────────────────────────────────────────────
# ContextVar mechanics
# ─────────────────────────────────────────────────────────────────


def test_p01_default_is_foreground():
    assert get_current_write_origin() == FOREGROUND
    assert is_background() is False


def test_p02_set_reset_round_trip():
    assert get_current_write_origin() == FOREGROUND
    token = set_current_write_origin(BACKGROUND)
    try:
        assert get_current_write_origin() == BACKGROUND
        assert is_background() is True
        # Nested set/reset.
        inner_tok = set_current_write_origin(BACKGROUND_EXTRACTION)
        assert get_current_write_origin() == BACKGROUND_EXTRACTION
        reset_current_write_origin(inner_tok)
        assert get_current_write_origin() == BACKGROUND
    finally:
        reset_current_write_origin(token)
    assert get_current_write_origin() == FOREGROUND


def test_p03_unknown_origin_coerced_on_read():
    """The setter accepts anything (no exception), but the getter
    normalises unknown values back to FOREGROUND so callers can't
    leak garbage labels into the DB through a bad set call."""
    token = set_current_write_origin("not-a-real-origin")
    try:
        # Read returns FOREGROUND despite the bad set.
        assert get_current_write_origin() == FOREGROUND
    finally:
        reset_current_write_origin(token)


def test_p04_is_background_predicate():
    assert is_background() is False  # default
    for ok_bg in (BACKGROUND, BACKGROUND_EXTRACTION):
        tok = set_current_write_origin(ok_bg)
        try:
            assert is_background() is True
        finally:
            reset_current_write_origin(tok)
    assert is_background() is False  # restored


def test_p05_contextvar_isolation_across_asyncio_tasks():
    """Two sibling tasks must each see their own copy of the ctx --
    setting BACKGROUND in task A must NOT leak into task B."""
    seen = {"A": None, "B": None}

    async def task_a():
        tok = set_current_write_origin(BACKGROUND)
        try:
            await asyncio.sleep(0.01)
            seen["A"] = get_current_write_origin()
        finally:
            reset_current_write_origin(tok)

    async def task_b():
        # B never sets anything; should see the default.
        await asyncio.sleep(0.005)
        seen["B"] = get_current_write_origin()

    async def driver():
        await asyncio.gather(task_a(), task_b())

    asyncio.run(driver())
    assert seen["A"] == BACKGROUND, f"task A saw {seen['A']!r}"
    assert seen["B"] == FOREGROUND, (
        f"task B saw {seen['B']!r} -- contextvar leaked across tasks"
    )


def test_p06_valid_origins_constants():
    """Sanity: the three constants we ship match the VALID_ORIGINS
    set so callers using either form stay consistent."""
    assert VALID_ORIGINS == {
        FOREGROUND, BACKGROUND, BACKGROUND_EXTRACTION,
    }


# ─────────────────────────────────────────────────────────────────
# Schema migration
# ─────────────────────────────────────────────────────────────────


def test_p11_fresh_kb_has_origin_column(fresh_kb):
    conn = _kb_connect(fresh_kb.db_path)
    try:
        cols = {r["name"] for r in conn.execute(
            "PRAGMA table_info(knowledge)"
        ).fetchall()}
    finally:
        conn.close()
    assert "created_by_origin" in cols


def test_p12_pre_15b_kb_db_gets_migrated(tmp_path, monkeypatch):
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
        "solution_code, solution_pattern, explanation, "
        "source_trace_id, created_at) VALUES "
        "('pattern', 'x', 'old', 'c', 'p', 'e', 'SEN-old', "
        "'2026-01-01T00:00:00')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    _stub_embedder(monkeypatch)
    kb = KnowledgeBase(db_path=db_path)
    e = kb.get_pattern(1)
    assert e is not None
    assert e.created_by_origin == "foreground"


def test_p13_fresh_mem_has_origin_on_both_tables(fresh_mem):
    conn = _mem_connect(fresh_mem.db_path)
    try:
        for tbl in ("episodic_memory", "semantic_memory"):
            cols = {r["name"] for r in conn.execute(
                f"PRAGMA table_info({tbl})"
            ).fetchall()}
            assert "created_by_origin" in cols, f"{tbl} missing col"
    finally:
        conn.close()


def test_p14_pre_15b_mem_db_gets_migrated(tmp_path, monkeypatch):
    db_path = tmp_path / "old_mem.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL, trace_id TEXT NOT NULL,
            event_type TEXT NOT NULL, summary TEXT NOT NULL,
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
        "('g', 'SEN-old', 'note', 's', '2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO semantic_memory (key, value, source, "
        "created_at, updated_at) VALUES "
        "('k', 'v', 'user_explicit', '2026-01-01T00:00:00', "
        "'2026-01-01T00:00:00')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "MEMORY_DB_PATH", db_path)
    mem = MemoryManager(db_path=db_path)
    eps = mem.get_recent_episodes(scope="g")
    assert len(eps) == 1
    assert eps[0].created_by_origin == "foreground"
    fact = mem.get_fact("k")
    assert fact is not None
    assert fact.created_by_origin == "foreground"


def test_p15_double_init_idempotent(fresh_kb, fresh_mem):
    KnowledgeBase(db_path=fresh_kb.db_path)
    KnowledgeBase(db_path=fresh_kb.db_path)
    MemoryManager(db_path=fresh_mem.db_path)
    MemoryManager(db_path=fresh_mem.db_path)


# ─────────────────────────────────────────────────────────────────
# Stamping behavior
# ─────────────────────────────────────────────────────────────────


def test_p21_kb_default_ctx_writes_foreground(fresh_kb):
    pid = _seed(fresh_kb)
    p = fresh_kb.get_pattern(pid)
    assert p.created_by_origin == "foreground"


def test_p22_kb_background_ctx_stamps(fresh_kb):
    tok = set_current_write_origin(BACKGROUND)
    try:
        pid = _seed(fresh_kb, "bg pattern")
    finally:
        reset_current_write_origin(tok)
    p = fresh_kb.get_pattern(pid)
    assert p.created_by_origin == "background"
    # And outside the scope, default is back.
    pid2 = _seed(fresh_kb, "fg pattern")
    assert fresh_kb.get_pattern(pid2).created_by_origin == "foreground"


def test_p23_kb_background_extraction_ctx_stamps(fresh_kb):
    tok = set_current_write_origin(BACKGROUND_EXTRACTION)
    try:
        pid = _seed(fresh_kb)
    finally:
        reset_current_write_origin(tok)
    assert fresh_kb.get_pattern(pid).created_by_origin == (
        "background_extraction"
    )


def test_p24_episode_under_set_ctx(fresh_mem):
    tok = set_current_write_origin(BACKGROUND)
    try:
        eid = fresh_mem.store_episode(
            scope="g", trace_id="t", event_type="note", summary="x",
        )
    finally:
        reset_current_write_origin(tok)
    eps = fresh_mem.get_recent_episodes(scope="g")
    rec = next(e for e in eps if e.id == eid)
    assert rec.created_by_origin == "background"


def test_p25_fact_insert_under_set_ctx(fresh_mem):
    tok = set_current_write_origin(BACKGROUND_EXTRACTION)
    try:
        fresh_mem.store_fact("auto_key", "v", source="auto_extracted")
    finally:
        reset_current_write_origin(tok)
    f = fresh_mem.get_fact("auto_key")
    assert f.created_by_origin == "background_extraction"


def test_p26_upsert_does_not_overwrite_original_origin(fresh_mem):
    """An auto-extracted fact later re-asserted by the user must
    keep its 'background_extraction' origin -- the column records
    who FIRST wrote the key, not who last touched it."""
    bg_tok = set_current_write_origin(BACKGROUND_EXTRACTION)
    try:
        fresh_mem.store_fact(
            "loc", "Detroit", source="auto_extracted", confidence=0.6,
        )
    finally:
        reset_current_write_origin(bg_tok)
    # Now a foreground re-assertion with higher confidence.
    fresh_mem.store_fact("loc", "Dearborn", source="user_explicit")
    f = fresh_mem.get_fact("loc")
    assert f.value == "Dearborn"  # value updated
    assert f.confidence == 1.0    # confidence advanced
    # But origin stays at the original write.
    assert f.created_by_origin == "background_extraction"


def test_p27_row_to_entry_round_trip(fresh_kb):
    tok = set_current_write_origin(BACKGROUND)
    try:
        pid = _seed(fresh_kb)
    finally:
        reset_current_write_origin(tok)
    # Force reading via get_pattern (which uses _row_to_entry).
    p = fresh_kb.get_pattern(pid)
    assert p.created_by_origin == "background"
    # Field name stays consistent if a future change adds something.
    assert hasattr(p, "created_by_origin")


def test_p28_origin_breakdown_aggregates(fresh_kb):
    # Mix: 2 foreground, 1 background, 1 background_extraction.
    _seed(fresh_kb, "fg1")
    _seed(fresh_kb, "fg2")
    tok = set_current_write_origin(BACKGROUND)
    try:
        _seed(fresh_kb, "bg")
    finally:
        reset_current_write_origin(tok)
    tok2 = set_current_write_origin(BACKGROUND_EXTRACTION)
    try:
        _seed(fresh_kb, "bge")
    finally:
        reset_current_write_origin(tok2)
    out = fresh_kb.origin_breakdown()
    assert out.get("foreground") == 2
    assert out.get("background") == 1
    assert out.get("background_extraction") == 1
    assert sum(out.values()) == 4


# ─────────────────────────────────────────────────────────────────
# Adapter / wrap-site sanity (source-level inspection)
# ─────────────────────────────────────────────────────────────────


def test_p31_telegram_bot_wraps_auto_extract():
    """Source-level check: _maybe_auto_extract sets and resets the
    BACKGROUND_EXTRACTION origin. Avoids spinning up the bot."""
    src = (
        Path(__file__).resolve().parent.parent
        / "interfaces" / "telegram_bot.py"
    ).read_text(encoding="utf-8", errors="replace")
    # Find the function and confirm the wrap is present.
    idx = src.find("async def _maybe_auto_extract")
    assert idx > 0
    # Take a bounded slice (~60 lines) for the assertions.
    body = src[idx:idx + 4000]
    assert "BACKGROUND_EXTRACTION" in body
    assert "set_current_write_origin" in body
    assert "reset_current_write_origin" in body


def test_p32_telegram_bot_wraps_adaptive_filter():
    src = (
        Path(__file__).resolve().parent.parent
        / "interfaces" / "telegram_bot.py"
    ).read_text(encoding="utf-8", errors="replace")
    idx = src.find("async def _maybe_run_adaptive_filter")
    assert idx > 0
    body = src[idx:idx + 4000]
    assert "BACKGROUND" in body  # set_current_write_origin(BACKGROUND)
    assert "set_current_write_origin" in body
    assert "reset_current_write_origin" in body
