"""Phase 15g -- KB preload script integrity + insertion tests.

The Phase 15e regression destroyed 28 post-Phase-14b patterns and
left the KB with stale May-4 Phase-9 examples that don't match the
current codebase shape. tools/preload_kb.py seeds 39 hand-curated
patterns covering the major surfaces (telegram bot, KB, memory,
SQLite, skills, tests, internal handlers, brain/Claude/Qwen,
math/util/config) so the bot can be productive from a cold start
instead of organically rebuilding context.

This file enforces three invariants:

  1. Every PATTERNS entry is structurally valid (required keys,
     non-empty fields, parseable recipe, valid diff body).
  2. The preload() function actually inserts every pattern into a
     fresh KB and pins each one (Phase 15a guarantee).
  3. The shadow recipes + agreement scores landed -- /kb planning
     gets non-zero data day-one.

Coverage:
  Structural integrity:
    G01 -- PATTERNS list non-empty (>= 30)
    G02 -- every entry has all 6 required keys
    G03 -- problem_summaries are unique
    G04 -- tags is a non-empty list of strings
    G05 -- explanation is substantive (>= 100 chars)
    G06 -- solution_pattern parses to >= 2 STEP-N blocks
    G07 -- shadow_recipe parses to >= 1 STEP-N block
    G08 -- solution_code passes _is_real_solution gate (Phase 15d)

  Insertion behavior (against fresh tmp KB):
    G11 -- preload() inserts every pattern
    G12 -- every inserted row is pinned (state='active', pinned=1)
    G13 -- every inserted row has shadow data populated
    G14 -- agreement scores are in [0.0, 1.0]
    G15 -- re-running preload() is a no-op (idempotent)
    G16 -- --replace flag would NOT touch the test (smoke check)

  Coverage breadth (regression guard against narrowing too far):
    G21 -- at least 5 patterns mention 'telegram'
    G22 -- at least 3 patterns mention 'sqlite' or 'schema'
    G23 -- at least 2 patterns mention 'test' or 'pytest'
    G24 -- at least 2 patterns mention 'memory'
    G25 -- at least 2 patterns mention 'skill' or 'agent'
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
from core.knowledge_base import KnowledgeBase
from core.qwen_agent import _parse_recipe_steps
from skills.code_assist import _is_real_solution
from tools.preload_kb import PATTERNS, preload


# ─────────────────────────────────────────────────────────────────
# fixtures
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
    db = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db)
    _stub_embedder(monkeypatch)
    return KnowledgeBase(db_path=db)


# ─────────────────────────────────────────────────────────────────
# Structural integrity
# ─────────────────────────────────────────────────────────────────


def test_g01_patterns_list_non_empty():
    assert len(PATTERNS) >= 30, (
        f"PATTERNS shrunk to {len(PATTERNS)} -- regression?"
    )


def test_g02_every_entry_has_required_keys():
    required = (
        "tags", "problem_summary", "solution_pattern",
        "solution_code", "explanation", "shadow_recipe",
    )
    for i, p in enumerate(PATTERNS):
        missing = [k for k in required if k not in p]
        assert not missing, (
            f"pattern {i} ({p.get('problem_summary', '?')[:60]}) "
            f"missing keys: {missing}"
        )


def test_g03_problem_summaries_unique():
    seen: set[str] = set()
    dupes: list[str] = []
    for p in PATTERNS:
        s = p["problem_summary"]
        if s in seen:
            dupes.append(s)
        seen.add(s)
    assert not dupes, f"duplicate problem_summaries: {dupes}"


def test_g04_tags_is_non_empty_list_of_str():
    for i, p in enumerate(PATTERNS):
        tags = p["tags"]
        assert isinstance(tags, list), (
            f"pattern {i}: tags is {type(tags).__name__}, not list"
        )
        assert tags, f"pattern {i}: tags is empty"
        for t in tags:
            assert isinstance(t, str) and t.strip(), (
                f"pattern {i}: bad tag {t!r}"
            )


def test_g05_explanation_is_substantive():
    """A useful explanation teaches the pattern. Below 100 chars
    means it's just a summary, not actionable guidance."""
    for i, p in enumerate(PATTERNS):
        e = p["explanation"]
        assert len(e) >= 100, (
            f"pattern {i} ({p['problem_summary'][:50]}): "
            f"explanation only {len(e)} chars -- needs more substance"
        )


def test_g06_solution_pattern_parses():
    """The recipe must produce >= 2 STEP-N blocks (verifier + done at
    minimum), or the future Claude pre-teach can't use it as a few-
    shot example."""
    for i, p in enumerate(PATTERNS):
        steps = _parse_recipe_steps(p["solution_pattern"])
        assert len(steps) >= 2, (
            f"pattern {i} ({p['problem_summary'][:50]}): "
            f"only {len(steps)} STEP blocks parsed"
        )


def test_g07_shadow_recipe_parses():
    """Shadow recipes should parse to at least 1 STEP block too --
    that's what makes the agreement score non-trivial."""
    for i, p in enumerate(PATTERNS):
        steps = _parse_recipe_steps(p["shadow_recipe"])
        assert len(steps) >= 1, (
            f"pattern {i}: shadow recipe parsed to 0 steps"
        )


def test_g08_solution_code_passes_quality_gate():
    """The Phase 15d-bugfix-2 cleanup_low_quality_patterns will run
    on every bot startup. If our seeded patterns fail the gate, they
    get archived on the next restart -- defeating the whole point."""
    for i, p in enumerate(PATTERNS):
        ok = _is_real_solution(p["solution_code"])
        assert ok, (
            f"pattern {i} ({p['problem_summary'][:50]}): "
            f"solution_code FAILS _is_real_solution -- "
            f"would be archived on next bot restart"
        )


# ─────────────────────────────────────────────────────────────────
# Insertion behavior
# ─────────────────────────────────────────────────────────────────


def test_g11_preload_inserts_every_pattern(fresh_kb):
    result = preload(kb=fresh_kb)
    assert len(result["inserted"]) == len(PATTERNS)
    assert result["skipped_count"] == 0
    # Verify rows actually landed in the DB
    import sqlite3
    conn = sqlite3.connect(str(fresh_kb.db_path))
    n = conn.execute(
        "SELECT COUNT(*) FROM knowledge WHERE category = 'pattern'"
    ).fetchone()[0]
    conn.close()
    assert n == len(PATTERNS)


def test_g12_every_inserted_row_is_pinned(fresh_kb):
    preload(kb=fresh_kb)
    import sqlite3
    conn = sqlite3.connect(str(fresh_kb.db_path))
    rows = conn.execute(
        "SELECT id, state, pinned FROM knowledge "
        "WHERE category = 'pattern'"
    ).fetchall()
    conn.close()
    for r in rows:
        rid, state, pinned = r
        assert state == "active", f"row {rid}: state is {state}"
        assert pinned == 1, f"row {rid}: pinned is {pinned}"


def test_g13_shadow_data_populated(fresh_kb):
    preload(kb=fresh_kb)
    import sqlite3
    conn = sqlite3.connect(str(fresh_kb.db_path))
    rows = conn.execute(
        "SELECT qwen_plan_recipe IS NOT NULL, "
        "qwen_plan_agreement IS NOT NULL FROM knowledge "
        "WHERE category = 'pattern'"
    ).fetchall()
    conn.close()
    for has_recipe, has_agreement in rows:
        assert has_recipe, "shadow recipe NULL on a preloaded row"
        assert has_agreement, "agreement NULL on a preloaded row"


def test_g14_agreement_scores_in_unit_interval(fresh_kb):
    preload(kb=fresh_kb)
    import sqlite3
    conn = sqlite3.connect(str(fresh_kb.db_path))
    scores = [
        r[0] for r in conn.execute(
            "SELECT qwen_plan_agreement FROM knowledge "
            "WHERE category = 'pattern'"
        ).fetchall()
    ]
    conn.close()
    for s in scores:
        assert s is not None
        assert 0.0 <= s <= 1.0, f"agreement {s} outside [0, 1]"


def test_g15_preload_is_idempotent(fresh_kb):
    """Re-running preload on the same DB skips already-present
    patterns by problem_summary match."""
    first = preload(kb=fresh_kb)
    second = preload(kb=fresh_kb)
    assert len(first["inserted"]) == len(PATTERNS)
    assert len(second["inserted"]) == 0
    assert second["skipped_count"] == len(PATTERNS)


def test_g16_replace_flag_forces_reinsertion(fresh_kb):
    """--replace mode re-inserts even when problem_summary matches.
    Used for forcing re-seed when canonical patterns evolve."""
    preload(kb=fresh_kb)
    second = preload(kb=fresh_kb, replace=True)
    assert len(second["inserted"]) == len(PATTERNS)


# ─────────────────────────────────────────────────────────────────
# Coverage breadth (regression guard)
# ─────────────────────────────────────────────────────────────────


def _count_with_keyword(keyword: str) -> int:
    """Count patterns where any tag OR the problem_summary mentions
    the keyword (case-insensitive)."""
    kl = keyword.lower()
    n = 0
    for p in PATTERNS:
        if kl in p["problem_summary"].lower():
            n += 1
            continue
        if any(kl in t.lower() for t in p["tags"]):
            n += 1
    return n


def test_g21_telegram_coverage():
    n = _count_with_keyword("telegram")
    assert n >= 5, f"only {n} telegram-related patterns (want >= 5)"


def test_g22_schema_coverage():
    n = _count_with_keyword("sqlite") + _count_with_keyword("schema")
    assert n >= 3, f"only {n} schema-related patterns (want >= 3)"


def test_g23_test_coverage():
    n = _count_with_keyword("pytest") + _count_with_keyword("test")
    assert n >= 2, f"only {n} test-related patterns (want >= 2)"


def test_g24_memory_coverage():
    n = _count_with_keyword("memory")
    assert n >= 2, f"only {n} memory-related patterns (want >= 2)"


def test_g25_skill_agent_coverage():
    n = _count_with_keyword("skill") + _count_with_keyword("agent")
    assert n >= 2, f"only {n} skill/agent patterns (want >= 2)"
