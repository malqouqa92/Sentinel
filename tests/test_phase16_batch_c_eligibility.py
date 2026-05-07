"""Phase 16 Batch C -- skip-eligibility gate ECC tests."""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from core import config, embeddings as emb
from core.knowledge_base import KnowledgeBase


def _stub_embedder(monkeypatch):
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


def _seed(
    kb, *, problem="p", passes=0, attempts=0, pinned=0,
    state="active", needs_reteach=0, last_verified=None, agreement=None,
):
    pid = kb.add_pattern(
        tags=["t"], problem_summary=problem,
        solution_code="(diff)", solution_pattern="STEP 1: done",
        explanation="x", trace_id="SEN-test",
        qwen_plan_agreement=agreement,
    )
    conn = sqlite3.connect(kb.db_path)
    conn.execute(
        "UPDATE knowledge SET solo_passes=?, solo_attempts=?, "
        "pinned=?, state=?, needs_reteach=?, last_verified_at=? "
        "WHERE id=?",
        (passes, attempts, pinned, state, needs_reteach,
         last_verified, pid),
    )
    conn.commit()
    conn.close()
    return pid


def _now(offset_days=0):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc)
            + timedelta(days=offset_days)).isoformat()


def test_e01_missing_pattern_rejects(fresh_kb):
    eligible, reason = fresh_kb.is_skip_eligible(pattern_id=999)
    assert not eligible
    assert reason == fresh_kb.SKIP_REASON_MISSING


def test_e02_limitation_rejects(fresh_kb):
    pid = fresh_kb.add_limitation(
        tags=["x"], problem_summary="hopeless",
        explanation="failed", trace_id="SEN-t",
    )
    e, r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert not e
    assert r == fresh_kb.SKIP_REASON_NOT_PATTERN


def test_e03_archived_rejects(fresh_kb):
    pid = _seed(fresh_kb, passes=5, attempts=5, pinned=1, last_verified=_now())
    conn = sqlite3.connect(fresh_kb.db_path)
    conn.execute("UPDATE knowledge SET state='archived' WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    e, r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert not e
    assert r == fresh_kb.SKIP_REASON_ARCHIVED


def test_e04_needs_reteach_rejects(fresh_kb):
    pid = _seed(
        fresh_kb, passes=5, attempts=5, pinned=1, needs_reteach=1,
        last_verified=_now(),
    )
    e, r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert not e
    assert r == fresh_kb.SKIP_REASON_NEEDS_RETEACH


def test_e10_pinned_eligible_despite_low_stats(fresh_kb):
    pid = _seed(
        fresh_kb, passes=1, attempts=1, pinned=1, last_verified=_now(),
    )
    e, r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert e
    assert r == fresh_kb.SKIP_REASON_OK_PINNED


def test_e11_pinned_eligible_despite_stale(fresh_kb):
    pid = _seed(
        fresh_kb, passes=5, attempts=5, pinned=1,
        last_verified=_now(offset_days=-365),
    )
    e, _r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert e


def test_e20_low_passes_rejects(fresh_kb):
    pid = _seed(fresh_kb, passes=2, attempts=2, last_verified=_now())
    e, r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert not e
    assert r == fresh_kb.SKIP_REASON_LOW_PASSES


def test_e21_imperfect_rate_rejects(fresh_kb):
    pid = _seed(fresh_kb, passes=4, attempts=5, last_verified=_now())
    e, r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert not e
    assert r == fresh_kb.SKIP_REASON_IMPERFECT_RATE


def test_e22_null_last_verified_rejects(fresh_kb):
    pid = _seed(fresh_kb, passes=3, attempts=3, last_verified=None)
    e, r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert not e
    assert r == fresh_kb.SKIP_REASON_STALE


def test_e23_old_last_verified_rejects(fresh_kb):
    pid = _seed(
        fresh_kb, passes=3, attempts=3,
        last_verified=_now(offset_days=-60),
    )
    e, r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert not e
    assert r == fresh_kb.SKIP_REASON_STALE


def test_e24_low_agreement_rejects(fresh_kb):
    pid = _seed(
        fresh_kb, passes=3, attempts=3, last_verified=_now(),
        agreement=0.3,
    )
    e, r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert not e
    assert r == fresh_kb.SKIP_REASON_LOW_AGREEMENT


def test_e30_at_threshold_eligible(fresh_kb):
    pid = _seed(
        fresh_kb, passes=3, attempts=3, last_verified=_now(),
        agreement=None,
    )
    e, r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert e
    assert r == fresh_kb.SKIP_REASON_OK_TRUSTED


def test_e31_well_above_threshold_eligible(fresh_kb):
    pid = _seed(
        fresh_kb, passes=10, attempts=10, last_verified=_now(),
        agreement=1.0,
    )
    e, _r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert e


def test_e32_null_agreement_does_not_block(fresh_kb):
    pid = _seed(
        fresh_kb, passes=5, attempts=5, last_verified=_now(),
        agreement=None,
    )
    e, _r = fresh_kb.is_skip_eligible(pattern_id=pid)
    assert e


def test_e40_row_dict_arg_avoids_db_fetch(fresh_kb):
    row = {
        "id": 9999, "category": "pattern", "state": "active",
        "pinned": 1, "needs_reteach": 0,
        "solo_attempts": 5, "solo_passes": 5,
        "last_verified_at": _now(), "qwen_plan_agreement": 0.9,
    }
    e, r = fresh_kb.is_skip_eligible(row=row)
    assert e
    assert r == fresh_kb.SKIP_REASON_OK_PINNED


def test_e42_reason_tokens_are_stable(fresh_kb):
    K = fresh_kb
    assert K.SKIP_REASON_OK_PINNED == "OK_PINNED"
    assert K.SKIP_REASON_OK_TRUSTED == "OK_TRUSTED"
    assert K.SKIP_REASON_MISSING == "MISSING"
    assert K.SKIP_REASON_NOT_PATTERN == "LIMITATION"
    assert K.SKIP_REASON_ARCHIVED == "ARCHIVED"
    assert K.SKIP_REASON_NEEDS_RETEACH == "NEEDS_RETEACH"
    assert K.SKIP_REASON_LOW_PASSES == "LOW_PASSES"
    assert K.SKIP_REASON_IMPERFECT_RATE == "IMPERFECT_RATE"
    assert K.SKIP_REASON_STALE == "STALE"
    assert K.SKIP_REASON_LOW_AGREEMENT == "LOW_AGREEMENT"
