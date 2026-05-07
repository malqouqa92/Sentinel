"""Phase 16 Batch C -- few-shot exclusion in KB retrieval ECC."""
from __future__ import annotations

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
def kb_with_three(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    _stub_embedder(monkeypatch)
    kb = KnowledgeBase(db_path=db_path)
    ids = []
    for i, p in enumerate([
        "alpha bravo charlie",
        "alpha delta echo",
        "alpha foxtrot golf",
    ]):
        pid = kb.add_pattern(
            tags=["t"], problem_summary=p,
            solution_code="(diff)",
            solution_pattern=f"STEP 1: done summary=\"v{i}\"",
            explanation=f"explanation {i}",
            trace_id="SEN-test",
        )
        ids.append(pid)
    return kb, ids


def test_x01_no_exclusion_returns_all(kb_with_three):
    kb, ids = kb_with_three
    rs = kb.search("alpha", max_results=10, hybrid=False)
    found = sorted(r.id for r in rs)
    assert sorted(ids) == found


def test_x02_excludes_single_id(kb_with_three):
    kb, ids = kb_with_three
    rs = kb.search(
        "alpha", max_results=10, hybrid=False,
        exclude_pattern_ids=[ids[1]],
    )
    found = [r.id for r in rs]
    assert ids[1] not in found
    assert ids[0] in found and ids[2] in found


def test_x03_excludes_multiple_ids(kb_with_three):
    kb, ids = kb_with_three
    rs = kb.search(
        "alpha", max_results=10, hybrid=False,
        exclude_pattern_ids=[ids[0], ids[2]],
    )
    assert [r.id for r in rs] == [ids[1]]


def test_x04_empty_or_none_is_noop(kb_with_three):
    kb, ids = kb_with_three
    rs_a = kb.search(
        "alpha", max_results=10, hybrid=False, exclude_pattern_ids=None,
    )
    rs_b = kb.search(
        "alpha", max_results=10, hybrid=False, exclude_pattern_ids=[],
    )
    assert sorted(r.id for r in rs_a) == sorted(ids)
    assert sorted(r.id for r in rs_b) == sorted(ids)


def test_x05_nonexistent_ids_safe(kb_with_three):
    kb, ids = kb_with_three
    rs = kb.search(
        "alpha", max_results=10, hybrid=False,
        exclude_pattern_ids=[999, 998],
    )
    assert sorted(r.id for r in rs) == sorted(ids)


def test_x06_malformed_ids_no_crash(kb_with_three):
    kb, _ids = kb_with_three
    rs = kb.search(
        "alpha", max_results=10, hybrid=False,
        exclude_pattern_ids=["bad", None],
    )
    # No crash; some results returned (exclusion ignored on bad input)
    assert len(rs) >= 1


def test_x10_get_context_forwards_exclude(kb_with_three):
    kb, ids = kb_with_three
    text = kb.get_context_for_prompt(
        "alpha", max_chars=10000, exclude_pattern_ids=[ids[1]],
    )
    assert "delta echo" not in text


def test_x11_get_context_no_exclude(kb_with_three):
    kb, _ids = kb_with_three
    text = kb.get_context_for_prompt("alpha", max_chars=10000)
    assert "delta echo" in text
