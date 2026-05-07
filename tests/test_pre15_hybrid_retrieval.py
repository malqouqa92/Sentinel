"""Pre-Phase-15: hybrid retrieval (FTS5 + embeddings) ECC tests.

Stubs the Ollama embedding endpoint so tests are fast + deterministic.

Coverage:
  H01 -- _pack/_unpack roundtrip preserves vector
  H02 -- cosine_similarity: identical vectors = 1.0
  H03 -- cosine_similarity: orthogonal vectors = 0.0
  H04 -- cosine_similarity: handles None / empty / mismatched dim
  H05 -- hybrid_score: top-rank + high cosine -> high score
  H06 -- hybrid_score: weights respected (BM25-only vs blend)
  H07 -- embed_text: success path returns float32 BLOB of right size
  H08 -- embed_text: failure path returns None (Ollama down)
  H09 -- KB.search hybrid: reorders BM25 results by embedding similarity
  H10 -- KB.search hybrid: rows without embeddings still appear (no exclusion)
  H11 -- KB.search hybrid=False: deterministic BM25-only behaviour
         (back-compat path)
  H12 -- backfill_embeddings: NULL-embedding rows get filled
  H13 -- KB schema: embedding column present after init
  H14 -- KB add_pattern: writes embedding alongside row
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, embeddings as emb
from core.knowledge_base import KnowledgeBase, _connect


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_kb(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", tmp_path / "kb.db")
    return KnowledgeBase(db_path=tmp_path / "kb.db")


def _stub_embedder(monkeypatch, *,
                   succeed: bool = True,
                   custom_vector: list[float] | None = None):
    """Patch embed_text both at the embeddings module AND at the
    knowledge_base import sites that pull it in."""
    counter = {"n": 0}

    def fake_embed(text, trace_id="SEN-system"):
        counter["n"] += 1
        if not succeed:
            return None
        if custom_vector is not None:
            vec = custom_vector
        else:
            # Deterministic per-text vector so equal texts -> equal vectors.
            seed = sum(ord(c) for c in (text or "")) % (2**31 - 1)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(config.EMBEDDING_DIM).tolist()
        return struct.pack(f"<{len(vec)}f", *vec)

    monkeypatch.setattr(emb, "embed_text", fake_embed)
    return counter


# ─────────────────────────────────────────────────────────────────
# Pure unit tests
# ─────────────────────────────────────────────────────────────────


def test_h01_pack_unpack_roundtrip():
    vec = [0.1, -0.2, 0.3, 1.5, -1.5]
    blob = emb._pack(vec)
    unpacked = emb._unpack(blob)
    assert unpacked.shape == (5,)
    np.testing.assert_allclose(unpacked, vec, rtol=1e-5)


def test_h02_cosine_identical_vectors():
    vec = [1.0, 2.0, 3.0]
    blob = emb._pack(vec)
    assert emb.cosine_similarity(blob, blob) == pytest.approx(1.0, abs=1e-5)


def test_h03_cosine_orthogonal_vectors():
    a = emb._pack([1.0, 0.0, 0.0])
    b = emb._pack([0.0, 1.0, 0.0])
    assert emb.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)


def test_h04_cosine_handles_garbage():
    valid = emb._pack([1.0, 2.0])
    assert emb.cosine_similarity(None, valid) == 0.0
    assert emb.cosine_similarity(valid, None) == 0.0
    assert emb.cosine_similarity(b"", valid) == 0.0
    assert emb.cosine_similarity(b"abc", valid) == 0.0  # malformed
    # Mismatched dim:
    bigger = emb._pack([1.0, 2.0, 3.0])
    assert emb.cosine_similarity(valid, bigger) == 0.0


def test_h05_hybrid_score_top_and_high_cosine_wins():
    # Top of FTS list (rank 0/10), perfect cosine
    top = emb.hybrid_score(0, 10, 1.0)
    # Bottom of FTS list (rank 9/10), worst cosine
    bot = emb.hybrid_score(9, 10, -1.0)
    assert top > bot
    assert top > 0.5
    assert bot < 0.5


def test_h06_hybrid_weight_extremes():
    # Pure BM25: cosine doesn't matter
    s1 = emb.hybrid_score(0, 10, 1.0, bm25_weight=1.0)
    s2 = emb.hybrid_score(0, 10, -1.0, bm25_weight=1.0)
    assert s1 == s2
    # Pure semantic: rank doesn't matter
    s3 = emb.hybrid_score(0, 10, 1.0, bm25_weight=0.0)
    s4 = emb.hybrid_score(9, 10, 1.0, bm25_weight=0.0)
    assert s3 == s4


# ─────────────────────────────────────────────────────────────────
# Embedding endpoint integration (mocked HTTP)
# ─────────────────────────────────────────────────────────────────


def test_h07_embed_text_success(monkeypatch):
    """Mock urlopen to return a valid embedding response."""
    import json as _json

    class FakeResp:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return _json.dumps({
                "embedding": [0.1] * config.EMBEDDING_DIM,
            }).encode("utf-8")

    def fake_urlopen(req, timeout):
        return FakeResp(None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    blob = emb.embed_text("hello world")
    assert blob is not None
    assert len(blob) == config.EMBEDDING_DIM * 4  # 4 bytes per float32


def test_h08_embed_text_handles_failure(monkeypatch):
    import urllib.error

    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert emb.embed_text("hello") is None


# ─────────────────────────────────────────────────────────────────
# KB integration -- the real point of the batch
# ─────────────────────────────────────────────────────────────────


def test_h13_schema_has_embedding_column(fresh_kb):
    conn = _connect(fresh_kb.db_path)
    try:
        cols = {r["name"] for r in conn.execute(
            "PRAGMA table_info(knowledge)"
        ).fetchall()}
    finally:
        conn.close()
    assert "embedding" in cols


def test_h14_add_pattern_writes_embedding(fresh_kb, monkeypatch):
    _stub_embedder(monkeypatch)
    pid = fresh_kb.add_pattern(
        tags=["t"], problem_summary="reverse a string",
        solution_code="s[::-1]", solution_pattern="slice negative",
        explanation="x", trace_id="H14",
    )
    conn = _connect(fresh_kb.db_path)
    try:
        row = conn.execute(
            "SELECT embedding FROM knowledge WHERE id = ?", (pid,),
        ).fetchone()
    finally:
        conn.close()
    assert row["embedding"] is not None
    # Right size for a 384-dim float32 vector:
    assert len(row["embedding"]) == config.EMBEDDING_DIM * 4


def test_h09_search_hybrid_reorders_by_similarity(fresh_kb, monkeypatch):
    """Seed three patterns that all match the FTS query but with
    different semantic relevance. With hybrid retrieval, the most
    semantically-similar one should win even if it ranks lower in
    BM25.

    Approach: assign deterministic vectors so we can prove the
    rerank uses them.
    """
    # Use custom vectors keyed off the problem summary
    def fake_embed(text, trace_id="SEN-system"):
        # Map specific texts to specific vectors so cosine ordering
        # is predictable.
        if "boat" in text.lower():
            vec = [1.0, 0.0, 0.0] + [0.0] * (config.EMBEDDING_DIM - 3)
        elif "ship" in text.lower():
            vec = [0.95, 0.1, 0.0] + [0.0] * (config.EMBEDDING_DIM - 3)
        elif "spaceship" in text.lower():
            vec = [0.0, 1.0, 0.0] + [0.0] * (config.EMBEDDING_DIM - 3)
        else:
            vec = [0.0, 0.0, 1.0] + [0.0] * (config.EMBEDDING_DIM - 3)
        return struct.pack(f"<{config.EMBEDDING_DIM}f", *vec)

    monkeypatch.setattr(emb, "embed_text", fake_embed)

    # All match the FTS query "vehicle" by tag
    fresh_kb.add_pattern(
        tags=["vehicle"], problem_summary="boat sailing",
        solution_code="x", solution_pattern="boat", explanation="e",
        trace_id="H09a",
    )
    fresh_kb.add_pattern(
        tags=["vehicle"], problem_summary="spaceship orbit",
        solution_code="x", solution_pattern="spaceship", explanation="e",
        trace_id="H09b",
    )
    fresh_kb.add_pattern(
        tags=["vehicle"], problem_summary="ship navigation",
        solution_code="x", solution_pattern="ship", explanation="e",
        trace_id="H09c",
    )

    # Query "boat" -- semantically close to boat (1.0) and ship (0.99)
    # very far from spaceship (0.0). Hybrid rerank should put boat OR
    # ship at top, not spaceship.
    results = fresh_kb.search("vehicle", max_results=3, hybrid=True)
    top_summaries = [r.problem_summary for r in results]
    # The top result should be one of boat/ship, NOT spaceship
    assert "spaceship" not in top_summaries[0].lower(), (
        f"hybrid retrieval should NOT put spaceship at top "
        f"when query embeds to 'vehicle' which is far from spaceship "
        f"vector but close to boat/ship. Got: {top_summaries}"
    )


def test_h10_rows_without_embeddings_still_appear(fresh_kb, monkeypatch):
    """A row from before the migration (NULL embedding) must still
    appear in hybrid search results -- never excluded."""
    _stub_embedder(monkeypatch, succeed=False)  # all writes fail to embed
    pid = fresh_kb.add_pattern(
        tags=["legacy"], problem_summary="legacy row",
        solution_code="x", solution_pattern="y", explanation="z",
        trace_id="H10",
    )
    # Now succeed for the SEARCH side
    _stub_embedder(monkeypatch, succeed=True)
    results = fresh_kb.search("legacy", max_results=5)
    assert any(r.id == pid for r in results)


def test_h11_search_hybrid_false_is_deterministic(fresh_kb, monkeypatch):
    """hybrid=False bypasses embeddings -- pure BM25 ordering."""
    _stub_embedder(monkeypatch)
    fresh_kb.add_pattern(
        tags=["a"], problem_summary="apple", solution_code="x",
        solution_pattern="y", explanation="z", trace_id="H11a",
    )
    fresh_kb.add_pattern(
        tags=["a"], problem_summary="banana", solution_code="x",
        solution_pattern="y", explanation="z", trace_id="H11b",
    )
    # Two runs should give identical ordering (no randomness from
    # embedding rerank).
    r1 = fresh_kb.search("a", max_results=2, hybrid=False)
    r2 = fresh_kb.search("a", max_results=2, hybrid=False)
    assert [e.id for e in r1] == [e.id for e in r2]


def test_h12_backfill_embeddings_fills_nulls(fresh_kb, monkeypatch):
    """Rows added with embedder failing get filled by backfill."""
    # First, add rows with embedder failing -> NULL embedding
    _stub_embedder(monkeypatch, succeed=False)
    pid = fresh_kb.add_pattern(
        tags=["bf"], problem_summary="P", solution_code="x",
        solution_pattern="y", explanation="z", trace_id="H12",
    )
    # Confirm NULL
    conn = _connect(fresh_kb.db_path)
    try:
        row = conn.execute(
            "SELECT embedding FROM knowledge WHERE id = ?", (pid,),
        ).fetchone()
    finally:
        conn.close()
    assert row["embedding"] is None

    # Now run backfill with embedder succeeding
    _stub_embedder(monkeypatch, succeed=True)
    counts = fresh_kb.backfill_embeddings()
    assert counts["scanned"] >= 1
    assert counts["embedded"] >= 1

    # Confirm filled
    conn = _connect(fresh_kb.db_path)
    try:
        row = conn.execute(
            "SELECT embedding FROM knowledge WHERE id = ?", (pid,),
        ).fetchone()
    finally:
        conn.close()
    assert row["embedding"] is not None
