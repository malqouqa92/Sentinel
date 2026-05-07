"""Pre-Phase-15: embedding wrapper for hybrid retrieval.

Sentinel runs on a 4GB-VRAM constraint. We use ``nomic-embed-text``
(~270 MB, 768-dim, 2048-token context) via Ollama's native embedding
endpoint. Combined with worker (qwen2.5:3b ~1.9 GB) = ~2.2 GB of
4 GB total VRAM, leaving ~1.8 GB for KV cache. The 2048-token context
(~6000 chars) handles our long recipe patterns without truncation
that smaller embedders (all-minilm @ 256 tokens) couldn't.

Embeddings are stored as raw float32 BLOBs (4 bytes × 768 = 3 KB
per row) in SQLite. Cosine similarity uses numpy. Search blends FTS5
keyword score with cosine similarity for hybrid retrieval -- the
proven pattern for small-model RAG (FTS5 catches exact matches,
embeddings catch semantic neighbours).

Best-effort throughout: any embedding failure (Ollama down, model
missing, network glitch) returns None and the caller gracefully
degrades to FTS5-only behaviour.
"""
from __future__ import annotations

import json
import struct
import urllib.error
import urllib.request
from typing import Iterable

import numpy as np

from core import config
from core.logger import log_event


# Model + dim are config so a future swap (e.g. mxbai-embed-large)
# doesn't require code changes. Defaults match production.
EMBEDDING_MODEL = getattr(config, "EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = getattr(config, "EMBEDDING_DIM", 768)
# Cap on the embedding endpoint -- nomic typically returns in <500ms;
# 10s safety margin handles cold-load when embedder isn't yet resident.
EMBEDDING_TIMEOUT_S = 10.0


def embed_text(text: str, trace_id: str = "SEN-system") -> bytes | None:
    """Compute an embedding for ``text`` via Ollama. Returns the raw
    float32 BLOB ready for SQLite storage, or ``None`` on any failure.

    Caller MUST handle None to mean "no embedding available, skip
    semantic part of hybrid retrieval for this row."
    """
    text = (text or "").strip()
    if not text:
        return None
    # nomic-embed-text has a 2048-token context (~6000 chars). Cap
    # at 6000 to give a small safety margin. If you swap to a smaller
    # embedder (all-minilm @ 256 tokens / ~1000 chars), narrow this
    # to text[:1000] to avoid HTTP 500 "input length exceeds context".
    payload = json.dumps({
        "model": EMBEDDING_MODEL,
        "prompt": text[:6000],
    }).encode("utf-8")
    url = f"{config.OLLAMA_BASE_URL}/api/embeddings"
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=EMBEDDING_TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        log_event(
            trace_id, "DEBUG", "embeddings",
            f"embed failed ({EMBEDDING_MODEL}): "
            f"{type(e).__name__}: {e}",
        )
        return None
    vec = body.get("embedding")
    if not isinstance(vec, list) or len(vec) != EMBEDDING_DIM:
        log_event(
            trace_id, "WARNING", "embeddings",
            f"unexpected embedding shape: dim={len(vec) if isinstance(vec, list) else 'n/a'} "
            f"(expected {EMBEDDING_DIM})",
        )
        return None
    return _pack(vec)


def _pack(vec: list[float]) -> bytes:
    """float32 little-endian -- compact, deterministic, numpy-friendly."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> np.ndarray:
    """Round-trip the BLOB back to a numpy float32 vector."""
    n = len(blob) // 4
    return np.array(struct.unpack(f"<{n}f", blob), dtype=np.float32)


def cosine_similarity(query_blob: bytes, candidate_blob: bytes) -> float:
    """Cosine in [-1, 1]. Returns 0.0 if either side is empty/malformed
    (graceful no-signal, not a crash)."""
    if not query_blob or not candidate_blob:
        return 0.0
    try:
        q = _unpack(query_blob)
        c = _unpack(candidate_blob)
    except (struct.error, ValueError):
        return 0.0
    if q.size == 0 or c.size == 0 or q.size != c.size:
        return 0.0
    qn = float(np.linalg.norm(q))
    cn = float(np.linalg.norm(c))
    if qn == 0.0 or cn == 0.0:
        return 0.0
    return float(np.dot(q, c) / (qn * cn))


def hybrid_score(
    bm25_rank: int, total_candidates: int,
    cosine: float,
    bm25_weight: float = 0.4,
) -> float:
    """Blend FTS5 BM25 rank with cosine similarity into one score.

    BM25 is normalized by rank-position (1.0 for top, 0 for last) since
    raw BM25 scores are unbounded and incomparable across queries.
    Cosine is already in [-1, 1] -- we shift to [0, 1] for blending.

    bm25_weight tunes how much keyword vs semantic matters. 0.4 leans
    slightly semantic which empirically works well for code/pattern
    retrieval where exact tokens matter less than concept overlap.
    """
    bm25_norm = (
        1.0 - (bm25_rank / max(1, total_candidates))
        if total_candidates > 1 else 1.0
    )
    cosine_norm = (cosine + 1.0) / 2.0
    return bm25_weight * bm25_norm + (1.0 - bm25_weight) * cosine_norm


def rerank_by_hybrid(
    query_text: str,
    candidates: list[tuple[int, bytes | None]],
    trace_id: str = "SEN-system",
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Given FTS5-ordered (id, embedding_blob) candidates and a query,
    compute a hybrid score per candidate and return the reordered list
    (most relevant first) as ``(id, score)`` tuples.

    Candidates with no embedding fall back to BM25-only score (their
    rank position) so old patterns from before embeddings shipped
    still appear, just without the semantic boost.
    """
    if not candidates:
        return []
    n = len(candidates)
    query_blob = embed_text(query_text, trace_id)
    scored: list[tuple[int, float]] = []
    for rank_idx, (cid, cblob) in enumerate(candidates):
        cosine = (
            cosine_similarity(query_blob, cblob)
            if (query_blob and cblob)
            else 0.0
        )
        score = hybrid_score(rank_idx, n, cosine)
        scored.append((cid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        scored = scored[:top_k]
    return scored
