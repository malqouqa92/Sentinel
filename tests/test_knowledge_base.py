import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.knowledge_base import KnowledgeBase


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(db_path=tmp_path / "kb.db")


def test_a_add_pattern(kb):
    eid = kb.add_pattern(
        tags=["asyncio", "timeout", "wait_for"],
        problem_summary="bound an awaitable with a timeout",
        solution_code="result = await asyncio.wait_for(coro, timeout=5)",
        solution_pattern="wrap any awaitable with asyncio.wait_for",
        explanation="wait_for raises TimeoutError if exceeded",
        trace_id="SEN-test-A",
    )
    assert isinstance(eid, int) and eid > 0


def test_b_search_finds_and_bumps_usage(kb):
    eid = kb.add_pattern(
        tags=["asyncio", "timeout"],
        problem_summary="bound a coroutine with a timeout",
        solution_code="await asyncio.wait_for(coro, timeout=5)",
        solution_pattern="use asyncio.wait_for to bound any coroutine",
        explanation="raises TimeoutError on expiration",
        trace_id="SEN-test-B",
    )
    results = kb.search("asyncio")
    assert len(results) == 1
    assert results[0].id == eid
    # usage_count was 0 before search; should be 1 after.
    assert results[0].usage_count == 1
    # Search again -> usage_count should reach 2.
    again = kb.search("asyncio")
    assert again[0].usage_count == 2


def test_c_search_no_match(kb):
    kb.add_pattern(
        tags=["http"], problem_summary="x", solution_code="x",
        solution_pattern="x", explanation="x", trace_id="SEN-test-C",
    )
    assert kb.search("completely unrelated topic xyz") == []


def test_d_get_context_for_prompt(kb):
    kb.add_pattern(
        tags=["asyncio", "timeout"],
        problem_summary="bound a coroutine with a timeout",
        solution_code=(
            "import asyncio\n"
            "async def safe(c):\n"
            "    return await asyncio.wait_for(c, timeout=5)\n"
        ),
        solution_pattern="wrap with asyncio.wait_for(coro, timeout=N)",
        explanation="raises TimeoutError after N seconds; cancels coro",
        trace_id="SEN-test-D",
    )
    ctx = kb.get_context_for_prompt("asyncio timeout")
    assert "KNOWN PATTERN" in ctx
    assert "asyncio.wait_for" in ctx
    assert len(ctx) <= 4000


def test_e_prune_drops_lowest_usage(kb):
    """Add 10 entries with varying usage_count, prune to 5."""
    import sqlite3
    ids = []
    for i in range(10):
        eid = kb.add_pattern(
            tags=[f"t{i}"], problem_summary=f"problem {i}",
            solution_code=f"code {i}",
            solution_pattern=f"pattern {i}",
            explanation=f"why {i}",
            trace_id=f"SEN-test-E-{i}",
        )
        ids.append(eid)
    # Set usage_count on each entry: id i -> usage_count = i.
    conn = sqlite3.connect(str(kb.db_path), isolation_level=None)
    try:
        for i, eid in enumerate(ids):
            conn.execute(
                "UPDATE knowledge SET usage_count = ? WHERE id = ?",
                (i, eid),
            )
    finally:
        conn.close()
    deleted = kb.prune(max_entries=5)
    assert deleted == 5
    # The 5 lowest-usage entries (i=0..4) should be gone.
    remaining_summaries = {
        e.problem_summary for e in kb.search("problem")
    }
    # Remaining should be problems 5..9.
    for i in range(5, 10):
        assert f"problem {i}" in remaining_summaries
    for i in range(0, 5):
        assert f"problem {i}" not in remaining_summaries


def test_f_stats(kb):
    kb.add_pattern(
        tags=["a"], problem_summary="p1", solution_code="c",
        solution_pattern="pat", explanation="ex",
        trace_id="SEN-test-F-1",
    )
    kb.add_pattern(
        tags=["b"], problem_summary="p2", solution_code="c",
        solution_pattern="pat", explanation="ex",
        trace_id="SEN-test-F-2",
    )
    kb.add_limitation(
        tags=["c"], problem_summary="lim1",
        explanation="qwen could not learn",
        trace_id="SEN-test-F-3",
    )
    s = kb.stats()
    assert s["total_entries"] == 3
    assert s["patterns_count"] == 2
    assert s["limitations_count"] == 1
    assert s["avg_usage_count"] == 0.0
