"""Phase 10 -- Tests E-K: three-tier memory + Telegram round-trip + auto-extract.

E -- store_fact/get_fact round-trip
F -- upsert by key advances updated_at; confidence-aware merge
G -- episodic store/retrieval respects scope
H -- get_agent_context capped + sorted by relevance
I -- decay_relevance reduces scores; prune_episodes drops lowest first
J -- /remember, /recall, /forget Telegram handlers round-trip
K -- auto-extraction from a 5-message conversation stores >=1 fact
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from core import config
from core.memory import (
    WORKING_MEMORY, MemoryManager, WorkingMemory, get_memory,
)


# -------------------------------------------------------------------
# E -- store_fact + get_fact round-trip
# -------------------------------------------------------------------

def test_e_store_and_get_fact():
    mem = get_memory()
    mem.store_fact("salary_minimum", "90000", source="user_explicit")
    fact = mem.get_fact("salary_minimum")
    assert fact is not None
    assert fact.value == "90000"
    assert fact.confidence == 1.0
    assert fact.source == "user_explicit"


# -------------------------------------------------------------------
# F -- upsert advances updated_at
# -------------------------------------------------------------------

def test_f_upsert_advances_updated_at():
    mem = get_memory()
    mem.store_fact("location", "Detroit", source="user_explicit")
    f1 = mem.get_fact("location")
    assert f1 is not None
    time.sleep(0.05)  # SQLite ISO ts has sub-second granularity
    mem.store_fact("location", "Dearborn Heights", source="user_explicit")
    f2 = mem.get_fact("location")
    assert f2 is not None
    assert f2.value == "Dearborn Heights"
    assert f2.updated_at > f1.updated_at, (
        f"updated_at did not advance: {f1.updated_at} -> {f2.updated_at}"
    )
    # Confidence-aware merge: a lower-confidence write must NOT
    # overwrite a higher-confidence existing fact.
    mem.store_fact(
        "location", "Garbage", source="auto_extracted", confidence=0.6,
    )
    f3 = mem.get_fact("location")
    assert f3 is not None
    assert f3.value == "Dearborn Heights", (
        f"low-conf write overwrote high-conf fact: {f3.value}"
    )


# -------------------------------------------------------------------
# G -- episodic scope filtering
# -------------------------------------------------------------------

def test_g_episodic_scope_filtering():
    mem = get_memory()
    mem.store_episode(
        "job_searcher", "SEN-test", "search",
        "scraped 5 RSM postings", tags=["jobs"],
    )
    mem.store_episode(
        "researcher", "SEN-test", "research",
        "AI trends Q1 brief", tags=["research"],
    )
    mem.store_episode(
        "global", "SEN-test", "chat",
        "user asked about salary", tags=["chat"],
    )

    job_eps = mem.get_recent_episodes(scope="job_searcher", limit=10)
    assert len(job_eps) == 1
    assert "RSM" in job_eps[0].summary

    all_eps = mem.get_recent_episodes(limit=10)
    assert len(all_eps) == 3


# -------------------------------------------------------------------
# H -- get_agent_context capped + sorted
# -------------------------------------------------------------------

def test_h_get_agent_context_capped_and_sorted():
    mem = get_memory()
    # 6 episodes for "researcher", varying relevance.
    for i in range(6):
        mem.store_episode(
            "researcher", f"SEN-{i:08x}", "research",
            f"topic-{i} brief that mentions trends",
            relevance_score=0.5 + 0.05 * i,  # higher i = higher rel
        )
    ctx = mem.get_agent_context(
        "researcher", "trends", max_chars=3000,
    )
    assert len(ctx) <= 3000
    assert "topic-" in ctx
    # Highest-relevance episode (topic-5) should appear before topic-0.
    pos5 = ctx.find("topic-5")
    pos0 = ctx.find("topic-0")
    assert pos5 != -1, "expected topic-5 in context"
    assert pos5 < pos0 or pos0 == -1, (
        f"highest-relevance not first: pos5={pos5} pos0={pos0}"
    )


# -------------------------------------------------------------------
# I -- decay + prune
# -------------------------------------------------------------------

def test_i_decay_relevance_and_prune():
    mem = get_memory()
    eid_recent = mem.store_episode(
        "global", "SEN-test", "test", "recent episode",
        relevance_score=1.0,
    )
    # Manually backdate one row so decay catches it.
    import sqlite3
    conn = sqlite3.connect(mem.db_path)
    try:
        conn.execute(
            "UPDATE episodic_memory SET created_at = ? WHERE id = ?",
            ("2020-01-01T00:00:00+00:00", eid_recent),
        )
        conn.commit()
    finally:
        conn.close()
    n_decayed = mem.decay_relevance(days_old=30, factor=0.5)
    assert n_decayed >= 1
    after = mem.get_recent_episodes(limit=10)
    target = next(e for e in after if e.id == eid_recent)
    assert target.relevance_score == pytest.approx(0.5, abs=0.01)

    # Prune: insert > cap, lowest-relevance ones drop first.
    cap = 5
    for i in range(cap + 3):
        mem.store_episode(
            "prune_test", "SEN-prune", "test", f"ep-{i}",
            relevance_score=0.1 + 0.1 * i,
        )
    deleted = mem.prune_episodes(scope="prune_test", max_per_scope=cap)
    assert deleted >= 3
    remaining = mem.get_recent_episodes(scope="prune_test", limit=20)
    assert len(remaining) == cap
    # The lowest-rel one (ep-0) should be gone.
    assert all(e.summary != "ep-0" for e in remaining)


# -------------------------------------------------------------------
# J -- /remember, /recall, /forget round-trip via handlers
# -------------------------------------------------------------------

def test_j_telegram_memory_handlers_round_trip(monkeypatch):
    """Construct a SentinelTelegramBot just enough to call its handlers
    directly. Mock Telegram Update/Context with attribute-only stubs.
    """
    from interfaces.telegram_bot import SentinelTelegramBot

    # Authorize the test user.
    monkeypatch.setattr(
        config, "TELEGRAM_AUTHORIZED_USERS", [42],
    )

    bot = SentinelTelegramBot.__new__(SentinelTelegramBot)

    class _StubApp:
        def __init__(self):
            self.bot = SimpleNamespace(send_message=lambda **kw: None)
            self.handlers: list = []

        def add_handler(self, h):
            self.handlers.append(h)

    bot.app = _StubApp()
    bot.brain = SimpleNamespace()
    bot.claude_cli = SimpleNamespace()
    bot.inference = SimpleNamespace()
    bot.kb = SimpleNamespace(stats=lambda: {})
    bot._started_at = 0.0
    from core.file_guard import FileGuard
    bot.file_guard = FileGuard(
        directory=config.PERSONA_DIR, alert_callback=lambda _: None,
    )
    bot._heartbeat_task = None

    replies: list[str] = []

    def _make_update():
        async def reply_text(text):
            replies.append(text)
        msg = SimpleNamespace(
            text="(unused)",
            reply_text=reply_text,
        )
        return SimpleNamespace(
            message=msg,
            effective_user=SimpleNamespace(id=42, username="tester"),
        )

    def _ctx(text: str):
        return SimpleNamespace(args=text.split())

    # /remember
    asyncio.run(bot.handle_remember(
        _make_update(),
        _ctx("salary_minimum: 92000"),
    ))
    assert replies, "expected a reply from /remember"
    assert "salary_minimum" in replies[-1]
    fact = get_memory().get_fact("salary_minimum")
    assert fact is not None and fact.value == "92000"

    # /recall
    replies.clear()
    asyncio.run(bot.handle_recall(
        _make_update(),
        _ctx("salary"),
    ))
    assert replies, "expected a reply from /recall"
    combined = "\n".join(replies)
    assert "salary_minimum" in combined and "92000" in combined

    # /forget
    replies.clear()
    asyncio.run(bot.handle_forget(
        _make_update(),
        _ctx("salary_minimum"),
    ))
    assert replies and "Forgot" in replies[-1]
    assert get_memory().get_fact("salary_minimum") is None


# -------------------------------------------------------------------
# K -- auto-extraction from a 5-message conversation
# -------------------------------------------------------------------

class _Result(BaseModel):
    text: str
    model_used: str = "fake"
    backend: str = "fake"


class _FakeInference:
    async def generate(self, **kwargs):
        return _Result(
            text='[{"key": "favorite_language", "value": "Python"}, '
                 '{"key": "timezone", "value": "America/Detroit"}]',
        )


def test_k_auto_extraction_stores_facts():
    mem = get_memory()
    fake_brain = SimpleNamespace(
        inference=_FakeInference(),
        model="fake-brain",
    )
    messages = [
        {"role": "user", "message": "I work in Python every day."},
        {"role": "user", "message": "I'm in the Detroit area."},
        {"role": "user", "message": "What are some good libraries?"},
        {"role": "user", "message": "How does asyncio compare to threading?"},
        {"role": "user", "message": "I prefer dataclasses to namedtuples."},
    ]
    n = asyncio.run(mem.extract_facts_from_conversation(
        messages, "SEN-autoex", brain=fake_brain,
    ))
    assert n >= 1, f"expected at least 1 fact extracted, got {n}"
    f1 = mem.get_fact("favorite_language")
    assert f1 is not None
    assert f1.confidence == pytest.approx(0.6, abs=1e-6)
    assert f1.source == "auto_extracted"
