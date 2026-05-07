"""Phase 9 integration tests.

Some are mocked (no live Telegram) -- O, R-marker, S, T, U.
Live live tests against the real bot live in workspace/sim_phase9.py
once the user has set SENTINEL_TELEGRAM_USER_ID.
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config


def make_update(user_id: int, text: str = "") -> MagicMock:
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = f"user{user_id}"
    update.message.text = text
    update.message.reply_text = AsyncMock()
    return update


def make_context(args: list[str] | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.args = args or []
    return ctx


def _make_bot(uid: int = 12345):
    config.TELEGRAM_AUTHORIZED_USERS = [uid]
    config.TELEGRAM_TOKEN = "fake-token"
    from core.brain import BrainRouter
    from core.claude_cli import ClaudeCLI
    from core.knowledge_base import KnowledgeBase
    from core.llm import InferenceClient
    from interfaces.telegram_bot import SentinelTelegramBot
    return SentinelTelegramBot(
        token="fake-token",
        brain=BrainRouter(),
        claude_cli=ClaudeCLI(),
        inference_client=InferenceClient(),
        knowledge_base=KnowledgeBase(),
    )


def test_o_natural_language_dispatch_full_chain(monkeypatch):
    """End-to-end natural-language flow with the brain mocked to dispatch
    to /search, the route + worker mocked, and the brain summarizing.
    Verifies the orchestration glue, not the live LLM."""
    bot = _make_bot()

    # Mock brain.process to return a dispatch decision
    from core.brain import BrainResult
    async def fake_process(msg, trace_id):
        return BrainResult(
            intent="dispatch",
            agent="web_search", command="/search",
            args="RSM jobs Michigan",
            summary="Searching the web for RSM jobs near Michigan.",
            trace_id=trace_id,
        )
    monkeypatch.setattr(bot.brain, "process", fake_process)

    # Mock route() in the bot's namespace to return ok with a fake task_id
    fake_task_id = "task-abc-123"
    fake_route_result = MagicMock(
        status="ok", task_id=fake_task_id, message="routed",
    )
    monkeypatch.setattr(
        "interfaces.telegram_bot.route",
        lambda cmd: fake_route_result,
    )

    # Mock _wait_for_task to "complete" immediately
    async def fake_wait(task_id, timeout, progress_message=None, trace_id=None):
        return {
            "task_id": task_id, "status": "completed",
            "result": json.dumps({
                "query": "RSM jobs Michigan",
                "results": [{"title": "RSM at AcmeCorp",
                             "url": "https://example.com",
                             "snippet": "Great role"}],
                "result_count": 1,
            }),
        }
    monkeypatch.setattr(bot, "_wait_for_task", fake_wait)

    # Mock brain.summarize_result
    async def fake_summarize(original_request, raw_result, trace_id):
        return "I found 1 RSM role in Michigan: 'RSM at AcmeCorp'."
    monkeypatch.setattr(bot.brain, "summarize_result", fake_summarize)

    update = make_update(user_id=12345, text="find RSM jobs near Michigan")
    asyncio.run(bot.handle_message(update, make_context()))

    sent = [c.args[0] for c in update.message.reply_text.call_args_list]
    # First message: the brain's "summary" + working trace
    assert any("Searching the web" in s for s in sent)
    # Final summary
    assert any("RSM at AcmeCorp" in s for s in sent)


def test_q_long_message_chunked():
    """A reply > 4096 chars must be split. Each chunk under the cap."""
    bot = _make_bot()
    update = make_update(user_id=12345)
    long = ("Lorem ipsum " * 1000)  # ~12000 chars
    asyncio.run(bot._send_long(update, long))
    chunks = [c.args[0] for c in update.message.reply_text.call_args_list]
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= config.TELEGRAM_MAX_MESSAGE_LENGTH


def test_s_task_timeout_message(monkeypatch):
    """When _wait_for_task returns None (timeout), user gets a clear msg."""
    bot = _make_bot()
    fake_route = MagicMock(status="ok", task_id="t1", message="routed")
    monkeypatch.setattr("interfaces.telegram_bot.route", lambda c: fake_route)

    async def fake_wait_timeout(task_id, timeout, progress_message=None, trace_id=None):
        return None
    monkeypatch.setattr(bot, "_wait_for_task", fake_wait_timeout)

    update = make_update(user_id=12345)
    ctx = make_context(args=["any", "query"])
    asyncio.run(bot.handle_search(update, ctx))
    sent = [c.args[0] for c in update.message.reply_text.call_args_list]
    assert any("timed out" in s.lower() for s in sent)


def test_u_regression_marker():
    """Phase 3-8 still intact: registries populated, key commands routable."""
    from core.agent_registry import AGENT_REGISTRY
    from core.registry import SKILL_REGISTRY

    skills = {s["name"] for s in SKILL_REGISTRY.list_skills()}
    assert {"job_extract", "web_search", "file_io", "code_execute",
            "code_assist"}.issubset(skills)
    assert AGENT_REGISTRY.has("job_analyst")
    assert AGENT_REGISTRY.has("code_assistant")
    for cmd in ("/ping", "/help", "/status", "/extract", "/search",
                "/file", "/exec", "/code", "/models", "/complexity"):
        assert cmd in config.REGISTERED_COMMANDS
        assert cmd in config.COMMAND_AGENT_MAP
