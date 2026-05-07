"""Telegram bot tests. Mocked Update/Context so we don't need a live
bot connection. Live integration tests live in test_pipeline_phase9."""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config


# ---------- helpers ---------------------------------------------------

def make_update(user_id: int, text: str = "") -> MagicMock:
    """Build a minimal Update double good enough for our handlers."""
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


def make_bot(authorized_user_id: int = 12345):
    """Construct a SentinelTelegramBot with mocked dependencies. The
    Application is real (built via builder) but never actually polls."""
    # Only flip authorization for the duration of the test
    config.TELEGRAM_AUTHORIZED_USERS = [authorized_user_id]
    config.TELEGRAM_TOKEN = "fake-token"
    from core.brain import BrainRouter
    from core.claude_cli import ClaudeCLI
    from core.knowledge_base import KnowledgeBase
    from core.llm import InferenceClient
    from interfaces.telegram_bot import SentinelTelegramBot
    bot = SentinelTelegramBot(
        token="fake-token",
        brain=BrainRouter(),       # Real brain (mocked inference per test)
        claude_cli=ClaudeCLI(),
        inference_client=InferenceClient(),
        knowledge_base=KnowledgeBase(),
    )
    return bot


# ---------- the tests -------------------------------------------------

def test_h_unauthorized_user_blocked():
    bot = make_bot(authorized_user_id=99999)
    update = make_update(user_id=11111, text="/code reverse a string")
    asyncio.run(bot.handle_code(update, make_context(args=["reverse", "a", "string"])))
    update.message.reply_text.assert_called_once_with("Unauthorized.")


def test_help_command(monkeypatch):
    bot = make_bot(authorized_user_id=12345)
    update = make_update(user_id=12345)
    asyncio.run(bot.handle_help(update, make_context()))
    msg = update.message.reply_text.call_args.args[0]
    assert "/code" in msg and "/claude" in msg and "/help" in msg


def test_status_command_returns_counts(monkeypatch):
    """Status reports queue counts. We don't assert exact numbers --
    just that the message has the right shape and the loaded-model
    line is present."""
    bot = make_bot(authorized_user_id=12345)
    monkeypatch.setattr(bot.inference, "get_loaded_model",
                        lambda: "qwen3:1.7b")
    update = make_update(user_id=12345)
    asyncio.run(bot.handle_status(update, make_context()))
    msg = update.message.reply_text.call_args.args[0]
    assert "Queue:" in msg
    assert "GPU:" in msg
    assert "Brain:" in msg


def test_models_command_lists_registry(monkeypatch):
    bot = make_bot(authorized_user_id=12345)
    update = make_update(user_id=12345)
    asyncio.run(bot.handle_models(update, make_context()))
    msg = update.message.reply_text.call_args.args[0]
    assert "Registered Models:" in msg
    assert "sentinel-brain" in msg or "qwen3-brain" in msg


def test_kb_command_returns_stats(monkeypatch):
    bot = make_bot(authorized_user_id=12345)
    update = make_update(user_id=12345)
    asyncio.run(bot.handle_kb(update, make_context()))
    msg = update.message.reply_text.call_args.args[0]
    assert "Knowledge Base:" in msg
    assert "Total entries:" in msg


def test_send_long_message_splits_at_4k(monkeypatch):
    bot = make_bot(authorized_user_id=12345)
    update = make_update(user_id=12345)
    long_text = "\n".join([f"line {i}: {'x' * 100}" for i in range(80)])
    asyncio.run(bot._send_long(update, long_text))
    # Each call's first arg is the message text
    sent_chunks = [c.args[0] for c in update.message.reply_text.call_args_list]
    assert len(sent_chunks) >= 2
    for chunk in sent_chunks:
        assert len(chunk) <= config.TELEGRAM_MAX_MESSAGE_LENGTH


def test_claude_handler_calls_claude_cli(monkeypatch):
    bot = make_bot(authorized_user_id=12345)

    async def fake_generate(**kwargs):
        from core.claude_cli import GenerateResult
        return GenerateResult(success=True, text="Hello from Claude.")

    monkeypatch.setattr(bot.claude_cli, "generate", fake_generate)
    update = make_update(user_id=12345)
    ctx = make_context(args=["what", "is", "1+1"])
    asyncio.run(bot.handle_claude(update, ctx))
    sent = [c.args[0] for c in update.message.reply_text.call_args_list]
    # First reply: "Asking Claude..."; second: the answer
    assert any("Asking Claude" in s for s in sent)
    assert any("Hello from Claude" in s for s in sent)


def test_unauthorized_helper_logs_warning(monkeypatch):
    """The auth check must log a WARNING with the rejected user_id."""
    from core.logger import log_event
    bot = make_bot(authorized_user_id=99999)
    update = make_update(user_id=22222)
    captured = []
    real_log = log_event

    def spy_log(trace_id, level, component, message):
        captured.append((level, component, message))
        return real_log(trace_id, level, component, message)

    import core.logger as logger_mod
    monkeypatch.setattr(logger_mod, "log_event", spy_log)
    # The bot's module imported log_event by name -- monkeypatch there too
    import interfaces.telegram_bot as tg
    monkeypatch.setattr(tg, "log_event", spy_log)

    asyncio.run(bot._check_auth(update))
    warns = [c for c in captured
             if c[0] == "WARNING" and c[1] == "telegram"
             and "22222" in c[2]]
    assert warns, f"expected WARNING log mentioning 22222; got {captured}"
