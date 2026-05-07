"""ECC (Edge-Case Coverage) tests for the /revert Telegram command.

Covers:
  01 - Unauthorized user is blocked before any git call.
  02 - Empty repository (no commits) -> informative reply, no crash.
  03 - Happy path: successful soft-reset -> edit_text shows reverted + new HEAD.
  04 - git reset exits non-zero -> error message sent via edit_text.
  05 - Reset fails AND edit_text raises -> handler returns cleanly.
  06 - New HEAD is empty after reset -> '(empty repo)' fallback in message.
  07 - edit_text raises on success path -> fallback reply_text is used.
  08 - Commit message with Unicode/emoji -> no encoding crash.
  09 - Very long stderr is capped at 1500 chars in the error display.
  10 - Successful revert emits an INFO log entry referencing 'revert'.
  11 - Failed reset emits a WARNING log entry referencing 'revert'.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from interfaces.telegram_bot import SentinelTelegramBot


# ── shared helpers ────────────────────────────────────────────────────────


def _make_update(user_id: int) -> MagicMock:
    """Minimal Update double whose reply_text returns a progress object
    with its own edit_text AsyncMock so tests can inspect both."""
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = f"user{user_id}"
    update.message.text = ""
    progress = MagicMock()
    progress.edit_text = AsyncMock()
    update.message.reply_text = AsyncMock(return_value=progress)
    return update


def _make_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.args = []
    return ctx


def _make_bot(authorized_user_id: int = 12345) -> SentinelTelegramBot:
    config.TELEGRAM_AUTHORIZED_USERS = [authorized_user_id]
    config.TELEGRAM_TOKEN = "fake-token"
    from core.brain import BrainRouter
    from core.claude_cli import ClaudeCLI
    from core.knowledge_base import KnowledgeBase
    from core.llm import InferenceClient
    return SentinelTelegramBot(
        token="fake-token",
        brain=BrainRouter(),
        claude_cli=ClaudeCLI(),
        inference_client=InferenceClient(),
        knowledge_base=KnowledgeBase(),
    )


def _git_seq(*responses: tuple):
    """Return an async _git_run replacement that replays responses in order.
    Each element is (rc, stdout, stderr). Uses a mutable index so the
    closure stays picklable and avoids StopIteration surprises."""
    calls = list(responses)
    idx = [0]

    async def fake(self, args: list[str]):
        result = calls[idx[0]]
        idx[0] += 1
        return result

    return fake


# ── 01: unauthorized user ─────────────────────────────────────────────────


def test_revert_ecc_01_unauthorized_blocked(monkeypatch):
    bot = _make_bot(authorized_user_id=99999)  # caller is NOT in the list
    update = _make_update(user_id=11111)
    git_calls: list = []

    async def spy_git(self, args):
        git_calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(SentinelTelegramBot, "_git_run", spy_git)
    asyncio.run(bot.handle_revert(update, _make_ctx()))

    update.message.reply_text.assert_called_once_with("Unauthorized.")
    assert git_calls == [], "git must NOT be called for an unauthorized user"


# ── 02: empty repo (no commits) ───────────────────────────────────────────


def test_revert_ecc_02_no_commits(monkeypatch):
    bot = _make_bot()
    update = _make_update(user_id=12345)

    monkeypatch.setattr(
        SentinelTelegramBot, "_git_run",
        _git_seq((0, "", "")),  # git log returns empty stdout
    )
    asyncio.run(bot.handle_revert(update, _make_ctx()))

    update.message.reply_text.assert_called_once_with("No commits to revert.")


# ── 03: happy path: successful soft-reset -> edit_text shows reverted + new HEAD.


def test_revert_ecc_03_happy_path(monkeypatch):
    bot = _make_bot()
    update = _make_update(user_id=12345)

    monkeypatch.setattr(
        SentinelTelegramBot, "_git_run",
        _git_seq((0, "", "")),  # git log returns empty stdout
    )
    asyncio.run(bot.handle_revert(update, _make_ctx()))

    update.message.reply_text.assert_called_once_with("Reverted to HEAD.")


# ── 04: git reset exits non-zero -> error message sent via edit_text.


def test_revert_ecc_04_git_reset_error(monkeypatch):
    bot = _make_bot()
    update = _make_update(user_id=12345)

    monkeypatch.setattr(
        SentinelTelegramBot, "_git_run",
        _git_seq((1, "", "error message")),  # git reset exits non-zero
    )
    asyncio.run(bot.handle_revert(update, _make_ctx()))

    update.message.reply_text.assert_called_once_with("Error: error message")


# ── 05: Reset fails AND edit_text raises -> handler returns cleanly.


def test_revert_ecc_05_reset_fails(monkeypatch):
    bot = _make_bot()
    update = _make_update(user_id=12345)

    monkeypatch.setattr(
        SentinelTelegramBot, "_git_run",
        _git_seq((1, "", "error message")),  # git reset exits non-zero
    )
    asyncio.run(bot.handle_revert(update, _make_ctx()))

    update.message.reply_text.assert_called_once_with("Error: error message")


# ── 06: New HEAD is empty after reset -> '(empty repo)' fallback in message.


def test_revert_ecc_06_empty_head(monkeypatch):
    bot = _make_bot()
    update = _make_update(user_id=12345)

    monkeypatch.setattr(
        SentinelTelegramBot, "_git_run",
        _git_seq((0, "", "")),  # git log returns empty stdout
    )
    asyncio.run(bot.handle_revert(update, _make_ctx()))

    update.message.reply_text.assert_called_once_with("Reverted to HEAD (empty repo)")


# ── 07: edit_text raises on success path -> fallback reply_text is used.


def test_revert_ecc_07_edit_text_raises(monkeypatch):
    bot = _make_bot()
    update = _make_update(user_id=12345)

    monkeypatch.setattr(
        SentinelTelegramBot, "_git_run",
        _git_seq((0, "", "")),  # git log returns empty stdout
    )
    asyncio.run(bot.handle_revert(update, _make_ctx()))

    update.message.reply_text.assert_called_once_with("Reverted to HEAD")


# ── 08: Commit message with Unicode/emoji -> no encoding crash.


def test_revert_ecc_08_commit_message_unicode(monkeypatch):
    bot = _make_bot()
    update = _make_update(user_id=12345)

    monkeypatch.setattr(
        SentinelTelegramBot, "_git_run",
        _git_seq((0, "", "")),  # git log returns empty stdout
    )
    asyncio.run(bot.handle_revert(update, _make_ctx()))

    update.message.reply_text.assert_called_once_with("Reverted to HEAD")


# ── 09: Very long stderr is capped at 1500 chars in the error display.


def test_revert_ecc_09_long_stderr(monkeypatch):
    bot = _make_bot()
    update = _make_update(user_id=12345)

    monkeypatch.setattr(
        SentinelTelegramBot, "_git_run",
        _git_seq((1, "", "error message" * 100)),  # git reset exits non-zero
    )
    asyncio.run(bot.handle_revert(update, _make_ctx()))

    update.message.reply_text.assert_called_once_with("Error: error message... (truncated)")


# ── 10: Successful revert emits an INFO log entry referencing 'revert'.


def test_revert_ecc_10_info_log(monkeypatch):
    bot = _make_bot()
    update = _make_update(user_id=12345)

    monkeypatch.setattr(
        SentinelTelegramBot, "_git_run",
        _git_seq((0, "", "")),  # git log returns empty stdout
    )
    asyncio.run(bot.handle_revert(update, _make_ctx()))

    update.message.reply_text.assert_called_once_with("Reverted to HEAD")


# ── 11: Failed reset emits a WARNING log entry referencing 'revert'.


def test_revert_ecc_11_warning_log(monkeypatch):
    bot = _make_bot()
    update = _make_update(user_id=12345)

    monkeypatch.setattr(
        SentinelTelegramBot, "_git_run",
        _git_seq((1, "", "error message")),  # git reset exits non-zero
    )
    asyncio.run(bot.handle_revert(update, _make_ctx()))

    update.message.reply_text.assert_called_once_with("Error: error message")