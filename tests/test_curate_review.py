"""Tests for the /curate review subcommand and its renderer.

Mocks the curation flow at the boundary -- no Claude CLI invoked.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interfaces.telegram_bot import (
    SentinelTelegramBot, _format_curation_proposal,
)


# ---------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------

def test_format_proposal_with_all_three_sections():
    record = {
        "token": "ABCD12",
        "episodes_reviewed": 17,
        "lookback_hours": 24,
        "created_at": "2026-05-05T12:34:56+00:00",
        "proposal": {
            "memory_additions": ["Prefers terse responses."],
            "memory_removals": ["(stale) lives in PT"],
            "user_updates": [
                {"section": "tz", "change": "EST", "reason": "confirmed today"},
            ],
        },
    }
    out = _format_curation_proposal(record)
    assert "[ABCD12]" in out
    assert "17 episodes" in out
    assert "Prefers terse responses." in out
    assert "(stale) lives in PT" in out
    assert "(tz) EST" in out
    assert "confirmed today" in out
    assert "/curate_approve ABCD12" in out
    assert "/curate_reject ABCD12" in out


def test_format_proposal_no_changes():
    record = {
        "token": "NOOP01",
        "episodes_reviewed": 5,
        "lookback_hours": 24,
        "created_at": "2026-05-05T00:00:00+00:00",
        "proposal": {"no_changes": True},
    }
    out = _format_curation_proposal(record)
    assert "[NOOP01]" in out
    assert "No durable changes" in out
    assert "/curate_approve" not in out  # nothing to approve


def test_format_proposal_only_additions():
    """Render must omit empty sections (no MEMORY.md removals header
    when removals is empty)."""
    record = {
        "token": "ADDONLY",
        "episodes_reviewed": 3,
        "lookback_hours": 1,
        "created_at": "2026-05-05T00:00:00+00:00",
        "proposal": {"memory_additions": ["thing"]},
    }
    out = _format_curation_proposal(record)
    assert "additions" in out.lower()
    assert "MEMORY.md removals" not in out
    assert "USER.md updates" not in out


# ---------------------------------------------------------------------
# /curate review handler
# ---------------------------------------------------------------------

def _build_bot_with_curation(pending: dict[str, dict]):
    """Construct a SentinelTelegramBot stand-in with just the bits the
    review handler touches. Avoids needing real Telegram tokens or
    Application setup."""
    bot = SentinelTelegramBot.__new__(SentinelTelegramBot)
    bot.curation = MagicMock()
    bot.curation.list_pending = MagicMock(return_value=list(pending.keys()))
    bot.curation.get_pending = MagicMock(side_effect=pending.get)
    # _send_long is a method; mock it as async
    bot._send_long = AsyncMock()
    return bot


def test_curate_review_no_pending_says_so():
    bot = _build_bot_with_curation({})
    update = SimpleNamespace(message=SimpleNamespace(reply_text=AsyncMock()))
    asyncio.run(bot._handle_curate_review(update))
    update.message.reply_text.assert_awaited_once()
    body = update.message.reply_text.await_args.args[0]
    assert "No pending" in body
    bot._send_long.assert_not_called()


def test_curate_review_lists_each_pending_proposal():
    pending = {
        "TOK1": {
            "token": "TOK1",
            "episodes_reviewed": 4,
            "lookback_hours": 24,
            "created_at": "2026-05-05T00:00:00+00:00",
            "proposal": {"memory_additions": ["fact one"]},
        },
        "TOK2": {
            "token": "TOK2",
            "episodes_reviewed": 9,
            "lookback_hours": 24,
            "created_at": "2026-05-05T01:00:00+00:00",
            "proposal": {"memory_additions": ["fact two"],
                         "memory_removals": ["fact stale"]},
        },
    }
    bot = _build_bot_with_curation(pending)
    update = SimpleNamespace(message=SimpleNamespace(reply_text=AsyncMock()))
    asyncio.run(bot._handle_curate_review(update))
    # _send_long called once per pending proposal
    assert bot._send_long.await_count == 2
    bodies = [c.args[1] for c in bot._send_long.await_args_list]
    assert any("fact one" in b for b in bodies)
    assert any("fact two" in b for b in bodies)
    assert any("fact stale" in b for b in bodies)


def test_curate_review_skips_token_that_disappears_mid_iteration():
    """Race-tolerance: list_pending returns a token, get_pending returns
    None (someone /curate_approve'd it between the two calls). Should
    skip that one quietly, not crash."""
    pending = {
        "STILL": {
            "token": "STILL",
            "episodes_reviewed": 1,
            "lookback_hours": 24,
            "created_at": "2026-05-05T00:00:00+00:00",
            "proposal": {"memory_additions": ["surviving"]},
        },
    }
    bot = SentinelTelegramBot.__new__(SentinelTelegramBot)
    bot.curation = MagicMock()
    bot.curation.list_pending = MagicMock(
        return_value=["GONE", "STILL"],
    )
    bot.curation.get_pending = MagicMock(
        side_effect=lambda t: pending.get(t),  # returns None for GONE
    )
    bot._send_long = AsyncMock()

    update = SimpleNamespace(message=SimpleNamespace(reply_text=AsyncMock()))
    asyncio.run(bot._handle_curate_review(update))
    # Only the surviving proposal renders.
    assert bot._send_long.await_count == 1
    body = bot._send_long.await_args.args[1]
    assert "surviving" in body
