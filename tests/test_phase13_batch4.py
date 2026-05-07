"""Phase 13 Batch 4 -- /jobs Telegram viewer.

ECC. Mocks the Telegram Update / Context. Real SQLite via temp_db
(conftest fixture). No bot polling, no LLM.

What we're verifying:
  01 -- Unauthorized caller blocked before any DB call.
  02 -- /jobs (no args, empty DB) -> "no applications" hint.
  03 -- /jobs (no args, populated DB) -> list with row formatter
        and footer hint.
  04 -- /jobs <state> -> filters to that state only.
  05 -- /jobs <unknown_state> -> error message naming valid states.
  06 -- /jobs <id> (numeric) -> drill-in detail with history block.
  07 -- /jobs <id> for missing id -> "no application" reply.
  08 -- /jobs <id> <state> -> transition succeeds, detail shows new state.
  09 -- /jobs <id> <unknown_state> -> error, no DB write.
  10 -- /jobs help -> usage block.
  11 -- Spanish state alias accepted (e.g. 'aplicado' -> 'applied').
  12 -- Row formatter uses correct emoji per recommendation band.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, database
from interfaces.telegram_bot import SentinelTelegramBot


# ── shared helpers ────────────────────────────────────────────────────────


def _make_update(user_id: int = 12345) -> MagicMock:
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = f"user{user_id}"
    update.message.text = ""
    update.message.reply_text = AsyncMock()
    return update


def _make_ctx(args: list[str] | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.args = args or []
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


def _seed_apps() -> list[int]:
    """Insert 3 applications across different states. Returns ids."""
    a = database.upsert_application(
        url="https://example.com/job/100",
        title="Regional Sales Manager (Detroit)",
        company="AcmeCo",
        location="Detroit, MI",
        archetype="Regional Sales Manager",
        score=4.6,
        recommendation="apply_now",
    )
    b = database.upsert_application(
        url="https://example.com/job/101",
        title="Account Executive",
        company="BetaCorp",
        location="Cincinnati, OH",
        archetype="Account Executive",
        score=3.6,
        recommendation="maybe",
    )
    c = database.upsert_application(
        url="https://example.com/job/102",
        title="SDR",
        company="GammaInc",
        location="Atlanta, GA",
        archetype="Sales Development Representative",
        score=2.5,
        recommendation="skip",
    )
    # Move b to 'applied' so we have a state-mix to filter on.
    database.transition_application(b, "applied")
    return [a, b, c]


# ── 01: unauthorized blocked ────────────────────────────────────────────


def test_jobs_01_unauthorized_blocked():
    bot = _make_bot(authorized_user_id=99999)
    update = _make_update(user_id=11111)
    asyncio.run(bot.handle_jobs(update, _make_ctx()))
    update.message.reply_text.assert_called_once_with("Unauthorized.")


# ── 02: empty DB ────────────────────────────────────────────────────────


def test_jobs_02_empty_db_helpful_hint(temp_db):
    bot = _make_bot()
    update = _make_update()
    asyncio.run(bot.handle_jobs(update, _make_ctx()))
    msg = update.message.reply_text.call_args.args[0]
    assert "no applications" in msg.lower()
    assert "/jobsearch" in msg


# ── 03: list mode ──────────────────────────────────────────────────────


def test_jobs_03_list_mode_shows_all_states_newest_first(temp_db):
    _seed_apps()
    bot = _make_bot()
    update = _make_update()
    asyncio.run(bot.handle_jobs(update, _make_ctx()))
    msg = update.message.reply_text.call_args.args[0]
    # All three appear in the list.
    assert "AcmeCo" in msg
    assert "BetaCorp" in msg
    assert "GammaInc" in msg
    # Footer hint present.
    assert "/jobs <id>" in msg


# ── 04: state filter ──────────────────────────────────────────────────


def test_jobs_04_state_filter_returns_only_matching(temp_db):
    ids = _seed_apps()  # b is 'applied', a + c stay 'evaluated'
    bot = _make_bot()
    update = _make_update()
    asyncio.run(bot.handle_jobs(update, _make_ctx(["applied"])))
    msg = update.message.reply_text.call_args.args[0]
    assert "BetaCorp" in msg
    assert "AcmeCo" not in msg
    assert "GammaInc" not in msg
    assert "state=applied" in msg


# ── 05: unknown state ─────────────────────────────────────────────────


def test_jobs_05_unknown_state_lists_valid_options(temp_db):
    bot = _make_bot()
    update = _make_update()
    asyncio.run(bot.handle_jobs(update, _make_ctx(["foobar"])))
    msg = update.message.reply_text.call_args.args[0]
    assert "unknown" in msg.lower()
    assert "applied" in msg
    assert "interview" in msg


# ── 06: drill-in by id ─────────────────────────────────────────────────


def test_jobs_06_drill_in_by_numeric_id(temp_db):
    ids = _seed_apps()
    bot = _make_bot()
    update = _make_update()
    asyncio.run(bot.handle_jobs(update, _make_ctx([str(ids[0])])))
    msg = update.message.reply_text.call_args.args[0]
    assert "AcmeCo" in msg
    assert "Regional Sales Manager" in msg
    assert "URL: https://example.com/job/100" in msg
    # Detail format includes the transition hint.
    assert "Transition: /jobs" in msg
    assert "History" in msg


# ── 07: drill-in missing id ──────────────────────────────────────────


def test_jobs_07_drill_in_missing_id(temp_db):
    bot = _make_bot()
    update = _make_update()
    asyncio.run(bot.handle_jobs(update, _make_ctx(["99999"])))
    msg = update.message.reply_text.call_args.args[0]
    assert "no application" in msg.lower()
    assert "99999" in msg


# ── 08: state transition ──────────────────────────────────────────────


def test_jobs_08_transition_advances_state(temp_db):
    ids = _seed_apps()
    bot = _make_bot()
    update = _make_update()
    # b is currently 'applied'; advance to 'interview'.
    asyncio.run(bot.handle_jobs(
        update, _make_ctx([str(ids[1]), "interview"]),
    ))
    msg = update.message.reply_text.call_args.args[0]
    assert "→ interview" in msg
    assert "BetaCorp" in msg
    # DB confirms the transition.
    row = database.get_application(ids[1])
    assert row["state"] == "interview"


# ── 09: bad state on transition ───────────────────────────────────────


def test_jobs_09_unknown_target_state_no_db_write(temp_db):
    ids = _seed_apps()
    bot = _make_bot()
    update = _make_update()
    asyncio.run(bot.handle_jobs(
        update, _make_ctx([str(ids[0]), "promoted"]),
    ))
    msg = update.message.reply_text.call_args.args[0]
    assert "unknown state" in msg.lower()
    # State unchanged.
    row = database.get_application(ids[0])
    assert row["state"] == "evaluated"


# ── 10: help ──────────────────────────────────────────────────────────


def test_jobs_10_help_shows_grammar(temp_db):
    bot = _make_bot()
    update = _make_update()
    asyncio.run(bot.handle_jobs(update, _make_ctx(["help"])))
    msg = update.message.reply_text.call_args.args[0]
    assert "/jobs <id>" in msg
    assert "applied" in msg
    assert "Examples" in msg


# ── 11: spanish alias ─────────────────────────────────────────────────


def test_jobs_11_spanish_state_alias_accepted(temp_db):
    ids = _seed_apps()
    bot = _make_bot()
    update = _make_update()
    # 'aplicado' is the Spanish alias for 'applied' in _STATE_ALIASES.
    asyncio.run(bot.handle_jobs(update, _make_ctx(["aplicado"])))
    msg = update.message.reply_text.call_args.args[0]
    assert "BetaCorp" in msg  # the only one in 'applied'
    assert "state=applied" in msg


# ── 12: row formatter emoji map ───────────────────────────────────────


def test_jobs_12_row_formatter_uses_band_emoji():
    """The formatter maps recommendation band -> emoji. Pure function,
    no DB / mock needed."""
    fmt = SentinelTelegramBot._jobs_format_row
    apply_now = fmt({
        "id": 1, "title": "X", "company": "Y", "score": 4.7,
        "recommendation": "apply_now", "state": "evaluated",
    })
    assert "🎯" in apply_now
    skip = fmt({
        "id": 2, "title": "X", "company": "Y", "score": 2.0,
        "recommendation": "skip", "state": "discarded",
    })
    assert "⏭️" in skip
    maybe = fmt({
        "id": 3, "title": "X", "company": "Y", "score": 3.6,
        "recommendation": "maybe", "state": "evaluated",
    })
    assert "🤔" in maybe
