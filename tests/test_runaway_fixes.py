"""Phase 12.5 hardening -- the three runaway-prevention fixes:

  Fix 1: worker.execute_task sleeps WORKER_GPU_REQUEUE_BACKOFF_S after
         requeue-on-busy so the loop can't hammer at 50/sec.
  Fix 2: database.recover_stale releases locks held by tasks that are
         not currently 'processing' (orphan detection), even if the
         lock is fresh.
  Fix 3: telegram_bot.handle_restart waits up to RESTART_DRAIN_TIMEOUT_S
         for in-flight tasks before triggering shutdown.

All mocked. No real LLM, no real Telegram, no real subprocess.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, database


# ---------------------------------------------------------------------
# Fix 1 -- worker backoff
# ---------------------------------------------------------------------

def test_worker_backoff_after_gpu_busy_requeue(monkeypatch):
    """execute_task must sleep WORKER_GPU_REQUEUE_BACKOFF_S after a
    requeue-on-busy. We measure by capturing the asyncio.sleep arg."""
    from core import worker
    from core.database import TaskRow

    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    # Force acquire_lock to fail (simulate busy).
    monkeypatch.setattr(database, "acquire_lock", lambda *_: False)
    monkeypatch.setattr(database, "requeue_task", lambda *_: None)
    monkeypatch.setattr("core.orchestrator.needs_gpu", lambda *_: True)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    task = TaskRow(
        task_id="t-x", trace_id="SEN-x", command="/code", args={},
        status="processing", priority=0,
        retry_count=0, max_retries=3,
        recovery_count=0, max_recoveries=5,
        result=None, error=None,
        created_at="2026-05-05T00:00:00+00:00",
        updated_at="2026-05-05T00:00:00+00:00",
    )
    asyncio.run(worker.execute_task(task))

    assert config.WORKER_GPU_REQUEUE_BACKOFF_S in sleeps, (
        f"expected backoff sleep; got sleeps={sleeps}"
    )


def test_worker_backoff_constant_is_at_least_one_second():
    """Sanity: the constant must be >= 1s. A value < 1s would still
    spam the loop fast enough to OOM the log."""
    assert config.WORKER_GPU_REQUEUE_BACKOFF_S >= 1.0


# ---------------------------------------------------------------------
# Fix 2 -- recover_stale orphan-lock detection
# ---------------------------------------------------------------------

def _seed_lock_with_task_status(task_id: str, status: str,
                                lock_ts_iso: str | None = None):
    """Insert a tasks row + a lock row pointing to it."""
    if lock_ts_iso is None:
        # FRESH (within STALE_LOCK_TIMEOUT) so the old code wouldn't
        # touch this lock. The orphan path must release it anyway.
        lock_ts_iso = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.execute(
            "INSERT INTO tasks (task_id, trace_id, command, args, status, "
            "priority, retry_count, max_retries, recovery_count, "
            "max_recoveries, result, error, created_at, updated_at) "
            "VALUES (?, ?, '/code', '{}', ?, 0, 0, 3, 0, 5, NULL, NULL, "
            "?, ?)",
            (task_id, "SEN-test", status,
             datetime.now(timezone.utc).isoformat(),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.execute(
            "INSERT INTO locks (resource, locked_by, locked_at) "
            "VALUES ('gpu', ?, ?)",
            (task_id, lock_ts_iso),
        )
        conn.commit()
    finally:
        conn.close()


def test_recover_stale_releases_orphan_lock_on_failed_task():
    """A FRESH lock held by a task whose row says 'failed' must be
    released by the orphan-detection path."""
    _seed_lock_with_task_status("orph-failed", "failed")
    assert database.get_lock("gpu") is not None
    summary = database.recover_stale()
    assert summary["locks_released"] >= 1
    assert database.get_lock("gpu") is None


def test_recover_stale_releases_lock_when_task_row_missing():
    """A lock held by a task_id with NO matching row in tasks table
    is also an orphan. (Could happen after manual DB surgery.)"""
    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.execute(
            "INSERT INTO locks (resource, locked_by, locked_at) "
            "VALUES ('gpu', 'ghost-task-id', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()
    finally:
        conn.close()
    summary = database.recover_stale()
    assert summary["locks_released"] >= 1
    assert database.get_lock("gpu") is None


def test_recover_stale_does_not_release_lock_for_processing_task():
    """The legitimate case: a task is genuinely processing and holds
    a fresh GPU lock. recover_stale must NOT touch it."""
    _seed_lock_with_task_status("alive", "processing")
    summary = database.recover_stale()
    assert summary["locks_released"] == 0
    assert database.get_lock("gpu") is not None
    assert database.get_lock("gpu")["locked_by"] == "alive"


def test_recover_stale_releases_old_lock_unchanged():
    """Pre-existing behavior preserved: locks older than
    STALE_LOCK_TIMEOUT get released even if the task is 'processing'."""
    old_ts = (
        datetime.now(timezone.utc)
        - timedelta(seconds=config.STALE_LOCK_TIMEOUT + 60)
    ).isoformat()
    _seed_lock_with_task_status("old-task", "processing", lock_ts_iso=old_ts)
    summary = database.recover_stale()
    assert summary["locks_released"] >= 1
    assert database.get_lock("gpu") is None


# ---------------------------------------------------------------------
# Fix 3 -- handle_restart drain wait
# ---------------------------------------------------------------------

def test_handle_restart_waits_for_in_flight_then_proceeds(monkeypatch):
    """When 1 task is in flight, handle_restart should warn the user,
    poll until count drops to 0, then proceed with shutdown."""
    from interfaces.telegram_bot import SentinelTelegramBot

    bot = SentinelTelegramBot.__new__(SentinelTelegramBot)
    bot._check_auth = AsyncMock(return_value=True)

    # First two polls: 1 in flight; third: 0 (drained).
    counts = iter([1, 1, 1, 0])
    monkeypatch.setattr(
        database, "count_tasks_by_status",
        lambda status: next(counts) if status == "processing" else 0,
    )
    monkeypatch.setattr(config, "RESTART_DRAIN_TIMEOUT_S", 30)
    # Stub out the spawn + shutdown so the test stays in-process.
    sleep_calls: list[float] = []
    real_sleep = asyncio.sleep
    async def fake_sleep(s):
        sleep_calls.append(s)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        "subprocess.Popen", MagicMock(return_value=MagicMock()),
    )
    create_task_calls: list = []
    monkeypatch.setattr(
        asyncio, "create_task",
        lambda coro: create_task_calls.append(coro) or MagicMock(),
    )

    update = SimpleNamespace(message=SimpleNamespace(reply_text=AsyncMock()))
    real_sleep_run = asyncio.run
    real_sleep_run(bot.handle_restart(
        update, SimpleNamespace(args=[]),
    ))

    # Three messages should have been sent: drain-warning, then OK
    # (no still-running warning), then "Restarting...".
    assert update.message.reply_text.await_count >= 2
    msgs = [c.args[0] for c in update.message.reply_text.await_args_list]
    assert any("in flight" in m for m in msgs), msgs
    assert any("Restarting" in m for m in msgs), msgs


def test_handle_restart_proceeds_with_warning_on_timeout(monkeypatch):
    """If the drain wait expires with tasks still running, warn the
    user but still proceed to shutdown."""
    from interfaces.telegram_bot import SentinelTelegramBot

    bot = SentinelTelegramBot.__new__(SentinelTelegramBot)
    bot._check_auth = AsyncMock(return_value=True)

    monkeypatch.setattr(
        database, "count_tasks_by_status",
        lambda status: 1 if status == "processing" else 0,
    )
    monkeypatch.setattr(config, "RESTART_DRAIN_TIMEOUT_S", 0.01)
    monkeypatch.setattr(
        "subprocess.Popen", MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        asyncio, "create_task",
        lambda coro: MagicMock(),
    )

    update = SimpleNamespace(message=SimpleNamespace(reply_text=AsyncMock()))
    asyncio.run(bot.handle_restart(update, SimpleNamespace(args=[])))

    msgs = [c.args[0] for c in update.message.reply_text.await_args_list]
    assert any("still running" in m for m in msgs), msgs
    assert any("Restarting" in m for m in msgs), msgs


def test_handle_restart_no_drain_wait_when_idle(monkeypatch):
    """If no tasks are in flight, the drain block should be skipped
    (no '⏳ ... in flight' message)."""
    from interfaces.telegram_bot import SentinelTelegramBot

    bot = SentinelTelegramBot.__new__(SentinelTelegramBot)
    bot._check_auth = AsyncMock(return_value=True)
    monkeypatch.setattr(
        database, "count_tasks_by_status",
        lambda status: 0,
    )
    monkeypatch.setattr(
        "subprocess.Popen", MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        asyncio, "create_task",
        lambda coro: MagicMock(),
    )
    update = SimpleNamespace(message=SimpleNamespace(reply_text=AsyncMock()))
    asyncio.run(bot.handle_restart(update, SimpleNamespace(args=[])))

    msgs = [c.args[0] for c in update.message.reply_text.await_args_list]
    assert not any("in flight" in m for m in msgs), msgs
    assert any("Restarting" in m for m in msgs), msgs
