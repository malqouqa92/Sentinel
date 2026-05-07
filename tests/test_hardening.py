"""Phase 11 -- hardening + integration tests.

Q-X. Mocks all I/O at boundary edges; doesn't touch real Telegram,
real Ollama, or real Claude CLI. Backup test uses tmp directories.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, database
from core import internal_handlers as _ih  # noqa: F401 -- registers handlers
from core.scheduler import INTERNAL_HANDLERS


# ---------------------------------------------------------------------
# Test Q -- log rotation creates backup file when size exceeds threshold
# ---------------------------------------------------------------------

def test_q_log_rotation_creates_backup(tmp_path):
    """RotatingFileHandler-equivalent rolls when current file > maxBytes
    and writes continue to a fresh file. Use a fresh handler in tmp_path
    to avoid races with the long-running bot's log file."""
    log_path = tmp_path / "test.jsonl"
    handler = RotatingFileHandler(
        str(log_path), mode="a", encoding="utf-8",
        maxBytes=200, backupCount=3,
    )
    import logging as _l
    logger = _l.getLogger("rotation_test")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(_l.INFO)
    try:
        # Write enough data to force at least one rollover.
        for i in range(40):
            logger.info(json.dumps({"i": i, "padding": "x" * 30}))
        handler.close()
        # Rolled file (.1) should exist and current file should be small.
        assert log_path.exists(), "current log file missing"
        rolled = list(tmp_path.glob("test.jsonl.*"))
        assert len(rolled) >= 1, f"expected rotated sibling, got {rolled}"
    finally:
        handler.close()


# ---------------------------------------------------------------------
# Test R -- /internal_wal_checkpoint runs without error on all DBs
# ---------------------------------------------------------------------

def test_r_wal_checkpoint_internal_handler_runs():
    handler = INTERNAL_HANDLERS["wal_checkpoint"]
    result = asyncio.run(handler(""))
    # Each configured DB should report ok or fail; never crash.
    # conftest monkeypatches DB_PATH/MEMORY_DB_PATH to tmp files, so use
    # the actual configured paths to build expectations.
    for p in (config.DB_PATH, config.MEMORY_DB_PATH, config.KNOWLEDGE_DB_PATH):
        assert p.name in result, f"missing {p.name} in {result!r}"


# ---------------------------------------------------------------------
# Test S -- low disk free triggers Telegram alert
# ---------------------------------------------------------------------

def test_s_low_disk_triggers_alert(monkeypatch):
    from core import internal_handlers
    # Patch shutil.disk_usage seen by internal_handlers.
    monkeypatch.setattr(
        "core.internal_handlers.shutil.disk_usage",
        lambda *_a, **_k: type("DU", (), {
            "free": 100_000_000,  # 100 MB << 1 GB threshold
            "used": 1, "total": 1,
        })(),
    )
    sent: list[str] = []

    async def fake_alert(msg: str) -> None:
        sent.append(msg)

    internal_handlers._set_alert_callback(fake_alert)
    try:
        result = asyncio.run(
            internal_handlers.resource_check("")
        )
    finally:
        internal_handlers._set_alert_callback(None)
    assert any("LOW DISK SPACE" in m for m in sent), sent
    assert "disk_low" in result


# ---------------------------------------------------------------------
# Tests T, U -- backup creates files + retention prunes
# ---------------------------------------------------------------------

def test_t_backup_writes_sqlite_and_persona_files(tmp_path, monkeypatch):
    # Redirect every path the backup handler touches into tmp_path.
    src_db = tmp_path / "sentinel.db"
    src_mem = tmp_path / "memory.db"
    src_kb = tmp_path / "knowledge.db"
    for p in (src_db, src_mem, src_kb):
        with sqlite3.connect(str(p)) as c:
            c.executescript(
                "CREATE TABLE x (id INTEGER); INSERT INTO x VALUES (1);"
            )
    backup_root = tmp_path / "backups"
    persona_src = tmp_path / "persona"
    persona_src.mkdir()
    for name in ("IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md"):
        (persona_src / name).write_text(f"# {name}\n", encoding="utf-8")

    monkeypatch.setattr(config, "DB_PATH", src_db)
    monkeypatch.setattr(config, "MEMORY_DB_PATH", src_mem)
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", src_kb)
    monkeypatch.setattr(config, "BACKUP_DIR", backup_root)
    monkeypatch.setattr(config, "PERSONA_DIR", persona_src)
    monkeypatch.setattr(config, "BACKUP_KEEP_DAYS", 7)

    from core import internal_handlers
    summary = asyncio.run(internal_handlers.backup(""))
    assert "copied=" in summary
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_dir = backup_root / today
    assert today_dir.exists()
    # All three SQLite DBs and four persona files copied.
    for name in ("sentinel.db", "memory.db", "knowledge.db"):
        assert (today_dir / name).exists(), f"missing {name}"
    for name in ("IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md"):
        assert (today_dir / "persona" / name).exists(), f"missing persona/{name}"


def test_u_backup_retention_prunes_older_than_keep_days(
    tmp_path, monkeypatch,
):
    backup_root = tmp_path / "backups"
    backup_root.mkdir()
    # Seed: 9 day-folders (today + 8 older). With keep_days=7, cutoff is
    # today-7 and the prune deletes anything STRICTLY older than that.
    # So delta=8 (today-8) is the only one removed; today..today-7 stay.
    today = datetime.now(timezone.utc).date()
    for delta in range(9):
        d = today - timedelta(days=delta)
        (backup_root / d.strftime("%Y-%m-%d")).mkdir()
    # Also seed one bogus folder that doesn't match the date pattern --
    # should be left alone.
    (backup_root / "not-a-date").mkdir()

    monkeypatch.setattr(config, "BACKUP_DIR", backup_root)
    monkeypatch.setattr(config, "BACKUP_KEEP_DAYS", 7)
    # Skip SQLite copies so the test stays focused on retention.
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "absent.db")
    monkeypatch.setattr(config, "MEMORY_DB_PATH", tmp_path / "absent.db")
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", tmp_path / "absent.db")
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path / "missing-persona")

    from core import internal_handlers
    summary = asyncio.run(internal_handlers.backup(""))

    # delta=8 should be gone; today..today-7 (8 folders) remain plus bogus.
    remaining = sorted(
        d.name for d in backup_root.iterdir() if d.is_dir()
    )
    expected = sorted(
        [(today - timedelta(days=i)).strftime("%Y-%m-%d")
         for i in range(8)] + ["not-a-date"]
    )
    assert remaining == expected
    assert "pruned=1" in summary


# ---------------------------------------------------------------------
# Test V -- end-to-end: scheduled job fires, result delivered to bot
# ---------------------------------------------------------------------

def test_v_e2e_scheduled_job_delivers_via_bot_alert():
    """End-to-end with everything mocked at boundaries: scheduler ticks,
    fake route returns success, bot.send_alert receives the completion
    message."""
    from core.scheduler import Scheduler
    bot = MagicMock()
    bot.send_alert = AsyncMock()
    fake_route = MagicMock(return_value=type("RR", (), {
        "status": "ok", "task_id": "task-V",
        "trace_id": "SEN-V", "message": "routed",
    })())

    async def fake_wait(task_id, timeout):
        return {
            "status": "completed",
            "result": {"summary": "everything fine"},
        }

    sched = Scheduler(
        router_fn=fake_route,
        wait_for_task_fn=fake_wait,
        shutdown_event=asyncio.Event(),
        bot=bot,
    )
    job_id = database.add_job(
        name="E2E", schedule_type="interval", schedule_value="1h",
        command="/ping",
        next_run_at=datetime.now(timezone.utc).isoformat(),
    )

    async def _drain():
        await sched._tick()
        if sched._running_tasks:
            await asyncio.gather(*list(sched._running_tasks))

    asyncio.run(_drain())

    # Route was called with the job's command
    fake_route.assert_called_once_with("/ping")
    # Bot received an alert containing the job name
    bot.send_alert.assert_awaited_once()
    msg = bot.send_alert.await_args.args[0]
    assert "E2E" in msg
    assert "completed" in msg.lower()
    # DB reflects success
    runs = database.last_runs(job_id)
    assert len(runs) == 1 and runs[0]["status"] == "completed"


# ---------------------------------------------------------------------
# Test W -- shutdown event causes scheduler_loop to exit promptly
# ---------------------------------------------------------------------

def test_w_shutdown_event_exits_scheduler_loop_quickly():
    """scheduler_loop should respond to shutdown_event within ~1s,
    not wait the full SCHEDULER_POLL_INTERVAL (30s) -- it's the
    asyncio.wait_for trick that gives us this property."""
    from core.scheduler import Scheduler
    sched = Scheduler(
        router_fn=MagicMock(),
        wait_for_task_fn=AsyncMock(),
        shutdown_event=asyncio.Event(),
    )

    async def _go():
        loop_task = asyncio.create_task(sched.scheduler_loop())
        # Let the first tick start
        await asyncio.sleep(0.2)
        sched.shutdown_event.set()
        # Should exit promptly
        await asyncio.wait_for(loop_task, timeout=2.0)

    # If this hangs/times out, the loop didn't exit on shutdown_event.
    asyncio.run(_go())


# ---------------------------------------------------------------------
# Test X -- file_guard_check + curate handlers don't crash when the
# Phase 10 components aren't installed
# ---------------------------------------------------------------------

def test_x_file_guard_check_handles_missing_install():
    from core import file_guard
    # Force "no FileGuard installed" path.
    saved = file_guard.FILE_GUARD
    file_guard.FILE_GUARD = None
    try:
        result = asyncio.run(INTERNAL_HANDLERS["file_guard_check"](""))
        assert "not installed" in result
    finally:
        file_guard.FILE_GUARD = saved


def test_x2_curate_handles_missing_install():
    from core import curation
    saved = curation.CURATION
    curation.CURATION = None
    try:
        result = asyncio.run(INTERNAL_HANDLERS["curate"](""))
        assert "not installed" in result
    finally:
        curation.CURATION = saved
