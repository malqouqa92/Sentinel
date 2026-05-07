"""Phase 11 -- scheduler tests A-J.

ECC coverage: every input/output equivalence class for the scheduler
is exercised here. No real LLM, no real Ollama, no real HTTP. The
worker pipeline is mocked via a fake `route` function that synthesizes
task rows directly in the test DB.

Tests use the autouse `temp_db` fixture from conftest, which gives
each test an isolated SQLite file with the full schema (now including
scheduled_jobs + job_runs).
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, database
from core.scheduler import (
    INTERNAL_HANDLERS,
    Scheduler,
    _within_active_hours,
    compute_next_run,
    parse_active_hours,
    parse_interval,
)


# ---------------------------------------------------------------------
# Pure parsing / time math
# ---------------------------------------------------------------------

def test_a_cron_next_run_weekday_morning():
    """Test A: cron '0 7 * * 1-5' from a Sunday returns next Monday 07:00 EST."""
    # Sun 2026-05-03 12:00 UTC = Sun 08:00 EDT (May = DST in NY)
    base = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)
    nxt = compute_next_run("cron", "0 7 * * 1-5", base=base)
    # Mon 2026-05-04 07:00 EDT = 11:00 UTC
    assert nxt == datetime(2026, 5, 4, 11, 0, tzinfo=timezone.utc), nxt


def test_a2_cron_skips_weekend():
    """Cron '0 7 * * 1-5' from a Friday afternoon returns Monday."""
    base = datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc)  # Fri 14:00 EDT
    nxt = compute_next_run("cron", "0 7 * * 1-5", base=base)
    # Next weekday Mon 2026-05-04 07:00 EDT = 11:00 UTC
    assert nxt == datetime(2026, 5, 4, 11, 0, tzinfo=timezone.utc), nxt


def test_b_interval_next_run():
    """Test B: interval '30m' returns base + 30 minutes."""
    base = datetime(2026, 5, 5, 10, 0, tzinfo=timezone.utc)
    nxt = compute_next_run("interval", "30m", base=base)
    assert nxt == base + timedelta(minutes=30)


def test_b2_interval_other_units():
    base = datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc)
    assert compute_next_run("interval", "45s", base=base) == base + timedelta(seconds=45)
    assert compute_next_run("interval", "2h", base=base) == base + timedelta(hours=2)
    assert compute_next_run("interval", "1d", base=base) == base + timedelta(days=1)


def test_b3_interval_invalid_raises():
    with pytest.raises(ValueError):
        parse_interval("xyz")
    with pytest.raises(ValueError):
        parse_interval("0m")
    with pytest.raises(ValueError):
        parse_interval("-5m")


def test_c_once_with_iso_timezone():
    """Test C: one-shot ISO datetime with explicit UTC tz round-trips."""
    nxt = compute_next_run("once", "2026-05-06T10:00:00+00:00")
    assert nxt == datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc)


def test_c2_once_with_naive_iso_assumed_utc():
    """Naive ISO is assumed UTC."""
    nxt = compute_next_run("once", "2026-05-06T10:00:00")
    assert nxt == datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc)


def test_active_hours_normal_window():
    """07:00-22:00 EST: 12:00 EST inside, 03:00 EST outside."""
    job = {"active_hours_start": "07:00", "active_hours_end": "22:00"}
    # 16:00 UTC = 12:00 EDT (May)
    assert _within_active_hours(job, datetime(2026, 5, 5, 16, 0, tzinfo=timezone.utc))
    # 07:00 UTC = 03:00 EDT
    assert not _within_active_hours(job, datetime(2026, 5, 5, 7, 0, tzinfo=timezone.utc))


def test_active_hours_wraps_midnight():
    """22:00-02:00 EST window covers nights."""
    job = {"active_hours_start": "22:00", "active_hours_end": "02:00"}
    # 04:00 UTC = 00:00 EDT -> inside
    assert _within_active_hours(job, datetime(2026, 5, 5, 4, 0, tzinfo=timezone.utc))
    # 14:00 UTC = 10:00 EDT -> outside
    assert not _within_active_hours(job, datetime(2026, 5, 5, 14, 0, tzinfo=timezone.utc))


def test_active_hours_unset_passes():
    job = {"active_hours_start": None, "active_hours_end": None}
    assert _within_active_hours(job, datetime(2026, 5, 5, 3, 0, tzinfo=timezone.utc))


# ---------------------------------------------------------------------
# Helpers + fixtures for execution tests
# ---------------------------------------------------------------------

def _add_job(name, schedule_type, schedule_value, command,
             next_run_at=None, **kwargs):
    """Insert a job; default next_run_at = now (so it's due immediately)."""
    if next_run_at is None:
        next_run_at = datetime.now(timezone.utc).isoformat()
    return database.add_job(
        name=name,
        schedule_type=schedule_type,
        schedule_value=schedule_value,
        command=command,
        next_run_at=next_run_at,
        **kwargs,
    )


def _make_scheduler(route_fn=None, wait_fn=None, bot=None):
    if route_fn is None:
        route_fn = MagicMock(return_value=SimpleNamespace(
            status="ok", task_id="t-stub", trace_id="SEN-stub",
            message="routed",
        ))
    if wait_fn is None:
        async def _wait(task_id, timeout):
            return {"status": "completed", "result": {"ok": True}}
        wait_fn = _wait
    return Scheduler(
        router_fn=route_fn,
        wait_for_task_fn=wait_fn,
        shutdown_event=asyncio.Event(),
        bot=bot,
    )


# ---------------------------------------------------------------------
# Test D -- successful routed execution updates DB rows
# ---------------------------------------------------------------------

def _drain(sched):
    async def _go():
        await sched._tick()
        if sched._running_tasks:
            await asyncio.gather(*list(sched._running_tasks))
    asyncio.run(_go())


def test_d_due_job_executes_via_route():
    """Test D: a due job calls route(), records run row, advances next_run."""
    job_id = _add_job("Morning ping", "interval", "30m", "/ping")
    route_calls = []
    def fake_route(cmd):
        route_calls.append(cmd)
        return SimpleNamespace(
            status="ok", task_id="task-D", trace_id="SEN-D",
            message="routed",
        )
    async def fake_wait(task_id, timeout):
        assert task_id == "task-D"
        return {"status": "completed", "result": {"ok": "done"}}
    sched = _make_scheduler(fake_route, fake_wait)

    _drain(sched)

    assert route_calls == ["/ping"]
    runs = database.last_runs(job_id)
    assert len(runs) == 1 and runs[0]["status"] == "completed"
    job = database.get_job(job_id)
    assert job["last_status"] == "completed"
    nxt = datetime.fromisoformat(job["next_run_at"])
    assert nxt > datetime.now(timezone.utc) + timedelta(minutes=29)


# ---------------------------------------------------------------------
# Test E -- skip-if-running
# ---------------------------------------------------------------------

def test_e_skip_if_previous_run_still_active():
    """Test E: when a prior job_runs row is still 'running', the next
    tick records a skip + advances next_run + does NOT call route."""
    job_id = _add_job("Long-running", "interval", "5m", "/ping")
    database.start_job_run(job_id, "SEN-prior")

    route_called = MagicMock()
    sched = _make_scheduler(route_fn=route_called)
    _drain(sched)

    route_called.assert_not_called()
    runs = database.last_runs(job_id)
    statuses = [r["status"] for r in runs]
    assert "skipped" in statuses
    skip_row = next(r for r in runs if r["status"] == "skipped")
    assert "previous run still active" in (skip_row["error"] or "")
    job = database.get_job(job_id)
    assert job["last_status"] == "skipped"


# ---------------------------------------------------------------------
# Test F -- active hours
# ---------------------------------------------------------------------

def test_f_outside_active_hours_skipped_no_run_row():
    """Test F: due job at 03:00 EDT with hours 07:00-22:00 is NOT executed
    (no run row written); next_run is advanced."""
    # Make 'now' (default = real wall-clock) irrelevant by pinning the
    # job's schedule + active hours such that it can never be in-window.
    # Easier: set a window that excludes ALL hours by making it 0-0 (which
    # is technically a 1-minute window). To avoid clock dependence, use a
    # window that's the inverse of the current hour.
    now = datetime.now(timezone.utc)
    from zoneinfo import ZoneInfo
    local_now = now.astimezone(ZoneInfo(config.SCHEDULER_TIMEZONE))
    # window = exactly 01:00-01:01 in local time, almost certainly excluding now.
    if 1 <= local_now.hour < 2:  # current hour is 01:xx; pick 14:00-15:00 instead
        start, end = "14:00", "15:00"
    else:
        start, end = "01:00", "01:01"

    job_id = _add_job(
        "Off-hours", "interval", "5m", "/ping",
        active_hours_start=start, active_hours_end=end,
    )
    sched = _make_scheduler()
    _drain(sched)

    runs = database.last_runs(job_id)
    assert runs == [], f"expected no run rows, got {runs}"
    job = database.get_job(job_id)
    nxt = datetime.fromisoformat(job["next_run_at"])
    assert nxt > now + timedelta(minutes=4)


# ---------------------------------------------------------------------
# Test G -- one-shot with delete_after_run
# ---------------------------------------------------------------------

def test_g_one_shot_delete_after_run():
    """Test G: a one-shot job with delete_after_run=1 is removed after a
    successful execution."""
    past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    job_id = _add_job(
        "Boom-and-gone", "once", past, "/ping",
        delete_after_run=True, next_run_at=past,
    )
    sched = _make_scheduler()
    _drain(sched)

    assert database.get_job(job_id) is None, "job should have been deleted"


def test_g2_one_shot_without_delete_just_disables():
    """Without delete_after_run, one-shot stays in DB but enabled=0."""
    past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    job_id = _add_job(
        "Boom-but-stays", "once", past, "/ping",
        delete_after_run=False, next_run_at=past,
    )
    sched = _make_scheduler()
    _drain(sched)

    job = database.get_job(job_id)
    assert job is not None
    assert job["enabled"] == 0, "one-shot should disable after firing"


# ---------------------------------------------------------------------
# Tests H, I, J -- /schedule subcommand parsing + DB effects
# ---------------------------------------------------------------------

def test_h_parse_schedule_add_cron_with_hours():
    from interfaces.telegram_bot import _parse_schedule_add
    out = _parse_schedule_add([
        "Morning", "--cron", "0 7 * * 1-5",
        "--hours", "7-22",
        "--command", "/jobsearch", "RSM", "Michigan",
    ])
    assert out["name"] == "Morning"
    assert out["schedule_type"] == "cron"
    assert out["schedule_value"] == "0 7 * * 1-5"
    assert out["command"] == "/jobsearch RSM Michigan"
    assert out["active_hours_start"] == "07:00"
    assert out["active_hours_end"] == "22:00"
    assert out["session_type"] == "main"
    assert out["delete_after_run"] is False


def test_h2_parse_schedule_add_interval_isolated():
    from interfaces.telegram_bot import _parse_schedule_add
    out = _parse_schedule_add([
        "Quick", "--interval", "30m",
        "--isolated",
        "--command", "/status",
    ])
    assert out["schedule_type"] == "interval"
    assert out["schedule_value"] == "30m"
    assert out["session_type"] == "isolated"


def test_h3_parse_schedule_add_once_delete_after():
    from interfaces.telegram_bot import _parse_schedule_add
    out = _parse_schedule_add([
        "OneShot", "--once", "2026-05-06T10:00:00+00:00",
        "--delete-after",
        "--command", "/research",
    ])
    assert out["schedule_type"] == "once"
    assert out["delete_after_run"] is True


def test_h4_parse_schedule_add_rejects_missing_command():
    from interfaces.telegram_bot import _ScheduleArgError, _parse_schedule_add
    with pytest.raises(_ScheduleArgError):
        _parse_schedule_add(["Bad", "--cron", "0 7 * * *"])


def test_h5_parse_schedule_add_rejects_missing_schedule():
    from interfaces.telegram_bot import _ScheduleArgError, _parse_schedule_add
    with pytest.raises(_ScheduleArgError):
        _parse_schedule_add(["Bad", "--command", "/ping"])


def test_h6_parse_schedule_add_rejects_non_slash_command():
    from interfaces.telegram_bot import _ScheduleArgError, _parse_schedule_add
    with pytest.raises(_ScheduleArgError):
        _parse_schedule_add([
            "Bad", "--cron", "0 7 * * *",
            "--command", "echo hi",
        ])


def test_h7_parse_schedule_add_rejects_unknown_flag():
    from interfaces.telegram_bot import _ScheduleArgError, _parse_schedule_add
    with pytest.raises(_ScheduleArgError):
        _parse_schedule_add([
            "Bad", "--cron", "0 7 * * *",
            "--banana",
            "--command", "/ping",
        ])


def test_i_pause_disables_job_in_db():
    """Test I: pause subcommand sets enabled=0."""
    job_id = _add_job("Paused", "interval", "30m", "/ping")
    database.set_job_enabled(job_id, False)
    j = database.get_job(job_id)
    assert j["enabled"] == 0


def test_j_resume_re_enables_and_recomputes_next_run():
    """Test J: resume sets enabled=1 AND recomputes next_run from now
    (so a long-paused job doesn't immediately fire on every poll)."""
    far_past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    job_id = _add_job(
        "Stale", "interval", "30m", "/ping", next_run_at=far_past,
    )
    database.set_job_enabled(job_id, False)
    # Re-enable + recompute
    database.set_job_enabled(job_id, True)
    new_nxt = compute_next_run("interval", "30m")
    database.set_next_run(job_id, new_nxt)

    j = database.get_job(job_id)
    assert j["enabled"] == 1
    assert datetime.fromisoformat(j["next_run_at"]) > \
        datetime.now(timezone.utc) + timedelta(minutes=29)


# ---------------------------------------------------------------------
# Bonus -- internal handler dispatch (bypasses router)
# ---------------------------------------------------------------------

def test_internal_handler_invoked_directly_no_router():
    """Job with command starting /internal_<name> calls the registered
    Python function in INTERNAL_HANDLERS, not the router."""
    captured = []

    async def my_handler(arg: str) -> str:
        captured.append(arg)
        return f"handled {arg!r}"

    INTERNAL_HANDLERS["unittest_probe"] = my_handler
    try:
        job_id = _add_job(
            "Probe", "interval", "1h",
            "/internal_unittest_probe hello world",
        )
        route_called = MagicMock(side_effect=AssertionError(
            "route should not be called for /internal_*"
        ))
        sched = _make_scheduler(route_fn=route_called)
        _drain(sched)

        assert captured == ["hello world"]
        runs = database.last_runs(job_id)
        assert len(runs) == 1 and runs[0]["status"] == "completed"
        assert "handled" in (runs[0]["result_summary"] or "")
    finally:
        INTERNAL_HANDLERS.pop("unittest_probe", None)


def test_route_failure_marks_run_failed_and_alerts_bot():
    """Routed command returning status='error' -> run row 'failed' +
    bot.send_alert called once."""
    job_id = _add_job("Bad", "interval", "30m", "/ping")
    bot = SimpleNamespace(send_alert=AsyncMock())
    bad_route = MagicMock(return_value=SimpleNamespace(
        status="error", task_id=None, trace_id="SEN-x",
        message="nope",
    ))
    sched = _make_scheduler(route_fn=bad_route, bot=bot)
    _drain(sched)

    runs = database.last_runs(job_id)
    assert len(runs) == 1 and runs[0]["status"] == "failed"
    bot.send_alert.assert_awaited_once()
    msg = bot.send_alert.await_args.args[0]
    assert "failed" in msg.lower()


def test_isolated_session_does_not_alert():
    """session_type='isolated' suppresses Telegram alerts on completion."""
    job_id = _add_job(
        "Quiet", "interval", "30m", "/ping",
        session_type="isolated",
    )
    bot = SimpleNamespace(send_alert=AsyncMock())
    sched = _make_scheduler(bot=bot)
    _drain(sched)

    bot.send_alert.assert_not_called()
    runs = database.last_runs(job_id)
    assert runs[0]["status"] == "completed"
