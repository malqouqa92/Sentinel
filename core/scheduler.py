"""Phase 11 -- cron / interval / one-shot job scheduler.

EST-anchored: cron expressions and HH:MM active-hour windows are
interpreted in `config.SCHEDULER_TIMEZONE` (default America/New_York).
All `next_run_at` values are stored UTC ISO 8601 in the DB.

Skip-if-running is enforced by checking job_runs for an existing
`status='running'` row before firing. The worker's GPU lock owns
real serialization; this scheduler defers to it via the standard
router/queue path.

Internal maintenance commands (/internal_*) bypass the router and
call registered Python functions directly -- saves ~1-2s of queue
round-trip on jobs that are pure SQLite/disk work (WAL checkpoint,
backups, file-guard heartbeat, resource probe).
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from croniter import croniter

from core import config, database
from core.logger import log_event
from core.telemetry import generate_trace_id


# Map of /internal_<name> -> async callable. Populated at import time
# in core/orchestrator.py (Batch 3). Scheduler invokes these directly
# without going through the router or queue.
INTERNAL_HANDLERS: dict[str, Callable[[str], Awaitable[Any]]] = {}


def _tz() -> ZoneInfo:
    return ZoneInfo(config.SCHEDULER_TIMEZONE)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------
# Schedule parsing
# ---------------------------------------------------------------------

_INTERVAL_RE = re.compile(r"^\s*(\d+)\s*(s|m|h|d)\s*$", re.IGNORECASE)
_INTERVAL_UNIT_S = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_interval(value: str) -> timedelta:
    """'30m' -> 1800s. Raises ValueError on bad input."""
    m = _INTERVAL_RE.match(value)
    if not m:
        raise ValueError(
            f"interval must look like '30m'/'2h'/'45s'/'1d'; got {value!r}"
        )
    n, unit = int(m.group(1)), m.group(2).lower()
    if n <= 0:
        raise ValueError(f"interval must be positive; got {value!r}")
    return timedelta(seconds=n * _INTERVAL_UNIT_S[unit])


def parse_active_hours(s: str | None) -> dtime | None:
    """'07:00' -> datetime.time(7, 0). None passes through."""
    if s is None or s == "":
        return None
    try:
        hh, mm = s.split(":")
        return dtime(int(hh), int(mm))
    except Exception as e:
        raise ValueError(
            f"active-hours must be HH:MM (24h); got {s!r}: {e}"
        )


def compute_next_run(
    schedule_type: str,
    schedule_value: str,
    base: datetime | None = None,
) -> datetime:
    """Returns the next firing time as UTC datetime.

    - once: schedule_value is an ISO datetime (UTC if no tz info).
    - interval: schedule_value like '30m'; next = base + interval.
    - cron: schedule_value is a 5-field cron expr; evaluated in EST,
      converted to UTC.
    """
    base = base or _now_utc()
    if schedule_type == "once":
        # Accept both naive (assume UTC) and aware ISO strings.
        dt = datetime.fromisoformat(schedule_value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if schedule_type == "interval":
        return base + parse_interval(schedule_value)
    if schedule_type == "cron":
        base_local = base.astimezone(_tz())
        nxt_local = croniter(schedule_value, base_local).get_next(datetime)
        # croniter loses tz on the returned datetime; re-attach.
        if nxt_local.tzinfo is None:
            nxt_local = nxt_local.replace(tzinfo=_tz())
        return nxt_local.astimezone(timezone.utc)
    raise ValueError(f"unknown schedule_type: {schedule_type!r}")


def _within_active_hours(job: dict, now_utc: datetime) -> bool:
    """True if no active-hours window OR now-in-EST falls inside it.

    Window can wrap midnight (e.g. start=22:00, end=02:00).
    """
    start = parse_active_hours(job.get("active_hours_start"))
    end = parse_active_hours(job.get("active_hours_end"))
    if start is None or end is None:
        return True
    now_local = now_utc.astimezone(_tz()).time()
    if start <= end:
        return start <= now_local <= end
    # wraps midnight
    return now_local >= start or now_local <= end


# ---------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------

class Scheduler:
    """Polls scheduled_jobs, dispatches due ones via router_fn (for user
    commands) or INTERNAL_HANDLERS (for /internal_* maintenance).

    `bot` is set by main.py after Telegram bot is constructed (chicken-
    and-egg avoidance) and is used to send completion alerts when
    session_type='main'."""

    def __init__(
        self,
        router_fn: Callable[[str], Any],
        wait_for_task_fn: Callable[[str, float], Awaitable[dict | None]],
        shutdown_event: asyncio.Event,
        bot: Any | None = None,
    ) -> None:
        self.route = router_fn
        self.wait_for_task = wait_for_task_fn
        self.shutdown_event = shutdown_event
        self.bot = bot
        self._running_tasks: set[asyncio.Task] = set()

    # ---------------- lifecycle ----------------

    async def scheduler_loop(self) -> None:
        log_event("SEN-system", "INFO", "scheduler", "scheduler loop started")
        while not self.shutdown_event.is_set():
            try:
                await self._tick()
            except Exception as e:
                log_event("SEN-system", "ERROR", "scheduler",
                          f"tick failed: {e}")
            try:
                await asyncio.wait_for(
                    self.shutdown_event.wait(),
                    timeout=config.SCHEDULER_POLL_INTERVAL,
                )
                break
            except asyncio.TimeoutError:
                pass
        log_event("SEN-system", "INFO", "scheduler", "scheduler loop exiting")

    async def _tick(self) -> None:
        now = _now_utc()
        due = await asyncio.to_thread(database.get_due_jobs, now)
        for job in due:
            if await asyncio.to_thread(database.has_running_run, job["id"]):
                await asyncio.to_thread(
                    database.record_skip, job["id"],
                    "previous run still active",
                )
                await asyncio.to_thread(
                    database.update_job_status, job["id"], "skipped",
                )
                self._advance_next_run(job)
                log_event("SEN-system", "INFO", "scheduler",
                          f"skipped job {job['id']} '{job['name']}' "
                          f"-- previous run still active")
                continue
            if not _within_active_hours(job, now):
                self._advance_next_run(job)
                log_event("SEN-system", "DEBUG", "scheduler",
                          f"job {job['id']} '{job['name']}' "
                          f"outside active hours; advanced next_run")
                continue
            if len(self._running_tasks) >= config.SCHEDULER_MAX_CONCURRENT:
                log_event("SEN-system", "DEBUG", "scheduler",
                          f"job {job['id']} '{job['name']}' "
                          f"deferred -- {len(self._running_tasks)} "
                          f"already in flight")
                continue
            t = asyncio.create_task(self._execute(job))
            self._running_tasks.add(t)
            t.add_done_callback(self._running_tasks.discard)

    # ---------------- execution ----------------

    async def _execute(self, job: dict) -> None:
        trace_id = generate_trace_id()
        run_id = await asyncio.to_thread(
            database.start_job_run, job["id"], trace_id,
        )
        await asyncio.to_thread(
            database.update_job_status, job["id"], "running",
        )
        log_event(trace_id, "INFO", "scheduler",
                  f"executing job {job['id']} '{job['name']}': "
                  f"{job['command']}")
        try:
            cmd = job["command"]
            if cmd.startswith("/internal_"):
                summary = await self._run_internal(cmd, trace_id)
            else:
                summary = await self._run_routed(cmd, trace_id)
            await asyncio.to_thread(
                database.complete_job_run, run_id, "completed", summary,
            )
            await asyncio.to_thread(
                database.update_job_status, job["id"], "completed",
            )
            log_event(trace_id, "INFO", "scheduler",
                      f"job {job['id']} completed")
            if job.get("session_type") == "main" and self.bot is not None:
                try:
                    await self.bot.send_alert(
                        f"✅ Scheduled '{job['name']}' completed.\n"
                        f"{(summary or '')[:500]}"
                    )
                except Exception as e:
                    log_event(trace_id, "WARNING", "scheduler",
                              f"send_alert failed: {e}")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            await asyncio.to_thread(
                database.complete_job_run, run_id, "failed",
                None, err,
            )
            await asyncio.to_thread(
                database.update_job_status, job["id"], "failed",
            )
            log_event(trace_id, "ERROR", "scheduler",
                      f"job {job['id']} failed: {err}")
            if job.get("session_type") == "main" and self.bot is not None:
                try:
                    await self.bot.send_alert(
                        f"❌ Scheduled '{job['name']}' failed: {err[:300]}"
                    )
                except Exception:
                    pass
        finally:
            self._advance_next_run(job)
            if (job.get("delete_after_run")
                    and job.get("schedule_type") == "once"):
                await asyncio.to_thread(database.delete_job, job["id"])

    async def _run_routed(self, command: str, trace_id: str) -> str:
        rr = await asyncio.to_thread(self.route, command)
        if getattr(rr, "status", None) == "error":
            raise RuntimeError(f"route error: {rr.message}")
        result = await self.wait_for_task(
            rr.task_id, config.SCHEDULER_TASK_TIMEOUT,
        )
        if not result:
            raise TimeoutError(
                f"task {rr.task_id} did not complete within "
                f"{config.SCHEDULER_TASK_TIMEOUT}s"
            )
        if result.get("status") != "completed":
            raise RuntimeError(
                f"task ended with status={result.get('status')} "
                f"error={result.get('error')!r}"
            )
        out = result.get("result")
        return str(out)[:2000] if out is not None else ""

    async def _run_internal(self, command: str, trace_id: str) -> str:
        # `/internal_foo bar baz` -> handler "foo", arg "bar baz"
        head, _, arg = command.partition(" ")
        name = head[len("/internal_"):]
        handler = INTERNAL_HANDLERS.get(name)
        if handler is None:
            raise KeyError(f"no INTERNAL_HANDLERS entry for {name!r}")
        out = await handler(arg)
        return str(out)[:2000] if out is not None else ""

    # ---------------- next-run math ----------------

    def _advance_next_run(self, job: dict) -> None:
        try:
            if job["schedule_type"] == "once":
                # one-shot: disable so it doesn't re-fire
                database.disable_job(job["id"])
                return
            nxt = compute_next_run(
                job["schedule_type"], job["schedule_value"],
                base=_now_utc(),
            )
            database.set_next_run(job["id"], nxt)
        except Exception as e:
            log_event("SEN-system", "ERROR", "scheduler",
                      f"_advance_next_run failed for job {job['id']}: {e}")
            database.disable_job(job["id"])  # poison-pill quarantine

    # ---------------- startup spreading ----------------

    async def spread_overdue_jobs(self) -> int:
        now = _now_utc()
        overdue = await asyncio.to_thread(database.get_due_jobs, now)
        if not overdue:
            return 0
        spread = (
            config.SCHEDULER_STARTUP_SPREAD_SECONDS / max(len(overdue), 1)
        )
        for i, job in enumerate(overdue):
            new_t = now + timedelta(seconds=i * spread)
            await asyncio.to_thread(database.set_next_run, job["id"], new_t)
        log_event("SEN-system", "INFO", "scheduler",
                  f"spread {len(overdue)} overdue jobs across "
                  f"{config.SCHEDULER_STARTUP_SPREAD_SECONDS}s")
        return len(overdue)
