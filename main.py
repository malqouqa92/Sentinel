"""Sentinel OS -- main entry point.

Starts the async worker loop, brain router, and Telegram bot in one
event loop. Cleans up gracefully on SIGINT / SIGTERM.

Required environment:
  SENTINEL_TELEGRAM_TOKEN     -- bot token from @BotFather
  SENTINEL_TELEGRAM_USER_ID   -- your Telegram user_id (or comma-list)
"""
from __future__ import annotations

import asyncio
import signal
import sqlite3
import sys
from datetime import datetime, timezone

from core import config, database
from core.agent_registry import AGENT_REGISTRY
from core.brain import BrainRouter
from core.claude_cli import ClaudeCLI
from core import internal_handlers as _internal_handlers  # registers /internal_*
from core.health import HealthMonitor, start_health_server
from core.knowledge_base import KnowledgeBase
from core.llm import INFERENCE_CLIENT, LLMError, OllamaClient
from core.logger import log_event
from core.model_registry import MODEL_REGISTRY
from core.registry import SKILL_REGISTRY
from core.router import route as _route
from core.scheduler import Scheduler
from core.telemetry import generate_trace_id
from core.worker import (
    get_or_create_shutdown_event, install_signal_handlers, worker_loop,
)
from interfaces.telegram_bot import SentinelTelegramBot


async def _prewarm_qwen_coder(trace_id: str) -> None:
    """Phase 16 (2026-05-06): pre-load qwen2.5-coder:3b into VRAM at
    startup so the first /code doesn't pay cold-load cost (~10-20s for
    weights + ~30-60s extra on stepfed's first step).

    User-observed symptom (SEN-2dea7bea): a /code submitted 17s after
    bot startup ran for 3 min before manual kill, vs the documented
    average of 60-120s. Cold-load on stepfed's first step compounded
    the slowness. Prewarm sends a 1-token generate so Ollama loads
    the weights into VRAM with the configured keep_alive timer.

    Best-effort: any failure (Ollama down, timeout, network) is
    logged at INFO and never raises. Trade-off: this puts qwen-coder
    in VRAM ahead of qwen3-brain. If the user's first action is free-
    text chat instead of /code, the brain will cold-load instead --
    acceptable since /code is the heavy path."""
    try:
        client = OllamaClient()
        await asyncio.to_thread(
            client.generate,
            "qwen2.5-coder:3b",
            "ok",
            None,    # system
            0.0,     # temperature
            30,      # timeout
            False,   # format_json
            trace_id,
        )
        log_event(
            trace_id, "INFO", "main",
            "qwen-coder prewarm ok (loaded into VRAM with keep_alive)",
        )
    except LLMError as e:
        log_event(
            trace_id, "INFO", "main",
            f"qwen-coder prewarm skipped (Ollama issue): {e}",
        )
    except Exception as e:
        log_event(
            trace_id, "INFO", "main",
            f"qwen-coder prewarm exception: {type(e).__name__}: {e}",
        )


_DEFAULT_SCHEDULED_JOBS: list[dict] = [
    # name, schedule_type, schedule_value, command
    {"name": "WAL checkpoint",
     "schedule_type": "interval", "schedule_value": "1h",
     "command": "/internal_wal_checkpoint",
     "session_type": "isolated"},
    {"name": "Resource probe",
     "schedule_type": "interval", "schedule_value": "10m",
     "command": "/internal_resource_check",
     "session_type": "isolated"},
    {"name": "Nightly backup",
     "schedule_type": "cron", "schedule_value": "30 3 * * *",
     "command": "/internal_backup",
     "session_type": "isolated"},
    # Phase 15a -- nightly KB + memory lifecycle walker. Runs at
    # 03:45 EST so it lands AFTER the 03:30 backup snapshots a
    # consistent state. Pure SQLite (no GPU, no LLM); cheap even at
    # the new 50K cap. Pinned rows are skipped by definition.
    {"name": "KB lifecycle transition",
     "schedule_type": "cron", "schedule_value": "45 3 * * *",
     "command": "/internal_kb_lifecycle",
     "session_type": "isolated"},
]


def _seed_default_scheduled_jobs(trace_id: str) -> None:
    """Idempotently seed default maintenance jobs. Skips any whose
    `name` already exists. The bot's in-process _heartbeat_task and
    _curation_scheduler_task remain authoritative for file_guard +
    curation; the scheduler defaults here cover the new Phase 11
    /internal_* maintenance commands only."""
    from core.scheduler import compute_next_run
    existing = {j["name"] for j in database.list_jobs()}
    seeded = []
    for spec in _DEFAULT_SCHEDULED_JOBS:
        if spec["name"] in existing:
            continue
        try:
            nxt = compute_next_run(
                spec["schedule_type"], spec["schedule_value"],
            )
            database.add_job(
                name=spec["name"],
                schedule_type=spec["schedule_type"],
                schedule_value=spec["schedule_value"],
                command=spec["command"],
                next_run_at=nxt.isoformat(),
                session_type=spec.get("session_type", "main"),
            )
            seeded.append(spec["name"])
        except Exception as e:
            log_event(trace_id, "WARNING", "main",
                      f"failed to seed default job {spec['name']!r}: {e}")
    if seeded:
        log_event(trace_id, "INFO", "main",
                  f"seeded {len(seeded)} default scheduled jobs: "
                  f"{', '.join(seeded)}")


def _print_startup_banner() -> None:
    """Short intro + quick-reference printed to stdout on every boot.
    Appears BEFORE any error so the user always has context."""
    bar = "+" + "=" * 60 + "+"
    print(bar)
    print("|" + " " * 60 + "|")
    print("|   SENTINEL  -  local agent framework"
          + " " * 23 + "|")
    print("|   Backends: Ollama (Qwen 3 1.7B + Coder 3B) + Claude CLI"
          + " " * 3 + "|")
    print("|   Interface: Telegram bot"
          + " " * 34 + "|")
    print("|" + " " * 60 + "|")
    print(bar)
    print()
    print("Quick reference (Telegram):")
    print("  /help              list all commands")
    print("  /dashboard         live system health snapshot")
    print("  /gwenask <idea>    have local Qwen author a recipe")
    print("  /gwen <recipe>     execute a literal recipe")
    print("  /code <problem>    Qwen with Claude-CLI ceiling")
    print("  /commit            commit working-tree changes")
    print()
    print("Docs:  CLAUDE.md (architecture)  |  PHASES.md (change log)")
    print("Logs:  logs/sentinel.jsonl       |  Health: 127.0.0.1:18700/health")
    print()


async def run() -> int:
    _print_startup_banner()
    trace_id = generate_trace_id()
    log_event(trace_id, "INFO", "main", "Sentinel OS starting...")

    if not config.TELEGRAM_TOKEN:
        msg = "SENTINEL_TELEGRAM_TOKEN not set"
        log_event(trace_id, "ERROR", "main", msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        print("\nFix: run setup.ps1, or set the env var manually:",
              file=sys.stderr)
        print("  setx SENTINEL_TELEGRAM_TOKEN \"<your-bot-token>\"",
              file=sys.stderr)
        return 2
    if not config.TELEGRAM_AUTHORIZED_USERS:
        msg = "SENTINEL_TELEGRAM_USER_ID not set (no authorized users)"
        log_event(trace_id, "ERROR", "main", msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        print("\nFix: run setup.ps1, or set the env var manually:",
              file=sys.stderr)
        print("  setx SENTINEL_TELEGRAM_USER_ID \"<your-numeric-user-id>\"",
              file=sys.stderr)
        return 2

    database.init_db()
    summary = await asyncio.to_thread(database.recover_stale)
    if summary["recovered"] or summary["failed"] or summary["locks_released"]:
        log_event(trace_id, "WARNING", "main",
                  f"crash recovery on startup: {summary}")

    SKILL_REGISTRY.reset()
    SKILL_REGISTRY.discover()
    AGENT_REGISTRY.reset()
    AGENT_REGISTRY.discover()

    avail = await asyncio.to_thread(MODEL_REGISTRY.check_availability)
    for name, ok in avail.items():
        log_event(trace_id, "INFO", "main",
                  f"model {name}: {'available' if ok else 'NOT AVAILABLE'}")

    # Phase 9 hygiene: clean up any pattern entries that fail the
    # quality gate (e.g. solution='200' poison from prior runs).
    kb_for_cleanup = KnowledgeBase()
    cleaned = await asyncio.to_thread(
        kb_for_cleanup.cleanup_low_quality_patterns,
    )
    if cleaned:
        log_event(trace_id, "WARNING", "main",
                  f"removed {cleaned} low-quality KB patterns on startup")

    brain = BrainRouter()
    claude_cli = ClaudeCLI()
    bot = SentinelTelegramBot(
        token=config.TELEGRAM_TOKEN,
        brain=brain,
        claude_cli=claude_cli,
        inference_client=INFERENCE_CLIENT,
        knowledge_base=KnowledgeBase(),
    )

    shutdown_event = get_or_create_shutdown_event()
    loop = asyncio.get_running_loop()
    install_signal_handlers(loop, shutdown_event)

    # Phase 11: scheduler + health endpoint. Constructed BEFORE bot.start()
    # so we can wire the scheduler back to the bot for completion alerts.
    health = HealthMonitor(start_time=datetime.now(timezone.utc))
    health.set_model_availability(avail)

    async def _scheduler_wait(task_id: str, timeout: int) -> dict | None:
        return await bot._wait_for_task(task_id, timeout)

    scheduler = Scheduler(
        router_fn=_route,
        wait_for_task_fn=_scheduler_wait,
        shutdown_event=shutdown_event,
        bot=bot,
    )
    health.scheduler = scheduler
    health.bot = bot
    bot.health_monitor = health

    # Wire alert callback so /internal_resource_check can ping Telegram.
    _internal_handlers._set_alert_callback(bot.send_alert)

    worker_task = asyncio.create_task(worker_loop(shutdown_event))
    await bot.start()

    # Health server + scheduler loop, started after the bot is up.
    health_runner = await start_health_server(health)
    await asyncio.to_thread(_seed_default_scheduled_jobs, trace_id)
    spread_count = await scheduler.spread_overdue_jobs()
    if spread_count:
        log_event(trace_id, "INFO", "main",
                  f"spread {spread_count} overdue scheduled jobs on startup")
    scheduler_task = asyncio.create_task(scheduler.scheduler_loop())

    log_event(trace_id, "INFO", "main",
              "Sentinel OS ready. Listening on Telegram.")
    print("Sentinel OS is running. Send a message to your Telegram bot. "
          "Ctrl+C to stop.")

    # Phase 16 (2026-05-06): prewarm qwen-coder so the first /code
    # doesn't pay cold-load cost. Background task -- doesn't block
    # bot from accepting messages. Best-effort, never raises.
    prewarm_task = asyncio.create_task(_prewarm_qwen_coder(trace_id))

    try:
        await shutdown_event.wait()
    finally:
        log_event(trace_id, "INFO", "main", "Shutting down...")
        # Phase 16: cancel prewarm if still in flight (typically done
        # within 30s; this catches the case where shutdown fires fast).
        prewarm_task.cancel()
        try:
            await prewarm_task
        except (asyncio.CancelledError, Exception):
            pass
        scheduler_task.cancel()
        try:
            await scheduler_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await health_runner.cleanup()
        except Exception as e:
            log_event(trace_id, "WARNING", "main",
                      f"health server cleanup raised: {e}")
        try:
            await bot.stop()
        except Exception as e:
            log_event(trace_id, "WARNING", "main",
                      f"bot.stop raised: {e}")
        worker_task.cancel()
        try:
            await worker_task
        except (asyncio.CancelledError, Exception):
            pass
        # Phase 11: SQLite hygiene on the way out.
        for db_path in [
            config.DB_PATH, config.MEMORY_DB_PATH, config.KNOWLEDGE_DB_PATH,
        ]:
            try:
                with sqlite3.connect(str(db_path), timeout=10.0) as c:
                    c.execute("PRAGMA optimize")
            except Exception as e:
                log_event(trace_id, "WARNING", "main",
                          f"PRAGMA optimize failed for {db_path}: {e}")
        log_event(trace_id, "INFO", "main", "Sentinel OS shut down cleanly.")
        print("Sentinel OS stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
