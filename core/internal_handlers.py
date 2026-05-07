"""Phase 11 -- /internal_* maintenance handlers.

These are registered into core.scheduler.INTERNAL_HANDLERS so the
scheduler can dispatch them directly (no router round-trip) for pure
SQLite/disk work that doesn't need agent or skill machinery.

Registered handlers (all async; receive a single string `arg`):
  wal_checkpoint     -- PRAGMA wal_checkpoint(TRUNCATE) on all 3 DBs
  backup             -- copy SQLite + persona files to BACKUP_DIR/YYYY-MM-DD
                        + prune older than BACKUP_KEEP_DAYS
  resource_check     -- disk-free + log-size probes; alert on threshold
  file_guard_check   -- replaces the bot's _heartbeat_task
  curate             -- replaces the bot's _curation_scheduler_task

Importing this module registers the handlers as a side-effect so
main.py (or scheduled jobs in tests) only needs `from core import
internal_handlers  # noqa`.
"""
from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core import config
from core.logger import log_event
from core.scheduler import INTERNAL_HANDLERS

# Optional alert callback; main.py wires this to the bot's send_alert
# after the bot is constructed. Tests can replace it directly.
ALERT_CALLBACK = None  # async (message: str) -> None


def _set_alert_callback(cb) -> None:
    global ALERT_CALLBACK
    ALERT_CALLBACK = cb


async def _alert(msg: str) -> None:
    if ALERT_CALLBACK is None:
        log_event("SEN-system", "WARNING", "internal",
                  f"alert (no callback wired): {msg}")
        return
    try:
        await ALERT_CALLBACK(msg)
    except Exception as e:
        log_event("SEN-system", "WARNING", "internal",
                  f"alert callback failed: {e}")


# ---------------------------------------------------------------------
# wal_checkpoint
# ---------------------------------------------------------------------

async def wal_checkpoint(arg: str) -> str:
    paths = [
        config.DB_PATH,
        config.MEMORY_DB_PATH,
        config.KNOWLEDGE_DB_PATH,
    ]
    results: list[str] = []
    for p in paths:
        try:
            with sqlite3.connect(str(p), timeout=30.0) as c:
                c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            results.append(f"{p.name}=ok")
        except Exception as e:
            results.append(f"{p.name}=fail({type(e).__name__})")
            log_event("SEN-system", "WARNING", "internal",
                      f"wal_checkpoint failed for {p}: {e}")
    return "; ".join(results)


INTERNAL_HANDLERS["wal_checkpoint"] = wal_checkpoint


# ---------------------------------------------------------------------
# backup -- copy SQLite (online) + persona files; prune retention
# ---------------------------------------------------------------------

def _sqlite_backup(src: Path, dst: Path) -> None:
    """Online SQLite backup -- safe while the source is being written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(src), timeout=30.0)
    try:
        dst_conn = sqlite3.connect(str(dst))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def _prune_old_backups(root: Path, keep_days: int) -> int:
    """Delete YYYY-MM-DD subdirectories older than keep_days. Returns
    the number deleted."""
    if not root.exists():
        return 0
    cutoff = (datetime.now(timezone.utc).date()
              - timedelta(days=keep_days))
    deleted = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            d = datetime.strptime(child.name, "%Y-%m-%d").date()
        except ValueError:
            continue  # skip directories that don't match the date pattern
        if d < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            deleted += 1
    return deleted


async def backup(arg: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dst_root = config.BACKUP_DIR / today
    dst_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    failed: list[str] = []
    # SQLite databases via online backup API.
    for src in [
        config.DB_PATH, config.MEMORY_DB_PATH, config.KNOWLEDGE_DB_PATH,
    ]:
        if not src.exists():
            continue
        try:
            _sqlite_backup(src, dst_root / src.name)
            copied.append(src.name)
        except Exception as e:
            failed.append(f"{src.name}({type(e).__name__})")
            log_event("SEN-system", "WARNING", "internal",
                      f"backup failed for {src}: {e}")
    # Persona files -- plain copy (small, human-edited).
    persona_dst = dst_root / "persona"
    persona_dst.mkdir(parents=True, exist_ok=True)
    if config.PERSONA_DIR.exists():
        for name in config.PROTECTED_FILES:
            src = config.PERSONA_DIR / name
            if src.exists():
                try:
                    shutil.copy2(src, persona_dst / name)
                    copied.append(f"persona/{name}")
                except Exception as e:
                    failed.append(f"persona/{name}({type(e).__name__})")
    pruned = _prune_old_backups(
        config.BACKUP_DIR, config.BACKUP_KEEP_DAYS,
    )
    summary = (
        f"backup -> {dst_root.name}: "
        f"copied={len(copied)} failed={len(failed)} pruned={pruned}"
    )
    log_event("SEN-system", "INFO", "internal", summary)
    return summary


INTERNAL_HANDLERS["backup"] = backup


# ---------------------------------------------------------------------
# resource_check -- disk + log-dir size + VRAM (via health.py probe)
# ---------------------------------------------------------------------

async def resource_check(arg: str) -> str:
    findings: list[str] = []
    # Disk free
    try:
        d = shutil.disk_usage(str(config.PROJECT_ROOT))
        free = int(d.free)
        if free < config.DISK_FREE_ALERT_BYTES:
            await _alert(
                f"⚠️ LOW DISK SPACE: {free // 1_000_000} MB free at "
                f"{config.PROJECT_ROOT}"
            )
            findings.append(f"disk_low={free}")
        else:
            findings.append(f"disk_ok={free}")
    except Exception as e:
        findings.append(f"disk_err={type(e).__name__}")
    # Log dir size
    try:
        total = 0
        if config.LOG_DIR.exists():
            for f in config.LOG_DIR.iterdir():
                if f.is_file():
                    total += f.stat().st_size
        if total > config.LOG_DIR_ALERT_BYTES:
            await _alert(
                f"⚠️ LARGE LOG DIRECTORY: {total // 1_000_000} MB"
            )
            findings.append(f"logs_large={total}")
        else:
            findings.append(f"logs_ok={total}")
    except Exception as e:
        findings.append(f"logs_err={type(e).__name__}")
    # VRAM (best-effort; skips silently if nvidia-smi missing)
    try:
        from core.health import _vram_used_mb
        vram = _vram_used_mb()
        if vram is not None:
            findings.append(f"vram={vram}")
    except Exception as e:
        findings.append(f"vram_err={type(e).__name__}")
    return ", ".join(findings)


INTERNAL_HANDLERS["resource_check"] = resource_check


# ---------------------------------------------------------------------
# file_guard_check -- replaces the bot's _heartbeat_task
# ---------------------------------------------------------------------

async def file_guard_check(arg: str) -> str:
    try:
        from core.file_guard import get_file_guard
        fg = get_file_guard()
        if fg is None:
            return "file_guard not installed"
        result = fg.check_integrity()
        return f"checked: {result}"
    except Exception as e:
        log_event("SEN-system", "WARNING", "internal",
                  f"file_guard_check failed: {type(e).__name__}: {e}")
        return f"error: {type(e).__name__}: {e}"


INTERNAL_HANDLERS["file_guard_check"] = file_guard_check


# ---------------------------------------------------------------------
# curate -- replaces the bot's _curation_scheduler_task
# ---------------------------------------------------------------------

async def curate(arg: str) -> str:
    """Auto-curation: propose facts from the last 24h and apply
    immediately (auto-approve) when run from the scheduler. The
    interactive /curate Telegram command stays in handle_curate
    for manual reviews."""
    try:
        from core.curation import get_curation_flow
        flow = get_curation_flow()
        if flow is None:
            return "curation flow not installed"
        proposal = await flow.propose_facts(
            lookback_hours=config.CURATION_LOOKBACK_HOURS,
        )
        if not proposal or not getattr(proposal, "items", None):
            return "no curation proposals generated"
        applied = await flow.apply_proposal(proposal, auto_approved=True)
        return (
            f"curate: proposed={len(proposal.items)} "
            f"applied={applied}"
        )
    except Exception as e:
        log_event("SEN-system", "WARNING", "internal",
                  f"curate failed: {type(e).__name__}: {e}")
        return f"error: {type(e).__name__}: {e}"


INTERNAL_HANDLERS["curate"] = curate


# ---------------------------------------------------------------------
# kb_lifecycle -- Phase 15a nightly age-based transition walker
# ---------------------------------------------------------------------

async def kb_lifecycle(arg: str) -> str:
    """Walk active->stale and stale->archived on the configured age
    windows. Pure SQLite, no GPU, no LLM. Pinned rows are immune.
    Cheap enough to run nightly even at the new 50K cap."""
    try:
        from core.knowledge_base import KnowledgeBase
        # Run on the same DB the bot writes to. Lazy KB instance --
        # this handler may fire before the bot's KB is constructed
        # (e.g. test harness), so spin one up locally.
        kb = KnowledgeBase()
        kb_result = await _run_blocking(kb.auto_transition_lifecycle)
    except Exception as e:
        log_event("SEN-system", "WARNING", "internal",
                  f"kb_lifecycle (kb) failed: {type(e).__name__}: {e}")
        kb_result = {"error": f"{type(e).__name__}: {e}"}
    try:
        from core.memory import get_memory
        mem = get_memory()
        mem_result = await _run_blocking(mem.auto_transition_lifecycle)
    except Exception as e:
        log_event("SEN-system", "WARNING", "internal",
                  f"kb_lifecycle (memory) failed: {type(e).__name__}: {e}")
        mem_result = {"error": f"{type(e).__name__}: {e}"}
    summary = f"kb={kb_result} memory={mem_result}"
    log_event("SEN-system", "INFO", "internal", f"kb_lifecycle: {summary}")
    return summary


async def _run_blocking(fn, *args, **kwargs):
    """Run a sync DB function on a worker thread so the scheduler
    coroutine doesn't block the event loop on the SQLite walk."""
    import asyncio as _aio
    return await _aio.to_thread(fn, *args, **kwargs)


INTERNAL_HANDLERS["kb_lifecycle"] = kb_lifecycle
