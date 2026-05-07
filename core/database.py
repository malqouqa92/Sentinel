import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from core import config
from core.logger import log_event


class TaskRow(BaseModel):
    task_id: str
    trace_id: str
    command: str
    args: dict
    status: str
    priority: int
    retry_count: int
    max_retries: int
    recovery_count: int
    max_recoveries: int
    result: dict | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    # Phase 17a -- /kill polling flag (idempotent ALTER, defaults 0).
    kill_requested: int = 0
    # Phase 17b -- chain runner parenting + depth cap.
    parent_task_id: str | None = None
    chain_depth: int = 0


class LockRow(BaseModel):
    resource: str
    locked_by: str | None
    locked_at: str | None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    # Phase 8 perf tuning: 64MB page cache, in-memory temp tables,
    # 256MB mmap, 5s busy timeout for contended writes.
    conn.execute("PRAGMA cache_size = -64000;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA mmap_size = 268435456;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def _row_to_task(row: sqlite3.Row) -> TaskRow:
    # Phase 17a/b: new columns may not exist in OLD rows from before
    # the ALTER migrations ran (defensive: try/except on each lookup).
    def _opt(name: str, default):
        try:
            v = row[name]
            return default if v is None else v
        except (KeyError, IndexError):
            return default
    return TaskRow(
        task_id=row["task_id"],
        trace_id=row["trace_id"],
        command=row["command"],
        args=json.loads(row["args"]),
        status=row["status"],
        priority=row["priority"],
        retry_count=row["retry_count"],
        max_retries=row["max_retries"],
        recovery_count=row["recovery_count"],
        max_recoveries=row["max_recoveries"],
        result=json.loads(row["result"]) if row["result"] is not None else None,
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        kill_requested=int(_opt("kill_requested", 0)),
        parent_task_id=_opt("parent_task_id", None),
        chain_depth=int(_opt("chain_depth", 0)),
    )


def init_db() -> None:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id        TEXT PRIMARY KEY,
                trace_id       TEXT NOT NULL,
                command        TEXT NOT NULL,
                args           TEXT NOT NULL,
                status         TEXT NOT NULL,
                priority       INTEGER NOT NULL DEFAULT 0,
                retry_count    INTEGER NOT NULL DEFAULT 0,
                max_retries    INTEGER NOT NULL DEFAULT 3,
                recovery_count INTEGER NOT NULL DEFAULT 0,
                max_recoveries INTEGER NOT NULL DEFAULT 5,
                result         TEXT,
                error          TEXT,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status_priority "
            "ON tasks(status, priority, created_at)"
        )
        # Phase 17a -- /kill command + emergency abort polling.
        # `kill_requested` is checked between attempts in the agentic
        # /code pipeline; setter is `request_kill`, reader is
        # `is_kill_requested`. Idempotent ALTER (sqlite3.OperationalError
        # on duplicate column is the expected no-op).
        try:
            conn.execute(
                "ALTER TABLE tasks ADD COLUMN kill_requested "
                "INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass
        # Phase 17b -- parent_task_id for the auto-decompose chain
        # runner. NULL on standalone /code tasks; non-NULL on child
        # tasks queued by `_run_agentic_pipeline` when Claude emitted
        # a DECOMPOSE block. Lets the bot/worker trace a child back
        # to its origin and detect "all siblings done" for chain
        # completion notifications. Idempotent ALTER.
        try:
            conn.execute(
                "ALTER TABLE tasks ADD COLUMN parent_task_id TEXT"
            )
        except sqlite3.OperationalError:
            pass
        # Phase 17b -- chain_depth so children can't infinitely
        # decompose. Default 0 (top-level), incremented per child.
        # `_run_agentic_pipeline` checks against
        # config.CODE_CHAIN_MAX_DEPTH before queuing further children.
        try:
            conn.execute(
                "ALTER TABLE tasks ADD COLUMN chain_depth "
                "INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS locks (
                resource   TEXT PRIMARY KEY,
                locked_by  TEXT,
                locked_at  TEXT
            )
            """
        )
        # Phase 11 -- scheduler.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduled_jobs (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                name               TEXT NOT NULL,
                schedule_type      TEXT NOT NULL,
                schedule_value     TEXT NOT NULL,
                command            TEXT NOT NULL,
                session_type       TEXT NOT NULL DEFAULT 'main',
                active_hours_start TEXT,
                active_hours_end   TEXT,
                enabled            INTEGER NOT NULL DEFAULT 1,
                next_run_at        TEXT,
                last_run_at        TEXT,
                last_status        TEXT,
                created_at         TEXT NOT NULL,
                delete_after_run   INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_due "
            "ON scheduled_jobs(enabled, next_run_at)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id          INTEGER NOT NULL,
                started_at      TEXT NOT NULL,
                finished_at     TEXT,
                status          TEXT NOT NULL,
                result_summary  TEXT,
                error           TEXT,
                trace_id        TEXT,
                FOREIGN KEY(job_id) REFERENCES scheduled_jobs(id)
                    ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_job_status "
            "ON job_runs(job_id, status, started_at)"
        )
        # Phase 12 -- job applications tracker.
        # url_hash is sha256(url) so dedup is index-friendly even if
        # the user pastes URLs in different cases or with/without
        # tracking params (we hash the canonicalized form).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                url_hash      TEXT NOT NULL UNIQUE,
                url           TEXT NOT NULL,
                title         TEXT NOT NULL,
                company       TEXT NOT NULL,
                location      TEXT,
                archetype     TEXT,
                score         REAL,
                recommendation TEXT,
                state         TEXT NOT NULL DEFAULT 'evaluated',
                history       TEXT NOT NULL DEFAULT '[]',
                first_seen_at TEXT NOT NULL,
                last_seen_at  TEXT NOT NULL,
                applied_at    TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_apps_state "
            "ON applications(state, last_seen_at)"
        )
    finally:
        conn.close()


def add_task(
    trace_id: str,
    command: str,
    args: dict,
    priority: int = 0,
    max_retries: int = 3,
    *,
    parent_task_id: str | None = None,
    chain_depth: int = 0,
) -> str:
    """Phase 17b: ``parent_task_id`` + ``chain_depth`` are keyword-only
    additions for the auto-decompose chain runner. Standalone /code
    invocations leave both at default (None / 0). Children queued by
    `_run_agentic_pipeline` set parent_task_id to the parent's id and
    chain_depth = parent's chain_depth + 1. The depth is checked
    against ``config.CODE_CHAIN_MAX_DEPTH`` to prevent infinite
    decomposition recursion."""
    task_id = uuid.uuid4().hex
    now = _utcnow_iso()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO tasks (
                task_id, trace_id, command, args, status, priority,
                retry_count, max_retries, recovery_count, max_recoveries,
                result, error, created_at, updated_at,
                parent_task_id, chain_depth
            ) VALUES (?, ?, ?, ?, 'pending', ?, 0, ?, 0, ?, NULL, NULL,
                      ?, ?, ?, ?)
            """,
            (task_id, trace_id, command, json.dumps(args), priority,
             max_retries, config.MAX_RECOVERIES, now, now,
             parent_task_id, chain_depth),
        )
    finally:
        conn.close()
    log_event(trace_id, "INFO", "database",
              f"task added task_id={task_id} command={command}"
              + (f" parent={parent_task_id[:12]}"
                 f" depth={chain_depth}" if parent_task_id else ""))
    return task_id


def list_children(parent_task_id: str) -> list["TaskRow"]:
    """Phase 17b: list child tasks of a chain parent, ordered by
    creation time (so the bot can render 'subtask 1/N')."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE parent_task_id = ? "
            "ORDER BY created_at ASC",
            (parent_task_id,),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_task(r) for r in rows]


def chain_status_summary(parent_task_id: str) -> dict:
    """Phase 17b: aggregate for chain progress reporting.

    Returns ``{'total': int, 'completed': int, 'failed': int,
    'pending': int, 'processing': int}``.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM tasks "
            "WHERE parent_task_id = ? GROUP BY status",
            (parent_task_id,),
        ).fetchall()
    finally:
        conn.close()
    out = {
        "total": 0, "completed": 0, "failed": 0,
        "pending": 0, "processing": 0,
    }
    for r in rows:
        out["total"] += r["n"]
        key = r["status"]
        if key in out:
            out[key] = r["n"]
    return out


def requeue_task(task_id: str) -> None:
    """Set a processing task back to pending without changing retry_count.
    Used by the worker when a required resource lock is unavailable."""
    now = _utcnow_iso()
    conn = _connect()
    try:
        conn.execute(
            "UPDATE tasks SET status = 'pending', updated_at = ? "
            "WHERE task_id = ? AND status = 'processing'",
            (now, task_id),
        )
    finally:
        conn.close()


def count_pending() -> int:
    return count_tasks_by_status("pending")


def count_tasks_by_status(status: str) -> int:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM tasks WHERE status = ?",
            (status,),
        ).fetchone()
    finally:
        conn.close()
    return int(row["n"])


def get_task(task_id: str) -> TaskRow | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
    finally:
        conn.close()
    return _row_to_task(row) if row else None


def list_tasks(status: str | None = None) -> list[TaskRow]:
    conn = _connect()
    try:
        if status is None:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at",
                (status,),
            ).fetchall()
    finally:
        conn.close()
    return [_row_to_task(r) for r in rows]


def claim_next_task() -> TaskRow | None:
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM tasks
            WHERE status = 'pending'
            ORDER BY priority ASC, created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        now = _utcnow_iso()
        conn.execute(
            "UPDATE tasks SET status = 'processing', updated_at = ? "
            "WHERE task_id = ? AND status = 'pending'",
            (now, row["task_id"]),
        )
        # Re-read to confirm we won the race and get the fresh row.
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)
        ).fetchone()
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    task = _row_to_task(row) if row else None
    if task:
        log_event(task.trace_id, "INFO", "database",
                  f"task claimed task_id={task.task_id}")
    return task


def complete_task(task_id: str, result: dict) -> None:
    now = _utcnow_iso()
    conn = _connect()
    try:
        conn.execute(
            "UPDATE tasks SET status = 'completed', result = ?, "
            "error = NULL, updated_at = ? WHERE task_id = ?",
            (json.dumps(result), now, task_id),
        )
        row = conn.execute(
            "SELECT trace_id FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
    finally:
        conn.close()
    if row:
        log_event(row["trace_id"], "INFO", "database",
                  f"task completed task_id={task_id}")


def fail_task(task_id: str, error: str) -> None:
    now = _utcnow_iso()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT trace_id, retry_count, max_retries FROM tasks "
            "WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return
        new_retry = row["retry_count"] + 1
        if new_retry >= row["max_retries"]:
            new_status = "failed"
        else:
            new_status = "pending"
        conn.execute(
            "UPDATE tasks SET status = ?, retry_count = ?, error = ?, "
            "updated_at = ? WHERE task_id = ?",
            (new_status, new_retry, error, now, task_id),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    log_event(
        row["trace_id"] if row else "SEN-unknown",
        "WARNING",
        "database",
        f"task failed task_id={task_id} retry={new_retry}/"
        f"{row['max_retries']} status={new_status} error={error[:200]!r}",
    )


def acquire_lock(resource: str, task_id: str) -> bool:
    now = _utcnow_iso()
    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(seconds=config.STALE_LOCK_TIMEOUT)
    ).isoformat()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT locked_by, locked_at FROM locks WHERE resource = ?",
            (resource,),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO locks (resource, locked_by, locked_at) "
                "VALUES (?, ?, ?)",
                (resource, task_id, now),
            )
            conn.execute("COMMIT")
            return True
        if row["locked_by"] is None or (row["locked_at"] or "") < cutoff:
            conn.execute(
                "UPDATE locks SET locked_by = ?, locked_at = ? "
                "WHERE resource = ?",
                (task_id, now, resource),
            )
            conn.execute("COMMIT")
            if row["locked_by"]:
                log_event("SEN-system", "WARNING", "database",
                          f"stale lock seized resource={resource} "
                          f"prev_owner={row['locked_by']}")
            return True
        conn.execute("ROLLBACK")
        return False
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def release_lock(resource: str, task_id: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE locks SET locked_by = NULL, locked_at = NULL "
            "WHERE resource = ? AND locked_by = ?",
            (resource, task_id),
        )
    finally:
        conn.close()


def recover_stale(
    timeout_seconds: int | None = None,
    *,
    force_all_processing: bool = False,
) -> dict[str, Any]:
    """Reset orphaned `processing` tasks + release dangling locks.

    Phase 17a: ``force_all_processing=True`` (used on bot startup)
    selects EVERY `processing` task regardless of `updated_at`. Without
    this, the `updated_at < cutoff` filter would miss zombie tasks
    whose `updated_at` was refreshed by the dying worker's last claim
    -- exactly the bug that wiped uncommitted source repeatedly on
    2026-05-06 (task `4386481397f3` survived 3+ bot restarts because
    its `updated_at` was newer than the cutoff). On startup, by
    definition there is no live worker making progress on any
    `processing` task, so they're ALL stale.
    """
    if timeout_seconds is None:
        timeout_seconds = config.STALE_LOCK_TIMEOUT
    cutoff = (
        datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
    ).isoformat()
    now = _utcnow_iso()
    conn = _connect()
    recovered_tasks: list[tuple[str, str, int, int]] = []
    failed_tasks: list[tuple[str, str]] = []
    released_locks: list[tuple[str, str | None]] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        if force_all_processing:
            # Startup mode: ignore updated_at filter entirely. Every
            # `processing` task at startup is orphaned by definition.
            rows = conn.execute(
                "SELECT task_id, trace_id, recovery_count, max_recoveries "
                "FROM tasks WHERE status = 'processing'",
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT task_id, trace_id, recovery_count, max_recoveries "
                "FROM tasks WHERE status = 'processing' AND updated_at < ?",
                (cutoff,),
            ).fetchall()
        for r in rows:
            new_recovery = r["recovery_count"] + 1
            if new_recovery >= r["max_recoveries"]:
                conn.execute(
                    "UPDATE tasks SET status = 'failed', "
                    "recovery_count = ?, error = ?, updated_at = ? "
                    "WHERE task_id = ?",
                    (
                        new_recovery,
                        f"exhausted {r['max_recoveries']} recoveries "
                        f"(worker crashes)",
                        now,
                        r["task_id"],
                    ),
                )
                failed_tasks.append((r["task_id"], r["trace_id"]))
            else:
                conn.execute(
                    "UPDATE tasks SET status = 'pending', "
                    "recovery_count = ?, updated_at = ? WHERE task_id = ?",
                    (new_recovery, now, r["task_id"]),
                )
                recovered_tasks.append((
                    r["task_id"], r["trace_id"],
                    new_recovery, r["max_recoveries"],
                ))

        # Phase 12.5: orphan-lock detection. Releases locks whose owning
        # task is no longer in 'processing' state (failed/completed/
        # missing entirely) IMMEDIATELY, not after STALE_LOCK_TIMEOUT.
        # Without this, a hard-killed worker leaves the GPU lock held
        # by a task ID whose row says 'failed' -- newer workers see
        # 'gpu busy', requeue, hammer the queue.
        lock_rows = conn.execute(
            """
            SELECT l.resource, l.locked_by, l.locked_at,
                   t.status AS owner_status
              FROM locks l
              LEFT JOIN tasks t ON t.task_id = l.locked_by
             WHERE l.locked_by IS NOT NULL
               AND (
                   l.locked_at < ?
                   OR t.task_id IS NULL
                   OR t.status != 'processing'
               )
            """,
            (cutoff,),
        ).fetchall()
        for lr in lock_rows:
            conn.execute(
                "UPDATE locks SET locked_by = NULL, locked_at = NULL "
                "WHERE resource = ?",
                (lr["resource"],),
            )
            released_locks.append((lr["resource"], lr["locked_by"]))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    for task_id, trace_id, n, mx in recovered_tasks:
        log_event(trace_id, "WARNING", "database",
                  f"stale task recovered task_id={task_id} "
                  f"recovery={n}/{mx} reset to pending")
    for task_id, trace_id in failed_tasks:
        log_event(trace_id, "WARNING", "database",
                  f"stale task exhausted recoveries task_id={task_id} "
                  f"marked failed permanently")
    for resource, prev_owner in released_locks:
        log_event("SEN-system", "WARNING", "database",
                  f"stale lock released resource={resource} "
                  f"prev_owner={prev_owner}")
    return {
        "recovered": len(recovered_tasks),
        "failed": len(failed_tasks),
        "locks_released": len(released_locks),
    }


# ─────────────────────────────────────────────────────────────────────
# Phase 17a -- /kill command helpers + trace_id lookup
# ─────────────────────────────────────────────────────────────────────


def get_task_by_trace_id(trace_id: str) -> TaskRow | None:
    """Find the task associated with a trace_id. Trace IDs are
    1:1 with task IDs in practice (router generates one trace per
    incoming command). Used by `_run_agentic_pipeline` to look up
    its own task_id for kill-poll checks."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM tasks WHERE trace_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (trace_id,),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_task(row) if row else None


def request_kill(task_id: str) -> bool:
    """Set kill_requested=1 on a task. Returns True if the task was
    found AND was in a killable state (`pending` or `processing`).
    Returns False if task is missing OR already finished. Idempotent
    -- calling twice on the same task is safe (no-op the second time).
    """
    now = _utcnow_iso()
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE tasks SET kill_requested = 1, updated_at = ? "
            "WHERE task_id = ? AND status IN ('pending','processing')",
            (now, task_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def is_kill_requested(task_id: str) -> bool:
    """Check whether a task has been kill-requested. Pure read,
    no state change. Pipeline polls this between attempts."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT kill_requested FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return False
    return int(row["kill_requested"] or 0) == 1


def find_kill_target() -> dict | None:
    """Find the most-recently-claimed processing task. Used by /kill
    Telegram handler when no specific task_id is given.

    Returns ``{'task_id': str, 'command': str, 'trace_id': str,
    'updated_at': str}`` or ``None`` if nothing is killable.
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT task_id, command, trace_id, updated_at "
            "FROM tasks WHERE status = 'processing' "
            "ORDER BY updated_at DESC LIMIT 1",
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "task_id": row["task_id"],
        "command": row["command"],
        "trace_id": row["trace_id"],
        "updated_at": row["updated_at"],
    }


def _test_only_force_processing(
    task_id: str, updated_at_iso: str
) -> None:
    """Test helper: directly set a task to processing with a chosen
    updated_at. Not for production use."""
    conn = _connect()
    try:
        conn.execute(
            "UPDATE tasks SET status = 'processing', updated_at = ? "
            "WHERE task_id = ?",
            (updated_at_iso, task_id),
        )
    finally:
        conn.close()


def _test_only_force_lock(
    resource: str, task_id: str, locked_at_iso: str
) -> None:
    """Test helper: directly insert a lock with a chosen locked_at.
    Not for production use."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO locks (resource, locked_by, locked_at) "
            "VALUES (?, ?, ?)",
            (resource, task_id, locked_at_iso),
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Phase 11 -- scheduler helpers (scheduled_jobs + job_runs)
# ---------------------------------------------------------------------

def _row_to_job(row: sqlite3.Row) -> dict:
    return dict(row)


def add_job(
    name: str,
    schedule_type: str,
    schedule_value: str,
    command: str,
    next_run_at: str,
    session_type: str = "main",
    active_hours_start: str | None = None,
    active_hours_end: str | None = None,
    delete_after_run: bool = False,
) -> int:
    now = _utcnow_iso()
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO scheduled_jobs (
                name, schedule_type, schedule_value, command, session_type,
                active_hours_start, active_hours_end, enabled, next_run_at,
                last_run_at, last_status, created_at, delete_after_run
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, NULL, NULL, ?, ?)
            """,
            (name, schedule_type, schedule_value, command, session_type,
             active_hours_start, active_hours_end, next_run_at, now,
             1 if delete_after_run else 0),
        )
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_job(job_id: int) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM scheduled_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    finally:
        conn.close()
    return _row_to_job(row) if row else None


def list_jobs(enabled_only: bool = False) -> list[dict]:
    conn = _connect()
    try:
        if enabled_only:
            rows = conn.execute(
                "SELECT * FROM scheduled_jobs WHERE enabled = 1 "
                "ORDER BY next_run_at"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM scheduled_jobs ORDER BY id"
            ).fetchall()
    finally:
        conn.close()
    return [_row_to_job(r) for r in rows]


def get_due_jobs(now: datetime) -> list[dict]:
    """Return enabled jobs whose next_run_at <= now (UTC ISO)."""
    cutoff = now.isoformat()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM scheduled_jobs "
            "WHERE enabled = 1 AND next_run_at IS NOT NULL "
            "AND next_run_at <= ? "
            "ORDER BY next_run_at",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_job(r) for r in rows]


def get_overdue_jobs(now: datetime, grace_seconds: int = 0) -> list[dict]:
    """Like get_due_jobs but for startup spreading: returns jobs whose
    next_run_at is in the past (now - grace_seconds)."""
    cutoff = (now - timedelta(seconds=grace_seconds)).isoformat()
    return get_due_jobs(datetime.fromisoformat(cutoff))


def set_next_run(job_id: int, next_run_at: datetime) -> None:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE scheduled_jobs SET next_run_at = ? WHERE id = ?",
            (next_run_at.isoformat(), job_id),
        )
    finally:
        conn.close()


def set_job_enabled(job_id: int, enabled: bool) -> None:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE scheduled_jobs SET enabled = ? WHERE id = ?",
            (1 if enabled else 0, job_id),
        )
    finally:
        conn.close()


def disable_job(job_id: int) -> None:
    set_job_enabled(job_id, False)


def delete_job(job_id: int) -> None:
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM scheduled_jobs WHERE id = ?", (job_id,)
        )
    finally:
        conn.close()


def update_job_status(job_id: int, status: str) -> None:
    """Set last_status + bump last_run_at on any terminal scheduler outcome."""
    now = _utcnow_iso()
    conn = _connect()
    try:
        conn.execute(
            "UPDATE scheduled_jobs SET last_status = ?, last_run_at = ? "
            "WHERE id = ?",
            (status, now, job_id),
        )
    finally:
        conn.close()


def has_running_run(job_id: int) -> bool:
    """True iff there is a job_runs row for this job with status='running'."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM job_runs WHERE job_id = ? AND status = 'running' "
            "LIMIT 1",
            (job_id,),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def start_job_run(job_id: int, trace_id: str) -> int:
    now = _utcnow_iso()
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO job_runs (job_id, started_at, status, trace_id)
            VALUES (?, ?, 'running', ?)
            """,
            (job_id, now, trace_id),
        )
        return int(cur.lastrowid)
    finally:
        conn.close()


def complete_job_run(
    run_id: int, status: str,
    result_summary: str | None = None,
    error: str | None = None,
) -> None:
    now = _utcnow_iso()
    summary = (result_summary or "")[:2000]
    conn = _connect()
    try:
        conn.execute(
            """
            UPDATE job_runs SET status = ?, finished_at = ?,
                                result_summary = ?, error = ?
            WHERE id = ?
            """,
            (status, now, summary, error, run_id),
        )
    finally:
        conn.close()


def record_skip(job_id: int, reason: str) -> int:
    """Append a job_runs row marked 'skipped' with the reason in error."""
    now = _utcnow_iso()
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO job_runs (job_id, started_at, finished_at, status, error)
            VALUES (?, ?, ?, 'skipped', ?)
            """,
            (job_id, now, now, reason),
        )
        return int(cur.lastrowid)
    finally:
        conn.close()


def last_runs(job_id: int, limit: int = 10) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM job_runs WHERE job_id = ? "
            "ORDER BY started_at DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_lock(resource: str) -> dict | None:
    """Phase 11: read-only inspection of a resource lock for /health.
    Returns {locked_by, locked_at} or None when nothing's holding it."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT locked_by, locked_at FROM locks WHERE resource = ?",
            (resource,),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["locked_by"] is None:
        return None
    return {"locked_by": row["locked_by"], "locked_at": row["locked_at"]}


# ---------------------------------------------------------------------
# Phase 12 -- applications tracker
# ---------------------------------------------------------------------

# Canonical states (mirrors career-ops/templates/states.yml). Aliases
# accepted on the way in via _normalize_state.
_APPLICATION_STATES = (
    "evaluated", "applied", "responded", "interview",
    "offer", "rejected", "discarded",
)
_STATE_ALIASES = {
    "evaluada": "evaluated", "evaluacion": "evaluated",
    "aplicado": "applied", "enviada": "applied", "aplicada": "applied", "sent": "applied",
    "respondido": "responded",
    "entrevista": "interview",
    "oferta": "offer",
    "rechazado": "rejected", "rechazada": "rejected",
    "descartado": "discarded", "descartada": "discarded",
    "cerrada": "discarded", "cancelada": "discarded",
}


def _normalize_state(s: str) -> str:
    s = (s or "").strip().lower()
    s = _STATE_ALIASES.get(s, s)
    if s not in _APPLICATION_STATES:
        raise ValueError(
            f"unknown application state {s!r}; "
            f"must be one of {_APPLICATION_STATES}"
        )
    return s


def _hash_url(url: str) -> str:
    """sha256 of the canonicalized URL. Lowercased + stripped + tracking
    params dropped (utm_*, gclid, fbclid)."""
    import hashlib
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
    raw = (url or "").strip().lower()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
        kept = [
            (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not (k.startswith("utm_") or k in ("gclid", "fbclid"))
        ]
        canonical = urlunsplit((
            parts.scheme, parts.netloc, parts.path,
            urlencode(kept), parts.fragment,
        ))
    except Exception:
        canonical = raw
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def application_exists(url: str) -> bool:
    h = _hash_url(url)
    if not h:
        return False
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM applications WHERE url_hash = ? LIMIT 1",
            (h,),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def upsert_application(
    url: str, title: str, company: str,
    location: str | None = None,
    archetype: str | None = None,
    score: float | None = None,
    recommendation: str | None = None,
    state: str = "evaluated",
) -> int:
    """Insert a new application or refresh last_seen_at + score on an
    existing row. Never overwrites a non-evaluated state (so an in-
    flight 'interview' record isn't reset to 'evaluated' by a re-scrape).
    Returns the row id."""
    h = _hash_url(url)
    if not h:
        raise ValueError("upsert_application requires a non-empty url")
    state = _normalize_state(state)
    now = _utcnow_iso()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id, state, history FROM applications WHERE url_hash = ?",
            (h,),
        ).fetchone()
        if row is None:
            history = json.dumps([{"ts": now, "from": None, "to": state}])
            cur = conn.execute(
                """
                INSERT INTO applications
                  (url_hash, url, title, company, location, archetype,
                   score, recommendation, state, history,
                   first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (h, url, title, company, location, archetype, score,
                 recommendation, state, history, now, now),
            )
            new_id = int(cur.lastrowid)
        else:
            # Refresh last_seen + score, but keep the existing state if
            # it has progressed past 'evaluated'.
            keep_state = row["state"] if row["state"] != "evaluated" else state
            conn.execute(
                """
                UPDATE applications SET
                    title = ?, company = ?, location = ?, archetype = ?,
                    score = ?, recommendation = ?, state = ?,
                    last_seen_at = ?
                WHERE id = ?
                """,
                (title, company, location, archetype, score,
                 recommendation, keep_state, now, row["id"]),
            )
            new_id = int(row["id"])
        conn.execute("COMMIT")
        return new_id
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def transition_application(
    app_id: int, to_state: str, note: str | None = None,
) -> dict | None:
    """Move an application to a new state, append the transition to
    history, return the updated row dict (or None if not found)."""
    to_state = _normalize_state(to_state)
    now = _utcnow_iso()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM applications WHERE id = ?", (app_id,),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        history = json.loads(row["history"] or "[]")
        history.append({
            "ts": now, "from": row["state"], "to": to_state,
            "note": note,
        })
        applied_at = row["applied_at"]
        if to_state == "applied" and not applied_at:
            applied_at = now
        conn.execute(
            """
            UPDATE applications
               SET state = ?, history = ?, last_seen_at = ?, applied_at = ?
             WHERE id = ?
            """,
            (to_state, json.dumps(history), now, applied_at, app_id),
        )
        updated = conn.execute(
            "SELECT * FROM applications WHERE id = ?", (app_id,),
        ).fetchone()
        conn.execute("COMMIT")
        return dict(updated) if updated else None
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def find_recent_company_postings(
    company_normalized: str, days: int = 90,
) -> list[dict]:
    """Phase 13 Batch 6: rows from `applications` whose company matches
    (case-insensitive, suffix-stripped) AND last_seen_at is within the
    given window. Used by core.legitimacy.detect_repost_cadence to
    flag ghost-job repost cadence.

    The match is intentionally LOOSE (LIKE substring on the lowercased
    company column with common suffixes stripped) so 'AcmeCo' matches
    'AcmeCo Inc', 'ACMECO LLC', 'AcmeCo, Corp.', etc. False positives
    are acceptable here -- the consumer fuzzy-matches titles too and
    only acts at >=3 matches.
    """
    co = (company_normalized or "").strip().lower()
    if not co:
        return []
    cutoff = (
        datetime.now(timezone.utc).replace(microsecond=0)
        - timedelta(days=max(1, int(days)))
    ).isoformat()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, url, title, company, last_seen_at
              FROM applications
             WHERE LOWER(company) LIKE ?
               AND last_seen_at >= ?
             ORDER BY last_seen_at DESC
             LIMIT 50
            """,
            (f"%{co}%", cutoff),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_application(app_id: int) -> dict | None:
    """Phase 13: fetch a single application row by id. Used by the
    /jobs Telegram viewer for the drill-in detail view. Returns the
    row dict or None if not found."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM applications WHERE id = ?", (app_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def list_applications(state: str | None = None,
                      limit: int = 100) -> list[dict]:
    conn = _connect()
    try:
        if state is not None:
            rows = conn.execute(
                "SELECT * FROM applications WHERE state = ? "
                "ORDER BY last_seen_at DESC LIMIT ?",
                (_normalize_state(state), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM applications "
                "ORDER BY last_seen_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def count_runs_today(status: str) -> int:
    """Phase 11: count job_runs rows started today (UTC) with given status."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    ).isoformat()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM job_runs "
            "WHERE status = ? AND started_at >= ?",
            (status, today_start),
        ).fetchone()
    finally:
        conn.close()
    return int(row["n"])
