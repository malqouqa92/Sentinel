"""Phase 17a -- /kill command + zombie-task recover_stale fix.

Three groups:
  R -- recover_stale force_all_processing (zombie-task fix)
  K -- request_kill / is_kill_requested / find_kill_target / get_task_by_trace_id
  W -- wiring source-checks (worker startup, telegram handler, pipeline poll)

Live trigger this fix addresses (2026-05-06 evening session):
  Task 4386481397f3 created at 22:09 when user killed bot mid-/qcode.
  Task stayed `processing` because recover_stale's `updated_at < cutoff`
  filter missed it (the dying worker's last claim refreshed updated_at).
  Every bot restart resumed the zombie task -> _git_reset_hard scope blast
  -> wiped uncommitted source files. Three confirmed wipes in one session.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Sentinel uses config.DB_PATH module-globally. Patch to a tmp DB."""
    from core import config, database
    db_path = tmp_path / "sentinel_test.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    database.init_db()
    yield db_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _back(seconds: int) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(seconds=seconds)
    ).isoformat()


# ============================================================
# Group R: recover_stale force_all_processing
# ============================================================


def test_r01_force_all_processing_marks_recently_updated_zombie(temp_db):
    """The headline bug: a `processing` task with FRESH updated_at
    (within cutoff) must still be recovered when force_all_processing=True.
    Default behavior (cutoff filter) would miss this — that's the bug."""
    from core import database
    tid = database.add_task("SEN-test", "code", {"x": 1})
    # Force to processing with NOW updated_at (mimicking the zombie's
    # state: dying worker refreshed updated_at on its last claim).
    database._test_only_force_processing(tid, _now())

    # Default mode: should NOT touch the task (updated_at is fresh).
    s1 = database.recover_stale(timeout_seconds=300)
    assert s1["recovered"] == 0
    row = database.get_task(tid)
    assert row.status == "processing"

    # Force mode: MUST recover.
    s2 = database.recover_stale(force_all_processing=True)
    assert s2["recovered"] == 1
    row2 = database.get_task(tid)
    assert row2.status == "pending"


def test_r02_force_all_processing_marks_old_zombies_too(temp_db):
    """Task that's been processing for an hour also gets recovered
    -- proves force mode is a superset, not a different filter."""
    from core import database
    tid = database.add_task("SEN-test2", "code", {"x": 2})
    database._test_only_force_processing(tid, _back(3600))
    s = database.recover_stale(force_all_processing=True)
    assert s["recovered"] == 1


def test_r03_force_mode_does_not_touch_pending_or_completed(temp_db):
    """Only `processing` rows are affected by force mode."""
    from core import database
    pid = database.add_task("SEN-pending", "code", {"a": 1})
    cid = database.add_task("SEN-completed", "code", {"a": 2})
    database.complete_task(cid, {"ok": True})
    s = database.recover_stale(force_all_processing=True)
    assert s["recovered"] == 0
    assert database.get_task(pid).status == "pending"
    assert database.get_task(cid).status == "completed"


def test_r04_default_mode_unchanged(temp_db):
    """Without force_all_processing, behavior is identical to pre-17a.
    Old tasks (updated_at < cutoff) recovered; fresh ones ignored."""
    from core import database
    fresh = database.add_task("SEN-fresh", "code", {})
    stale = database.add_task("SEN-stale", "code", {})
    database._test_only_force_processing(fresh, _now())
    database._test_only_force_processing(stale, _back(7200))
    s = database.recover_stale(timeout_seconds=300)
    assert s["recovered"] == 1
    assert database.get_task(fresh).status == "processing"  # untouched
    assert database.get_task(stale).status == "pending"


def test_r05_force_mode_releases_orphan_locks_too(temp_db):
    """Orphan-lock release path still runs in force mode (locks
    associated with non-processing tasks get cleared)."""
    from core import database
    tid = database.add_task("SEN-locked", "code", {})
    database._test_only_force_lock("gpu", tid, _now())
    # Task is `pending` but lock claims it -- orphan condition.
    s = database.recover_stale(force_all_processing=True)
    assert s["locks_released"] == 1


def test_r06_force_mode_kwarg_only_not_positional(temp_db):
    """API safety: force_all_processing must be keyword-only so
    legacy callers passing positional timeout_seconds don't acci-
    dentally enable force mode."""
    from core import database
    # This must work (positional timeout_seconds, default force=False):
    database.recover_stale(60)
    # This must work (explicit kwarg):
    database.recover_stale(force_all_processing=True)
    # The signature should reject positional force:
    import inspect
    sig = inspect.signature(database.recover_stale)
    p = sig.parameters["force_all_processing"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY


# ============================================================
# Group K: kill helpers
# ============================================================


def test_k01_kill_requested_column_exists(temp_db):
    """init_db migration must add the kill_requested column."""
    from core import database
    conn = sqlite3.connect(database.config.DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)")]
    conn.close()
    assert "kill_requested" in cols


def test_k02_request_kill_sets_flag(temp_db):
    from core import database
    tid = database.add_task("SEN-kill1", "code", {})
    database._test_only_force_processing(tid, _now())
    assert database.is_kill_requested(tid) is False
    ok = database.request_kill(tid)
    assert ok is True
    assert database.is_kill_requested(tid) is True


def test_k03_request_kill_idempotent(temp_db):
    from core import database
    tid = database.add_task("SEN-kill2", "code", {})
    database._test_only_force_processing(tid, _now())
    assert database.request_kill(tid) is True
    # Second call: still returns True (still in killable state),
    # but flag is already set. No double-event.
    assert database.request_kill(tid) is True
    assert database.is_kill_requested(tid) is True


def test_k04_request_kill_returns_false_on_completed(temp_db):
    from core import database
    tid = database.add_task("SEN-kill3", "code", {})
    database.complete_task(tid, {"ok": True})
    assert database.request_kill(tid) is False
    assert database.is_kill_requested(tid) is False


def test_k05_request_kill_returns_false_on_missing(temp_db):
    from core import database
    assert database.request_kill("nonexistent_id") is False


def test_k06_is_kill_requested_returns_false_on_missing(temp_db):
    from core import database
    assert database.is_kill_requested("nonexistent_id") is False


def test_k07_find_kill_target_returns_none_when_idle(temp_db):
    from core import database
    assert database.find_kill_target() is None


def test_k08_find_kill_target_returns_most_recent_processing(temp_db):
    from core import database
    a = database.add_task("SEN-a", "code", {})
    b = database.add_task("SEN-b", "code", {})
    c = database.add_task("SEN-c", "code", {})
    database._test_only_force_processing(a, _back(3600))
    database._test_only_force_processing(b, _back(60))  # most recent
    database._test_only_force_processing(c, _back(120))
    target = database.find_kill_target()
    assert target is not None
    assert target["task_id"] == b
    assert target["command"] == "code"


def test_k09_get_task_by_trace_id_round_trip(temp_db):
    from core import database
    tid = database.add_task("SEN-trace1", "search", {"q": "x"})
    row = database.get_task_by_trace_id("SEN-trace1")
    assert row is not None
    assert row.task_id == tid


def test_k10_get_task_by_trace_id_missing_returns_none(temp_db):
    from core import database
    assert database.get_task_by_trace_id("SEN-nope") is None


# ============================================================
# Group W: wiring source-checks
# ============================================================


def test_w01_worker_startup_calls_recover_stale_with_force_flag():
    src = (PROJECT / "core" / "worker.py").read_text(encoding="utf-8")
    # The force_all_processing=True kwarg must appear in worker.main.
    assert "force_all_processing=True" in src, (
        "worker startup must invoke recover_stale with "
        "force_all_processing=True"
    )


def test_w02_telegram_registers_kill_handler():
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    assert 'CommandHandler("kill"' in src
    assert "handle_kill" in src


def test_w03_kill_command_in_botfather_menu():
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    assert '("kill", ' in src, (
        "/kill must be advertised in BOT_COMMAND_MENU so the "
        "Telegram client suggestions surface include it"
    )


def test_w04_pipeline_polls_kill_between_attempts():
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert "is_kill_requested(" in src
    assert "bailed_on_kill" in src
    assert "qwen_killed" in src


def test_w05_pipeline_skips_kb_add_limitation_on_kill():
    """Kill-bail path must NOT call kb.add_limitation -- /kill is
    user-initiated, not a capability signal."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    # Find the bailed_on_kill branch.
    idx = src.find("if bailed_on_kill:")
    assert idx > 0
    # Within ~600 chars after that, must mention skipping limitation.
    branch = src[idx:idx + 800]
    assert "qwen_killed" in branch
    # The else branch (non-kill, non-repetition) is what calls
    # add_limitation. Bail branch shouldn't.
    assert "kb.add_limitation" not in branch.split("else:")[0]


def test_w06_telegram_render_branch_includes_qwen_killed():
    """qwen_killed must be in the ready-to-display markdown tuple,
    NOT fall into the legacy 'Here's the code:' wrapper."""
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    # Find the solved_by tuple.
    idx = src.find('"qwen_skip_path"')
    assert idx > 0
    # Within ~200 chars there must be qwen_killed.
    nearby = src[idx:idx + 400]
    assert '"qwen_killed"' in nearby


def test_w07_handle_kill_uses_find_kill_target_first():
    """The /kill handler must find the most-recent processing task
    via find_kill_target, not blindly iterate all tasks."""
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    idx = src.find("async def handle_kill(")
    assert idx > 0
    body = src[idx:idx + 2000]
    assert "find_kill_target" in body
    assert "request_kill" in body
