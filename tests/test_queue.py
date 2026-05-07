import asyncio
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import database, worker
from core.router import route


def _iso_minutes_ago(minutes: int) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(minutes=minutes)
    ).isoformat()


@pytest.fixture
def fail_handlers():
    """Register __fail_once and __fail_always handlers for the duration
    of the test by injecting them as built-in commands in the orchestrator.
    Yields the call-count dict so tests can inspect it."""
    from core import config, orchestrator
    counts = {"once": 0, "always": 0}

    async def fail_once(task):
        counts["once"] += 1
        if counts["once"] == 1:
            raise RuntimeError("first attempt fails (by design)")
        return {"response": "succeeded after retry"}

    async def fail_always(task):
        counts["always"] += 1
        raise RuntimeError("this handler always fails (by design)")

    orchestrator.BUILTIN_HANDLERS["__fail_once"] = fail_once
    orchestrator.BUILTIN_HANDLERS["__fail_always"] = fail_always
    config.COMMAND_AGENT_MAP["/__fail_once"] = None
    config.COMMAND_AGENT_MAP["/__fail_always"] = None
    try:
        yield counts
    finally:
        orchestrator.BUILTIN_HANDLERS.pop("__fail_once", None)
        orchestrator.BUILTIN_HANDLERS.pop("__fail_always", None)
        config.COMMAND_AGENT_MAP.pop("/__fail_once", None)
        config.COMMAND_AGENT_MAP.pop("/__fail_always", None)


async def _drain(timeout: float = 15.0) -> None:
    """Run worker_loop until no pending tasks remain, then return.
    idle_exit=True makes the worker stop the moment claim returns None."""
    shutdown = asyncio.Event()
    await asyncio.wait_for(
        worker.worker_loop(shutdown, idle_exit=True), timeout=timeout
    )


def test_a_submission_via_router():
    """Routing /ping hello produces a pending task in the DB whose
    fields match the router output."""
    result = route("/ping hello")
    assert result.status == "ok"
    assert result.task_id is not None

    task = database.get_task(result.task_id)
    assert task is not None
    assert task.status == "pending"
    assert task.trace_id == result.trace_id
    assert task.command == "ping"
    assert task.args == {"text": "hello"}
    assert task.priority == 0
    assert task.retry_count == 0
    assert task.recovery_count == 0


def test_f_no_double_processing():
    """Two concurrent claim_next_task() calls on the same pending task —
    only one wins. BEGIN IMMEDIATE serializes the transactions."""
    task_id = database.add_task("SEN-test001", "ping", {"text": "race"})
    results: list = []
    lock = threading.Lock()

    def claim():
        result = database.claim_next_task()
        with lock:
            results.append(result)

    t1 = threading.Thread(target=claim)
    t2 = threading.Thread(target=claim)
    t1.start(); t2.start()
    t1.join(); t2.join()

    winners = [r for r in results if r is not None]
    losers = [r for r in results if r is None]
    assert len(winners) == 1, f"expected 1 winner, got {len(winners)}"
    assert len(losers) == 1, f"expected 1 loser, got {len(losers)}"
    assert winners[0].task_id == task_id
    assert winners[0].status == "processing"


def test_g_stale_recovery():
    """A task stuck in 'processing' for >threshold gets reset to pending,
    recovery_count increments, and a WARNING is logged."""
    task_id = database.add_task("SEN-test002", "ping", {"text": "stale"})
    # Force into processing with an updated_at 10 minutes in the past.
    database._test_only_force_processing(task_id, _iso_minutes_ago(10))

    # Confirm it's stuck before recovery.
    before = database.get_task(task_id)
    assert before is not None and before.status == "processing"
    assert before.recovery_count == 0

    summary = database.recover_stale(timeout_seconds=60)
    assert summary["recovered"] == 1
    assert summary["failed"] == 0

    after = database.get_task(task_id)
    assert after is not None
    assert after.status == "pending"
    assert after.recovery_count == 1


def test_g_stale_lock_release():
    """A lock older than the threshold gets released by recover_stale."""
    database._test_only_force_lock("gpu", "ghost-task", _iso_minutes_ago(10))
    summary = database.recover_stale(timeout_seconds=60)
    assert summary["locks_released"] == 1
    # Now the lock should be acquirable by a new task.
    assert database.acquire_lock("gpu", "new-task") is True
    database.release_lock("gpu", "new-task")


def test_b_processing():
    """Worker picks up the pending /ping task and completes it with
    {'response': 'pong'}."""
    result = route("/ping hello")
    asyncio.run(_drain())
    task = database.get_task(result.task_id)
    assert task is not None
    assert task.status == "completed"
    assert task.result == {"response": "pong"}
    assert task.error is None


def test_c_serial_ordering():
    """Three tasks submitted rapidly all complete; their completion order
    matches their insertion order."""
    r1 = route("/ping one")
    r2 = route("/ping two")
    r3 = route("/ping three")

    asyncio.run(_drain())

    tasks = [database.get_task(r.task_id) for r in (r1, r2, r3)]
    for t in tasks:
        assert t is not None and t.status == "completed", \
            f"task not completed: {t}"
    # Completion order = updated_at order. Insertion order = created_at.
    # If the worker is serial and FIFO, both orderings match.
    assert tasks[0].created_at <= tasks[1].created_at <= tasks[2].created_at
    assert tasks[0].updated_at <= tasks[1].updated_at <= tasks[2].updated_at


def test_d_retry_on_failure(fail_handlers):
    """A task whose handler fails on first attempt: retry_count → 1,
    status returns to pending; on the second attempt it succeeds."""
    task_id = database.add_task("SEN-d-test", "__fail_once",
                                {"text": "retry me"})
    asyncio.run(_drain())
    task = database.get_task(task_id)
    assert task is not None
    assert task.status == "completed"
    assert task.retry_count == 1
    assert task.result == {"response": "succeeded after retry"}
    assert fail_handlers["once"] == 2  # one fail + one success


def test_e_permanent_failure(fail_handlers):
    """A task that always fails, with max_retries=2: after 2 retries,
    status=failed, retry_count=2, error populated."""
    task_id = database.add_task("SEN-e-test", "__fail_always",
                                {"text": "doomed"}, max_retries=2)
    asyncio.run(_drain())
    task = database.get_task(task_id)
    assert task is not None
    assert task.status == "failed"
    assert task.retry_count == 2
    assert task.error is not None and "always fails" in task.error
    assert fail_handlers["always"] == 2


def test_h_graceful_shutdown():
    """Submit a task, start the worker, signal shutdown mid-execution.
    The current task must complete; no orphaned 'processing' rows remain;
    worker exits."""
    result = route("/ping shutdown-test")

    async def scenario():
        shutdown = asyncio.Event()

        async def signaller():
            # Wait until the worker has actually claimed the task.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                t = await asyncio.to_thread(
                    database.get_task, result.task_id
                )
                if t and t.status == "processing":
                    break
                await asyncio.sleep(0.02)
            shutdown.set()

        await asyncio.gather(
            worker.worker_loop(shutdown),
            signaller(),
        )

    asyncio.run(asyncio.wait_for(scenario(), timeout=10.0))

    task = database.get_task(result.task_id)
    assert task is not None
    assert task.status == "completed", \
        f"task should have finished before shutdown: {task}"

    orphans = [t for t in database.list_tasks()
               if t.status == "processing"]
    assert orphans == [], f"orphaned processing tasks: {orphans}"
