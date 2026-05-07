import asyncio
import signal
import sys
import traceback

from core import config, database, orchestrator
from core.agent_registry import AGENT_REGISTRY
from core.database import TaskRow
from core.llm import INFERENCE_CLIENT
from core.logger import log_event
from core.registry import SKILL_REGISTRY


# Module-level shutdown event so main.py can set it on SIGINT/SIGTERM
# without needing a reference to a per-call event inside worker_loop().
shutdown_event: asyncio.Event | None = None


def get_or_create_shutdown_event() -> asyncio.Event:
    global shutdown_event
    if shutdown_event is None:
        shutdown_event = asyncio.Event()
    return shutdown_event


async def execute_task(task: TaskRow) -> None:
    log_event(task.trace_id, "INFO", "worker",
              f"executing task_id={task.task_id} command={task.command}")

    needs_gpu = orchestrator.needs_gpu(task.command)
    if needs_gpu:
        got_lock = await asyncio.to_thread(
            database.acquire_lock, "gpu", task.task_id
        )
        if not got_lock:
            await asyncio.to_thread(database.requeue_task, task.task_id)
            log_event(task.trace_id, "INFO", "worker",
                      f"gpu busy -- requeued task_id={task.task_id}")
            # Phase 12.5 hardening: backoff before returning so the
            # worker doesn't claim->requeue at 50/sec when the lock
            # is held by a stale/orphaned task. Without this we hit
            # the runaway pattern observed during /restart races.
            await asyncio.sleep(config.WORKER_GPU_REQUEUE_BACKOFF_S)
            return

    try:
        result = await orchestrator.dispatch(task)
        await asyncio.to_thread(database.complete_task, task.task_id, result)
        log_event(task.trace_id, "INFO", "worker",
                  f"completed task_id={task.task_id} result={result}")
    except Exception:
        tb = traceback.format_exc()
        await asyncio.to_thread(database.fail_task, task.task_id, tb)
        log_event(task.trace_id, "WARNING", "worker",
                  f"raised task_id={task.task_id} error={tb.splitlines()[-1]}")
    finally:
        if needs_gpu:
            await asyncio.to_thread(
                database.release_lock, "gpu", task.task_id
            )
            # Pre-Phase-15b: previously we ALWAYS unloaded the worker
            # model after a GPU task to free VRAM for the brain. That
            # was correct on tight 4 GB when the brain (1.4 GB) + worker
            # (1.9 GB) couldn't co-reside with KV cache + all-minilm
            # embedder. With pre-15a the embedder is ~45 MB; combined
            # ~3.4 GB leaves ~600 MB for KV cache -- comfortable for
            # short prompts, occasional evict for long ones.
            #
            # Trust OLLAMA_KEEP_ALIVE (2m) to manage eviction now:
            #   - If user runs /code then chats: worker stays warm for
            #     2m AND brain loads alongside it (no swap penalty).
            #   - If 2m of idle: Ollama evicts naturally.
            #   - If VRAM pressured: Ollama evicts the older model
            #     before loading the new one (built-in behaviour).
            #
            # Net win: subsequent calls to the same model no longer
            # pay the cold-load tax. Pattern #52's 15-minute graduation
            # hang was caused exactly by this -- worker unloaded right
            # before graduation needed it. The 60s graduation cap in
            # Phase 14a polish was a band-aid; this is the root fix.
            pass


async def worker_loop(
    shutdown_event: asyncio.Event,
    idle_exit: bool = False,
) -> None:
    log_event("SEN-system", "INFO", "worker", "worker loop started")
    while not shutdown_event.is_set():
        task = await asyncio.to_thread(database.claim_next_task)
        if task is None:
            if idle_exit:
                log_event("SEN-system", "INFO", "worker",
                          "idle — exiting (idle_exit)")
                return
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=config.WORKER_POLL_INTERVAL,
                )
            except asyncio.TimeoutError:
                pass
            continue
        await execute_task(task)
        if shutdown_event.is_set():
            log_event(task.trace_id, "INFO", "worker",
                      f"worker shutting down gracefully — finished "
                      f"task_id={task.task_id}")
            return
    log_event("SEN-system", "INFO", "worker",
              "worker loop exited (shutdown signaled)")


def install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    shutdown_event: asyncio.Event,
) -> None:
    """Cross-platform shutdown wiring. On POSIX uses loop.add_signal_handler
    where available; on Windows falls back to signal.signal that schedules
    the event-set on the loop thread."""
    def _request_shutdown() -> None:
        loop.call_soon_threadsafe(shutdown_event.set)

    if sys.platform == "win32":
        signal.signal(signal.SIGINT, lambda *_: _request_shutdown())
        try:
            signal.signal(signal.SIGBREAK, lambda *_: _request_shutdown())
        except (AttributeError, ValueError):
            pass
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_shutdown)
            except NotImplementedError:
                signal.signal(sig, lambda *_: _request_shutdown())


async def main() -> None:
    await asyncio.to_thread(database.init_db)
    # Phase 17a: ALL `processing` tasks at startup are zombies (no live
    # worker existed before this one). Skip the `updated_at < cutoff`
    # filter that would miss zombies whose updated_at was refreshed by
    # the dying worker's last claim.
    summary = await asyncio.to_thread(
        database.recover_stale, force_all_processing=True,
    )
    if summary["recovered"] or summary["failed"] or summary["locks_released"]:
        log_event("SEN-system", "WARNING", "worker",
                  f"crash recovery on startup: {summary}")

    skill_summary = await asyncio.to_thread(SKILL_REGISTRY.discover)
    log_event("SEN-system", "INFO", "worker",
              f"skill discovery: {skill_summary}")

    agent_summary = await asyncio.to_thread(AGENT_REGISTRY.discover)
    log_event("SEN-system", "INFO", "worker",
              f"agent discovery: {agent_summary}")

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    install_signal_handlers(loop, shutdown_event)
    log_event("SEN-system", "INFO", "worker", "worker ready")
    await worker_loop(shutdown_event)


if __name__ == "__main__":
    asyncio.run(main())
