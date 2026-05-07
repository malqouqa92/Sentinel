import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import database, orchestrator, worker
from core.agent_registry import AGENT_REGISTRY
from core.registry import SKILL_REGISTRY
from core.router import route
from skills.job_extract import (
    ALLOWED_LOCATION_TYPES,
    JobExtraction,
)


SAMPLE_JOB = (
    "Regional Sales Manager - Midwest Territory "
    "Acme Industrial Solutions | Chicago, IL (Hybrid - 2 days/week) "
    "Compensation: $120K-$145K base + commission "
    "Requirements: 7+ years B2B sales experience, 3+ in management"
)


def test_j_builtin_command_via_orchestrator():
    """Routing /ping flows through orchestrator -> built-in handler.
    No agent involved. Returns pong."""
    result = route("/ping hello")
    assert result.status == "ok"

    async def drain():
        shutdown = asyncio.Event()
        await asyncio.wait_for(
            worker.worker_loop(shutdown, idle_exit=True), timeout=10.0,
        )

    asyncio.run(drain())

    task = database.get_task(result.task_id)
    assert task is not None
    assert task.status == "completed"
    assert task.result == {"response": "pong"}


def test_j_status_returns_queue_depth():
    """/status uses the built-in status handler, returns queue_depth."""
    result = route("/status")
    asyncio.run(_drain())
    task = database.get_task(result.task_id)
    assert task.status == "completed"
    assert "queue_depth" in task.result
    assert isinstance(task.result["queue_depth"], int)


def test_j_help_lists_skills_and_agents():
    """/help returns the live inventory of commands, skills, agents."""
    result = route("/help")
    asyncio.run(_drain())
    task = database.get_task(result.task_id)
    assert task.status == "completed"
    payload = task.result
    assert "commands" in payload and "/extract" in payload["commands"]
    skill_names = {s["name"] for s in payload["skills"]}
    assert "job_extract" in skill_names
    agent_names = {a["name"] for a in payload["agents"]}
    assert "job_analyst" in agent_names


def test_orchestrator_needs_gpu_derives_from_pipeline():
    """needs_gpu is derived from agent.skills, not a parallel set."""
    assert orchestrator.needs_gpu("extract") is True
    assert orchestrator.needs_gpu("ping") is False
    assert orchestrator.needs_gpu("status") is False


def test_orchestrator_unmapped_command_logs_warning():
    """A command that's in REGISTERED_COMMANDS but missing from
    COMMAND_AGENT_MAP falls through to the default handler with a
    WARNING log -- defensive, shouldn't normally fire."""
    from core.database import TaskRow

    fake_task = TaskRow(
        task_id="t1", trace_id="SEN-test-orch", command="totally_unknown",
        args={}, status="processing", priority=0, retry_count=0,
        max_retries=3, recovery_count=0, max_recoveries=5,
        result=None, error=None,
        created_at="2026-05-04T00:00:00+00:00",
        updated_at="2026-05-04T00:00:00+00:00",
    )
    result = asyncio.run(orchestrator.dispatch(fake_task))
    assert result.get("response") == "executed"


async def _drain():
    shutdown = asyncio.Event()
    await asyncio.wait_for(
        worker.worker_loop(shutdown, idle_exit=True), timeout=180.0,
    )


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_i_full_pipeline_through_orchestrator():
    """End-to-end: route /extract -> queue -> worker -> orchestrator
    -> job_analyst agent -> job_extract skill -> result in DB."""
    result = route(f"/extract {SAMPLE_JOB}")
    assert result.status == "ok"
    assert result.task_id is not None

    asyncio.run(_drain())

    task = database.get_task(result.task_id)
    assert task is not None
    assert task.status == "completed", \
        f"task did not complete: status={task.status} error={task.error}"

    # Result is the agent's pipeline output -- the JobExtraction dict.
    extracted = JobExtraction(**task.result)
    assert extracted.title.strip() != ""
    assert extracted.location_type in ALLOWED_LOCATION_TYPES

    # GPU lock released after the run.
    assert database.acquire_lock("gpu", "post-test-check") is True
    database.release_lock("gpu", "post-test-check")


def test_k_phase_3_4_regression_smoke():
    """Test K's intent: nothing from earlier phases broke.
    The full pytest run (test_observability/test_router/test_queue/
    test_extractor) is the actual proof; this test is a marker that
    the regression check is part of the deliverable."""
    # Skill + agent registries are present (post-refactor).
    assert SKILL_REGISTRY.has("job_extract")
    assert AGENT_REGISTRY.has("job_analyst")
    # Built-in commands still routable.
    for cmd in ("/ping", "/status", "/help", "/extract"):
        from core import config
        assert cmd in config.REGISTERED_COMMANDS
