"""End-to-end pipeline tests for Phase 7."""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, database, worker
from core.knowledge_base import KnowledgeBase
from core.router import route


@pytest.fixture
def fresh_kb(tmp_path, monkeypatch):
    db = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db)
    yield KnowledgeBase(db_path=db)


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_DIR", ws)
    yield ws


async def _drain(timeout: float = 240.0) -> None:
    shutdown = asyncio.Event()
    await asyncio.wait_for(
        worker.worker_loop(shutdown, idle_exit=True), timeout=timeout,
    )


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_m_code_through_pipeline(fresh_kb, temp_workspace):
    """Routes /code through full pipeline: router -> queue -> worker
    -> orchestrator -> code_assistant agent -> code_assist skill ->
    KB lookup -> Qwen -> executor validation -> result in DB."""
    r = route('/code "write a python function to check if a string is a palindrome"')
    assert r.status == "ok" and r.task_id is not None

    asyncio.run(_drain(timeout=900.0))

    task = database.get_task(r.task_id)
    assert task is not None
    assert task.status == "completed", \
        f"task did not complete: status={task.status} error={task.error}"
    # Result should be the CodeAssistOutput dict.
    assert "solution" in task.result
    assert "solved_by" in task.result
    assert task.result["solved_by"] in (
        "qwen", "qwen_taught", "claude_direct",
        "qwen_agent", "qwen_failed",
    )
    # GPU lock released after.
    assert database.acquire_lock("gpu", "__post_M") is True
    database.release_lock("gpu", "__post_M")


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_n_code_with_context_flag(fresh_kb, temp_workspace):
    """Routes /code --context "<bad code>" "fix this" through pipeline."""
    r = route(
        '/code --context "def add(a, b): return a - b" '
        '"fix the bug in this function so add(2,3) returns 5"'
    )
    assert r.status == "ok" and r.task_id is not None

    asyncio.run(_drain(timeout=900.0))

    task = database.get_task(r.task_id)
    assert task.status == "completed", \
        f"task did not complete: status={task.status} error={task.error}"
    assert "solution" in task.result
    # We don't strict-assert that the solution is correct -- Qwen 3B is
    # unreliable -- but we assert the pipeline completed cleanly.


def test_o_regression_marker():
    """Test O's intent: Phase 3/4/5/6 tests still pass alongside this
    one. The full pytest suite is the actual proof."""
    from core.agent_registry import AGENT_REGISTRY
    from core.registry import SKILL_REGISTRY

    skill_names = {s["name"] for s in SKILL_REGISTRY.list_skills()}
    assert {
        "job_extract", "web_search", "file_io", "code_execute",
        "code_assist",
    }.issubset(skill_names)

    assert AGENT_REGISTRY.has("job_analyst")
    assert AGENT_REGISTRY.has("code_assistant")

    for cmd in ("/ping", "/help", "/status",
                "/extract", "/search", "/file", "/exec", "/code"):
        assert cmd in config.REGISTERED_COMMANDS
        assert cmd in config.COMMAND_AGENT_MAP
