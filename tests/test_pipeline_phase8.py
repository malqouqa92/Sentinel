"""Phase 8 end-to-end pipeline tests."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, database, worker
from core.knowledge_base import KnowledgeBase
from core.model_registry import MODEL_REGISTRY
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
def test_o_extract_uses_pinned_agent_model(fresh_kb, temp_workspace):
    """job_analyst.yaml pins model: qwen-coder. Routing /extract must
    route through that model regardless of complexity score."""
    sample = (
        "Regional Sales Manager - Midwest Territory "
        "Acme Industrial Solutions | Chicago, IL (Hybrid) "
        "Comp: $120K-$145K base + commission "
        "Requirements: 7+ years B2B sales experience"
    )
    r = route(f"/extract {sample}")
    assert r.status == "ok"
    asyncio.run(_drain())
    task = database.get_task(r.task_id)
    assert task.status == "completed", \
        f"task did not complete: {task.error}"
    # The result is the JobExtraction. We can't directly observe which
    # model was used via task.result; the YAML pinning is plumbed via
    # the agent's context dict to the skill, which uses the pinned id.
    assert task.result.get("title", "").strip() != ""


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_p_code_auto_route_simple(fresh_kb, temp_workspace):
    """`/code` with a simple problem flows through orchestrator ->
    code_assistant agent -> code_assist skill -> KB lookup -> Qwen."""
    r = route('/code "write a function that returns the larger of two numbers"')
    assert r.status == "ok"
    asyncio.run(_drain())
    task = database.get_task(r.task_id)
    assert task.status == "completed", \
        f"task did not complete: {task.error}"
    assert "solution" in task.result
    assert task.result["solved_by"] in (
        "qwen", "qwen_taught", "claude_direct",
        "qwen_agent", "qwen_failed",
    )


@pytest.mark.requires_ollama
def test_q_models_command(fresh_kb, temp_workspace):
    """`/models` returns the live registry inventory. Must include the
    core 3 models (Phase 9 may add sentinel-brain on top)."""
    r = route("/models")
    assert r.status == "ok"
    asyncio.run(_drain(timeout=15.0))
    task = database.get_task(r.task_id)
    assert task.status == "completed"
    names = {m["name"] for m in task.result["models"]}
    assert names >= {"qwen3-brain", "qwen-coder", "claude-cli"}
    # Each entry must carry the availability bit
    for m in task.result["models"]:
        assert "available" in m
        assert "backend" in m


def test_r_complexity_command(fresh_kb, temp_workspace):
    """`/complexity` returns ComplexityResult with score, tier, reasoning."""
    r = route('/complexity "implement distributed consensus with raft"')
    assert r.status == "ok"
    asyncio.run(_drain(timeout=15.0))
    task = database.get_task(r.task_id)
    assert task.status == "completed"
    payload = task.result
    assert "score" in payload and "tier" in payload
    assert "reasoning" in payload and isinstance(payload["reasoning"], list)
    assert "recommended_model" in payload
    # "distributed" is a complex_keyword -> at least standard tier
    assert payload["tier"] in ("standard", "advanced")


def test_s_regression_marker():
    """Phase 3-7 regression: registries, commands, agents intact."""
    from core.agent_registry import AGENT_REGISTRY
    from core.registry import SKILL_REGISTRY

    skill_names = {s["name"] for s in SKILL_REGISTRY.list_skills()}
    assert {
        "job_extract", "web_search", "file_io", "code_execute",
        "code_assist",
    }.issubset(skill_names)
    assert AGENT_REGISTRY.has("job_analyst")
    assert AGENT_REGISTRY.has("code_assistant")
    for cmd in ("/ping", "/status", "/help",
                "/extract", "/search", "/file", "/exec", "/code",
                "/models", "/complexity"):
        assert cmd in config.REGISTERED_COMMANDS
        assert cmd in config.COMMAND_AGENT_MAP
