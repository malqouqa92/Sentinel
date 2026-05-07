import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agents import Agent, AgentConfig
from core.agent_registry import AgentRegistry
from core.registry import SKILL_REGISTRY
from core.skills import BaseSkill, SkillError
from skills.job_extract import (
    ALLOWED_LOCATION_TYPES,
    JobExtractInput,
    JobExtraction,
)


SAMPLE_JOB = (
    "Regional Sales Manager - Midwest Territory\n"
    "Acme Industrial Solutions | Chicago, IL (Hybrid - 2 days/week)\n"
    "Compensation: $120K-$145K base + commission\n"
    "Requirements: 7+ years B2B sales experience, 3+ in management\n"
)


def test_e_agent_creation():
    """Creating an agent with a valid skill pipeline succeeds; pipeline
    is validated against the global SkillRegistry."""
    cfg = AgentConfig(
        name="job_analyst",
        description="Analyzes job postings",
        persona_prompt="You are a technical recruiter.",
        skill_pipeline=["job_extract"],
    )
    agent = Agent(cfg)
    assert agent.config.name == "job_analyst"
    assert len(agent._skills) == 1
    assert agent._skills[0].name == "job_extract"


def test_g_missing_skill_in_pipeline_fails_at_init():
    """An agent referencing a non-existent skill fails at init with
    a clear error -- not at first run."""
    cfg = AgentConfig(
        name="broken_agent",
        description="references a skill that doesn't exist",
        persona_prompt="x",
        skill_pipeline=["nonexistent_skill"],
    )
    with pytest.raises(ValueError) as exc_info:
        Agent(cfg)
    msg = str(exc_info.value)
    assert "nonexistent_skill" in msg
    assert "broken_agent" in msg


def test_h_pipeline_error_propagation(monkeypatch):
    """When a skill raises SkillError, the agent halts the pipeline
    and returns a structured error envelope."""
    cfg = AgentConfig(
        name="job_analyst",
        description="x",
        persona_prompt="x",
        skill_pipeline=["job_extract"],
    )
    agent = Agent(cfg)

    # Replace the skill's execute with one that raises.
    skill = SKILL_REGISTRY.get("job_extract")

    async def boom(input_data, trace_id, context=None):
        raise SkillError(
            "job_extract", "deliberate failure for test_h", trace_id,
        )

    monkeypatch.setattr(skill, "execute", boom)

    result = asyncio.run(
        agent.run({"text": "anything"}, trace_id="SEN-test-H"),
    )
    assert result.get("_error") is True
    assert result["failed_at"] == "job_extract"
    assert "deliberate failure" in result["error"]


def test_agent_registry_yaml_discovery(tmp_path):
    """AgentRegistry.discover() loads YAML files and validates against
    the skill registry."""
    yaml_path = tmp_path / "test_agent.yaml"
    yaml_path.write_text(
        "name: test_yaml_agent\n"
        "description: yaml-loaded test agent\n"
        "persona_prompt: be helpful\n"
        "skill_pipeline:\n"
        "  - job_extract\n"
        "model: null\n",
        encoding="utf-8",
    )
    reg = AgentRegistry()
    summary = reg.discover(tmp_path)
    assert summary["loaded"] == 1
    assert summary["registered"] == 1
    assert summary["errors"] == 0
    assert reg.has("test_yaml_agent")
    listed = reg.list_agents()
    assert listed[0]["skill_pipeline"] == ["job_extract"]


def test_agent_registry_bad_yaml_logged_skipped(tmp_path):
    """A YAML referencing a missing skill is logged + skipped; sibling
    YAMLs still load."""
    (tmp_path / "broken.yaml").write_text(
        "name: broken_agent\n"
        "description: bad\n"
        "persona_prompt: x\n"
        "skill_pipeline:\n"
        "  - skill_that_does_not_exist\n",
        encoding="utf-8",
    )
    (tmp_path / "good.yaml").write_text(
        "name: good_agent_yaml\n"
        "description: ok\n"
        "persona_prompt: x\n"
        "skill_pipeline:\n"
        "  - job_extract\n",
        encoding="utf-8",
    )
    reg = AgentRegistry()
    summary = reg.discover(tmp_path)
    assert summary["errors"] >= 1
    assert reg.has("good_agent_yaml")
    assert not reg.has("broken_agent")


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_f_agent_run_real_llm():
    """Agent.run() over a real job posting returns a JobExtraction-shaped
    dict (the final skill's model_dump). Logged with trace_id and
    component='agent.job_analyst'."""
    cfg = AgentConfig(
        name="job_analyst",
        description="x",
        persona_prompt="x",
        skill_pipeline=["job_extract"],
    )
    agent = Agent(cfg)
    result = asyncio.run(
        agent.run({"text": SAMPLE_JOB}, trace_id="SEN-test-F-agent"),
    )
    assert result.get("_error") is not True, \
        f"agent reported error: {result}"
    # Validate as JobExtraction.
    extracted = JobExtraction(**result)
    assert extracted.title.strip() != ""
    assert extracted.location_type in ALLOWED_LOCATION_TYPES
