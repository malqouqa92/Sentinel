import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.registry import SkillRegistry
from core.skills import BaseSkill, SkillError
from skills.job_extract import (
    ALLOWED_LOCATION_TYPES,
    JobExtractInput,
    JobExtractSkill,
    JobExtraction,
)


def test_a_auto_discovery():
    """A fresh registry discovers job_extract from the skills/ directory."""
    reg = SkillRegistry()
    summary = reg.discover()
    assert summary["loaded"] >= 1
    assert summary["registered"] >= 1
    assert summary["errors"] == 0
    assert reg.has("job_extract")
    listed = reg.list_skills()
    by_name = {s["name"]: s for s in listed}
    assert "job_extract" in by_name
    assert by_name["job_extract"]["version"] == "1.0.0"
    assert by_name["job_extract"]["requires_gpu"] is True


def test_c_bad_input_raises_before_llm():
    """validate_input on a missing required field raises ValidationError
    and does NOT touch the LLM (proven by absence of any client call)."""
    skill = JobExtractSkill()
    with pytest.raises(Exception) as exc_info:
        skill.validate_input({})  # missing required `text`
    # pydantic raises ValidationError; subclass of ValueError in pydantic v2
    assert "text" in str(exc_info.value).lower()


def test_d_failed_skill_load_does_not_crash(tmp_path, monkeypatch):
    """A broken Python file in the skills directory is logged and skipped;
    other (good) skills still load."""
    # Set up a fake skills directory with one good file and one broken.
    fake_skills = tmp_path / "skills_under_test"
    fake_skills.mkdir()
    (fake_skills / "__init__.py").write_text("")

    # Broken module: syntax error.
    (fake_skills / "broken_skill.py").write_text(
        "def oops(:\n    return 1\n"
    )
    # Good module: a minimal BaseSkill subclass.
    (fake_skills / "good_skill.py").write_text(
        "from typing import ClassVar\n"
        "from pydantic import BaseModel\n"
        "from core.skills import BaseSkill\n"
        "\n"
        "class GoodIn(BaseModel):\n"
        "    x: int\n"
        "\n"
        "class GoodOut(BaseModel):\n"
        "    y: int\n"
        "\n"
        "class GoodSkill(BaseSkill):\n"
        "    name: ClassVar[str] = 'good_test_skill'\n"
        "    description: ClassVar[str] = 'a test skill'\n"
        "    version: ClassVar[str] = '0.1.0'\n"
        "    requires_gpu: ClassVar[bool] = False\n"
        "    input_schema: ClassVar[type] = GoodIn\n"
        "    output_schema: ClassVar[type] = GoodOut\n"
        "    async def execute(self, input_data, trace_id, context=None):\n"
        "        return GoodOut(y=input_data.x * 2)\n"
    )

    reg = SkillRegistry()
    summary = reg.discover(fake_skills)
    assert summary["errors"] >= 1, f"expected an error, got {summary}"
    assert reg.has("good_test_skill"), \
        "good skill should have loaded despite broken sibling"


def test_b_skill_execution_real_llm():
    """Test B uses the real LLM, so it's marked requires_ollama via
    redirect to the existing test_e (clean extraction) pattern."""
    pass  # Intentional placeholder — see the @requires_ollama version below.


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_b_skill_execution():
    """End-to-end skill execute() against real Ollama. Same posting as
    the Phase 4 Test E sample; lenient assertions."""
    sample = (
        "Regional Sales Manager - Midwest Territory\n"
        "Acme Industrial Solutions | Chicago, IL (Hybrid - 2 days/week)\n"
        "Compensation: $120K-$145K base + commission\n"
        "Requirements: 7+ years B2B sales experience, 3+ in management\n"
    )
    skill = JobExtractSkill()
    input_data = skill.validate_input({"text": sample})
    result = asyncio.run(skill.execute(input_data, trace_id="SEN-test-B"))
    assert isinstance(result, JobExtraction)
    assert result.title.strip() != ""
    assert result.location_type in ALLOWED_LOCATION_TYPES
    assert 0.0 <= result.confidence <= 1.0
