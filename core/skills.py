"""Skill foundation: BaseSkill abstract class + SkillError.

A skill is a single, named, versioned, schema-validated unit of work.
Skills live in the `skills/` directory, one file per skill, and are
auto-discovered by SkillRegistry at startup.
"""
from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel


class SkillError(Exception):
    """Raised by a skill's execute() when it cannot produce a valid result.
    The trace_id field allows correlating the failure to the originating
    request across the JSONL log."""

    def __init__(self, skill_name: str, message: str, trace_id: str):
        self.skill_name = skill_name
        self.trace_id = trace_id
        super().__init__(f"[{skill_name}] {message}")


class BaseSkill(ABC):
    """Base class for all skills.

    Subclasses MUST set the class attributes below and implement
    execute(). The class attributes are ClassVars because they describe
    the skill type, not per-instance state.

    Phase 10 fan-out:
    - ``output_is_list`` -- output_schema wraps a single list field
      (e.g. {"postings": [...]}); agent unwraps to bare list.
    - ``accepts_list`` -- skill consumes a list whole; agent passes
      the list as-is via validate_input. Skills MUST override
      validate_input to wrap the list into the expected field name.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    version: ClassVar[str]
    requires_gpu: ClassVar[bool] = False
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]
    accepts_list: ClassVar[bool] = False
    output_is_list: ClassVar[bool] = False

    @abstractmethod
    async def execute(
        self,
        input_data: BaseModel,
        trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        """Run the skill. Return a validated output_schema instance."""

    def validate_input(self, raw: dict) -> BaseModel:
        """Parse a raw dict against this skill's input_schema."""
        return self.input_schema(**raw)
