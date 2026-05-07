"""Agent factory: composes skills into ordered pipelines.

An Agent is a configured composition of skills with a persona prompt and
optional model override. Pipelines are deterministic (fixed sequence
chosen by the human) -- not LLM-selected -- because small models
(<=3B) cannot reliably do tool selection. Dynamic selection is a future
phase when a stronger model and a tool-call validation layer exist.

Phase 10: every successful pipeline run also appends one episodic
memory entry (scope=agent_name) so the brain can recall what each
agent has been doing recently. Failures don't append -- error
envelopes are surfaced to chat, not memorialized.
"""
import traceback
from typing import Any

from pydantic import BaseModel

from core.logger import log_event
from core.registry import SKILL_REGISTRY, SkillRegistry
from core.skills import BaseSkill, SkillError


def _record_episode(
    scope: str, trace_id: str, summary: str,
    detail: str = "", tags: list[str] | None = None,
) -> None:
    """Best-effort episodic memory append. Late import + try/except
    so a memory hiccup never breaks a successful pipeline."""
    try:
        from core.memory import get_memory  # noqa: PLC0415
        get_memory().store_episode(
            scope=scope, trace_id=trace_id,
            event_type="pipeline_completed",
            summary=summary[:500], detail=detail[:2000],
            tags=tags or [], relevance_score=1.0,
        )
    except Exception as e:
        log_event(
            trace_id, "DEBUG", f"agent.{scope}",
            f"episodic memory append skipped: "
            f"{type(e).__name__}: {e}",
        )


class AgentConfig(BaseModel):
    name: str
    description: str
    persona_prompt: str
    skill_pipeline: list[str]
    model: str | None = None


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        registry: SkillRegistry | None = None,
    ) -> None:
        self.config = config
        self._registry = registry or SKILL_REGISTRY

        if not config.skill_pipeline:
            raise ValueError(
                f"agent '{config.name}' has empty skill_pipeline"
            )

        missing = [
            s for s in config.skill_pipeline if not self._registry.has(s)
        ]
        if missing:
            raise ValueError(
                f"agent '{config.name}' references missing skills: "
                f"{missing}. Registered skills: "
                f"{[s['name'] for s in self._registry.list_skills()]}"
            )

        self._skills: list[BaseSkill] = [
            self._registry.get(name) for name in config.skill_pipeline
        ]

        # Soft I/O compatibility check: if skill[i].output_schema is not
        # exactly skill[i+1].input_schema, warn about required fields
        # that won't be available. Doesn't block agent creation -- the
        # validator at runtime is the source of truth.
        for i in range(len(self._skills) - 1):
            cur = self._skills[i]
            nxt = self._skills[i + 1]
            if cur.output_schema is nxt.input_schema:
                continue
            cur_fields = set(cur.output_schema.model_fields.keys())
            req_next = {
                name for name, field in nxt.input_schema.model_fields.items()
                if field.is_required()
            }
            missing_in = req_next - cur_fields
            if missing_in:
                log_event(
                    "SEN-system", "WARNING", f"agent.{config.name}",
                    f"pipeline I/O mismatch: {cur.name}.output is missing "
                    f"required fields for {nxt.name}.input: "
                    f"{sorted(missing_in)}",
                )

    async def run(
        self, input_data: dict, trace_id: str
    ) -> dict[str, Any]:
        """Execute the skill pipeline. Returns the final skill's
        model_dump() on success. On any skill failure, returns an error
        envelope: {"_error": True, "error": str, "failed_at": skill_name,
        "trace_id": trace_id}. Pipeline halts on first failure."""
        log_event(
            trace_id, "INFO", f"agent.{self.config.name}",
            f"pipeline starting steps={self.config.skill_pipeline}",
        )

        current: Any = input_data
        context = (
            {"model": self.config.model}
            if self.config.model is not None
            else None
        )

        for skill in self._skills:
            log_event(
                trace_id, "INFO", f"agent.{self.config.name}",
                f"step start skill={skill.name}",
            )

            # Phase 10 fan-out: when current is a list and the next
            # skill expects a single item (accepts_list=False), call
            # the skill once per item and aggregate results.
            try:
                if isinstance(current, list) and not skill.accepts_list:
                    aggregated: list[dict] = []
                    for idx, item in enumerate(current):
                        sub_in = skill.validate_input(item)
                        sub_result = await skill.execute(
                            sub_in, trace_id, context=context,
                        )
                        if not isinstance(sub_result, BaseModel):
                            raise SkillError(
                                skill.name,
                                f"item {idx}: returned "
                                f"{type(sub_result).__name__}, "
                                f"expected BaseModel",
                                trace_id,
                            )
                        aggregated.append(sub_result.model_dump())
                    current = aggregated
                    log_event(
                        trace_id, "INFO", f"agent.{self.config.name}",
                        f"step ok (fan-out) skill={skill.name} "
                        f"items={len(aggregated)}",
                    )
                    continue

                input_obj = skill.validate_input(current)
            except SkillError as e:
                log_event(
                    trace_id, "ERROR", f"agent.{self.config.name}",
                    f"step failed skill={skill.name} SkillError: {e}",
                )
                return {
                    "_error": True, "error": str(e),
                    "failed_at": skill.name, "trace_id": trace_id,
                }
            except Exception as e:
                err = (f"input validation failed for skill {skill.name}: "
                       f"{e}")
                log_event(
                    trace_id, "ERROR", f"agent.{self.config.name}", err,
                )
                return {
                    "_error": True, "error": str(e),
                    "failed_at": skill.name, "trace_id": trace_id,
                }

            try:
                result = await skill.execute(
                    input_obj, trace_id, context=context,
                )
            except SkillError as e:
                log_event(
                    trace_id, "ERROR", f"agent.{self.config.name}",
                    f"step failed skill={skill.name} SkillError: {e}",
                )
                return {
                    "_error": True, "error": str(e),
                    "failed_at": skill.name, "trace_id": trace_id,
                }
            except Exception as e:
                tb = traceback.format_exc()
                log_event(
                    trace_id, "ERROR", f"agent.{self.config.name}",
                    f"step failed skill={skill.name} unexpected: "
                    f"{tb.splitlines()[-1]}",
                )
                return {
                    "_error": True,
                    "error": f"{type(e).__name__}: {e}",
                    "failed_at": skill.name, "trace_id": trace_id,
                }

            if not isinstance(result, BaseModel):
                err = (f"skill {skill.name} returned "
                       f"{type(result).__name__}, expected BaseModel")
                log_event(
                    trace_id, "ERROR", f"agent.{self.config.name}", err,
                )
                return {
                    "_error": True, "error": err,
                    "failed_at": skill.name, "trace_id": trace_id,
                }

            dump = result.model_dump()
            # Phase 10: if the skill declares output_is_list, unwrap
            # the single list field so the next iteration can fan-out
            # over its items. Refuse to unwrap if the schema doesn't
            # actually have a single list field -- that's a skill bug,
            # not silently swallowable.
            if skill.output_is_list:
                if (
                    isinstance(dump, dict)
                    and len(dump) == 1
                    and isinstance(next(iter(dump.values())), list)
                ):
                    current = next(iter(dump.values()))
                else:
                    err = (
                        f"skill {skill.name} declared output_is_list "
                        f"but dump shape is "
                        f"{type(dump).__name__} keys={list(dump) if isinstance(dump, dict) else 'n/a'}"
                    )
                    log_event(
                        trace_id, "ERROR",
                        f"agent.{self.config.name}", err,
                    )
                    return {
                        "_error": True, "error": err,
                        "failed_at": skill.name, "trace_id": trace_id,
                    }
            else:
                current = dump
            log_event(
                trace_id, "INFO", f"agent.{self.config.name}",
                f"step ok skill={skill.name}",
            )

        log_event(
            trace_id, "INFO", f"agent.{self.config.name}",
            "pipeline completed",
        )
        _record_episode(
            scope=self.config.name,
            trace_id=trace_id,
            summary=(
                f"{self.config.name} completed: "
                f"{', '.join(self.config.skill_pipeline)}"
            ),
            detail=(
                f"input_keys={sorted(input_data.keys()) if isinstance(input_data, dict) else type(input_data).__name__} "
                f"output_keys={sorted(current.keys()) if isinstance(current, dict) else type(current).__name__}"
            ),
            tags=[self.config.name, *self.config.skill_pipeline],
        )
        return current
