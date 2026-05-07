# Qwen-solo code assist: KB search -> recipe generation -> stepfed execution.
# No Claude, no graduation, no retries. Pure Qwen pipeline for /qcode.
#
# Flow:
#   1. KB lookup (patterns only, max 5)
#   2. Build system prompt via _qwen_shadow_system_prompt()
#   3. Ask Qwen for a STEP-N recipe via _qwen_generate(format_json=False)
#   4. Execute recipe via run_agent_stepfed()
#   5. Post-stepfed verifier: git diff against base_sha to confirm real changes
#   6. Return QcodeAssistOutput(solved_by='qwen_solo' | 'qwen_solo_failed')
import asyncio
from typing import ClassVar

from pydantic import BaseModel

from core import config
from core.knowledge_base import KnowledgeBase
from core.llm import LLMError
from core.logger import log_event
from core.qwen_agent import run_agent_stepfed
from core.skills import BaseSkill, SkillError
from skills.code_assist import (
    _git_diff_stat,
    _git_snapshot,
    _qwen_generate,
    _qwen_shadow_system_prompt,
)


# ----------------------- I/O schemas ---------------------------------

class QcodeAssistInput(BaseModel):
    text: str  # maps to problem


class QcodeAssistOutput(BaseModel):
    solution: str
    solved_by: str = "qwen_solo"
    steps_executed: int
    completed: bool
    diff_stat: str = ""  # git diff --stat since base_sha; empty = no changes


# ----------------------- Skill ---------------------------------------

class QcodeAssistSkill(BaseSkill):
    name: ClassVar[str] = "qcode_assist"
    description: ClassVar[str] = (
        "Qwen-only code assist: KB search + recipe generation + stepfed "
        "execution. No Claude, no retries, no graduation."
    )
    version: ClassVar[str] = "1.0.0"
    requires_gpu: ClassVar[bool] = True
    input_schema: ClassVar[type[BaseModel]] = QcodeAssistInput
    output_schema: ClassVar[type[BaseModel]] = QcodeAssistOutput

    async def execute(
        self,
        input_data: BaseModel,
        trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        inp: QcodeAssistInput = input_data  # type: ignore[assignment]
        problem = inp.text
        model = config.WORKER_MODEL

        # 1. KB lookup -- filter to patterns only after search
        try:
            kb = KnowledgeBase()
            entries = kb.search(problem, max_results=5)
            pattern_entries = [e for e in entries if e.category == "pattern"]
        except Exception as exc:
            log_event(trace_id, "WARNING", "qcode_assist",
                      f"KB search failed: {exc}")
            pattern_entries = []

        # Build KB context block
        kb_block = ""
        if pattern_entries:
            lines = []
            for e in pattern_entries:
                body = e.solution_code or e.solution_pattern or ""
                lines.append(f"# {e.problem_summary}\n{body}")
            kb_block = "\n\n".join(lines)

        # 2. Build system prompt
        system = _qwen_shadow_system_prompt()

        # 3. Build user message
        user_parts = [f"Problem: {problem}"]
        if kb_block:
            user_parts.append(
                f"\nKB context (similar solved patterns):\n{kb_block}"
            )
        user_msg = "\n".join(user_parts)

        # 4. Generate recipe via Qwen
        # format_json=False so Ollama does not override the STEP-N
        # format directive with JSON mode (Phase 15d-bugfix)
        log_event(trace_id, "INFO", "qcode_assist",
                  f"generating recipe for: {problem[:80]!r}")
        try:
            recipe = await asyncio.to_thread(
                _qwen_generate,
                system, user_msg, trace_id, model,
                900, False,  # timeout=900, format_json=False
            )
        except LLMError as exc:
            raise SkillError("qcode_assist",
                             f"Ollama failure: {exc}", trace_id)

        # Snapshot HEAD before stepfed so we can diff after
        base_sha = await _git_snapshot(trace_id)

        # Capture pre-stepfed diff so we can isolate changes from THIS run only
        pre_diff_stat = ""
        try:
            pre_diff_stat = await _git_diff_stat(base_sha)
        except Exception as exc:
            log_event(trace_id, "WARNING", "qcode_assist",
                      f"pre-stepfed diff check failed: {exc}")

        # 5. Execute recipe via stepfed
        log_event(trace_id, "INFO", "qcode_assist",
                  f"executing recipe via stepfed ({len(recipe)} chars)")
        try:
            result = await asyncio.to_thread(
                run_agent_stepfed,
                problem, recipe, trace_id, model,
            )
        except LLMError as exc:
            raise SkillError("qcode_assist",
                             f"Ollama failure in stepfed: {exc}", trace_id)

        completed = result.get("completed_via_done", False)
        steps_executed = result.get("steps", 0)
        summary = result.get("summary", recipe[:200])

        # 6. Post-stepfed verifier: compare pre vs post to isolate NEW changes only
        post_diff_stat = ""
        try:
            post_diff_stat = await _git_diff_stat(base_sha)
        except Exception as exc:
            log_event(trace_id, "WARNING", "qcode_assist",
                      f"post-stepfed diff check failed: {exc}")

        if post_diff_stat == pre_diff_stat:
            # Stepfed produced zero new changes -- prior dirty state would have
            # fooled a simple non-empty check into falsely claiming success
            solved_by = "qwen_solo_failed"
            new_diff = ""
            summary = (
                "qwen claimed done but no new files changed -- "
                "likely silent edit_file anchor mismatch in the solution"
            )
            log_event(trace_id, "WARNING", "qcode_assist",
                      f"stepfed no-op: completed={completed} "
                      f"steps={steps_executed} -- pre==post diff (no new changes)")
        else:
            solved_by = "qwen_solo"
            # Show only this run's contribution: if tree was clean before, post
            # IS the new diff; if prior changes existed, annotate both baselines
            if not pre_diff_stat:
                new_diff = post_diff_stat
            else:
                new_diff = (
                    f"[prior uncommitted]\n{pre_diff_stat}\n"
                    f"[after this run]\n{post_diff_stat}"
                )
            log_event(trace_id, "INFO", "qcode_assist",
                      f"stepfed done: completed={completed} "
                      f"steps={steps_executed} new_diff={new_diff[:120]!r}")

        return QcodeAssistOutput(
            solution=summary,
            solved_by=solved_by,
            steps_executed=steps_executed,
            completed=completed,
            diff_stat=new_diff,
        )
