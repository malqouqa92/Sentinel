"""Gwen open-system skill (v2).

Routes /gwen invocations through `core.gwen_agent.run_gwen_open`:
  - LITERAL RECIPE (input starts with `STEP N:` or `/gwen STEP N:`)
    -> server-side parser dispatches each step against
    OPEN_TOOL_DISPATCH. No LLM call. 100% deterministic.
  - ENGLISH -> Qwen writes a STEP-N recipe via `_qwen_generate`,
    then the same executor runs it.

Tools are UNSANDBOXED (real filesystem + bash, paths can escape
PROJECT_ROOT). Authorized via Telegram TELEGRAM_AUTHORIZED_USERS
in the bot handler -- do not expose this skill outside that
authorization boundary.

History: v1 used `core.qwen_agent.run_agent` (Ollama native
tool-calling). qwen2.5-coder:3b consistently faked `done()` on
step 1 without dispatching real tools, so the bot would report
`completed=True steps=1` while nothing happened on disk. v2
replaces that path with the deterministic stepfed-style executor
that the rest of Sentinel already trusts.
"""
import asyncio
from typing import ClassVar

from pydantic import BaseModel

from core import config
from core.gwen_agent import run_gwen_open
from core.logger import log_event
from core.skills import BaseSkill, SkillError


# ----------------------- I/O schemas ---------------------------------

class GwenAssistInput(BaseModel):
    text: str  # the user's request (English or literal STEP-N)


class GwenAssistOutput(BaseModel):
    solution: str
    solved_by: str = "gwen_ok"   # 'gwen_ok' | 'gwen_failed'
    steps_executed: int
    completed: bool
    mode: str = "literal"        # 'literal' | 'english'


# ----------------------- Render helpers ------------------------------

def _format_session(session: list[dict]) -> str:
    """Compact human-readable trace for the Telegram reply.

    Each step renders as `STEP N: tool -> ok | error: <msg>`.
    Tool result bodies (stdout, file contents) are NOT included --
    they would blow past Telegram's 4096-char message cap and noise
    up the report.
    """
    lines = []
    for entry in session:
        i = entry.get("step", "?")
        tool = entry.get("tool", "?")
        result = entry.get("result", {}) or {}
        if "error" in result:
            lines.append(f"STEP {i}: {tool} -> ❌ {result['error'][:160]}")
        else:
            # Surface the most useful single field per tool
            if tool == "run_bash":
                rc = result.get("return_code", "?")
                stdout = (result.get("stdout") or "").strip()
                head = stdout.splitlines()[0] if stdout else "(no stdout)"
                lines.append(
                    f"STEP {i}: run_bash rc={rc} -> {head[:120]}"
                )
            elif tool == "read_file":
                tot = result.get("total_chars", "?")
                lines.append(f"STEP {i}: read_file ({tot} chars)")
            elif tool == "write_file":
                lines.append(
                    f"STEP {i}: write_file "
                    f"({result.get('bytes_written', '?')} bytes)"
                )
            elif tool == "edit_file":
                delta = result.get("lines_changed", "?")
                lines.append(
                    f"STEP {i}: edit_file (lines_changed={delta})"
                )
            elif tool == "list_dir":
                n = len(result.get("items", []) or [])
                lines.append(f"STEP {i}: list_dir ({n} entries)")
            elif tool == "done":
                lines.append(f"STEP {i}: done ✅")
            else:
                lines.append(f"STEP {i}: {tool} -> ok")
    return "\n".join(lines)


def _build_solution(result: dict) -> str:
    """Render the complete /gwen reply: header, per-step trace, summary."""
    mode = result.get("mode", "?")
    completed = result.get("completed_via_done", False)
    steps = result.get("steps", 0)
    summary = (result.get("summary") or "").strip()
    session = result.get("session", [])
    parser_err = result.get("error")
    recipe = result.get("recipe", "")

    head_icon = "✅" if completed else "⚠️"
    parts = [
        f"{head_icon} /gwen ({mode} mode, {steps} step(s))",
    ]
    if parser_err:
        parts.append(f"parser: {parser_err}")
    if mode == "english" and recipe:
        # Show the recipe Qwen authored so the user can audit and re-run
        parts.append("Qwen recipe:")
        parts.append(f"```\n{recipe[:1200]}\n```")
    if session:
        parts.append("Trace:")
        parts.append(_format_session(session))
    if summary:
        parts.append(f"Summary: {summary[:600]}")
    return "\n".join(parts)


# ----------------------- Skill ---------------------------------------

class GwenAssistSkill(BaseSkill):
    name: ClassVar[str] = "gwen_assist"
    description: ClassVar[str] = (
        "Gwen open-system: literal STEP-N recipe (no LLM) or "
        "English-to-recipe via Qwen. Unsandboxed file/bash access."
    )
    version: ClassVar[str] = "2.0.0"
    requires_gpu: ClassVar[bool] = True  # English path uses Qwen
    input_schema: ClassVar[type[BaseModel]] = GwenAssistInput
    output_schema: ClassVar[type[BaseModel]] = GwenAssistOutput

    async def execute(
        self,
        input_data: BaseModel,
        trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        inp: GwenAssistInput = input_data  # type: ignore[assignment]
        text = inp.text
        model = config.WORKER_MODEL

        log_event(trace_id, "INFO", "gwen_assist",
                  f"dispatching: {text[:80]!r}")
        try:
            result = await asyncio.to_thread(
                run_gwen_open, text, trace_id, model,
            )
        except Exception as exc:
            raise SkillError(
                "gwen_assist",
                f"run_gwen_open failed: {type(exc).__name__}: {exc}",
                trace_id,
            )

        completed = bool(result.get("completed_via_done"))
        steps_executed = int(result.get("steps", 0))
        mode = result.get("mode", "literal")
        solved_by = "gwen_ok" if completed else "gwen_failed"
        solution = _build_solution(result)

        log_event(
            trace_id,
            "INFO" if completed else "WARNING",
            "gwen_assist",
            f"finished: mode={mode} steps={steps_executed} "
            f"completed={completed} solved_by={solved_by!r}",
        )

        return GwenAssistOutput(
            solution=solution,
            solved_by=solved_by,
            steps_executed=steps_executed,
            completed=completed,
            mode=mode,
        )
