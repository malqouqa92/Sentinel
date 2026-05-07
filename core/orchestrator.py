"""Deterministic orchestrator: command -> agent (or built-in) lookup.

This is NOT an LLM call. It is a static mapping from `config.COMMAND_AGENT_MAP`.
The router decides whether the input is well-formed; the orchestrator
decides who handles it. Built-in commands (ping/status/help) are handled
inline; agent-backed commands route through the AgentRegistry.

Design rationale: dynamic LLM-based tool selection (ReAct etc.) is a
future phase that needs a stronger model and a tool-call validation
layer. For now humans (you) own the pipeline definitions; the
orchestrator just executes.
"""
from typing import Awaitable, Callable

from pydantic import BaseModel

from core import config, database
from core.agent_registry import AGENT_REGISTRY
from core.database import TaskRow
from core.logger import log_event
from core.registry import SKILL_REGISTRY
from core.skills import BaseSkill, SkillError

SKILL_PREFIX = "skill:"

BuiltinHandler = Callable[[TaskRow], Awaitable[dict]]


async def _ping_handler(task: TaskRow) -> dict:
    return {"response": "pong"}


async def _status_handler(task: TaskRow) -> dict:
    pending = await _to_thread(database.count_pending)
    return {"queue_depth": pending}


async def _help_handler(task: TaskRow) -> dict:
    """Inventory what's available. Useful as a self-discovery surface."""
    return {
        "commands": sorted(list(config.REGISTERED_COMMANDS)),
        "skills": SKILL_REGISTRY.list_skills(),
        "agents": AGENT_REGISTRY.list_agents(),
    }


async def _models_handler(task: TaskRow) -> dict:
    """Inventory all registered models + live availability."""
    from core.model_registry import MODEL_REGISTRY
    MODEL_REGISTRY.check_availability()
    return {
        "models": [m.model_dump() for m in MODEL_REGISTRY.list_models()],
    }


async def _complexity_handler(task: TaskRow) -> dict:
    """Run the heuristic complexity classifier on the supplied input."""
    from core.complexity import classify_complexity
    text = task.args.get("text", "")
    result = classify_complexity(
        command="code",  # use /code base scoring as a reasonable default
        args={"text": text},
    )
    return result.model_dump()


async def _default_handler(task: TaskRow) -> dict:
    """Fallback for an unmapped command. Should not normally fire because
    the router rejects unregistered commands; here as defense in depth."""
    return {"response": "executed", "note": "no specific handler"}


async def _to_thread(fn, *args):
    import asyncio
    return await asyncio.to_thread(fn, *args)


BUILTIN_HANDLERS: dict[str, BuiltinHandler] = {
    "ping": _ping_handler,
    "status": _status_handler,
    "help": _help_handler,
    "models": _models_handler,
    "complexity": _complexity_handler,
}


def needs_gpu(command: str) -> bool:
    """True iff executing this command requires the GPU lock.
    Derived from the agent's skill pipeline (or the skill itself for
    skill:-prefixed direct dispatch). No parallel GPU set to maintain."""
    cmd_with_slash = f"/{command.lstrip('/')}"
    target = config.COMMAND_AGENT_MAP.get(cmd_with_slash)
    if not target:
        return False
    if target.startswith(SKILL_PREFIX):
        skill = SKILL_REGISTRY.get(target[len(SKILL_PREFIX):])
        return bool(skill and skill.requires_gpu)
    agent = AGENT_REGISTRY.get(target)
    if not agent:
        return False
    return any(s.requires_gpu for s in agent._skills)


async def _dispatch_skill_directly(
    skill: BaseSkill, task: TaskRow,
) -> dict:
    """Run a single skill without an agent wrapper. Preserves the same
    error envelope shape as Agent.run() for consistency."""
    try:
        input_data = skill.validate_input(task.args)
    except Exception as e:
        log_event(
            task.trace_id, "ERROR", "orchestrator",
            f"input validation failed for skill={skill.name}: {e}",
        )
        return {
            "_error": True,
            "error": str(e),
            "failed_at": skill.name,
            "trace_id": task.trace_id,
        }
    try:
        result = await skill.execute(input_data, task.trace_id)
    except SkillError as e:
        log_event(
            task.trace_id, "ERROR", "orchestrator",
            f"skill {skill.name} raised SkillError: {e}",
        )
        return {
            "_error": True, "error": str(e),
            "failed_at": skill.name, "trace_id": task.trace_id,
        }
    if not isinstance(result, BaseModel):
        err = (f"skill {skill.name} returned {type(result).__name__}, "
               f"expected BaseModel")
        log_event(task.trace_id, "ERROR", "orchestrator", err)
        return {
            "_error": True, "error": err,
            "failed_at": skill.name, "trace_id": task.trace_id,
        }
    return result.model_dump()


async def dispatch(task: TaskRow) -> dict:
    """Resolve task.command and run it. Returns the dict to be stored
    as task.result. Raises on truly exceptional conditions (missing
    agent registration); skill-level failures are surfaced inside the
    returned dict via the agent's error envelope."""
    cmd_with_slash = f"/{task.command.lstrip('/')}"

    if cmd_with_slash not in config.COMMAND_AGENT_MAP:
        log_event(
            task.trace_id, "WARNING", "orchestrator",
            f"unmapped command (router accepted but no executor): "
            f"{cmd_with_slash}",
        )
        return await _default_handler(task)

    target = config.COMMAND_AGENT_MAP[cmd_with_slash]

    if target is None:
        handler = BUILTIN_HANDLERS.get(task.command, _default_handler)
        log_event(
            task.trace_id, "INFO", "orchestrator",
            f"dispatching builtin command={task.command}",
        )
        return await handler(task)

    if target.startswith(SKILL_PREFIX):
        skill_name = target[len(SKILL_PREFIX):]
        skill = SKILL_REGISTRY.get(skill_name)
        if skill is None:
            msg = (f"command {cmd_with_slash} maps to skill "
                   f"'{skill_name}' but no such skill is registered")
            log_event(task.trace_id, "ERROR", "orchestrator", msg)
            raise RuntimeError(msg)
        log_event(
            task.trace_id, "INFO", "orchestrator",
            f"dispatching directly to skill='{skill_name}' "
            f"command={task.command}",
        )
        return await _dispatch_skill_directly(skill, task)

    agent = AGENT_REGISTRY.get(target)
    if agent is None:
        msg = (f"command {cmd_with_slash} maps to agent '{target}' "
               f"but no such agent is registered")
        log_event(task.trace_id, "ERROR", "orchestrator", msg)
        raise RuntimeError(msg)

    log_event(
        task.trace_id, "INFO", "orchestrator",
        f"dispatching to agent='{target}' command={task.command}",
    )
    return await agent.run(task.args, task.trace_id)
