"""Gwen open-system agent.

Direct user -> Qwen path that bypasses KB lookup, Claude pre-teach,
graduation, and the PROJECT_ROOT sandbox. Use to drive Sentinel's
host as a general-purpose shell + filesystem tool.

Two execution modes (chosen by `_is_literal_recipe`):

  1. LITERAL RECIPE (fast path) -- input that starts with `STEP N:`
     bypasses the LLM entirely. core.qwen_agent's recipe parser
     dispatches each step server-side against OPEN_TOOL_DISPATCH.
     Zero VRAM cost, fully deterministic. Intended for recipes
     pasted from an external AI (the /prompt brief teaches
     ChatGPT/Claude/Gemini to emit `/gwen STEP N: ...` lines).

  2. ENGLISH (LLM path) -- intent in plain English. Qwen writes a
     STEP-N recipe via `_qwen_generate(format_json=False)`, then
     the executor runs each step. Adds 5-15s for recipe authoring
     plus per-step dispatch.

Tools are EXPLICITLY UNSANDBOXED:
  - Paths accept `~`, absolute, and relative (relative -> PROJECT_ROOT
    for back-compat).
  - `run_bash` runs with cwd=None (process cwd, typically
    PROJECT_ROOT for the bot). Caller can `cd /target && ...` per
    command.
  - There is no sandbox check. Paths can escape PROJECT_ROOT freely.

Authorization is via Telegram TELEGRAM_AUTHORIZED_USERS in the bot
handler. Never expose this module's tool dispatch outside that
authorization boundary.
"""
from __future__ import annotations

import json
import re as _re
import subprocess as _subprocess
from pathlib import Path
from typing import Any

from core import config
from core.logger import log_event
from core.qwen_agent import (
    _parse_recipe_steps,
    _parse_step_text_to_tool_call,
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

PROJECT_ROOT = config.PROJECT_ROOT
BASH_TIMEOUT = 60
MAX_TOOL_OUTPUT = 4000


# ---------------------------------------------------------------------
# Path resolution (NO sandbox)
# ---------------------------------------------------------------------

def _open_resolve(path: str) -> Path:
    """Resolve a path with home expansion and no sandbox check.

    - Backslashes -> forward slashes (Windows recipe-string safety;
      same normalization as Phase 17e in core.qwen_agent).
    - `~` and `~user` expanded via Path.expanduser().
    - Absolute paths returned as-is (after expansion).
    - Relative paths interpreted against PROJECT_ROOT (back-compat
      with /qcode-style recipes).

    The returned path is .resolve()'d so symlinks and `..` components
    collapse, but membership in PROJECT_ROOT is NOT enforced.
    """
    path = path.replace("\\", "/")
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


# ---------------------------------------------------------------------
# Open tools (no sandbox)
# ---------------------------------------------------------------------

def open_read_file(path: str) -> dict:
    try:
        target = _open_resolve(path)
        if not target.exists():
            return {"error": f"file not found: {path}"}
        if not target.is_file():
            return {"error": f"not a file: {path}"}
        text = target.read_text(encoding="utf-8", errors="replace")
        return {
            "ok": True,
            "content": text[:MAX_TOOL_OUTPUT],
            "truncated": len(text) > MAX_TOOL_OUTPUT,
            "total_chars": len(text),
            "resolved": str(target),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def open_write_file(path: str, content: str) -> dict:
    try:
        target = _open_resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "bytes_written": len(content),
            "resolved": str(target),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def open_edit_file(path: str, old: str, new: str) -> dict:
    try:
        target = _open_resolve(path)
        if not target.exists():
            return {"error": f"file not found: {path}"}
        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return {"error": "`old` string not found in file"}
        if count > 1:
            return {
                "error": f"`old` string appears {count} times; "
                f"include more context to make it unique"
            }
        new_text = text.replace(old, new, 1)
        target.write_text(new_text, encoding="utf-8")
        return {
            "ok": True,
            "lines_changed": new_text.count("\n") - text.count("\n"),
            "resolved": str(target),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def open_list_dir(path: str = ".") -> dict:
    try:
        target = _open_resolve(path)
        if not target.is_dir():
            return {"error": f"not a directory: {path}"}
        items = []
        for p in sorted(target.iterdir()):
            kind = "dir" if p.is_dir() else "file"
            items.append(f"{kind}: {p.name}")
        return {
            "ok": True,
            "items": items[:200],
            "truncated": len(items) > 200,
            "resolved": str(target),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def open_run_bash(
    command: str | None = None,
    cmd: str | None = None,
    cwd: str | None = None,
    timeout: int = BASH_TIMEOUT,
) -> dict:
    """Run a shell command. cwd defaults to current process directory
    (no PROJECT_ROOT lock). Accepts both `command` and `cmd` aliases
    -- recipes from external AIs sometimes use either.

    Use `cwd=` to set working directory; or just `cd <dir> && ...`
    inside the command (works because shell=True spawns a real shell).
    """
    command = command or cmd
    if not command:
        return {"error": "missing required arg: command (or cmd)"}
    try:
        run_cwd = _open_resolve(cwd) if cwd else None
        if run_cwd is not None and not run_cwd.is_dir():
            return {"error": f"cwd not a directory: {cwd}"}
        result = _subprocess.run(
            command, shell=True,
            cwd=str(run_cwd) if run_cwd else None,
            capture_output=True, text=True,
            timeout=min(timeout, BASH_TIMEOUT),
        )
        return {
            "ok": True,
            "stdout": (result.stdout or "")[:MAX_TOOL_OUTPUT],
            "stderr": (result.stderr or "")[:MAX_TOOL_OUTPUT],
            "return_code": result.returncode,
        }
    except _subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


OPEN_TOOL_DISPATCH = {
    "read_file": open_read_file,
    "write_file": open_write_file,
    "edit_file": open_edit_file,
    "list_dir": open_list_dir,
    "run_bash": open_run_bash,
}


# ---------------------------------------------------------------------
# System prompt (used only when Qwen translates English -> recipe)
# ---------------------------------------------------------------------

GWEN_SYSTEM = (
    "You are Gwen, an autonomous shell + filesystem agent with FULL "
    "system access on the user's machine. You can read/write/edit "
    "any file (including `~/Desktop/...` and absolute paths like "
    "`C:/Users/<you>/...`) and run any shell command.\n"
    "\n"
    "Your output MUST be a recipe in this exact format. No prose, no "
    "preamble, no markdown fences -- just numbered STEP lines:\n"
    "\n"
    "STEP 1: <tool> <key>=\"<value>\" ...\n"
    "STEP 2: <tool> ...\n"
    "STEP N: done summary=\"<one-line summary of what you did>\"\n"
    "\n"
    "Tools (exact arg names required):\n"
    "  read_file(path)\n"
    "  list_dir(path)\n"
    "  write_file(path, content)\n"
    "  edit_file(path, old, new)\n"
    "  run_bash(command)            -- cwd defaults to project root\n"
    "  run_bash(command, cwd)       -- override cwd\n"
    "  done(summary)                -- ALWAYS the final step\n"
    "\n"
    "Rules:\n"
    "- First characters of your output MUST be `STEP 1:`.\n"
    "- Forward slashes in paths only. NEVER backslashes.\n"
    "- For `edit_file`, `old` must be byte-for-byte present in the "
    "file. If you have not read the file in this session, prefer "
    "`write_file` (overwrite) or `run_bash` instead.\n"
    "- Target 2-6 STEPs. If you need more, you are over-scoping; "
    "stop at a checkpoint and let the user run a follow-up.\n"
    "- The final step is ALWAYS `done summary=\"...\"`.\n"
)


# ---------------------------------------------------------------------
# Recipe detection
# ---------------------------------------------------------------------

LITERAL_RECIPE_RE = _re.compile(
    r"^\s*(?:/gwen\s+)?STEP\s+\d+\s*:", _re.IGNORECASE,
)


def _is_literal_recipe(text: str) -> bool:
    """True if `text` is a STEP-N recipe (or starts with one).

    Accepts an optional `/gwen ` prefix on the first line so users can
    paste recipes the external AI emitted with the `/gwen` prefix
    intact. The bot's CommandHandler already strips one `/gwen` token,
    but pasted multi-line recipes may have additional `/gwen ` prefixes
    on later STEP lines -- those are stripped at parse time below.
    """
    if not text:
        return False
    return bool(LITERAL_RECIPE_RE.match(text))


def _strip_gwen_prefixes(text: str) -> str:
    """Remove `/gwen ` (case-insensitive) prefixes from line starts so
    multi-line pastes from external AI parse cleanly.

    Phase 18d-gz polish (live trigger 2026-05-07 16:21Z, Gemini-emitted
    recipe): also strip INLINE `/gwen ` patterns when they appear
    immediately before a `STEP N:` marker. Defense-in-depth for the
    case where an AI prefixes EVERY step (against brief rule) AND the
    user's Telegram client collapses newlines into spaces -- mid-line
    `/gwen STEP N:` would otherwise smush into the previous step's
    arg value, breaking everything downstream."""
    # First pass: strip inline `/gwen ` immediately before STEP markers.
    text = _re.sub(
        r'/gwen\s+(?=STEP\s+\d+\s*:)',
        '',
        text,
        flags=_re.IGNORECASE,
    )
    # Second pass: per-line prefix strip (the original behavior).
    out_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.lower().startswith("/gwen "):
            indent_len = len(line) - len(stripped)
            out_lines.append(line[:indent_len] + stripped[len("/gwen "):])
        elif stripped.lower() == "/gwen":
            continue
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------
# Stepfed-open executor (no sandbox, no Qwen-fallback transcription)
# ---------------------------------------------------------------------

def _execute_recipe(recipe: str, trace_id: str) -> dict:
    """Parse `recipe` into STEP blocks and dispatch each against
    OPEN_TOOL_DISPATCH. No LLM in this path -- pure server-side
    dispatch.

    Returns:
      {
        "summary": str,         # done(summary=...) or autogen
        "session": list[dict],  # per-step trace
        "steps": int,           # how many STEPs ran
        "completed_via_done": bool,
        "error": str | None,    # parser-level error or None
      }
    """
    recipe = _strip_gwen_prefixes(recipe)
    steps = _parse_recipe_steps(recipe)
    if not steps:
        return {
            "summary": "(no STEP-N blocks parsed from input)",
            "session": [],
            "steps": 0,
            "completed_via_done": False,
            "error": "recipe parser returned 0 steps",
        }

    log_event(trace_id, "INFO", "gwen_agent",
              f"executing {len(steps)} step(s) "
              f"(literal recipe, no LLM)")

    session: list[dict] = []
    final_summary = ""
    completed = False

    aborted_unparseable = False
    for i, step_text in enumerate(steps, start=1):
        log_event(trace_id, "INFO", "gwen_agent",
                  f"step {i}/{len(steps)}: {step_text[:100]!r}")
        call = _parse_step_text_to_tool_call(step_text)
        if call is None:
            session.append({
                "step": i, "tool": "?", "args": {},
                "result": {"error": "step text did not parse to a "
                           "known tool call -- recipe ABORTED here"},
            })
            log_event(trace_id, "ERROR", "gwen_agent",
                      f"step {i} unparseable, ABORTING recipe: "
                      f"{step_text[:160]!r}")
            aborted_unparseable = True
            break

        fn = call.get("function", {}) or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args or {}

        if name == "done":
            final_summary = args.get("summary", "")
            completed = True
            session.append({
                "step": i, "tool": "done", "args": args,
                "result": {"ok": True},
            })
            log_event(trace_id, "INFO", "gwen_agent",
                      f"step {i} done summary={final_summary[:120]!r}")
            break

        handler = OPEN_TOOL_DISPATCH.get(name)
        if handler is None:
            result: dict[str, Any] = {"error": f"unknown tool: {name}"}
        else:
            try:
                result = handler(**args)
            except TypeError as e:
                result = {"error": f"bad args: {e}"}
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}

        session.append({
            "step": i, "tool": name, "args": args, "result": result,
        })
        if "error" in result:
            log_event(trace_id, "ERROR", "gwen_agent",
                      f"step {i} tool={name} error, ABORTING recipe: "
                      f"{result['error'][:160]}")
            aborted_unparseable = True  # reuse abort signal
            break

    if aborted_unparseable:
        bad = session[-1] if session else {}
        final_summary = (
            f"❌ ABORTED at step {bad.get('step', '?')}: "
            f"unparseable step text (no known tool match). "
            f"Recipe halted -- subsequent steps NOT executed."
        )
    elif not final_summary:
        final_summary = (
            f"Ran {len(session)} step(s); "
            f"completed_via_done={completed}"
        )

    return {
        "summary": final_summary,
        "session": session,
        "steps": len(session),
        "completed_via_done": completed,
        "aborted_unparseable": aborted_unparseable,
        "error": None,
    }


# ---------------------------------------------------------------------
# English -> recipe (uses Qwen text-gen, then dispatches)
# ---------------------------------------------------------------------

def _english_to_recipe(text: str, trace_id: str, model: str) -> str:
    """Ask Qwen to translate plain English to a STEP-N recipe.

    Imports `_qwen_generate` lazily so this module does not pull
    skills.code_assist at import time (avoids circular imports if
    code_assist ever imports core.gwen_agent in the future).
    """
    from skills.code_assist import _qwen_generate
    user_prompt = f"User request: {text}\n\nRecipe:"
    return _qwen_generate(
        GWEN_SYSTEM, user_prompt, trace_id, model,
        timeout=120, format_json=False,
    )


# ---------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------

def run_gwen_open(text: str, trace_id: str, model: str) -> dict:
    """Top-level entry. Sync; caller wraps in asyncio.to_thread.

    Routing:
      - input matches /^STEP N:/ (with optional `/gwen ` prefix) ->
        literal-recipe fast path (no LLM call)
      - otherwise -> Qwen translates English to recipe, then executes

    Returns the same shape as `_execute_recipe` plus a `mode` field:
    "literal" or "english".
    """
    if _is_literal_recipe(text):
        log_event(trace_id, "INFO", "gwen_agent",
                  "fast path: literal recipe detected, "
                  "no LLM call")
        out = _execute_recipe(text, trace_id)
        out["mode"] = "literal"
        return out

    log_event(trace_id, "INFO", "gwen_agent",
              f"english path: asking Qwen for recipe "
              f"(input chars={len(text)})")
    try:
        recipe = _english_to_recipe(text, trace_id, model)
    except Exception as e:
        log_event(trace_id, "ERROR", "gwen_agent",
                  f"recipe generation failed: {type(e).__name__}: {e}")
        return {
            "summary": f"(Qwen recipe generation failed: {e})",
            "session": [],
            "steps": 0,
            "completed_via_done": False,
            "error": f"{type(e).__name__}: {e}",
            "mode": "english",
        }

    log_event(trace_id, "INFO", "gwen_agent",
              f"qwen recipe ({len(recipe)} chars): "
              f"{recipe[:120]!r}")
    out = _execute_recipe(recipe, trace_id)
    out["mode"] = "english"
    out["recipe"] = recipe  # surface so callers can render it
    return out
