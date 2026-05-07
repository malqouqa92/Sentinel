"""Qwen tool-calling agent for /code.

Architecture:
  - Claude pre-teaches: produces a step-by-step recipe before Qwen starts
  - Qwen executes: tool-calling loop via Ollama /api/chat (Read, Write,
    Edit, ListDir, Bash, Done) sandboxed to PROJECT_ROOT
  - Claude reviews: after Qwen's session, verdicts pass/fail
  - KB stores: the full session as a learnable pattern

Sandbox: every path tool resolves relative to config.PROJECT_ROOT.
Paths that escape the project root are rejected.
"""
from __future__ import annotations

import json
import subprocess as _subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from core import config
from core.logger import log_event


PROJECT_ROOT = config.PROJECT_ROOT
CHAT_TIMEOUT = 300       # one Ollama /api/chat round
BASH_TIMEOUT = 60
MAX_TOOL_OUTPUT = 4000   # truncate stdout/stderr/file contents
MAX_AGENT_STEPS = 30
MAX_CONSECUTIVE_FAILURES = 5  # bail out if Qwen keeps erroring on the same kind of call
STEPFED_PER_STEP_TIMEOUT = 60   # per-step Ollama call cap in stepfed
STEPFED_TOTAL_BUDGET_SEC = 600  # whole-stepfed-loop wall clock cap


# ----------------------------------------------------------------------
# Tool implementations
# ----------------------------------------------------------------------

def _safe_resolve(rel_path: str) -> Path:
    # Phase 17e: normalize path separators BEFORE resolution. The
    # recipe parser sometimes hands us Windows-style backslashes
    # ("interfaces\telegram_bot.py") which Python's string-literal
    # decoder interprets as escape sequences (\t = TAB, \n = newline,
    # etc.). Result was paths like "interfaces<TAB>elegram_bot.py"
    # silently failing every edit_file in the recipe. Normalize ALL
    # backslashes to forward slashes -- POSIX paths always work on
    # Windows + Linux + macOS. Live trigger: 2026-05-07 ~01:15Z chain
    # run, attempt 1 lost 4 of 11 STEPs to this bug.
    rel_path = rel_path.replace("\\", "/")
    target = (PROJECT_ROOT / rel_path).resolve()
    target.relative_to(PROJECT_ROOT.resolve())  # raises ValueError if outside
    return target


def tool_read_file(path: str) -> dict:
    try:
        target = _safe_resolve(path)
        if not target.exists():
            return {"error": f"file not found: {path}"}
        if not target.is_file():
            return {"error": f"not a file: {path}"}
        text = target.read_text(encoding="utf-8")
        return {
            "ok": True,
            "content": text[:MAX_TOOL_OUTPUT],
            "truncated": len(text) > MAX_TOOL_OUTPUT,
            "total_chars": len(text),
        }
    except ValueError:
        return {"error": "path escapes sandbox"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def tool_write_file(path: str, content: str) -> dict:
    try:
        target = _safe_resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "bytes_written": len(content)}
    except ValueError:
        return {"error": "path escapes sandbox"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def tool_edit_file(path: str, old: str, new: str) -> dict:
    try:
        target = _safe_resolve(path)
        if not target.exists():
            return {"error": f"file not found: {path}"}
        text = target.read_text(encoding="utf-8")
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
        return {"ok": True, "lines_changed": new_text.count("\n") - text.count("\n")}
    except ValueError:
        return {"error": "path escapes sandbox"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def tool_list_dir(path: str = ".") -> dict:
    try:
        target = _safe_resolve(path)
        if not target.is_dir():
            return {"error": f"not a directory: {path}"}
        items = []
        for p in sorted(target.iterdir()):
            kind = "dir" if p.is_dir() else "file"
            items.append(
                f"{kind}: {p.relative_to(PROJECT_ROOT).as_posix()}"
            )
        return {"ok": True, "items": items[:200]}
    except ValueError:
        return {"error": "path escapes sandbox"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def tool_run_bash(
    command: str | None = None,
    cmd: str | None = None,
    timeout: int = BASH_TIMEOUT,
) -> dict:
    """Run shell command in PROJECT_ROOT. Output truncated.
    Accepts both `command` (canonical) and `cmd` (alias) since teacher
    Claude has been observed to use both in recipes."""
    command = command or cmd
    if not command:
        return {"error": "missing required arg: command (or cmd)"}
    try:
        result = _subprocess.run(
            command, shell=True, cwd=str(PROJECT_ROOT),
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


TOOL_DISPATCH = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_dir": tool_list_dir,
    "run_bash": tool_run_bash,
}

# OpenAI/Ollama tool-calling schema
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents. Returns up to 4KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to project root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace exactly one occurrence of `old` with `new` in "
                "a file. Fails if `old` is not unique. Prefer this "
                "over write_file for surgical changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file, overwriting existing content. "
                "Use only for new files; use edit_file for changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (default: project root).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a shell command in the project root. Use for "
                "`git status`, `git diff`, `pytest`, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Signal task complete. Provide a summary of what you "
                "did. Stop calling tools after this."
            ),
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]


# ----------------------------------------------------------------------
# Ollama /api/chat client (sync, wrapped by caller in to_thread)
# ----------------------------------------------------------------------

def _ollama_chat(model: str, messages: list[dict],
                 tools: list[dict] | None = None,
                 timeout: int = CHAT_TIMEOUT) -> dict:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": config.MODEL_KEEP_ALIVE,
        "options": {"temperature": 0.1},
    }
    if tools:
        body["tools"] = tools
    data = json.dumps(body).encode("utf-8")
    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ----------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------

AGENT_SYSTEM = (
    "You are a coding agent operating on a Python project. The project "
    "root is your sandbox -- you can read, edit, write files within "
    "it, and run shell commands. Your job: apply the user's requested "
    "change to the codebase. A senior engineer has provided a recipe; "
    "follow it. Use the tools to read existing code, make minimal "
    "edits, run `git diff` to verify, and call `done` with a summary "
    "when finished. Be concise. Don't over-explore. Don't ask questions.\n"
    "\n"
    "TOOL-CALL FORMAT: Every turn, output ONLY a single JSON object "
    "(in a ```json``` fence is fine) matching this shape:\n"
    "  {\"name\": \"<tool_name>\", \"arguments\": { ... }}\n"
    "Where <tool_name> is one of: read_file, edit_file, write_file, "
    "list_dir, run_bash, done. No prose outside the JSON. One tool "
    "call per turn."
)


def _parse_tool_calls_from_content(content: str) -> list[dict]:
    """Some models (notably qwen2.5-coder) don't populate the native
    tool_calls field -- they emit tool invocations as JSON inside the
    content. Detect and parse them, returning OpenAI-shaped tool_calls.

    Uses json.JSONDecoder.raw_decode for proper brace-balanced parsing
    -- regex-based extraction breaks on nested objects (which our tool
    args ALWAYS have), so this scans the content top-level."""
    if not content:
        return []
    # Strip markdown fences first to expose the JSON payload, but DON'T
    # try to parse fence-by-fence -- a single fence often contains many
    # JSON objects separated by newlines.
    text = content
    import re as _re
    # Remove ```json and ``` markers but keep the content
    text = _re.sub(r"```(?:json)?\s*", "", text)
    text = _re.sub(r"\s*```", "", text)

    decoder = json.JSONDecoder()
    objects: list[dict] = []
    i = 0
    n = len(text)
    while i < n:
        # Fast skip to next opening brace
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        i = end

    out: list[dict] = []
    for obj in objects:
        if "name" in obj:
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            out.append({"function": {"name": obj["name"], "arguments": args}})
        elif "function" in obj and isinstance(obj["function"], dict):
            out.append(obj)
    return out


def _project_map() -> str:
    """Auto-generate a tree-style overview of key project files so Qwen
    knows what exists before it starts editing. Reduces blind-edit
    failures. Capped at ~80 lines to keep the prompt lean."""
    lines: list[str] = []
    skip_dirs = {"__pycache__", ".git", "logs", "workspace", ".pytest_cache"}
    for top in sorted(PROJECT_ROOT.iterdir()):
        if top.name in skip_dirs:
            continue
        rel = top.relative_to(PROJECT_ROOT).as_posix()
        if top.is_dir():
            lines.append(f"{rel}/")
            for child in sorted(top.iterdir()):
                if child.name in skip_dirs or child.name.endswith(".pyc"):
                    continue
                if child.is_dir():
                    lines.append(f"  {child.relative_to(PROJECT_ROOT).as_posix()}/")
                else:
                    lines.append(f"  {child.relative_to(PROJECT_ROOT).as_posix()}")
        else:
            lines.append(rel)
        if len(lines) > 80:
            lines.append("(...truncated)")
            break
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Step-feeder: parse Claude's recipe into atomic steps and feed Qwen
# ONE step at a time. Each step is a tiny prompt -- Qwen's job becomes
# transcription (recipe text -> JSON tool call), not synthesis. Avoids
# the "long recipe overwhelms 3B and drops to prose" failure mode.
# ----------------------------------------------------------------------

import re as _re

STEPFED_SYSTEM = (
    "You are a TOOL-CALL EMITTER. Read the instruction. Output ONE "
    "JSON object on a single line, NOTHING ELSE -- no markdown, no "
    "fences, no commentary, no explanation, no prose.\n"
    "\n"
    "Format: {\"name\": \"<tool_name>\", \"arguments\": {...}}\n"
    "\n"
    "Valid tool names: read_file, list_dir, write_file, edit_file, "
    "run_bash, done.\n"
    "\n"
    "Examples:\n"
    "Instruction: read_file path=\"core/foo.py\"\n"
    "Output: {\"name\": \"read_file\", \"arguments\": {\"path\": \"core/foo.py\"}}\n"
    "\n"
    "Instruction: write_file path=\"x.py\" content=\"def f():\\n    pass\\n\"\n"
    "Output: {\"name\": \"write_file\", \"arguments\": {\"path\": \"x.py\", "
    "\"content\": \"def f():\\n    pass\\n\"}}\n"
    "\n"
    "Instruction: done summary=\"created x.py\"\n"
    "Output: {\"name\": \"done\", \"arguments\": {\"summary\": \"created x.py\"}}\n"
)


def _parse_recipe_steps(recipe: str) -> list[str]:
    """Split a recipe into per-step instruction strings.

    Matches `STEP N: <body>` blocks. Body extends until the next STEP
    or end of input. Notes/prose between STEP lines are dropped.
    Returns a list of body strings (no "STEP N:" prefix).

    Phase 18b: two-tier parse. Strict pattern requires newline before
    subsequent STEPs (avoids false positives from "STEP 2:" inside a
    value or prose). If strict produces <=1 step AND the input clearly
    has multiple STEP markers, retry with a relaxed boundary that
    accepts any whitespace run before STEPs -- recovers single-line
    pastes (some Telegram clients collapse newlines on copy).
    """
    if not recipe:
        return []
    strict = _re.compile(
        r"STEP\s+\d+\s*:\s*(.+?)(?=\n\s*STEP\s+\d+\s*:|\Z)",
        _re.DOTALL | _re.IGNORECASE,
    )
    matches = [m.strip() for m in strict.findall(recipe) if m.strip()]
    if len(matches) <= 1:
        marker_count = len(
            _re.findall(r"\bSTEP\s+\d+\s*:", recipe, _re.IGNORECASE)
        )
        if marker_count >= 2:
            relaxed = _re.compile(
                r"STEP\s+\d+\s*:\s*(.+?)(?=\s+STEP\s+\d+\s*:|\Z)",
                _re.DOTALL | _re.IGNORECASE,
            )
            relaxed_matches = [
                m.strip() for m in relaxed.findall(recipe) if m.strip()
            ]
            if len(relaxed_matches) > len(matches):
                return relaxed_matches
    return matches


_TOOL_NAME_RE = _re.compile(
    r"^\s*(?:STEP\s*\d+\s*:\s*)?"          # optional STEP N: leftover
    r"([a-z_][a-z_0-9]*)"                  # tool name
    r"\s*\(?",                             # optional opening paren
    _re.IGNORECASE,
)
# Phase 18b: accept ASCII straight-quotes AND Unicode curly variants
# (some Telegram clients auto-convert " to U+201C/U+201D on paste).
#
# Phase 18c (2026-05-07): REMOVED single-quote acceptance. Python
# code inside content="..." values commonly uses single-quoted
# strings (e.g. `encoding='utf-8'`, `pathlib.Path('C:/...')`) which
# the regex was matching as bogus kv pairs and passing as kwargs to
# the tool dispatch. Live failure 10:17Z: the dashboard recipe's
# write_file got `encoding='utf-8'` as a stray kwarg from inside its
# own content. Recipes only use double-quotes per the brief; keeping
# only double + curly-double is the correct scope.
_KV_RE = _re.compile(
    r'(\w+)\s*=\s*'
    r'(?:'
    r'"((?:\\.|[^"\\])*)"'      # straight double quotes
    r'|“((?:\\.|[^”\\])*)”'  # curly double quotes
    r')',
    _re.DOTALL,
)
_VALID_TOOLS = {"read_file", "list_dir", "write_file",
                "edit_file", "run_bash", "done"}


# Phase 18c (2026-05-07): tools that accept a single string arg can
# be parsed permissively when quotes have been stripped (Telegram
# auto-format ate them). Multi-arg tools cannot use permissive parse
# because `\w+=` patterns inside values would be misparsed as kwargs.
_SINGLE_ARG_TOOLS = {
    "run_bash": "command",
    "read_file": "path",
    "list_dir": "path",
    "done": "summary",
}


def _parse_step_text_to_tool_call(step_text: str) -> dict | None:
    """Parse a recipe step into an OpenAI-shaped tool call WITHOUT
    asking Qwen. The recipe IS the tool call -- Qwen as a transcription
    layer just adds JSON-escape bugs, so we bypass it.

    Handles:
      tool_name key="value" key2="value with \\\"escape\\\""
      tool_name(key="value", key2="value")
      done(summary="text")
      done summary="text"
      tool_name key=unquoted_value           (Phase 18c, single-arg tools only)
    """
    if not step_text:
        return None
    name_match = _TOOL_NAME_RE.match(step_text)
    if not name_match:
        return None
    tool_name = name_match.group(1).lower()
    if tool_name not in _VALID_TOOLS:
        return None
    args: dict[str, Any] = {}
    for m in _KV_RE.finditer(step_text):
        key = m.group(1)
        # Phase 18b: regex has 2 alt groups (straight-double,
        # curly-double); exactly one is non-None per match.
        raw_value = (
            m.group(2) if m.group(2) is not None else m.group(3)
        )
        # JSON-decode the value so \n becomes newline, \" becomes ",
        # \\ becomes backslash. The recipe uses JSON-style escapes.
        try:
            decoded = json.loads(f'"{raw_value}"')
        except json.JSONDecodeError:
            # Fallback path: fires when raw_value contains characters
            # invalid in JSON strings -- almost always REAL newlines
            # introduced by paste-time soft-wrap (Telegram clients
            # wrap long lines, the wraps land inside content="..." or
            # command="..." and cmd.exe / Python see broken input).
            #
            # Phase 18 fix: collapse `\s*\n\s*` runs into a single
            # space BEFORE doing escape replacement. Intended newlines
            # come in as the escape sequence `\n` (per PROMPT_BRIEF.md);
            # those survive the collapse (still backslash-n in raw_value)
            # and decode correctly in the next replace() call. Real
            # newlines that snuck in via wrap are paste artifacts and
            # get normalized to a space.
            collapsed = _re.sub(r"\s*\n\s*", " ", raw_value)
            decoded = (
                collapsed
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        args[key] = decoded
    # Phase 18d (2026-05-07): _b64 arg suffix for paste-mangle immunity.
    # Any arg whose key ends in `_b64` is decoded from base64 and the
    # decoded value replaces the plain key in the args dict. Base64
    # alphabet ([A-Za-z0-9+/=]) cannot be mangled by Telegram smart-quote
    # conversion or paste-wrap (b64decode ignores whitespace). This is
    # the robust solution for long content that real-world Telegram
    # paste corruption was breaking. Recipe author base64-encodes:
    #     STEP 1: write_file path="x.py" content_b64="<base64>"
    # Parser decodes -> args = {path: "x.py", content: "<plain text>"}.
    # Phase 18d-gz: `_b64gz` suffix decodes via base64 THEN zlib-decompress.
    # For source files >2KB the recipe-encoded form is roughly 3x smaller,
    # which matters because Telegram's per-message limit is 4096 chars.
    # Process _b64gz BEFORE _b64 so the suffix-stripping logic doesn't
    # collide.
    b64gz_keys = [k for k in args if k.endswith("_b64gz")]
    if b64gz_keys:
        import base64 as _b64m
        import zlib as _zlib
        for k in b64gz_keys:
            plain_key = k[:-6]  # strip "_b64gz"
            try:
                raw = _b64m.b64decode(args[k], validate=False)
            except Exception as e:
                # Substitute a clear error so tool dispatch surfaces a
                # useful message instead of silently dropping the arg.
                args[plain_key] = (
                    f"[ERROR: base64-decode of `{k}` value failed "
                    f"({type(e).__name__}: {e}). Recipe author "
                    f"likely hallucinated the base64 string -- LLMs "
                    f"cannot reliably hand-compute base64.]"
                )
                args.pop(k)
                continue
            try:
                decompressed = _zlib.decompress(raw)
                args[plain_key] = decompressed.decode("utf-8", errors="replace")
                args.pop(k)
            except Exception as e:
                # b64 OK, zlib FAIL -- bytes are corrupt. Most common
                # cause: AI hallucinated the b64gz value (the bytes
                # look like b64 but the gzip stream inside is garbage).
                # Surface a CLEAR error rather than silently dropping
                # the content arg (the prior failure mode confused
                # the user during the 2026-05-07 16:34Z Gemini test).
                args[plain_key] = (
                    f"[ERROR: `{k}` value base64-decoded to {len(raw)} "
                    f"bytes but zlib-decompress failed ({e}). The "
                    f"recipe author (typically an AI) emitted base64 "
                    f"that looks valid but does not decompress. LLMs "
                    f"cannot reliably hand-compute base64+zlib; ask "
                    f"the AI to use plain `content=` with \\n escapes "
                    f"instead, or split source across multiple "
                    f"smaller write_file steps.]"
                )
                args.pop(k)
    b64_keys = [k for k in args if k.endswith("_b64")]
    if b64_keys:
        import base64 as _b64m
        for k in b64_keys:
            plain_key = k[:-4]  # strip "_b64"
            try:
                decoded_bytes = _b64m.b64decode(args[k], validate=False)
                args[plain_key] = decoded_bytes.decode("utf-8", errors="replace")
                args.pop(k)
            except Exception:
                # Leave the _b64 key as-is on decode failure; tool will
                # reject it as an unknown kwarg with a clear error.
                pass
    # Phase 18d-multi: when the strict (quoted) regex finds 0 args AND
    # the unquoted step text contains at least one `_b64` marker, do
    # a permissive multi-arg parse. The `_b64` marker is the safety
    # signal -- it tells us the recipe author intended a paste-mangle-
    # immune form, so we can confidently split on `\w+=` boundaries.
    if not args and "_b64" in step_text:
        import base64 as _b64m
        # Find ALL `\w+_b64=` and `\w+_b64gz=` marker positions FIRST.
        # Each b64 value extends from after its `=` to either the next
        # marker or end-of-step. This avoids the bug where base64
        # padding `=` at end of value triggers the `\w+=` boundary
        # lookahead falsely.
        b64_marker_re = _re.compile(r'\b(\w+?)(_b64gz|_b64)\b\s*=\s*')
        b64_markers = list(b64_marker_re.finditer(step_text))
        for i, m in enumerate(b64_markers):
            key = m.group(1)
            suffix = m.group(2)  # "_b64" or "_b64gz"
            start = m.end()
            end = (b64_markers[i + 1].start()
                   if i + 1 < len(b64_markers) else len(step_text))
            raw_b64 = step_text[start:end]
            # Strip whitespace (paste-wrap), stray quote fragments
            cleaned = _re.sub(r'\s+', '', raw_b64)
            cleaned = cleaned.strip('"\'').strip("“”‘’")
            try:
                decoded = _b64m.b64decode(cleaned, validate=False)
            except Exception as e:
                # Substitute a clear error so tool dispatch surfaces
                # something useful instead of silently dropping the arg.
                args[key] = (
                    f"[ERROR: base64-decode of `{key}{suffix}` value "
                    f"failed ({type(e).__name__}: {e}). LLMs cannot "
                    f"reliably hand-compute base64; ask the AI to use "
                    f"plain `content=\"...\"` with \\n escapes instead.]"
                )
                continue
            if suffix == "_b64gz":
                import zlib as _zlib
                try:
                    decoded = _zlib.decompress(decoded)
                except Exception as ze:
                    # b64 OK, zlib FAIL -- AI hallucinated the gz stream.
                    # Surface the diagnostic instead of silent-drop (live
                    # trigger 2026-05-07 17:23Z, trace SEN-fc90dfff).
                    args[key] = (
                        f"[ERROR: `{key}{suffix}` base64-decoded to "
                        f"{len(decoded)} bytes but zlib-decompress "
                        f"failed ({ze}). Recipe author (typically an AI) "
                        f"emitted base64 that looks valid but does not "
                        f"decompress to valid gzip. LLMs cannot reliably "
                        f"hand-compute base64+zlib; ask the AI to use "
                        f"plain `content=\"...\"` with \\n escapes "
                        f"instead, or split source across smaller "
                        f"write_file steps.]"
                    )
                    continue
            try:
                args[key] = decoded.decode("utf-8", errors="replace")
            except Exception as ue:
                args[key] = f"[ERROR: utf-8 decode failed for `{key}`: {ue}]"
        # Now find plain `\w+=value` markers in the prefix BEFORE the
        # first b64 marker (b64 markers consume the rest of the step).
        plain_region = (
            step_text[:b64_markers[0].start()] if b64_markers else step_text
        )
        plain_re = _re.compile(
            r'\b(\w+)\s*=\s*(.+?)(?=\s+\w+\s*=|\Z)',
            _re.DOTALL,
        )
        for m in plain_re.finditer(plain_region):
            key = m.group(1)
            value = m.group(2).strip()
            if key == tool_name or key in args:
                continue
            v = value.strip('"\'').strip("“”‘’")
            if v:
                args[key] = v
    # Phase 18c permissive fallback: when the strict (quoted) regex
    # finds 0 args AND this is a single-arg tool, take everything
    # after `<arg_name>=` to end-of-step as the value. Recovers the
    # unquoted-value case caused by Telegram client auto-formatting
    # that strips quotes from messages (live trigger: 2026-05-07
    # 10:17Z dashboard recipe).
    #
    # Phase 18d: also try `<arg_name>_b64=` and decode -- so that
    # users who paste a recipe like `run_bash command_b64=<base64>`
    # (no quotes) still get full recovery.
    if not args and tool_name in _SINGLE_ARG_TOOLS:
        arg_name = _SINGLE_ARG_TOOLS[tool_name]
        # 1) try plain `arg=value` permissive
        permissive = _re.search(
            rf'\b{arg_name}\s*=\s*(.+?)\s*$',
            step_text,
            _re.DOTALL,
        )
        # 2) try `arg_b64=base64` permissive (and decode)
        b64_perm = _re.search(
            rf'\b{arg_name}_b64\s*=\s*(\S+)',
            step_text,
        )
        if b64_perm:
            import base64 as _b64m
            try:
                decoded_bytes = _b64m.b64decode(
                    b64_perm.group(1).strip('"').strip("'").strip("“”‘’"),
                    validate=False,
                )
                args[arg_name] = decoded_bytes.decode("utf-8", errors="replace")
            except Exception:
                pass
        elif permissive:
            value = permissive.group(1).strip()
            # Strip stray surrounding quote-fragments left over from
            # half-eaten quotes (e.g. opening " survived but closing
            # didn't, or vice-versa).
            value = value.strip('"').strip("'").strip("“”‘’")
            if value:
                args[arg_name] = value
    if not args and tool_name != "done":
        # Suspect parse: tool name found but no args. Probably the
        # step uses a shape we didn't anticipate -- let Qwen handle.
        return None
    return {
        "function": {"name": tool_name, "arguments": args},
        "_parsed_locally": True,
    }


def _stepfed_one_call(
    step_text: str, model: str, trace_id: str, step_num: int,
) -> dict | None:
    """Convert ONE step instruction into ONE tool-call dict.

    Tries server-side parse FIRST (deterministic, no JSON-escape bugs).
    Falls back to Qwen ONLY if server parser can't recognize the step
    shape. Returns OpenAI-shaped {function: {name, arguments}}, or
    None on total failure."""
    parsed = _parse_step_text_to_tool_call(step_text)
    if parsed is not None:
        log_event(trace_id, "INFO", "qwen_agent",
                  f"stepfed step {step_num} parsed locally "
                  f"tool={parsed['function']['name']} "
                  f"args_keys={list(parsed['function']['arguments'].keys())}")
        return parsed
    log_event(trace_id, "INFO", "qwen_agent",
              f"stepfed step {step_num} server-parse failed; "
              f"falling back to Qwen for transcription")
    user_prompt = f"Instruction: {step_text[:2000]}\nOutput:"
    messages = [
        {"role": "system", "content": STEPFED_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = _ollama_chat(model, messages, tools=None,
                                timeout=STEPFED_PER_STEP_TIMEOUT)
    except (urllib.error.URLError, json.JSONDecodeError,
            TimeoutError) as e:
        log_event(trace_id, "ERROR", "qwen_agent",
                  f"stepfed step {step_num} chat failed: "
                  f"{type(e).__name__}: {e}")
        return None
    msg = response.get("message", {}) or {}
    content = (msg.get("content") or "").strip()
    log_event(trace_id, "INFO", "qwen_agent",
              f"stepfed step {step_num} qwen response chars={len(content)}")
    # Qwen sometimes wraps in ``` or adds prose. Try strict parse first,
    # then progressively lenient (already implemented in
    # _parse_tool_calls_from_content).
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "name" in obj:
            return {"function": {
                "name": obj["name"],
                "arguments": obj.get("arguments", {}),
            }}
    except json.JSONDecodeError:
        pass
    parsed = _parse_tool_calls_from_content(content)
    if parsed:
        return parsed[0]
    log_event(trace_id, "WARNING", "qwen_agent",
              f"stepfed step {step_num} could not parse tool call from: "
              f"{content[:200]!r}")
    return None


def run_agent_stepfed(
    problem: str,
    recipe: str,
    trace_id: str,
    model: str = "qwen2.5-coder:3b",
) -> dict:
    """Step-fed variant of run_agent: parse recipe into atomic steps,
    feed each ONE at a time so Qwen never sees the full recipe.

    Falls back to run_agent if recipe doesn't parse into >= 2 steps."""
    steps = _parse_recipe_steps(recipe)
    if len(steps) < 2:
        log_event(trace_id, "WARNING", "qwen_agent",
                  f"stepfed: only parsed {len(steps)} steps from recipe; "
                  f"falling back to legacy run_agent")
        return run_agent(problem, recipe, trace_id, model)

    log_event(trace_id, "INFO", "qwen_agent",
              f"stepfed: parsed {len(steps)} steps from recipe")

    session: list[dict] = []
    final_summary = ""
    completed = False
    last_error: str | None = None
    started = time.monotonic()

    for i, step_text in enumerate(steps, start=1):
        # Wall-clock budget guard -- abort if total stepfed time
        # exceeds the cap; better to bail and let claude review what
        # we have than block the worker for the full Telegram timeout.
        elapsed = time.monotonic() - started
        if elapsed > STEPFED_TOTAL_BUDGET_SEC:
            log_event(trace_id, "WARNING", "qwen_agent",
                      f"stepfed wall-clock budget "
                      f"{STEPFED_TOTAL_BUDGET_SEC}s exceeded at step "
                      f"{i}/{len(steps)} -- aborting loop")
            last_error = f"stepfed budget exceeded at {elapsed:.0f}s"
            break
        log_event(trace_id, "INFO", "qwen_agent",
                  f"stepfed step {i}/{len(steps)}: {step_text[:80]!r}")
        call = _stepfed_one_call(step_text, model, trace_id, i)
        if call is None:
            session.append({"step": i, "tool": "?",
                            "args": {},
                            "result": {"error": "qwen failed to emit "
                                       "valid tool call for this step"}})
            continue
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

        log_event(trace_id, "INFO", "qwen_agent",
                  f"stepfed step {i} tool={name} "
                  f"args_keys={list(args.keys())}")

        if name == "done":
            final_summary = args.get("summary", "")
            completed = True
            session.append({"step": i, "tool": "done",
                            "args": args, "result": {"ok": True}})
            break

        handler = TOOL_DISPATCH.get(name)
        if handler is None:
            result = {"error": f"unknown tool: {name}"}
        else:
            try:
                result = handler(**args)
            except TypeError as e:
                result = {"error": f"bad args: {e}"}
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
        session.append({"step": i, "tool": name,
                        "args": args, "result": result})
        if "error" in result:
            log_event(trace_id, "WARNING", "qwen_agent",
                      f"stepfed step {i} tool error: "
                      f"{result['error'][:120]}")

    if not final_summary:
        final_summary = (
            f"Step-fed execution finished {len(session)} step(s); "
            f"completed_via_done={completed}"
        )

    return {
        "summary": final_summary,
        "session": session,
        "steps": len(session),
        "completed_via_done": completed,
        "error": last_error,
    }


# ─────────────────────────────────────────────────────────────────
# Phase 16 Batch A -- read-only shadow planner.
#
# The Phase 15c shadow plan was a one-shot text-out call: Qwen got
# the user prompt + KB context + project_map and had to write a
# STEP-N recipe BLIND, no ability to actually look at the code it
# was planning against. That capped agreement scores at ~0.75 and
# made Qwen confabulate (e.g. SEN-fec42f17's Telegram-bot-tutorial
# in shadow recipe).
#
# This function gives Qwen a READ-ONLY tool surface (read_file,
# list_dir) so it can explore the actual codebase before writing
# its recipe. Edit / write / bash are explicitly absent -- shadow
# planning is a measurement, not an execution path. The production
# /code path keeps the full TOOLS_SCHEMA via run_agent_stepfed; only
# shadow has the locked-down view.
#
# Loop: at most ``max_tool_calls`` exploration steps, then the model
# is expected to emit a STEP-N recipe in plain content (no further
# tool calls). Best-effort everywhere -- any failure returns an
# empty recipe so /code never blocks on shadow.
# ─────────────────────────────────────────────────────────────────


# Tool subset Qwen sees in shadow mode. read_file + list_dir only.
# Definitions are independent of TOOLS_SCHEMA so we can word them
# specifically for "explore, then plan" rather than "execute".
SHADOW_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file's contents to see what's actually there "
                "before you reference it in your recipe. Returns up "
                "to 4KB. Use generously -- you cannot write a good "
                "edit_file recipe without seeing the current bytes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to project root."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List a directory to discover what files exist. Use "
                "before read_file when you're not sure of the exact "
                "filename."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to project root, or '.' "
                            "for the root."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
]

SHADOW_PLANNER_SYSTEM = (
    "You are planning a code change. You have read-only tools "
    "(read_file, list_dir). Use ONE OR TWO reads at MOST -- then "
    "STOP and write the recipe.\n"
    "\n"
    "STRICT TWO-STEP WORKFLOW:\n"
    "1. ONE read_file (or list_dir then read_file) on the file you "
    "   will modify. That's it -- no chains of reads.\n"
    "2. IMMEDIATELY after reading, your VERY NEXT response MUST be "
    "   the recipe in STEP-N format. No more tool calls. No prose.\n"
    "\n"
    "RECIPE FORMAT (output exactly this shape, nothing else):\n"
    "  STEP 1: <tool> <key>=\"<value>\" ...\n"
    "  STEP 2: <tool> ...\n"
    "  STEP N: done summary=\"<one-line>\"\n"
    "\n"
    "RULES:\n"
    "- Each STEP is exactly ONE tool call, no prose between steps.\n"
    "- Tools the EXECUTOR has: read_file, list_dir, write_file, "
    "  edit_file, run_bash, done.\n"
    "- For edit_file, the `old` arg must be a VERBATIM substring "
    "  from a read_file you ran. NEVER invent it.\n"
    "- Final step is ALWAYS done(summary=\"...\").\n"
    "- Output ONLY the recipe -- no narration, no markdown fences.\n"
    "\n"
    "CRITICAL: do NOT chain read_file calls hoping to fully explore "
    "the codebase. You have a small budget (3 tool calls TOTAL). "
    "After your first read, the next response MUST be the recipe."
)


# Hard cap on exploration steps in shadow. A 3B coder model that
# hasn't planned its recipe in 3 reads isn't going to plan a good
# one in 10. Phase 16 Batch A polish: dropped from 5 -> 3 after
# SEN-6a8a6539 trace showed Qwen burning all 5 slots on read_file
# calls without ever emitting the recipe. Tighter budget +
# strengthened system prompt forces transition to recipe output
# sooner.
SHADOW_MAX_TOOL_CALLS = 3


def run_shadow_planner(
    problem: str,
    kb_context_block: str,
    project_map: str,
    trace_id: str,
    model: str = "qwen2.5-coder:3b",
    max_tool_calls: int = SHADOW_MAX_TOOL_CALLS,
) -> dict:
    """Phase 16 Batch A: tools-enabled shadow planner.

    Runs an agentic loop where Qwen has read_file + list_dir AND
    is asked to output a STEP-N recipe. Caller wraps in
    ``asyncio.to_thread`` + ``asyncio.wait_for`` for the timeout.

    Returns ``{"recipe": str, "tool_calls": int, "error": str|None}``.
    Empty recipe on any failure (caller should fall back).
    """
    user_prompt = (
        f"PROBLEM:\n{problem}\n\n"
        f"PROJECT FILES (partial map):\n{project_map[:1500]}\n\n"
        + (
            f"KB CONTEXT (prior winning recipes):\n"
            f"{kb_context_block[:2500]}\n\n"
            if kb_context_block else ""
        )
        + "Now: explore the relevant files with read_file/list_dir, "
        "then output the recipe."
    )
    messages: list[dict] = [
        {"role": "system", "content": SHADOW_PLANNER_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    tool_calls_total = 0
    last_content = ""
    last_error: str | None = None

    # Loop exactly max_tool_calls + 1 turns where the LAST turn is
    # the recipe-output turn (no tool calls expected). Inside each
    # turn, refuse any call beyond max_tool_calls dispatches so the
    # cap is hard, not advisory.
    for step in range(max_tool_calls + 1):
        # Hard cap: after max_tool_calls dispatches, stop entering
        # the loop body for further dispatches. The final ollama
        # call below is forced to output a recipe via the "you've
        # used your budget" instruction injected last turn.
        if tool_calls_total >= max_tool_calls and step > 0:
            # One more chat turn to give the model a chance to emit
            # the recipe, then break regardless.
            try:
                response = _ollama_chat(
                    model, messages, tools=SHADOW_TOOLS_SCHEMA,
                )
                last_content = (
                    (response.get("message", {}) or {}).get("content")
                    or last_content
                )
            except (urllib.error.URLError, json.JSONDecodeError,
                    TimeoutError) as e:
                last_error = f"{type(e).__name__}: {e}"
                log_event(
                    trace_id, "WARNING", "qwen_agent",
                    f"shadow planner cap-exit chat failed: {last_error}",
                )
            log_event(
                trace_id, "INFO", "qwen_agent",
                f"shadow planner hit tool-call cap "
                f"({tool_calls_total}/{max_tool_calls}); ending",
            )
            break
        try:
            response = _ollama_chat(
                model, messages, tools=SHADOW_TOOLS_SCHEMA,
            )
        except (urllib.error.URLError, json.JSONDecodeError,
                TimeoutError) as e:
            last_error = f"{type(e).__name__}: {e}"
            log_event(trace_id, "WARNING", "qwen_agent",
                      f"shadow planner step {step + 1} chat failed: "
                      f"{last_error}")
            break
        msg = response.get("message", {}) or {}
        content = msg.get("content") or ""
        last_content = content
        tool_calls = msg.get("tool_calls") or []
        # qwen2.5-coder fallback: tool calls in content
        if not tool_calls:
            tool_calls = _parse_tool_calls_from_content(content)
        # No tool calls -> Qwen is done exploring; content should be
        # the recipe (or junk we'll discover when scoring).
        if not tool_calls:
            log_event(trace_id, "INFO", "qwen_agent",
                      f"shadow planner ended at step {step + 1} "
                      f"(no more tool calls; content_chars={len(content)})")
            break
        # Reject any tool call NOT in the read-only set. This is the
        # safety boundary for shadow planning -- if the model tries
        # write_file or run_bash, we refuse and feed back an error.
        # (Phase 16 design: shadow is measurement, never execution.)
        readonly_names = {"read_file", "list_dir"}
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })
        for call in tool_calls:
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
            tool_calls_total += 1
            if name not in readonly_names:
                # Refuse + feed back a friendly error so Qwen learns
                # its constraint within this turn.
                err_msg = (
                    f"tool '{name}' is not allowed in shadow planning "
                    f"(read-only mode: read_file, list_dir only)"
                )
                log_event(trace_id, "INFO", "qwen_agent",
                          f"shadow planner rejected tool={name}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "content": json.dumps({"error": err_msg}),
                })
                continue
            # Dispatch the read-only tool.
            try:
                if name == "read_file":
                    result = tool_read_file(args.get("path", ""))
                elif name == "list_dir":
                    result = tool_list_dir(args.get("path", "."))
                else:
                    result = {"error": "unknown tool"}
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            log_event(trace_id, "INFO", "qwen_agent",
                      f"shadow planner step {step + 1} tool={name} "
                      f"result_keys={list(result.keys())}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": json.dumps(result)[:4000],
            })
        # Cap reached -- ask Qwen to stop exploring next turn.
        if tool_calls_total >= max_tool_calls:
            messages.append({
                "role": "user",
                "content": (
                    "You've used your exploration budget. Output "
                    "the STEP-N recipe now."
                ),
            })

    # Extract a recipe from whatever Qwen produced last. Reuse
    # _parse_recipe_steps so we share the canonical parser with
    # the production scoring path (core.plan_agreement uses it too).
    recipe_steps = _parse_recipe_steps(last_content)
    recipe = "\n".join(f"STEP {i + 1}: {s}" for i, s in enumerate(recipe_steps))
    return {
        "recipe": recipe,
        "tool_calls": tool_calls_total,
        "error": last_error,
    }


def run_agent(
    problem: str,
    recipe: str,
    trace_id: str,
    model: str = "qwen2.5-coder:3b",
    max_steps: int = MAX_AGENT_STEPS,
) -> dict:
    """Run the Qwen tool-calling agent loop. Returns:
      {
        "summary": str,         # final summary from done() or last content
        "session": list[dict],  # full trace of tool calls + results
        "steps": int,           # how many turns executed
        "completed_via_done": bool,
        "error": str | None,
      }
    Sync function -- caller wraps in asyncio.to_thread.
    """
    project_map = _project_map()
    user_prompt = (
        f"USER REQUEST:\n{problem}\n\n"
        f"RECIPE FROM SENIOR ENGINEER:\n{recipe}\n\n"
        f"CURRENT PROJECT STRUCTURE:\n{project_map}\n\n"
        "Apply the change. Use tools. If you need a file that doesn't "
        "appear above, use `write_file` to create it. If you're "
        "editing existing code, `read_file` it first to see the exact "
        "current contents (your `old` argument to `edit_file` must "
        "match BYTE-EXACT including indentation). Call `done` when "
        "finished."
    )
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    session: list[dict] = []
    steps = 0
    final_summary = ""
    completed = False
    last_error: str | None = None
    consecutive_failures = 0

    for step in range(max_steps):
        steps = step + 1
        try:
            response = _ollama_chat(model, messages, tools=TOOLS_SCHEMA)
        except (urllib.error.URLError, json.JSONDecodeError,
                TimeoutError) as e:
            last_error = f"{type(e).__name__}: {e}"
            log_event(trace_id, "ERROR", "qwen_agent",
                      f"step {steps} chat call failed: {last_error}")
            break

        msg = response.get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        # qwen2.5-coder fallback: tool calls emitted as JSON in content
        if not tool_calls:
            tool_calls = _parse_tool_calls_from_content(content)
            if tool_calls:
                log_event(trace_id, "INFO", "qwen_agent",
                          f"step {steps} parsed "
                          f"{len(tool_calls)} tool_calls from content")

        # If still no tool calls, treat as final summary
        if not tool_calls:
            final_summary = content.strip() or final_summary
            log_event(trace_id, "INFO", "qwen_agent",
                      f"step {steps} no tool_calls; ending "
                      f"(content_chars={len(content)})")
            break

        # Append the assistant's tool_calls turn
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": tool_calls})

        # Dispatch each tool call
        for call in tool_calls:
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

            log_event(trace_id, "INFO", "qwen_agent",
                      f"step {steps} tool={name} "
                      f"args_keys={list(args.keys())}")

            if name == "done":
                final_summary = args.get("summary", "")
                completed = True
                session.append({"step": steps, "tool": "done",
                                "args": args, "result": {"ok": True}})
                break

            handler = TOOL_DISPATCH.get(name)
            if handler is None:
                result = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = handler(**args)
                except TypeError as e:
                    result = {"error": f"bad args: {e}"}
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
            session.append({"step": steps, "tool": name,
                            "args": args, "result": result})

            # Track consecutive failures
            if "error" in result:
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            messages.append({
                "role": "tool",
                "content": json.dumps(result)[:MAX_TOOL_OUTPUT],
                "name": name,
            })

        if completed:
            break

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log_event(
                trace_id, "WARNING", "qwen_agent",
                f"step {steps} -- {consecutive_failures} consecutive "
                f"tool failures; bailing out for Claude to review",
            )
            final_summary = (
                f"(stopped after {consecutive_failures} consecutive "
                f"tool failures -- Qwen couldn't proceed)"
            )
            break

    if not final_summary:
        final_summary = (
            "(Qwen ran out of steps without calling done())"
            if steps >= max_steps
            else "(Qwen exited without producing a summary)"
        )

    return {
        "summary": final_summary,
        "session": session,
        "steps": steps,
        "completed_via_done": completed,
        "error": last_error,
    }
