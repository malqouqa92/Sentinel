"""Phase 15c -- structural agreement scorer for shadow planning.

We measure whether Qwen-with-KB-context produces a recipe broadly
similar to Claude's, NOT exact-token match. The signal we want is
"do the plans agree on what to touch and how" -- a baseline for
Qwen's planning ability that compounds as the KB grows. On bigger
hardware (or more shots) we'll know whether Qwen could become
Claude-optional from this column alone.

Pure heuristic, no LLM judging: file paths touched (Jaccard),
tool names used (Jaccard), step count proximity. Weighted blend
clamped to [0.0, 1.0]. Best-effort throughout: any parse failure
returns 0.0 rather than crashing the /code path.
"""
from __future__ import annotations

import re
from typing import Iterable

# Reuse the recipe parser from qwen_agent so we score against EXACTLY
# what the executor would parse. Late import keeps this module light
# and avoids circular dep through skills.code_assist.
from core.qwen_agent import (
    _parse_recipe_steps as _parse_recipe_steps,
    _parse_step_text_to_tool_call as _parse_step_to_tool_call,
)


# Weights from the Phase 15c spec. file_jaccard dominates because
# "are we touching the same files?" is the strongest single signal
# of whether two plans agree on the work; tool overlap is a secondary
# structural check; step count proximity is a coarse "did we plan to
# do roughly as much?" signal.
W_FILES = 0.5
W_TOOLS = 0.3
W_STEPS = 0.2

# Path arg names across the executor's tool surface.
_PATH_KEYS = ("path", "file", "filepath", "filename")
_PATH_RE = re.compile(
    r"(?:" + "|".join(_PATH_KEYS) + r")\s*=\s*\"((?:\\.|[^\"\\])*)\"",
    re.IGNORECASE,
)


def _files_in_step(step_text: str) -> set[str]:
    """Pull every path-like arg out of a step body. Falls back to a
    direct regex if the structured tool-call parser can't decode the
    step (e.g. malformed recipe; we still want to count the strings
    that LOOK like paths). Note: qwen_agent's parser returns the
    OpenAI-shaped envelope {function: {name, arguments}} so we have
    to dig down a level."""
    parsed = _parse_step_to_tool_call(step_text or "")
    if parsed:
        fn = parsed.get("function") or {}
        args = fn.get("arguments") or {}
        out: set[str] = set()
        for key, val in args.items():
            if not isinstance(val, str):
                continue
            if key.lower() in _PATH_KEYS:
                v = val.strip()
                if v:
                    out.add(v)
        if out:
            return out
    # Fallback regex (unparseable steps).
    return {m.group(1).strip() for m in _PATH_RE.finditer(step_text or "")
            if m.group(1).strip()}


def _tool_in_step(step_text: str) -> str | None:
    parsed = _parse_step_to_tool_call(step_text or "")
    if parsed:
        fn = parsed.get("function") or {}
        return str(fn.get("name", "")).lower() or None
    return None


def _files_and_tools(recipe: str) -> tuple[set[str], list[str]]:
    """Return (set-of-files-touched, list-of-tool-names) for a
    recipe. Tool names come back as a list (not a set) so a recipe
    that reads_file four times still gets credit for the verb -- the
    Jaccard call de-dupes when computing overlap."""
    if not recipe:
        return set(), []
    files: set[str] = set()
    tools: list[str] = []
    for body in _parse_recipe_steps(recipe):
        for f in _files_in_step(body):
            files.add(f)
        t = _tool_in_step(body)
        if t:
            tools.append(t)
    return files, tools


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Standard Jaccard similarity. Both empty -> 1.0 (no work
    proposed, no work proposed -- they agree). One empty -> 0.0."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _step_count_proximity(claude_n: int, qwen_n: int) -> float:
    """Closer step counts -> higher score. Both zero -> 0.0 (no
    plans at all means no agreement signal). The proximity formula
    is 1 - |diff| / max -- simple, monotonic, bounded in [0, 1]."""
    if claude_n <= 0 or qwen_n <= 0:
        return 0.0
    diff = abs(claude_n - qwen_n)
    return max(0.0, 1.0 - (diff / max(claude_n, qwen_n)))


def score_plan_agreement(
    claude_recipe: str | None, qwen_recipe: str | None,
) -> float:
    """Heuristic [0.0, 1.0] structural agreement.

    Returns 0.0 on:
      - either input None / empty / unparseable (no STEP N: matches)
      - exception during parsing (best-effort: never raise)

    Returns 1.0 only when files, tools, AND step counts all match
    perfectly. Anything in between is a weighted blend.
    """
    try:
        if not claude_recipe or not qwen_recipe:
            return 0.0
        c_files, c_tools = _files_and_tools(claude_recipe)
        q_files, q_tools = _files_and_tools(qwen_recipe)
        # If Qwen produced literally no parseable steps, treat as a
        # zero -- callers want "did Qwen plan a real recipe?" to be
        # the gating signal, not a partial-credit guess based on the
        # other recipe's count.
        if not c_tools and not q_tools:
            return 0.0
        if not q_tools:
            return 0.0

        file_score = _jaccard(c_files, q_files)
        tool_score = _jaccard(c_tools, q_tools)
        step_score = _step_count_proximity(len(c_tools), len(q_tools))
        blended = (
            W_FILES * file_score
            + W_TOOLS * tool_score
            + W_STEPS * step_score
        )
        return max(0.0, min(1.0, blended))
    except Exception:
        # Best-effort: a malformed recipe must never break /code.
        return 0.0
