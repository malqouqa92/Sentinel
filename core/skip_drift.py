"""Phase 16 Batch C drift-detect (bail-no-penalty).

Pre-flight check for ``skills.code_assist._execute_skip_replay``:
parses the recipe for every ``edit_file`` step's ``old="..."``
literal and verifies that substring exists in the target file ON
DISK NOW. If any old= is missing, the replay would silently no-op
when stepfed runs, producing a wrong/empty diff and tanking
diff-match -- diff-match correctly rejects, but the failed-replay
counter increment unfairly penalizes the pattern for environmental
state drift (e.g. the user's emoji-bar prompt picks a different
emoji each successful run, so the next replay's stored ``old=``
is now stale).

Drift-detect catches this BEFORE stepfed runs and returns a sentinel
so the caller can return a 'failed' status with reason
``environmental:state_drift:<context>`` without incrementing the
counter (state drift is environmental, not a recipe defect).

This module is intentionally separate from skills/code_assist.py so
the bot's ``_git_reset_hard`` (which scopes to ``core skills agents
tests interfaces``) can't wipe its source mid-development. The skill
imports from here.

Public API:
    detect_recipe_drift(recipe, project_root) -> (bool, str)
        Returns (drift_detected, reason).
        On True, caller should bail without penalty.
"""
from __future__ import annotations

import re
from pathlib import Path

# Regex pulls the old= literal even with escaped quotes inside.
# We only call this on step texts that already start with edit_file
# (the caller filters via str.startswith), so we don't anchor the
# pattern to `edit_file` -- a `[^"\n]*?` prefix can't traverse the
# `path="..."` arg without matching across the quote.
_RECIPE_EDIT_OLD_RE = re.compile(
    r'\bold\s*=\s*"((?:[^"\\]|\\.)*)"',
)
_RECIPE_PATH_RE = re.compile(r'\bpath\s*=\s*"([^"]+)"')


def _unescape_recipe_literal(s: str) -> str:
    """Reverse the standard recipe-arg escapes: \\n -> newline,
    \\t -> tab, \\" -> ", \\\\ -> \\. Defensive on malformed input
    (returns the raw string)."""
    if "\\" not in s:
        return s
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            elif nxt == '"':
                out.append('"')
            elif nxt == "\\":
                out.append("\\")
            elif nxt == "'":
                out.append("'")
            else:
                # Unknown escape -- preserve verbatim so we don't
                # alter the substring meaningfully.
                out.append(c)
                out.append(nxt)
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def detect_recipe_drift(
    recipe: str,
    project_root: Path | str,
) -> tuple[bool, str]:
    """Returns ``(drift_detected, reason_token)``. False on:
      - empty recipe
      - recipe has no edit_file steps (write_file is idempotent;
        no drift possible)
      - every edit_file's old= literal is found in its target

    Returns True on:
      - target file referenced by an edit_file step doesn't exist
      - target file can't be read
      - old= literal is not in the target file's current bytes

    Reason tokens (grep-able for log filters):
      - target_missing:<path>
      - target_unreadable:<path>
      - old_not_found:<path>
    """
    if not recipe:
        return False, ""

    # Parse step-by-step so we can pair `path=` with `old=` per step.
    # Reuse the canonical step parser from qwen_agent if available.
    try:
        from core.qwen_agent import _parse_recipe_steps
        steps = _parse_recipe_steps(recipe)
    except Exception:
        # Fallback: treat the whole recipe as one block. Still
        # better than crashing.
        steps = [recipe]

    project_root = Path(project_root)

    for step_text in steps:
        # Only edit_file is state-dependent.
        if "edit_file" not in step_text:
            continue
        # Some steps could be `read_file` etc. that contain the
        # token "edit_file" in args; require it to be the leading
        # tool-name token.
        stripped = step_text.lstrip()
        if not stripped.startswith("edit_file"):
            continue

        m_path = _RECIPE_PATH_RE.search(step_text)
        if not m_path:
            continue
        rel_path = m_path.group(1).strip()

        m_old = _RECIPE_EDIT_OLD_RE.search(step_text)
        if not m_old:
            continue
        old_literal = _unescape_recipe_literal(m_old.group(1))

        target = project_root / rel_path
        if not target.exists():
            return True, f"target_missing:{rel_path}"
        try:
            content = target.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            return True, f"target_unreadable:{rel_path}"

        if old_literal and old_literal not in content:
            return True, f"old_not_found:{rel_path}"

    return False, ""
