"""Phase 16 Batch C-safety -- recipe linter for skip-eligible patterns.

Before a stored KB pattern is allowed to take the Batch C "skip
Claude pre-teach + reviewer" path, its recipe must pass deterministic
safety gates. The linter answers ONE question per pattern:

    "Is this recipe safe to execute without Claude review?"

Failures here -> pattern is excluded from skip-eligibility for THIS
/code attempt. Patterns aren't permanently demoted; the linter is a
per-replay safety check.

Checks:

    L1. Recipe parses to >= 2 STEPs (anything less is malformed; the
        same threshold the stepfed parser uses).
    L2. Final step is `done(summary=...)`.
    L3. There exists a `run_bash` verification step BEFORE `done`.
        Skip-eligibility requires runtime verification; a recipe that
        edits files but never runs a verification command produces
        only an empty diff-match positive (Batch B's M21 reject case).
    L4. All tool names are in the executor's known set
        (read_file, list_dir, write_file, edit_file, run_bash, done).
    L5. Every `run_bash` step's command passes the bash whitelist
        (core/bash_whitelist.is_recipe_bash_safe).
    L6. No write_file or edit_file step targets a path that looks
        like a runtime / sensitive file (logs/, *.db, .env*, secrets,
        /etc/, etc.) -- skip-eligible recipes only touch source.
    L7. The recipe's stored solution_pattern was not truncated at
        the RECIPE_MAX_CHARS_STEPFED boundary -- truncated recipes
        may be missing their final `done` or verification step,
        making them unsafe to replay.
    L8. Recipe contains at least one `edit_file` or `write_file`
        step. A "recipe" of read_file + done (with the done summary
        being a clarification question) is a legitimate Claude
        REFUSAL coerced into recipe shape by Option 1's reformat,
        not a real solution. Producing one of these as a 1/1 PASS
        was the regression caught in trace SEN-b46a27cf
        (pattern id=96, 2026-05-06): Claude correctly refused
        ("provide the emoji pair you want"), reformat coerced it
        into shape, reviewer-Claude saw existing emojis on disk
        and passed it. L8 prevents that pattern from ever reaching
        skip-eligibility AND -- when the same check is applied
        inline in the /code attempt loop -- prevents storage of
        such patterns in the first place.

Returns LintResult with verdict + per-check details so logs can
explain WHICH check fired.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.bash_whitelist import is_recipe_bash_safe


# Mirrors RECIPE_MAX_CHARS_STEPFED in skills/code_assist.py. Kept as
# a separate constant here to avoid an import cycle (recipe_linter
# is consumed by skills/code_assist.py).
_RECIPE_MAX_CHARS_STEPFED = 8000

# Tool names recognized by core.qwen_agent's executor.
_VALID_TOOLS = {
    "read_file", "list_dir", "write_file",
    "edit_file", "run_bash", "done",
}

# Path patterns that disqualify a write_file/edit_file step from
# skip-eligibility. Curated conservatively: the goal is "skip-
# eligible recipes only modify SOURCE code, never runtime / data /
# secret files."
_FORBIDDEN_PATH_PREFIXES = (
    "logs/", "logs\\",
    "__pycache__/", "__pycache__\\",
    ".git/", ".git\\",
    "/etc/", "/var/", "/sys/", "/proc/",
    "C:\\Windows\\", "C:/Windows/",
    "C:\\Users", "C:/Users",  # absolute user-home access
)
_FORBIDDEN_PATH_EXACT = (
    ".env", ".env.bot", ".env.local", ".envrc",
    "id_rsa", "id_ed25519",
    "knowledge.db", "sentinel.db", "memory.db",
)
_FORBIDDEN_PATH_SUBSTRINGS = (
    ".db", ".sqlite", ".sqlite3",
    "secret", "credential", "password",
    "id_rsa", "id_ed25519",
)


@dataclass
class LintResult:
    """Outcome of linting a recipe.

    safe: True iff every check passed.
    reason: short string explaining the verdict (rejection cause if
            safe is False, or summary if True).
    failed_checks: list of check IDs ("L1".."L7") that failed.
    """
    safe: bool
    reason: str
    failed_checks: list[str] = field(default_factory=list)


def _is_forbidden_path(path: str) -> tuple[bool, str]:
    """Return (forbidden, why). Conservative: any prefix/exact/substr
    match disqualifies."""
    if not path:
        return True, "empty path"
    p = path.strip().replace("\\", "/")
    # Strip a leading `./` PREFIX specifically (not individual `.`/`/`
    # chars -- lstrip("./") would eat the leading dot of `.env`).
    if p.startswith("./"):
        p = p[2:]
    for pre in _FORBIDDEN_PATH_PREFIXES:
        norm_pre = pre.replace("\\", "/")
        if p.startswith(norm_pre):
            return True, f"path starts with {pre!r}"
    base = p.rsplit("/", 1)[-1]
    for ex in _FORBIDDEN_PATH_EXACT:
        if base == ex:
            return True, f"path basename is exact {ex!r}"
    p_lower = p.lower()
    for sub in _FORBIDDEN_PATH_SUBSTRINGS:
        if sub in p_lower:
            return True, f"path contains forbidden substring {sub!r}"
    return False, ""


def lint_recipe_for_skip(
    recipe_text: str,
    parsed_steps: list[dict] | None = None,
) -> LintResult:
    """Run all linter checks on a recipe.

    `recipe_text`: the raw stored recipe string (solution_pattern).
    `parsed_steps`: optionally a pre-parsed list of {tool, args}
                    dicts; if None, parses from recipe_text via
                    core.qwen_agent's parser.
    """
    failed: list[str] = []
    reasons: list[str] = []

    # L7 -- truncation check (cheapest, run first; if the stored
    # text is at the cap, the recipe is suspect regardless of what
    # the parsed steps look like).
    if recipe_text and len(recipe_text) >= _RECIPE_MAX_CHARS_STEPFED:
        failed.append("L7")
        reasons.append(
            f"L7: recipe length {len(recipe_text)} hit truncation "
            f"cap {_RECIPE_MAX_CHARS_STEPFED}; recipe may be incomplete"
        )

    # Parse if caller didn't provide. Late-imported to avoid cycle.
    if parsed_steps is None:
        try:
            from core.qwen_agent import (
                _parse_recipe_steps, _parse_step_text_to_tool_call,
            )
            step_strs = _parse_recipe_steps(recipe_text or "")
            parsed_steps = []
            for s in step_strs:
                call = _parse_step_text_to_tool_call(s)
                if call:
                    parsed_steps.append({
                        "tool": (
                            call.get("function", {}).get("name")
                            if "function" in call
                            else call.get("name", "")
                        ).lower(),
                        "args": (
                            call.get("function", {}).get("arguments", {})
                            if "function" in call
                            else call.get("arguments", {})
                        ),
                    })
        except Exception as e:
            return LintResult(
                safe=False,
                reason=f"linter: recipe parse failed: {type(e).__name__}",
                failed_checks=failed + ["L1"],
            )

    # L1 -- minimum step count.
    if not parsed_steps or len(parsed_steps) < 2:
        failed.append("L1")
        reasons.append(
            f"L1: recipe parsed {len(parsed_steps or [])} STEP(s); "
            f"need >= 2"
        )
        return LintResult(
            safe=False, reason="; ".join(reasons), failed_checks=failed,
        )

    # L4 -- all tool names valid.
    for i, step in enumerate(parsed_steps):
        tool = (step.get("tool") or "").lower()
        if tool not in _VALID_TOOLS:
            failed.append("L4")
            reasons.append(
                f"L4: step {i + 1} unknown tool {tool!r}"
            )

    # L2 -- final step is `done`.
    final_tool = (parsed_steps[-1].get("tool") or "").lower()
    if final_tool != "done":
        failed.append("L2")
        reasons.append(
            f"L2: final step is {final_tool!r}, expected 'done'"
        )

    # L3 -- there's a run_bash verification step somewhere before done.
    bash_steps_before_done = [
        s for s in parsed_steps[:-1]
        if (s.get("tool") or "").lower() == "run_bash"
    ]
    if not bash_steps_before_done:
        failed.append("L3")
        reasons.append(
            "L3: no run_bash verification step before done; replay "
            "would have no runtime check"
        )

    # L6 -- write_file / edit_file paths not forbidden.
    for i, step in enumerate(parsed_steps):
        tool = (step.get("tool") or "").lower()
        if tool not in ("write_file", "edit_file"):
            continue
        path = (step.get("args") or {}).get("path") or ""
        forbidden, why = _is_forbidden_path(path)
        if forbidden:
            failed.append("L6")
            reasons.append(f"L6: step {i + 1} {tool} {why}")

    # L5 -- bash whitelist on every run_bash command.
    bash_ok, bash_reason = is_recipe_bash_safe(parsed_steps)
    if not bash_ok:
        failed.append("L5")
        reasons.append(f"L5: {bash_reason}")

    # L8 -- recipe contains at least one edit_file or write_file
    # step. Without one, the recipe is a degenerate "read + done"
    # shape that wasn't a real solution; storing/skipping it would
    # be the regression that caused pattern id=96.
    edit_steps = [
        s for s in parsed_steps
        if (s.get("tool") or "").lower() in ("edit_file", "write_file")
    ]
    if not edit_steps:
        failed.append("L8")
        reasons.append(
            "L8: recipe has no edit_file or write_file step -- not "
            "a real solution (likely a clarification question coerced "
            "into recipe shape)"
        )

    if failed:
        # Dedup checks (L4 / L6 may fire multiple times for multi-step
        # violations) while preserving order.
        seen = set()
        dedup_failed = []
        for c in failed:
            if c not in seen:
                seen.add(c)
                dedup_failed.append(c)
        return LintResult(
            safe=False,
            reason="; ".join(reasons),
            failed_checks=dedup_failed,
        )

    return LintResult(
        safe=True,
        reason=(
            f"all {len(parsed_steps)} steps lint-clean "
            f"({len(edit_steps)} edit/write + "
            f"{len(bash_steps_before_done)} run_bash verify + done)"
        ),
        failed_checks=[],
    )
