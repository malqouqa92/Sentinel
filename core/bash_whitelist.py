"""Phase 16 Batch C-safety -- bash command whitelist for replay-skip.

When Batch C ships and a stored recipe is replayed without Claude
review (because it's marked skip-eligible), we MUST gate any
`run_bash` step against a strict allowlist. The recipe came from a
prior /code run that DID get Claude review, so it should only contain
safe commands -- but defensive: a corrupted KB row, a malicious past
attempt, or a stored recipe that was reviewed-and-approved with a
borderline command shouldn't get a free pass on subsequent replays.

This module is allowlist-only: an unrecognized command is REJECTED.
We don't try to parse and analyze arbitrary shell -- we match the
command against curated regex patterns covering the small set of
verification operations recipes legitimately need.

Usage:

    from core.bash_whitelist import is_bash_safe_for_replay

    ok, reason = is_bash_safe_for_replay(cmd_str)
    if not ok:
        # Reject the recipe from skip-eligibility; fall back to full
        # Claude pipeline.
        ...

Wired into recipe_linter's "is this recipe safe to replay without
Claude review?" check. NOT applied to /code's normal `run_bash` path
-- regular /code runs still get reviewer-Claude as the safety gate.
"""
from __future__ import annotations

import re


# Allowlist patterns. Each entry is a regex that the FULL command
# string must match (re.fullmatch). Deliberately conservative: only
# pure verification + introspection commands. No file-mutating ops
# (cp/mv/rm/touch), no network (curl/wget/ssh), no installers
# (pip/apt/conda), no privilege escalation (sudo/runas), no destructive
# git ops (push/reset --hard/checkout).
#
# The allowlist mirrors actual recipe usage observed in production:
# `python -c "..."` to import-test a module + verify behavior, and
# `python -m pytest tests/...` to run tests after an edit.
_ALLOWLIST_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "python -c",
        re.compile(r'^python(?:3)?\s+-c\s+["\'][\s\S]*["\']\s*$'),
    ),
    (
        "python -m pytest",
        re.compile(r"^python(?:3)?\s+-m\s+pytest(?:\s+[\w./_-]+)*\s*$"),
    ),
    (
        "pytest",
        re.compile(r"^pytest(?:\s+[\w./_-]+)*\s*$"),
    ),
    (
        "python script.py (project-relative)",
        re.compile(r"^python(?:3)?\s+[\w./_-]+\.py(?:\s+[\w./_-]+)*\s*$"),
    ),
    (
        "ls / pwd (read-only inspection)",
        re.compile(r"^(?:ls|pwd)(?:\s+[\w./_-]+)*\s*$"),
    ),
]


# Hard denylist: substrings that NEVER appear in a safe command. Even
# if a command somehow passes the allowlist regex, these substrings
# are an immediate reject (defense in depth against allowlist
# regex bugs).
_DENY_SUBSTRINGS = [
    " rm ", "rm -",      # delete files
    " mv ", "mv -",      # move/rename
    "git push", "git reset --hard",
    "git checkout", "git rebase",
    "sudo", "runas",
    "curl ", "wget ",
    " ssh ", "scp ",
    "pip install", "pip uninstall",
    "apt ", "apt-get",
    "rmdir", " del ",
    ">/dev/", ">>/dev/",  # redirect to devices
    "$(", "`",            # command substitution
    " && rm", " ; rm",
    " > /etc", " >> /etc",
]


def is_bash_safe_for_replay(command: str) -> tuple[bool, str]:
    """Return (allowed, reason) for a `run_bash` command on the
    skip-eligible replay path.

    Allowed iff the command matches an allowlist pattern AND
    contains no denylist substring. Reason is a short human-readable
    string explaining the verdict (used in log lines + telemetry)."""
    if not command:
        return False, "empty command"
    cmd = command.strip()
    # Wrap with leading/trailing space so " rm " denylist substrings
    # match even at command boundaries.
    cmd_padded = f" {cmd} "
    for deny in _DENY_SUBSTRINGS:
        if deny in cmd_padded:
            return False, f"contains denylist substring {deny.strip()!r}"
    for label, pattern in _ALLOWLIST_PATTERNS:
        if pattern.fullmatch(cmd):
            return True, f"matches allowlist: {label}"
    return False, "no allowlist pattern matched"


def is_recipe_bash_safe(recipe_steps: list[dict]) -> tuple[bool, str]:
    """Walk a parsed recipe (list of {tool, args} dicts) and verify
    every `run_bash` step is whitelist-safe.

    Returns (all_safe, reason). On the first reject, returns the
    rejecting step's reason. If no run_bash steps exist, returns
    (True, "no run_bash steps")."""
    if not recipe_steps:
        return True, "empty recipe"
    bash_count = 0
    for i, step in enumerate(recipe_steps):
        tool = (step.get("tool") or "").lower()
        if tool != "run_bash":
            continue
        bash_count += 1
        args = step.get("args") or {}
        cmd = args.get("command") or args.get("cmd") or ""
        ok, reason = is_bash_safe_for_replay(cmd)
        if not ok:
            return False, f"step {i + 1}: {reason} (cmd={cmd[:80]!r})"
    if bash_count == 0:
        return True, "no run_bash steps in recipe"
    return True, f"all {bash_count} run_bash step(s) whitelist-safe"
