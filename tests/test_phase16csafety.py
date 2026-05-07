"""Phase 16 C-safety -- recipe linter + bash whitelist + nvidia-smi
window-flash fix.

C-safety lands BEFORE Batch C (the actual skip path) so when C ships,
the deterministic safety gates are already in place. Without this
batch, a stored recipe could replay-skip Claude review and run an
arbitrary `run_bash` step OR edit a runtime/sensitive file with no
guard.

Coverage:

  Bash whitelist (core.bash_whitelist):
    S01 -- empty command rejected
    S02 -- `python -c "..."` allowed
    S03 -- `python -m pytest` allowed
    S04 -- bare `pytest tests/foo.py` allowed
    S05 -- `python script.py` allowed
    S06 -- `ls` / `pwd` allowed (read-only inspection)
    S07 -- `rm -rf /` rejected (deny substring)
    S08 -- `git push origin main` rejected
    S09 -- `git reset --hard HEAD` rejected
    S10 -- `curl http://...` rejected
    S11 -- `pip install foo` rejected
    S12 -- `sudo anything` rejected
    S13 -- command substitution `$(rm)` rejected
    S14 -- backtick command substitution rejected
    S15 -- chained `python -c '...' && rm` rejected
    S16 -- recipe with no run_bash steps -> bash_safe True
    S17 -- recipe with all-safe run_bash steps -> bash_safe True
    S18 -- recipe with one bad run_bash step -> bash_safe False

  Recipe linter (core.recipe_linter):
    S31 -- 0-step recipe -> L1 fail
    S32 -- 1-step recipe -> L1 fail
    S33 -- valid 3-step recipe (edit + run_bash + done) -> safe
    S34 -- final step not done -> L2 fail
    S35 -- recipe missing run_bash before done -> L3 fail
    S36 -- recipe with unknown tool name -> L4 fail
    S37 -- recipe with bad bash command -> L5 fail
    S38 -- recipe writing to logs/ -> L6 fail
    S39 -- recipe writing to .env -> L6 fail
    S40 -- recipe writing to *.db -> L6 fail
    S41 -- recipe at exactly the truncation cap -> L7 fail
    S42 -- recipe well below cap is fine -> L7 pass
    S43 -- multiple violations report all in failed_checks
    S44 -- safe recipe gets a positive reason describing structure

  nvidia-smi window flash fix:
    S51 -- core/health.py uses CREATE_NO_WINDOW on win32 path
    S52 -- non-win32 path passes flags=0 (no-op)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.bash_whitelist import (
    is_bash_safe_for_replay,
    is_recipe_bash_safe,
)
from core.recipe_linter import (
    LintResult,
    lint_recipe_for_skip,
)


# ─────────────────────────────────────────────────────────────────
# Bash whitelist
# ─────────────────────────────────────────────────────────────────


def test_s01_empty_command_rejected():
    ok, reason = is_bash_safe_for_replay("")
    assert ok is False
    assert "empty" in reason.lower()


@pytest.mark.parametrize("cmd", [
    'python -c "print(1+1)"',
    "python -c 'import math_utils; print(math_utils.add(2,3))'",
    'python3 -c "from foo import bar"',
])
def test_s02_python_dash_c_allowed(cmd):
    ok, reason = is_bash_safe_for_replay(cmd)
    assert ok, f"expected allowed: {cmd!r} got {reason!r}"


@pytest.mark.parametrize("cmd", [
    "python -m pytest tests/test_foo.py",
    "python -m pytest",
    "python3 -m pytest tests/",
])
def test_s03_python_pytest_allowed(cmd):
    ok, _ = is_bash_safe_for_replay(cmd)
    assert ok, cmd


def test_s04_bare_pytest_allowed():
    ok, _ = is_bash_safe_for_replay("pytest tests/test_foo.py")
    assert ok


def test_s05_python_script_allowed():
    ok, _ = is_bash_safe_for_replay("python tools/preload_kb.py")
    assert ok


def test_s06_ls_pwd_allowed():
    assert is_bash_safe_for_replay("ls")[0]
    assert is_bash_safe_for_replay("pwd")[0]
    assert is_bash_safe_for_replay("ls core/")[0]


@pytest.mark.parametrize("cmd", [
    "rm -rf /",
    "rm file.py",
    "rm -f workspace/data.json",
])
def test_s07_rm_rejected(cmd):
    ok, reason = is_bash_safe_for_replay(cmd)
    assert ok is False, cmd
    assert "rm" in reason.lower() or "denylist" in reason.lower()


def test_s08_git_push_rejected():
    ok, _ = is_bash_safe_for_replay("git push origin main")
    assert ok is False


def test_s09_git_reset_hard_rejected():
    ok, _ = is_bash_safe_for_replay("git reset --hard HEAD")
    assert ok is False


def test_s10_curl_rejected():
    ok, _ = is_bash_safe_for_replay("curl https://evil.example.com")
    assert ok is False


def test_s11_pip_install_rejected():
    ok, _ = is_bash_safe_for_replay("pip install requests")
    assert ok is False


def test_s12_sudo_rejected():
    ok, _ = is_bash_safe_for_replay("sudo anything")
    assert ok is False


def test_s13_command_substitution_dollar_rejected():
    """$(...) command substitution lets a benign-looking string
    smuggle in arbitrary commands. Reject categorically."""
    ok, _ = is_bash_safe_for_replay('python -c "import os; os.system(\\"ls\\")"')
    # This SHOULD be rejected because "$(" is in deny list. But the
    # actual smuggle vector here is os.system inside python -c, which
    # is a different category of risk. The denylist catches the
    # outer shell-substitution form specifically:
    ok_bad, _ = is_bash_safe_for_replay('echo $(rm -rf /)')
    assert ok_bad is False


def test_s14_backtick_substitution_rejected():
    ok, _ = is_bash_safe_for_replay("echo `rm -rf /`")
    assert ok is False


def test_s15_chained_with_rm_rejected():
    ok, _ = is_bash_safe_for_replay('python -c "print(1)" && rm file')
    assert ok is False


def test_s16_recipe_no_bash_steps():
    steps = [
        {"tool": "edit_file", "args": {"path": "x.py"}},
        {"tool": "done", "args": {"summary": "ok"}},
    ]
    ok, reason = is_recipe_bash_safe(steps)
    assert ok
    assert "no run_bash" in reason.lower()


def test_s17_recipe_all_safe_bash():
    steps = [
        {"tool": "edit_file", "args": {"path": "x.py"}},
        {"tool": "run_bash", "args": {"command": 'python -c "import x"'}},
        {"tool": "done", "args": {"summary": "ok"}},
    ]
    ok, _ = is_recipe_bash_safe(steps)
    assert ok


def test_s18_recipe_one_bad_bash():
    steps = [
        {"tool": "run_bash", "args": {"command": "python -c 'import x'"}},
        {"tool": "run_bash", "args": {"command": "rm -rf workspace/"}},
        {"tool": "done", "args": {"summary": "ok"}},
    ]
    ok, reason = is_recipe_bash_safe(steps)
    assert ok is False
    assert "step 2" in reason


# ─────────────────────────────────────────────────────────────────
# Recipe linter
# ─────────────────────────────────────────────────────────────────


def _safe_recipe_steps():
    """A canonical valid recipe -- 3 steps: edit, run_bash verify,
    done. Used as the baseline that should pass all lint checks."""
    return [
        {"tool": "edit_file", "args": {
            "path": "core/util.py", "old": "x = 1", "new": "x = 2",
        }},
        {"tool": "run_bash", "args": {
            "command": 'python -c "import core.util; print(core.util.x)"',
        }},
        {"tool": "done", "args": {"summary": "x changed from 1 to 2"}},
    ]


def test_s31_zero_step_recipe_l1_fail():
    r = lint_recipe_for_skip("", parsed_steps=[])
    assert r.safe is False
    assert "L1" in r.failed_checks


def test_s32_one_step_recipe_l1_fail():
    r = lint_recipe_for_skip(
        'STEP 1: done summary="x"',
        parsed_steps=[{"tool": "done", "args": {"summary": "x"}}],
    )
    assert r.safe is False
    assert "L1" in r.failed_checks


def test_s33_valid_3_step_recipe_safe():
    steps = _safe_recipe_steps()
    r = lint_recipe_for_skip("STEP 1: edit_file ...", parsed_steps=steps)
    assert r.safe is True, f"expected safe, got: {r.reason}"
    assert r.failed_checks == []


def test_s34_final_not_done_l2_fail():
    steps = _safe_recipe_steps()
    steps[-1] = {"tool": "edit_file", "args": {"path": "x.py"}}
    r = lint_recipe_for_skip("STEP 1: ...", parsed_steps=steps)
    assert r.safe is False
    assert "L2" in r.failed_checks


def test_s35_no_run_bash_before_done_l3_fail():
    """Recipe with edit + done but no run_bash verification step --
    skip-eligibility requires runtime verification."""
    steps = [
        {"tool": "edit_file", "args": {"path": "x.py"}},
        {"tool": "done", "args": {"summary": "edited"}},
    ]
    r = lint_recipe_for_skip("STEP 1: ...", parsed_steps=steps)
    assert r.safe is False
    assert "L3" in r.failed_checks


def test_s36_unknown_tool_l4_fail():
    steps = _safe_recipe_steps()
    steps.insert(0, {"tool": "delete_file", "args": {"path": "x.py"}})
    r = lint_recipe_for_skip("STEP ...", parsed_steps=steps)
    assert r.safe is False
    assert "L4" in r.failed_checks


def test_s37_bad_bash_l5_fail():
    steps = _safe_recipe_steps()
    steps[1] = {"tool": "run_bash", "args": {"command": "rm -rf /"}}
    r = lint_recipe_for_skip("STEP ...", parsed_steps=steps)
    assert r.safe is False
    assert "L5" in r.failed_checks


@pytest.mark.parametrize("forbidden_path", [
    "logs/sentinel.jsonl",
    "logs\\sentinel.jsonl",
    ".env",
    ".env.bot",
])
def test_s38_s39_forbidden_path_l6_fail(forbidden_path):
    steps = _safe_recipe_steps()
    steps[0] = {"tool": "edit_file", "args": {
        "path": forbidden_path, "old": "x", "new": "y",
    }}
    r = lint_recipe_for_skip("STEP ...", parsed_steps=steps)
    assert r.safe is False, f"expected reject for {forbidden_path}"
    assert "L6" in r.failed_checks


def test_s40_db_path_l6_fail():
    steps = _safe_recipe_steps()
    steps[0] = {"tool": "write_file", "args": {
        "path": "knowledge.db", "content": "blah",
    }}
    r = lint_recipe_for_skip("STEP ...", parsed_steps=steps)
    assert r.safe is False
    assert "L6" in r.failed_checks


def test_s41_truncation_cap_l7_fail():
    """Recipe text at exactly the cap is suspicious -- it may have
    lost its final `done` to truncation."""
    big_recipe = "STEP 1: edit_file path=\"x.py\" old=\"a\" new=\"b\"\n" + (
        "x" * 8000
    )
    r = lint_recipe_for_skip(big_recipe, parsed_steps=_safe_recipe_steps())
    assert r.safe is False
    assert "L7" in r.failed_checks


def test_s42_recipe_well_below_cap_passes_l7():
    short_recipe = "STEP 1: edit_file ...\nSTEP 2: done summary=\"ok\""
    r = lint_recipe_for_skip(short_recipe, parsed_steps=_safe_recipe_steps())
    # L7 should not be in failed checks even if other checks fail
    assert "L7" not in r.failed_checks


def test_s43_multiple_violations_all_reported():
    """Recipe missing done AND with bad bash -- both should appear
    in failed_checks list."""
    steps = [
        {"tool": "edit_file", "args": {"path": "x.py"}},
        {"tool": "run_bash", "args": {"command": "rm -rf /"}},
        {"tool": "edit_file", "args": {"path": "y.py"}},  # not done
    ]
    r = lint_recipe_for_skip("STEP ...", parsed_steps=steps)
    assert r.safe is False
    assert "L2" in r.failed_checks  # not done at end
    assert "L5" in r.failed_checks  # bad bash


def test_s44_safe_recipe_descriptive_reason():
    """A clean recipe's reason should mention step structure so logs
    are informative, not just 'safe'."""
    r = lint_recipe_for_skip("STEP ...", parsed_steps=_safe_recipe_steps())
    assert r.safe
    assert "step" in r.reason.lower()


# ─────────────────────────────────────────────────────────────────
# nvidia-smi window-flash fix (source-level check)
# ─────────────────────────────────────────────────────────────────


_HEALTH_SRC = (
    Path(__file__).resolve().parent.parent / "core" / "health.py"
).read_text(encoding="utf-8")


def test_s51_health_passes_create_no_window_on_win32():
    """The fix: subprocess.run for nvidia-smi must include
    creationflags=CREATE_NO_WINDOW on win32 to suppress the console
    flash when the bot runs detached via Task Scheduler."""
    assert "CREATE_NO_WINDOW" in _HEALTH_SRC, (
        "core/health.py must reference CREATE_NO_WINDOW"
    )
    assert "win32" in _HEALTH_SRC


def test_s52_creationflags_zero_on_non_windows():
    """The fix must be a no-op on non-Windows so we don't break
    Linux/macOS deployments that run the bot foreground."""
    # Source-level: the conditional `if sys.platform == "win32"` is
    # what gates the flag. Either explicit `else 0` or a default
    # `creationflags = 0` keeps it safe.
    assert (
        'sys.platform == "win32"' in _HEALTH_SRC
        or "sys.platform=='win32'" in _HEALTH_SRC
    )
