"""Phase 17f -- reviewer recipe-promise verification + corrective tool-sig discipline.

Trigger: 2026-05-07 ~01:42-01:48Z chain run on /qcode prompt.
- Child 1 PASSED but its core/config.py edit silently no-op'd.
  Reviewer Read'd skills/qcode_assist.py (the new file) and confirmed
  it, but never verified the recipe's CLAIM that core/config.py was
  modified. /qcode missing from COMMAND_AGENT_MAP.
- Child 2 attempt 2 wrote `edit_file path="x" text="..."` instead
  of `edit_file path="x" old="..." new="..."`. 5 of 6 STEPs failed
  with "missing 2 required positional arguments" or "unexpected
  keyword argument 'text'". Claude was confusing tool signatures
  under the high-pressure corrective prompt.

Two source-level groups:
  R -- reviewer prompt now mandates recipe-promise verification
  C -- corrective prompt now has explicit tool-signature discipline
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent


# ============================================================
# Group R: reviewer recipe-promise verification
# ============================================================


def test_r01_reviewer_mentions_recipe_promise_verification():
    """Reviewer system prompt must explicitly reference verifying
    each recipe step's claimed change against actual file state."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    review_idx = src.find("async def _claude_review")
    assert review_idx > 0
    body = src[review_idx:review_idx + 4000]
    # Key load-bearing phrase
    assert "RECIPE-PROMISE VERIFICATION" in body or "recipe-promise verification" in body.lower()


def test_r02_reviewer_calls_out_silent_noop_failure_mode():
    """Reviewer must be told that a step which silently no-op'd
    (anchor mismatch / wrong path / malformed args) is a FAIL even
    if other steps succeeded."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    review_idx = src.find("async def _claude_review")
    body = src[review_idx:review_idx + 4000]
    # Must mention silent no-ops + the verdict consequence
    assert "silent no-op" in body.lower() or "Silent no-op" in body
    assert "FAIL" in body or "fail" in body


def test_r03_reviewer_must_check_each_recipe_step():
    """Prompt must use 'each' (or 'every') to scope verification
    to ALL recipe steps, not just one."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    review_idx = src.find("async def _claude_review")
    body = src[review_idx:review_idx + 4000]
    assert "EACH" in body or "every" in body.lower()


def test_r04_reviewer_reasoning_format_includes_step_number():
    """The example reasoning in the prompt must show citing a
    SPECIFIC recipe step + its claim being verified."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    review_idx = src.find("async def _claude_review")
    body = src[review_idx:review_idx + 4000]
    # Should include 'step' + concrete example
    assert "Recipe step" in body or "recipe step" in body.lower()


def test_r05_reviewer_budget_increased():
    """Verification of each recipe step requires more tool calls
    than the old 'just verify the specific claim' bound."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    review_idx = src.find("async def _claude_review")
    body = src[review_idx:review_idx + 4000]
    # Expect a larger budget number
    assert "~7 calls" in body or "7 calls" in body or "10 calls" in body


def test_r06_reviewer_cross_references_tool_trace():
    """Trace shows ERR for failed steps; prompt must instruct
    Claude to look at it."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    review_idx = src.find("async def _claude_review")
    body = src[review_idx:review_idx + 4000]
    assert "TOOL TRACE" in body
    # And the prompt body must reference cross-checking it
    assert "trace" in body.lower()


# ============================================================
# Group C: corrective tool-signature discipline
# ============================================================


def test_c01_corrective_lists_exact_tool_signatures():
    """Corrective system prompt must include explicit signatures
    with arg names so Claude doesn't improvise."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    # Find CORRECTIVE_SYSTEM constant
    idx = src.find("CORRECTIVE_SYSTEM = (")
    assert idx > 0
    body = src[idx:idx + 4000]
    # All 6 tool signatures present
    assert "read_file(path=" in body
    assert "write_file(path=" in body
    assert "edit_file(path=" in body
    assert "run_bash(command=" in body
    assert "done(summary=" in body


def test_c02_corrective_warns_against_text_arg():
    """The exact failure mode (`text=`) must be called out as an
    anti-pattern so Claude doesn't repeat it."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = src.find("CORRECTIVE_SYSTEM = (")
    body = src[idx:idx + 4000]
    assert "text=" in body
    assert "TOOL-SIGNATURE DISCIPLINE" in body or "tool-signature discipline" in body.lower()


def test_c03_corrective_handles_missing_args_failure_mode():
    """Prompt must mention the 'missing 2 required positional
    arguments' failure mode + recommend write_file as the recovery.
    Import the constant so adjacent string concatenation across
    source lines is collapsed."""
    from skills.code_assist import CORRECTIVE_SYSTEM
    assert "missing 2 required positional arguments" in CORRECTIVE_SYSTEM
    assert "write_file" in CORRECTIVE_SYSTEM


def test_c04_corrective_signature_list_uses_exact_arg_names():
    """The arg-name examples must be EXACTLY old/new/path/content/command/summary,
    not abbreviations -- otherwise Claude could mimic the wrong forms."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = src.find("CORRECTIVE_SYSTEM = (")
    body = src[idx:idx + 4000]
    # Each arg name in its expected position
    assert "edit_file(path=..., old=..., new=...)" in body
    assert "write_file(path=..., content=...)" in body


def test_c05_corrective_cites_live_failure_for_credibility():
    """Including the actual date+UTC of the live failure makes the
    'don't use text=' warning concrete, not abstract."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = src.find("CORRECTIVE_SYSTEM = (")
    body = src[idx:idx + 4000]
    assert "2026-05-07" in body or "2026-05" in body


# ============================================================
# Group I: import + integration sanity
# ============================================================


def test_i01_imports_clean():
    from skills import code_assist  # noqa: F401


def test_i02_review_function_still_returns_dict_shape():
    """The review function signature + return shape contract is
    unchanged -- caller still expects {"verdict": "pass"|"fail"|
    "unknown", "reasoning": "..."}. A breaking signature change
    would break the agentic pipeline."""
    import inspect
    from skills.code_assist import _claude_review
    sig = inspect.signature(_claude_review)
    # Required params: problem, recipe, qwen_summary, qwen_session, git_diff, trace_id
    assert "problem" in sig.parameters
    assert "recipe" in sig.parameters
    assert "trace_id" in sig.parameters


def test_i03_corrective_function_signature_unchanged():
    import inspect
    from skills.code_assist import _claude_corrective_teach
    sig = inspect.signature(_claude_corrective_teach)
    assert "problem" in sig.parameters
    assert "trace_id" in sig.parameters
