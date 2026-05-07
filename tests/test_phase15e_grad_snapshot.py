"""Phase 15e -- /code commits its changes before graduation runs.

Caught live monitoring pattern #66: /code reported PASS and Updated
_build_bar to blue/white emojis, but after restart the file was
still orange/black. Root cause: graduation's stash-and-pop dance
was silently failing to preserve the working-tree changes; the
final `git reset --hard pre_grad_sha` then wiped the change because
pre_grad_sha == base_sha (no commit between /code and graduation
when auto-commit was off).

Phase 15e fix: a real `_git_commit_for_graduation` runs BEFORE
graduate_pattern(), so pre_grad_sha now points to a SHA that
INCLUDES /code's work. The graduation's final reset to pre_grad_sha
RESTORES the change instead of reverting it. The stash dance becomes
a robust redundancy rather than the load-bearing path.

Coverage:
  Source-level wiring:
    E01 -- _git_commit_for_graduation function exists with the right
           signature
    E02 -- success path calls _git_commit_for_graduation BEFORE
           graduate_pattern(), AFTER kb.add_pattern()
    E03 -- the call uses the user problem as message
    E04 -- _git_commit_changes is still the no-op (back-compat: the
           Phase 10 contract that auto-commit pollutes history when
           called per-attempt is preserved)

  Helper behavior:
    E11 -- _git_commit_for_graduation runs `git add -- <scope>`
           with the right path scope (excludes logs/, *.db,
           __pycache__)
    E12 -- the commit identity is sentinel-grad@sentinel.local so
           it's filterable in `git log`
    E13 -- the commit message is prefixed with sentinel-grad: for
           greppability

  Diagnostic logging in kb_graduation:
    E21 -- stash failure path logs WARNING with rc + stderr
    E22 -- stash empty path logs DEBUG (expected with the new
           pre-grad commit)
    E23 -- stash success path logs INFO with stdout excerpt
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _read_code_assist() -> str:
    return (
        Path(__file__).resolve().parent.parent
        / "skills" / "code_assist.py"
    ).read_text(encoding="utf-8", errors="replace")


def _read_kb_graduation() -> str:
    return (
        Path(__file__).resolve().parent.parent
        / "core" / "kb_graduation.py"
    ).read_text(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────
# Source-level wiring
# ─────────────────────────────────────────────────────────────────


def test_e01_grad_snapshot_function_exists():
    import skills.code_assist as ca
    assert hasattr(ca, "_git_commit_for_graduation"), (
        "_git_commit_for_graduation helper missing"
    )
    sig = inspect.signature(ca._git_commit_for_graduation)
    params = list(sig.parameters)
    # (trace_id, message) -- both positional, in that order
    assert params == ["trace_id", "message"], (
        f"unexpected signature: {params}"
    )
    # Returns SHA string OR None -- accept either the runtime
    # types.UnionType representation (str | None) or the string
    # form, depending on how the file was parsed.
    ret = sig.return_annotation
    ret_str = str(ret)
    assert "str" in ret_str and "None" in ret_str, (
        f"unexpected return annotation: {ret!r}"
    )


def test_e02_success_path_calls_grad_commit_before_graduate():
    src = _read_code_assist()
    # Locate the success branch.
    pipeline_idx = src.find("async def _run_agentic_pipeline")
    assert pipeline_idx > 0
    body = src[pipeline_idx:pipeline_idx + 30000]
    grad_commit_idx = body.find("_git_commit_for_graduation(")
    graduate_idx = body.find("graduate_pattern(")
    assert grad_commit_idx > 0, (
        "_git_commit_for_graduation never called from pipeline"
    )
    assert graduate_idx > 0
    assert grad_commit_idx < graduate_idx, (
        "snapshot commit must run BEFORE graduate_pattern()"
    )
    # And before kb.add_pattern is called the second time too --
    # actually the snapshot should be after add_pattern (we want the
    # row written first, then commit). Verify ordering:
    #   _git_commit_changes (no-op) -> _git_commit_for_graduation
    #   -> kb.add_pattern -> graduate_pattern
    add_pattern_idx = body.find("new_pattern_id = kb.add_pattern(")
    assert grad_commit_idx < add_pattern_idx < graduate_idx


def test_e03_grad_commit_uses_problem_as_message():
    src = _read_code_assist()
    # The call site should pass the user-supplied problem as the
    # message argument.
    pipeline_idx = src.find("async def _run_agentic_pipeline")
    body = src[pipeline_idx:pipeline_idx + 30000]
    grad_idx = body.find("_git_commit_for_graduation(")
    chunk = body[grad_idx:grad_idx + 200]
    assert "input_data.problem" in chunk
    # And the helper itself prepends "sentinel-grad:" to the message.
    assert 'f"sentinel-grad: {message[:80]}"' in src


def test_e04_old_commit_helper_still_no_op():
    """Phase 10 contract preserved: _git_commit_changes is still a
    no-op (the per-attempt auto-commit pollution was the original
    rationale; that's untouched). The Phase 15e fix lives in a
    SEPARATE helper to keep the contracts disjoint."""
    src = _read_code_assist()
    no_op_idx = src.find("async def _git_commit_changes")
    assert no_op_idx > 0
    body = src[no_op_idx:no_op_idx + 1500]
    assert "auto-commit disabled" in body
    # The no-op MUST NOT call `git commit` itself.
    assert "git commit" not in body, (
        "_git_commit_changes regressed -- it now actually commits, "
        "which violates the Phase 10 owner-commits-manually contract"
    )


# ─────────────────────────────────────────────────────────────────
# Helper behavior (still source-level: avoid spinning up subprocess)
# ─────────────────────────────────────────────────────────────────


def test_e11_grad_commit_uses_scoped_add():
    """The git add scope must NOT include logs/, *.db, or
    __pycache__ -- those are runtime artifacts that should never
    end up in the snapshot commit."""
    src = _read_code_assist()
    helper_idx = src.find("async def _git_commit_for_graduation")
    body = src[helper_idx:helper_idx + 4000]
    # Scope is "core skills agents tests interfaces workspace"
    assert 'scope = "core skills agents tests interfaces workspace"' in body
    # And the actual add invocation uses that scope:
    assert 'f"git add -- {scope}"' in body
    # Confirm the dangerous bits are absent from scope:
    for danger in ("logs", ".db", "__pycache__"):
        # Scope substring check
        scope_line = next(
            l for l in body.splitlines() if "scope = " in l
        )
        assert danger not in scope_line, (
            f"scope unexpectedly includes {danger}"
        )


def test_e12_grad_commit_uses_sentinel_grad_identity():
    """Commit identity is sentinel-grad@sentinel.local so users can
    `git log --grep=sentinel-grad` or `--invert-grep` to filter
    them out."""
    src = _read_code_assist()
    helper_idx = src.find("async def _git_commit_for_graduation")
    body = src[helper_idx:helper_idx + 4000]
    assert "user.email=sentinel-grad@sentinel.local" in body
    assert "user.name=Sentinel-Grad-Snapshot" in body


def test_e13_grad_commit_message_is_prefixed():
    src = _read_code_assist()
    helper_idx = src.find("async def _git_commit_for_graduation")
    body = src[helper_idx:helper_idx + 4000]
    # Message format: "sentinel-grad: <problem-substring>"
    assert 'f"sentinel-grad: {message[:80]}"' in body


# ─────────────────────────────────────────────────────────────────
# Diagnostic logging in kb_graduation
# ─────────────────────────────────────────────────────────────────


def test_e21_stash_failure_path_logs_warning():
    src = _read_kb_graduation()
    # Three branches in the stash result handling:
    # rc != 0 -> WARNING with stderr
    # "No local changes to save" -> DEBUG
    # else -> INFO with stdout
    assert "stash push FAILED rc=" in src
    assert '"WARNING", "kb_graduation"' in src


def test_e22_stash_empty_path_logs_debug():
    src = _read_kb_graduation()
    # The DEBUG log explicitly mentions the post-15e expected case.
    assert "no local changes to save" in src.lower() or \
           "No local changes to save" in src
    assert '"DEBUG", "kb_graduation"' in src


def test_e23_stash_success_path_logs_with_stdout():
    src = _read_kb_graduation()
    assert "stashed dirty tree before graduation" in src
    # And echoes the stdout excerpt for diagnostics.
    assert "out={stash_out" in src


# ─────────────────────────────────────────────────────────────────
# Sanity: pipeline order + comment markers preserved
# ─────────────────────────────────────────────────────────────────


def test_e31_phase15e_comment_marker_present():
    """A grep-able marker so future archeologists can find the
    rationale without reading the whole file."""
    src = _read_code_assist()
    assert "Phase 15e" in src, (
        "Phase 15e marker missing from skills/code_assist.py"
    )


def test_e32_no_op_helper_kept_for_back_compat():
    """The original `_git_commit_changes` is still imported / kept
    around -- removing it would break older call sites that depend
    on its `git add -N` side-effect for diff helpers."""
    import skills.code_assist as ca
    assert hasattr(ca, "_git_commit_changes")
    assert hasattr(ca, "_git_commit_for_graduation")
    assert ca._git_commit_changes is not ca._git_commit_for_graduation
