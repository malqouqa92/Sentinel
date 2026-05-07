"""Phase 14b ECC -- 10 scenarios stressing agentic graduation.

Each scenario simulates a different real-world /code-style situation
the graduation path might hit. All stubs avoid actual git/Qwen/Claude
calls so the suite stays fast and deterministic.

Scenarios:
  S01 -- pass rate exactly at threshold (1/3 = 33%) flips needs_reteach
  S02 -- pass rate just above threshold (2/3 = 67%) does NOT flip
  S03 -- mode dispatch: missing recipe + base_sha -> text-gen fallback
  S04 -- mode dispatch: missing base_sha + recipe -> text-gen fallback
  S05 -- very long recipe (5000+ chars) -> graduation handles
  S06 -- stepfed crashes mid-run -> graduation FAIL, finally still runs
  S07 -- error in MIDDLE step (not first) -> graduation FAIL with that step
  S08 -- syntax-check fail on changed files -> graduation FAIL
  S09 -- stash subprocess returns failure -> graduation still runs
  S10 -- multi-file recipe (3 edits + done) -> graduation PASS
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.knowledge_base import KnowledgeBase


# ─────────────────────────────────────────────────────────────────
# Shared fixtures + helpers
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def kb(tmp_path, monkeypatch):
    db_path = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    return KnowledgeBase(db_path=db_path)


def _seed_phase14b(kb, *, recipe="STEP 1: edit_file\nSTEP 2: done",
                   base_sha="abc1234", summary="P"):
    """Seed a pattern eligible for the agentic graduation path."""
    return kb.add_pattern(
        tags=["s"], problem_summary=summary,
        solution_code="diff",
        solution_pattern=recipe,
        explanation="seed",
        trace_id="SEN-seed",
        base_sha=base_sha,
    )


def _stub_agentic_pipeline(monkeypatch, *,
                           stepfed_result=None,
                           stepfed_raises=False,
                           stepfed_timeout=False,
                           reset_results=None,
                           syntax_ok=True,
                           syntax_err="",
                           stash_rc=0,
                           stash_out=b"No local changes to save"):
    """Patch git + stepfed + syntax helpers for an agentic graduation
    test. Default: stepfed succeeds with completed_via_done=True and
    one clean step. Override via kwargs.

    stepfed_result: dict to return from stepfed (else default OK shape).
    stepfed_raises: if True, stepfed raises RuntimeError.
    stepfed_timeout: if True, stepfed sleeps 2s (caller must set tight
        GRADUATION_TIMEOUT_S to actually trigger).
    reset_results: list of bools for successive reset calls (default
        [True, True] = both resets succeed).
    """
    import skills.code_assist as ca
    import core.qwen_agent as qa
    import asyncio as _aio

    if stepfed_result is None:
        stepfed_result = {
            "summary": "ok",
            "session": [
                {"step": 1, "tool": "edit_file",
                 "args": {}, "result": {"status": "ok"}},
                {"step": 2, "tool": "done",
                 "args": {}, "result": {"status": "ok"}},
            ],
            "steps": 2, "completed_via_done": True, "error": None,
        }
    if reset_results is None:
        reset_results = [True, True]

    state = {"reset_idx": 0, "stepfed_calls": 0, "syntax_calls": 0}

    async def fake_snapshot(trace_id):
        return "PRE_GRAD"

    async def fake_reset(sha, trace_id):
        idx = state["reset_idx"]
        state["reset_idx"] += 1
        if idx < len(reset_results):
            return reset_results[idx]
        return True

    def fake_stepfed(*a, **k):
        state["stepfed_calls"] += 1
        if stepfed_raises:
            raise RuntimeError("stepfed simulated crash")
        if stepfed_timeout:
            import time as _t
            _t.sleep(2.0)
        return stepfed_result

    async def fake_diff_stat(base):
        return " interfaces/x.py | 2 +-"

    async def fake_syntax(diff, trace_id):
        state["syntax_calls"] += 1
        return (syntax_ok, syntax_err)

    class FakeProc:
        def __init__(self, rc, out, err=b""):
            self.returncode = rc
            self._out = out
            self._err = err

        async def communicate(self):
            return (self._out, self._err)

    async def fake_subprocess_exec(*a, **k):
        return FakeProc(stash_rc, stash_out)

    monkeypatch.setattr(ca, "_git_snapshot", fake_snapshot)
    monkeypatch.setattr(ca, "_git_reset_hard", fake_reset)
    monkeypatch.setattr(ca, "_git_diff_stat", fake_diff_stat)
    monkeypatch.setattr(ca, "_verify_syntax_of_changed_files", fake_syntax)
    monkeypatch.setattr(qa, "run_agent_stepfed", fake_stepfed)
    monkeypatch.setattr(_aio, "create_subprocess_exec", fake_subprocess_exec)

    return state


# ─────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────


def test_s01_pass_rate_at_threshold_flips_flag(kb):
    """3 attempts, 1 pass -> 33% < 50% threshold -> needs_reteach=True."""
    pid = _seed_phase14b(kb)
    kb.record_solo_attempt(pid, True, "S01")    # 1/1 (above, no flip yet)
    kb.record_solo_attempt(pid, False, "S01")   # 1/2 (still <3 tries)
    a, p, flag = kb.record_solo_attempt(pid, False, "S01")  # 1/3 = 33%
    assert (a, p) == (3, 1)
    assert flag is True


def test_s02_pass_rate_above_threshold_does_not_flip(kb):
    """2 of 3 = 67% > 50% -> no flag."""
    pid = _seed_phase14b(kb)
    kb.record_solo_attempt(pid, True, "S02")
    kb.record_solo_attempt(pid, True, "S02")
    a, p, flag = kb.record_solo_attempt(pid, False, "S02")
    assert (a, p) == (3, 2)
    assert flag is False


def test_s03_dispatch_missing_recipe_falls_back_to_text_gen(kb, monkeypatch):
    """Pattern has base_sha but solution_pattern is empty -> text-gen
    path (agentic requires both)."""
    pid = kb.add_pattern(
        tags=[], problem_summary="P",
        solution_code="diff",
        solution_pattern="",   # empty recipe
        explanation="", trace_id="SEN-seed",
        base_sha="abc1234",
    )
    # Stub the text-gen path; assert mode comes back text_gen.
    import skills.code_assist as ca

    def fake_qwen(s, u, t, m):
        return '{"code": "def f():\\n    return 1\\nprint(f())"}'

    async def fake_validate(c, t):
        return (True, {"return_code": 0, "stdout": "1", "stderr": ""})

    monkeypatch.setattr(ca, "_qwen_generate", fake_qwen)
    monkeypatch.setattr(ca, "_validate_code", fake_validate)

    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=kb, model_id="qwen", trace_id="SEN-S03",
    ))
    assert out["mode"] == "text_gen"


def test_s04_dispatch_missing_base_sha_falls_back_to_text_gen(kb, monkeypatch):
    """Pattern has recipe but base_sha is empty/None -> text-gen."""
    pid = kb.add_pattern(
        tags=[], problem_summary="P",
        solution_code="diff",
        solution_pattern="STEP 1: done",
        explanation="", trace_id="SEN-seed",
        base_sha=None,
    )
    import skills.code_assist as ca

    def fake_qwen(s, u, t, m):
        return '{"code": "def f():\\n    return 1\\nprint(f())"}'

    async def fake_validate(c, t):
        return (True, {"return_code": 0, "stdout": "1", "stderr": ""})

    monkeypatch.setattr(ca, "_qwen_generate", fake_qwen)
    monkeypatch.setattr(ca, "_validate_code", fake_validate)

    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=kb, model_id="qwen", trace_id="SEN-S04",
    ))
    assert out["mode"] == "text_gen"


def test_s05_very_long_recipe_handled(kb, monkeypatch):
    """5000-char recipe must not crash the agentic path."""
    huge = "STEP 1: edit_file path=x.py old=A new=B\n" * 100  # ~4400 chars
    pid = _seed_phase14b(kb, recipe=huge)
    state = _stub_agentic_pipeline(monkeypatch)
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=kb, model_id="qwen", trace_id="SEN-S05",
    ))
    assert out["mode"] == "agentic"
    assert out["passed"] is True
    assert state["stepfed_calls"] == 1


def test_s06_stepfed_crash_marks_failed_runs_finally(kb, monkeypatch):
    """If stepfed raises an exception, graduation marks fail and the
    finally block still runs (cleanup reset)."""
    pid = _seed_phase14b(kb)
    state = _stub_agentic_pipeline(monkeypatch, stepfed_raises=True)
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=kb, model_id="qwen", trace_id="SEN-S06",
    ))
    assert out["mode"] == "agentic"
    assert out["passed"] is False
    assert "crashed" in out["stderr_excerpt"].lower()
    # Both resets ran (base_sha + cleanup).
    assert state["reset_idx"] == 2


def test_s07_error_in_middle_step_fails_graduation(kb, monkeypatch):
    """Error in step 2 of 3 (with completed_via_done=True) must FAIL.
    This is the bug class that hit pattern #54 -- step error masked
    by overall completion."""
    pid = _seed_phase14b(kb)
    state = _stub_agentic_pipeline(monkeypatch, stepfed_result={
        "summary": "completed despite error",
        "session": [
            {"step": 1, "tool": "edit_file",
             "args": {}, "result": {"status": "ok"}},
            {"step": 2, "tool": "edit_file",
             "args": {}, "result": {"error": "old string not found"}},
            {"step": 3, "tool": "done",
             "args": {}, "result": {"status": "ok"}},
        ],
        "steps": 3, "completed_via_done": True, "error": None,
    })
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=kb, model_id="qwen", trace_id="SEN-S07",
    ))
    assert out["passed"] is False
    assert "step 2" in out["stderr_excerpt"]
    assert "old string not found" in out["stderr_excerpt"]


def test_s08_syntax_check_failure_fails_graduation(kb, monkeypatch):
    """Stepfed completed cleanly but syntax check on changed files
    failed -> graduation FAIL."""
    pid = _seed_phase14b(kb)
    state = _stub_agentic_pipeline(
        monkeypatch,
        syntax_ok=False,
        syntax_err="SyntaxError: unterminated string literal at line 17",
    )
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=kb, model_id="qwen", trace_id="SEN-S08",
    ))
    assert out["passed"] is False
    assert "syntax" in out["stderr_excerpt"].lower()
    assert "line 17" in out["stderr_excerpt"]


def test_s09_stash_failure_does_not_crash_graduation(kb, monkeypatch):
    """If git stash returns non-zero (e.g. nothing to stash) the
    graduation must still run -- stash is best-effort, not required."""
    pid = _seed_phase14b(kb)
    state = _stub_agentic_pipeline(
        monkeypatch,
        stash_rc=1, stash_out=b"warning: stash push failed",
    )
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=kb, model_id="qwen", trace_id="SEN-S09",
    ))
    # Graduation still ran the stepfed path despite stash issue.
    assert out["mode"] == "agentic"
    assert state["stepfed_calls"] == 1
    assert out["passed"] is True


def test_s10_multi_step_recipe_passes(kb, monkeypatch):
    """A realistic recipe (3 edits + done) all succeeding produces a
    PASS verdict."""
    pid = _seed_phase14b(kb, recipe=(
        'STEP 1: edit_file path="a.py" old="x" new="y"\n'
        'STEP 2: edit_file path="b.py" old="x" new="y"\n'
        'STEP 3: write_file path="c.py" content="new"\n'
        'STEP 4: done summary="three edits done"'
    ))
    state = _stub_agentic_pipeline(monkeypatch, stepfed_result={
        "summary": "three edits + done",
        "session": [
            {"step": 1, "tool": "edit_file",
             "args": {}, "result": {"status": "ok"}},
            {"step": 2, "tool": "edit_file",
             "args": {}, "result": {"status": "ok"}},
            {"step": 3, "tool": "write_file",
             "args": {}, "result": {"status": "ok"}},
            {"step": 4, "tool": "done",
             "args": {}, "result": {"status": "ok"}},
        ],
        "steps": 4, "completed_via_done": True, "error": None,
    })
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="multi-edit task", code_context=None,
        kb=kb, model_id="qwen", trace_id="SEN-S10",
    ))
    assert out["mode"] == "agentic"
    assert out["passed"] is True
    assert out["solo_passes"] == 1
    assert "completed cleanly" in out["stderr_excerpt"].lower()
