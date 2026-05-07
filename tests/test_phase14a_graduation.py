"""Phase 14a -- KB graduation test transfer-verification.

ECC. No GPU, no real LLM, no executor subprocess (we stub both
``_qwen_generate`` and ``_validate_code``).

Coverage:
  Schema migration (idempotent ALTER):
    01 -- fresh DB has the new columns + partial indexes
    02 -- existing DB without columns gets them on KB() init
    03 -- second init is a no-op (no duplicate-column error)

  KB methods:
    11 -- get_pattern returns full row including new columns
    12 -- get_pattern on missing id returns None
    13 -- record_solo_attempt(passed=True) increments both counters
    14 -- record_solo_attempt(passed=False) increments only attempts
    15 -- record_solo_attempt below MIN_TRIES doesn't flip flag
    16 -- record_solo_attempt at MIN_TRIES with low pass rate flips
    17 -- record_solo_attempt at MIN_TRIES with high pass rate doesn't flip
    18 -- needs_reteach is one-way: passing later doesn't unflip
    19 -- clear_needs_reteach(id) resets the flag
    20 -- list_needs_reteach uses the partial index
    21 -- list_stale: NULL last_verified_at is treated as stale
    22 -- list_stale: row newer than cutoff is excluded
    23 -- graduation_stats aggregates correctly

  graduate_pattern integration:
    31 -- happy path: Qwen returns working code -> passed=True,
          counters incremented, no flag
    32 -- Qwen returns garbage -> passed=False, counter still
          incremented
    33 -- Qwen raises -> passed=False, no crash
    34 -- pattern_id missing -> graceful fail
    35 -- code passes exec but fails quality gate -> still treated
          as graduation FAIL (no false-positive reward)
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.knowledge_base import KnowledgeBase, _connect


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_kb(tmp_path, monkeypatch):
    """Brand-new KB at a temp path."""
    db_path = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    return KnowledgeBase(db_path=db_path)


def _seed_pattern(kb: KnowledgeBase, summary: str = "test problem") -> int:
    return kb.add_pattern(
        tags=["test"],
        problem_summary=summary,
        solution_code="print(2+2)",
        solution_pattern="just print",
        explanation="trivial",
        trace_id="SEN-test",
    )


# ─────────────────────────────────────────────────────────────────
# Schema migration
# ─────────────────────────────────────────────────────────────────


def test_g_01_fresh_db_has_grad_columns(fresh_kb):
    conn = _connect(fresh_kb.db_path)
    try:
        cols = {r["name"] for r in conn.execute(
            "PRAGMA table_info(knowledge)"
        ).fetchall()}
        idx = {r["name"] for r in conn.execute(
            "PRAGMA index_list(knowledge)"
        ).fetchall()}
    finally:
        conn.close()
    for c in (
        "solo_attempts", "solo_passes", "last_verified_at", "needs_reteach",
    ):
        assert c in cols, f"missing column: {c}"
    assert "idx_knowledge_needs_reteach" in idx
    assert "idx_knowledge_last_verified" in idx


def test_g_02_pre_phase14_db_gets_migrated(tmp_path, monkeypatch):
    """Simulate an existing pre-Phase14 KB.db that lacks the new
    columns. KnowledgeBase() init should ALTER it cleanly."""
    db_path = tmp_path / "old.db"
    # Manually create an old-shape table with just the Phase 8 columns.
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            tags TEXT NOT NULL,
            problem_summary TEXT NOT NULL,
            solution_code TEXT,
            solution_pattern TEXT,
            explanation TEXT NOT NULL,
            source_trace_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            usage_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # Existing row from "before" Phase 14a.
    conn.execute(
        "INSERT INTO knowledge (category, tags, problem_summary, "
        "solution_code, solution_pattern, explanation, source_trace_id, "
        "created_at) VALUES ('pattern', 'x', 'old problem', 'old code', "
        "'old pattern', 'old expl', 'SEN-old', '2026-01-01T00:00:00')"
    )
    conn.commit()
    conn.close()
    # Now KnowledgeBase init must ALTER the table.
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    kb = KnowledgeBase(db_path=db_path)
    pattern = kb.get_pattern(1)
    assert pattern is not None
    assert pattern.solo_attempts == 0
    assert pattern.solo_passes == 0
    assert pattern.last_verified_at is None
    assert pattern.needs_reteach is False


def test_g_03_double_init_is_idempotent(fresh_kb):
    """Second KnowledgeBase(db_path=...) on the same file must not
    error on duplicate ADD COLUMN."""
    KnowledgeBase(db_path=fresh_kb.db_path)
    KnowledgeBase(db_path=fresh_kb.db_path)


# ─────────────────────────────────────────────────────────────────
# get_pattern + record_solo_attempt
# ─────────────────────────────────────────────────────────────────


def test_g_11_get_pattern_includes_new_fields(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    p = fresh_kb.get_pattern(pid)
    assert p is not None
    assert p.solo_attempts == 0
    assert p.solo_passes == 0
    assert p.last_verified_at is None
    assert p.needs_reteach is False


def test_g_12_get_pattern_missing(fresh_kb):
    assert fresh_kb.get_pattern(99999) is None


def test_g_13_record_pass_increments_both(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    a, p, flag = fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    assert (a, p, flag) == (1, 1, False)
    p2 = fresh_kb.get_pattern(pid)
    assert p2.solo_attempts == 1 and p2.solo_passes == 1
    assert p2.last_verified_at is not None


def test_g_14_record_fail_increments_only_attempts(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    a, p, flag = fresh_kb.record_solo_attempt(pid, False, "SEN-test")
    assert (a, p, flag) == (1, 0, False)


def test_g_15_below_min_tries_no_flip(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    # Two failures, but MIN_TRIES is 3 -> no flag yet.
    fresh_kb.record_solo_attempt(pid, False, "SEN-test")
    a, p, flag = fresh_kb.record_solo_attempt(pid, False, "SEN-test")
    assert (a, p, flag) == (2, 0, False)


def test_g_16_min_tries_low_rate_flips(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    fresh_kb.record_solo_attempt(pid, False, "SEN-test")
    fresh_kb.record_solo_attempt(pid, False, "SEN-test")
    a, p, flag = fresh_kb.record_solo_attempt(pid, False, "SEN-test")
    assert (a, p, flag) == (3, 0, True)


def test_g_17_min_tries_high_rate_does_not_flip(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    a, p, flag = fresh_kb.record_solo_attempt(pid, False, "SEN-test")
    # 2/3 = 67% > 50% threshold, no flip.
    assert (a, p, flag) == (3, 2, False)


def test_g_18_flag_is_one_way(fresh_kb):
    """Once flagged, more passes don't auto-unflip -- only the
    explicit clear_needs_reteach() does."""
    pid = _seed_pattern(fresh_kb)
    for _ in range(3):
        fresh_kb.record_solo_attempt(pid, False, "SEN-test")
    assert fresh_kb.get_pattern(pid).needs_reteach is True
    # Even after 5 passes, flag stays
    for _ in range(5):
        fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    assert fresh_kb.get_pattern(pid).needs_reteach is True


def test_g_19_clear_needs_reteach_resets_flag(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    for _ in range(3):
        fresh_kb.record_solo_attempt(pid, False, "SEN-test")
    assert fresh_kb.get_pattern(pid).needs_reteach is True
    assert fresh_kb.clear_needs_reteach(pid) is True
    assert fresh_kb.get_pattern(pid).needs_reteach is False


def test_g_20_list_needs_reteach_uses_partial_index(fresh_kb):
    pid_bad = _seed_pattern(fresh_kb, "bad pattern")
    pid_ok = _seed_pattern(fresh_kb, "ok pattern")
    for _ in range(3):
        fresh_kb.record_solo_attempt(pid_bad, False, "SEN-test")
    for _ in range(3):
        fresh_kb.record_solo_attempt(pid_ok, True, "SEN-test")
    flagged = fresh_kb.list_needs_reteach()
    assert len(flagged) == 1
    assert flagged[0].id == pid_bad


def test_g_21_list_stale_includes_unverified(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    stale = fresh_kb.list_stale(days=1)
    # Never-verified pattern shows up in stale list.
    assert any(s.id == pid for s in stale)


def test_g_22_list_stale_excludes_recent(fresh_kb):
    pid = _seed_pattern(fresh_kb)
    fresh_kb.record_solo_attempt(pid, True, "SEN-test")
    # Just verified -> not stale at 30 days.
    stale = fresh_kb.list_stale(days=30)
    assert not any(s.id == pid for s in stale)


def test_g_23_graduation_stats_aggregates(fresh_kb):
    pid_a = _seed_pattern(fresh_kb, "A")
    pid_b = _seed_pattern(fresh_kb, "B")
    fresh_kb.record_solo_attempt(pid_a, True, "SEN-test")
    fresh_kb.record_solo_attempt(pid_a, True, "SEN-test")
    fresh_kb.record_solo_attempt(pid_b, False, "SEN-test")
    s = fresh_kb.graduation_stats()
    assert s["total_patterns"] == 2
    assert s["verified_patterns"] == 2
    assert s["solo_attempts"] == 3
    assert s["solo_passes"] == 2
    assert s["needs_reteach"] == 0
    assert abs(s["solo_pass_rate"] - 0.667) < 0.01


# ─────────────────────────────────────────────────────────────────
# graduate_pattern integration
# ─────────────────────────────────────────────────────────────────


def _stub_qwen_validate(monkeypatch, *,
                        qwen_text: str = '{"code": "print(42)"}',
                        qwen_raises: bool = False,
                        exec_passes: bool = True,
                        exec_stdout: str = "42",
                        exec_stderr: str = ""):
    """Patch the Qwen call + executor in skills.code_assist before
    graduate_pattern's late imports resolve. Returns a dict of call
    counters for assertions."""
    import skills.code_assist as ca

    counters = {"qwen": 0, "exec": 0}

    def fake_qwen(system, user, trace_id, model_id):
        counters["qwen"] += 1
        if qwen_raises:
            raise RuntimeError("simulated qwen crash")
        return qwen_text

    async def fake_validate(code, trace_id):
        counters["exec"] += 1
        return (
            exec_passes,
            {"return_code": 0 if exec_passes else 1,
             "stdout": exec_stdout, "stderr": exec_stderr},
        )

    monkeypatch.setattr(ca, "_qwen_generate", fake_qwen)
    monkeypatch.setattr(ca, "_validate_code", fake_validate)
    return counters


def test_g_31_graduate_happy_path(fresh_kb, monkeypatch):
    pid = _seed_pattern(fresh_kb, "compute 2+2")
    counters = _stub_qwen_validate(
        monkeypatch,
        # Real-looking code -- has structural markers (def + return).
        qwen_text='{"code": "def add(a, b):\\n    return a + b\\nprint(add(2,2))"}',
        exec_passes=True,
    )
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="compute 2+2",
        code_context=None, kb=fresh_kb,
        model_id="qwen2.5-coder:3b", trace_id="SEN-test-31",
    ))
    assert out["passed"] is True
    assert out["solo_attempts"] == 1
    assert out["solo_passes"] == 1
    assert out["needs_reteach"] is False
    assert out["mode"] == "text_gen"  # no base_sha -> legacy path
    assert counters["qwen"] == 1
    assert counters["exec"] == 1


def test_g_32_graduate_garbage_qwen_response(fresh_kb, monkeypatch):
    pid = _seed_pattern(fresh_kb)
    _stub_qwen_validate(monkeypatch, qwen_text="not json at all")
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=fresh_kb, model_id="qwen", trace_id="SEN-test-32",
    ))
    assert out["passed"] is False
    assert out["solo_attempts"] == 1
    assert out["solo_passes"] == 0


def test_g_33_graduate_qwen_raises(fresh_kb, monkeypatch):
    pid = _seed_pattern(fresh_kb)
    _stub_qwen_validate(monkeypatch, qwen_raises=True)
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=fresh_kb, model_id="qwen", trace_id="SEN-test-33",
    ))
    assert out["passed"] is False
    assert out["solo_attempts"] == 1
    assert "qwen failed" in out["stderr_excerpt"].lower()


def test_g_34_graduate_missing_pattern(fresh_kb, monkeypatch):
    _stub_qwen_validate(monkeypatch)
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=99999, problem="x", code_context=None,
        kb=fresh_kb, model_id="qwen", trace_id="SEN-test-34",
    ))
    assert out["passed"] is False
    assert out["solo_attempts"] == 0  # never recorded


def test_g_36_graduation_times_out_at_60s(fresh_kb, monkeypatch):
    """Phase 14a polish: a Qwen call that doesn't return within
    GRADUATION_TIMEOUT_S must short-circuit with passed=False rather
    than block the /code reply for the full OllamaClient default
    (900s). Reproduces what hit pattern #52 in production."""
    pid = _seed_pattern(fresh_kb)
    import skills.code_assist as ca
    import core.kb_graduation as kg

    # Override the timeout to 0.1s so the test is fast.
    monkeypatch.setattr(kg, "GRADUATION_TIMEOUT_S", 0.1)

    def slow_qwen(system, user, trace_id, model_id):
        # Sleep longer than the timeout to force the wait_for to fire.
        import time as _t
        _t.sleep(2.0)
        return '{"code": "print(0)"}'

    async def stub_validate(code, trace_id):
        return (True, {"return_code": 0, "stdout": "", "stderr": ""})

    monkeypatch.setattr(ca, "_qwen_generate", slow_qwen)
    monkeypatch.setattr(ca, "_validate_code", stub_validate)

    out = asyncio.run(kg.graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=fresh_kb, model_id="qwen", trace_id="SEN-test-36",
    ))
    assert out["passed"] is False
    assert out["solo_attempts"] == 1
    assert out["solo_passes"] == 0
    assert "timed out" in out["stderr_excerpt"].lower()
    # And the duration should be CLOSE to the timeout, not 2s+.
    assert out["duration_s"] < 1.0, (
        f"expected fast bail, got {out['duration_s']}s"
    )


def test_g_40_dispatcher_picks_text_gen_when_no_base_sha(fresh_kb, monkeypatch):
    """Old patterns (no base_sha) must route to text-gen graduation,
    not the agentic path which requires base_sha for the tree reset."""
    pid = _seed_pattern(fresh_kb)  # no base_sha
    counters = _stub_qwen_validate(
        monkeypatch,
        qwen_text='{"code": "def f():\\n    return 1\\nprint(f())"}',
        exec_passes=True,
    )
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=fresh_kb, model_id="qwen", trace_id="SEN-test-40",
    ))
    assert out["mode"] == "text_gen"
    # text-gen path called Qwen + executor
    assert counters["qwen"] == 1
    assert counters["exec"] == 1


def test_g_41_dispatcher_picks_agentic_when_base_sha_present(
    fresh_kb, monkeypatch,
):
    """A pattern with both base_sha + recipe must route to the
    agentic path. We stub git + stepfed + Claude review so the test
    doesn't actually touch the working tree."""
    # Seed pattern with full Phase 14b metadata.
    pid = fresh_kb.add_pattern(
        tags=["phase14b"],
        problem_summary="agentic-eligible pattern",
        solution_code="(diff text)",
        solution_pattern='STEP 1: edit_file path="x.py" old="a" new="b"\n'
                        'STEP 2: done summary="ok"',
        explanation="seeded for test",
        trace_id="SEN-seed",
        base_sha="abc1234",
    )
    # Stub the agentic-graduation dependencies before the late
    # imports inside _graduate_via_agentic resolve.
    import skills.code_assist as ca
    import core.qwen_agent as qa
    calls = {"snapshot": 0, "reset": 0, "stepfed": 0, "syntax": 0}

    async def fake_snapshot(trace_id):
        calls["snapshot"] += 1
        return "PRE_GRAD_SHA"

    async def fake_reset(sha, trace_id):
        calls["reset"] += 1
        return True

    def fake_stepfed(problem, recipe, trace_id, model_id):
        calls["stepfed"] += 1
        return {
            "summary": "stub agent ran 1 step",
            "session": [
                {"step": 1, "tool": "edit_file",
                 "args": {}, "result": {"status": "ok"}},
            ],
            "steps": 1,
            "completed_via_done": True,
            "error": None,
        }

    async def fake_diff_stat(base):
        return " x.py | 1 +"

    async def fake_syntax(diff_stat, trace_id):
        calls["syntax"] += 1
        return (True, "")

    monkeypatch.setattr(ca, "_git_snapshot", fake_snapshot)
    monkeypatch.setattr(ca, "_git_reset_hard", fake_reset)
    monkeypatch.setattr(ca, "_git_diff_stat", fake_diff_stat)
    monkeypatch.setattr(
        ca, "_verify_syntax_of_changed_files", fake_syntax,
    )
    monkeypatch.setattr(qa, "run_agent_stepfed", fake_stepfed)
    # Also stub git stash subprocess so we don't actually touch the
    # working tree.
    import asyncio as _aio

    class FakeProc:
        def __init__(self, rc=0, out=b"No local changes to save", err=b""):
            self.returncode = rc
            self._out = out
            self._err = err

        async def communicate(self):
            return (self._out, self._err)

    async def fake_subprocess_exec(*args, **kwargs):
        # Pretend stash always succeeds with "no changes".
        return FakeProc()

    monkeypatch.setattr(
        _aio, "create_subprocess_exec", fake_subprocess_exec,
    )

    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=fresh_kb, model_id="qwen", trace_id="SEN-test-41",
    ))
    assert out["mode"] == "agentic"
    assert out["passed"] is True
    assert out["solo_passes"] == 1
    # Verify the pipeline order: snapshot -> reset(base) -> stepfed
    # -> syntax check, then reset(pre_grad) in finally. NO Claude
    # review (Phase 14b bugfix: deterministic check from stepfed
    # output, not LLM review of stale tree state).
    assert calls["snapshot"] == 1
    assert calls["stepfed"] == 1
    assert calls["syntax"] == 1
    # Reset called twice: once to base_sha, once to pre_grad_sha.
    assert calls["reset"] == 2


def test_g_42_agentic_stepfed_error_marks_failed(fresh_kb, monkeypatch):
    """Phase 14b bugfix: a stepfed step with an error in its result
    must FAIL graduation even though completed_via_done=True. This
    is the exact bug pattern #54 hit -- stepfed step 1 errored
    ('old string not found') but Claude reviewed leftover state
    and falsely said pass."""
    pid = fresh_kb.add_pattern(
        tags=[], problem_summary="pat", solution_code="",
        solution_pattern="STEP 1: edit_file path=x.py old=A new=B\n"
                        "STEP 2: done summary=ok",
        explanation="", trace_id="SEN-seed",
        base_sha="deadbee",
    )
    import skills.code_assist as ca
    import core.qwen_agent as qa
    import asyncio as _aio

    async def ok_async(*a, **k):
        return True

    async def fake_snapshot(trace_id):
        return "PRE"

    async def fake_diff(base):
        return "diff"

    async def fake_syntax(*a):
        return (True, "")

    def fake_stepfed_with_error(*a, **k):
        # Simulates exactly what hit pattern #54.
        return {
            "summary": "executed but errored on step 1",
            "session": [
                {"step": 1, "tool": "edit_file",
                 "args": {}, "result": {"error": "old string not found in file"}},
                {"step": 2, "tool": "done",
                 "args": {}, "result": {"status": "ok"}},
            ],
            "steps": 2,
            "completed_via_done": True,
            "error": None,
        }

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"No local changes to save", b"")

    async def fake_subprocess_exec(*a, **k):
        return FakeProc()

    monkeypatch.setattr(ca, "_git_snapshot", fake_snapshot)
    monkeypatch.setattr(ca, "_git_reset_hard", ok_async)
    monkeypatch.setattr(ca, "_git_diff_stat", fake_diff)
    monkeypatch.setattr(
        ca, "_verify_syntax_of_changed_files", fake_syntax,
    )
    monkeypatch.setattr(qa, "run_agent_stepfed", fake_stepfed_with_error)
    monkeypatch.setattr(_aio, "create_subprocess_exec", fake_subprocess_exec)

    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=fresh_kb, model_id="qwen", trace_id="SEN-test-42",
    ))
    assert out["mode"] == "agentic"
    assert out["passed"] is False, (
        f"step error must fail graduation; got result={out!r}"
    )
    assert out["solo_passes"] == 0
    assert "errored" in out["stderr_excerpt"].lower()
    assert "old string not found" in out["stderr_excerpt"].lower()


def test_g_43_agentic_reset_failure_handled(fresh_kb, monkeypatch):
    """If reset to base_sha fails, graduation marks fail without
    crashing. Tree-state finally still runs."""
    pid = fresh_kb.add_pattern(
        tags=[], problem_summary="pat", solution_code="",
        solution_pattern="STEP 1: done summary=ok",
        explanation="", trace_id="SEN-seed",
        base_sha="badsha",
    )
    import skills.code_assist as ca
    import asyncio as _aio

    async def fake_snapshot(trace_id):
        return "PRE"

    reset_calls = {"n": 0}

    async def fake_reset(sha, trace_id):
        reset_calls["n"] += 1
        # First reset (to base_sha) fails; second reset (cleanup) ok.
        return reset_calls["n"] != 1

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"No local changes to save", b"")

    async def fake_subprocess_exec(*a, **k):
        return FakeProc()

    monkeypatch.setattr(ca, "_git_snapshot", fake_snapshot)
    monkeypatch.setattr(ca, "_git_reset_hard", fake_reset)
    monkeypatch.setattr(_aio, "create_subprocess_exec", fake_subprocess_exec)

    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=fresh_kb, model_id="qwen", trace_id="SEN-test-43",
    ))
    assert out["passed"] is False
    assert out["mode"] == "agentic"
    # Cleanup reset should still have been attempted.
    assert reset_calls["n"] == 2


def test_g_44_solution_code_now_stores_diff_text():
    """Phase 14b B: KB.add_pattern accepts the diff text (not just
    a stat) and round-trips it via get_pattern."""
    from core.knowledge_base import KnowledgeBase
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "kb.db"
        kb = KnowledgeBase(db_path=db)
        pid = kb.add_pattern(
            tags=["x"], problem_summary="P",
            solution_code="diff --git a/foo b/foo\n-old\n+new\n",
            solution_pattern="STEP 1: edit foo",
            explanation="e", trace_id="SEN-test-44",
            base_sha="abc1234",
        )
        p = kb.get_pattern(pid)
        assert p.solution_code.startswith("diff --git")
        assert p.base_sha == "abc1234"


def test_g_35_quality_gate_rejects_degenerate_pass(fresh_kb, monkeypatch):
    """A bare '200' that runs but is degenerate must NOT count as a
    graduation pass (would falsely confirm transfer)."""
    pid = _seed_pattern(fresh_kb)
    _stub_qwen_validate(
        monkeypatch,
        qwen_text='{"code": "200"}',
        exec_passes=True, exec_stdout="200",
    )
    from core.kb_graduation import graduate_pattern
    out = asyncio.run(graduate_pattern(
        pattern_id=pid, problem="x", code_context=None,
        kb=fresh_kb, model_id="qwen", trace_id="SEN-test-35",
    ))
    assert out["passed"] is False  # quality gate rejected
    assert out["solo_attempts"] == 1
    assert out["solo_passes"] == 0
