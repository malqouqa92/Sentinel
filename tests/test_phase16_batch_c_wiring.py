"""Phase 16 Batch C -- skip-path wiring + dispatcher ECC.

Source-level + behavioral checks (mocked LLM). Live replay is in
the separate stress test runner.
"""
from __future__ import annotations

import asyncio
import struct
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core import config, embeddings as emb
from core.knowledge_base import KnowledgeBase
from skills import code_assist


def _stub_embedder(monkeypatch):
    def fake_embed(text, trace_id="SEN-system"):
        seed = sum(ord(c) for c in (text or "")) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(config.EMBEDDING_DIM).tolist()
        return struct.pack(f"<{len(vec)}f", *vec)
    monkeypatch.setattr(emb, "embed_text", fake_embed)


@pytest.fixture
def kb_with_skip_eligible(tmp_path: Path, monkeypatch) -> KnowledgeBase:
    db_path = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    _stub_embedder(monkeypatch)
    kb = KnowledgeBase(db_path=db_path)
    pid = kb.add_pattern(
        tags=["t"], problem_summary="add a helper",
        solution_code=(
            "diff --git a/x.py b/x.py\nnew file mode 100644\n"
            "--- /dev/null\n+++ b/x.py\n"
            "@@ -0,0 +1,1 @@\n+def helper(): return 42\n"
        ),
        solution_pattern=(
            'STEP 1: write_file path="x.py" '
            'content="def helper(): return 42\\n"\n'
            'STEP 2: run_bash command="python -c \\"from x import helper; '
            'assert helper() == 42; print(\'ok\')\\""\n'
            'STEP 3: done summary="created x.py"'
        ),
        explanation="trivial", trace_id="SEN-test",
    )
    kb.pin_pattern(pid)
    return kb


def _async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


_CA_SRC = (Path(__file__).resolve().parent.parent
           / "skills" / "code_assist.py").read_text(encoding="utf-8")


# ─── Source-level wiring ───


def test_w01_skip_path_called_before_attempt_loop():
    skip_idx = _CA_SRC.find("_maybe_skip_path(")
    loop_idx = _CA_SRC.find("for attempt in range(1, MAX_TEACH_ATTEMPTS")
    assert 0 < skip_idx < loop_idx


def test_w02_success_returns_qwen_skip_path():
    assert "solved_by=\"qwen_skip_path\"" in _CA_SRC


def test_w03_failed_excludes_pattern_from_fallback():
    assert "excluded_skip_id" in _CA_SRC
    assert "p.id != excluded_skip_id" in _CA_SRC


def test_w04_uses_surgical_revert_not_scoped_reset():
    """CRITICAL guard against the v1 self-wipe: the failure cleanup
    in _execute_skip_replay must use surgical_revert (recipe-only
    paths) and NEVER call _git_reset_hard (which would scope-blast
    core/skills/agents/tests/interfaces and wipe uncommitted source)."""
    # Find the _execute_skip_replay function body
    start = _CA_SRC.find("async def _execute_skip_replay(")
    end = _CA_SRC.find("async def _run_agentic_pipeline(", start)
    body = _CA_SRC[start:end]
    assert "surgical_revert" in body, (
        "skip-replay failure cleanup must use surgical_revert"
    )
    assert "_git_reset_hard" not in body, (
        "skip-replay failure must NEVER call _git_reset_hard "
        "(scope-blasts uncommitted source)"
    )


def test_w05_extract_recipe_paths_helper_present():
    assert "_extract_recipe_paths" in _CA_SRC
    assert "_RECIPE_PATH_RE" in _CA_SRC


# ─── _maybe_skip_path behavior ───


def test_w10_no_kb_match_ineligible(kb_with_skip_eligible):
    input_data = MagicMock(problem="x", code_context=None)
    r = _async(code_assist._maybe_skip_path(
        input_data=input_data, trace_id="SEN-t",
        kb=kb_with_skip_eligible, kb_patterns=[],
        backend_model="qwen2.5-coder:3b", base_sha="abc1234",
    ))
    assert r["status"] == "ineligible"
    assert r["reason"] == "no_kb_match"


def test_w11_low_passes_pattern_ineligible(tmp_path, monkeypatch):
    db_path = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    _stub_embedder(monkeypatch)
    kb = KnowledgeBase(db_path=db_path)
    pid = kb.add_pattern(
        tags=["t"], problem_summary="x",
        solution_code="(diff)", solution_pattern="STEP 1: done",
        explanation="x", trace_id="SEN-t",
    )
    entry = kb.get_pattern(pid)
    input_data = MagicMock(problem="x", code_context=None)
    r = _async(code_assist._maybe_skip_path(
        input_data=input_data, trace_id="SEN-t",
        kb=kb, kb_patterns=[entry],
        backend_model="qwen2.5-coder:3b", base_sha="abc1234",
    ))
    assert r["status"] == "ineligible"
    assert r["reason"].startswith("eligibility:LOW_PASSES")


def test_w12_lint_failure_ineligible(kb_with_skip_eligible):
    import sqlite3
    bad = (
        'STEP 1: read_file path="x.py"\n'
        'STEP 2: done summary="just looked"'
    )
    conn = sqlite3.connect(kb_with_skip_eligible.db_path)
    conn.execute(
        "UPDATE knowledge SET solution_pattern=? WHERE id=1", (bad,)
    )
    conn.commit()
    conn.close()
    entry = kb_with_skip_eligible.get_pattern(1)
    input_data = MagicMock(problem="x", code_context=None)
    r = _async(code_assist._maybe_skip_path(
        input_data=input_data, trace_id="SEN-t",
        kb=kb_with_skip_eligible, kb_patterns=[entry],
        backend_model="qwen2.5-coder:3b", base_sha="abc1234",
    ))
    assert r["status"] == "ineligible"
    assert r["reason"].startswith("lint:")


def test_w13_telemetry_mode_does_not_replay(
    kb_with_skip_eligible, monkeypatch,
):
    monkeypatch.setattr(config, "SKIP_PATH_ENABLED", False)
    entry = kb_with_skip_eligible.get_pattern(1)
    input_data = MagicMock(problem="x", code_context=None)
    with patch.object(
        code_assist, "_execute_skip_replay", AsyncMock(),
    ) as mock_replay:
        r = _async(code_assist._maybe_skip_path(
            input_data=input_data, trace_id="SEN-t",
            kb=kb_with_skip_eligible, kb_patterns=[entry],
            backend_model="qwen2.5-coder:3b", base_sha="abc1234",
        ))
        mock_replay.assert_not_called()
    assert r["status"] == "telemetry_only"


def test_w14_live_mode_calls_replay(kb_with_skip_eligible, monkeypatch):
    monkeypatch.setattr(config, "SKIP_PATH_ENABLED", True)
    entry = kb_with_skip_eligible.get_pattern(1)
    input_data = MagicMock(problem="x", code_context=None)
    fake = {"status": "success", "pattern_id": 1, "reason": "x",
            "solution": "ok", "diff_stat": ""}

    async def fake_replay(**kw):
        return fake

    with patch.object(
        code_assist, "_execute_skip_replay", side_effect=fake_replay,
    ):
        r = _async(code_assist._maybe_skip_path(
            input_data=input_data, trace_id="SEN-t",
            kb=kb_with_skip_eligible, kb_patterns=[entry],
            backend_model="qwen2.5-coder:3b", base_sha="abc1234",
        ))
    assert r["status"] == "success"


# ─── _execute_skip_replay failure classification ───


def test_w20_environmental_does_not_pollute_counter(
    kb_with_skip_eligible, monkeypatch,
):
    monkeypatch.setattr(config, "SKIP_PATH_ENABLED", True)
    entry = kb_with_skip_eligible.get_pattern(1)
    input_data = MagicMock(problem="x", code_context=None)

    async def boom(*a, **kw):
        raise RuntimeError("git not available")

    record_calls = []

    def track(pid, passed, trace_id="?"):
        record_calls.append((pid, passed))
        return (1, 1, False)

    with patch("core.tree_state.snapshot_dirty_tree",
               side_effect=boom), \
         patch.object(kb_with_skip_eligible, "record_solo_attempt",
                      side_effect=track):
        r = _async(code_assist._execute_skip_replay(
            pattern=entry, input_data=input_data, trace_id="SEN-t",
            kb=kb_with_skip_eligible,
            backend_model="qwen2.5-coder:3b", base_sha="abc1234",
        ))
    assert r["status"] == "failed"
    assert r["reason"].startswith("environmental:")
    assert record_calls == []


def test_w23_replay_success_increments_counter(
    kb_with_skip_eligible, monkeypatch,
):
    monkeypatch.setattr(config, "SKIP_PATH_ENABLED", True)
    entry = kb_with_skip_eligible.get_pattern(1)
    input_data = MagicMock(problem="x", code_context=None)
    fake_handle = MagicMock(had_dirty=False, patch_path=None)
    fake_match = MagicMock(accept=True, score=0.95, reason="match")

    async def fake_snap(_root):
        return fake_handle

    async def fake_restore(_h):
        return MagicMock(restored=True, reason="ok",
                         leftover_patch=None)

    async def fake_diff_full(_sha):
        return ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
                "@@ -1,1 +1,2 @@\n+def helper(): return 42\n")

    async def fake_diff_stat(_sha):
        return "x.py | 1 +"

    record_calls = []

    def track(pid, passed, trace_id="?"):
        record_calls.append((pid, passed))
        return (3, 3, False)

    def fake_step(problem, recipe, trace_id, model):
        return {"summary": "ok", "session": [], "steps": 3,
                "completed_via_done": True}

    with patch("core.tree_state.snapshot_dirty_tree",
               side_effect=fake_snap), \
         patch("core.tree_state.restore_dirty_tree",
               side_effect=fake_restore), \
         patch("core.qwen_agent.run_agent_stepfed",
               side_effect=fake_step), \
         patch.object(code_assist, "_git_diff_full",
                      side_effect=fake_diff_full), \
         patch.object(code_assist, "_git_diff_stat",
                      side_effect=fake_diff_stat), \
         patch("core.diff_match.evaluate_diff_match",
               return_value=fake_match), \
         patch.object(kb_with_skip_eligible, "record_solo_attempt",
                      side_effect=track):
        r = _async(code_assist._execute_skip_replay(
            pattern=entry, input_data=input_data, trace_id="SEN-t",
            kb=kb_with_skip_eligible,
            backend_model="qwen2.5-coder:3b", base_sha="abc1234",
        ))
    assert r["status"] == "success"
    assert record_calls == [(entry.id, True)]


def test_w30_solution_mentions_skip_path():
    assert "Solved via skip-path" in _CA_SRC
    assert "NO Claude calls" in _CA_SRC


# ─── extract_recipe_paths helper ───


def test_w40_extract_recipe_paths_basic():
    recipe = (
        'STEP 1: write_file path="a.py" content="..."\n'
        'STEP 2: edit_file path="b.py" old="x" new="y"\n'
        'STEP 3: run_bash command="..."\n'
        'STEP 4: done summary="..."'
    )
    paths = code_assist._extract_recipe_paths(recipe)
    assert paths == ["a.py", "b.py"]


def test_w41_extract_recipe_paths_dedup():
    recipe = (
        'STEP 1: read_file path="a.py"\n'
        'STEP 2: edit_file path="a.py" old="x" new="y"\n'
        'STEP 3: done'
    )
    paths = code_assist._extract_recipe_paths(recipe)
    assert paths == ["a.py"]


def test_w42_extract_recipe_paths_empty():
    assert code_assist._extract_recipe_paths("") == []
    assert code_assist._extract_recipe_paths(None) == []
