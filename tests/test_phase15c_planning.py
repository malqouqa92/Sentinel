"""Phase 15c -- Qwen-planning shadow tests.

ECC. No GPU, no real LLM, no executor subprocess. The shadow Qwen
call site is verified at the source level (the contextvar pattern
plus the wait_for + best-effort try/except shape) so we never spin
up Ollama. The scoring heuristic + KB schema + round-trip are
unit-tested directly.

Coverage:
  Schema migration:
    Q01 -- fresh KB has qwen_plan_recipe + qwen_plan_agreement cols
    Q02 -- pre-15c DB gets ALTERed cleanly (NULL on existing rows)
    Q03 -- double-init is idempotent

  KB round-trip:
    Q11 -- add_pattern accepts the new kwargs and stores them
    Q12 -- _row_to_entry round-trips both fields
    Q13 -- agreement clamped to [0,1] on out-of-range input
    Q14 -- agreement=None preserved (distinguishes "no data" from 0.0)
    Q15 -- planning_stats: empty DB returns clean zeros
    Q16 -- planning_stats: mean and percentiles
    Q17 -- planning_stats: by_archetype tag rollup

  score_plan_agreement:
    Q21 -- identical recipes -> ~1.0
    Q22 -- completely different recipes -> low
    Q23 -- same files different tools -> partial credit
    Q24 -- empty inputs -> 0.0
    Q25 -- malformed (no STEP lines) -> 0.0
    Q26 -- exception inside parser does not propagate

  Hook integration (source-level):
    Q31 -- _run_agentic_pipeline calls _qwen_shadow_plan after
           _claude_pre_teach with proper try/except guard
    Q32 -- shadow_recipe + shadow_agreement piped into add_pattern
"""
from __future__ import annotations

import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core import embeddings as emb
from core.knowledge_base import KnowledgeBase, _connect as _kb_connect
from core.plan_agreement import score_plan_agreement


def _stub_embedder(monkeypatch):
    def fake_embed(text, trace_id="SEN-system"):
        seed = sum(ord(c) for c in (text or "")) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(config.EMBEDDING_DIM).tolist()
        return struct.pack(f"<{len(vec)}f", *vec)
    monkeypatch.setattr(emb, "embed_text", fake_embed)


@pytest.fixture
def fresh_kb(tmp_path, monkeypatch):
    db_path = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    _stub_embedder(monkeypatch)
    return KnowledgeBase(db_path=db_path)


CLAUDE_RECIPE_REF = (
    'STEP 1: write_file path="math_utils.py" content="def gcd(a, b):\\n'
    '    return b if a == 0 else gcd(b % a, a)\\n"\n'
    'STEP 2: run_bash command="python -c \\"from math_utils import gcd; '
    'assert gcd(12, 8) == 4\\""\n'
    'STEP 3: done summary="added gcd"\n'
)


# ─────────────────────────────────────────────────────────────────
# Schema migration
# ─────────────────────────────────────────────────────────────────


def test_q01_fresh_kb_has_shadow_columns(fresh_kb):
    conn = _kb_connect(fresh_kb.db_path)
    try:
        cols = {r["name"] for r in conn.execute(
            "PRAGMA table_info(knowledge)"
        ).fetchall()}
    finally:
        conn.close()
    assert "qwen_plan_recipe" in cols
    assert "qwen_plan_agreement" in cols


def test_q02_pre_15c_db_gets_migrated(tmp_path, monkeypatch):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL, tags TEXT NOT NULL,
            problem_summary TEXT NOT NULL,
            solution_code TEXT, solution_pattern TEXT,
            explanation TEXT NOT NULL,
            source_trace_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            usage_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO knowledge (category, tags, problem_summary, "
        "solution_code, solution_pattern, explanation, "
        "source_trace_id, created_at) VALUES "
        "('pattern', 'x', 'old', 'c', 'p', 'e', 'SEN-old', "
        "'2026-01-01T00:00:00')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    _stub_embedder(monkeypatch)
    kb = KnowledgeBase(db_path=db_path)
    p = kb.get_pattern(1)
    assert p is not None
    assert p.qwen_plan_recipe is None
    assert p.qwen_plan_agreement is None


def test_q03_double_init_idempotent(fresh_kb):
    KnowledgeBase(db_path=fresh_kb.db_path)
    KnowledgeBase(db_path=fresh_kb.db_path)


# ─────────────────────────────────────────────────────────────────
# KB round-trip
# ─────────────────────────────────────────────────────────────────


def test_q11_add_pattern_stores_shadow_fields(fresh_kb):
    pid = fresh_kb.add_pattern(
        tags=["test"], problem_summary="P",
        solution_code="c", solution_pattern="STEP 1: done",
        explanation="e", trace_id="SEN-q11",
        qwen_plan_recipe="STEP 1: done summary=\"x\"",
        qwen_plan_agreement=0.75,
    )
    p = fresh_kb.get_pattern(pid)
    assert p.qwen_plan_recipe == "STEP 1: done summary=\"x\""
    assert p.qwen_plan_agreement == pytest.approx(0.75, abs=1e-6)


def test_q12_row_to_entry_handles_both_fields(fresh_kb):
    pid = fresh_kb.add_pattern(
        tags=["t"], problem_summary="p", solution_code="c",
        solution_pattern="x", explanation="e", trace_id="SEN-q12",
        qwen_plan_recipe="STEP 1: read_file path=\"a.py\"",
        qwen_plan_agreement=0.42,
    )
    p = fresh_kb.get_pattern(pid)
    assert hasattr(p, "qwen_plan_recipe")
    assert hasattr(p, "qwen_plan_agreement")
    assert p.qwen_plan_recipe.startswith("STEP 1:")
    assert 0.0 <= p.qwen_plan_agreement <= 1.0


def test_q13_agreement_clamped(fresh_kb):
    pid_hi = fresh_kb.add_pattern(
        tags=[], problem_summary="hi", solution_code="c",
        solution_pattern="x", explanation="e", trace_id="SEN-q13a",
        qwen_plan_recipe="STEP 1: done", qwen_plan_agreement=2.5,
    )
    pid_lo = fresh_kb.add_pattern(
        tags=[], problem_summary="lo", solution_code="c",
        solution_pattern="x", explanation="e", trace_id="SEN-q13b",
        qwen_plan_recipe="STEP 1: done", qwen_plan_agreement=-0.4,
    )
    assert fresh_kb.get_pattern(pid_hi).qwen_plan_agreement == 1.0
    assert fresh_kb.get_pattern(pid_lo).qwen_plan_agreement == 0.0


def test_q14_agreement_none_preserved(fresh_kb):
    """None stays None -- must be distinguishable from 0.0 so
    'no shadow data' isn't conflated with 'shadow ran and scored 0'."""
    pid = fresh_kb.add_pattern(
        tags=[], problem_summary="p", solution_code="c",
        solution_pattern="x", explanation="e", trace_id="SEN-q14",
        # No shadow kwargs.
    )
    p = fresh_kb.get_pattern(pid)
    assert p.qwen_plan_recipe is None
    assert p.qwen_plan_agreement is None


def test_q15_planning_stats_empty_db(fresh_kb):
    s = fresh_kb.planning_stats()
    assert s["patterns_total"] == 0
    assert s["patterns_with_shadow"] == 0
    assert s["mean_agreement"] is None
    assert s["p25"] is None and s["p50"] is None and s["p75"] is None
    assert s["by_archetype"] == []


def test_q16_planning_stats_aggregates(fresh_kb):
    # One row without shadow data, three with various scores.
    fresh_kb.add_pattern(
        tags=["a"], problem_summary="no shadow", solution_code="c",
        solution_pattern="x", explanation="e", trace_id="SEN-q16-0",
    )
    for i, score in enumerate([0.2, 0.5, 0.9], start=1):
        fresh_kb.add_pattern(
            tags=["a"], problem_summary=f"p{i}", solution_code="c",
            solution_pattern="x", explanation="e",
            trace_id=f"SEN-q16-{i}",
            qwen_plan_recipe="STEP 1: done",
            qwen_plan_agreement=score,
        )
    s = fresh_kb.planning_stats()
    assert s["patterns_total"] == 4
    assert s["patterns_with_shadow"] == 3
    # Mean of {0.2, 0.5, 0.9} == ~0.533
    assert s["mean_agreement"] == pytest.approx(0.5333, abs=0.01)
    # Sorted: [0.2, 0.5, 0.9]; bucket-quantiles at idx 0/1/2.
    assert s["p25"] == pytest.approx(0.2, abs=0.01)
    assert s["p50"] == pytest.approx(0.5, abs=0.01)
    assert s["p75"] == pytest.approx(0.9, abs=0.01)


def test_q17_planning_stats_per_tag(fresh_kb):
    fresh_kb.add_pattern(
        tags=["alpha", "x"], problem_summary="a1", solution_code="c",
        solution_pattern="x", explanation="e", trace_id="SEN-q17-1",
        qwen_plan_recipe="STEP 1: done", qwen_plan_agreement=0.4,
    )
    fresh_kb.add_pattern(
        tags=["alpha", "y"], problem_summary="a2", solution_code="c",
        solution_pattern="x", explanation="e", trace_id="SEN-q17-2",
        qwen_plan_recipe="STEP 1: done", qwen_plan_agreement=0.6,
    )
    fresh_kb.add_pattern(
        tags=["beta"], problem_summary="b1", solution_code="c",
        solution_pattern="x", explanation="e", trace_id="SEN-q17-3",
        qwen_plan_recipe="STEP 1: done", qwen_plan_agreement=0.8,
    )
    s = fresh_kb.planning_stats()
    by = {row["tag"]: row for row in s["by_archetype"]}
    assert by["alpha"]["n"] == 2
    assert by["alpha"]["mean_agreement"] == pytest.approx(0.5, abs=0.01)
    assert by["beta"]["n"] == 1
    assert by["beta"]["mean_agreement"] == pytest.approx(0.8, abs=0.01)


# ─────────────────────────────────────────────────────────────────
# score_plan_agreement
# ─────────────────────────────────────────────────────────────────


def test_q21_identical_recipes_score_high():
    s = score_plan_agreement(CLAUDE_RECIPE_REF, CLAUDE_RECIPE_REF)
    assert s == pytest.approx(1.0, abs=1e-6), (
        f"identical recipes should score 1.0, got {s}"
    )


def test_q22_completely_different_recipes_score_low():
    other = (
        'STEP 1: read_file path="totally/unrelated.py"\n'
        'STEP 2: list_dir path="/tmp"\n'
        'STEP 3: list_dir path="/var"\n'
        'STEP 4: list_dir path="/etc"\n'
        'STEP 5: done summary="explored"\n'
    )
    s = score_plan_agreement(CLAUDE_RECIPE_REF, other)
    assert s < 0.4, (
        f"unrelated recipes should score under 0.4, got {s}"
    )


def test_q23_same_files_different_tools_partial_credit():
    """Both recipes touch math_utils.py, but Qwen uses edit_file
    instead of write_file and skips the shell verification. Partial
    credit on the file Jaccard, less on the tool Jaccard, partial
    on step count."""
    qwen = (
        'STEP 1: edit_file path="math_utils.py" old="x" new="y"\n'
        'STEP 2: done summary="ok"\n'
    )
    s = score_plan_agreement(CLAUDE_RECIPE_REF, qwen)
    # File Jaccard = 1/1 = 1.0 (both touch math_utils.py only).
    # Tool Jaccard depends: claude has {write_file, run_bash, done},
    # qwen has {edit_file, done} -> intersection {done}, union 4 -> 0.25
    # Step proximity: 3 vs 2 -> 1 - 1/3 = 0.667
    # blend = 0.5*1.0 + 0.3*0.25 + 0.2*0.667 = 0.5 + 0.075 + 0.133 = 0.708
    assert 0.50 < s < 0.85, f"expected partial credit, got {s}"


def test_q24_empty_inputs_score_zero():
    assert score_plan_agreement("", CLAUDE_RECIPE_REF) == 0.0
    assert score_plan_agreement(CLAUDE_RECIPE_REF, "") == 0.0
    assert score_plan_agreement(None, None) == 0.0
    assert score_plan_agreement(None, CLAUDE_RECIPE_REF) == 0.0


def test_q25_malformed_no_step_lines_score_zero():
    """Recipes lacking STEP N: structure parse to zero steps and
    must score 0.0 (we want a real recipe, not prose)."""
    junk = "Sure, I'd write the function and test it."
    assert score_plan_agreement(CLAUDE_RECIPE_REF, junk) == 0.0
    assert score_plan_agreement(junk, CLAUDE_RECIPE_REF) == 0.0


def test_q26_exception_does_not_propagate():
    """Pass non-string objects to force a parser exception. The
    function must return 0.0, not propagate."""
    # contextvars.Token objects don't have the .lower / .find that
    # the parser would call -- forces the broad except path.
    class Boom:
        def __str__(self): raise RuntimeError("nope")
    s = score_plan_agreement(Boom(), Boom())  # type: ignore[arg-type]
    assert s == 0.0


# ─────────────────────────────────────────────────────────────────
# Hook integration (source-level: avoid spinning up Ollama)
# ─────────────────────────────────────────────────────────────────


def _read_code_assist():
    return (
        Path(__file__).resolve().parent.parent
        / "skills" / "code_assist.py"
    ).read_text(encoding="utf-8", errors="replace")


def test_q31_shadow_hook_present_in_pipeline():
    src = _read_code_assist()
    # The shadow plan helper exists.
    assert "async def _qwen_shadow_plan" in src
    # 30s timeout cap is wired.
    assert "QWEN_SHADOW_TIMEOUT_S" in src
    assert "asyncio.wait_for" in src
    # The pipeline calls the helper after _claude_pre_teach on
    # attempt 1 and inside a try/except guard.
    pipeline_idx = src.find("async def _run_agentic_pipeline")
    assert pipeline_idx > 0
    # Window grew in Phase 17b (chain runner added ~80 lines inside
    # the decomposition branch which sits between _claude_pre_teach
    # and _qwen_shadow_plan). Bumped 8000 -> 16000 to keep the
    # invariant "shadow plan is wired into attempt 1" testable.
    body = src[pipeline_idx:pipeline_idx + 16000]
    assert "_claude_pre_teach" in body
    assert "_qwen_shadow_plan" in body
    assert "score_plan_agreement" in body
    # Best-effort: a try/except wraps the shadow block.
    assert "shadow_recipe = None" in body
    assert "shadow_agreement = None" in body


def test_q32_shadow_piped_into_add_pattern():
    src = _read_code_assist()
    # The kwargs reach add_pattern at the success-path storage call.
    # A simple substring check is enough; we don't need to AST parse.
    assert "qwen_plan_recipe=shadow_recipe" in src
    assert "qwen_plan_agreement=shadow_agreement" in src


def test_q33_shadow_timeout_constant_is_set():
    """Sanity: the cap is non-trivial. Phase 15c shipped at 30s
    (one-shot text-out budget). Phase 16 Batch A bumped to 90s
    because the tools-enabled agentic shadow does multiple LLM
    round-trips. Either is acceptable; zero or absurd values are
    not."""
    from skills import code_assist as ca
    assert 10 <= ca.QWEN_SHADOW_TIMEOUT_S <= 600
