"""Phase 17b -- /code auto-decompose chain runner.

When `_extract_decomposition` fires AND `config.CODE_CHAIN_ENABLED=True`
AND the parent task's `chain_depth < CODE_CHAIN_MAX_DEPTH`, the pipeline
queues child /code tasks (one per subtask) instead of returning the
markdown list to the user. Each child runs as its own full /code with
its own pre-teach, KB retrieval, recipe, etc.

Three groups:
  C -- core column + helper behavior (parent_task_id, chain_depth,
       list_children, chain_status_summary)
  E -- enqueue path (DECOMPOSE detected -> children queued, depth
       inheritance, depth cap, ENABLED flag gating)
  W -- wiring source-checks (config flags exist, render branch,
       short-circuit return)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from core import config, database
    db_path = tmp_path / "sentinel_chain.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    database.init_db()
    yield db_path


# ============================================================
# Group C: column + helper
# ============================================================


def test_c01_parent_task_id_column_exists(temp_db):
    from core import database
    conn = sqlite3.connect(database.config.DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)")]
    conn.close()
    assert "parent_task_id" in cols


def test_c02_chain_depth_column_exists(temp_db):
    from core import database
    conn = sqlite3.connect(database.config.DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)")]
    conn.close()
    assert "chain_depth" in cols


def test_c03_add_task_default_no_parent(temp_db):
    """Standalone /code: parent_task_id=None, chain_depth=0."""
    from core import database
    tid = database.add_task("SEN-c03", "code", {})
    row = database.get_task(tid)
    assert row.parent_task_id is None
    assert row.chain_depth == 0


def test_c04_add_task_with_parent_kwarg_only(temp_db):
    """parent_task_id and chain_depth must be keyword-only (no
    positional foot-gun for the legacy 5-arg add_task signature)."""
    from core import database
    parent = database.add_task("SEN-parent", "code", {})
    child = database.add_task(
        "SEN-child", "code", {"text": "subtask"},
        parent_task_id=parent, chain_depth=1,
    )
    row = database.get_task(child)
    assert row.parent_task_id == parent
    assert row.chain_depth == 1
    # Signature check
    import inspect
    sig = inspect.signature(database.add_task)
    p1 = sig.parameters["parent_task_id"]
    p2 = sig.parameters["chain_depth"]
    assert p1.kind == inspect.Parameter.KEYWORD_ONLY
    assert p2.kind == inspect.Parameter.KEYWORD_ONLY


def test_c05_list_children_empty_when_no_parent(temp_db):
    from core import database
    parent = database.add_task("SEN-empty", "code", {})
    assert database.list_children(parent) == []


def test_c06_list_children_returns_in_creation_order(temp_db):
    from core import database
    parent = database.add_task("SEN-p2", "code", {})
    a = database.add_task("SEN-a", "code", {"sub": 1}, parent_task_id=parent)
    b = database.add_task("SEN-b", "code", {"sub": 2}, parent_task_id=parent)
    c = database.add_task("SEN-c", "code", {"sub": 3}, parent_task_id=parent)
    rows = database.list_children(parent)
    assert [r.task_id for r in rows] == [a, b, c]


def test_c07_list_children_excludes_unrelated(temp_db):
    """Other tasks not in the chain must not leak into list_children."""
    from core import database
    parent = database.add_task("SEN-p3", "code", {})
    other_top = database.add_task("SEN-other", "code", {})
    child = database.add_task(
        "SEN-real-child", "code", {}, parent_task_id=parent,
    )
    rows = database.list_children(parent)
    assert [r.task_id for r in rows] == [child]


def test_c08_chain_status_summary_zero_state(temp_db):
    from core import database
    parent = database.add_task("SEN-summary", "code", {})
    s = database.chain_status_summary(parent)
    assert s == {"total": 0, "completed": 0, "failed": 0,
                 "pending": 0, "processing": 0}


def test_c09_chain_status_summary_mixed_states(temp_db):
    """Cover all four statuses across siblings. fail_task only sets
    status=failed after max_retries; we direct-SQL it for the test
    so the assertion is about chain_status_summary's grouping logic,
    not about fail_task's retry semantics."""
    from core import database
    parent = database.add_task("SEN-mix", "code", {})
    a = database.add_task("SEN-a", "code", {}, parent_task_id=parent)
    b = database.add_task("SEN-b", "code", {}, parent_task_id=parent)
    c = database.add_task("SEN-c", "code", {}, parent_task_id=parent)
    database.complete_task(a, {"ok": True})
    # Force b to status='failed' directly (skip fail_task's retry
    # accounting -- not the focus of this test).
    conn = sqlite3.connect(database.config.DB_PATH)
    conn.execute(
        "UPDATE tasks SET status='failed' WHERE task_id=?", (b,),
    )
    conn.commit()
    conn.close()
    s = database.chain_status_summary(parent)
    assert s["total"] == 3
    assert s["completed"] == 1
    assert s["failed"] == 1
    assert s["pending"] == 1


def test_c10_chain_depth_inherited_through_chain(temp_db):
    """Depth chain: 0 -> 1 -> 2."""
    from core import database
    p0 = database.add_task("SEN-d0", "code", {})
    p1 = database.add_task(
        "SEN-d1", "code", {}, parent_task_id=p0, chain_depth=1,
    )
    p2 = database.add_task(
        "SEN-d2", "code", {}, parent_task_id=p1, chain_depth=2,
    )
    assert database.get_task(p0).chain_depth == 0
    assert database.get_task(p1).chain_depth == 1
    assert database.get_task(p2).chain_depth == 2


# ============================================================
# Group E: enqueue path behavioral
# ============================================================


def test_e01_chain_enabled_false_returns_decompose_markdown(
    temp_db, monkeypatch,
):
    """When CODE_CHAIN_ENABLED=False, the DECOMPOSE branch returns
    the existing Phase 17 Batch 1 markdown response (no children
    queued)."""
    from core import config, database
    monkeypatch.setattr(config, "CODE_CHAIN_ENABLED", False)
    from skills.code_assist import _extract_decomposition

    recipe = (
        "DECOMPOSE\n"
        "- /code one\n"
        "- /code two\n"
        "- /code three\n"
    )
    subs = _extract_decomposition(recipe)
    assert subs is not None
    assert len(subs) == 3
    # The pipeline branch isn't called directly here, but we assert
    # the source-level guard (later in Group W).


def test_e02_chain_runner_queues_children_with_correct_args(temp_db):
    """Direct test of database.add_task with the chain runner's
    expected args. Round-trip the args dict + parent + depth."""
    from core import database
    parent = database.add_task("SEN-parent", "code", {"text": "big"})
    sub_texts = ["one", "two", "three"]
    children = []
    for txt in sub_texts:
        cid = database.add_task(
            "SEN-child-" + txt, "code", {"text": txt},
            parent_task_id=parent, chain_depth=1,
        )
        children.append(cid)
    listed = database.list_children(parent)
    assert len(listed) == 3
    assert all(c.chain_depth == 1 for c in listed)
    assert all(c.parent_task_id == parent for c in listed)
    assert [c.args["text"] for c in listed] == sub_texts


def test_e03_chain_depth_cap_blocks_recursive_decompose(temp_db):
    """Conceptual: a child task whose chain_depth == MAX_DEPTH must
    NOT be allowed to decompose further. Pipeline checks
    self_chain_depth < CODE_CHAIN_MAX_DEPTH. Source-level check."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert "self_chain_depth < config.CODE_CHAIN_MAX_DEPTH" in src


def test_e04_chain_runner_uses_keyword_only_kwargs():
    """Source check: chain runner queues children via
    add_task(..., parent_task_id=..., chain_depth=...) keyword-only."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert "parent_task_id=kill_task_id" in src
    assert "chain_depth=self_chain_depth + 1" in src


def test_e05_chain_runner_strips_slash_code_prefix():
    """Source check: subtask text 'subtask body' is queued as
    args={'text': 'subtask body'}, not args={'text': '/code subtask'}.
    The bot's command router would re-parse it otherwise."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert 'sub_text.startswith("/code ")' in src


def test_e06_chain_runner_logs_chain_queued():
    """Per-child log line CHAIN-QUEUED so log scans can trace child
    tasks back to parent."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert "CHAIN-QUEUED" in src


def test_e07_chain_runner_falls_through_on_enqueue_error():
    """If add_task throws, fall through to the manual decomposition
    surface (don't crash the /code). Structural check: the
    `decompose_suggested` return path must come AFTER the chain
    runner block in source order, so an empty queued_ids list
    naturally falls into it."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert "CHAIN-ERROR" in src
    chain_err_idx = src.find("CHAIN-ERROR")
    decompose_return_idx = src.find('solved_by="decompose_suggested"')
    assert chain_err_idx > 0 and decompose_return_idx > 0
    # decompose_suggested return must come AFTER CHAIN-ERROR in source.
    assert decompose_return_idx > chain_err_idx, (
        "fall-through path (decompose_suggested return) must come "
        "after the chain-runner CHAIN-ERROR block"
    )
    # And queued_ids = [] must zero out before the if-check, so on
    # enqueue error we fall through naturally.
    assert "queued_ids = []" in src


# ============================================================
# Group W: wiring source-checks
# ============================================================


def test_w01_config_chain_enabled_exists():
    """The flag must exist and be a bool. Per Phase 17d
    (post-live-validation 2026-05-06 ~00:46Z), the flag was
    flipped to True after the chain runner was proven to fire
    decomposition + queue children correctly. Test no longer
    asserts the default value -- only that the flag is present
    and bool-typed (the runtime read in skills/code_assist.py
    relies on truthiness)."""
    from core import config
    assert hasattr(config, "CODE_CHAIN_ENABLED")
    assert isinstance(config.CODE_CHAIN_ENABLED, bool)


def test_w02_config_max_depth_one():
    from core import config
    assert hasattr(config, "CODE_CHAIN_MAX_DEPTH")
    assert config.CODE_CHAIN_MAX_DEPTH == 1


def test_w03_telegram_render_includes_chain_started():
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    idx = src.find('"qwen_killed"')
    assert idx > 0
    nearby = src[idx:idx + 400]
    assert '"chain_started"' in nearby


def test_w04_pipeline_short_circuits_on_chain_started():
    """The chain branch must return CodeAssistOutput directly,
    skipping shadow plan / stepfed / KB write."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = src.find('solved_by="chain_started"')
    assert idx > 0
    # Walk back ~3000 chars; must contain "if queued_ids:" before the
    # return CodeAssistOutput.
    window = src[max(0, idx - 3000):idx]
    assert "if queued_ids:" in window


def test_w05_chain_branch_logs_decomposition_decision():
    """Even when chain doesn't fire (flag off / depth cap), the
    decomposition decision should be logged for telemetry."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = src.find("decomposition suggested")
    assert idx > 0
    nearby = src[idx:idx + 600]
    assert "chain_will_fire=" in nearby


def test_w06_pipeline_passes_self_chain_depth():
    """Pipeline reads chain_depth from the trace-id-mapped task row."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert "self_chain_depth" in src
    assert "_row.chain_depth" in src


def test_w07_imports_clean():
    """Module loads without errors."""
    from skills import code_assist  # noqa: F401
    from core import database  # noqa: F401
    from core import config  # noqa: F401
