"""Phase 17d -- chain-child auto-commit + bot child-result relay + /revert chain.

Three groups:
  C -- _chain_child_auto_commit helper (the load-bearing fix)
  R -- bot-side _relay_chain_children + _send_chain_child_result (UX)
  V -- /revert chain handler (cleanup convenience)

Why 17d exists: Phase 17b's chain runner (live 2026-05-06 ~00:46Z)
shipped child task spawning, but each child runs its own /code
pipeline which uses _git_reset_hard("core skills agents tests
interfaces") between attempts AND on failure. When child 2 had any
reset event, child 1's uncommitted work was wiped. Subtask 1's
qcode_assist.py + agents/qcode_assistant.yaml survived (untracked
NEW files outside reset's removal scope) but child 1's edits to
core/config.py were reverted. End-to-end autonomy was therefore
broken even though subtask 1 was reported as PASS.

Fix: scoped exception to NO-AUTO-COMMITS for chain-child success
path. Auto-commit happens ONLY when (a) parent_task_id is non-NULL
AND (b) recipe_paths is non-empty AND (c) attempt PASSED. Standalone
/code (no parent) still respects the directive.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent


def _git(*args, cwd):
    """Sync git wrapper for tests. Returns (rc, stdout)."""
    p = subprocess.run(
        ["git", *args], cwd=str(cwd),
        capture_output=True, text=True,
    )
    return p.returncode, p.stdout + p.stderr


@pytest.fixture
def chain_repo(tmp_path):
    """Throwaway git repo with a baseline commit + a 'core/' file
    so chain-commit can stage something realistic."""
    _git("init", "-q", cwd=tmp_path)
    _git("-c", "user.email=a@b", "-c", "user.name=test",
         "commit", "--allow-empty", "-m", "init", cwd=tmp_path)
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "config.py").write_text(
        "ORIGINAL = True\n", encoding="utf-8",
    )
    _git("add", "core/config.py", cwd=tmp_path)
    _git("-c", "user.email=a@b", "-c", "user.name=test",
         "commit", "-m", "baseline", cwd=tmp_path)
    return tmp_path


# ============================================================
# Group C: _chain_child_auto_commit
# ============================================================


def test_c01_helper_exists_and_async():
    from skills.code_assist import _chain_child_auto_commit
    import inspect
    assert inspect.iscoroutinefunction(_chain_child_auto_commit)


def test_c02_helper_returns_false_on_no_recipe_paths(monkeypatch):
    from skills.code_assist import _chain_child_auto_commit
    result = asyncio.run(_chain_child_auto_commit(
        recipe_paths=[],
        child_task_id="child123abc",
        parent_task_id="parent456def",
        problem="anything",
        trace_id="SEN-test01",
    ))
    assert result is False


def test_c03_helper_actually_commits(chain_repo, monkeypatch):
    """The headline behavior: helper stages + commits the recipe
    paths and creates a real git commit."""
    from core import config
    from skills.code_assist import _chain_child_auto_commit
    monkeypatch.setattr(config, "PROJECT_ROOT", chain_repo)
    # Modify the file (simulating what the recipe did).
    (chain_repo / "core" / "config.py").write_text(
        "ORIGINAL = True\nNEW = 42\n", encoding="utf-8",
    )
    head_before = _git("rev-parse", "HEAD", cwd=chain_repo)[1].strip()
    result = asyncio.run(_chain_child_auto_commit(
        recipe_paths=["core/config.py"],
        child_task_id="abcd1234deadbeef",
        parent_task_id="parent5678cafebabe",
        problem="add NEW=42 to core/config.py",
        trace_id="SEN-c03test",
    ))
    assert result is True
    head_after = _git("rev-parse", "HEAD", cwd=chain_repo)[1].strip()
    assert head_before != head_after, "must create new commit"


def test_c04_commit_message_uses_sentinel_chain_prefix(chain_repo, monkeypatch):
    """Commit msg must start 'sentinel-chain:' so /revert chain can
    detect chain commits and the user can grep for them."""
    from core import config
    from skills.code_assist import _chain_child_auto_commit
    monkeypatch.setattr(config, "PROJECT_ROOT", chain_repo)
    (chain_repo / "core" / "config.py").write_text(
        "X=1\n", encoding="utf-8",
    )
    asyncio.run(_chain_child_auto_commit(
        recipe_paths=["core/config.py"],
        child_task_id="aaaa1111bbbb2222",
        parent_task_id="cccc3333dddd4444",
        problem="x",
        trace_id="SEN-c04",
    ))
    msg = _git("log", "-1", "--format=%s", cwd=chain_repo)[1].strip()
    assert msg.startswith("sentinel-chain:")
    assert "aaaa1111" in msg  # child id short
    assert "cccc3333" in msg  # parent id short


def test_c05_commit_identity_is_chain_child(chain_repo, monkeypatch):
    """Commit author email must be chain-child@sentinel.local so
    /revert chain can filter on it (won't accidentally undo manual
    /commits or sentinel-grad commits)."""
    from core import config
    from skills.code_assist import _chain_child_auto_commit
    monkeypatch.setattr(config, "PROJECT_ROOT", chain_repo)
    (chain_repo / "core" / "config.py").write_text(
        "X=1\n", encoding="utf-8",
    )
    asyncio.run(_chain_child_auto_commit(
        recipe_paths=["core/config.py"],
        child_task_id="abcd",
        parent_task_id="defg",
        problem="x",
        trace_id="SEN-c05",
    ))
    email = _git("log", "-1", "--format=%ae", cwd=chain_repo)[1].strip()
    assert email == "chain-child@sentinel.local"


def test_c06_only_recipe_paths_committed(chain_repo, monkeypatch):
    """Critical: helper must NOT sweep up unrelated dirty work.
    Modify TWO files but list only ONE in recipe_paths -- only that
    one should land in the commit."""
    from core import config
    from skills.code_assist import _chain_child_auto_commit
    monkeypatch.setattr(config, "PROJECT_ROOT", chain_repo)
    # Two modifications.
    (chain_repo / "core" / "config.py").write_text("X=1\n", encoding="utf-8")
    (chain_repo / "core" / "other.py").write_text("Y=2\n", encoding="utf-8")
    # Only one in recipe_paths.
    asyncio.run(_chain_child_auto_commit(
        recipe_paths=["core/config.py"],
        child_task_id="abcd",
        parent_task_id="defg",
        problem="x",
        trace_id="SEN-c06",
    ))
    files = _git("show", "--name-only", "--format=", "HEAD", cwd=chain_repo)[1].strip()
    assert "config.py" in files
    assert "other.py" not in files
    # other.py should still be on disk + dirty
    assert (chain_repo / "core" / "other.py").exists()


def test_c07_returns_false_when_nothing_to_commit(chain_repo, monkeypatch):
    """If the recipe paths are already in HEAD with no dirty edits,
    helper should detect 'nothing to commit' and return False (not
    create an empty commit)."""
    from core import config
    from skills.code_assist import _chain_child_auto_commit
    monkeypatch.setattr(config, "PROJECT_ROOT", chain_repo)
    head_before = _git("rev-parse", "HEAD", cwd=chain_repo)[1].strip()
    # No modifications -- core/config.py is identical to HEAD.
    result = asyncio.run(_chain_child_auto_commit(
        recipe_paths=["core/config.py"],
        child_task_id="abcd",
        parent_task_id="defg",
        problem="x",
        trace_id="SEN-c07",
    ))
    assert result is False
    head_after = _git("rev-parse", "HEAD", cwd=chain_repo)[1].strip()
    assert head_before == head_after, "must NOT create empty commit"


def test_c08_handles_new_file(chain_repo, monkeypatch):
    """Recipes commonly create NEW files (write_file). Helper
    should add + commit them via 'git add'."""
    from core import config
    from skills.code_assist import _chain_child_auto_commit
    monkeypatch.setattr(config, "PROJECT_ROOT", chain_repo)
    (chain_repo / "skills").mkdir(exist_ok=True)
    (chain_repo / "skills" / "newfile.py").write_text(
        "def hello(): pass\n", encoding="utf-8",
    )
    result = asyncio.run(_chain_child_auto_commit(
        recipe_paths=["skills/newfile.py"],
        child_task_id="abcd",
        parent_task_id="defg",
        problem="x",
        trace_id="SEN-c08",
    ))
    assert result is True
    files = _git("show", "--name-only", "--format=", "HEAD", cwd=chain_repo)[1].strip()
    assert "newfile.py" in files


# ============================================================
# Group W: pipeline wiring source-checks
# ============================================================


def test_w01_pipeline_calls_chain_commit_on_success():
    """Source check: _run_agentic_pipeline calls
    _chain_child_auto_commit when the task has a parent_task_id."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    assert "_chain_child_auto_commit(" in src


def _call_site_idx(src: str) -> int:
    """Find the CALL site of _chain_child_auto_commit (await ...),
    not the definition. Definition appears earlier in the file."""
    return src.find("await _chain_child_auto_commit(")


def test_w02_chain_commit_only_in_success_branch():
    """The chain-commit call must be inside the
    'if solved_by == "qwen_agent"' (success) branch -- NOT in the
    qwen_failed branch (failed children should NOT auto-commit
    a partial recipe)."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    success_idx = src.find('solved_by = "qwen_agent"')
    failed_idx = src.find('solved_by = "qwen_failed"')
    chain_idx = _call_site_idx(src)
    assert success_idx > 0 and failed_idx > 0 and chain_idx > 0
    assert success_idx < chain_idx < failed_idx, (
        f"chain-commit call must be between success and failed "
        f"branches (success={success_idx} chain={chain_idx} "
        f"failed={failed_idx})"
    )


def test_w03_chain_commit_gated_on_parent_task_id():
    """Must check parent_task_id is non-NULL before committing
    (so standalone /code doesn't auto-commit)."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = _call_site_idx(src)
    window = src[max(0, idx - 800):idx]
    assert "parent_task_id" in window


def test_w04_chain_commit_uses_recipe_paths():
    """Must pass _extract_recipe_paths(recipe), not the wide
    _COMMIT_INCLUDE scope."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = _call_site_idx(src)
    window = src[idx:idx + 800]
    assert "_extract_recipe_paths" in window


# ============================================================
# Group R: bot-side child relay
# ============================================================


def test_r01_relay_method_exists():
    from interfaces.telegram_bot import SentinelTelegramBot
    assert hasattr(SentinelTelegramBot, "_relay_chain_children")
    assert hasattr(SentinelTelegramBot, "_send_chain_child_result")


def test_r02_handle_code_calls_relay_on_chain_started():
    """After parent task completes with chain_started, handle_code
    must invoke _relay_chain_children."""
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    idx = src.find("async def handle_code(")
    body = src[idx:idx + 5000]
    assert '"chain_started"' in body
    assert "_relay_chain_children(" in body


def test_r03_relay_polls_list_children():
    """The polling mechanism must use database.list_children(parent_task_id)."""
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    idx = src.find("async def _relay_chain_children(")
    body = src[idx:idx + 3000]
    assert "list_children" in body
    assert "parent_task_id" in body


def test_r04_relay_dedups_already_relayed():
    """Must NOT re-send the same child's completion twice when the
    poll loop sees it again."""
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    idx = src.find("async def _relay_chain_children(")
    body = src[idx:idx + 3000]
    assert "already_relayed" in body


def test_r05_send_child_uses_subtask_marker():
    """Each relayed child message should be prefixed with a
    'Subtask' marker so chat readers know it's a chain child."""
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    idx = src.find("async def _send_chain_child_result(")
    body = src[idx:idx + 2000]
    assert "Subtask" in body


# ============================================================
# Group V: /revert chain
# ============================================================


def test_v01_handle_revert_dispatches_to_chain():
    """handle_revert must dispatch to _handle_revert_chain when
    args[0] == 'chain'."""
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    idx = src.find("async def handle_revert(")
    body = src[idx:idx + 1500]
    assert '"chain"' in body
    assert "_handle_revert_chain(" in body


def test_v02_revert_chain_filters_on_email_AND_message():
    """The chain detection must require BOTH the email AND the
    'sentinel-chain:' message prefix -- not just one. Defensive
    against accidentally matching unrelated commits."""
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    idx = src.find("async def _handle_revert_chain(")
    body = src[idx:idx + 3000]
    assert "chain-child@sentinel.local" in body
    assert "sentinel-chain:" in body


def test_v03_revert_chain_uses_hard_reset_not_soft():
    """For chain revert we want the working tree to also revert
    (so the chain edits are gone from disk too, not just unstaged
    like the normal /revert)."""
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    idx = src.find("async def _handle_revert_chain(")
    body = src[idx:idx + 3000]
    assert '"reset", "--hard"' in body or 'reset --hard' in body


def test_v04_revert_chain_handles_no_chain_at_head():
    """When HEAD is not a chain commit, must reply with a friendly
    'nothing to revert' instead of doing a destructive reset."""
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    idx = src.find("async def _handle_revert_chain(")
    body = src[idx:idx + 3000]
    assert "Nothing to" in body or "nothing to" in body.lower()


# ============================================================
# Group I: import sanity
# ============================================================


def test_i01_imports_clean():
    from skills import code_assist  # noqa: F401
    from interfaces import telegram_bot  # noqa: F401
    from core import database  # noqa: F401
