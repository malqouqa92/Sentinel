"""Phase 17j -- _git_reset_hard exclude_paths so graduation's
finally block doesn't wipe the recipe outcome.

Live trigger 2026-05-07 ~04:09Z: /code added /prompt handler to
interfaces/telegram_bot.py and created workspace/persona/PROMPT_BRIEF.md.
Reviewer PASSED. Phase 17i snapshot/replay/restore correctly produced
the recipe outcome in working tree. Then graduation's finally block
ran `_git_reset_hard(pre_grad_sha, trace_id)` which scope-blasted
core/skills/agents/tests/interfaces -- wiping the handler edit.
PROMPT_BRIEF.md survived because workspace/persona is outside the
reset scope. Net: half the work landed, half got wiped.

Fix: _git_reset_hard accepts exclude_paths (list of paths to skip).
kb_graduation passes recipe_paths_for_exclude (extracted earlier
in the same function for snapshot_dirty_tree's exclude_paths) so
the same paths are skipped on the final reset.

Two groups:
  R -- _git_reset_hard exclude_paths behavior
  G -- kb_graduation finally block wires recipe paths through
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent


def _git(*args, cwd):
    p = subprocess.run(["git", *args], cwd=str(cwd),
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Tmp git repo with tracked baseline + monkeypatch PROJECT_ROOT."""
    _git("init", "-q", cwd=tmp_path)
    _git("-c", "user.email=a@b", "-c", "user.name=t",
         "commit", "--allow-empty", "-m", "init", cwd=tmp_path)
    for d in ("core", "skills", "interfaces", "agents", "tests"):
        (tmp_path / d).mkdir()
    # Add a placeholder file in each scope dir so they all exist in
    # HEAD (git pathspec rejects unknown paths).
    (tmp_path / "agents" / ".keep").write_text("", encoding="utf-8")
    (tmp_path / "tests" / ".keep").write_text("", encoding="utf-8")
    (tmp_path / "skills" / ".keep").write_text("", encoding="utf-8")
    (tmp_path / "core" / "config.py").write_text(
        "ORIGINAL = True\n", encoding="utf-8",
    )
    (tmp_path / "interfaces" / "bot.py").write_text(
        "BOT_ORIGINAL = True\n", encoding="utf-8",
    )
    _git("add", ".", cwd=tmp_path)
    _git("-c", "user.email=a@b", "-c", "user.name=t",
         "commit", "-m", "baseline", cwd=tmp_path)
    from core import config
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    return tmp_path


# ============================================================
# Group R: _git_reset_hard exclude_paths
# ============================================================


def test_r01_signature_has_exclude_paths():
    import inspect
    from skills.code_assist import _git_reset_hard
    sig = inspect.signature(_git_reset_hard)
    assert "exclude_paths" in sig.parameters
    p = sig.parameters["exclude_paths"]
    assert p.default is None  # backwards-compat default


def test_r02_no_exclude_resets_everything_in_scope(repo):
    """Without exclude_paths, _git_reset_hard wipes ALL dirty work
    in the scope dirs. (Pre-17j behavior.)"""
    from skills.code_assist import _git_reset_hard
    # Get current HEAD
    rc, head = _git("rev-parse", "HEAD", cwd=repo)
    head_sha = head.strip()
    # Dirty edit
    (repo / "core" / "config.py").write_text("MODIFIED = True\n", encoding="utf-8")
    asyncio.run(_git_reset_hard(head_sha, "SEN-test"))
    # Edit gone
    assert (repo / "core" / "config.py").read_text(encoding="utf-8") \
        == "ORIGINAL = True\n"


def test_r03_exclude_paths_preserves_listed_files(repo):
    """The headline fix: with exclude_paths, files in the list
    are NOT touched by the reset."""
    from skills.code_assist import _git_reset_hard
    rc, head = _git("rev-parse", "HEAD", cwd=repo)
    head_sha = head.strip()
    (repo / "core" / "config.py").write_text(
        "RECIPE_EDIT = True\n", encoding="utf-8",
    )
    (repo / "interfaces" / "bot.py").write_text(
        "OTHER_DIRTY = True\n", encoding="utf-8",
    )
    # Exclude only core/config.py -- it should survive
    asyncio.run(_git_reset_hard(
        head_sha, "SEN-test",
        exclude_paths=["core/config.py"],
    ))
    # core/config.py preserved, interfaces/bot.py reset
    assert (repo / "core" / "config.py").read_text(encoding="utf-8") \
        == "RECIPE_EDIT = True\n"
    assert (repo / "interfaces" / "bot.py").read_text(encoding="utf-8") \
        == "BOT_ORIGINAL = True\n"


def test_r04_exclude_paths_handles_new_untracked_files(repo):
    """git clean -fd should also honor the exclusion -- new
    untracked files in the excluded path should survive."""
    from skills.code_assist import _git_reset_hard
    rc, head = _git("rev-parse", "HEAD", cwd=repo)
    head_sha = head.strip()
    (repo / "skills" / "newfile.py").write_text(
        "def new(): pass\n", encoding="utf-8",
    )
    asyncio.run(_git_reset_hard(
        head_sha, "SEN-test",
        exclude_paths=["skills/newfile.py"],
    ))
    assert (repo / "skills" / "newfile.py").exists()


def test_r05_exclude_with_no_dirty_in_scope_is_noop(repo):
    """Edge: clean tree + excludes -> nothing changes, no crash."""
    from skills.code_assist import _git_reset_hard
    rc, head = _git("rev-parse", "HEAD", cwd=repo)
    head_sha = head.strip()
    asyncio.run(_git_reset_hard(
        head_sha, "SEN-test",
        exclude_paths=["core/config.py"],
    ))
    # Files unchanged
    assert (repo / "core" / "config.py").read_text(encoding="utf-8") \
        == "ORIGINAL = True\n"


def test_r06_multiple_exclude_paths(repo):
    from skills.code_assist import _git_reset_hard
    rc, head = _git("rev-parse", "HEAD", cwd=repo)
    head_sha = head.strip()
    (repo / "core" / "config.py").write_text(
        "EDIT_A = True\n", encoding="utf-8",
    )
    (repo / "interfaces" / "bot.py").write_text(
        "EDIT_B = True\n", encoding="utf-8",
    )
    asyncio.run(_git_reset_hard(
        head_sha, "SEN-test",
        exclude_paths=["core/config.py", "interfaces/bot.py"],
    ))
    # Both preserved
    assert (repo / "core" / "config.py").read_text(encoding="utf-8") \
        == "EDIT_A = True\n"
    assert (repo / "interfaces" / "bot.py").read_text(encoding="utf-8") \
        == "EDIT_B = True\n"


def test_r07_legacy_callers_unaffected(repo):
    """Pre-17j callers that pass only (base_sha, trace_id) must
    keep working identically -- no regression to the broad reset."""
    from skills.code_assist import _git_reset_hard
    rc, head = _git("rev-parse", "HEAD", cwd=repo)
    head_sha = head.strip()
    (repo / "core" / "config.py").write_text(
        "DIRTY = True\n", encoding="utf-8",
    )
    # Legacy 2-arg call
    asyncio.run(_git_reset_hard(head_sha, "SEN-legacy"))
    # Wiped (no exclude_paths)
    assert (repo / "core" / "config.py").read_text(encoding="utf-8") \
        == "ORIGINAL = True\n"


# ============================================================
# Group G: kb_graduation wires recipe paths to finally reset
# ============================================================


def test_g01_kb_graduation_finally_passes_exclude_paths():
    """Source-level: the finally block's _git_reset_hard call must
    pass exclude_paths derived from recipe_paths_for_exclude (same
    list used by Phase 17i for snapshot_dirty_tree)."""
    src = (PROJECT / "core" / "kb_graduation.py").read_text(encoding="utf-8")
    # Find the finally block
    finally_idx = src.find("finally:")
    assert finally_idx > 0
    body = src[finally_idx:finally_idx + 3000]
    assert "_git_reset_hard(" in body
    assert "exclude_paths=" in body
    assert "recipe_paths_for_exclude" in body


def test_g02_kb_graduation_uses_same_recipe_paths_for_both_exclusions():
    """The variable recipe_paths_for_exclude is computed ONCE near
    the snapshot_dirty_tree call (Phase 17i) and reused in the
    finally block (Phase 17j). One source of truth."""
    src = (PROJECT / "core" / "kb_graduation.py").read_text(encoding="utf-8")
    # Both occurrences exist
    assert src.count("recipe_paths_for_exclude") >= 2
    # _extract_recipe_paths called only once (for that variable)
    assert src.count("_extract_recipe_paths(recipe)") == 1


def test_g03_finally_reset_failure_path_logs_warning():
    """Pre-17j path had `except Exception as e: log ERROR`.
    Phase 17j must keep that error logging."""
    src = (PROJECT / "core" / "kb_graduation.py").read_text(encoding="utf-8")
    finally_idx = src.find("finally:")
    body = src[finally_idx:finally_idx + 3000]
    # Reset call wrapped in try/except with FAILED log
    assert "FAILED to restore pre-grad SHA" in body


# ============================================================
# Group I: integration sanity
# ============================================================


def test_i01_imports_clean():
    from skills import code_assist  # noqa
    from core import kb_graduation  # noqa


def test_i02_exclude_paths_uses_read_aside_restore():
    """Source-level: _git_reset_hard implements exclusion via
    read-aside-then-restore (read_bytes/write_bytes around the
    reset) NOT git pathspec `:(exclude)` syntax. Pathspec was
    tried first but fought cmd.exe shell parsing on Windows
    (live trigger 2026-05-07 04:18Z: `pathspec 'core' did not
    match any file(s) known to git`)."""
    src = (PROJECT / "skills" / "code_assist.py").read_text(encoding="utf-8")
    idx = src.find("async def _git_reset_hard")
    body = src[idx:idx + 6000]
    assert "read_bytes()" in body
    assert "write_bytes(" in body
    # The actual `git checkout` command must NOT be wrapped in
    # `:(exclude)<path>` magic syntax. (The comment block may still
    # mention `:(exclude)` to explain why we DON'T use it; we check
    # the cmds list construction specifically.)
    cmds_idx = body.find("cmds = [")
    assert cmds_idx > 0
    cmds_block = body[cmds_idx:cmds_idx + 300]
    assert ":(exclude)" not in cmds_block
