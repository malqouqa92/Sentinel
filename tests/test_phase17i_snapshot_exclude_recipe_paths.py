"""Phase 17i -- snapshot_dirty_tree exclude_paths to prevent
graduation/restore conflicts on recipe-touched files.

Live trigger 2026-05-07 ~03:43Z: /code 'change progress bar to
orange' applied a real edit (BAR_FILLED_EMOJI 🟦 -> 🟠), reviewer
PASSED, then graduation snapshot captured BOTH the orange edit AND
an unrelated pending core/progress.py deletion. Replay recreated
the orange edit in the clean tree. restore_dirty_tree tried to
apply the snapshot patch (which had orange-edit hunks based on
pre-replay HEAD content) on top of post-replay state -- patch
conflict on core/progress.py:1 -- restore failed. Net: BOTH the
orange edit AND the progress.py deletion were lost.

Fix: snapshot_dirty_tree gains exclude_paths param. kb_graduation
passes recipe_paths_for_exclude (extracted from the recipe being
graduated) so the snapshot only carries UNRELATED dirty work --
recipe paths are skipped because replay recreates them anyway.

Two groups:
  S -- snapshot_dirty_tree exclude_paths behavior
  G -- kb_graduation wires recipe paths through to exclude_paths
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
def repo(tmp_path):
    """Tmp git repo with a tracked baseline. Tests can dirty + snapshot."""
    _git("init", "-q", cwd=tmp_path)
    _git("-c", "user.email=a@b", "-c", "user.name=t",
         "commit", "--allow-empty", "-m", "init", cwd=tmp_path)
    (tmp_path / "core").mkdir()
    (tmp_path / "skills").mkdir()
    (tmp_path / "core" / "config.py").write_text(
        "ORIGINAL = True\n", encoding="utf-8",
    )
    (tmp_path / "skills" / "foo.py").write_text(
        "def foo(): return 1\n", encoding="utf-8",
    )
    _git("add", "core/config.py", "skills/foo.py", cwd=tmp_path)
    _git("-c", "user.email=a@b", "-c", "user.name=t",
         "commit", "-m", "baseline", cwd=tmp_path)
    return tmp_path


# ============================================================
# Group S: snapshot_dirty_tree exclude_paths
# ============================================================


def test_s01_signature_has_exclude_paths():
    import inspect
    from core.tree_state import snapshot_dirty_tree
    sig = inspect.signature(snapshot_dirty_tree)
    assert "exclude_paths" in sig.parameters
    p = sig.parameters["exclude_paths"]
    assert p.default is None  # backwards-compat default


def test_s02_no_exclude_captures_all_dirty_in_paths(repo):
    """Without exclude_paths, snapshot captures both dirty files."""
    from core.tree_state import snapshot_dirty_tree
    # Dirty BOTH files
    (repo / "core" / "config.py").write_text("X=1\n", encoding="utf-8")
    (repo / "skills" / "foo.py").write_text("def foo(): return 99\n",
                                              encoding="utf-8")
    handle = asyncio.run(snapshot_dirty_tree(repo, paths=["core", "skills"]))
    assert handle.had_dirty
    assert sorted(handle.captured_files) == [
        "core/config.py", "skills/foo.py",
    ]


def test_s03_exclude_paths_skips_listed_files(repo):
    """Recipe paths excluded -> snapshot has only unrelated dirty work."""
    from core.tree_state import snapshot_dirty_tree
    (repo / "core" / "config.py").write_text("X=1\n", encoding="utf-8")
    (repo / "skills" / "foo.py").write_text("def foo(): return 99\n",
                                              encoding="utf-8")
    # Simulate: recipe touched core/config.py; graduation will replay
    # that. Exclude it so restore won't conflict.
    handle = asyncio.run(snapshot_dirty_tree(
        repo, paths=["core", "skills"],
        exclude_paths=["core/config.py"],
    ))
    assert handle.had_dirty
    # Only skills/foo.py captured
    assert handle.captured_files == ["skills/foo.py"]


def test_s04_exclude_paths_with_dir_prefix(repo):
    """Excluding 'core/' should skip everything under core/."""
    from core.tree_state import snapshot_dirty_tree
    (repo / "core" / "config.py").write_text("X=1\n", encoding="utf-8")
    (repo / "skills" / "foo.py").write_text("def foo(): return 99\n",
                                              encoding="utf-8")
    handle = asyncio.run(snapshot_dirty_tree(
        repo, paths=["core", "skills"],
        exclude_paths=["core"],
    ))
    assert handle.captured_files == ["skills/foo.py"]


def test_s05_all_paths_excluded_returns_no_dirty(repo):
    """Exclude all dirty -> handle.had_dirty=False (no patch needed)."""
    from core.tree_state import snapshot_dirty_tree
    (repo / "core" / "config.py").write_text("X=1\n", encoding="utf-8")
    handle = asyncio.run(snapshot_dirty_tree(
        repo, paths=["core", "skills"],
        exclude_paths=["core/config.py"],
    ))
    assert handle.had_dirty is False


def test_s06_exclude_with_no_paths_filter(repo):
    """exclude_paths works even when paths=None (full-tree scan)."""
    from core.tree_state import snapshot_dirty_tree
    (repo / "core" / "config.py").write_text("X=1\n", encoding="utf-8")
    (repo / "skills" / "foo.py").write_text("def foo(): return 99\n",
                                              encoding="utf-8")
    handle = asyncio.run(snapshot_dirty_tree(
        repo, paths=None, exclude_paths=["core/config.py"],
    ))
    assert handle.captured_files == ["skills/foo.py"]


def test_s07_diff_only_includes_captured_files(repo):
    """The patch file itself must contain ONLY hunks for captured
    files -- not for excluded ones. Otherwise restore would still
    try to apply the excluded file's hunks."""
    from core.tree_state import snapshot_dirty_tree
    (repo / "core" / "config.py").write_text("X=1\n", encoding="utf-8")
    (repo / "skills" / "foo.py").write_text("def foo(): return 99\n",
                                              encoding="utf-8")
    handle = asyncio.run(snapshot_dirty_tree(
        repo, paths=["core", "skills"],
        exclude_paths=["core/config.py"],
    ))
    patch_text = handle.patch_path.read_text(encoding="utf-8")
    assert "skills/foo.py" in patch_text
    assert "core/config.py" not in patch_text


def test_s08_excluded_file_still_checkout_to_HEAD(repo):
    """Excluded files DO get checked out to HEAD (so the caller's
    reset/replay starts clean) -- they're just not in the snapshot
    patch. This is critical: graduation's reset wipes everything in
    the broad path scope; excluded files come back via reset, then
    replay recreates them. Restore reapplies only unrelated dirty."""
    from core.tree_state import snapshot_dirty_tree
    (repo / "core" / "config.py").write_text("X=1\n", encoding="utf-8")
    asyncio.run(snapshot_dirty_tree(
        repo, paths=["core", "skills"],
        exclude_paths=["core/config.py"],
    ))
    # After snapshot, even the excluded file should be at HEAD state
    # (because checkout HEAD -- core/ skills/ ran).
    assert (repo / "core" / "config.py").read_text(encoding="utf-8") \
        == "ORIGINAL = True\n"


def test_s09_full_replay_restore_cycle_no_conflict(repo):
    """End-to-end simulation of the live failure scenario:
    1. /code edits core/config.py (the recipe's path)
    2. Unrelated dirty: skills/foo.py changed
    3. snapshot with exclude_paths=['core/config.py']
    4. (graduation reset would wipe both)
    5. (graduation replay recreates core/config.py)
    6. restore reapplies snapshot
    Expected: restore succeeds (no conflict) AND skills/foo.py is back."""
    import asyncio
    from core.tree_state import (
        restore_dirty_tree, snapshot_dirty_tree,
    )
    # Step 1: recipe-edit
    (repo / "core" / "config.py").write_text(
        "RECIPE_EDIT = True\n", encoding="utf-8",
    )
    # Step 2: unrelated dirty
    (repo / "skills" / "foo.py").write_text(
        "def foo(): return 'unrelated'\n", encoding="utf-8",
    )
    # Step 3: snapshot excluding recipe path
    handle = asyncio.run(snapshot_dirty_tree(
        repo, paths=["core", "skills"],
        exclude_paths=["core/config.py"],
    ))
    # Step 4 (simulate graduation reset): tree is now clean at HEAD
    assert (repo / "core" / "config.py").read_text(encoding="utf-8") \
        == "ORIGINAL = True\n"
    # Step 5 (simulate replay recreating recipe edit):
    (repo / "core" / "config.py").write_text(
        "RECIPE_EDIT = True\n", encoding="utf-8",
    )
    # Step 6: restore -- should NOT conflict since core/config.py
    # is NOT in snapshot patch.
    result = asyncio.run(restore_dirty_tree(handle))
    assert result.restored is True
    # Both edits should be in working tree now
    assert (repo / "core" / "config.py").read_text(encoding="utf-8") \
        == "RECIPE_EDIT = True\n"
    assert (repo / "skills" / "foo.py").read_text(encoding="utf-8") \
        == "def foo(): return 'unrelated'\n"


# ============================================================
# Group G: kb_graduation wiring
# ============================================================


def test_g01_kb_graduation_extracts_recipe_paths():
    """kb_graduation source must call _extract_recipe_paths(recipe)
    and pass result as exclude_paths to snapshot_dirty_tree."""
    src = (PROJECT / "core" / "kb_graduation.py").read_text(encoding="utf-8")
    assert "_extract_recipe_paths(recipe)" in src
    assert "exclude_paths=" in src


def test_g02_kb_graduation_passes_exclude_to_snapshot():
    """Specifically: the snapshot_dirty_tree call must include the
    exclude_paths kwarg derived from the recipe."""
    src = (PROJECT / "core" / "kb_graduation.py").read_text(encoding="utf-8")
    snap_idx = src.find("snapshot_dirty_tree(")
    assert snap_idx > 0
    body = src[snap_idx:snap_idx + 800]
    assert "exclude_paths=" in body


def test_g03_kb_graduation_recipe_paths_extraction_is_best_effort():
    """If _extract_recipe_paths import fails or raises, graduation
    must NOT crash -- just falls back to no exclusion (empty list)."""
    src = (PROJECT / "core" / "kb_graduation.py").read_text(encoding="utf-8")
    # The try/except around the import + extraction
    extract_idx = src.find("_extract_recipe_paths")
    assert extract_idx > 0
    # Within ~300 chars before, must have a try:
    window = src[max(0, extract_idx - 300):extract_idx]
    assert "try:" in window


# ============================================================
# Group I: imports
# ============================================================


def test_i01_imports_clean():
    from core import tree_state  # noqa
    from core import kb_graduation  # noqa
    from skills.code_assist import _extract_recipe_paths  # noqa
