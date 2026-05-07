"""Phase 16 Batch C -- tree-state snapshot/restore + surgical_revert ECC."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from core.tree_state import (
    RestoreResult, SnapshotHandle,
    restore_dirty_tree, snapshot_dirty_tree, surgical_revert,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()

    def run(*args):
        out = subprocess.run(
            ["git", *args], cwd=str(root),
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {out.stderr}"
            )
        return out

    run("init", "-q")
    run("config", "user.email", "test@local")
    run("config", "user.name", "test")
    run("config", "commit.gpgsign", "false")
    (root / "a.txt").write_text("v1\n", encoding="utf-8")
    run("add", "a.txt")
    run("commit", "-q", "-m", "init")
    return root


def _async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ─── Snapshot / restore round-trip ───


def test_t01_snapshot_clean_tree_is_noop(repo):
    h = _async(snapshot_dirty_tree(repo))
    assert h.had_dirty is False
    assert h.patch_path is None


def test_t02_restore_noop_handle_succeeds(repo):
    h = _async(snapshot_dirty_tree(repo))
    r = _async(restore_dirty_tree(h))
    assert r.restored is True


def test_t10_snapshot_captures_and_resets(repo):
    (repo / "a.txt").write_text("v2 dirty\n", encoding="utf-8")
    h = _async(snapshot_dirty_tree(repo))
    assert h.had_dirty is True
    assert h.patch_path is not None and h.patch_path.exists()
    assert (repo / "a.txt").read_text(encoding="utf-8") == "v1\n"


def test_t11_restore_round_trips_byte_for_byte(repo):
    pre = "v2 dirty\nline 2\n"
    (repo / "a.txt").write_text(pre, encoding="utf-8")
    h = _async(snapshot_dirty_tree(repo))
    r = _async(restore_dirty_tree(h))
    assert r.restored is True
    assert (repo / "a.txt").read_text(encoding="utf-8") == pre


def test_t20_snapshot_does_not_touch_untracked(repo):
    """KEY GUARD against the v1 self-wipe: snapshot must NEVER touch
    untracked files. The dev's in-progress source files (tree_state.py
    itself, etc.) are untracked when first written; they MUST survive
    a snapshot/checkout cycle."""
    (repo / "untracked.txt").write_text("WIP code", encoding="utf-8")
    (repo / "a.txt").write_text("v2 dirty\n", encoding="utf-8")
    _async(snapshot_dirty_tree(repo))
    # Tracked file got reset, untracked file untouched.
    assert (repo / "a.txt").read_text(encoding="utf-8") == "v1\n"
    assert (repo / "untracked.txt").read_text(encoding="utf-8") == "WIP code"


def test_t30_apply_reject_keeps_state_and_patch(repo):
    pre = "ORIGINAL DIRTY\nshared\n"
    (repo / "a.txt").write_text(pre, encoding="utf-8")
    h = _async(snapshot_dirty_tree(repo))
    (repo / "a.txt").write_text(
        "REPLAY STATE\nother\n", encoding="utf-8",
    )
    r = _async(restore_dirty_tree(h))
    assert r.restored is False
    assert r.leftover_patch is not None
    assert r.leftover_patch.exists()
    assert "REPLAY STATE" in (repo / "a.txt").read_text(encoding="utf-8")


def test_t40_patch_vanishes_returns_error(repo):
    (repo / "a.txt").write_text("v2\n", encoding="utf-8")
    h = _async(snapshot_dirty_tree(repo))
    h.patch_path.unlink()
    r = _async(restore_dirty_tree(h))
    assert r.restored is False
    assert "vanished" in r.reason.lower()


def test_t41_restore_idempotent_on_clean(repo):
    h = _async(snapshot_dirty_tree(repo))
    r1 = _async(restore_dirty_tree(h))
    r2 = _async(restore_dirty_tree(h))
    assert r1.restored and r2.restored


# ─── surgical_revert ───


def test_s01_surgical_reverts_tracked_file_only(repo):
    """surgical_revert reverts ONLY the listed paths; everything else
    untouched. This is the CRITICAL property that prevents the v1
    self-wipe -- core/tree_state.py and tests/* never get touched
    when only workspace/skip_test/file.py is in the list."""
    (repo / "a.txt").write_text("dirty\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("WIP", encoding="utf-8")
    rc, reverted, removed = _async(
        surgical_revert(repo, ["a.txt"])
    )
    assert rc == 0
    assert "a.txt" in reverted
    # Untracked file must not be touched.
    assert (repo / "untracked.txt").read_text(encoding="utf-8") == "WIP"
    # Tracked file reverted to HEAD.
    assert (repo / "a.txt").read_text(encoding="utf-8") == "v1\n"


def test_s02_surgical_removes_new_untracked_file(repo):
    """When path is NOT tracked at HEAD, surgical_revert REMOVES it
    (the recipe created it new; on failure we want it gone)."""
    (repo / "new_file.py").write_text("from replay\n", encoding="utf-8")
    rc, reverted, removed = _async(
        surgical_revert(repo, ["new_file.py"])
    )
    assert rc == 0
    assert "new_file.py" in removed
    assert not (repo / "new_file.py").exists()


def test_s03_surgical_idempotent_on_missing_path(repo):
    """Path doesn't exist anywhere -- not an error, just a no-op."""
    rc, reverted, removed = _async(
        surgical_revert(repo, ["does_not_exist.py"])
    )
    assert rc == 0
    assert reverted == []
    assert removed == []


def test_s04_surgical_does_not_touch_unrelated_files(repo):
    """The headline guard. Multiple files in repo; surgical_revert is
    asked to revert ONE. All others must survive."""
    (repo / "a.txt").write_text("DIRTY a\n", encoding="utf-8")
    (repo / "b_untracked.txt").write_text("UNTRACKED B", encoding="utf-8")
    (repo / "c_untracked.txt").write_text("UNTRACKED C", encoding="utf-8")
    rc, _r, _rm = _async(surgical_revert(repo, ["a.txt"]))
    assert rc == 0
    # b and c are untracked; they had nothing to revert and should
    # stay.
    assert (repo / "b_untracked.txt").exists()
    assert (repo / "c_untracked.txt").exists()
    assert (repo / "b_untracked.txt").read_text() == "UNTRACKED B"
    assert (repo / "c_untracked.txt").read_text() == "UNTRACKED C"


def test_s05_dataclass_shape():
    h = SnapshotHandle(
        patch_path=Path("/tmp/x"), had_dirty=True,
        project_root=Path("/tmp"), captured_files=["a.py"],
    )
    assert h.had_dirty is True
    r = RestoreResult(restored=True, reason="ok")
    assert r.restored is True
