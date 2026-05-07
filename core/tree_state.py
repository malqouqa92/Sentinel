"""Phase 16 Batch C -- working-tree snapshot/restore via tempfile diff.

Replaces the stash-based dance from Phase 14b/15e graduation. The
stash dance silently failed in production (Phase 15e regression) and
auto-commits are now permanently disabled (NO AUTO-COMMITS owner
directive 2026-05-06), so a different mechanism is needed for the
Batch C replay-skip path.

Approach: tempfile.mkstemp() + git diff > tmp.patch + git checkout
to reset tracked changes; restore via git apply. Untracked files
and runtime artifacts (knowledge.db, logs/) are NOT touched -- per
Phase 15f those are git-untracked, so `git checkout -- .` ignores
them.

Two-step contract:
    handle = await snapshot_dirty_tree(project_root)
    try:
        # do replay / experiment
        ...
    finally:
        result = await restore_dirty_tree(handle)
        # result.restored is True on clean apply, False on
        # apply-reject (in which case result.leftover_patch is the
        # saved patch path the user can use to merge manually).
"""
from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SnapshotHandle:
    patch_path: Path | None
    had_dirty: bool
    project_root: Path
    captured_files: list[str]


@dataclass
class RestoreResult:
    restored: bool
    reason: str
    leftover_patch: Path | None = None


async def _git(*args: str, cwd: Path) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return (
        proc.returncode or 0,
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


async def snapshot_dirty_tree(
    project_root: Path,
    paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> SnapshotHandle:
    """Capture tracked dirty changes to a temp patch file, then reset
    the tracked tree to HEAD. Untracked + gitignored files are left
    alone. No-op when the tree has no tracked changes.

    Phase 16 Batch C IMPORTANT: this function does NOT touch
    untracked files. Specifically, brand-new files written during a
    development session (e.g. core/tree_state.py while it was being
    drafted) are SAFE -- snapshot/checkout never affects them.
    Only modifications to files already tracked at HEAD are touched.

    Phase 17i (2026-05-07): ``exclude_paths`` lets the caller skip
    files matching the listed paths. Critical for graduation: the
    recipe being graduated will be REPLAYED in the clean tree, so
    its target files MUST be excluded from the snapshot -- otherwise
    restore_dirty_tree tries to apply the recipe's edits ON TOP of
    the replay's recreated edits and rejects with patch conflict.
    Live trigger 2026-05-07 ~03:43Z: /code orange edit was wiped
    when graduation snapshot included the orange edit + an unrelated
    pending core/progress.py deletion; restore failed on patch
    conflict, graduation reset tree, both edits lost.
    """
    project_root = Path(project_root)

    rc, status_out, status_err = await _git(
        "status", "--porcelain", cwd=project_root,
    )
    if rc != 0:
        raise RuntimeError(
            f"git status failed in {project_root}: {status_err.strip()}"
        )

    tracked_lines: list[str] = []
    for line in status_out.splitlines():
        if not line:
            continue
        code = line[:2]
        if code in ("??", "!!"):
            continue
        rel = line[3:]
        # Phase 16 Batch C: when scoped to specific paths, only
        # include rows whose path is under one of them. Prevents
        # the snapshot patch from containing files _git_reset_hard
        # won't touch (which would later cause apply-collision).
        if paths and not any(
            rel == p or rel.startswith(p.rstrip('/') + '/')
            for p in paths
        ):
            continue
        # Phase 17i: explicit exclude. When the caller knows certain
        # files will be RECREATED by a subsequent operation (e.g.,
        # graduation's recipe replay), excluding them here prevents
        # the restore_dirty_tree apply-conflict.
        if exclude_paths and any(
            rel == p or rel.startswith(p.rstrip('/') + '/')
            for p in exclude_paths
        ):
            continue
        tracked_lines.append(rel)

    # Phase 17i: even when all dirty is excluded, we MUST still run
    # the cleanup checkout below so the caller's reset/replay starts
    # clean. Don't early-return -- skip only patch-creation. Test
    # case s08: exclude_paths catches the only dirty file -> we still
    # need to checkout HEAD on it so graduation sees a clean tree.
    if not tracked_lines:
        # No patch to write; still cleanup the broad scope below.
        rc, _, co_err = await _git(
            "checkout", "HEAD", "--",
            *(paths if paths else ["."]),
            cwd=project_root,
        )
        if rc != 0:
            raise RuntimeError(f"git checkout failed: {co_err.strip()}")
        return SnapshotHandle(
            patch_path=None, had_dirty=False,
            project_root=project_root, captured_files=[],
        )

    # Phase 17i: diff EXACTLY the captured files (post-include +
    # post-exclude filtering), not the broad `paths` scope. Without
    # this, the diff would include files we just excluded above.
    rc, diff_out, diff_err = await _git(
        "diff", "HEAD", "--binary",
        "--", *tracked_lines,
        cwd=project_root,
    )
    if rc != 0:
        raise RuntimeError(f"git diff HEAD failed: {diff_err.strip()}")

    fd, tmp_path_str = tempfile.mkstemp(
        prefix="sentinel_snap_", suffix=".patch",
    )
    tmp_path = Path(tmp_path_str)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(diff_out)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    # Phase 17i: checkout the broad `paths` scope (or '.') to clean
    # ALL dirty tracked files in scope -- the caller's reset/replay
    # cycle expects a clean tree, not a partial one. The excluded
    # files get checkout'd back to HEAD too (so the recipe replay
    # starts clean); they don't appear in the snapshot patch though,
    # so restore won't try to reapply them.
    rc, _, co_err = await _git(
        "checkout", "HEAD", "--",
        *(paths if paths else ["."]),
        cwd=project_root,
    )
    if rc != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"git checkout failed: {co_err.strip()}")

    return SnapshotHandle(
        patch_path=tmp_path, had_dirty=True,
        project_root=project_root, captured_files=tracked_lines,
    )


async def restore_dirty_tree(handle: SnapshotHandle) -> RestoreResult:
    """Reapply the saved patch onto the current working tree.

    Idempotent on clean handles. On apply-reject: keep replay state
    AND patch on disk (returned via leftover_patch). On clean apply:
    delete temp patch."""
    if not handle.had_dirty or handle.patch_path is None:
        return RestoreResult(
            restored=True, reason="no dirty state to restore",
            leftover_patch=None,
        )

    if not handle.patch_path.exists():
        return RestoreResult(
            restored=False,
            reason=(
                f"patch file vanished at {handle.patch_path}; "
                f"replay state kept; original dirty changes lost"
            ),
            leftover_patch=None,
        )

    rc, _, err = await _git(
        "apply", "--whitespace=nowarn", str(handle.patch_path),
        cwd=handle.project_root,
    )
    if rc == 0:
        handle.patch_path.unlink(missing_ok=True)
        return RestoreResult(
            restored=True,
            reason=(
                f"patch applied cleanly "
                f"({len(handle.captured_files)} file(s))"
            ),
            leftover_patch=None,
        )

    return RestoreResult(
        restored=False,
        reason=(
            f"git apply rejected: {err.strip()[:200]}; "
            f"replay state preserved; patch saved at "
            f"{handle.patch_path}"
        ),
        leftover_patch=handle.patch_path,
    )


# ─────────────────────────────────────────────────────────────────────
# Phase 16 Batch C surgical revert helpers.
#
# Replaces the scoped `git checkout HEAD -- core skills ...` pattern
# (which wiped the author's uncommitted source mid-session during
# Batch C development -- 2026-05-06 lesson) with a path-list approach.
# Caller passes ONLY the files the recipe actually edited; nothing
# else is touched.
# ─────────────────────────────────────────────────────────────────────


async def surgical_revert(
    project_root: Path, paths: list[str],
) -> tuple[int, list[str], list[str]]:
    """Revert specific paths. For each path:
      * if tracked at HEAD: `git checkout HEAD -- <path>`
      * if not tracked (recipe created it new): unlink the file

    Returns ``(rc, reverted_existing, removed_new)`` where rc=0 only
    if all operations succeeded. Idempotent per-path: a missing file
    that wasn't tracked is a no-op, not an error.

    DOES NOT touch any path NOT in the input list. This is the safe
    counterpart to ``_git_reset_hard`` for skip-path failure cleanup
    -- never wipes the author's uncommitted source.
    """
    project_root = Path(project_root)
    reverted: list[str] = []
    removed: list[str] = []
    overall_rc = 0
    for path in paths:
        path_str = str(path).replace("\\", "/")
        # Is it tracked at HEAD?
        rc_ls, _, _ = await _git(
            "ls-files", "--error-unmatch", path_str,
            cwd=project_root,
        )
        if rc_ls == 0:
            # Tracked: revert via checkout.
            rc_co, _, err_co = await _git(
                "checkout", "HEAD", "--", path_str,
                cwd=project_root,
            )
            if rc_co == 0:
                reverted.append(path_str)
            else:
                overall_rc = max(overall_rc, rc_co)
        else:
            # Not tracked: if file exists, remove it.
            full_path = project_root / path_str
            try:
                if full_path.exists():
                    full_path.unlink()
                    removed.append(path_str)
            except OSError:
                overall_rc = max(overall_rc, 1)
    return overall_rc, reverted, removed
