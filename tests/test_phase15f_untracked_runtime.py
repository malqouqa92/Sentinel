"""Phase 15f -- runtime files must NEVER be tracked by git.

The Phase 15e graduation flow added a `_git_commit_for_graduation`
that commits /code's working-tree changes. Graduation then calls
`git stash push -u` to stash any leftover dirty files. If the
working tree contains tracked-but-gitignored files (knowledge.db,
sentinel.db, logs/sentinel.jsonl), stash's internal "reset working
tree to HEAD" step iterates them and overwrites each one with the
HEAD blob. On Windows, when one of those files is held open by the
bot (logs/sentinel.jsonl), the unlink fails and stash bails -- but
files it already overwrote stay overwritten. Phase 15e shipped on
top of this latent bug; the FIRST /code under 15e wiped knowledge.db
back to a Phase-9-era schema.

Phase 15f untracks every file that should never have been in the
index in the first place. Permanent fix: even if a future stash
fails the same way, there are no SQLite databases or active log
files in the working tree for it to overwrite.

Coverage:
  Critical regression vectors:
    F01 -- knowledge.db is NOT tracked by git
    F02 -- sentinel.db is NOT tracked by git
    F03 -- memory.db is NOT tracked by git
    F04 -- logs/sentinel.jsonl is NOT tracked by git

  Wider cleanup:
    F11 -- no .pyc files are tracked by git
    F12 -- no workspace/job_searches/* output files are tracked
    F13 -- no workspace/research/* output files are tracked

  .gitignore sanity:
    F21 -- *.db pattern still in .gitignore
    F22 -- __pycache__/ pattern still in .gitignore
    F23 -- workspace/job_searches/, workspace/research/ in .gitignore

  Source files that SHOULD stay tracked (regression guard):
    F31 -- core/*.py files are still tracked
    F32 -- skills/*.py files are still tracked
    F33 -- tests/*.py files are still tracked
    F34 -- CLAUDE.md and PHASES.md still tracked
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _git_ls_files() -> set[str]:
    """Return the set of currently-tracked file paths (POSIX-style)
    relative to repo root."""
    r = subprocess.run(
        ["git", "ls-files"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, check=True,
    )
    # Strip any \r on Windows-checked-out lines.
    return {l.rstrip("\r") for l in r.stdout.splitlines() if l.strip()}


def _gitignore_text() -> str:
    return (PROJECT_ROOT / ".gitignore").read_text(
        encoding="utf-8", errors="replace",
    )


# ─────────────────────────────────────────────────────────────────
# Critical regression vectors
# ─────────────────────────────────────────────────────────────────


def test_f01_knowledge_db_not_tracked():
    tracked = _git_ls_files()
    assert "knowledge.db" not in tracked, (
        "knowledge.db is tracked -- a future graduation stash will "
        "overwrite it with HEAD's blob. This was the Phase 15e "
        "regression vector that wiped 28 patterns."
    )


def test_f02_sentinel_db_not_tracked():
    tracked = _git_ls_files()
    assert "sentinel.db" not in tracked


def test_f03_memory_db_not_tracked():
    tracked = _git_ls_files()
    assert "memory.db" not in tracked


def test_f04_log_file_not_tracked():
    """The bot writes logs/sentinel.jsonl while running. If git
    tries to overwrite it via stash/checkout, it fails on Windows
    file locking AND can leave the index inconsistent."""
    tracked = _git_ls_files()
    assert "logs/sentinel.jsonl" not in tracked


# ─────────────────────────────────────────────────────────────────
# Wider cleanup (defense in depth)
# ─────────────────────────────────────────────────────────────────


def test_f11_no_pyc_files_tracked():
    """Pyc bytecode is gitignored by `__pycache__/` and `*.pyc`.
    Tracking them ate ~128 files of index space and made every
    git operation potentially destructive to the bot's running
    state."""
    tracked = _git_ls_files()
    pyc = [p for p in tracked if p.endswith(".pyc")]
    assert pyc == [], (
        f"{len(pyc)} .pyc file(s) still tracked: {pyc[:5]}..."
    )


def test_f12_no_job_search_output_tracked():
    tracked = _git_ls_files()
    js = [p for p in tracked if p.startswith("workspace/job_searches/")]
    assert js == [], (
        f"{len(js)} workspace/job_searches/* file(s) tracked: {js[:3]}"
    )


def test_f13_no_research_output_tracked():
    tracked = _git_ls_files()
    rs = [p for p in tracked if p.startswith("workspace/research/")]
    assert rs == [], (
        f"{len(rs)} workspace/research/* file(s) tracked: {rs[:3]}"
    )


# ─────────────────────────────────────────────────────────────────
# .gitignore sanity (the rules backing the untracking)
# ─────────────────────────────────────────────────────────────────


def test_f21_db_pattern_in_gitignore():
    text = _gitignore_text()
    assert "*.db" in text, "*.db pattern missing from .gitignore"


def test_f22_pycache_pattern_in_gitignore():
    text = _gitignore_text()
    assert "__pycache__/" in text


def test_f23_workspace_output_in_gitignore():
    text = _gitignore_text()
    assert "workspace/job_searches/" in text
    assert "workspace/research/" in text


# ─────────────────────────────────────────────────────────────────
# Source files MUST stay tracked (regression guard against
# accidentally untracking source by glob)
# ─────────────────────────────────────────────────────────────────


def test_f31_core_python_still_tracked():
    tracked = _git_ls_files()
    expected_present = {
        "core/config.py",
        "core/knowledge_base.py",
        "core/memory.py",
        "core/kb_graduation.py",
    }
    missing = expected_present - tracked
    assert not missing, f"core source files vanished from index: {missing}"


def test_f32_skills_python_still_tracked():
    tracked = _git_ls_files()
    assert "skills/code_assist.py" in tracked
    assert "skills/job_score.py" in tracked


def test_f33_tests_python_still_tracked():
    tracked = _git_ls_files()
    # Spot-check the test files we built this session.
    expected = {
        "tests/test_phase15a_lifecycle.py",
        "tests/test_phase15b_provenance.py",
        "tests/test_phase15c_planning.py",
        "tests/test_phase15d_resilience.py",
        "tests/test_phase15e_grad_snapshot.py",
        "tests/test_phase15f_untracked_runtime.py",
    }
    missing = expected - tracked
    assert not missing, f"phase 15 test files missing: {missing}"


def test_f34_docs_still_tracked():
    tracked = _git_ls_files()
    assert "CLAUDE.md" in tracked
    assert "PHASES.md" in tracked
    # Persona files
    assert "workspace/persona/QWENCODER.md" in tracked
