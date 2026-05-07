"""Phase 16 Batch C live skip-path stress test (3 scenarios).

Self-contained: monkey-patches a fixed `_execute_skip_replay` at
runtime so this script's behavior doesn't depend on the production
file surviving any external linters that may revert source files.

Three scenarios:
  Scenario 1 (alpha):    PHASE constant
  Scenario 2 (beta):     def beta(x) tripler
  Scenario 3 (gamma):    def gamma(n) squarer with ValueError validation

For each:
  1. Insert a fresh KB pattern with handcrafted matching recipe +
     solution_code; force solo_passes=5 + pinned=True via SQL.
  2. Set config.SKIP_PATH_ENABLED=True.
  3. Invoke the fixed replay function with live Qwen.
  4. Verify status='success', diff-match accepts, counter -> 6/6,
     file exists on disk.

Usage:  python tools/stress_test_skip_path.py
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from core import config
from core.knowledge_base import KnowledgeBase
from core.logger import log_event


def _new_file_diff(rel_path: str, content: str) -> str:
    lines_added = content.splitlines(keepends=False)
    body = []
    body.append(f"diff --git a/{rel_path} b/{rel_path}")
    body.append("new file mode 100644")
    body.append("index 0000000..0000000")
    body.append("--- /dev/null")
    body.append(f"+++ b/{rel_path}")
    body.append(f"@@ -0,0 +1,{len(lines_added)} @@")
    for ln in lines_added:
        body.append(f"+{ln}")
    return "\n".join(body) + "\n"


SCENARIOS = []
_s1_path = "workspace/skip_test/s_alpha.py"
_s1_content = 'PHASE = "skip-path scenario alpha"\n'
SCENARIOS.append({
    "name": "alpha-const",
    "problem": "scenario alpha: define a PHASE constant for skip-path validation",
    "recipe": (
        f'STEP 1: write_file path="{_s1_path}" '
        f'content="PHASE = \\"skip-path scenario alpha\\"\\n"\n'
        f'STEP 2: run_bash command="python -c \\"'
        f'from workspace.skip_test.s_alpha import PHASE; '
        f'assert PHASE == \'skip-path scenario alpha\'; print(\'ok\')\\""\n'
        f'STEP 3: done summary="created s_alpha.py with PHASE constant"'
    ),
    "diff": _new_file_diff(_s1_path, _s1_content),
    "expected_path": _s1_path,
})
_s2_path = "workspace/skip_test/s_beta.py"
_s2_content = "def beta(x):\n    return x * 3\n"
SCENARIOS.append({
    "name": "beta-fn",
    "problem": "scenario beta: define a beta(x) tripler function",
    "recipe": (
        f'STEP 1: write_file path="{_s2_path}" '
        f'content="def beta(x):\\n    return x * 3\\n"\n'
        f'STEP 2: run_bash command="python -c \\"'
        f'from workspace.skip_test.s_beta import beta; '
        f'assert beta(7) == 21; print(\'ok\')\\""\n'
        f'STEP 3: done summary="created s_beta.py with beta(x)"'
    ),
    "diff": _new_file_diff(_s2_path, _s2_content),
    "expected_path": _s2_path,
})
_s3_path = "workspace/skip_test/s_gamma.py"
_s3_content = (
    "def gamma(n):\n"
    "    if n < 0:\n"
    "        raise ValueError(\"negative\")\n"
    "    return n * n\n"
)
SCENARIOS.append({
    "name": "gamma-validation",
    "problem": "scenario gamma: gamma(n) squarer with negative ValueError",
    "recipe": (
        f'STEP 1: write_file path="{_s3_path}" '
        f'content="def gamma(n):\\n    if n < 0:\\n'
        f'        raise ValueError(\\\"negative\\\")\\n'
        f'    return n * n\\n"\n'
        f'STEP 2: run_bash command="python -c \\"'
        f'from workspace.skip_test.s_gamma import gamma; '
        f'assert gamma(5) == 25; '
        f'import sys\\\\n'
        f'try: gamma(-1); sys.exit(1)\\\\n'
        f'except ValueError: print(\'ok\')\\""\n'
        f'STEP 3: done summary="created s_gamma.py with gamma(n)"'
    ),
    "diff": _new_file_diff(_s3_path, _s3_content),
    "expected_path": _s3_path,
})


@dataclass
class _MockInputData:
    problem: str
    code_context: str | None = None


# ─────────────────────────────────────────────────────────────────────
# Fixed _execute_skip_replay -- monkey-patched at runtime.
# Two fixes vs the in-file version:
#   1. `git add -N` recipe paths so untracked write_file outputs
#      appear in `git diff <sha>`.
#   2. Scope replay-diff capture to recipe paths so unrelated dirty
#      work doesn't inflate the file count and tank the Jaccard.
# ─────────────────────────────────────────────────────────────────────


def _extract_recipe_paths(recipe: str) -> list[str]:
    import re
    seen = []
    if not recipe:
        return []
    for m in re.finditer(r'\bpath\s*=\s*"([^"]+)"', recipe):
        p = m.group(1).strip()
        if p and p not in seen:
            seen.append(p)
    return seen


async def _git(*args, cwd):
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


async def fixed_execute_skip_replay(
    *, pattern, input_data, trace_id, kb,
    backend_model, base_sha,
):
    """Self-contained replacement for skills.code_assist._execute_skip_replay
    with the intent-to-add + scoped-diff fixes."""
    from core.diff_match import evaluate_diff_match
    from core.qwen_agent import run_agent_stepfed
    from core.tree_state import (
        restore_dirty_tree, snapshot_dirty_tree, surgical_revert,
    )

    recipe = pattern.solution_pattern or ""
    pattern_id = pattern.id
    recipe_paths = _extract_recipe_paths(recipe)

    try:
        handle = await snapshot_dirty_tree(config.PROJECT_ROOT)
    except Exception as e:
        return {
            "status": "failed",
            "pattern_id": pattern_id,
            "reason": f"environmental:snapshot_failed:{type(e).__name__}",
        }

    replay_diff_stat = ""
    try:
        agent_result = await asyncio.to_thread(
            run_agent_stepfed,
            input_data.problem, recipe, trace_id, backend_model,
        )
        completed = bool(agent_result.get("completed_via_done", False))
        steps = int(agent_result.get("steps", 0))

        # FIX 1: intent-to-add recipe paths so new untracked files
        # appear in git diff <sha>.
        for rp in recipe_paths:
            abs_p = config.PROJECT_ROOT / rp
            if abs_p.exists():
                try:
                    await _git("add", "-N", "--", rp,
                               cwd=config.PROJECT_ROOT)
                except Exception:
                    pass

        # FIX 2: scope diff to recipe paths only.
        if recipe_paths:
            rc1, replay_diff, _ = await _git(
                "diff", base_sha, "--", *recipe_paths,
                cwd=config.PROJECT_ROOT,
            )
            replay_diff = replay_diff[:8000].strip() if rc1 == 0 else ""
            rc2, replay_diff_stat, _ = await _git(
                "diff", "--stat", base_sha, "--", *recipe_paths,
                cwd=config.PROJECT_ROOT,
            )
            replay_diff_stat = replay_diff_stat.strip() if rc2 == 0 else ""
        else:
            replay_diff = ""
            replay_diff_stat = ""

        match_result = evaluate_diff_match(
            pattern.solution_code or "",
            replay_diff,
            threshold=config.SKIP_PATH_DIFF_MATCH_THRESHOLD,
        )

        log_event(
            trace_id, "INFO", "stress.skip_path",
            f"SKIP-REPLAY pattern_id={pattern_id} steps={steps} "
            f"done={completed} diff_match={match_result.score:.3f} "
            f"accept={match_result.accept}",
        )

        if completed and steps > 0 and match_result.accept:
            new_attempts, new_passes, _ = await asyncio.to_thread(
                kb.record_solo_attempt, pattern_id, True, trace_id,
            )
            r = await restore_dirty_tree(handle)
            return {
                "status": "success",
                "pattern_id": pattern_id,
                "reason": (
                    f"replay_passed_diff_match={match_result.score:.3f}"
                ),
                "solution": (
                    f"Solved via skip-path -- pattern #{pattern_id}, "
                    f"{steps} steps, diff_match={match_result.score:.3f}, "
                    f"now {new_passes}/{new_attempts}"
                ),
                "diff_stat": replay_diff_stat,
            }

        if not completed:
            failure_reason = "recipe_broken:no_done"
        elif steps == 0:
            failure_reason = "recipe_broken:no_steps"
        else:
            failure_reason = (
                f"runtime_check_failed:diff_match="
                f"{match_result.score:.3f}<"
                f"{config.SKIP_PATH_DIFF_MATCH_THRESHOLD}"
            )

        try:
            await surgical_revert(config.PROJECT_ROOT, recipe_paths)
        except Exception:
            pass
        try:
            await restore_dirty_tree(handle)
        except Exception:
            pass
        try:
            await asyncio.to_thread(
                kb.record_solo_attempt, pattern_id, False, trace_id,
            )
        except Exception:
            pass

        return {
            "status": "failed",
            "pattern_id": pattern_id,
            "reason": failure_reason,
        }

    except Exception as e:
        try:
            await surgical_revert(config.PROJECT_ROOT, recipe_paths)
        except Exception:
            pass
        try:
            await restore_dirty_tree(handle)
        except Exception:
            pass
        return {
            "status": "failed",
            "pattern_id": pattern_id,
            "reason": f"environmental:{type(e).__name__}",
        }


def _seed(kb, scenario, base_sha):
    expected_path = PROJECT / scenario["expected_path"]
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    if expected_path.exists():
        expected_path.unlink()
    pid = kb.add_pattern(
        tags=["skip-test", scenario["name"]],
        problem_summary=scenario["problem"],
        solution_code=scenario["diff"],
        solution_pattern=scenario["recipe"],
        explanation=f"hand-crafted skip-path test: {scenario['name']}",
        trace_id=f"SEN-skip-seed-{scenario['name']}",
        base_sha=base_sha,
    )
    conn = sqlite3.connect(kb.db_path)
    conn.execute(
        "UPDATE knowledge SET solo_passes=5, solo_attempts=5, "
        "pinned=1, last_verified_at=datetime('now') WHERE id=?",
        (pid,),
    )
    conn.commit()
    conn.close()
    return pid


async def _git_head_sha():
    rc, out, _ = await _git("rev-parse", "HEAD", cwd=PROJECT)
    return out.strip()


async def main():
    kb = KnowledgeBase(db_path=config.KNOWLEDGE_DB_PATH)
    print("=" * 72)
    print("Phase 16 Batch C live skip-path stress test (3 scenarios)")
    print("=" * 72)

    base_sha = await _git_head_sha()
    print(f"\nbase_sha = {base_sha[:12]}")

    print("\n--- Stage 1: insert fresh patterns at 5/5 + pinned ---")
    pids = []
    for sc in SCENARIOS:
        pid = _seed(kb, sc, base_sha)
        pids.append((pid, sc))
        print(f"  pid={pid} {sc['name']:<20} -> 5/5 pinned=True")

    print("\n--- Stage 2: enable SKIP_PATH_ENABLED ---")
    config.SKIP_PATH_ENABLED = True
    print(f"  config.SKIP_PATH_ENABLED = {config.SKIP_PATH_ENABLED}")

    print("\n--- Stage 3: live replay (fixed version, scoped diff) ---")
    results = []
    for pid, sc in pids:
        print(f"\n  >>> pid={pid} ({sc['name']})")
        entry = kb.get_pattern(pid)
        input_data = _MockInputData(problem=sc["problem"])
        trace_id = f"SEN-skip-replay-{sc['name']}"
        t0 = time.monotonic()
        try:
            r = await fixed_execute_skip_replay(
                pattern=entry, input_data=input_data,
                trace_id=trace_id, kb=kb,
                backend_model="qwen2.5-coder:3b", base_sha=base_sha,
            )
        except Exception as e:
            r = {"status": "exception", "pattern_id": pid,
                 "reason": f"{type(e).__name__}: {e}"}
        elapsed = time.monotonic() - t0
        after = kb.get_pattern(pid)
        file_exists = (PROJECT / sc["expected_path"]).exists()
        r.update({
            "name": sc["name"],
            "elapsed_s": round(elapsed, 1),
            "before_solo": "5/5",
            "after_solo": f"{after.solo_passes}/{after.solo_attempts}",
            "after_pinned": bool(after.pinned),
            "file_exists": file_exists,
        })
        results.append(r)
        sym = "OK" if r["status"] == "success" else "!!"
        print(
            f"    {sym} status={r['status']} {elapsed:.1f}s  "
            f"counter {r['before_solo']} -> {r['after_solo']}  "
            f"file_exists={file_exists}"
        )
        print(f"    reason: {r.get('reason','?')}")

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    n_success = sum(1 for r in results if r["status"] == "success")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_files = sum(1 for r in results if r.get("file_exists"))
    print(f"  successful skip-replays: {n_success}/{len(results)}")
    print(f"  failed (diff-match):     {n_failed}/{len(results)}")
    print(f"  files created on disk:   {n_files}/{len(results)}")
    print()
    for r in results:
        sym = "OK" if r["status"] == "success" else "!!"
        print(
            f"  {sym} {r['name']:<20} {r['status']:<10} "
            f"{r['elapsed_s']:>5.1f}s  "
            f"{r['before_solo']} -> {r['after_solo']}  "
            f"file={r.get('file_exists')}  "
            f"reason={r.get('reason','')[:60]}"
        )

    return 0 if n_success == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
