"""Phase 14a + 14b -- KB graduation tests (transfer verification).

After Claude teaches Qwen a new pattern, immediately verify that the
production pipeline can reproduce it WITHOUT a fresh Claude pre-teach.
Result updates ``solo_attempts`` / ``solo_passes`` on the pattern row.
Once we have ≥ ``GRAD_MIN_TRIES`` (default 3) and the pass rate drops
below ``GRAD_FAIL_THRESHOLD`` (default 0.5), the pattern auto-flags
``needs_reteach=1`` and the next match escalates straight to Claude.

TWO MODES:

  A) Agentic-pipeline graduation (Phase 14b, default for new patterns).
     If the pattern has both a stored recipe (``solution_pattern``) and
     a ``base_sha``, we:
       1. Snapshot current tree state
       2. ``git reset --hard base_sha``  (clean known starting point)
       3. Run Qwen's stepfed agent on the STORED recipe (no Claude
          pre-teach needed -- it's already cached as the recipe).
       4. ``git diff`` the result and run Claude review for verdict.
       5. Apply the same syntax-check gate as production.
       6. ALWAYS reset back to the snapshot SHA (graduation is a
          probe, not an edit). Stash + pop for any uncommitted dirt.
       7. Update KB.
     This mirrors what production /code actually does, so failure here
     is a real signal that production would fail too.

  B) Legacy text-gen graduation (Phase 14a, fallback).
     For patterns from before Phase 14b (no base_sha / no recipe):
     ask Qwen to one-shot generate code via ``_qwen_generate``, run
     ``_validate_code``. Less representative but back-compat.

Why split the modes: pre-Phase-14b patterns don't have base_sha and
running A would fail at the reset step. Old patterns get the legacy
treatment automatically.

Cost (mode A): 1 Qwen stepfed call + 1 Claude review (~30-60s total).
Cost (mode B): 1 Qwen text-gen call + 1 executor run (~10-30s).
Both gated by GRADUATION_TIMEOUT_S so a hung Qwen doesn't block /code
for the OllamaClient default 900s.
"""
from __future__ import annotations

import asyncio
import time

from core.knowledge_base import KnowledgeBase
from core.logger import log_event


# Cap the graduation Qwen call at 60s. Anything slower means the
# pattern isn't easily reproducible and the user-facing /code reply
# would be blocked unacceptably long. Past production observation
# (pattern #52, 2026-05-05): the OllamaClient default of 900s caused
# a single cold-load hang to delay /code by 15 minutes.
GRADUATION_TIMEOUT_S = 60


async def graduate_pattern(
    pattern_id: int,
    problem: str,
    code_context: str | None,
    kb: KnowledgeBase,
    model_id: str,
    trace_id: str,
) -> dict:
    """Verify ``pattern_id``. Dispatches on mode:

      - Mode A (Phase 14b): pattern has base_sha + recipe -> replay
        recipe through stepfed + Claude review on a clean tree.
      - Mode B (Phase 14a): no base_sha -> legacy text-gen + executor.

    Returns:
      {
        "pattern_id": int, "passed": bool,
        "solo_attempts": int, "solo_passes": int,
        "needs_reteach": bool, "duration_s": float,
        "mode": "agentic" | "text_gen",
        "code_excerpt": str, "stdout_excerpt": str,
        "stderr_excerpt": str,
      }

    On any setup failure (missing pattern, Qwen raises, git failure)
    returns a result with ``passed=False`` so the caller's UX path
    stays uniform.
    """
    started = time.monotonic()
    pattern = kb.get_pattern(pattern_id)
    if pattern is None:
        log_event(
            trace_id, "WARNING", "kb_graduation",
            f"pattern_id={pattern_id} not found; skip",
        )
        return {
            "pattern_id": pattern_id, "passed": False,
            "solo_attempts": 0, "solo_passes": 0,
            "needs_reteach": False, "duration_s": 0.0,
            "mode": "none",
            "code_excerpt": "", "stdout_excerpt": "",
            "stderr_excerpt": "pattern not found",
        }

    # Mode dispatch. Phase 14b patterns have base_sha + a real recipe
    # in solution_pattern. Old (pre-14b) patterns fall back to
    # text-gen so they keep getting verified instead of just sitting.
    has_base = bool((pattern.base_sha or "").strip())
    has_recipe = bool((pattern.solution_pattern or "").strip())
    if has_base and has_recipe:
        return await _graduate_via_agentic(
            pattern, problem, kb, model_id, trace_id, started,
        )
    return await _graduate_via_text_gen(
        pattern, problem, code_context, kb, model_id,
        trace_id, started,
    )


async def _graduate_via_text_gen(
    pattern, problem, code_context, kb, model_id, trace_id, started,
):
    """Phase 14a legacy path: ask Qwen to one-shot generate, run
    executor, update KB. Used for old patterns without base_sha."""
    # Late imports to avoid skills→core→skills cycle.
    from skills.code_assist import (
        _clean_solution_text, _coerce_str, _extract_code_fallback,
        _extract_json, _is_real_solution, _qwen_generate,
        _qwen_user_prompt, _validate_code, QWEN_SYSTEM_BASE,
    )

    pattern_id = pattern.id

    # KB context: the SAME few-shot context the next /code call would
    # see for this problem -- this is the actual ecological test of
    # transfer, not a contrived no-context one.
    kb_examples = kb.get_context_for_prompt(problem)
    user_prompt = _qwen_user_prompt(problem, code_context, kb_examples)
    log_event(
        trace_id, "INFO", "kb_graduation",
        f"start pattern_id={pattern_id} model={model_id} "
        f"kb_chars={len(kb_examples)} timeout={GRADUATION_TIMEOUT_S}s",
    )

    try:
        # Phase 14a observability fix: cap graduation at 60s. The base
        # OllamaClient default is 900s -- way too patient for a yes/no
        # verification that should be quick. If Qwen can't reproduce
        # in 60s, the pattern by definition isn't "easily transferred"
        # and should count as a graduation FAIL. The /code response
        # path runs this inline, so a 15-min hang here blocks the
        # user-facing reply for 15 min (which is what hit pattern #52).
        qwen_raw = await asyncio.wait_for(
            asyncio.to_thread(
                _qwen_generate,
                QWEN_SYSTEM_BASE, user_prompt, trace_id, model_id,
            ),
            timeout=GRADUATION_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        log_event(
            trace_id, "WARNING", "kb_graduation",
            f"qwen graduation timed out after {GRADUATION_TIMEOUT_S}s "
            f"-- counts as fail (pattern too slow to reproduce solo)",
        )
        attempts, passes, flag = kb.record_solo_attempt(
            pattern_id, passed=False, trace_id=trace_id,
        )
        return {
            "pattern_id": pattern_id, "passed": False,
            "solo_attempts": attempts, "solo_passes": passes,
            "needs_reteach": flag,
            "duration_s": round(time.monotonic() - started, 2),
            "mode": "text_gen",
            "code_excerpt": "",
            "stdout_excerpt": "",
            "stderr_excerpt": (
                f"graduation timed out after {GRADUATION_TIMEOUT_S}s"
            ),
        }
    except Exception as e:
        log_event(
            trace_id, "WARNING", "kb_graduation",
            f"qwen call failed: {type(e).__name__}: {e}",
        )
        attempts, passes, flag = kb.record_solo_attempt(
            pattern_id, passed=False, trace_id=trace_id,
        )
        return {
            "pattern_id": pattern_id, "passed": False,
            "solo_attempts": attempts, "solo_passes": passes,
            "needs_reteach": flag,
            "duration_s": round(time.monotonic() - started, 2),
            "mode": "text_gen",
            "code_excerpt": "",
            "stdout_excerpt": "",
            "stderr_excerpt": f"qwen failed: {type(e).__name__}: {e}",
        }

    parsed = _extract_json(qwen_raw)
    qwen_code = _coerce_str(
        (parsed or {}).get("code") if isinstance(parsed, dict) else None
    )
    if not qwen_code:
        qwen_code = _coerce_str(_extract_code_fallback(qwen_raw))
    qwen_code = _clean_solution_text(qwen_code)

    if not qwen_code:
        log_event(
            trace_id, "WARNING", "kb_graduation",
            f"qwen returned no parseable code (raw_len={len(qwen_raw)})",
        )
        attempts, passes, flag = kb.record_solo_attempt(
            pattern_id, passed=False, trace_id=trace_id,
        )
        return {
            "pattern_id": pattern_id, "passed": False,
            "solo_attempts": attempts, "solo_passes": passes,
            "needs_reteach": flag,
            "duration_s": round(time.monotonic() - started, 2),
            "mode": "text_gen",
            "code_excerpt": "",
            "stdout_excerpt": "",
            "stderr_excerpt": "qwen returned no parseable code",
        }

    passed_exec, exec_result = await _validate_code(qwen_code, trace_id)
    # Same quality gate as the main /code path -- a degenerate "200"
    # that runs but solves nothing should NOT count as a graduation
    # pass (would falsely mark a real pattern as transferred).
    if passed_exec and not _is_real_solution(qwen_code):
        log_event(
            trace_id, "WARNING", "kb_graduation",
            "code passed exec but failed quality gate; "
            "treating as graduation FAIL",
        )
        passed_exec = False

    attempts, passes, flag = kb.record_solo_attempt(
        pattern_id, passed=passed_exec, trace_id=trace_id,
    )
    return {
        "pattern_id": pattern_id,
        "passed": passed_exec,
        "solo_attempts": attempts,
        "solo_passes": passes,
        "needs_reteach": flag,
        "duration_s": round(time.monotonic() - started, 2),
        "mode": "text_gen",
        "code_excerpt": qwen_code[:400],
        "stdout_excerpt": (exec_result.get("stdout") or "")[:400],
        "stderr_excerpt": (exec_result.get("stderr") or "")[:400],
    }


# ─────────────────────────────────────────────────────────────────
# Phase 14b -- Solution A: agentic-pipeline graduation
# ─────────────────────────────────────────────────────────────────

async def _graduate_via_agentic(
    pattern, problem, kb, model_id, trace_id, started,
):
    """Phase 14b graduation: replay the stored recipe through Qwen's
    stepfed agent + Claude review on a tree reset to the pattern's
    base_sha. Mirrors production /code so failure here is a real
    signal that production would fail too.

    Tree state discipline:
      pre_grad_sha = current HEAD (whatever it is)
      stash any uncommitted dirty changes
      reset --hard to pattern.base_sha
      <do the work>
      reset --hard to pre_grad_sha
      pop the stash

    The reset+pop is in a finally block so the user's working tree is
    NEVER left in a bad state regardless of what fails inside.
    """
    pattern_id = pattern.id
    base_sha = (pattern.base_sha or "").strip()
    recipe = (pattern.solution_pattern or "").strip()
    log_event(
        trace_id, "INFO", "kb_graduation",
        f"start AGENTIC pattern_id={pattern_id} base_sha={base_sha[:8]} "
        f"recipe_chars={len(recipe)} model={model_id} "
        f"timeout={GRADUATION_TIMEOUT_S}s",
    )

    # Late imports to avoid skills→core→skills cycle.
    from skills.code_assist import (
        _git_diff_stat, _git_reset_hard, _git_snapshot,
        _verify_syntax_of_changed_files,
    )
    from core.qwen_agent import run_agent_stepfed

    pre_grad_sha = await _git_snapshot(trace_id)

    # Phase 17 graduation tree-state redesign (2026-05-06):
    # Replace the stash dance with tree_state.snapshot_dirty_tree
    # + restore_dirty_tree. The stash dance was unreliable per
    # Phase 15e (silent failures) AND the pre-15e reliance on
    # auto-commit was disabled (NO AUTO-COMMITS owner directive).
    # tree_state writes the dirty diff to a tempfile and applies
    # it back deterministically -- failure modes are explicit
    # (apply-reject leaves replay state + patch on disk for
    # manual merge), no silent loss.
    from core.tree_state import (
        restore_dirty_tree, snapshot_dirty_tree,
    )
    try:
        # Phase 17i (2026-05-07): exclude the recipe's target paths
        # from the snapshot. Replay will recreate those edits in the
        # clean tree, so capturing them in the snapshot causes
        # restore_dirty_tree to apply edits ON TOP of replay's
        # recreated edits -> patch conflict -> /code's work lost.
        # Without this exclusion: orange-edit + unrelated dirty
        # core/progress.py deletion both wiped (live trigger
        # 2026-05-07 ~03:43Z).
        try:
            from skills.code_assist import _extract_recipe_paths
            recipe_paths_for_exclude = _extract_recipe_paths(recipe)
        except Exception:
            recipe_paths_for_exclude = []
        # Scope to the same paths _git_reset_hard touches, so the
        # snapshot patch contains only files that will be reverted.
        snap_handle = await snapshot_dirty_tree(
            _project_root(),
            paths=["core", "skills", "agents", "tests", "interfaces"],
            exclude_paths=recipe_paths_for_exclude or None,
        )
    except Exception as e:
        log_event(
            trace_id, "WARNING", "kb_graduation",
            f"snapshot_dirty_tree failed: {type(e).__name__}: {e}; "
            f"graduation will NOT restore /code's changes",
        )
        snap_handle = None
    else:
        if snap_handle.had_dirty:
            log_event(
                trace_id, "INFO", "kb_graduation",
                f"snapshotted dirty tree before graduation; "
                f"{len(snap_handle.captured_files)} file(s) captured "
                f"to {snap_handle.patch_path}",
            )
        else:
            log_event(
                trace_id, "DEBUG", "kb_graduation",
                "snapshot: no tracked changes to capture (working "
                "tree was clean at graduation start)",
            )

    passed = False
    review = {"verdict": "fail", "reasoning": "(graduation incomplete)"}
    agent_summary = ""
    diff_excerpt = ""

    try:
        ok = await _git_reset_hard(base_sha, trace_id)
        if not ok:
            log_event(
                trace_id, "WARNING", "kb_graduation",
                f"could not reset to base_sha={base_sha[:8]}; "
                "skipping graduation",
            )
            review = {
                "verdict": "fail",
                "reasoning": f"git reset to {base_sha[:8]} failed",
            }
        else:
            try:
                agent_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_agent_stepfed,
                        problem, recipe, trace_id, model_id,
                    ),
                    timeout=GRADUATION_TIMEOUT_S,
                )
                agent_summary = agent_result.get("summary", "") or ""
                session = agent_result.get("session", []) or []

                # Phase 14b bugfix: graduation determines pass DETER-
                # MINISTICALLY from stepfed output, NOT from a Claude
                # review of the diff. Reason: pattern #54 caught a
                # false positive -- Claude reviewed leftover state
                # from a prior /code attempt and marked PASS even
                # though stepfed actually errored on step 1.
                #
                # New criteria, all required:
                #   1. completed_via_done == True (recipe finished)
                #   2. steps > 0 (at least one tool call ran)
                #   3. NO step in session has an "error" key in result
                #   4. syntax check on changed files passes
                completed = bool(agent_result.get("completed_via_done"))
                step_count = int(agent_result.get("steps", 0) or 0)
                step_errors = [
                    s for s in session
                    if isinstance(s.get("result"), dict)
                    and "error" in s["result"]
                ]
                diff_stat = await _git_diff_stat(base_sha)
                diff_excerpt = (diff_stat or "")[:400]
                syntax_ok, syntax_err = (
                    await _verify_syntax_of_changed_files(
                        diff_stat, trace_id,
                    )
                )

                if not completed:
                    fail_reason = "stepfed did not call done()"
                elif step_count == 0:
                    fail_reason = "stepfed ran zero tool calls"
                elif step_errors:
                    first_err = step_errors[0]["result"]["error"]
                    fail_reason = (
                        f"stepfed step {step_errors[0].get('step', '?')} "
                        f"errored: {str(first_err)[:120]}"
                    )
                elif not syntax_ok:
                    fail_reason = (
                        f"syntax check failed on changed files: "
                        f"{syntax_err[:200]}"
                    )
                else:
                    fail_reason = ""

                passed = not fail_reason
                review = {
                    "verdict": "pass" if passed else "fail",
                    "reasoning": fail_reason or "stepfed completed cleanly",
                }
            except asyncio.TimeoutError:
                log_event(
                    trace_id, "WARNING", "kb_graduation",
                    f"agentic graduation timed out after "
                    f"{GRADUATION_TIMEOUT_S}s",
                )
                review = {
                    "verdict": "fail",
                    "reasoning": (
                        f"timed out after {GRADUATION_TIMEOUT_S}s"
                    ),
                }
            except Exception as e:
                log_event(
                    trace_id, "WARNING", "kb_graduation",
                    f"agentic graduation crashed: "
                    f"{type(e).__name__}: {e}",
                )
                review = {
                    "verdict": "fail",
                    "reasoning": f"crashed: {type(e).__name__}: {e}",
                }
    finally:
        # Tree state discipline: ALWAYS restore the pre-grad state
        # regardless of what failed. Reset first (gets us back to
        # pre_grad_sha), then pop the stash (gets back the dirty
        # working changes). Best-effort each step -- log and continue.
        #
        # Phase 17j (2026-05-07): pass recipe_paths_for_exclude to
        # _git_reset_hard so the final reset SKIPS the recipe paths.
        # Without this, the reset wipes /code's edits AGAIN even
        # though Phase 17i's snapshot/replay/restore cycle correctly
        # produced them. The replay's outcome IS the /code outcome
        # we want to preserve. Live trigger 2026-05-07 ~04:09Z
        # (PROMPT_BRIEF.md survived but the handler edit in
        # telegram_bot.py was wiped here).
        try:
            await _git_reset_hard(
                pre_grad_sha, trace_id,
                exclude_paths=recipe_paths_for_exclude or None,
            )
        except Exception as e:
            log_event(
                trace_id, "ERROR", "kb_graduation",
                f"FAILED to restore pre-grad SHA={pre_grad_sha[:8]}: "
                f"{type(e).__name__}: {e}",
            )
        if snap_handle is not None and snap_handle.had_dirty:
            try:
                restore_result = await restore_dirty_tree(snap_handle)
                if restore_result.restored:
                    log_event(
                        trace_id, "INFO", "kb_graduation",
                        f"restored dirty tree after graduation "
                        f"({restore_result.reason})",
                    )
                else:
                    log_event(
                        trace_id, "WARNING", "kb_graduation",
                        f"restore_dirty_tree apply-rejected: "
                        f"{restore_result.reason}",
                    )
            except Exception as e:
                log_event(
                    trace_id, "ERROR", "kb_graduation",
                    f"restore_dirty_tree raised: {type(e).__name__}: {e}",
                )

    attempts, passes_count, flag = kb.record_solo_attempt(
        pattern_id, passed=passed, trace_id=trace_id,
    )
    return {
        "pattern_id": pattern_id,
        "passed": passed,
        "solo_attempts": attempts,
        "solo_passes": passes_count,
        "needs_reteach": flag,
        "duration_s": round(time.monotonic() - started, 2),
        "mode": "agentic",
        "code_excerpt": diff_excerpt,
        "stdout_excerpt": agent_summary[:400],
        "stderr_excerpt": (review.get("reasoning") or "")[:400],
    }


def _project_root():
    """Late-import shim so module-level loading doesn't pull config."""
    from core import config
    return config.PROJECT_ROOT
