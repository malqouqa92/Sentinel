# CODE_TIERS.md — Claude pre-teach playbook (Phase 17c)

This file is loaded into Claude's PRE_TEACH_SYSTEM prompt on every
agentic /code call. Claude reads the matching tier's rules + uses the
heuristic complexity score the system computes from the user's prompt
as guidance.

The tiers are advisory, not enforced. Claude still makes the final
call on STEP-N vs DECOMPOSE. But these rules tighten the judgment:
on tier-3 tasks Claude should DECOMPOSE rather than emit a 12-step
recipe that hits the parser cap.

---

## Tier 1 — basic
*Trivial tasks. ≤2 STEPs. Single file. Idempotent.*

- Examples: "add a constant to config.py", "rename a variable",
  "fix a typo in a docstring".
- **Action:** emit STEP-N directly. Skip Read/Grep when the change
  is fully described by the prompt — go straight to write_file or
  edit_file.
- **Anti-pattern:** over-thinking. Don't read 3 files for a 2-line fix.

## Tier 2 — standard
*Mid-size tasks. 3-7 STEPs. ≤2 files of substantial work.*

- Examples: "add a helper function to core/foo.py with a unit test",
  "extend a Pydantic model with a new field + migrate the DB",
  "refactor function X to take an extra param".
- **Action:** STEP-N is the default. Read first to grab anchors for
  edit_file. Verify with run_bash. write_file > edit_file if >30%
  of the file changes.
- **Anti-pattern:** fanning out into more files than the task needs.

## Tier 3 — advanced (multi-component)
*Multi-file changes. >7 STEPs likely. New command lane / pipeline.*

- Examples: "add a /qcode command", "build a new agent for X",
  "wire skill Y into the recipe pipeline", "add a new column +
  migration + handler + tests".
- **Action:** **DO NOT one-shot.** Emit DECOMPOSE format. Each
  subtask must be tier-1 or tier-2 (3-7 STEPs each). The user (or
  the chain runner if `CODE_CHAIN_ENABLED=True`) runs each subtask
  separately.
- **Anti-pattern:** "I'll just write a long recipe." A 12-step
  recipe will hit the 8000-char parser cap, get truncated at a step
  boundary, and only the first few STEPs will execute. Decomposition
  is the structurally correct response.
- **Example DECOMPOSE for "add /qcode command":**
  ```
  DECOMPOSE
  - /code add empty /qcode handler stub in interfaces/telegram_bot.py that just replies 'not implemented'
  - /code wire /qcode handler to call core.qwen_agent's run_agent_stepfed with KB context (no Claude pre-teach)
  - /code wire /qcode result back to telegram with pass/fail on attempt 1 only
  ```

## Tier 4 — pipeline rebuild
*Schema migration. New skill + new agent. Cross-cutting refactor.*

- Examples: "migrate the tasks table to add column X across all
  callers", "rebuild the job pipeline with new scoring", "swap the
  brain model and update all prompts".
- **Action:** DECOMPOSE with extra care. Each subtask must be
  independently revertible (`/commit`-able after each). Order the
  subtasks so partial completion leaves the system in a coherent
  state — schema migration first, callers updated second, tests
  added third. Never let a child subtask break the test suite.
- **Anti-pattern:** assuming the user will let the chain run end-
  to-end without intervention. Tier-4 chains are typically
  reviewed-per-subtask.
