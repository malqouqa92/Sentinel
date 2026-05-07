# Sentinel — Phase History

Append-only log of what was built in each phase, key decisions, and known limits. Read this before changing the architecture.

---

## Phase 1 — Observability + Config

**Built:** `core/config.py`, `core/telemetry.py`, `core/logger.py`.

- `config.py` is the single source of truth for paths, model names, thresholds, env-var names. Nothing hardcoded elsewhere. PROJECT_ROOT anchors all paths.
- TraceID format: `SEN-{8hex}`. Generated per request. Threads through every log line.
- Logger writes JSONL to `logs/sentinel.jsonl`. 5 fields: `timestamp` (ISO8601 UTC), `trace_id`, `level`, `component`, `message`. No rotation.

**Decision:** ISO 8601 with UTC offset. Component tag on every line so we can grep by subsystem.

---

## Phase 2 — Deterministic Router

**Built:** `core/router.py` with `RouteResult` Pydantic model.

- First-token rule: input must start with `/command`. Anything else is an `INVALID_POSITION` error.
- Flag parsing: `--key value` pairs; bare flags become `True`.
- Error codes: `INVALID_POSITION`, `UNKNOWN_COMMAND`, `null` on success.

**Decision:** Lowercase only the command token. Flag values and free text preserve case. Literal "lowercase entire input" mangles paths/identifiers.

**Decision:** No LLM in routing. String parsing + dict lookup only. VRAM is too scarce to spend on intent classification.

---

## Phase 3-4 — Skills, Registry, Agents

**Built:** `core/skills.py` (BaseSkill abstract), `core/registry.py` (auto-discovery), `core/agents.py` (Agent with fixed `skill_pipeline`), `core/agent_registry.py` (YAML discovery).

- Every skill inherits `BaseSkill`, has Pydantic `input_schema`/`output_schema`, and an async `execute(input_data, trace_id, context)` method.
- Agents are YAML configs in `agents/`. Each lists a `skill_pipeline` — output of skill N becomes input to skill N+1.
- I/O compatibility check at agent init time: warn if `skill[i].output_schema` doesn't satisfy `skill[i+1].input_schema`'s required fields.

**Decision:** Fixed pipelines, no ReAct. 3B models can't reliably do dynamic tool selection. Humans define pipelines in YAML.

**Footgun (resolved 2026-05-04):** `core/agents.py` was once corrupted by a Qwen /code attempt that replaced `skill.execute(...)` with `skill.run(...)` (no such method) and stripped error handling. Restored from baseline `3a04678`. Always restore from baseline if a /code attempt damages a core file.

---

## Phase 5 — Database, Worker, Locks

**Built:** `core/database.py`, `core/worker.py`.

- SQLite tables: `tasks` (with `retry_count`, `recovery_count`, `result`, `error`) and `locks`.
- Worker is `asyncio.run(worker_loop)`. Polls queue, claims a task, acquires GPU lock if needed, dispatches via orchestrator, releases lock in `finally`.
- Signal handlers (SIGINT/SIGTERM and Windows SIGBREAK) set a shutdown event; worker drains gracefully.
- Crash recovery on startup: `database.recover_stale()` resets in-flight tasks and releases stale locks.

**Decision:** GPU lock lives in DB, not memory. Survives crashes. Stale-lock timeout is `STALE_LOCK_TIMEOUT` seconds.

---

## Phase 6 — Skills v1

**Built:** `skills/job_extract.py`, `skills/web_search.py`, `skills/file_io.py`, `skills/code_execute.py`. Each registered with command in `config.COMMAND_AGENT_MAP`.

- File and web skills are CPU-only. `code_execute` runs Python in subprocess (timeout 30s, never `exec()`/`eval()` in the main process).

---

## Phase 7 — Orchestrator + KB

**Built:** `core/orchestrator.py`, `core/knowledge_base.py`.

- Orchestrator: deterministic command → executor map. Three target types: `None` (built-in handler), `"agent_name"` (registry lookup), `"skill:NAME"` (single-skill direct dispatch, bypasses agent).
- KB: SQLite FTS5 over `tags + problem_summary + solution_pattern + explanation`. `add_pattern` / `add_limitation`. Auto-prune at `KNOWLEDGE_MAX_ENTRIES` (500), oldest-low-usage first.

**Decision:** No vector embeddings. FTS5 over curated tags is enough at our scale and frees VRAM.

---

## Phase 8 — Model Roster + Routing + Claude CLI

**Built:** `core/model_registry.py`, `core/complexity.py`, `core/llm.py` (added `InferenceClient` over `OllamaClient`), `core/claude_cli.py`.

- Roster: `qwen3-brain` (Qwen 3 1.7B), `qwen-coder` (Qwen 2.5 Coder 3B), `claude-cli` (subprocess to local `claude` CLI). Plus `sentinel-brain` (custom Modelfile, see Phase 9).
- Heuristic complexity classifier (no LLM). KB-informed: a problem class with a "limitation" entry bumps the tier to `advanced`.
- Fallback chain: recommended → next-tier-up → claude-cli.

**Decision:** Sentinel makes ZERO outbound API calls. Claude integration is via the local `claude` CLI subprocess (uses the user's existing Claude Code login).

**Decision:** No basic-tier local model. The model-swap penalty (5-10s) on 4GB VRAM wipes any speed gain. Two standard locals + one advanced (claude-cli).

**Ollama env:** `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KEEP_ALIVE=2m`.

---

## Phase 9 — Telegram + Brain + Agentic /code

**Built:** `core/brain.py`, `interfaces/telegram_bot.py`, `core/qwen_agent.py`, agentic rewrite of `skills/code_assist.py`, `main.py` entry point.

### Brain
- Custom Modelfile `sentinel-brain` derived from `qwen3:1.7b`: `num_ctx=8192`, `num_predict=1024`, `temperature=0.1`. Uses `/no_think` mode (skips Qwen 3 chain-of-thought, ~10s faster per call).
- Used for intent classification (free-text Telegram → dispatch JSON or chat reply) and result summarization.
- After every GPU task, worker actively unloads (`keep_alive=0`) so the brain can reload without waiting for the keep-alive timeout.

### Telegram bot — three lanes
- `/code <task>` → enqueue task → worker → `code_assist` skill (agentic pipeline).
- `/claude <prompt>` → direct claude CLI subprocess passthrough (no skill, no KB).
- Free text → brain → either chat reply OR dispatch as a `/command`.

**Auth:** every handler checks `update.effective_user.id ∈ TELEGRAM_AUTHORIZED_USERS`. Bot is public; anyone can find it. Env vars: `SENTINEL_TELEGRAM_TOKEN`, `SENTINEL_TELEGRAM_USER_ID` (comma-separated for multiple).

### Agentic /code (final architecture)
The flow per `/code` request:

1. **KB lookup** — fetch top-N patterns (prior winning recipes) and limitations (known failure modes) for similar tasks.
2. **Claude pre-teach** — generates a numbered atomic-step recipe (`STEP 1: read_file path="..." \n STEP 2: edit_file path="..." old="..." new="..."`). KB patterns are injected so Claude adapts proven sequences.
3. **Qwen agent loop** — `core/qwen_agent.py` executes the recipe via tools: `read_file`, `list_dir`, `write_file`, `edit_file`, `run_bash`, `done`. All paths sandboxed to `PROJECT_ROOT`. Bails on 5 consecutive tool failures.
4. **Claude review** — pass/fail verdict on the diff. JSON output: `{verdict, reasoning}`.
5. **If fail and attempts left** — `git checkout <base> -- core skills agents tests && git clean -fd` (scoped reset, won't touch live log), then Claude `_claude_corrective_teach` produces a sharper recipe based on prior session + diff + verdict, and we go back to step 3.
6. **If pass** — single commit at the end, KB stores the winning recipe as a pattern. Solved-by: `qwen_agent`.
7. **If exhausted (5 attempts)** — reset tree to clean, KB stores limitation with the last recipe Claude tried. Solved-by: `qwen_failed`. Claude does NOT finish the job — the philosophy is teach Qwen, not replace it.

**Constants:** `MAX_TEACH_ATTEMPTS = 5`. `MAX_AGENT_STEPS = 30`. `MAX_CONSECUTIVE_FAILURES = 5`. `CLAUDE_CLI_TIMEOUT = 1800s`. `TELEGRAM_TASK_TIMEOUT = 1800s`.

### Key decisions logged here
- **Iterative teaching, never falls back to Claude.** Claude is teacher + reviewer only. The point is for Qwen to learn over time. KB hits should make later runs faster (shorter recipes) and eventually self-sufficient.
- **Atomic-step recipe format.** Each step is one tool call with literal args. No prose between steps. 3B models can't interpret abstract instructions.
- **Recipe length cap (1800 chars).** Longer recipes push Qwen into prose-mode (no tool calls). Less is more.
- **Workdir diff, never commit failed attempts.** Earlier versions committed every Qwen attempt directly to history, poisoning the tree with broken code that future runs would inherit. Now: `git add -N` for untracked → diff against base from working tree → commit ONCE on final pass; reset on fail.
- **Loud-failure to chat.** When a worker stores an `_error: True` envelope as a "completed" task result, the Telegram /code handler now surfaces the full error + traceback to chat instead of rendering it as empty code. Bot stays alive; failures are visible.
- **`qwen2.5-coder:3b` doesn't emit native `tool_calls` field.** It returns calls as JSON text in `content`. `qwen_agent.py` parses with `json.JSONDecoder.raw_decode` to handle multiple JSON objects per response.

### Known limits / footguns
- **3B ceiling for multi-file integration.** Qwen reliably edits one file but struggles to wire a new file into existing call sites without explicit step-by-step instructions. The teaching loop is the workaround.
- **File-revert mystery.** During this session, edits to `skills/code_assist.py` were silently reverted twice by something on the dev machine (not git hooks — only `.sample` files in `.git/hooks`, no `core.fsmonitor`). Likely an open IDE buffer auto-discarding. If edits don't persist, close any editor on the file before retrying.
- **Telegram message editing is the progress bar.** No separate UI — the initial "🔧 Starting..." message is edited in place to show stage and elapsed time. See `_latest_stage_for_trace`.

---

## Conventions across all phases

- Python 3.12, deps pinned in `requirements.txt`.
- Pydantic v2 for all schemas.
- Async everywhere the worker touches.
- ISO 8601 UTC timestamps.
- TraceID on every operation.
- All paths from `core/config.py`. Never hardcode.
- No `exec()` / `eval()` in main process. Code execution → `subprocess`.
- No outbound API calls from Sentinel itself. Claude is via local CLI subprocess.

## What NOT to do

- Don't add LLM calls to router or orchestrator.
- Don't bypass the skill registry. Every skill inherits BaseSkill and lives in `skills/`.
- Don't use in-memory state for anything that must survive a crash. Use the DB.
- Don't commit Qwen's intermediate /code attempts to git history. Use workdir diff.
- Don't ignore an `_error: True` result envelope. Surface it loudly.

---

## Phase 10 — Domain pipelines, persistent memory, persona files

**Goal:** Sentinel stops being stateless. Adds persona files always loaded into the brain, three-tier persistent memory (working/episodic/semantic), a job-search pipeline, a research pipeline, a nightly Claude-driven curation flow with Telegram approval, and hash-based diff-watch on the four protected persona files.

### Batch 1 — Persona + Memory + Diff-Watch (foundation)

**Built:**
- `workspace/persona/` with seed `IDENTITY.md`, `SOUL.md`, `USER.md`, `MEMORY.md`. Human-editable, git-tracked.
- `core/file_guard.py` — `FileGuard` class: SHA-256 baseline at startup, `check_integrity()` returns tampered list, `authorize_update(filename, new_content)` writes + refreshes hash. Alert callback wired to Telegram via `bot.send_alert_sync`.
- `core/memory.py` — `MemoryManager` (episodic + semantic with FTS5) plus `WorkingMemory` (in-process per-Telegram-session ring buffer, 20 messages). New DB at `memory.db` (separate from sentinel.db and knowledge.db). Triggers keep FTS in sync. Auto-prune per-scope at 200 episodic / 500 semantic.
- `core/brain.py` — `BrainRouter._load_persona_files()` on `__init__`; per-file inject caps (2000 default, 5000 for MEMORY.md); `reload_persona()` for post-curation refresh. Best-effort `_memory_context()` prepends recent episodic context to every chat call.
- `core/agents.py` — `Agent.run()` appends one episodic memory entry on every successful pipeline (scope=agent_name). Failures don't get memorialized.
- `interfaces/telegram_bot.py` — `/remember <key>: <value>`, `/forget <key>`, `/recall <query>`, `/memory` (stats), `/curate` (Batch 1 stub). `send_alert(text)` + `send_alert_sync(text)`. 5-min heartbeat task runs `file_guard.check_integrity()` and pings Telegram on tampering. Auto-extraction wired in `handle_message`: every 5 messages, `extract_facts_from_conversation` runs in background and stores facts at confidence 0.6.
- `core/config.py` — Phase 10 constants block: `PERSONA_DIR`, `PROTECTED_FILES`, `FILE_GUARD_CHECK_INTERVAL`, `PERSONA_INJECT_MAX_CHARS` (per-file dict), `MEMORY_DB_PATH`, `EPISODIC_MAX_PER_SCOPE`, `SEMANTIC_MAX_ENTRIES`, `MEMORY_DECAY_DAYS`/`FACTOR`, `AUTO_EXTRACT_THRESHOLD`, `MEMORY_SOURCE_CONFIDENCE` map, curation-time, `JOBSPY_*` defaults, output dirs.

**Decisions captured:**
- **Memory commands are bot-direct, not queued.** `/remember`, `/forget`, `/recall`, `/memory` do sub-second SQLite ops. Routing them through the worker queue would add ~1-2 s round-trip with no benefit. The spec said add them to `COMMAND_AGENT_MAP`; deviated and documented here. `/curate` is also bot-direct because it needs interactive multi-message Telegram approval (doesn't fit the task-row model).
- **Memory + file_guard are lazy singletons.** `get_memory()` and `install_file_guard()` instead of import-time construction. Lets tests monkeypatch `config.MEMORY_DB_PATH` / `config.PERSONA_DIR` cleanly without import-order gymnastics.
- **Persona load is forgiving.** Missing persona file → skipped silently. Bot should boot on a fresh install before the user has populated the files.
- **`_record_episode` is best-effort.** Wrapped in try/except so a memory hiccup never breaks a successful pipeline (debug log on skip). The agent is the truth, memory is the diary.
- **Confidence-aware semantic merge.** Higher confidence wins, equal → newer wins. Prevents auto-extracted (0.6) writes from overwriting user-explicit (1.0) facts.
- **Auto-extracted facts appear in `get_profile_context` flagged `UNCONFIRMED`.** Brain sees them immediately but knows their status. Curation promotes to 1.0 with owner approval. (Open question 2 from spec — picked this default.)
- **Persona-file mirrors hidden from `get_profile_context`.** `sync_persona_files` stores the full MD content as a fact under `persona:<name>`. Showing those big blobs in the agent profile context would crowd out actual user facts. The brain gets persona separately via `_build_system_prompt`.
- **Heartbeat is on the bot's event loop.** `_heartbeat_loop` runs every `FILE_GUARD_CHECK_INTERVAL` seconds (default 300 s). Cancelled cleanly in `stop()`. No new threads.
- **Branch hygiene.** Phase 10 work lives on branch `phase10`; merge to master after verified-complete.

**Footguns / open items:**
- The bot's `__init__` now constructs `FileGuard` if not passed in. The default reads from `config.PERSONA_DIR`. Tests that touch the bot constructor must either pass a tmp `file_guard` or set `config.TELEGRAM_AUTHORIZED_USERS` to keep auth happy.
- `_memory_context` swallows ALL exceptions — if memory.db is corrupt, the brain just drops context silently. Visible only in DEBUG logs. Trade-off: chat resilience over loud-failure on memory issues.
- Phase 9 file-revert mystery still applies. Before manually editing persona MDs, close any IDE that has them open with auto-discard.

### Batch 2 — Job search pipeline

**Built:**
- `skills/job_scrape.py` — wraps `python-jobspy` (lazy-imported so the bot still boots if not installed; SkillError surfaces a clean install message). Output schema is `{postings: [...]}`; `output_is_list=True` so the agent unwraps to a bare list for fan-out.
- `skills/job_score.py` — LLM-scored fit per posting. Reads `workspace/persona/USER.md` so scoring adapts to the human profile. Returns `{title, company, location, score, reasons, recommend}`.
- `skills/job_report.py` — `accepts_list=True` (consumes the full scored list); overrides `validate_input` to wrap a bare list in `{scored: [...]}`. Writes `jobs.csv` + `summary.md` to `workspace/job_searches/<timestamp>/` and stores one episodic entry under `scope=job_searcher`.
- `agents/job_searcher.yaml` — pipeline `[job_scrape, job_extract, job_score, job_report]`, pinned to `qwen-coder` (JSON-extraction-heavy).
- `core/skills.py` — added `accepts_list` and `output_is_list` `ClassVar`s to `BaseSkill`.
- `core/agents.py` — fan-out logic: when `current` is a list and `skill.accepts_list=False`, the agent calls the skill once per item and aggregates `model_dump()` results. After each step, if `skill.output_is_list=True`, unwraps the single-list-field model so the next iteration sees a bare list.
- `interfaces/telegram_bot.py` — `/jobsearch <role> [--location ...] [--distance N] [--hours H] [--results R] [--sites S]` handler.
- `core/config.py` — `JOBSPY_*` defaults and `JOB_DESCRIPTION_MAX_CHARS`.

**Decisions:**
- **Two-flag fan-out is more explicit than convention.** Considered auto-unwrap on "single list field" outputs, rejected because it'd silently fan-out when a skill genuinely meant to return a list as data. `output_is_list` and `accepts_list` make intent visible at skill-class declaration.
- **`job_report` is the canonical `accepts_list=True` skill.** Its `validate_input` override is the pattern other batch-style skills should follow.
- **Lazy-import `python-jobspy`.** Module-level `from jobspy import scrape_jobs` would crash bot startup if the dep isn't installed. Lazy + SkillError lets the bot run with /jobsearch failing gracefully until the user installs.
- **Sites flag accepts comma-separated string.** Pydantic field_validator splits `"indeed,linkedin"` into a list. Matches Telegram-friendly typing.

**Footguns:**
- The bot's `/code` agentic flow (Phase 9) uses `git checkout <base> -- core skills agents tests && git clean -fd` between attempts to undo failed work. While Phase 10 was being built CONCURRENTLY with `/code` traffic from the user testing, the clean step REPEATEDLY DELETED `core/curation.py` and reset `core/skills.py`/`core/agents.py` mid-edit. Worked around by atomic write-then-commit so HEAD always had my work; future fix should make `/code`'s reset scope tighter (only touch files it modified, not all of `core skills agents tests`).
- Test isolation: the new `test_pipelines.py` tests pass in isolation (`pytest tests/test_pipelines.py`) but a couple fail when run AFTER `test_pipeline_phase7/8/9` in the full suite. Suspect cross-test mutation of `JobExtractSkill.execute`. Fix is non-blocking — add explicit registry reset or instance-level monkeypatch in those tests later.

### Batch 3 — Research + nightly curation

**Built:**
- `skills/web_summarize.py` — per-result LLM brief, capped at `RESEARCH_SUMMARY_MAX_WORDS` (200). Iterates internally over results — no fan-out needed because `web_search`'s output is a multi-field dict, not a single-list-field. `accepts_list=False`.
- `skills/research_report.py` — pure-CPU markdown writer. Saves `report.md` to `workspace/research/<timestamp>/`, stores one episode under `scope=researcher`.
- `agents/researcher.yaml` — pipeline `[web_search, web_summarize, research_report]`, pinned to `sentinel-brain` (one-paragraph summarization is what it's tuned for).
- `core/curation.py` — `CurationFlow` class. `propose(trace_id)` reads the last 24h of episodes + current MEMORY.md + USER.md, calls the local Claude CLI for a JSON proposal, runs a server-side sanity check (rejects code-execution patterns / secret regexes), stores the pending proposal in an in-memory dict keyed by a 4-hex-char token. `apply(token)` writes through `file_guard.authorize_update` so diff-watch baseline stays in sync; calls `brain.reload_persona()` and `memory.sync_persona_files()`.
- `interfaces/telegram_bot.py` — `/research <topic>` handler, real `/curate` (kicks off proposal + posts to Telegram with `[TOKEN]`), `/curate_approve <token>`, `/curate_reject <token>`. Nightly scheduler `_curation_scheduler_loop` sleeps until next `CURATION_HOUR_LOCAL`:`CURATION_MINUTE_LOCAL` (default 03:00) and pushes a proposal automatically.

**Decisions:**
- **In-memory pending-proposal state.** No persistence: if the bot restarts before the user approves, the proposal is lost and the user runs `/curate` again. Trade-off chosen because (a) curation calls Claude CLI (~$0.01-0.05) — cheap to redo; (b) persisting would require schema additions + recovery logic for stale proposals; (c) the user's approval window is bounded (typically minutes-to-hours); (d) the proposal embeds the snapshot it was made from, so applying a stale one would be unsafe anyway.
- **Server-side sanity check before sending to Telegram.** Curator might propose `eval(...)` or paste a leaked token. Reject those server-side, don't even surface to the user. Forbidden patterns are conservative (eval/exec/subprocess/os.system/key prefixes) — false positives are fine because curator can re-propose.
- **MEMORY.md edits via in-place update; USER.md edits as appended curator notes.** USER.md is the owner's own profile; the curator should suggest, the human accepts inline. MEMORY.md is curator-driven by design.
- **Curator uses Claude (not Qwen).** Reasoning across episodes + structured-JSON output is exactly the case Phase 8 picks Claude for. Worth the ~$0.05/curation given the daily cadence.
- **`/no_think` in curator prompt** — same as the brain. Saves ~10s per call.
- **Scheduler sleeps to target time, doesn't poll.** `asyncio.sleep` of `(target - now).total_seconds()` is exact and non-busy.

**Tests:** L (curator JSON), M (apply round-trip + diff-watch clean + brain reload + persona-mirror update), Q (research pipeline writes report + episode). All pass alongside Batch 1-2 tests. See `tests/test_curation.py` and `tests/test_research.py`.

**Open follow-ups (not blockers):**
- The 80-char `user_message[:80]` log truncation in `core/brain.py` still applies (user explicitly declined to bump it during the build).
- Curation scheduler is daily-only; no "after N hours of activity" trigger. Add later if needed.
- Approval flow has no edit-in-place option — only approve/reject. If the user wants to tweak before applying, they edit MEMORY.md manually after approval.

---

## Phase 16 — Toward Qwen-only operation (in progress, 2026-05-06)

**Goal:** Move the system from "Claude-augmented every /code" to "Claude fades to zero as Qwen earns trust". The vision the owner articulated: "the ultimate AI assistant offline LLM" — search indexes (FTS5 + embeddings + reranking) over-compensate for the small model + 4GB GPU constraint, so a 3B-coder model with PERFECT context can outperform a larger model with poor context. Phase 16 is the first phase that builds the FADE-CLAUDE-OUT machinery instead of just patching defects.

**Originally scoped batches:**
- Batch 0 — `/kb show <id>` (transparency for inspection)
- Batch A — Tools-enabled shadow plan (Qwen sees files before planning)
- Batch B — Diff-match verification (deterministic safety check)
- Batch C — Claude-skip path (the headline; bypass Claude on high-trust patterns)
- Batch D — Auto-pin proven graduators (KB self-curates)
- Batch C-safety — recipe linter + per-attempt git snapshot + bash whitelist (deferred until Batch C ships)

### Batch 0 — `/kb show <id>` ✅ Shipped (commit `3441468`)

Per-pattern transparency surface. Single Telegram command prints the full record: lifecycle (state, pinned, archived_at, origin, base_sha), problem_summary, Claude's stored recipe (trimmed 600 chars), Qwen's shadow recipe (trimmed 600 chars), agreement score with per-component breakdown (file Jaccard × W=0.5, tool Jaccard × W=0.3, step proximity × W=0.2), graduation solo stats, tags. The component breakdown lets the user SEE why the score is whatever it is — knowing "file=1.0 tool=0.33 step=0.66" tells you Qwen got the right file but used a different tool, which is actionable feedback the aggregate doesn't surface. Read-only; can't mutate any row. **12 tests, 200/200 regression green.**

### Batch A — Tools-enabled shadow planner ⚠️ Architecture complete, DEFAULT OFF

**Original ship (commit `b87bf5c`):** New `run_shadow_planner` in `core/qwen_agent.py` gives Qwen a structurally-locked-down read-only tool surface (`read_file`, `list_dir` only). `SHADOW_TOOLS_SCHEMA` defines them independently of `TOOLS_SCHEMA` so write/edit/bash are absent by design, not by trust. Loop refuses non-read-only tool calls with an error message fed back to the model. Cap of 5 exploration steps + a final recipe-output turn; 90s timeout initially. **19 tests, 219/219 regression green.**

**Bugfix 1 (commit `7ac8719`):** First production trace (SEN-bbef3a4f) timed out at 90s on a cold qwen2.5-coder:3b load. Bumped `QWEN_SHADOW_TIMEOUT_S` 90 → 180. **38/38 targeted regression green.**

**Polish (commit `33f859b`):** Second production trace (SEN-6a8a6539) showed Qwen burning ALL 5 tool-call slots on file reads without ever transitioning to recipe output. Strengthened `SHADOW_PLANNER_SYSTEM` prompt with hard directives ("Use ONE OR TWO reads at MOST", "VERY NEXT response MUST be the recipe", "do NOT chain read_file calls"). Dropped `SHADOW_MAX_TOOL_CALLS` 5 → 3. Production result: pattern id=84 stored, /code wall-clock dropped from 4m31s to 55s, agreement score 0.567 (real signal, no longer NULL). **22/22 regression.**

**Stress test surfaced the real bottleneck.** Three diverse scenarios (S1: `lcm` math, S2: `clamp` util, S3: `/uptime` telegram cmd, cancelled mid-flight) showed:

| Scenario | Production | Shadow plan |
|---|---|---|
| S1 (SEN-9e28f685) | PASS attempt 1, ~4min, $0.22 | TIMED OUT at 180s after step 1 read_file |
| S2 (SEN-a95e89b2) | PASS attempt 1, ~4m20s, $0.25 | TIMED OUT at 180s WITHOUT a single tool call landing |
| S3 (SEN-69888f98) | (cancelled) | — |

The bottleneck is NOT cold-load — it's single-call inference time. A single Qwen chat call with KB-augmented prompt (5-10K char context) takes 180-210 seconds on this 4GB GPU. Bumping the timeout further would just push the cost. The agentic architecture works (tools available, sandbox enforced, refusal logic correct); it's blocked on inference speed, not design.

**Revert (commit `14611f8`):** Flipped `SHADOW_PLAN_USE_TOOLS=False` as default. Architecture preserved behind the flag; flip back to True once one of these lands:
- `num_predict` cap on shadow LLM calls (bounds inference time directly)
- Faster GPU (4GB → 8GB+ would change the math)
- Smarter context compression (5-10K → 2-3K)

The legacy one-shot shadow path with the format_json=False fix (Phase 15d-bugfix) gives reliable mid-quality shadow data in 10-15s without blocking /code. Default behavior reverts to that. **222/222 regression green.**

### Phase 15e snapshot scope bug (surfaced by stress test, recovered manually)

`_git_commit_for_graduation`'s scope is `core skills agents tests interfaces workspace` — excludes project-root files like `math_utils.py`. S1 added an `lcm` function to `math_utils.py`. Sequence:

1. Production attempt 1 modified `math_utils.py` (added lcm)
2. `_git_commit_for_graduation` ran `git add -- <scope>`, which excluded math_utils.py at project root
3. Helper logged `grad-snapshot: nothing to commit (no staged changes)`
4. Graduation stashed the dirty working tree (captured math_utils.py change in stash)
5. Graduation reset, ran stepfed (re-applied lcm), reset back, then `stash pop`
6. Stash pop FAILED with merge conflict because the working tree (post-second-reset, scoped to non-project-root) still had stepfed's lcm
7. Net result: math_utils.py change in working tree, NOT committed, stash orphaned

**Recovery (commit `91b16dc`):** Manually committed `math_utils.py` with the `sentinel-grad@sentinel.local` identity to capture S1's intent. Two orphaned stashes from this and an earlier session left in `git stash list` (harmless on disk).

**Permanent fix (Phase 17 candidate):** extend `_git_commit_for_graduation` scope to use `git add --all -- :!*.db :!logs/ :!__pycache__/` exclusion form so project-root source files are captured.

### Where Phase 16 stands

- ✅ Batch 0 shipped, working
- ⚠️ Batch A shipped, default OFF (architecture preserved, blocked on inference speed)
- ⏳ Batch B (diff-match verification) — pending, ready to ship but provides no behavior change without Batch C
- ⏳ Batch C (Claude-skip path) — pending, depends on Batch B + Batch C-safety
- ✅ Batch D (auto-pin proven graduators) — shipped commit `3ed0b27`, 14 tests, 236/236 regression
- ⏳ Batch C-safety (recipe linter + per-attempt snapshot + bash whitelist) — pending, must ship WITH Batch C

### Batch D — Auto-pin proven graduators ✅ Shipped (commit `3ed0b27`)

Smallest, lowest-risk Phase 16 batch — ships independently of Batch B and Batch C. Inside `KnowledgeBase.record_solo_attempt`, after the existing solo_attempts/solo_passes update, check if the pattern just crossed the trust threshold. If it did, set `pinned=1` in the same transaction. The kb_lifecycle walker (Phase 15a) can never archive a pinned row, so 5/5-proven patterns become permanent foundation without manual `/kb pin <id>` curation.

**Threshold (strict by design):** `AUTO_PIN_MIN_PASSES=5` AND `AUTO_PIN_REQUIRED_RATE=1.0`. Even one failure (4/5=0.8 rate) doesn't qualify — partial credit doesn't earn this trust level. Foundation set is conservative; future Batch C skip-eligibility will gate at lower bars (0.8 / 3 attempts) for actual SKIP behavior.

**Atomicity:** Both UPDATE statements run inside the same `BEGIN IMMEDIATE` transaction. No window where the row is "almost pinned". Idempotent on already-pinned rows (skip the auto-pin branch entirely). Reversible via `/kb unpin <id>`.

**Log surface:** Distinct marker on the transition only (`AUTO-PIN pattern_id=N (N/N solo passes, perfect rate) -- now permanent foundation, immune to kb_lifecycle archival`). Easy to grep, fires once per pattern.

**Tests:** 14 new in `test_phase16bd_autopin.py` covering thresholds, firing on 4/4 vs 5/5 vs 5/6, idempotency, atomicity, log shape, reversibility. Targeted regression: 236/236 green.

**Recovery point:** tag `pre-batch-D-20260506-091719` at HEAD `e073de1` set just before this batch. Revert via `git reset --hard pre-batch-D-20260506-091719` if anything regresses in heavy testing.

**Live expectation:** today's KB has zero patterns at 5/5 yet (most at 1/1). Pattern id=88 (agreement=1.0) is well-positioned to be among the first auto-pinned over the coming days as repeat /code teaches accumulate graduations.

### Decisions captured

- **Default OFF for risky paths.** Batch A's agentic shadow could in principle work, but the production cost (4-min /code) was unacceptable. The right call was flipping the flag rather than ripping out the architecture. When inference-time controls land, flip back; the code already exists.
- **Stress testing IS engineering work.** Discovered Phase 15e's project-root scope bug. Discovered the inference-speed bottleneck. Discovered Qwen's bipolar exploration behavior (5-reads-no-recipe vs 0-reads-stub-recipe). All real signals that wouldn't have surfaced in unit tests.
- **Trip-wires armed throughout.** Row count, schema columns, pinned count monitored after each scenario. KB integrity green throughout (rows 80 → 82, no schema regression, 39 pinned preserved).
- **Forensic snapshots before high-risk operations.** `knowledge.db.stresstest-pre-*.db` snapshot taken before stress test, plus a git tag at the recovery point. Standard practice for "I plan to test heavy" sessions.

### Session footguns surfaced 2026-05-06 (avoid in future phases)

**Ghost terminal spawning during "live monitor" requests.** The user asked the assistant to "live monitor the bot as I test my generic code". The assistant's instinct was an active polling loop — `until <check>; do sleep 5; done` patterns or Python `while time.time() < deadline: ... sleep(6)` loops invoked via the Bash tool. On Windows these leave **orphaned bash.exe + cmd.exe + conhost.exe + sleep.exe** processes alive even after the harness reports the call complete. Each iteration spawns a visible cmd window that flashes onscreen. Five orphan bash.exe processes accumulated before the user noticed and complained: *"there's a 'terminal' baby being spawned every 5 seconds and then it closes"*. Cleanup required mass-killing bash.exe (which the harness correctly blocks as risky on shared systems).

**Rule for future bots:** Never use `until ... sleep N; done` or `while true; do ... sleep N; done` patterns inside Bash calls. The system prompt explicitly forbids it. The sanctioned alternatives:
- The `Monitor` tool (deferred) is designed for streaming events from a background process — each stdout line becomes a notification.
- For "wait until X" use `Bash run_in_background=true` ONCE, then read with BashOutput.
- For periodic status checks, check ONCE per user prompt instead of polling.

If the user asks to "monitor", explain that you'll check on each prompt rather than poll. Do not silently start a polling loop. And if you tell the user you're monitoring then stop, SAY SO — silently going dead while claiming to monitor will get you called out (it did, here: *"actually live monitor it"* and later *"HELLLLLO??????? ITS BEEEN DONE WAKE UP"*).

**0.875 agreement is not a regression — read the breakdown first.** Pattern id=89 followed pattern id=88 with the same emoji-swap prompt but scored 0.875 vs 1.000. The user (correctly) flagged the drop and asked what happened. Decomposition via `/kb show 89`:
- file Jaccard (W=0.5): **1.000** (same file, `interfaces/telegram_bot.py`)
- tool Jaccard (W=0.3): **0.750** (Claude used `{edit_file, run_bash, done}`; Qwen used `{read_file, edit_file, run_bash, done}` — Qwen ADDED a read_file step)
- step proximity (W=0.2): **0.750** (Claude wrote 3 steps, Qwen wrote 4)
- Blend: 0.5 + 0.225 + 0.15 = **0.875**

Qwen's recipe was actually MORE robust than Claude's — read-before-edit per QWENCODER.md teaching. The agreement metric measures structural similarity to Claude, not recipe quality. **Healthy variance, not a defect.** Future bots: when an agreement score drops, ALWAYS check `/kb show <id>` per-component breakdown before treating it as a regression.

**Bash ghost-process cleanup on Windows is risky.** `taskkill /F /IM bash.exe` kills the bash running the current command (self-termination — exit code 1 expected) but the harness will block it on shared systems with explicit *"risks terminating other unrelated shells/jobs"* permission denial. If orphans accumulate, the safer path is killing by PID one at a time after identifying the culprit (largest RSS / earliest start / etc.). Better: don't create the orphans in the first place.

### Batch B — Diff-match deterministic verification ✅ Shipped (2026-05-06)

`core/diff_match.py` implements hunk-set Jaccard scoring on git-style diff text. Two public functions:
- `score_diff_match(stored_diff, replay_diff) -> float` in [0.0, 1.0] -- pure score, never raises.
- `evaluate_diff_match(stored, replay, threshold=0.7) -> DiffMatchResult` -- score + accept verdict + reason. Empty replay diff is a forced reject (silent-pass guard).

Algorithm: parse each diff into a `set[(file_path, hunk_signature)]` where `hunk_signature = (n_added, n_removed, sha256(added_lines), sha256(removed_lines))`. Line numbers are NOT in the signature -- a hunk that drifts position but keeps content matches. Compute Jaccard. Threshold 0.7 picked to allow same-file same-shape edits while rejecting wrong-file or wrong-content replays.

**This is the cheap deterministic gate Batch C uses** to decide "did the replayed recipe produce a structurally similar diff?". Telemetry-only at ship: every successful /code can call it against the closest stored pattern and log the score, but no behavior is yet gated on the result.

24 ECC tests in `tests/test_phase16b_diff_match.py` cover: hunk parsing boundaries, score correctness across (identical / completely-different / partial-overlap / both-empty / one-empty / line-drift / different-paths-same-content), evaluate verdict + reason, threshold parameter override, robustness (never raises on garbage / unicode / binary noise).

Bug caught during ECC: original parser absorbed `--- a/path` headers from a second file's section into the previous file's hunk body, corrupting signatures. Fix: also recognize `--- ` lines as hunk boundaries.

### Batch C-safety — Recipe linter + bash whitelist + nvidia-smi window-flash fix ✅ Shipped (2026-05-06)

**Three components, one batch:**

**`core/bash_whitelist.py`** -- allowlist-only command gate. The replay-skip path (Batch C) MUST refuse `run_bash` steps that aren't on the curated allowlist. Allowlist patterns: `python -c "..."`, `python -m pytest`, `pytest`, `python script.py`, `ls`, `pwd`. Deny substrings (defense-in-depth even if allowlist regex bugs out): ` rm `, `git push`, `git reset --hard`, `git checkout`, `git rebase`, `sudo`, `curl `, `wget `, ` ssh `, `pip install`, `apt`, `$(`, backtick, ` && rm`, etc. NOT applied to /code's normal `run_bash` path -- regular /code still gets reviewer-Claude as the safety gate. Only the skip-eligible replay path consumes this.

**`core/recipe_linter.py`** -- 7 deterministic checks per stored pattern, gating skip-eligibility:
- L1: recipe parses to >= 2 STEPs
- L2: final step is `done`
- L3: there exists a `run_bash` verification step BEFORE `done` (skip-eligibility requires runtime check; without it, Batch B's empty-diff guard would be the only safety net and could pass false positives on idempotent recipes)
- L4: all tool names are in `_VALID_TOOLS`
- L5: every `run_bash` step's command passes the bash whitelist
- L6: no `write_file` / `edit_file` step targets a forbidden path (`logs/`, `.env*`, `*.db`, `*.sqlite`, `id_rsa`, etc.)
- L7: stored recipe text is below `RECIPE_MAX_CHARS_STEPFED=8000` (a recipe at the truncation cap may have lost its final `done` or verification step)

Returns `LintResult(safe, reason, failed_checks)` so logs explain WHICH check fired. Per-replay check; failures don't permanently demote a pattern.

Bug caught during ECC: `path.lstrip("./")` strips individual chars (eats the leading `.` from `.env`). Fix: `if p.startswith("./"): p = p[2:]`.

**`core/health.py` nvidia-smi window-flash fix** -- the user noticed a terminal flash every ~5 minutes; root cause was the scheduler's Resource probe (every 10 min) calling `nvidia-smi` via `subprocess.run` without `CREATE_NO_WINDOW`. When the bot runs detached (Phase 11 Task Scheduler supervisor at-logon), Windows allocates a fresh console for each child process. Fix: pass `creationflags=subprocess.CREATE_NO_WINDOW` on `sys.platform == "win32"`. No-op on Linux/macOS.

42 ECC tests in `tests/test_phase16csafety.py` covering bash whitelist (S01-S18), recipe linter (S31-S44), and nvidia-smi fix source-level checks (S51-S52).

### Batch C — Claude-skip path ✅ Shipped + live-validated (2026-05-06)

**End-to-end live validation (3 fresh patterns):**
| Scenario | Status | Time | Diff-match | Counter |
|---|---|---|---|---|
| alpha-const | ✅ success | 0.2s | **1.000** | 5/5 → 6/6 |
| beta-fn | ✅ success | 0.2s | **1.000** | 5/5 → 6/6 |
| gamma-validation | ✅ success | 0.2s | **1.000** | 5/5 → 6/6 |

All three: Qwen-only execution, ZERO Claude calls, perfect diff-match, file changes applied to disk, working tree restored cleanly post-replay, counter advanced via `record_solo_attempt(passed=True)`. End-to-end validation of the full skip-path machinery.

**What landed (all in 432/432 regression green at land time):**

- **`core/knowledge_base.py::is_skip_eligible(pattern_id, *, row=None)`** — pure-Python gate returning `(eligible, reason_token)`. Reason tokens are class constants on KnowledgeBase: `SKIP_REASON_OK_PINNED`, `SKIP_REASON_OK_TRUSTED`, `SKIP_REASON_MISSING`, `SKIP_REASON_NOT_PATTERN`, `SKIP_REASON_ARCHIVED`, `SKIP_REASON_NEEDS_RETEACH`, `SKIP_REASON_LOW_PASSES`, `SKIP_REASON_IMPERFECT_RATE`, `SKIP_REASON_STALE`, `SKIP_REASON_LOW_AGREEMENT`. Pinned bypasses trust-path; non-pinned needs `solo_passes >= 3 AND solo_attempts == solo_passes AND last_verified < 30d AND (qwen_plan_agreement IS NULL OR >= 0.5)`. 18 ECC tests in `tests/test_phase16_batch_c_eligibility.py`.

- **`core/knowledge_base.py::search` + `get_context_for_prompt`** gain `exclude_pattern_ids` kwarg. Used by skip-path failure fallback so Claude doesn't re-derive the broken recipe via retrieval. Builds `NOT IN (...)` SQL clause; malformed input is logged and ignored. 9 ECC tests in `tests/test_phase16_batch_c_exclusion.py`.

- **`core/tree_state.py`** — new module replacing the stash dance:
  - `snapshot_dirty_tree(project_root) -> SnapshotHandle` — captures tracked dirty changes to a temp `.patch` file, resets tracked tree to HEAD. Untracked + gitignored files NEVER touched.
  - `restore_dirty_tree(handle) -> RestoreResult` — applies patch back. On apply-reject, KEEPS replay state + leaves patch on disk for manual merge; never silently loses work.
  - `surgical_revert(project_root, paths) -> (rc, reverted, removed)` — surgical replacement for `_git_reset_hard`. Only touches the listed paths; reverts tracked ones via `git checkout HEAD --`, removes untracked-new-since-replay ones via unlink. **CRITICAL guard**: this is the lesson learned from the v1 self-wipe (where `_git_reset_hard("core skills agents tests interfaces")` blasted away the author's uncommitted Batch C source mid-development). 13 ECC tests in `tests/test_phase16_batch_c_tree_state.py`.

- **`skills/code_assist.py`** wires skip-path into `_run_agentic_pipeline`:
  - `_extract_recipe_paths(recipe)` — regex-extracts `path="..."` args; used to scope failure cleanup to recipe-touched files only.
  - `_maybe_skip_path(...)` — orchestrates the three gates (eligibility, lint, telemetry-flag) and dispatches to `_execute_skip_replay` only if all pass AND `config.SKIP_PATH_ENABLED=True`. Returns dict with `status` in `{success, failed, ineligible, telemetry_only}`.
  - `_execute_skip_replay(...)` — runs `run_agent_stepfed` on the stored recipe, intent-to-adds recipe paths so `git diff` sees new files, scopes diff capture to recipe paths only (key fix found during stress test debugging — without scoping, leftover dirty work in the tree drove Jaccard to 1/N), evaluates diff-match, increments counter on success/real-failure (NOT on environmental failures), uses `surgical_revert` for cleanup.
  - Skip-path branch fires BEFORE the Claude pre-teach loop. On `success`: returns `CodeAssistOutput(solved_by="qwen_skip_path")` directly. On `failed`: excludes failed pattern_id from `kb_patterns` for the fallback Claude pipeline so Claude doesn't re-derive it. On `ineligible` / `telemetry_only`: silent fall-through. 16 ECC tests in `tests/test_phase16_batch_c_wiring.py`.

- **Config flags** (already shipped in `core/config.py` from prep batch): `SKIP_PATH_ENABLED=False` (telemetry-only default; flip to True after observing log evidence), `SKIP_PATH_MIN_PASSES=3`, `SKIP_PATH_FRESHNESS_DAYS=30`, `SKIP_PATH_AGREEMENT_FLOOR=0.5`, `SKIP_PATH_DIFF_MATCH_THRESHOLD=0.7`.

**Lessons from build (preserved as comments + as `tests/test_phase16_batch_c_wiring.py::test_w04_uses_surgical_revert_not_scoped_reset`):**

1. **DO NOT use `_git_reset_hard("core skills agents tests interfaces")` inside skip-path** — that's the v1 self-wipe footgun. The scoped reset blasts away ANY uncommitted source under those dirs, including the author's in-progress work. Always use `surgical_revert(recipe_paths)` instead.

2. **`git diff <sha>` doesn't see untracked files** — recipes that create new files (typical write_file scenarios) need `git add -N` first or the diff capture comes back empty and diff-match scores 0 (silent rejection of correct replays).

3. **Always SCOPE the replay-diff capture to recipe paths** — without scoping, unrelated dirty work / leftover intent-to-add files inflate the file count and tank the Jaccard. Live debugging surfaced exactly this: 25-file working tree → Jaccard 1/25 = 0.04 even on byte-perfect replays. Fix: `git diff <sha> -- <recipe_paths>`.

**Recovery point this session:** `pre-batch-C-20260506-133810` at HEAD `ae17f51` + KB snapshot `knowledge.db.pre-batch-C-20260506-133810.db`. Plus a stress-test source backup snapshotted alongside the repo (read-only).

**Counter semantics now extended via skip-path:** `solo_attempts`/`solo_passes` are incremented by replay-skip outcomes, not just initial graduation. Same prompt 3+×, each successful replay → counter climbs → eventually auto-pin (Batch D 5/5 threshold) fires. Failed replays count too (only for `recipe_broken` / `runtime_check_failed`; environmental failures like Ollama-down do NOT pollute the counter).

### Batch C -- live production validation (2026-05-06 ~16:20-16:24Z)

End-to-end visible-change-persists loop, four /code runs in alternation, observed in Telegram chat:

| Time  | Prompt                                       | Pattern | Wall-clock | Claude calls | Diff-match | Counter   |
|-------|----------------------------------------------|---------|------------|--------------|------------|-----------|
| 16:20 | "make a green progress bar"                  | #127    | 7s         | **0**        | **1.000**  | 9/10      |
| 16:21 | "change my progress bar to use red squares"  | #131    | 65s        | 2            | n/a        | 1/1 cold  |
| 16:23 | "make a green progress bar"                  | #127    | **11s**    | **0**        | **1.000**  | **10/11** |
| 16:24 | "change my progress bar to use red squares"  | #131    | **11s**    | **0**        | **1.000**  | **4/4**   |

Each /code above followed by manual `/commit` + `/restart`. Telegram chat visibly showed the bar emoji change between restarts. Both patterns auto-pinned (Phase 16 Batch D, 5/5+ perfect rate); they will skip-path forever as long as the user alternates between them.

**The "alternating prompts" insight (the design loop closing):**

The "(you pick)" semantics of the original emoji prompt were structurally incompatible with deterministic replay -- each run picked a DIFFERENT random emoji, so the next replay's stored `old="..."` was always stale. But two prompts with FIXED targets ("make a **green** progress bar", "change to **red** squares") form a perpetual two-state machine:

- /commit + /restart with file = green; user runs "red" prompt -> file green->red; Option A overwrites #131's stored recipe to old="green-pattern" new="red-pattern".
- /commit + /restart, file = red; user runs "green" prompt -> file red->green; Option A overwrites #127's stored recipe to old="red-pattern" new="green-pattern".
- /commit + /restart, file = green; user runs "red" prompt -> recipe-#131 says old="green-pattern", file IS green -> drift-detect satisfied -> SKIP-FIRING -> diff-match 1.000.

The alternation is the trick. Two patterns ping-pong, each one's stored `old=` always matches the OTHER pattern's `new=`. Skip-path fires perpetually for both.

This only works when alternating; running the SAME prompt twice in a row would still drift (file is already in target state, recipe wants to transition FROM the previous "before" state). For an idempotent always-greens-the-bar prompt that auto-skips on every single re-run regardless of order, see Phase 17 candidate "smart drift-recovery" -- the system would need to either re-plan via Qwen on drift OR refactor `_build_bar` to read emojis from a state-independent module-level constant.

**Storage-side scoped diff (separate fix, 2026-05-06 same session):**

`skills/code_assist._git_diff_full(paths=...)` extended; `_run_agentic_pipeline` success path passes `attempt_paths = _extract_recipe_paths(recipe)` so the stored `solution_code` only contains hunks for files the recipe actually edited. Without this, Option A dedup would store the wide whole-tree diff (Phase 17 snapshot-scope wide-flavor bug) including unrelated `workspace/*` intent-to-add'd leftovers, and future skip-path diff-match attempts would compare a narrow scoped replay diff against the wide stored diff -> hunk-set Jaccard tanks via `1 / N` (N = total dirty files) and skip-path rejects byte-perfect replays. Caught live on pattern #125 with Jaccard 0.167 before this fix landed. The user-visible "Files changed" stat output gets the same scoping treatment via `_git_diff_stat(paths=...)`.

**Snapshot scope param (2026-05-06 same session):**

`core.tree_state.snapshot_dirty_tree(project_root, paths=None)` accepts an optional path-list scope. Used by `core/kb_graduation.graduate_pattern` to scope the snapshot to the same directories `_git_reset_hard` resets (`core skills agents tests interfaces`), so the captured patch only contains files that will actually be reverted. Without scoping, intent-to-add'd workspace files outside the reset scope would be in the patch but NOT reset, causing apply-collision on restore. Symptom seen on the first /code under the new graduation flow: 29 files captured, restore_dirty_tree apply-rejected with "git diff header lacks filename information when removing 1 leading pathname component". Fixed by passing the scope list through.

**Bot-side render fix (2026-05-06 same session):**

`interfaces/telegram_bot.py` /code result handler treats `solved_by="qwen_skip_path"` as ready-to-display markdown (same path as `qwen_agent`/`qwen_failed`). Without this branch, skip-path successes fell into the legacy "Here's the code:" wrapper that strips the body, hiding the "Solved via skip-path" message. Cosmetic but important for the Phase 16 vision visibility -- when skip-path fires, users now see the full success body in chat.

---

## Phase 17 — Pre-teach discipline + decomposition (2026-05-06)

### Batch 1 ✅ Shipped

**Trigger:** trace SEN-b203948c (2026-05-06 22:09Z) — /qcode prompt produced a 14692-char Claude response, truncated to 8000 chars at a step boundary, parsed to 1 STEP, kicked off a Phase-16 reformat retry. Root cause: Claude tried to one-shot a multi-component change. STEP-N format pressure had no upper bound on output size; once it overflowed the parser cap, the recipe degenerated.

**Two changes:**

1. **`PRE_TEACH_SYSTEM` tightening** (`skills/code_assist.py`)
   - Explicit "FIRST CHARACTERS OF YOUR OUTPUT MUST BE `STEP 1:`" rule (load-bearing fix for prose-prefix degeneracy)
   - Labeled `WRONG STEP-N output` negative example showing what the parser rejects (prose before STEP 1, prose between STEPs)
   - Explicit 3-7 STEP target window with rationale ("If you need more than ~8 STEPs, the task is too big -- use DECOMPOSE")
   - `GOOD STEP-N output` and `GOOD DECOMPOSE output` blocks framed as "emit ONLY this, nothing else"

2. **DECOMPOSE escape valve** (new `_extract_decomposition` + `_format_decomposition_response` helpers + pipeline short-circuit)
   - Claude rates task size during pre-teach. If too big (>~8 STEPs / >2 files / brand-new command lane), emits a `DECOMPOSE` first line + `- /code <subtask>` bullets instead of STEP-N
   - `_extract_decomposition(recipe)` does strict first-line match (`DECOMPOSE` or `DECOMPOSE:` only — narrative use of the word in prose does NOT trigger), then regex-extracts `/code`-prefixed bullets
   - `_run_agentic_pipeline` checks for decomposition AFTER `_claude_pre_teach` returns, BEFORE shadow plan / stepfed / KB write. On a hit, returns `CodeAssistOutput(solved_by="decompose_suggested", ...)` directly — no LLM, no executor, no graduation
   - User-facing render via `_format_decomposition_response`: 📋 Task too big for one recipe. + numbered `\`/code <subtask>\`` list + "Run them in order. /commit between each."
   - `interfaces/telegram_bot.py` /code result handler extended to treat `decompose_suggested` as ready-to-display markdown (same branch as `qwen_skip_path`) so the body shows in chat instead of falling into the legacy "Here's the code:" stripper

**Why Qwen never sees the big picture:** Decomposition is a Claude-side responsibility. Each subtask is its own /code invocation with its own KB retrieval, its own pre-teach, its own small recipe. Qwen sees one small recipe at a time, the same as always. The user is the orchestrator (running each `/code` + `/commit` in sequence). This is the lightweight version of the chain-of-/code idea — no orchestrator agent, no shared state across subtasks beyond what /commit captures.

**34 ECC tests** in `tests/test_phase17_decompose.py`:
- Group P (10) — `PRE_TEACH_SYSTEM` source-level checks (first-chars rule, no-prose rule, 3-7 target, 8K-cap warning, WRONG example, DECOMPOSE rules, bullet format, scope-not-retries, default-path framing, size sanity)
- Group D (13) — `_extract_decomposition` parser cases (None-on-normal-recipe, None-on-empty/None, 3-subtask happy path, colon variant, leading whitespace, prose-narrative does NOT trigger, first-line strict, no-bullets returns None, asterisk bullets, non-/code bullets filtered, single-subtask still valid, whitespace stripped)
- Group F (5) — `_format_decomposition_response` rendering (numbered list, "Task too big" marker, "in order" hint, /commit mention, single-subtask case)
- Group W (4) — pipeline wiring source-checks (extract called after pre_teach, `solved_by="decompose_suggested"` present, telegram render branch, short-circuit returns early)
- Group I (2) — module-level helper imports + sanity

**Regression:** 485 passed / 3 known-stale (legacy stash-dance assertions, replaced by Phase 16 tree_state path) on Phase 15+16+17 scoped suite. Zero new regressions.

---

### Batch 17a ✅ Shipped — `recover_stale()` zombie-task fix + `/kill` command

**Trigger (live, 2026-05-06 evening):** task `4386481397f3` (created at 22:09Z when user killed bot mid-/qcode) survived **three** bot restarts and silently wiped uncommitted source files via `_git_reset_hard("core skills agents tests interfaces")` between retry attempts. `recover_stale()`'s `updated_at < cutoff` filter missed it because the dying worker's last claim refreshed `updated_at` to NOW. By definition, ANY `processing` task at bot startup is a zombie — there's no live worker making progress.

**What landed:**

1. **`core/database.py::recover_stale(*, force_all_processing: bool = False)`** — keyword-only kwarg. When True, skips the `updated_at < cutoff` filter and recovers EVERY `processing` task. Default (False) preserves legacy behavior for periodic mid-run sweeps. Worker startup at `core/worker.py:140` now calls with `force_all_processing=True`.

2. **`tasks.kill_requested`** — new `INTEGER NOT NULL DEFAULT 0` column added via idempotent ALTER. Schema migration in `init_db()`; existing rows get default 0. Polled between attempts in `_run_agentic_pipeline`.

3. **`core/database.py`** — three new helpers: `request_kill(task_id)` (sets flag, returns True if killable), `is_kill_requested(task_id)` (pure read), `find_kill_target()` (most-recent processing task, used by /kill handler). Plus `get_task_by_trace_id(trace_id)` so the pipeline can resolve its own task_id from its trace_id without signature changes upstream.

4. **`/kill` Telegram command** — `interfaces/telegram_bot.py::handle_kill`. Looks up the most-recent processing task, sets `kill_requested=1`, replies with confirmation. `/kill` advertised in `BOT_COMMAND_MENU` so Telegram suggests it. Idempotent: second /kill on same task is a no-op. /kill with no running task replies "nothing to kill".

5. **Pipeline poll** — `_run_agentic_pipeline` looks up `kill_task_id` once via `get_task_by_trace_id(trace_id)` (best-effort; failure means /kill is a no-op for that run, never a crash). Top of each attempt iteration polls `is_kill_requested(kill_task_id)`; on True sets `bailed_on_kill = True` and breaks the loop. Worst-case latency between /kill and bail = current attempt duration (~60-120s for Claude pre-teach). Cannot interrupt mid-Claude-CLI subprocess; refusing to start the next attempt is the strongest cancellation primitive available.

6. **Kill-bail outcome shape** — new `solved_by="qwen_killed"`. Working tree reset to base_sha. **Does NOT call `kb.add_limitation`** — /kill is user-initiated, not a capability signal, and limitation rows are reserved for genuine "Qwen+Claude could not solve". Bot render branch updated to surface `qwen_killed` body via the ready-to-display markdown path.

**23 ECC tests** in `tests/test_phase17a_kill_recovery.py`:
- Group R (6) — recover_stale force mode (recently-updated zombie recovered, old zombies recovered, pending/completed untouched, default mode unchanged, orphan locks released, kwarg-only signature)
- Group K (10) — kill helpers (column exists, request_kill sets flag, idempotent, returns False on completed/missing, is_kill_requested missing-row safe, find_kill_target idle/picks-most-recent, get_task_by_trace_id round-trip + missing)
- Group W (7) — wiring source-checks (worker startup uses force flag, kill handler registered, BotFather menu, pipeline polls, kill skips add_limitation, telegram render includes qwen_killed, handler uses find_kill_target)

**Why kwarg-only on `force_all_processing`:** existing callers pass `timeout_seconds` positionally. Making the new flag keyword-only prevents accidental enabling via positional arg (e.g. `recover_stale(True)` would have meant `timeout_seconds=True` if positional was allowed — defensive). `inspect.signature` enforces this in test_r06.

**What this blocks (intentionally):** Phase 17b (chain runner) needs `/kill` to safely abort an 8-15 min auto-decompose chain. Without 17a, /restart is the only abort, and /restart leaves zombie tasks that re-trigger the wipe bug. 17a is a hard prereq for 17b.

---

### Batch 17b ✅ Shipped — auto-decompose chain runner

**The headline user-friendly feature:** non-engineer types `/code add a /qcode command`, system detects the task is too big, queues N child /code tasks back into its own worker queue, processes them sequentially, user sees one /code completion message per subtask in chat. No manual orchestration required.

**Builds on Phase 17 Batch 1** — the DECOMPOSE detection was already in place (`_extract_decomposition`). 17b adds the runner that picks up the DECOMPOSE list and submits each subtask back into Sentinel's existing task queue.

**What landed:**

1. **`tasks.parent_task_id TEXT`** + **`tasks.chain_depth INTEGER NOT NULL DEFAULT 0`** — two new columns on the tasks table (idempotent ALTERs in `init_db`). NULL parent = standalone /code; non-NULL = child of a chain. Depth 0 = top-level, incremented per child.

2. **`core/database.py::add_task(*, parent_task_id=None, chain_depth=0)`** — keyword-only kwargs (positional-arg-foot-gun protection enforced via `inspect.signature` test).

3. **`core/database.py::list_children(parent_task_id)`** + **`chain_status_summary(parent_task_id)`** — helpers for chain progress tracking. `list_children` returns rows ordered by `created_at`. `chain_status_summary` returns `{'total': N, 'completed': X, 'failed': Y, 'pending': Z, 'processing': W}` for "subtask 2/3 done" UX in future polish.

4. **Config flags** in `core/config.py`:
   - `CODE_CHAIN_ENABLED = False` — default OFF on first ship. Phase 17 Batch 1's manual DECOMPOSE surface still runs unchanged when False. Flip to True after live validation that the runner produces clean chains.
   - `CODE_CHAIN_MAX_DEPTH = 1` — children of a chain cannot themselves decompose. Without this cap, runaway recursion is possible.

5. **Pipeline wiring in `skills/code_assist.py::_run_agentic_pipeline`** — the existing decomposition branch was extended:
   - When `_extract_decomposition` fires AND `CODE_CHAIN_ENABLED=True` AND `self_chain_depth < CODE_CHAIN_MAX_DEPTH`: queue children via `database.add_task(parent_task_id=kill_task_id, chain_depth=self_chain_depth+1)`. Each subtask gets its own fresh `trace_id` (via `core.telemetry.generate_trace_id`).
   - The `/code ` prefix on subtask text is stripped before queueing so the bot's command router doesn't double-parse.
   - Returns new `solved_by="chain_started"` outcome with markdown body listing the queued subtasks.
   - When chain doesn't fire (flag off, depth-capped, or enqueue error): falls through to Phase 17 Batch 1's existing `solved_by="decompose_suggested"` markdown response — graceful degradation, never crash the /code.
   - Each enqueue logs `CHAIN-QUEUED child task_id=...` for trace-back. Enqueue errors log `CHAIN-ERROR` and fall through.

6. **`self_chain_depth` lookup** reuses the trace-id-mapped task row already fetched for 17a's kill polling — single DB query covers both kill polling and chain depth.

7. **Telegram render branch** extended in `interfaces/telegram_bot.py` — `chain_started` joins the ready-to-display markdown tuple alongside `qwen_killed` / `qwen_skip_path` etc.

**24 ECC tests** in `tests/test_phase17b_chain_runner.py`:
- Group C (10) — schema migration + add_task with parent + list_children + chain_status_summary + depth inheritance
- Group E (7) — enqueue path (flag off behavior, correct args, depth cap, kwarg-only enforcement, slash-prefix stripping, log marker, fall-through on error)
- Group W (7) — wiring source-checks (config flags, render branch, short-circuit return, decision-logging, depth threading, imports clean)

**What this does NOT do (intentionally):**
- No auto-/commit between subtasks. User /commits at the end. Preserves the **NO AUTO-COMMITS** owner directive.
- No fancy progress bar. Each child task reports its own /code completion in chat — N separate replies. Cleaner UX (per-subtask visibility) at the cost of single-thread polish.
- No mid-chain rollback on failure. Chain is fail-stop: if subtask 2 fails, subtasks 3+ remain pending until the worker's natural error propagation marks them. User can `git restore` if they want to start over.
- No depth>1 chains. CODE_CHAIN_MAX_DEPTH=1 by design — top-level decomposes ONCE; children must succeed as small recipes or fail fast. Prevents runaway.

**Prereq dependency:** Phase 17a's `/kill` is the abort mechanism for an 8-15 min auto-chain. Without 17a, /restart was the only abort, and /restart resurrected zombie tasks via the `recover_stale` bug fixed in 17a.

---

---

### Batch 17c ✅ Shipped — `CODE_TIERS.md` playbook injection

**The polish layer.** Phase 17 Batch 1 added DECOMPOSE as Claude's escape valve; tonight we discovered Claude can choose NOT to use it (judged a multi-component /qcode task as "I'll one-shot this" → 14692-char response → truncated to 1 STEP). Batch 17c adds structured tier guidance to the pre-teach prompt so Claude has explicit rules per task size, not just judgment.

**What landed:**

1. **`workspace/persona/CODE_TIERS.md`** — new curated playbook file (~3.2K chars). Four tiers documented:
   - **Tier 1 (basic)** — ≤2 STEPs, single file, idempotent. STEP-N directly, skip Read/Grep when prompt fully describes the change.
   - **Tier 2 (standard)** — 3-7 STEPs, ≤2 files. STEP-N is default. Read first for edit_file anchors.
   - **Tier 3 (advanced)** — multi-file, >7 STEPs, new command lane. **DO NOT one-shot.** DECOMPOSE mandatory. Worked example for "add /qcode command" included verbatim.
   - **Tier 4 (pipeline rebuild)** — schema migration, new agent, cross-cutting refactor. DECOMPOSE with extra care; subtasks must be independently revertible.

2. **`core/config.py`**:
   - `CODE_TIERS.md` added to `PROTECTED_FILES` (file_guard hash-watch, edits via `authorize_update` only — same protection as QWENCODER.md).
   - `PERSONA_INJECT_MAX_CHARS["CODE_TIERS.md"] = 4000` (the file is 3.2K, leaving headroom for additions).

3. **`skills/code_assist.py`**:
   - **`_load_code_tiers_memo()`** — fresh load on every /code call (no bot restart needed for curated edits to take effect). Capped at `PERSONA_INJECT_MAX_CHARS`. Best-effort: missing file or read error → `""`, never raises.
   - **`_classify_complexity_tier(problem)`** — runs `core.complexity.classify_complexity` with `command="code"`, returns `(tier, score)`. Tiers: `basic` / `standard` / `advanced`. Best-effort: any exception → `("standard", 0.5)` (neutral fallback).
   - **`_claude_pre_teach`** — appends the CODE_TIERS memo + a complexity verdict block to `PRE_TEACH_SYSTEM` before invoking Claude:
     ```
     ==== CODE_TIERS playbook (curated tier guidance) ====
     <memo content>
     ==== complexity verdict ====
     This task scored complexity=0.72 which maps to tier='advanced'.
     Read the matching tier section above and follow its action +
     anti-pattern guidance.
     ```
   - The memo is APPENDED (not prepended) so `PRE_TEACH_SYSTEM`'s strict format rules are read first; tier guidance refines those rules.
   - Pre-teach log line now surfaces tier + score for telemetry.

**Why a Claude-side playbook (vs QWENCODER.md which is Qwen-side):** Claude is the pre-teach author. When Claude judges "is this task too big for one recipe?", it needs an explicit rubric. Phase 17 Batch 1's DECOMPOSE escape worked only when Claude USED it. Tonight's failure showed Claude was given the option but chose to one-shot a multi-component task anyway. The tier playbook adds deterministic nudges + a complexity verdict.

**Why the complexity verdict is included verbatim in the prompt:** The static playbook tells Claude the rules; the verdict tells Claude THIS specific task's tier. Without the verdict, Claude still has to judge complexity itself. With it, the heuristic classifier (`core.complexity.classify_complexity`) does the work, and Claude follows the matching tier's rules.

**23 ECC tests** in `tests/test_phase17c_code_tiers.py`:
- Group T (7) — file content (exists, size, all 4 tiers, tier-3 mandates DECOMPOSE, anti-patterns documented, includes worked example, /qcode example present)
- Group L (7) — helper behavior (loads text, capped at PERSONA_INJECT_MAX_CHARS, returns empty on missing dir, swallows IO errors, classifier returns valid tier, handles empty input, falls back on classifier exception)
- Group W (9) — wiring (file_guard list, persona caps, pre-teach loads memo, runs classifier, appends after PRE_TEACH_SYSTEM, logs tier decision, complexity verdict block, falls through when memo missing, imports clean)

**Coupled test-window updates** (defensive; not new bugs): `tests/test_phase15c_planning.py::test_q31` window 8000→16000 + `tests/test_phase17_decompose.py::test_w04` window 2000→6000. These are static substring-search tests over the `_run_agentic_pipeline` body; Phase 17b's chain runner pushed the targets past the original windows. Window expanded; assertions unchanged.

---

### Batch 17d ✅ Shipped — chain-child auto-commit + bot relay + `/revert chain`

**Trigger (live, 2026-05-06 ~00:54Z):** Phase 17b chain runner fired correctly (decomposition → 2 child tasks queued → child 1 PASSED → child 2 grinded then bailed via Phase 15d shape-repetition). But user observed two real gaps:

1. **State loss between siblings.** Child 2's `_git_reset_hard("core skills agents tests interfaces")` between attempts wiped child 1's uncommitted edits to `core/config.py`. Child 1's NEW files (`skills/qcode_assist.py`, `agents/qcode_assistant.yaml`) survived (untracked-new outside reset's removal scope), but the modified-file edits were blasted. End-to-end autonomy was therefore broken even though child 1 was reported PASS.

2. **Invisibility of children in chat.** Each child task completed in the DB but the bot had no `chat_id` binding back to the parent's Telegram message. User saw the parent's "🔗 Chain started" notice but never saw subtask 1's success message because the bot wasn't polling for children.

**What landed:**

1. **`skills/code_assist._chain_child_auto_commit(...)`** — scoped exception to **NO AUTO-COMMITS** owner directive: chain-runner subtasks MUST commit between siblings to preserve state.
   - Stages ONLY the recipe-touched paths (`_extract_recipe_paths(recipe)`), not the wide `_COMMIT_INCLUDE` scope. Won't sweep up unrelated dirty work.
   - Commit identity: `chain-child@sentinel.local` + message prefix `sentinel-chain: child <id> of parent <id> -- <problem>`. Uniquely identifiable, greppable, revertible.
   - Best-effort: returns False on any failure (no recipe paths, nothing to commit, git failure) and never raises into `/code`.
   - Wired in `_run_agentic_pipeline` AFTER `solved_by="qwen_agent"` (success branch only) AND only when the task has a non-NULL `parent_task_id`. Standalone /code (no parent) is unaffected — directive preserved for that path.

2. **`interfaces/telegram_bot._relay_chain_children(parent_task_id)` + `_send_chain_child_result(child)`** — after a parent task returns `solved_by="chain_started"`, `handle_code` calls `_relay_chain_children` which polls `database.list_children(parent_task_id)` until all siblings reach a terminal state. Each newly-completed child is rendered to chat (prefixed with `📦 Subtask <task_id_short>`) so the user sees per-subtask completion messages in real-time. A rolling status message edits in place to show `chain progress N/M done`. Dedup via `already_relayed` set prevents re-sending.

3. **`/revert chain` Telegram command** — extends `handle_revert` with a chain dispatch path. `_handle_revert_chain` walks back from HEAD counting contiguous commits whose author email is `chain-child@sentinel.local` AND whose message starts with `sentinel-chain:` (BOTH required, defensive). Stops at the first non-chain commit. Then `git reset --hard HEAD~N` to undo all chain commits in one shot AND restore the working tree (vs the regular `/revert` which is `--soft` — keeps changes staged). One-line undo for any chain run.

**22 ECC tests** in `tests/test_phase17d_chain_commit.py`:
- Group C (8) — helper behavior: exists + async, no-op on empty paths, real commits get created, message uses `sentinel-chain:` prefix, identity is `chain-child@sentinel.local`, ONLY recipe paths committed (NOT unrelated dirty work), nothing-to-commit returns False, handles new files
- Group W (4) — pipeline wiring: helper called, in success branch only, gated on `parent_task_id`, uses `_extract_recipe_paths`
- Group R (5) — bot relay: methods exist, `handle_code` calls relay on `chain_started`, polls `list_children`, dedups already-relayed, prefixed with `Subtask` marker
- Group V (4) — `/revert chain`: handle_revert dispatches, filter on email AND message prefix (both required), uses `--hard` reset, friendly "nothing to revert" when HEAD isn't a chain
- Group I (1) — imports clean

**Test update (coupled):** `test_phase17b_chain_runner.test_w01` no longer asserts `CODE_CHAIN_ENABLED is False` (the flag was flipped to True after live validation). Now asserts the flag exists and is bool.

**Owner directive trade-off (worth being explicit):**
- **Before 17d:** "NO AUTO-COMMITS — only manual `/commit` from Telegram." Strict, owner-purity.
- **After 17d:** "NO AUTO-COMMITS, EXCEPT chain-runner children (parent_task_id is non-NULL). Standalone `/code` and graduation are unchanged." Pragmatic — required for end-to-end autonomy.
- **Mitigations baked in:** scoped to recipe paths only (no scope-blast), unique identity for grep-ability, `/revert chain` for one-shot undo, only fires on actual success (not partial runs), best-effort (failures don't crash /code).

---

### Batch 17e ✅ Shipped — DECOMPOSE matcher loose + path normalize

**Triggers (live, 2026-05-07):**
- ~01:14Z — Claude DECOMPOSE response with markdown decoration was rejected by strict first-line matcher → reformatted into STEP-N → chain runner never fired (entire run became monolithic)
- ~01:15Z — Recipe with `path="interfaces\telegram_bot.py"` had `\t` decoded to TAB → 4 of 11 STEPs failed silently with "file not found"

**Fixes:**
1. **`_extract_decomposition`** matcher loosened to `\bDECOMPOSE\b` (case-insensitive, word-boundary). Bullet list (`- /code ...`) is the real safety net — bulletless responses always reject regardless of first line.
2. **`_safe_resolve`** in `core/qwen_agent.py` now does `rel_path.replace("\\", "/")` before resolution. POSIX paths cross-platform; sandbox check unchanged.
3. **`DECOMPOSE-MISS` diagnostic log** when matcher rejects a short non-STEP response — surfaces Claude's actual first line for future iteration.

37 ECC tests in `tests/test_phase17e_decompose_loosen_path_norm.py` (matcher accept variants, matcher reject regression-guards, path normalization across forms, sandbox escape still blocked, integration with `tool_read_file`/`tool_edit_file`/`tool_write_file`).

---

### Batch 17f ✅ Shipped — reviewer recipe-promise verification + corrective tool-sig discipline

**Triggers (live, 2026-05-07 ~01:42-01:48Z):**
- Child 1 reviewer Read'd `skills/qcode_assist.py` and confirmed the new file, but **never verified** that the recipe's claimed edit to `core/config.py` was actually applied. PASS verdict was premature; child 1's commit `a78c217` was missing the `/qcode` entry in `COMMAND_AGENT_MAP`. Child 2 then ran assuming the entry existed.
- Child 2 attempt 2 corrective Claude wrote `edit_file path="x" text="..."` instead of `edit_file path="x" old="..." new="..."` — 5 of 6 STEPs failed with `missing 2 required positional arguments` or `unexpected keyword argument 'text'`. Tool-signature confusion under prompt pressure (8760-char corrective context).

**Fixes:**

1. **`_claude_review` system prompt** gains an explicit "RECIPE-PROMISE VERIFICATION (load-bearing)" section. For EVERY `edit_file` / `write_file` step in the recipe given, the reviewer must Read or Grep the target file and confirm the claimed change is actually present. Silent no-ops (anchor mismatch, wrong path, malformed args) are SILENT FAILURES that show up only if you check. Reasoning format must include the specific recipe step number being verified. Tool-call budget bumped 5 → 7 to make multi-step verification practical.

2. **`CORRECTIVE_SYSTEM` prompt** now lists tool signatures in EXACT form (`edit_file(path=..., old=..., new=...)` with all arg names) under a "TOOL-SIGNATURE DISCIPLINE (load-bearing)" header. Calls out the live failure mode (`text=` ≠ `old=`/`new=`) by name + date so the warning is concrete. Also adds `missing 2 required positional arguments` to the list of prior-verdict signals that should push toward `write_file` instead of edit_file.

14 ECC tests in `tests/test_phase17f_reviewer_corrective_tightening.py` covering both prompt updates + signature/return-shape regression guards on `_claude_review` and `_claude_corrective_teach`.

---

### Phase 17 ship summary (a + b + c + d + e + f, 2026-05-06 — 2026-05-07)

| Batch | Tests | Lines | Headline |
|---|---|---|---|
| **17a** | 23 ECC | +303 LOC, +132/-26 src | `recover_stale` zombie-task fix + `/kill` |
| **17b** | 24 ECC | +263 LOC, +209 src | Auto-decompose chain runner |
| **17c** | 23 ECC | +250 LOC, +135 src + new `CODE_TIERS.md` | Tier playbook injection |
| **Total** | **70 ECC** | ~830 LOC | Self-feeding /code that scales from one prompt to a multi-subtask feature build |

**Combined effect:** non-engineer types `/code build /qcode`, system runs the heuristic classifier → tier=advanced (verified via test_l05) → Claude pre-teach reads CODE_TIERS Tier 3 rules → emits DECOMPOSE → chain runner queues N subtasks → worker drains them sequentially → user sees N completion messages → `/commit` once at the end. Plus `/kill` for the abort path that didn't exist before tonight.

---

---

### Batch C — original spec (for reference)

**Readiness update (2026-05-06 post-stress-test):**
- ✅ Batch B (`core/diff_match.py`) shipped — gate is in place
- ✅ Batch C-safety (recipe linter + bash whitelist + nvidia-smi fix) shipped
- ✅ Batch D (auto-pin) shipped + **first observed AUTO-PIN in production** on pattern #101 (5/5, trace SEN-1266f1c7)
- ✅ Option A (dedup at `add_pattern`) shipped — counter growth path validated
- ✅ Stress test (20/20 PASS, mean agreement 0.922, all graduated 1/1, zero leaks) — safety net validated under diverse loads
- ✅ Config flags added (`config.SKIP_PATH_ENABLED=False`, `SKIP_PATH_MIN_PASSES=3`, `SKIP_PATH_FRESHNESS_DAYS=30`, `SKIP_PATH_AGREEMENT_FLOOR=0.5`, `SKIP_PATH_DIFF_MATCH_THRESHOLD=0.7`) — Batch C feature flags wired
- ⏳ Batch C implementation itself — pending
- ⏳ Phase 17 graduation tree-state redesign (temp-file diff serialization to replace stash dance, since auto-commit is permanently disabled) — should ship WITH Batch C (or just before) so replay's snapshot/restore is rock-solid

**Ship plan when authorized:**
1. Recovery point: `git tag pre-batch-C-YYYYMMDD-HHMMSS` + `cp knowledge.db knowledge.db.pre-batch-C-...`
2. Implement `KnowledgeBase.is_skip_eligible(pattern)` (uses the config thresholds above)
3. Implement tree-state snapshot/restore via tempfile diff (NOT stash) — addresses Phase 17 candidate
4. Wire skip-path branch into `_run_agentic_pipeline` BEFORE Claude pre-teach: lookup top match → eligibility check → lint check → if all pass AND `SKIP_PATH_ENABLED`, do replay; if any fails OR flag is False, log telemetry and fall through normally
5. Counter increment classification: only `recipe_broken` and `runtime_check_failed` failures count; environmental (Ollama down) does NOT
6. Few-shot exclusion: if replay fails, pass `exclude_pattern_ids=[failed_id]` into KB retrieval for the fallback pipeline so Claude doesn't re-derive the same broken recipe
7. ECC tests (~25-30) covering: per-clause eligibility, tree-state snapshot/restore (incl apply-reject path), diff-match wiring, fall-through, counter classification, few-shot exclusion
8. Stress test 2-3 scenarios end-to-end with telemetry-only mode (`SKIP_PATH_ENABLED=False`); verify metric distributions are sane
9. Flip `SKIP_PATH_ENABLED=True` only after telemetry confirms safety
10. Document in CLAUDE.md + PHASES.md



**Goal:** when /code retrieves a high-trust KB match, run stepfed against the stored recipe FIRST (skipping Claude pre-teach + reviewer); if diff-match accepts AND the run_bash verification step succeeds, store the pattern attempt as a real `solo_passes` increment, return result, done. If anything fails, fall through to the full Claude pipeline as the safety net.

**Skip-eligibility (D1 revised) -- in code as `KnowledgeBase.is_skip_eligible(pattern)`:**
```sql
pinned = 1
OR (
    solo_passes >= 3
    AND solo_attempts = solo_passes        -- perfect rate at small N
    AND last_verified_at > datetime('now', '-30 days')
    AND (qwen_plan_agreement IS NULL OR qwen_plan_agreement >= 0.5)
)
```

**Per-replay safety chain (D2 + D3 + D4 + Batch C-safety):**
1. Lookup top-1 KB match by hybrid retrieval. If `is_skip_eligible(top) is False`, skip the skip path.
2. Lint the pattern via `recipe_linter.lint_recipe_for_skip(pattern.solution_pattern)`. If `not safe`, skip; log the failed checks.
3. Snapshot working tree: `tempfile.NamedTemporaryFile()` + `git diff > tmp.patch` + `git checkout -- :/ ':!*.db' ':!logs/' ':!__pycache__/'` (the D3 strategy that doesn't depend on stash internals).
4. Run stepfed against the stored recipe.
5. After stepfed: `git diff` the new state, call `evaluate_diff_match(pattern.solution_code, new_diff)`.
6. If `accept is True` AND stepfed's `completed_via_done is True` AND no step errors AND syntax check passes on edited Python files:
   - Increment `solo_passes` and `solo_attempts` on the existing pattern (no new pattern created).
   - Apply the original dirty patch on top via `git apply tmp.patch` (best-effort; if reject, keep replay state and log the patch path so user can manually merge).
   - Return success.
7. Otherwise (any fail mode): `git checkout -- :/` to clean replay's edits, `git apply tmp.patch` to restore dirty state, fall through to full Claude pipeline. Increment `solo_attempts` only if failure is `recipe_broken` or `runtime_check_failed` (D4 classification); environmental failures (Ollama down / timeout-on-load) do NOT pollute the counter.
8. **Few-shot exclusion in fallback:** when the fallback pipeline retrieves KB context for Claude, exclude the failing pattern's id so Claude doesn't re-derive the same broken recipe.

**Counter semantics (key):** `solo_attempts` and `solo_passes` are now incremented by REPLAY-SKIP attempts (Batch C path), not just initial graduation (Phase 14b path). Same prompt 3-5×, each replay-skip success → counter climbs → eventually auto-pin (Batch D 5/5 threshold) fires. This is what makes the user's mental model literal.

**Telemetry-first ship:** initial Batch C should land with `SKIP_PATH_ENABLED = False` flag and ONLY record diff-match scores + lint verdicts as observability. After 1-2 days of telemetry showing the metric distributions are sane, flip the flag and enable actual skipping. Reduces blast radius of any miss.

**Test coverage required:** ~25-30 ECC tests covering skip-eligibility boolean (per-clause), tree-state snapshot/restore (including apply-reject path), diff-match wiring, fall-through behavior, counter increment classification, few-shot exclusion. Plus 2-3 stress-test scenarios end-to-end.

**Recovery point set this session:** `pre-bundle-BCsafety-20260506-104217` -- before Batch B + C-safety. Revert with `git reset --hard pre-bundle-BCsafety-20260506-104217`.

### Batch Option 1 — Recipe reformat retry on parser failure ✅ Shipped

**Problem caught in trace SEN-ca56136c (2026-05-06):** When Claude's pre-teach recipe didn't have parseable `STEP N:` blocks (Claude wrote prose, or used a different shape the local regex couldn't match), `run_agent_stepfed` silently fell back to legacy `run_agent` — the Phase-9 free-form Qwen path that doesn't pin Qwen to literal recipe commands. That trace burned 66s on legacy execution doing the wrong edit, then 20s on the failed reviewer-Claude pass. Pattern repeated across multiple traces.

**Fix:** Inserted a parse-check at the call site in `skills/code_assist.py` (the agentic attempt loop). If `_parse_recipe_steps(recipe)` returns <2 STEPs, the loop calls a new helper `_claude_reformat_recipe(prior_recipe, problem, trace_id)` — re-asks Claude with a strict format-only system prompt and `tools=[]` (no exploration). If reformat succeeds (>=2 STEPs), use the reformatted recipe. If reformat ALSO fails, fall through to stepfed's existing legacy fallback as the last-resort safety net (preserves current behavior when Claude is unavailable / completely uncooperative).

Saves ~70-90s per affected `/code`. Only triggers on a path that was already failing. Bounded downside: one extra ~15s Claude call when this path fires.

**Tests:** 17 new ECC in `tests/test_phase16_reformat.py` covering parser-boundary cases (R01-R03), reformat function behavior (R11-R19: unavailable, CLI error, prose extraction, bare blocks, oversize truncation, empty response, no-tools enforcement, system-prompt strictness), and source-level integration (R21-R25: import wired, reformat called in loop, legacy fallback preserved on retry exhaustion, log markers). 166/166 targeted regression green (Phase 15a-g + Phase 16 Batch 0/A/D + Option 1).

**Recovery tags:** `pre-batch-option1-20260506-095716` (HEAD `0e9fced`, before any Option 1 code) and `post-batch-option1-20260506-101120` (HEAD `5bd01bd`, current state).

### Snapshot scope footgun — too WIDE flavor (caught 2026-05-06 during Option 1 ship)

**Phase 15e's `_git_commit_for_graduation` does `git add -- core skills agents tests interfaces workspace` to commit /code's working-tree changes before graduation runs.** The original Phase 15e bug was the scope being too NARROW (excludes project-root files like `math_utils.py` — caught during Batch A stress test). During the Option 1 ship session, the OPPOSITE flavor surfaced: the scope is also too WIDE — it sweeps in any uncommitted work that happens to be in those scoped directories.

Concrete instance: while implementing Option 1 (uncommitted edits to `skills/code_assist.py` and the new `tests/test_phase16_reformat.py`), the user ran two `/code` requests in parallel. The grad-snapshot at the end of the second `/code` run (`4662c97 sentinel-grad: /code: Write a Python function that returns the sum of two numbers`) **swept all 403 lines of in-progress Option 1 work into a commit whose message describes a 4-line `add(a, b)` function in `core/util.py`.**

Net effect: Option 1's code IS in HEAD and tests pass — but the commit history is misleading. `git log -- skills/code_assist.py` shows Option 1 landing under the wrong commit message. Future blame will point engineers at the wrong context.

**Phase 17 candidates (related and bundleable):**
1. **Snapshot scope: include project-root files** (the original Phase 15e ticket — Batch A stress-test S1 surfacing).
2. **Snapshot scope: only files touched by the recipe** — track which paths the agent_result actually edited and `git add` only those, not a directory scope. Eliminates BOTH flavors in one fix. Cleanest path.
3. **Refuse grad-snapshot if working tree has unrelated uncommitted changes** — defensive: detect `git diff --name-only` paths NOT in the recipe's edit set, abort with a warning instead of sweeping them in. Preserves user's in-progress work.

The user has discretion to leave the misleading commit as-is (work is preserved, tag `option1-code-landed-at -> 4662c97` documents the entanglement) or rewrite history with `git reset --soft HEAD~2` and a clean re-commit. No action taken without their authorization.

---

## Phase 15g — Preload KB with 39 curated patterns (2026-05-06)

**Goal:** The Phase 15e regression destroyed ~28 post-Phase-14b patterns and left only May-4 Phase-9 testbed examples describing a codebase that no longer exists. User-observable symptoms: /code iterating 3 attempts on familiar tasks (`SEN-fec42f17` spent 4m18s and $0.61 on a single emoji-swap before getting to PASS), shadow agreement stuck at 0.000 because Qwen had no relevant few-shot examples to pattern-match against, and the bot "felt less intelligent" than pre-regression. More /code teaches would organically rebuild context but slowly; Phase 15g rebuilds from a non-zero baseline.

### What landed (commit `f00d412`)

- **`tools/preload_kb.py`** — 39 hand-curated patterns covering the user's actual workflow surfaces. Patterns are REPRESENTATIVE (placeholder names like `NEW_COL`, `OLD_FILLED`/`NEW_FILLED`) not literal — Claude pre-teach reads them as "shape" and writes a real recipe targeting actual code. Durable across codebase evolution; doesn't go stale on refactor.
  - Telegram bot UX (8): /command + handler, _build_bar emoji swap, brain dispatch lane, send_alert wiring, authorize new user, _send_long chunking, BotFather menu, two-step approval token
  - KB / KnowledgeBase (6): idempotent ALTER, query method, GROUP BY stats, partial index, NULL-column backfill, hybrid search weight tuning
  - Memory (4): episodic store + recall, semantic store with origin context, cross-scope FTS5 search, custom Episode field migration
  - SQLite tables (3): tasks-table column ALTER, FTS5 table with insert/delete triggers, idempotent upsert
  - Skills + Agents (4): BaseSkill subclass, accepts_list/output_is_list fan-out, agent YAML + COMMAND_AGENT_MAP, skill validation step
  - Tests (4): phase ECC test file, monkeypatch-based LLM mock, fresh_kb tmp_path fixture, source-grep wiring test
  - Internal handlers + scheduler (2): new INTERNAL_HANDLERS entry + cron, modify scheduled job
  - Brain + Claude + Qwen (3): brain prompt extension, Claude CLI subprocess invocation, QWENCODER.md curator-entry
  - Math/util/config (4): math_utils with validation, core/util.py string + list, core/config constant
  - Async (1): asyncio.to_thread for blocking IO
- **Each pattern includes**: result-oriented problem_summary mirroring real /code prompts; STEP-N recipe following the QWENCODER.md contract (read→edit→verify→done); diff body that passes `_is_real_solution` (Phase 15d-bugfix-2); pre-computed shadow recipe + agreement score via `core.plan_agreement.score_plan_agreement`; PINNED via Phase 15a on insert so the auto-walker can never archive.
- **`tools/preload_kb.py` is idempotent on re-runs** — detects existing problem_summary matches and skips. `--replace` flag forces re-insertion when the canonical patterns evolve.
- **Test file `tests/test_phase15g_preload.py`** — 19 invariants:
  - Structural (8): list non-empty (≥30), required keys, unique problem_summaries, tags-list-of-str, explanation ≥100 chars, recipe parses ≥2 STEPs, shadow recipe parses ≥1 STEP, solution_code passes the quality gate
  - Insertion (6): inserts every pattern, every row pinned, shadow data populated, agreement in [0,1], idempotent, --replace works
  - Coverage breadth (5): ≥5 telegram, ≥3 sqlite/schema, ≥2 test, ≥2 memory, ≥2 skill/agent

### Live results

```
BEFORE preload:  37 rows,  2 shadow rows, mean agreement 0.000
AFTER preload:   76 rows, 41 shadow rows, mean agreement 0.598
FTS5 top hit for "telegram OR progress bar OR emoji":
  id=43 "change the telegram progress bar emojis" (preloaded)  ← top hit
  id=46 "make telegram messages over 4000 chars split correctly"
  id=41 "i want to change my progress bar in telegram to use emojis"
```

The preloaded pattern beats the May-4 stale ones at FTS5 retrieval for the canonical user prompt. `/kb planning` now reports real percentiles instead of the stuck-at-zero baseline.

### Decisions captured

- **39 patterns hits the sweet spot.** Below ~30 the FTS5 retrieval starts leaving common surfaces uncovered (the user types something the system hasn't seen, falls back to whatever's closest, gets bad recipes). Above ~50 the marginal value drops because each /code only injects top-5 hits at `KNOWLEDGE_CONTEXT_MAX_CHARS=4000`. 39 covers the user's observed workflow surface with headroom for organic accumulation.
- **REPRESENTATIVE not literal.** Recipes use placeholder names instead of current literal code. Three reasons: (1) durable across refactor — the pattern doesn't become stale every time `_build_bar` is renamed; (2) Claude pre-teach is good at reading shape and writing specifics — feeding it literals just constrains its judgement; (3) the production graduation (Phase 14b) replays the STORED recipe through stepfed, and a literal recipe targeting code-state-X breaks if the tree has moved on.
- **Shadow recipes are SHORTER than canonical.** A 3B coder model writing tools-less in shadow mode genuinely produces shorter recipes; the pre-computed agreement reflects this honestly (mean 0.598, not synthetically inflated). When the production shadow plan runs on a real /code, the score is comparable signal.
- **PINNED on insert.** They're foundation. Even if a pattern goes 30 days unused, the auto_transition_lifecycle won't archive it. `/kb unpin <id>` if you want a pattern in normal lifecycle later.
- **Idempotent + --replace.** Re-running preload on an already-seeded KB skips by problem_summary match. `--replace` re-inserts when canonical patterns evolve — for example, if Phase 16 changes the recipe contract, re-run preload --replace to re-seed updated shapes.

### Phase 15h candidates surfaced

- **Coverage gaps observed during live use.** When the user runs a /code that hits a pattern not in the preload, log it; if the same gap shows up 3+ times, propose adding a pattern via a curate flow.
- **Shadow agreement decay tracking.** As the codebase evolves, a preloaded pattern's `qwen_plan_agreement` will diverge from what the live shadow plan would compute today. A nightly check (alongside `kb_lifecycle`) could re-score preloaded patterns and flag drift.
- **Per-domain preload modules.** Right now `tools/preload_kb.py` is one big PATTERNS list. Could split into domain-specific files (`preload_telegram.py`, `preload_kb.py`, etc.) for easier curation when patterns proliferate.

---

## Phase 15f — Untrack runtime files: close the regression vector (2026-05-06)

**Goal:** Stop git operations from silently wiping the bot's runtime state. Caught monitoring the FIRST `/code` under Phase 15e: the new `sentinel-grad` commit shipped, graduation ran, `git stash push -u` was called — and stash's internal "reset working tree to HEAD" step **silently overwrote `knowledge.db`** with HEAD's blob (167936 bytes, the May-4-Phase-9 schema with 35 rows and no Phase 14a/14b/15a/15b/15c columns) before failing on `logs/sentinel.jsonl` (Windows file lock from the running bot). The reset didn't roll back what it had already done.

**Hash confirmed**: on-disk `knowledge.db` hash equals HEAD's blob hash (`4e7d4182f164eeb5e16b0fefc100762d62ae6af8`). Not corrupt — just the OLD version. Lost: ~28 patterns (40-67), all post-Phase-14 shadow data including the live agreement scores from this session.

### Root cause

`knowledge.db`, `sentinel.db`, and `logs/sentinel.jsonl` were committed during early Phase 9 testing on May 4 — **BEFORE** `.gitignore` was written. Once tracked, `.gitignore` is irrelevant; git keeps tracking them. Every git operation that touches HEAD's tree (stash, checkout, reset --hard, even some merge paths) becomes a loaded gun pointed at runtime state.

Phase 15e was the trigger, not the cause. The bug existed dormant since May 4 — Phase 15e's new `sentinel-grad` commit was the first git operation that consistently fired the stash-with-tracked-runtime-files path on every successful /code.

### What landed (commit `d9c522a`)

**142 files untracked** via `git rm --cached` (preserves on-disk content; only removes from index):

- **3 critical regression vectors** — `knowledge.db`, `sentinel.db`, `logs/sentinel.jsonl`
- **128 .pyc bytecode files** under all `__pycache__/` dirs (auto-regenerable)
- **7 `workspace/job_searches/*`** files (per-run user output)
- **7 `workspace/research/*`** files (per-run user output)

`memory.db` was already correctly untracked by accidental luck of timing.

The .gitignore patterns backing this were already in place (`*.db`, `__pycache__/`, `logs/`, `workspace/job_searches/`, `workspace/research/`) — Phase 15f just removed the ALREADY-tracked entries that were grandfathered through the index.

### Decisions captured

- **`git rm --cached` preserves disk state.** No `--force` flag needed; the working tree is unchanged. Only the index update is committed. Bot's running knowledge.db survives untouched (until it next reads from a stash/reset path, which 15f neutralises).
- **Why we didn't recover lost patterns from git history.** Every commit that ever touched `knowledge.db` (39 of them, all from May 4) has the same 167936-byte blob. No commit captured a post-Phase-14a state. `git fsck --full --unreachable` found 4 dangling blobs — none are SQLite databases. WAL/SHM files don't exist (clean checkpoint on bot kill). No `backups/` directory (the nightly cron at 03:30 EST hadn't run yet). Lost data is genuinely lost.
- **Forward-only.** The 35 surviving rows are real KB data from May 4. On next bot start, `KnowledgeBase()._init_schema()` runs the idempotent ALTER blocks; missing columns get added back; surviving rows get default values for the new columns (`state='active'`, `pinned=0`, `created_by_origin='foreground'`, etc.). No backfill possible for the lost shadow agreement scores or graduation results.
- **Defense in depth.** Untracking the .pyc files alone closes a separate latent bug — a stale checkout could have brought back old bytecode for currently-running modules. Untracking workspace output cleans up ~14 files that were never meant to be tracked.
- **Source files protected by regression guard.** `test_phase15f_untracked_runtime.py::test_f31-f34` assert that `core/*.py`, `skills/*.py`, recent test files, and the docs are still tracked — catches accidental over-untracking via glob.
- **The Phase 15e snapshot commit pattern is preserved.** Phase 15e's `_git_commit_for_graduation` is unaffected; it commits source files only (scope: `core skills agents tests interfaces workspace`), and now those scopes don't contain runtime artifacts.

### Tests

- `tests/test_phase15f_untracked_runtime.py` — **14 new tests, ~3s**:
  - Critical (4): `knowledge.db`, `sentinel.db`, `memory.db`, `logs/sentinel.jsonl` all NOT in `git ls-files`
  - Cleanup (3): no `.pyc`, no `workspace/job_searches/*`, no `workspace/research/*` tracked
  - .gitignore sanity (3): the patterns backing the cleanup are present
  - Source-stays-tracked guard (4): core/skills/tests/docs all still tracked
- Targeted regression: **169/169 green** (15f + 15e + 15d + 15c + 15b + KB + Phase 14 + 15a, 1m42s).

### Live verification

Bot was force-killed (PID 19796) before commit to prevent further damage. Next bot start runs the idempotent schema migrations on the regressed `knowledge.db`; surviving 35 rows get the missing Phase 14/15 columns added with default values. From `d9c522a` forward, no future git operation can silently wipe runtime state — there are no SQLite databases or active log files in the index for stash/checkout/reset to overwrite.

### Phase 15g candidates surfaced

- **Worktree-isolated graduation** (was already a Phase 15f candidate, still relevant). Run graduation via `git worktree add` — even with 15f's untracking, separate-worktree is architecturally cleanest because it removes ALL graduation-side git operations from the user's working tree.
- **CI guard against re-tracking.** A pre-commit hook (or just CI test) that fails when any `*.db`, `*.pyc`, `logs/*.jsonl`, or `workspace/job_searches/*` file appears in the index. Test_f01-f13 covers this for the running test suite, but a CI hook would catch it earlier.
- **Recover the 8 patterns from August's commits if any.** None of the 39 commits touching knowledge.db preserved post-Phase-14a state. Final check showed no recoverable SQLite blobs in dangling git objects either. Genuine loss; not actionable.
- **Document the bot-startup migration story.** When the bot starts against a regressed knowledge.db, the schema migrations run silently. Adding a startup log line `"applied N idempotent ALTER statements; rows recovered with defaults"` would make future regressions visible faster.

---

## Phase 15e — Snapshot /code's work before graduation runs (2026-05-06)

**Goal:** Stop graduation from silently destroying /code's working-tree changes. Caught monitoring pattern #66: the bot reported PASS, claimed it updated `_build_bar` to blue/white emojis, graduation passed 1/1, but after restart the file was STILL on orange/black. Working tree clean, no diff, last commit on the file was `cca4d6a` (an unrelated docs commit). The change had been silently rolled back by graduation's tree-state discipline.

### Root cause — two-step destruction

1. **/code attempt 1's `edit_file`** changes the working tree to blue/white. Claude review reads the file, confirms PASS.
2. **Phase 14a/14b graduation** runs immediately after PASS. With the Phase-10 owner-commits-manually contract, `pre_grad_sha = HEAD = base_sha` (no commit between /code and graduation). The discipline:
   - `git stash push -u` — silently FAILS to capture the working-tree change in production (the stash result handler had a silent path: rc != 0 OR "No local changes to save" both bypassed any log line)
   - `git reset --hard base_sha` — wipes the change
   - stepfed re-runs the recipe — working tree is blue/white again
   - `git reset --hard pre_grad_sha` (== base_sha) — wipes it AGAIN
   - `stash_token is None` — no pop attempt
   - Working tree left at base_sha state. Change GONE.

Graduation reports PASS because stepfed actually reproduced the recipe on the clean tree. The user-facing /code change was silently rolled back.

### What landed (commit `ef42e4a`)

**Fix A — real commit before graduation runs** (`skills/code_assist.py`):
- New `_git_commit_for_graduation()` helper, DISTINCT from the Phase-10 no-op `_git_commit_changes` (the per-attempt auto-commit-pollution rationale is preserved for that path; some downstream paths still call it for the `git add -N` side-effect that lets diff helpers see untracked files — test E04 enforces both helpers coexist).
- Runs ONCE on success, BEFORE `graduate_pattern()`, AFTER `kb.add_pattern()`.
- Scoped `git add -- core skills agents tests interfaces workspace` (no `logs/`, `*.db`, `__pycache__`).
- Sentinel identity: `user.email=sentinel-grad@sentinel.local`, `user.name=Sentinel-Grad-Snapshot`.
- Message prefix: `sentinel-grad: <problem[:80]>` for greppability.
- Returns the new commit SHA on success, `None` on no-change/fail (best-effort: a failed commit just reverts to the old behavior for that one /code).

Net effect: `pre_grad_sha` now points to a SHA that INCLUDES /code's work. Graduation's final reset to `pre_grad_sha` RESTORES the change instead of reverting it. The stash dance becomes a robust redundancy rather than the load-bearing path.

**Fix B — unconditional stash diagnostic logging** (`core/kb_graduation.py`):
- WARNING with rc + stderr on `git stash push` failure
- DEBUG when stash returns "No local changes to save" (expected post-15e because /code now committed first)
- INFO with stdout excerpt on success

Pre-15e the silent paths left no audit trail when graduation later destroyed working-tree changes. Pattern #66 diagnosis took several rounds because we couldn't tell whether stash captured anything. Now: every outcome leaves one log line.

### Decisions captured

- **The Phase-10 contract is partially relaxed, not abandoned.** Phase 10 was about avoiding PER-ATTEMPT pollution (5 commits per /code with concurrent edits). Phase 15e adds ONE commit per SUCCESSFUL /code, with `sentinel-grad:` prefix and `sentinel-grad@` identity. Owner can `git log --invert-grep --grep=sentinel-grad` to filter, or `git reset HEAD~N` to drop them. The original concern (per-attempt pollution) is unaffected — _git_commit_changes is still the no-op.
- **The alternative (separate worktree) was deferred.** Architecturally cleaner — graduation in `/tmp/grad-XXX` worktree, never touch user's working tree, no stash dance needed. ~5x more code than the snapshot-commit approach. Documented as Phase 15f candidate.
- **Helper returns SHA, caller doesn't consume it (yet).** Typed for future graduation-coupling work where graduation might want to verify the snapshot is what it expects.
- **Lost data: patterns #62, #64, #65** each ran /code with a real change that DID NOT persist. They show up in KB with diff text reflecting the intended change, but the actual on-disk state never matched. Not recoverable — the working-tree changes were destroyed in real time. From `ef42e4a` forward, every /code success leaves a permanent git commit before graduation can touch the tree.
- **Scope of `git add` excludes `logs/`, `*.db`, `__pycache__`.** Same scope used by `_git_reset_hard` (`core skills agents tests interfaces`) plus `workspace`. Prevents the snapshot commit from including runtime artifacts that constantly mutate.

### Tests

- `tests/test_phase15e_grad_snapshot.py` — **12 new tests, ~3s** (no GPU, no LLM, all source-level + signature inspection). Coverage:
  - Source-level wiring (4): helper exists with right signature, success path calls it BEFORE `graduate_pattern()` AFTER `add_pattern()`, uses `input_data.problem` as message, no-op `_git_commit_changes` is preserved (back-compat)
  - Helper behavior (3): scoped add excludes `logs`/`*.db`/`__pycache__`, sentinel-grad identity in source, `sentinel-grad:` message prefix
  - Diagnostic logging (3): stash failure WARNING + rc + stderr, empty DEBUG, success INFO with stdout excerpt
  - Sanity (2): `Phase 15e` marker in source, both helpers coexist
- Targeted regression: 155/155 green (15e + 15d + 15c + 15b + KB + Phase 14 + 15a, 1m42s).

### Live verification

Waits on next /code after restart. The next pattern stored should leave the working-tree change in place after restart for the first time since Phase 10. Easy to confirm: `git log --grep=sentinel-grad --since="1 hour ago"` should show one commit per successful /code, AND the user-observed file state should match what the bot's solution message describes.

### Phase 15f candidates surfaced

- **Run graduation in a separate worktree.** `git worktree add /tmp/sentinel-grad-XXX <base_sha>`, run stepfed there, no touching user's working tree. Architecturally cleaner; eliminates the stash dance entirely; never needs the snapshot commit.
- **`/kb forget <id>` true-delete command.** Phase 15a deferred this; remains relevant.
- **Drop sentinel-grad commits older than N days.** A scheduled `INTERNAL_HANDLERS["prune_grad_snapshots"]` could `git rebase --onto <newer> --keep` to drop commits matching the `sentinel-grad:` prefix. Or just leave them — they're filtered cosmetically already.
- **Surface "what did /code actually change?" in the bot's reply.** Currently the bot reports the diff stat, but if the grad-snapshot commit doesn't include changes (no-op recipe like Claude's old emoji-to-emoji loop), we should warn explicitly.

---

## Phase 15d — /code resilience hardening + QWENCODER.md (2026-05-06)

**Goal:** Six fixes targeting the multi-attempt failure mode caught live in the Phase 15c monitor session — owner ran `/code change the progress bar for all tasks except "code"...` and it ate 5/5 attempts in 16m52s, $1.05 in Claude calls, no pattern stored. Phase 15c shadow planning was visible in the trace (qwen recipe 59 chars, agreement=0.000) and worked exactly as designed; the failures were long-standing /code footguns the higher observability surface finally made undeniable. Phase 15d makes them stop happening.

### What landed (commit `24a614a`)

**Fix 1 — step-boundary recipe truncation.** Pre-15d cap was a hard char cut at 4000. Recipes 6000-8000 chars got chopped MID-string on the last step's `new=` arg, producing a malformed `edit_file path=... old="..."` that the parser ran and the executor crashed on. Hit attempt 1 step 6, attempt 2, and attempt 5 in the live trace. New `_truncate_recipe_to_steps()` walks STEP blocks in order and drops the trailing partial step at a boundary; falls back to char-cut for inputs with no STEP markers (defensive).

**Fix 2 — bigger recipe budget for stepfed mode.** 4000-char cap was Phase 9's defense against "long recipes push Qwen into prose-mode". Stepfed parses each step individually so prose drift across step boundaries doesn't apply. New constant `RECIPE_MAX_CHARS_STEPFED = 8000`. Multi-handler edits like the live trace's task finally fit unmolested.

**Fix 3 — bail on review-reasoning shape repetition.** Live trace spent ~9 minutes on attempts 3-5 chasing variants of the same Claude-recipe-quality issue. After attempt 2+, `_shape_repetition_phrase` compares the latest review reasoning against every prior one for a verbatim 5-gram of NON-stopword tokens. Match → bail, store limitation marked `"bailed on shape repetition (phrase X)"`, save the remaining attempts. Stopword filter blocks generic phrases like "Read confirms that the" from triggering.

**Fix 4 — already-read files hint to corrective_teach.** Each corrective attempt re-explored the same files (~120-200s of Claude time per attempt). Tree gets reset to `base_sha` between attempts so files are byte-for-byte identical. New `_extract_project_paths()` pulls project-relative-shaped paths out of the prior recipe + last review reasoning. Threaded into `_claude_corrective_teach(files_already_read=...)`; corrective prompt now includes a "FILES YOU'VE ALREADY READ — skip re-reading" section. Advisory hint, not a hard constraint.

**Fix 5 — shadow data on limitations.** Phase 15c only piped `qwen_plan_recipe` + `qwen_plan_agreement` into `add_pattern` (success path). Live trace ended in a limitation write with NULL shadow columns even though shadow data WAS captured on attempt 1 — wasteful. `add_limitation` now accepts the same two optional kwargs; the failure-path explanation also distinguishes "exhausted attempts" from "bailed on shape repetition" so the audit trail stays accurate. Signal is arguably MORE useful on a limitation: "could Qwen have planned this hopeless task?" answers Claude-quality vs Qwen-capacity at the failure point.

**Fix 6 — QWENCODER.md teaching memo.** New persona file at `workspace/persona/QWENCODER.md`, file_guard-protected like the existing four. Senior-engineer-level content: recipe contract reference, tool table with required args, argument-quoting rules, edit_file rules (where most failures live), verification step pattern, cross-file wiring footgun, failure-mode cheat sheet keyed to specific Claude-review phrases, recipe length guidance, KB context usage, and a `<!-- CURATOR-ENTRIES -->` section reserved for the future `/curate qwencoder` flow. Loaded fresh into the shadow-plan system prompt on every call via `_qwen_shadow_system_prompt()` (BASE + memo, memo APPENDED so strict contract reads first). Loader is best-effort: missing file or read error returns `""` and the BASE prompt still has the contract — never raises into /code.

### Decisions captured

- **Memo is APPENDED, not prepended.** Position matters for small-context models. The strict format contract reads first; the longer teaching memo follows. Reversing this risks Qwen "interpreting" the contract through the memo's worked examples.
- **Bail-on-repetition uses 5-grams.** 4-gram matches are too loose and trigger false bails on shared sentence fragments; 6-gram is too tight to catch the failure pattern we care about. Tuned to the ~10-token shared phrases observed in the live trace's reviews.
- **Stopword filter blocks all-stopwords windows.** Without it, "the diff has been applied" would trigger bail just because Claude uses similar review boilerplate. Filter requires >50% non-stopword tokens.
- **Truncator preserves at least one whole step.** Some output beats none; a downstream parser will catch any damage. The first-step-too-fat case still returns the truncated step (not empty).
- **Hint is advisory, not a hard skip.** Claude can still Read again if it genuinely needs to re-verify a file. Removing the option would force Claude to trust the hint blindly, which is unsafe — it's a teaching prompt, not a skip directive.
- **QWENCODER.md is in `PROTECTED_FILES`.** Same posture as the brain's persona files. Edits trigger the file_guard heartbeat; sanctioned writes go through `authorize_update`.
- **The 5-gram is symmetric across all prior reviews, not just N-1.** A failure shape that shows up on attempts 1, 3 (with attempt 2 different) still triggers bail on attempt 3. We're looking for "we're chasing variants", not "consecutive chase".

### Tests

- `tests/test_phase15d_resilience.py` — **26 new tests** covering all six fixes:
  - Truncation (4): short pass-through, drops trailing partial, first-step-alone, no-STEPs fallback
  - Budget (1): constant equals 8000
  - Bail (5): identical reasonings match, different ones don't, all-stopwords rejected, empty inputs no-bail, 4-gram threshold
  - Path extraction (5): from recipe, from reasoning, ignores absolute, ignores bare names, signature has kwarg
  - Limitation shadow (3): kwargs flow through, back-compat, source-level call site verification
  - QWENCODER.md (8): file exists + non-trivial size, in PROTECTED_FILES, has injection cap, loader reads file, missing returns "", composer appends memo, shadow plan source uses dynamic composer, memo content has canonical contract
- Targeted regression: 15d + 15c + 15b + KB + Phase 14 + 15a → **132/132 green** (1m43s).

### Phase 15e candidates surfaced

- **`/curate qwencoder` flow.** The CURATOR-ENTRIES section in QWENCODER.md is currently empty. A scheduled or on-demand curate variant would: scan `logs/sentinel.jsonl` for low-agreement traces, ask Claude to extract the failure shapes Qwen got wrong, propose deltas to QWENCODER.md, and apply via `file_guard.authorize_update` after user approval (Phase 10's curate flow shape).
- **Production stepfed reads QWENCODER.md too.** Today only the shadow plan injects the memo. The production executor's transcription LLM call could benefit too — though the risk-reward is lower (transcription is a much narrower task than recipe generation).
- **Bail-on-repetition counts toward solo_attempts in graduation.** Currently bail just stores a limitation; it doesn't update graduation stats. A bailed limitation could count as a "Qwen could-not-plan" data point in the planning_stats aggregate.
- **Shadow-data-on-limitation telemetry surface.** `/kb planning` aggregates only patterns. Add a `/kb limitations-planning` view to surface "limitations where Qwen actually agreed with Claude's plan but Qwen-the-executor still failed" — a strong signal for hardware-tier upgrades.
- **Stash-on-dirty-tree before /code.** Phase 14b's stash-on-graduation was best-effort. /code attempt-1 doesn't snapshot the dirty tree at all. If a /code crashes mid-attempt the user could lose uncommitted work.

---

## Phase 15c — Qwen-planning shadow tests (2026-05-06)

**Goal:** Phase 14 graduation answered "can Qwen *execute* Claude's recipe?" Phase 15c starts answering the prior question: "could Qwen have *planned* the same recipe?" If Qwen-with-KB-context produces a recipe structurally similar to Claude's, then on bigger hardware (or with more KB compounding), Qwen could become Claude-optional. We need a baseline column to track that signal over time.

### What landed (commit `c90488c`)

- **Schema (idempotent ALTER, both nullable on the `knowledge` table):**
  - `qwen_plan_recipe TEXT` — what Qwen wrote in shadow mode
  - `qwen_plan_agreement REAL` — structural similarity score, clamped to [0.0, 1.0]
  - NULL on patterns written before 15c, AND on /code attempts where the shadow call timed out / crashed. Both states stay distinguishable from "shadow ran and agreed on nothing" (0.0) — important for the planning_stats math.
- **`core/plan_agreement.py`:** pure-heuristic scorer. `score_plan_agreement(claude, qwen) -> float ∈ [0.0, 1.0]` blends 0.5×file-path Jaccard + 0.3×tool-name Jaccard + 0.2×step-count proximity. Reuses `core/qwen_agent._parse_recipe_steps` + `_parse_step_text_to_tool_call` so we score against EXACTLY what the executor would parse, not a parallel re-parser that could drift. Both-empty → 0.0; parser exception → 0.0; never raises (best-effort by design — must not break /code).
- **Shadow plan call site (`skills/code_assist.py::_run_agentic_pipeline`):** after `_claude_pre_teach` returns the canonical recipe and BEFORE Qwen executes, the new `_qwen_shadow_plan()` asks the worker model for the same recipe using identical KB context + project_map. Wrapped in `asyncio.wait_for(..., timeout=QWEN_SHADOW_TIMEOUT_S=30)`; ANY failure returns `None` and /code continues unaffected. The shadow state is captured on attempt 1 and survives across retries so the eventual `kb.add_pattern` stamps them on whichever attempt wins.
- **`KnowledgeBase.add_pattern` + `_add` signatures gain two optional kwargs** (`qwen_plan_recipe`, `qwen_plan_agreement`). Defaults are `None` — paths that don't have shadow data write NULL. Out-of-range floats are clamped server-side (defensive: callers already do this, but a stray NaN would cascade through stats math).
- **`KnowledgeBase.planning_stats()`:** aggregate query for the new Telegram surface. Returns `patterns_total`, `patterns_with_shadow`, `mean_agreement`, `p25` / `p50` / `p75` (sorted-vector bucket quantile — right shape for chat readout, not textbook stats), and a `by_archetype` per-tag rollup (groups by FIRST tag in the comma-separated tag list).
- **Telegram `/kb planning` subcommand:** header + percentile line + per-tag table (top 10 by row count). Friendly fallback message when no shadow data exists yet.

### Decisions captured

- **Attempt 1 only.** Corrective-teach loops use Claude's diff feedback that Qwen wouldn't have access to in a counterfactual "Qwen alone" world; running shadow there would confound the signal. Net cost: shadow runs once per /code teach, not once per attempt.
- **30s timeout chosen carefully.** Below graduation's 60s ceiling so a stuck shadow never starves verification; above the typical Qwen warm call (~5–15s) so a normal completion has headroom. On a cold model load shadow can use most of its budget — fine, it's a single call.
- **Heuristic, not LLM-judged.** Spec was explicit. Hardware budget already tight; an LLM-judged scorer would add another GPU swap per /code AND make the score reproducibility-flaky. The Jaccard-blend signal is "do the plans broadly agree on what to touch and how" — exactly what we want, and it's cheap.
- **`by_archetype` groups by FIRST tag, not all tags.** Multi-tag patterns get under-counted (they contribute to one bucket only). Accepted because the per-tag rollup is informational, not quota-critical, and the alternative (UNNEST-style tag explosion) needs more SQL gymnastics for marginal value.
- **No partial index for `qwen_plan_agreement IS NOT NULL`** despite the Phase 14a/15a precedent. `/kb planning` runs interactively on demand, not nightly; even at the 50K cap the scan is fast with SQLite's filter pushdown. Hardware-discipline guideline says "partial indexes if you need any" — we don't.

### Tests

- `tests/test_phase15c_planning.py` — **19 new tests, ~3s** (no LLM, no GPU). ECC across:
  - Schema migration (3): fresh DB, pre-15c migration, idempotent re-init
  - KB round-trip (7): kwargs flow through, `_row_to_entry` shape, clamp on out-of-range agreement, None preserved (distinct from 0.0), `planning_stats` empty + aggregates + per-tag
  - Scoring (6): identical=1.0, unrelated<0.4, partial credit on same-files different-tools, empty-input=0.0, malformed=0.0, parser exception swallowed
  - Hook integration (3): source-level checks that the shadow call is wired after `_claude_pre_teach` with proper try/except, the kwargs reach `add_pattern`, the timeout constant equals 30s
- Targeted regression: `test_phase15b_provenance.py` (21/21), `test_knowledge_base.py` (6/6), `test_phase14a_graduation.py` (27/27), `test_phase15a_lifecycle.py` (33/33). **106/106 green.**

### Live-verification

The bot at PID 10656 has Phase 15a + 15b loaded (post-restart earlier this session) but NOT 15c. The 15c migration runs on the next restart; existing rows come up with both new columns NULL. The next /code attempt 1 will be the first to populate shadow data on its pattern row.

### Phase 15d candidates surfaced

- **Plan-agreement targets in PROFILE.yml.** Threshold-driven Telegram nudge: "Qwen agrees with Claude ≥0.8 on RegSalesManager problems — consider lowering KB injection cap to verify."
- **Per-attempt shadow.** Currently attempt 1 only; could record shadow on every attempt and graph divergence as Claude's recipe corrects.
- **Plan-agreement-aware fallback chain.** When a problem class hits ≥0.85 mean agreement over N samples, the complexity classifier could try Qwen-only (no Claude pre-teach) on first attempt for that class, falling back to Claude on review failure.
- **Multi-tag rollup.** Replace `first_tag` GROUP BY with a JSON-tag join so multi-archetype patterns count in every relevant bucket.

---

## Phase 15b — Write-origin provenance via ContextVar (2026-05-06)

**Goal:** Every persistent write to KB/memory tables now carries a `created_by_origin` label so future curation jobs can distinguish a `/code` user-driven teach from an adaptive-filter background write or a Phase 10 auto-extracted fact. Until now the column was missing entirely; the loop was creating data without provenance.

### What landed (commit `b0e329c`)

- **`core/write_origin.py`:** `contextvars.ContextVar[str]` with default `FOREGROUND`. API: `set_current_write_origin(origin) -> Token`, `reset_current_write_origin(token)`, `get_current_write_origin()`, `is_background()`. Constants `FOREGROUND` / `BACKGROUND` / `BACKGROUND_EXTRACTION`. Asyncio-safe by construction — each task forks its own copy of the var so concurrent foreground + background paths never cross-contaminate. Unknown setter values are accepted (no exception) but normalised to `FOREGROUND` on read; best-effort character keeps the provenance system from ever blocking a write.
- **Schema (idempotent ALTER, NOT NULL DEFAULT 'foreground' backfills every pre-15b row in one statement):**
  - `knowledge + created_by_origin TEXT NOT NULL DEFAULT 'foreground'`
  - `episodic_memory + created_by_origin TEXT NOT NULL DEFAULT 'foreground'`
  - `semantic_memory + created_by_origin TEXT NOT NULL DEFAULT 'foreground'`
- **Stamp sites** (writers read the contextvar at INSERT time):
  - `core/knowledge_base._add()`
  - `core/memory.store_episode()`
  - `core/memory.store_fact()` — INSERT only; UPSERT path leaves the original origin alone so the column records who FIRST wrote the key.
- **Wrap sites** (`interfaces/telegram_bot.py`):
  - `_maybe_auto_extract` wraps with `BACKGROUND_EXTRACTION` for the whole hook (any future fact write inside picks up the right origin).
  - `_maybe_run_adaptive_filter` wraps with `BACKGROUND`. The current adaptive_filter only writes to PROFILE.yml, not KB; the wrapper is foundational and harmless today, attribution-correct tomorrow if/when adaptive_filter gains a KB sink.
- **`/kb` default reply** now ends with a third section:
  ```
  ✍️ Provenance (Phase 15b):
  Origin breakdown: foreground=N, background=N, background_extraction=N
  ```
  Single GROUP BY query (`KnowledgeBase.origin_breakdown()`), not three counts.

### Decisions captured

- **UPSERT does NOT overwrite `created_by_origin`.** The auto-extracted "loc=Detroit" later re-asserted as user_explicit "loc=Dearborn" keeps origin='background_extraction'. Audit trail beats latest-writer-wins; curation workflows want to know which keys *started life as* machine inference.
- **Unknown origin values: accept on set, normalise on read.** Best-effort character. Never raise during a write because the contextvar carries an unrecognised label.
- **Wrap sites scope the entire hook, not just the SQL write.** If a brain call inside the hook later triggers its own write, that write also gets the right origin without touching the call graph.
- **Adaptive filter's BACKGROUND wrap is foundation, not active.** It currently writes only PROFILE.yml. Wrapped now so any future KB sink is attribution-correct from day one.

### Tests

- `tests/test_phase15b_provenance.py` — **21 new tests, ~2s.** ECC across:
  - ContextVar mechanics (6): default = FOREGROUND, set/reset round-trip incl. nested, unknown coerced on read, predicate correctness, asyncio task isolation (P05), constants match VALID_ORIGINS
  - Schema migration (5): fresh + pre-15b on KB and memory (both tables), idempotent re-init
  - Stamping (8): KB/episodic/semantic stamped under each origin, UPSERT preserves original origin, _row_to_entry round-trips, origin_breakdown aggregates
  - Wrap-site verification (2): source-level checks that `_maybe_auto_extract` + `_maybe_run_adaptive_filter` set/reset around their bodies
- Targeted regression: 87/87 green (15b + KB + Phase 14 + Phase 15a).

### Phase 15c candidates surfaced (now landed)

- Qwen-planning shadow tests — see Phase 15c above.

---

## Phase 15a — Archive-not-delete lifecycle (KB + memory) (2026-05-05)

**Goal:** Stop the self-learning loop from silently deleting its own learnings. Three subsystems were each running their own destructive prune (`knowledge_base.prune` lowest-usage, `memory.prune_episodes` lowest-relevance, `memory._prune_semantic` lowest-confidence) under tiny caps — the caps were defensive paranoia, not real constraints. Disk is plentiful, FTS5 on 50K rows is microseconds, KB context injection caps at 4000 chars per /code regardless. The caps lost data without protecting anything.

### What landed (commit `38a725e`)

- **Lifecycle states + pinned flag + archived_at on three tables.** `knowledge`, `episodic_memory`, `semantic_memory` each gain `state TEXT NOT NULL DEFAULT 'active'` (values: `active` | `stale` | `archived`), `pinned INTEGER NOT NULL DEFAULT 0`, `archived_at TEXT`. Idempotent ALTER (PRAGMA `table_info` check, not try-catch) — pre-Phase-15a DBs come out the other side with every existing row at `state='active' pinned=0`.
- **Two partial indexes per table.** `idx_<table>_pinned WHERE pinned = 1` + `idx_<table>_archived WHERE state = 'archived'`. Match the Phase-14a discipline: cover only the small selective subsets we query. Stay tiny even at the new 50K cap.
- **Caps raised.** `KNOWLEDGE_MAX_ENTRIES` 500→50000, `EPISODIC_MAX_PER_SCOPE` 200→100000, `SEMANTIC_MAX_ENTRIES` 500→50000. New constants `KB_STALE_AFTER_DAYS=30`, `KB_ARCHIVE_AFTER_DAYS=90` for the auto-transition walker.
- **Prune is now archive-not-delete on all three surfaces.** `knowledge_base.prune` counts non-archived rows only, archives the lowest-usage ones, skips pinned. Same shape on `memory.prune_episodes` (lowest-relevance) and `memory._prune_semantic` (lowest-confidence). All three return the count archived (matching the prior `deleted` return semantics so existing tests like `test_e_prune_drops_lowest_usage` keep passing).
- **`search()` defaults to hiding archived.** `KB.search(include_archived=False)` adds `AND k.state != 'archived'` to the JOIN. Same defaults applied to `MemoryManager.search_episodes`, `get_recent_episodes`, `search_facts`, `list_facts`. Direct lookups (`get_pattern(id)`, `get_fact(key)`) bypass the filter — needed by `/kb verify <id>` + `/kb restore <id>` paths.
- **Pin/unpin/restore APIs.** `KB.pin_pattern / unpin_pattern / restore_pattern`; mirrored on memory as `pin_episode / unpin_episode / restore_episode` (id-keyed) and `pin_fact / unpin_fact / restore_fact` (key-keyed, matching `delete_fact`'s natural identifier).
- **`auto_transition_lifecycle` walker.** Two sequential UPDATEs in one call — `active→stale` on `created_at < stale_cutoff AND <low-usage predicate> AND pinned=0`, then `stale→archived` on `created_at < archive_cutoff AND pinned=0`. Low-usage predicate per surface: `usage_count <= 1` (KB), `relevance_score <= 0.5` (episodic), `confidence < 0.6` (semantic — auto-extracted-and-never-confirmed).
- **`store_fact` upsert revives archived keys.** A user explicitly re-asserting a fact (`/remember location: Detroit`) on a key the auto-walker had archived flips it back to `state='active', archived_at=NULL` in the same UPDATE. Without this, the auto-walker could silently swallow a user write.
- **Telegram `/kb` extended.** New subcommands `/kb pin <id>`, `/kb unpin <id>`, `/kb restore <id>`. Help text updated. Same dispatch-by-substring pattern as the existing verify/retake/stale/reteach.
- **Scheduler wiring.** New `INTERNAL_HANDLERS["kb_lifecycle"]` calls `KB.auto_transition_lifecycle` + `MemoryManager.auto_transition_lifecycle` in one job. Self-registers on `core.internal_handlers` import (already triggered by `main.py`). Seeded in `_DEFAULT_SCHEDULED_JOBS` as a nightly `45 3 * * *` cron (EST) — lands 15 minutes after the existing 03:30 backup so the snapshot captures a consistent pre-walk state.

### Decisions captured

- **Archive-not-delete, never both directions on the same surface.** Considered keeping `prune` deletion-capable behind a flag for "true cleanup" mode. Rejected: the whole point of Phase 15a is that the self-learning loop never destroys its own outputs. True deletion belongs to a future, deliberately-named `/kb forget` command run by a human, not the auto-pruner.
- **`cleanup_low_quality_patterns` still uses DELETE (intentional).** That function is startup hygiene that removes pattern rows whose `solution_code` failed the write-time quality gate (i.e. write bugs / corrupted state, not aged-out learnings). The Phase 15a "no DELETE" rule is about the self-learning prune path; quality-gate cleanup is a separate concern. Documented in CLAUDE.md.
- **Pinned is orthogonal to state, not a fourth state.** Considered modeling pinned-vs-archived as an enum (`active|stale|archived|pinned`). Rejected because pinning is genuinely independent: a row can be pinned-and-active or pinned-and-archived (the latter via manual UPDATE). The auto-transitions filter on `pinned=0`; default search filters on `state != 'archived'`. Two clean predicates beat one tangled enum.
- **`unpin` does NOT touch state.** A common confusion: unpinning could either leave the row alone or "release it back to natural lifecycle." Picked the former. Unpinning a pinned-archived row leaves it archived (you can `/kb restore` separately). Unpinning a pinned-active row leaves it active and now eligible for auto-transition. Predictable + reversible.
- **Auto-transition uses deterministic age + usage, not LLM-judged staleness.** Hardware budget is already tight (1.7B brain + 3B coder + nomic-embed-text resident). A nightly LLM-judged sweep would compete for VRAM with whatever the user is running. "Old + rarely retrieved" is a strong heuristic; `search()` increments `usage_count` so patterns that earn their keep get protected for free.
- **Single sweep can collapse active→stale→archived for very old rows.** Initial test `test_l42` assumed two nightly ticks were needed to fully archive a 120-day-old row. Wrong: both UPDATEs run sequentially in one call and the second sees the row in its newly-flipped `'stale'` state. Test rewritten to assert end state (`archived`) without prescribing the path. This is intentional — a months-old, never-retrieved pattern shouldn't need to wait an extra night to finally archive.
- **Memory's `auto_transition_lifecycle` lives on `MemoryManager`, not as a separate scheduled handler.** Spec named only the KB handler. Avoided splitting into two scheduler entries by having `kb_lifecycle` call BOTH walkers in the same job (KB + memory). One nightly sweep, one log line, one place to rebalance the windows if the heuristics drift.

### Tests

- `tests/test_phase15a_lifecycle.py` — **33 new tests, 1m40s.** ECC across schema migration (fresh + pre-15a DB on both `knowledge.db` and `memory.db`, idempotent re-init), `search()` filter behaviour with/without `include_archived`, pin/unpin/restore round-trips on KB + episodic + semantic, prune-archives-instead-of-deletes + skips-pinned + doesn't-re-archive on all three surfaces, `auto_transition_lifecycle` gates on age + usage/relevance/confidence + pinned, `store_fact` upsert revives archived keys.
- Targeted regression: `test_phase14a_graduation.py` (27/27), `test_knowledge_base.py` (6/6), `test_pre15_hybrid_retrieval.py` (14/14). **80/80 green.**

### Live-verification on the running bot

The live bot (PID 10656 at the time of this commit) loads source files only at startup, so this commit doesn't affect it until next restart. No hot-reload risk; no DB schema mutation runs against the live `knowledge.db` / `memory.db` until the bot reboots — at which point the idempotent ALTERs run once and the lifecycle columns appear with every existing row at `state='active' pinned=0`.

### Phase 15b candidates surfaced

- **Async lifecycle walker.** Currently `kb_lifecycle` runs synchronously inside the scheduler tick (~10-50ms at 50K rows; cheap). Async wrapper exists; could fire-and-forget if the walker ever grows expensive.
- **`/memory pin/unpin/restore` Telegram surface.** Memory pin/unpin/restore are exposed programmatically but not yet via Telegram. Add when the user has a concrete need.
- **`/kb forget <id>` true-delete command.** The deliberately-human-triggered destructive operation Phase 15a left as future work.
- **`KB.list_archived(limit)` + `MemoryManager.list_archived_episodes/facts(limit)`.** Browse-archived API for a future `/kb archive` viewer.
- **Confidence-based revive on episodic, parallel to `store_fact`.** Currently only semantic fact upserts revive archived rows; if an agent re-stores an episode with the same scope+summary it just inserts a new row. Could add an upsert path if churn becomes an issue.

---

## Phase 14 — Self-Teaching Revival: KB Graduation Tests (closed 2026-05-05)

**Goal:** Stop *hoping* Qwen absorbed Claude's teaching. Empirically *measure* whether the KB enables solo reproduction. Owner's framing: "once Claude CLI disconnects, will Qwen even know what to do alone?"

### Outcome

The graduation system is live. Every successful agentic `/code` now triggers a transfer-verification step that mirrors the production pipeline — and updates KB columns (`solo_attempts`, `solo_passes`, `last_verified_at`, `needs_reteach`) with the result. After ≥3 attempts at <50% pass rate, a pattern auto-flags `needs_reteach` and future matches escalate straight to Claude instead of being used as a few-shot example.

| Sub-phase | Commit | Scope | Tests |
|---|---|---|---|
| 14a | `e25a8ee` | Initial graduation: text-gen Qwen + executor + KB schema columns + partial indexes (`needs_reteach=1`, `last_verified_at NOT NULL`) + `/kb` subcommands (`verify`, `retake`, `stale`, `reteach`) | 21 |
| 14a polish | `4f07b15` | `.gitignore` (fixes /commit pollution from staged .pyc files); 60s graduation timeout (was inheriting OllamaClient's 900s default; pattern #52 hit a 15-min hang in production) | +1 |
| 14b A+B | `fe8fbf1` | Solution A: replay STORED Claude recipe through Qwen stepfed + Claude review on a tree reset to `pattern.base_sha`. Solution B: store the actual diff text (capped 2000 chars) instead of the diff stat; new `base_sha` column. Mode dispatch: agentic for new patterns, text-gen fallback for old patterns | +5 |
| 14b bugfix | `530f8f0` | Live-caught false positive on pattern #54: scoped reset missed `interfaces/`, AND Claude review was reading stale leftover state from a prior /code attempt and falsely passing. Two-pronged fix: expand reset scope to include `interfaces`, replace Claude review verdict with deterministic stepfed-output check (completed_via_done + step_count > 0 + no step errors + syntax check) | +2 (test_g_42 rewrote, no net new) |
| 14 ECC | (this commit) | 10 fresh scenarios stressing agentic graduation: pass-rate threshold math, mode dispatch fallbacks, very-long recipe, stepfed crash + finally cleanup, error-in-middle-step, syntax-fail, stash failure, multi-step recipe | +10 |

**Tests total: 31 new** (test_phase14a_graduation.py + test_phase14b_scenarios.py). Full Phase 13 + 14 regression: **255/255 passed**.

### Decisions captured during execution

- **Graduation tests the production skill, not blank-page generation.** Phase 14a's text-gen approach was structurally unfair to Qwen 3B — production /code is "Claude writes recipe, Qwen executes tool calls", not "given problem, generate code from scratch." Phase 14b graduation REPLAYS the stored recipe through Qwen's stepfed agent on a clean tree, mirroring production exactly.
- **Verdict is deterministic, not LLM-judged.** Phase 14b initially used `_claude_review` for the verdict to mirror production exactly, but pattern #54 caught a live false positive: Claude reviewed leftover state from a prior /code attempt and falsely passed. Switched to a deterministic check from stepfed's own output (completed_via_done + steps > 0 + no step errors + syntax check). Side benefit: ~10s faster per graduation, and reproducible (same inputs = same outcome).
- **Reset scope must include every directory `/code` can write to.** Original `_git_reset_hard` was scoped to `core/skills/agents/tests` for safety (don't touch logs/ DBs during retries), but `interfaces/` was missing — `/code` freely edits `interfaces/telegram_bot.py` per `BOT_COMMIT_INCLUDE`. Production retries were also affected (attempt 2 inheriting attempt 1's interfaces/ edits). Fix expanded scope to include `interfaces`.
- **Tree-state discipline: stash → reset → run → reset → pop, all in `try/finally`.** Graduation is a probe, not an edit. Even if the LLM crashes mid-graduation, the user's working tree comes back exactly as it was.
- **needs_reteach is one-way.** Once a pattern is flagged (3+ attempts at <50%), more passes don't auto-unflip — only explicit `/kb retake <id>` does. Prevents a single fluke pass from rehabilitating a known-bad teacher.
- **Mode dispatch keeps old patterns useful.** Pre-Phase-14b patterns (no `base_sha`) automatically fall back to the text-gen path so they keep getting verified instead of permanently abstaining.
- **Hardware-aware indexing.** Two PARTIAL indexes (`WHERE needs_reteach = 1`, `WHERE last_verified_at IS NOT NULL`) so `/kb stale` and `/kb reteach` queries never scan the full knowledge table even at the 500-row cap.
- **Skip vs ship browser-snapshot graduation.** Considered adding a browser-rendered execution check (e.g., for UI changes), deferred. Lightweight stepfed + syntax check is good enough for the code-pattern domain we have.

### Live-test journey (2026-05-05 22:25-22:40)

Owner ran `/code i want to change my progress bar in telegram to use emojis` three times back-to-back to stress the new system:

| Run | Pattern | Production | Graduation | Notes |
|---|---|---|---|---|
| 1 | #54 (after 14b) | PASS attempt 1 (🟦) | **PASS 1/1** ❌ false positive | Caught the scoped-reset + stale-state-review bugs. Reset didn't undo `interfaces/`, so graduation stepfed errored on step 1; Claude reviewed leftover diff and falsely passed. |
| 2 | #55 (after bugfix `530f8f0`) | FAIL attempt 1 (909-char recipe + chat-loop fallback + `done=False`); PASS attempt 2 (🟩) | **PASS 1/1** ✅ true positive | First honest graduation: the bugfix's wider reset cleaned `interfaces/`, attempt 2's tight 618-char recipe parsed cleanly, stepfed completed cleanly, deterministic verdict said pass. |

The journey itself validated the system end-to-end. Production live-caught a real bug, fix shipped in ~5 minutes, next test confirmed the fix.

### Phase 15 candidates surfaced

- **Stash-on-dirty-tree was best-effort** — a real graduation could lose the user's uncommitted work if pop fails. Worth hardening (e.g., write the stash content to a sidecar file too as belt-and-suspenders).
- **Async graduation** — currently inline on the /code response path (~30s). Could fire-and-forget, send Telegram notification when done, free up the user faster. Polish, not blocking.
- **Variant generation** — when Claude teaches, also produce 2-3 variant problems testing the same skill. Pass-rate measured across original + variants. Stronger transfer signal than single-problem graduation.
- **Periodic re-validation** — scheduler runs `/kb verify --stale` weekly. Catches drift (model update, library version change). Background job; aligns with existing scheduler infra.
- **Server-parse improvements for stepfed** — graduation falls back to "ask Qwen to transcribe the recipe step into JSON" when the deterministic parser fails. Each fallback adds ~5s. Better parser handling of Claude's exact recipe format would speed graduation toward instant.

---

## Phase 13 — Speed + Accuracy Pass (kicked off 2026-05-05)

**Goal:** Cut `/jobsearch` wall-clock time AND raise top-N precision by closing every actionable gap from the Phase 12.5 production runs. ONE phase, multiple coordinated batches.

**Inspiration sources:** career-ops (open-source, MIT) + owner's own prior Job Tracker Lite project (33-source scraper, ~560 deduped postings, MIT). Cite both where features map.

**Owner-confirmed scope (from kickoff Q&A):**

| # | Item | Decision |
|---|---|---|
| 1 | LLM batching in extract+score | ✅ in (~6 min off a 12-min run) |
| 2 | **Dedup at SCRAPE time** — drop URLs whose `url_hash` is in `applications` | ✅ in (option A: aggressive) |
| 3 | Pre-extract title-filter pass + region preference | ✅ in (BOTH: keyword boost AND state-list whitelist) |
| 4 | Q4_K_S quantize qwen-coder | ⚠️ risk-analyze first, then ship if low-risk |
| 5 | Force-gate Chicago leak (location_type=remote override) | ❌ skip — current behavior decent |
| 6 | Per-archetype dimension weights from PROFILE.yml | ✅ in |
| 7 | `/jobs` Telegram viewer (list/apply/interview/rejected/note/status) | ✅ in — copy from career-ops `modes/tracker.md` + JTL dashboard |
| 8 | `--rescan` smarter (a: adaptive title filter, b: brain query expansion) | ✅ in (BOTH a + b) |
| 9 | Browser-snapshot legitimacy signals | ⚠️ risk-gate — only if low-cost; otherwise Phase 14 |
| 10 | Canadian (pgeocode 'ca') | ❌ skip — US only |
| 11 | `--avoid` flag verification | ✅ + `Director of Sales Operations` removed from PROFILE.target_roles.primary per owner (director-class roles flooded the maybe-band) |

**Batch plan:**
1. Batch 1 — Dedup + region preference + verify --avoid
2. Batch 2 — Per-archetype weights + Q4 risk analysis
3. Batch 3 — LLM batching (the headline perf win)
4. Batch 4 — `/jobs` Telegram viewer
5. Batch 5 — Smarter rescan (adaptive filter + query expansion)
6. Batch 6 — Browser-snapshot risk-gate (defer if heavy)
7. Final — docs + report

Each batch: stop, show test results, then proceed. ECC tests, mocked LLM, no network in unit tests.

### Outcome (closed 2026-05-05)

All six batches shipped. **101 new tests, 212/212 targeted regression green.** Combined wall-clock improvement on a 40-posting `/jobsearch`: ~7 min → ~3.5 min (LLM batching dominant), with measurable accuracy lift from per-archetype weights + region nudges + scrape-time dedup.

| Batch | Commit | Scope | Tests |
|---|---|---|---|
| 1 | `eb37b3f` | Scrape-time dedup against `applications` table; region keyword boost/avoid in `north_star`; accepted-states whitelist as `cultural_signals` penalty; regression test for `--avoid` flag; removed Director of Sales Operations from PROFILE primary | 14 |
| 2 | `543e992` | Per-archetype dimension weights (`ARCHETYPE_DEFAULT_WEIGHTS` for the 6 sales archetypes; PROFILE override path with normalize-to-sum=1.0); Q4 quantize risk analysis (we're already on Q4_K_M — no change recommended) | 26 |
| 3 | `cdbdcee` | LLM batching: `job_score` now `accepts_list=True`; chunks N postings into `JOB_SCORE_BATCH_SIZE` (default 5); per-batch LLM call with three-tier failure recovery (whole-batch parse failure / partial item failure / `LLMError` → per-item fallback); commute-gated postings short-circuit before batch | 11 |
| 4 | `985bd61` | `/jobs` Telegram viewer: list / filter-by-state / drill-by-id / state-transition; Spanish state aliases; recommendation-band emoji map; new `core.database.get_application(id)` helper | 12 |
| 5 | `bc54c0d` | Smarter rescan: `core/query_expansion.py` (deterministic abbrev map: RSM ↔ Regional Sales Manager etc., capped at 3 variants); `core/adaptive_filter.py` (sample dropped titles, brain extracts negatives, validate against positive/seniority collisions, atomic YAML append); sidecar `logs/last_scrape_stats.json` for post-pipeline access without breaking the agent's single-list output contract | 20 |
| 6 | `a362766` | Lightweight legitimacy signals (no browser): `core/legitimacy.py` with apply-URL classification (ATS host whitelist + foreign TLD heuristic) and repost-cadence detection (same company, fuzzy title match, 90-day window in `applications`); new `core.database.find_recent_company_postings()` data source | 18 |

**Decisions captured during execution:**
- **Watcher mystery (Batch 2)**: long-running pytest in the background was clobbering the working tree mid-edit (likely `test_revert_ecc.py` doing real git ops). Workaround: only run targeted batch tests during edits, defer full regression to phase end. Documented mid-conversation; no code change needed.
- **JobScrapeOutput stays single-field** (Batch 5): the agent's `output_is_list=True` unwrap requires exactly one list-typed field. New telemetry (dropped titles, query variants) goes to a sidecar JSON file the bot reads post-pipeline instead.
- **No browser snapshot in Batch 6**: would need `playwright` (heavy dep). Per owner's "if low-risk don't add too much work" guidance, shipped the high-signal cheap checks (URL host classification + repost cadence from existing applications table) which cover the same ghost-job detection use case. Browser snapshot deferred to Phase 14 if ever.
- **Adaptive filter is auto-apply** (Batch 5): per owner request. Safety bounds: max 3 new negatives per run, won't add anything that overlaps a positive/seniority_boost keyword (would block target roles). Every change announced in Telegram so user can see what was added and edit via `/profile` if needed.
- **Query expansion is deterministic, not brain-based** (Batch 5): hardcoded abbrev map avoids ~5s GPU model swap (qwen3-brain ↔ qwen2.5:3b) for jargon coverage that doesn't actually need an LLM. Only known sales titles in the map; new jargon → add a row.

**Pre-existing test failures unchanged** (not caused by this phase): 10× `test_revert_ecc.py::test_revert_ecc_*` (assertion strings don't match current `/revert` handler output), 1× `test_phase10_ecc_az::test_o_job_pipeline_fanout` (test interference flake — passes in isolation, fails under specific full-suite ordering), 1× `test_l_qwen_garbage_falls_through_to_claude` (Qwen LLM behavior flake). Worth a Phase 14 cleanup pass.

**Phase 14 candidates surfaced during this work:**
- Application-feedback retraining loop (mine state-transition mismatches → adjust archetype weights / negatives)
- Cross-conversation persona drift detection (Aftermath OS contradiction-tracer pattern — flagged in their repo scan)
- `/code` teaching loop revival (KB has been dormant since job-pipeline work)
- Hypothesis property-based tests for `core/geo.py` and the scoring math

**This entry is now closed. Below this section is each batch's deltas as it lands.** ← (kept for historical accuracy; the per-batch detail is in the table above and in commit messages.)

---

## Phase 12.5 — Geographic Hardening + LinkedIn Fetch (2026-05-05)

**Built:** `core/geo.py`, hard commute gate in `skills/job_score.py`, scrape-radius narrowing + `linkedin_fetch_description` in `skills/job_scrape.py`. New dep: `pgeocode>=0.5,<1.0`.

**Why:** Audit of the first real `/jobsearch` run (40 postings, 2 worth_applying, 13 maybe, 25 skip) showed 7 of the top-10 were out-of-commute deal-breakers (Cincinnati, Chicago, Toronto, Mississauga, Indiana, Hudsonville, Milwaukee) but landed in the maybe band because (a) the scorer treated `workplace_pref='all'` as "no penalty" and (b) the 20-mile-from-48125 cap was background context, not a hard rule. Plus 8 of 13 maybe-band postings were LinkedIn with truncated 150-char bodies that defaulted scorer dimensions to neutral-3.

### What landed

- **`core/geo.py`** — pgeocode-backed offline lookups (no network, no API key). `zip_to_latlong` (5-digit + extended-zip), `parse_city_state` (handles `City, ST`, `City, ST 12345`, `City, ST, US` shapes; returns None on Canadian / state-only / unparseable), `city_state_to_latlong` (median across all postal codes in a city — crude but adequate for a 20-mi gate), `haversine_miles`, `distance_miles_from_zip` (tries embedded zip first, falls back to City+ST). `_looks_foreign` uses **word-boundary regex** (real bug found by test — substring `india` matched `Indiana`). `outside_commute(profile, location_type, location_text)` is the policy.
- **`skills/job_score.py`** — pre-LLM gate via `outside_commute`. If gated, `_commute_skipped()` returns `score=2.0, recommendation='skip', red_flags=1`, all other dims=2, hard reason in `reasons[]`. **LLM not called.** SYSTEM_PROMPT also gained an explicit HARD COMMUTE RULE + title-suffix tells (East/West/North/South/-Region) as defense-in-depth.
- **`skills/job_scrape.py`** — when candidate cannot relocate + on-site/hybrid + PROFILE has zip+miles, jobspy `distance` overrides 250 → `onsite_max_miles`. Honors explicit `--distance N`. `linkedin_fetch_description=True` passed unconditionally so LinkedIn postings come back with full bodies.

### Verified against the first run's actual data

| Location | mi from 48125 | Old (LLM only) | New (pre-LLM gate) |
|---|---|---|---|
| Detroit, MI | 9.6 | 4.08 ✓ | falls through to LLM |
| Cincinnati, OH | 225 | 3.93 maybe | **2.0 skip (gated)** |
| Chicago, IL | 229 | 3.93 maybe | **2.0 skip (gated)** |
| Hudsonville, MI | 139 | 3.75 maybe | **2.0 skip (gated)** |
| Milwaukee, WI | 243 | 3.75 maybe | **2.0 skip (gated)** |
| Toronto, Canada | foreign | 3.93 maybe | **2.0 skip (gated)** |
| Mississauga, Canada | foreign | 3.93 maybe | **2.0 skip (gated)** |
| Indiana, US (state-only) | unresolvable | 3.93 maybe | falls through to LLM (conservative) |

### Decisions captured

- **Conservative on geocode uncertainty.** When a location can't be resolved (state-only, "Remote", "(unknown)", company-name leak), the gate does NOT fire. LLM still scores. Better to over-LLM than to drop a real role on a parsing miss.
- **Word-boundary, not substring.** Regex `\b{tok}\b` instead of `tok in low`. Caught by `test_looks_foreign_uses_word_boundary_not_substring` — `Indianapolis, IN, US` would have been flagged as India otherwise.
- **Set iteration non-determinism is real.** First test draft asserted `_looks_foreign("Toronto, Ontario, Canada") == "canada"`. Both `ontario` and `canada` are in `_FOREIGN_TOKENS`; whichever python iterates first wins. Tests now accept either.
- **Score floor 2.0, not 1.0.** Floor stays for actual scoring failures (`_floor_scored`). The commute-gate path uses 2.0 to signal "not LLM-scored, hard deal-breaker" — distinct from "LLM crashed."
- **LinkedIn fetch added unconditionally.** Per-posting +1s scrape cost is acceptable; the alternative (post-process LinkedIn body via separate LLM call) would cost ~10s/posting. Same trade career-ops makes.
- **Scrape distance respects explicit `--distance N`.** Only overrides when caller is using the default. Power users keep control.
- **HARD COMMUTE RULE in the prompt is defense-in-depth.** The pre-LLM gate is the real enforcer; the prompt rule covers the case where the gate can't fire (e.g., unresolvable location but the LLM picks up "Cincinnati, OH" from the description body).

### Tests

- `tests/test_geo.py` — 30 tests, 4s. ECC: 8 zip lookup (incl. extended-zip + 6 garbage rejections), 12 city/state parse cases, 2 haversine sanity, 7 distance round-trips with expected ranges + 6 unresolvable returns None, 3 `_looks_foreign` (incl. the word-boundary regression), 8 `outside_commute` policy (distant US gated, local US passes, remote bypasses, foreign gated, willing-to-relocate disables, no-zip disables, unresolvable falls through, hybrid like onsite), 2 job_score integration (gate skips LLM via `fail_llm` stub, local onsite reaches LLM via canned response).

**Phase 12.5 total: 30 new tests, all passing. Combined regression (geo + archetypes + profile + applications + pipelines + telegram + skills): 143/143 + 1 deselected pre-existing flake.**

### Open follow-ups for Phase 13

**Performance**
- 🥇 **Batched LLM calls in `job_extract` + `job_score`** — pack 5 postings per Ollama request, parse JSON array. Realistic ~2x speedup on the LLM-bound phases (reduces a 12-min run to ~5-6 min). Per-call overhead saved + amortized prompt processing; generation time still scales with output tokens. Risk: malformed JSON output kills the whole batch — fallback path needed (retry at smaller batch, single-call mode on parse fail).
- URL-hash extraction cache: skip `job_extract` LLM call when the posting's `url_hash` already has a cached `JobExtraction` (re-runs of the same query become near-free).
- Quantize `qwen-coder` to Q4_K_S: ~20% faster, drops VRAM 2.8 GB → ~1.8 GB. Quality loss small for structured extraction.
- Pre-extract title-filter pass: if the title alone tells us archetype + reject reason, skip the LLM call entirely.

**Quality / observability**
- Per-archetype dimension weights from PROFILE.yml (Phase 12 deferred this).
- `/jobs` Telegram viewer for the applications table (state transitions from chat).
- `/jobsearch --rescan` flag (force re-evaluate even when in apps table).
- Tie-breaking in archetype detection (longest-keyword-wins) — Director-of-sales etc. landed on Unknown last run; user confirmed those are out of scope so this is lower priority.
- Browser-snapshot legitimacy signals (apply-button state, posting age, repost cadence from `applications.history`).
- Canadian city support (pgeocode 'ca' dataset) so we can compute distance instead of just flagging "foreign."

---

## Phase 12 — Job Pipeline Rebuild: Profile + Rubric + Tracker (2026-05-05)

**Built:** `core/job_profile.py`, `core/archetypes.py`, `workspace/persona/PROFILE.example.yml`, new `applications` table in `sentinel.db`. Rewrote `skills/job_score.py` end-to-end (HARD CUTOVER from 0-1 score to 1-5 weighted rubric). Threaded `url + workplace_pref` carry-through through `skills/job_scrape.py` → `skills/job_extract.py` → `skills/job_score.py` → `skills/job_report.py` → applications table. New `/profile` Telegram subcommands. New `--workplace` and `--avoid` flags on `/jobsearch`. Top-3 broadcast appended to every `/jobsearch` reply.

**Phase 12 is the FRAMEWORK pass. Phase 13 is the HARDENING pass.** Items deliberately deferred to 13:
- Browser-snapshot legitimacy signals (apply-button state, posting age, repost cadence detection from applications table).
- Per-archetype custom dimension weights from PROFILE.yml (currently a single weight set for all archetypes).
- Tie-breaking in archetype detection beyond first-registered-wins.
- `--rescan` flag to force re-evaluation of already-tracked URLs.
- `/jobs` Telegram viewer for the applications table (state transitions from chat).
- Hybrid post-filter is best-effort only (jobspy can't filter server-side; the scorer penalizes mismatches but doesn't drop).

### Batch 1 — PROFILE + filters (commit `222696b`)

- **`workspace/persona/PROFILE.example.yml`** — sales-flavored archetype preset (Regional Sales Manager, Territory Sales Manager, AE, SDR, RevOps, CSM). Each marked `primary` / `secondary` / `adjacent` / `skip`. Hand-edit-friendly with comments.
- **`core/job_profile.py`** — Pydantic `Profile` model + `load_profile()` that re-reads PROFILE.yml on every call (so `/profile edit` takes effect immediately). Returns defaults on missing/malformed/invalid — never raises. `title_passes()` is the pure pre-LLM filter (positive must match, negative rejects, avoid hits in title OR company reject). `has_seniority_boost()` for the rubric's north_star nudge.
- **`/profile` Telegram subcommands** — `init` (copy from example), `show` (parsed pretty-print), `set <dotted.path> <value>` (in-chat edit with type coercion + re-validation), `edit` (print full file + path for manual disk edit). Auth-gated.
- **`skills/job_scrape.py`** — `JobScrapeInput` gains `avoid` (comma-string → list) and `workplace` (on-site|hybrid|remote|all with aliases onsite/in-person/wfh/any). `_workplace_to_is_remote` translates to jobspy's `is_remote` tri-state. Pre-LLM title-filter pre-pass uses `load_profile()` + `title_passes()`. Drop counts logged separately as `dropped_title` vs `dropped_avoid`. Skill version 1.0.0 → 1.1.0.
- **20 tests** in `tests/test_job_profile.py`. ECC: 5 loader cases, 7 title_passes cases, 4 JobScrapeInput field validators, 1 `_workplace_to_is_remote`, 2 `_row_to_posting`.

### Batch 2 — 5-dim rubric + archetypes + legitimacy (commit `2ad52c4`)

- **`core/archetypes.py`** — `DEFAULT_ARCHETYPES` (the same 6 from PROFILE.example.yml), `DIMENSION_WEIGHTS` (cv_match=0.30, north_star=0.25, comp=0.20, cultural_signals=0.15, red_flags=0.10; sums to 1.0). `detect_archetype()` is deterministic substring count with title weighted 3:1 over description. `weighted_score()` falls back missing dims to neutral-3 + clamps to [1,5]. `recommendation_band()` thresholds at 4.5/4.0/3.5. `legitimacy_tier()` maps signal count to high/caution/suspicious. `_all_archetypes()` merges PROFILE archetypes over defaults; `fit='skip'` suppresses the matching default entirely (real bug found via test).
- **`skills/job_score.py` rewritten** — new `ScoredPosting` schema: `score: 1.0-5.0`, `dimensions: dict`, `archetype: str`, `recommendation: str` (band), `legitimacy: Legitimacy(tier, signals)`, `url: str`. Pydantic `mode='before'` validators clamp every value, coerce garbage to neutral-3, cap reasons at 5, snap invalid bands to 'skip'. Skill version 1.0.0 → 2.0.0. New `SYSTEM_PROMPT` asks the LLM for `{dimensions, reasons, legitimacy_signals}` and explains each dimension; legitimacy is rated SEPARATELY from score. Title-keyword 'senior' boost adds +0.5 to north_star.
- **`skills/job_report.py` updated** — new CSV columns (recommendation, archetype, url, legitimacy_tier). `_build_summary_md()` reports band counts with emoji and links apply URLs in the top-10. New `top_n_for_telegram()` helper — filters skip-band, friendly fallback when nothing eligible. `JobReportOutput` gains apply_now_count / worth_applying_count / maybe_count / skip_count + top_n_telegram string.
- **44 tests** in `tests/test_archetypes.py`. ECC: 12 detect_archetype (parametrized for all 6 archetypes + unknown + title-3x weighting + profile override + fit='skip' suppression), 5 weighted_score, 8 recommendation_band, 5 legitimacy_tier, 6 ScoredPosting validators, 4 job_report rendering.

### Batch 3 — applications table + dedup + top-3 Telegram (commit pending)

- **`sentinel.db::applications` table** — keyed by `url_hash` (sha256 of canonicalized URL — utm_*/gclid/fbclid stripped, lowercased). Columns: title, company, location, archetype, score, recommendation, state (canonical 7-state lifecycle), history (JSON transitions), first_seen_at, last_seen_at, applied_at. Indexed on (state, last_seen_at).
- **DB helpers** — `application_exists`, `upsert_application` (idempotent + never regresses advanced state), `transition_application` (appends history + sets applied_at on first 'applied'), `list_applications`, `_normalize_state` (canonical states + Spanish aliases).
- **URL plumbing** — extended `JobExtractInput` and `JobExtraction` with `url: str = ""`, `description: str = ""`, `workplace_pref: str = "all"` so the URL survives the fan-out hop and the apps writer has something to dedup against. `job_extract.execute` echoes them through.
- **`job_report` writes applications** — best-effort upsert for each scored posting with a non-empty URL. Failures logged but never break the report write.
- **`handle_jobsearch` broadcasts top-3** — after the standard brain-summary message, parses `result.top_n_telegram` and sends it as a SECOND message with apply links. Per owner's option-A choice (always broadcast on every `/jobsearch`).
- **Custom scenario test** — `tests/test_jobsearch_scenario.py::test_regional_sales_manager_scenario`. Real Indeed scrape for "Regional Sales Manager" in Detroit MI within 168h, 5 results. LLM steps mocked. Verifies: real postings come back, title filter keeps matches, fan-out fires extract+score per posting, CSV+MD written, applications table populated with real URLs, top_n_telegram non-empty with apply links. Marked `slow + requires_network`; auto-skips when scrape returns 0 (rate limit / outage). Owner's canonical query.
- **16 tests** in `tests/test_applications.py`. ECC: URL canonicalization (3 cases), upsert (4 cases including state-keep on rescrape), transitions (4 cases including history append + alias normalization), list filtering (3 cases), report→apps integration (2 cases including empty-URL skip).

### Decisions captured during this phase

- **HARD CUTOVER on the score schema, not gradual.** Owner explicitly declined backwards-compat. The old `score: 0-1, recommend: bool` is gone; every caller updated. Migration cost was small (3 callers: job_report, test_pipelines, test_o) — would have grown if we delayed.
- **PROFILE.yml is single source of truth, but title_filter accepts per-call overrides.** `--avoid "x,y"` merges with PROFILE.avoid_companies. Owner's preference: blanket avoid in PROFILE for permanent dislikes, per-call `--avoid` for "today I don't want recruiters from X".
- **Internal scoring weights are uniform across archetypes.** Career-ops has per-archetype weight tweaks (FDE prioritizes velocity, SA prioritizes architecture, etc). Skipped that for Phase 12 — the variance is small relative to the LLM's per-dimension scoring noise. Phase 13 can add it.
- **Workplace preference is best-effort, not authoritative.** jobspy supports `is_remote=True/False` server-side. `hybrid` has no server-side filter. Decision: pass `is_remote` for remote/on-site, leave hybrid + all as None, let the scorer penalize mismatches via the carry-through. Better than dropping postings on incomplete extraction.
- **Bot doesn't know the scoring schema.** `top_n_telegram` is a string built by the SKILL and stashed on the output model. The bot just `result.get("top_n_telegram")` and forwards. Decouples Telegram from rubric changes.
- **URL canonicalization is conservative.** Lowercase + strip whitespace + drop utm_*/gclid/fbclid. Don't strip path case (some servers care). Don't normalize trailing slash. Aggressive canonicalization would over-merge distinct postings.
- **Don't dedup at scrape time.** Considered skipping URLs already in `applications`. Rejected because re-scoring on profile changes is useful, and the title filter already drops the cheap-to-reject cases. Apps table is the dedup layer for downstream consumers, not for cost reduction.

### Tests

- `tests/test_job_profile.py` — 20.
- `tests/test_archetypes.py` — 44.
- `tests/test_applications.py` — 16.
- `tests/test_jobsearch_scenario.py` — 1 (real Indeed integration).
- Migrations of existing `tests/test_pipelines.py` (4) for the new schema.

**Phase 12 total: 81 new tests + 4 migrated, all passing.** 104/104 across the Phase-12-and-neighbors regression sweep (job_profile + archetypes + applications + jobsearch_scenario + pipelines + telegram + skills + extractor).

**Open follow-ups for Phase 13:**
- Posting legitimacy currently has only LLM-judged JD-text signals. Phase 13 will add browser-snapshot signals (apply-button state, posting age) and repost-cadence detection (query applications.history for url_hash repeats).
- Per-archetype dimension weights from PROFILE.yml.
- `/jobs` Telegram viewer (list/transition/notes).
- `/jobsearch --rescan` flag to force re-evaluation.
- Tie-breaking in archetype detection (longest-keyword-wins).
- Pre-existing failures (test_cache, test_code_assist::test_g/i/l, test_pipeline_phase7::test_m/n) untouched per owner instruction.

---

## Phase 11 — Scheduler, Health, Hardening (2026-05-05)

**Built:** `core/scheduler.py`, `core/health.py`, `core/internal_handlers.py`, `scripts/install_supervisor.ps1`. New schemas: `scheduled_jobs`, `job_runs` (in `sentinel.db`). Telegram surface: `/restart`, `/commit`, `/revert`, `/schedule add|list|pause|resume|delete|runs`, `/dashboard`, `/curate review`, plus `set_my_commands` BotFather menu sync.

### Scheduler (Batch 1)

- Three schedule types: `once` (ISO datetime), `interval` (`30m` / `2h` / `45s` / `1d`), `cron` (5-field via `croniter`).
- EST-anchored: cron + active-hours windows interpreted in `America/New_York` (`SCHEDULER_TIMEZONE`); `next_run_at` always stored UTC; `/schedule list` renders local. New dep `tzdata` so Windows `zoneinfo` works.
- **Skip-if-running** by checking `job_runs` for an existing `status='running'` row before firing — prevents pile-ups on the 4GB GPU. Worker's DB GPU lock is the real serializer; `SCHEDULER_MAX_CONCURRENT=1` is a sanity belt.
- **Active hours window** with midnight-wrap support (`22:00–02:00` covers nights).
- **One-shot semantics:** `delete_after_run=1` removes the row on success; default just sets `enabled=0`.
- **Internal-handler dispatch** for `/internal_<name>` commands bypasses the router and calls async Python in `INTERNAL_HANDLERS` directly. Saves ~1–2s of queue round-trip on pure SQLite/disk maintenance.
- **Startup spreading:** `spread_overdue_jobs()` redistributes overdue jobs across `SCHEDULER_STARTUP_SPREAD_SECONDS` (default 300) so a long downtime doesn't trigger a thundering herd on the GPU.

### Health + dashboard (Batch 2)

- `aiohttp` server bound to `127.0.0.1:18700/health` (never external). JSON snapshot: status/uptime/queue/gpu/models/scheduler/memory/kb/disk/logs/telegram.
- **GPU lock observability** is new — `gpu.lock_holder` + `lock_age_seconds` from the `locks` table. Previously the locks table was only queryable via acquire/release/recover_stale; now a stuck agentic `/code` is diagnosable at a glance.
- **Model availability is cached**, not probed per request. `HealthMonitor.set_model_availability()` is wired by `main.py` from the startup `MODEL_REGISTRY.check_availability()` result.
- **`/dashboard`** Telegram command renders the same snapshot as a chat-readable summary so the user can check status from their phone without the HTTP port.
- **nvidia-smi probe** is tri-state: missing → one-time WARNING, then `vram_used_mb=None` forever. No per-call retries.
- **Polling watchdog REMOVED** per owner instruction. The Phase 11 plan originally included an auto-restart-on-stale, but `/restart` (committed earlier same day) made it redundant; bot stop+start from inside itself was fragile on Windows.

### Hardening + main.py wiring (Batch 3)

- **Log rotation** via custom `_WindowsTolerantRotatingHandler` (10 MB / 5 backups). On Windows, rolling renames a held file with `PermissionError`/WinError 32; the subclass swallows that, re-opens the stream, and lets logs continue appending. Rotation retries on next overflow.
- **Internal handlers:**
  - `wal_checkpoint` — `PRAGMA wal_checkpoint(TRUNCATE)` on `sentinel.db`, `memory.db`, `knowledge.db`.
  - `backup` — online `sqlite3.Connection.backup()` of all 3 DBs + 4 persona files into `BACKUP_DIR/YYYY-MM-DD/`. Retention prunes folders older than `BACKUP_KEEP_DAYS` (default 7); preserves non-date directory names untouched.
  - `resource_check` — disk-free + log-dir size + best-effort VRAM; `_alert()` callback fires on threshold violations.
  - `file_guard_check` — replaces in-bot heartbeat (additive; bot's `_heartbeat_task` left in place pending verification).
  - `curate` — auto-approves on the scheduler path; interactive `/curate` Telegram command unchanged.
- **`main.py` wiring:** instantiate `HealthMonitor` + `Scheduler` after registries, before `bot.start()`. Wire `health.scheduler`, `health.bot`, `bot.health_monitor`, and `internal_handlers._set_alert_callback(bot.send_alert)`. `await scheduler.spread_overdue_jobs()` then `start_health_server()`. Seed three idempotent default jobs on first boot: hourly WAL checkpoint, 10-min resource probe, nightly 03:30 EST backup. `PRAGMA optimize` on all 3 DBs in shutdown finally block.
- **Supervisor:** `scripts/install_supervisor.ps1` registers a user-level Windows Task Scheduler entry at-logon with restart-on-failure 3× / 1-min interval. No admin needed. Idempotent (unregisters first if present).

### Decisions captured during this phase

- **Scheduler defers to the worker's DB GPU lock.** Rejected the original plan's in-process `max_concurrent_runs` counter — duplicates the lock and creates a second source of truth.
- **`route_fn` injected, not `BrainRouter`.** The brain is for natural-language → command parsing; the scheduler always has a structured command already.
- **Don't auto-rewrite the bot's existing background tasks.** The bot's Phase 10 `_heartbeat_task` and `_curation_scheduler_task` stayed in place. Scheduler-driven `/internal_file_guard_check` and `/internal_curate` are *available* but not seeded by default — user can swap once the scheduler has been verified live for a few days.
- **`_WindowsTolerantRotatingHandler`.** Filed under "things you only learn the hard way": Windows `RotatingFileHandler` rollover does an `os.rename` of the current file, which fails if any other process has it open. The bot logs into the same file from PID A while pytest imports it in PID B → rotation explodes. Subclass tolerates the failure; logs keep landing on disk.
- **Test allowlists relaxed (Phase 7 follow-up).** Pre-existing test_cache / test_code_assist / test_pipeline_phase7 failures were left untouched per owner instruction during Phase 11; only NEW failures introduced by this phase get fixed.

### Tests

- `tests/test_scheduler.py` — 27 tests, 3.75s. ECC: cron parsing (weekday + weekend skip), interval (4 unit cases + invalid), once (aware + naive ISO), 3 active-hours cases including midnight wrap, execution+route, skip-if-running, active-hours blocking, one-shot delete + just-disable, 7 `/schedule add` parser variants, pause, resume, internal-handler bypass, route failure → alert, isolated-mode silence.
- `tests/test_health.py` — 9 tests, 3.6s. K (3 overdue → spread), L (0 overdue noop), M (snapshot keys/types), N (lock holder + age 80–120s for 90s seed), N2 (unheld lock returns None), O (model availability uses cache, network probe asserted-not-called), P (dashboard non-empty), P2 (held lock shows holder not "(free)"), bonus end-to-end aiohttp bind on ephemeral port.
- `tests/test_hardening.py` — 9 tests, 3.8s. Q (rotation creates rolled sibling), R (wal_checkpoint runs on all 3 configured DBs), S (low disk → alert), T (backup writes 3 SQLite + 4 persona), U (retention prunes folders > keep_days, preserves bogus non-date), V (E2E job → bot alert), W (shutdown_event exits scheduler_loop within 2s), X / X2 (file_guard / curate degrade cleanly when component absent).
- `tests/test_curate_review.py` — 6 tests, 3s. Renderer with all 3 sections, no_changes path, only-additions, empty-pending message, two-pending list, race-tolerance (token disappears mid-iteration).

**Phase 11 total: 51 new tests, all passing.**

**Open follow-ups (not blockers):**
- Bot's `_heartbeat_task` + `_curation_scheduler_task` are now redundant with `/internal_file_guard_check` + `/internal_curate`. Delete + seed scheduled equivalents in a follow-up after a few days of scheduler verification.
- No historical record of past curation outcomes — `/curate review` only shows in-memory pending. Scheduled `/internal_curate` auto-applies and leaves no audit trail beyond the JSONL log line. Could add a `curation_runs` table if desired.
- Pre-existing test failures (test_cache, test_code_assist::test_g/i/l, test_pipeline_phase7::test_m/n) untouched per owner instruction.

---

## Phase 9 — VERIFIED COMPLETE (2026-05-04)

20-test ECC of /code spanning trivial → cross-file:

| # | Test class | Result |
|---|-----------|--------|
| 1-5 | trivial single-function (add, reverse, square, celsius_to_fahrenheit, is_even) | 5/5 PASS attempt 1 |
| 6-10 | easy single-function (factorial, merge_dicts, count_vowels, min_max, slugify) | 5/5 PASS attempt 1 |
| 11-15 | medium (chunked generator, flatten recursive, memoize, partition, retry sync+async) | 5/5 PASS attempt 1 |
| 16-20 | harder (Stack class, multiply, subtract, docstring, exports cross-file, divide) | 5/5 PASS attempt 1 |

**100% pass rate. ~36s avg per test. ~$1.50 total Claude cost.**

Reviewer-Claude verdict reasoning consistently quoted line numbers, used Grep, and ran live `python -c "import ..."` Bash checks. Qwen frequently exceeded spec (added ValueErrors, IndexErrors, dunder methods, sync+async support).

KB grew from ~13 to ~33 patterns; all portable (PROJECT_ROOT-relative paths) and survive restarts (quality gate accepts diff stats).

### Decisions captured during this phase

- **Server-side step parser** (`_parse_step_text_to_tool_call`) bypasses Qwen for the recipe→tool-call transcription. The recipe IS the answer; using Qwen as a transcription layer just introduced JSON over-escape bugs. Falls back to Qwen when parser can't recognize the step shape.
- **Reviewer with tools** (Read+Grep+Glob+Bash) plus a server-side `py_compile` gate. Verdicts now mean "I read the file and it compiles" instead of "the diff looks right."
- **Recipe extractor before truncation** (`_extract_recipe_steps_from_text`) pulls STEP blocks out of the full Claude response before applying any length cap. Otherwise tools-enabled Claude can write 25k char narration and the chop loses all the actionable content.
- **`run_bash` accepts both `command` and `cmd`** as a defensive fallback against prompt drift.
- **KB quality gate accepts diff-stat patterns** (not just Python source). Otherwise every successful agentic /code pattern would get wiped on next bot startup by `cleanup_low_quality_patterns`.

### Known minor caveats (not blockers)

- KB context block can hit the 3000-char cap when many same-tag patterns match; cap is hardcoded.
- Per-step Qwen fallback timeout is 60s; per-loop budget 600s.
- Recipes that don't produce `STEP N:` lines fall through to legacy `run_agent` (rare with tools-enabled teacher).


---

## Phase 18 — /gwen open-system + /gwenask in-house authoring (2026-05-07)

Stand up two new Telegram surfaces for "user idea → working .exe" without external AI:

1. **`/gwen <recipe>`** — paste a literal STEP-N recipe (read_file/list_dir/write_file/edit_file/run_bash/done), executes the steps verbatim. Bypasses brain + KB + agent — pure dispatcher. Fast-path detection: if first line matches `STEP \d+:` pattern, no LLM invoked.
2. **`/gwenask <english>`** — describe an app idea, local Qwen authors the recipe, server-side b64gz-wraps content, returns recipe text for user to paste into `/gwen`.

### Built (in batches a-j, then -tune-1/-tune-2/-tune-3/-savepoint)

- `core/gwen_agent.py`: open-tool dispatch (`OPEN_TOOL_DISPATCH`), `_execute_recipe` literal executor, `run_gwen_open` entry point. Tools resolve absolute paths via `Path.expanduser()`; no sandbox.
- `core/qwen_agent.py`: `_parse_recipe_steps` (strict→relaxed STEP boundary fallback), `_KV_RE` accepts straight-double + curly-double quotes, `_b64`/`_b64gz` arg suffix decode, multi-arg permissive parse for unquoted Telegram-paste-mangled values.
- `interfaces/telegram_bot.py`:
  - `handle_gwen` reads `update.message.text` directly (preserves newlines).
  - `handle_gwenask` server-side b64gz-wraps content + AST/name-consistency self-correction retry.
  - `handle_encode` — user pastes Python source, gets back `content_b64gz="..."` snippet.
  - `handle_prompt claude/chatgpt/gemini` — per-AI brief variants.
  - `_AI_PROMPT_HEADERS` ClassVar with per-AI failure-mode warnings.
- `workspace/persona/PROMPT_BRIEF.md` — grew from ~6KB to 25KB through batches a-j capturing every AI failure mode (paste-mangle immunity, format-locked, install policy, tilde-shell-only).

### Live-bug fixes (each surfaced by a real production /gwen execution)

| Date | Bug | Fix |
|---|---|---|
| 2026-05-07 ~02:31Z | `/gwen` newlines collapsed | `handle_gwen` reads `update.message.text` directly + router preserves whitespace via `stripped.split(maxsplit=1)[1]` |
| 02:55Z | Telegram strips quotes | Phase 18c permissive parser for unquoted single-arg tools |
| 03:30Z | AI hallucinated b64gz; silent missing-arg error | Polish-3/7 diagnostic substitution on zlib-decompress fail |
| 04:05Z | Claude added preamble + code fences | Tightened brief from "discouraged" to "FAILING any of these makes user switch AI" |
| 04:20Z | Claude used `~/Desktop` in PyInstaller args | Brief: "Tilde is shell-only" section |
| 04:31Z | Phase 17 a–j shipped (e3802cb) | (graduation) |
| 18:24Z | `/gwenask` silently swallowed Qwen reply | Added `import re` to `interfaces/telegram_bot.py` (NameError eaten in handler) |
| 18:32Z | False ✅ Done on broken recipe | Executor now sets `aborted_unparseable=True` and BREAKS on first unparseable step OR tool error |
| 18:42Z | Telegram quote-strip breaks Qwen-authored content | `_wrap_plain_content_as_b64gz` rewrites `content="<source>"` → `content_b64gz="<gzipped>"` server-side before reply |
| 18:48Z | Wrapper over-matched, ate next step | Step-aware split on STEP markers BEFORE regex |
| 18:50Z | Qwen forgot close-quote on multi-line content | Greedy-to-end-of-chunk wrapper; close-quote no longer required |
| 18:52Z | Source truncated mid `f"..."` (unescaped inner quote) | Wrapper consumes everything past `content="` to chunk end (no more first-`"` truncation) |
| 18:53Z | Qwen wrote `class App` but defined `class TicTacToe` | `_validate_recipe_source` AST-checks class-name vs module-level instantiation; one self-correction retry |
| 19:00Z | TclError mixing pack+grid in same parent | TKINTER ESSENTIALS rule #6/#7 + GRID worked example |
| 19:08Z | NameError `name 'root' is not defined` inside class methods | Worked examples rewritten to use `self.root` consistently |
| 19:19Z | Qwen used filedialog instead of fixed path | FILE I/O ESSENTIALS section |
| 19:31Z | `from tkinter import *` mixed with `tk.Button(...)` | TKINTER IMPORTS section: always `import tkinter as tk` |

### `GWENASK_SYSTEM` final structure (~5KB system prompt)

1. STRICT OUTPUT RULES (5 lines: only the recipe, prefix discipline, etc.)
2. RECIPE-QUOTING DISCIPLINE (every key=value paired, single-quotes inside content)
3. TKINTER IMPORTS (always `import tkinter as tk`, never `from tkinter import *`)
4. TKINTER ESSENTIALS (8 numbered rules: timer, command-method, self.X state, stop-flag, reset-and-update, no-pack-grid-mix, self.root not module-root, __main__ guard)
5. FILE I/O ESSENTIALS (honor explicit paths, mkdir parents, encoding='utf-8')
6. PICK THE RIGHT PATTERN (TIMER vs EVENT-DRIVEN — when to use root.after vs not)
7. THREE WORKED MINI-EXAMPLES (TIMER, EVENT-DRIVEN counter, GRID 3x3) — Qwen pattern-matches examples FIRST, so they all use `self.root` consistently
8. RECIPE SHAPE (exact 5-step boilerplate with `<Name>` placeholders)

### Validated scenarios (Qwen-authored end-to-end)

| Scenario | Status | Iters to lock |
|---|---|---|
| Stopwatch | ✅ 3/3 | 8 (drove the prompt foundation) |
| Counter | ✅ 3/3 | 6 (drove TIMER vs EVENT-DRIVEN split) |
| Tic-Tac-Toe | ✅ 5/5 | 13 (drove name-consistency self-correction loop) |
| Sticky Note | 🟡 partial | ongoing (file I/O works in some iters; `*` import bug) |
| Pomodoro / Calculator / NotesDB / CSVViewer / MarkdownPreview / JobAnalyzerV2 | ⏳ untested | queued |

### Decisions captured during Phase 18

- **Server-side b64gz wrap** is the only way to keep recipes paste-mangle-immune through Telegram. Qwen can't compute base64 in its head, so `handle_gwenask` does it AFTER generation, transparently to the user.
- **AST + name-consistency validation** is the right gate for Qwen-authored Python. Pure regex/syntactic checks miss class-vs-instance mismatches that AST catches in one shot.
- **One self-correction retry, not loop** — feeds the specific error back to Qwen as `PREVIOUS ATTEMPT FAILED VALIDATION: <error>`. More retries don't help; Qwen either fixes or doesn't.
- **Worked examples beat rules** — Qwen pattern-matches the WORKED example shape over abstract rules. Every prompt rule was paired with a corrected example to make the rule visible structurally.
- **Executor MUST hard-fail on tool error**, not just parse error — earlier "WARNING + continue" let PyInstaller + copy steps run against stale source from a prior build, returning false ✅.
- **`num_predict=6144` on `/gwenask` calls** — Ollama default ~1024 tokens was truncating Qwen mid-class for any recipe over ~2KB.

### Save point

Tag `phase18-tune-savepoint-20260507` (commit `3f7837f`). Revert: `git reset --hard phase18-tune-savepoint-20260507`.

### Known caveats / future work

- Sticky Note + scenarios 5-10 still need verification.
- Each new task surface (file I/O, sqlite, ttk.Treeview, etc.) tends to surface 1-2 new Qwen quirks needing a universal rule add.
- `GWENASK_SYSTEM` is ~5KB; Qwen 8K context budget remaining ~3KB for output (recipes typically 1.5-3KB so it works, but headroom is thin).
- Skip-path / KB graduation paths from Phase 14-16 are NOT used by `/gwen` — it bypasses agent, brain, KB, and orchestrator entirely. Pure dispatcher.

---

## Phase 18.5 — Fresh-PC setup tooling (2026-05-07)

After Phase 18 shipped, layered three install paths so getting
Sentinel onto a new machine takes zero manual config:

| Tool | Role |
|---|---|
| `setup.ps1` | 10-step idempotent installer for an already-cloned repo. |
| `bootstrap.ps1` | Single-file fetcher (Git URL / zip / folder). Chains into setup.ps1. |
| `Sentinel-Setup.exe` | ps2exe-compiled bootstrap.ps1 (~31 KB). Double-clickable. |
| `build-installer.ps1` | Rebuilds the .exe from bootstrap.ps1 (auto-installs ps2exe module). |
| `SETUP.md` | User-facing manual: decision tree, per-step explanation, Telegram credential walkthrough, troubleshooting, uninstall. |

### Decisions captured during 18.5

- **Three layered tools, not one mega-installer.** `setup.ps1` is
  source-of-truth for what gets installed. `bootstrap.ps1` only
  handles the "get the code onto the box" step. `.exe` is just a
  packaging detail. Each layer is testable independently.
- **PowerShell over Python bootstrapper.** Sentinel is Windows-only
  on this hardware (RTX 3050, owner directive). PS1 is native, no
  bootstrapping problem (Python doesn't exist yet on a fresh PC).
- **`winget` for everything.** Python 3.12, Ollama, Node.js, Git all
  install via winget. No manual download links to chase. Falls back
  to a clear error message if winget itself is missing (Windows 10
  pre-1809).
- **Idempotent steps over a "first-run flag" file.** Each step asks
  "is this already done?" before acting. Re-running setup.ps1 after
  a partial failure doesn't reinstall what's already there.
- **Inline credential instructions, not external doc links.** Step 9
  prints the @BotFather + @userinfobot walkthrough right in the
  terminal so users never alt-tab to find instructions.
- **`.env` for Telegram credentials, not setx.** `setx` is offered
  as a fallback, but `.env` keeps the token out of system-wide
  environment and survives Windows user-profile resets. `.env` is
  gitignored.
- **Don't bundle the 3 GB Qwen weights inside the .exe.** Models
  download at runtime via `ollama pull`. Keeps the .exe at 31 KB
  and avoids stale-weights problem if Qwen versions change.
- **No code-signing.** SmartScreen warning is the unavoidable cost
  of skipping a $~300/yr cert; documented in SETUP.md so users know
  it's expected and they should click "Run anyway." If we ever
  distribute beyond owner+friends, revisit.
- **Startup banner in main.py prints BEFORE any error.** Token-
  missing errors include the exact `setx SENTINEL_TELEGRAM_TOKEN
  "..."` command to fix manually -- error message is its own
  documentation.

### Open caveats

- ps2exe-wrapped `.exe`s trip Windows Defender heuristics on some
  systems; SETUP.md documents the workaround (Defender exclusion or
  distribute the .ps1 directly).
- `bootstrap.ps1` Folder mode uses robocopy `/MIR` -- mirrors the
  source, including any local edits the user has *not* committed.
  Doesn't carry `.git` (excluded), so the new PC has no commit
  history. Good for personal moves, not for fresh dev setup; for
  that use Git mode.
- `setup.ps1` step 7 (PyInstaller) prints WARN on failure but
  continues -- the bot runs fine without PyInstaller, only the
  `/gwenask`->.exe pipeline breaks. Step 8 (Claude CLI) similarly
  degrades gracefully -- bot runs Qwen-only without it.
