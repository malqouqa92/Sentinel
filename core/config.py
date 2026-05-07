from pathlib import Path

PROJECT_NAME = "sentinel"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = "sentinel.jsonl"

WORKSPACE_DIR = PROJECT_ROOT / "workspace"

# Where /gwenask + /gwen build their .exe demos. Override via env var
# SENTINEL_DEMOS_DIR if you want them somewhere other than ~/Sentinel-Demos.
import os as _os
SENTINEL_DEMOS_DIR = Path(_os.environ.get(
    "SENTINEL_DEMOS_DIR",
    str(Path.home() / "Desktop" / "Sentinel-Demos"),
)).resolve()
# POSIX-style string used in recipe templates / GWENASK_SYSTEM. Forward
# slashes work fine on Windows for Python and most shells we use.
SENTINEL_DEMOS_DIR_POSIX = SENTINEL_DEMOS_DIR.as_posix()

VRAM_LIMIT_MB = 4096

REGISTERED_COMMANDS = {
    "/ping", "/help", "/status",
    "/extract", "/search", "/file", "/exec", "/code", "/qcode", "/gwen",
    "/models", "/complexity",
    # Phase 10 -- memory + pipelines
    "/remember", "/forget", "/recall", "/memory",
    "/curate", "/curate_approve", "/curate_reject",
    "/jobsearch", "/research",
    "/commit",
}

DB_PATH = PROJECT_ROOT / "sentinel.db"
WORKER_POLL_INTERVAL = 1.0
STALE_LOCK_TIMEOUT = 300
MAX_RECOVERIES = 5
# Phase 12.5 hardening: backoff after worker requeues a task because
# the GPU lock is busy. Without this the loop hammers claim->requeue
# at ~50/sec when a stale lock isn't released. 2s caps the rate at
# 0.5 attempts/sec, well below the cost of a real GPU-bound task.
WORKER_GPU_REQUEUE_BACKOFF_S = 2.0
# /restart drain wait: max seconds handle_restart waits for in-flight
# 'processing' tasks to finish before triggering shutdown + spawn.
RESTART_DRAIN_TIMEOUT_S = 30

# Authoritative mapping of routed command to executor.
# - str   -> agent name (looked up in AGENT_REGISTRY)
# - None  -> built-in handler in core.orchestrator (ping/status/help)
# Adding a new agent-backed command means: add a skill to skills/, add
# an agent YAML to agents/, then add an entry here. Worker doesn't change.
COMMAND_AGENT_MAP: dict[str, str | None] = {
    "/ping": None,
    "/help": None,
    "/status": None,
    "/extract": "job_analyst",
    # "skill:NAME" form bypasses the agent layer for single-skill commands
    # that don't need a persona/pipeline. Resolved by the orchestrator.
    "/search": "skill:web_search",
    "/file": "skill:file_io",
    "/exec": "skill:code_execute",
    "/code": "code_assistant",
    "/qcode": "qcode_assistant",
    "/gwen": "gwen_assistant",
    "/models": None,
    "/complexity": None,
    # Phase 10 -- /remember, /forget, /recall, /memory, /curate are
    # handled directly in the Telegram bot. The four memory ops are
    # sub-second SQLite calls (queue round-trip would add ~1-2s for no
    # benefit). /curate kicks off a long-running curation flow as a
    # bg task in the bot process; not queued because it's interactive
    # (multi-message Telegram approval back-and-forth doesn't fit the
    # task-row model). /jobsearch and /research are real agents.
    "/jobsearch": "job_searcher",
    "/research": "researcher",
}

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"
MODEL_KEEP_ALIVE = "5m"
LLM_TIMEOUT = 120
LLM_TEMPERATURE = 0.1

SCRAPE_DELAY = 2.0
SCRAPE_TIMEOUT = 10
SCRAPE_MAX_BODY_CHARS = 2000

KNOWLEDGE_DB_PATH = PROJECT_ROOT / "knowledge.db"
# Phase 15a: caps raised drastically because they were defensive
# paranoia, not real constraints. FTS5 + partial indexes keep query
# time microseconds-level even at 50K rows; KB context injection is
# capped at KNOWLEDGE_CONTEXT_MAX_CHARS regardless of total. Disk is
# plentiful (~3KB per row at 768-dim embeddings = 150 MB at the cap).
# The OLD 500-cap defeated the self-learning loop by deleting
# patterns Qwen had already learned.
KNOWLEDGE_MAX_ENTRIES = 50000
KNOWLEDGE_CONTEXT_MAX_CHARS = 4000
# Phase 15a auto-transition windows (days). Active->stale walks
# patterns older than KB_STALE_AFTER_DAYS with usage_count <= 1;
# stale->archived walks rows further past KB_ARCHIVE_AFTER_DAYS.
# Pinned rows are NEVER touched by either step.
KB_STALE_AFTER_DAYS = 30
KB_ARCHIVE_AFTER_DAYS = 90

# Teaching loop uses the local `claude` CLI subprocess (Claude Code).
# Sentinel itself makes ZERO outbound API calls. The CLI handles auth
# via the user's existing Claude Code login -- no API key needed.
CLAUDE_CLI_MODEL = "sonnet"  # alias resolved by CLI to current Sonnet
CLAUDE_CLI_TIMEOUT = 1800    # 30 min: Claude can grind on hard tasks
TEACHING_EXECUTOR_TIMEOUT = 30

# Phase 8 -- model routing
AUTO_ROUTING_ENABLED = True
COMPLEXITY_TIER_THRESHOLDS = {"basic": 0.3, "standard": 0.6}
FALLBACK_RETRY_DELAY = 5      # seconds before retrying on timeout
MAX_FALLBACK_ATTEMPTS = 3     # max models to try before giving up

# Phase 9 -- Telegram + Brain
import os as _os9
TELEGRAM_TOKEN = _os9.environ.get("SENTINEL_TELEGRAM_TOKEN", "")
TELEGRAM_AUTHORIZED_USERS: list[int] = [
    int(uid) for uid in
    _os9.environ.get("SENTINEL_TELEGRAM_USER_ID", "").split(",")
    if uid.strip().isdigit()
]
TELEGRAM_TASK_TIMEOUT = 1800       # 30 minutes -- whole-task ceiling
TELEGRAM_CLAUDE_TIMEOUT = 1800     # 30 minutes -- /claude direct calls
TELEGRAM_MAX_MESSAGE_LENGTH = 4096 # Telegram hard limit

BRAIN_MODEL = "sentinel-brain"
BRAIN_TEMPERATURE = 0.1
BRAIN_SUMMARIZE_TEMPERATURE = 0.3
WORKER_MODEL = "qwen2.5-coder:3b"

# Pre-Phase-15: embedding model for hybrid retrieval. nomic-embed-text
# (~270 MB, 768-dim, 2048-token context) is the sweet spot at 4 GB
# VRAM. Combined with worker (qwen2.5:3b ~1.9 GB) = ~2.2 GB, leaving
# ~1.8 GB for KV cache. The 2048-token (~6000 char) context handles
# our long recipe patterns without truncation that smaller embedders
# (all-minilm @ 256 tokens) couldn't.
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

# Default model roster. The registry checks Ollama (/api/tags) and the
# claude CLI to flag what's actually available at startup; unpulled
# models stay registered but available=False. The user runs
# `ollama pull <model_id>` to enable additional local models.
MODEL_ROSTER: list[dict] = [
    # Sentinel-brain: custom Modelfile derived from qwen3:1.7b with
    # num_ctx=8192, num_predict=1024, temp=0.1 baked in. Used by the
    # Telegram-facing intent classifier + result summarizer.
    {
        "name": "sentinel-brain",
        "model_id": "sentinel-brain",
        "backend": "ollama",
        "context_window": 8192,
        "speed_tier": "fast",
        "capability_tier": "standard",
        "vram_mb": 1300,
        "max_output_tokens": 1024,
    },
    # Raw qwen3:1.7b also available if a caller wants vanilla settings.
    {
        "name": "qwen3-brain",
        "model_id": "qwen3:1.7b",
        "backend": "ollama",
        "context_window": 8192,
        "speed_tier": "fast",
        "capability_tier": "standard",
        "vram_mb": 1300,
        "max_output_tokens": 2000,
    },
    # Worker: code-specialized; better at JSON extraction and code gen
    {
        "name": "qwen-coder",
        "model_id": "qwen2.5-coder:3b",
        "backend": "ollama",
        "context_window": 8192,
        "speed_tier": "medium",
        "capability_tier": "standard",
        "vram_mb": 2800,
        "max_output_tokens": 2000,
    },
    # Ceiling: teaching loop, hard problems, /claude direct chat
    {
        "name": "claude-cli",
        "model_id": "claude-cli",
        "backend": "claude_cli",
        "context_window": 200000,
        "speed_tier": "slow",
        "capability_tier": "advanced",
        "vram_mb": None,
        "max_output_tokens": 4096,
    },
]

# ---------------------------------------------------------------------
# Phase 10 -- persona files, memory, pipelines, curation
# ---------------------------------------------------------------------

# Persona files live in workspace/ (human-editable, git-trackable).
PERSONA_DIR = WORKSPACE_DIR / "persona"
PROTECTED_FILES = {
    "SOUL.md", "IDENTITY.md", "USER.md", "MEMORY.md",
    # Phase 15d -- QWENCODER.md is the human-curated teaching memo
    # the worker model reads at the top of every shadow-plan call.
    # File_guard protects it the same way the brain's persona files
    # are protected; edits land via authorize_update (or via the
    # forthcoming /curate qwencoder flow).
    "QWENCODER.md",
    # Phase 17c -- CODE_TIERS.md is the Claude-side tier playbook.
    # Loaded into PRE_TEACH_SYSTEM based on the complexity classifier
    # to push Claude toward DECOMPOSE on tier-3+ tasks instead of
    # one-shotting them. Same protection pattern as QWENCODER.md.
    "CODE_TIERS.md",
}
FILE_GUARD_CHECK_INTERVAL = 300  # seconds

# Per-file injection caps when persona is loaded into the brain context.
# MEMORY.md gets more headroom because it's the dynamic, curation-driven
# fact store; the other three are hand-edited and meant to stay terse.
# QWENCODER.md is allowed a larger budget because it's a teaching
# document with annotated examples; the worker model needs the full
# memo, not a clipped excerpt.
PERSONA_INJECT_MAX_CHARS = {
    "IDENTITY.md": 2000,
    "SOUL.md": 2000,
    "USER.md": 2000,
    "MEMORY.md": 5000,
    "QWENCODER.md": 6000,
    # Phase 17c -- tier playbook is short by design. Whole file
    # loads (no per-tier extraction) -- the per-tier highlight
    # happens in skills/code_assist via formatting around the load.
    "CODE_TIERS.md": 4000,
}

# Three-tier persistent memory. SEPARATE DB from sentinel.db (task queue)
# and knowledge.db (code-pattern KB). Don't merge -- different lifecycles
# and the FTS triggers would interfere.
MEMORY_DB_PATH = PROJECT_ROOT / "memory.db"
WORKING_MEMORY_MAX_MESSAGES = 20      # per Telegram session, in-process
# Phase 15a: caps raised. Same rationale as KNOWLEDGE_MAX_ENTRIES --
# the old 200/500 caps were paranoia and turned the prune path into
# silent data loss. With archive-not-delete in core/memory.py, "over
# the cap" now just shifts low-relevance / low-confidence rows into
# state='archived' (still on disk, hidden from default queries).
EPISODIC_MAX_PER_SCOPE = 100000       # per agent (or "global")
SEMANTIC_MAX_ENTRIES = 50000
MEMORY_DECAY_DAYS = 30                # episodes older than this decay
MEMORY_DECAY_FACTOR = 0.95
AUTO_EXTRACT_THRESHOLD = 5            # messages between auto-extractions
EPISODIC_CONTEXT_MAX_CHARS = 3000     # per agent context window
PROFILE_CONTEXT_MAX_CHARS = 2000

# Source -> default confidence mapping for semantic_memory.store_fact.
# user_explicit > task_derived > auto_extracted.
MEMORY_SOURCE_CONFIDENCE = {
    "user_explicit": 1.0,
    "task_derived": 0.8,
    "persona_file": 1.0,
    "auto_extracted": 0.6,
}

# Nightly curation. Local time (24h). 03:00 = post-day, low activity.
CURATION_HOUR_LOCAL = 3
CURATION_MINUTE_LOCAL = 0
CURATION_LOOKBACK_HOURS = 24

# Job search defaults (python-jobspy). Override per-call via /jobsearch
# flags (--distance / --hours / --results / --sites).
JOBSPY_DELAY = 2.0
JOBSPY_DEFAULT_DISTANCE = 250
JOBSPY_DEFAULT_HOURS_OLD = 72
JOBSPY_DEFAULT_RESULTS = 20
JOBSPY_SITES = ["indeed", "linkedin", "google", "zip_recruiter"]
JOB_DESCRIPTION_MAX_CHARS = 2000

# Phase 13: number of postings packed into a single LLM scoring call.
# qwen2.5:3b has a 32k context window; a batch of 5 postings @ ~1500
# chars each + system prompt + output room is ~10k tokens, well under.
# Bigger batches reduce round-trip overhead but enlarge the blast
# radius if Qwen returns malformed JSON for the batch.
JOB_SCORE_BATCH_SIZE = 5

# Phase 13 -- smarter rescan
# (a) Adaptive title filter: when a /jobsearch drops a high fraction of
#     scraped postings to the title pre-filter, sample the dropped titles
#     and ask the brain to extract 1-N candidate negative keywords. The
#     bot writes them to PROFILE.title_filter.negative on the fly so the
#     next /jobsearch is more selective. Auto-apply (no confirmation).
JOB_ADAPTIVE_FILTER_ENABLED = True
JOB_ADAPTIVE_FILTER_MIN_DROPS = 10        # need at least N drops to act
JOB_ADAPTIVE_FILTER_DROP_RATIO = 0.4      # AND drop_ratio > this
JOB_ADAPTIVE_FILTER_MAX_NEW = 3           # max new negatives per run
JOB_ADAPTIVE_FILTER_SAMPLE_SIZE = 20      # how many titles to send to brain
# (b) Query expansion: a single /jobsearch search term is broadened into
#     N variants (the original + abbreviation-expansion + reverse). Each
#     variant scrapes independently; rows are deduped by URL before the
#     title filter. Doubles/triples coverage on jargon-y searches.
JOB_QUERY_EXPANSION_ENABLED = True
JOB_QUERY_EXPANSION_MAX = 3               # max total queries (incl. original)

# Sidecar file written by job_scrape on every run; read by the bot's
# /jobsearch handler to drive the adaptive filter without piping stats
# through the typed agent pipeline (which expects single-list outputs).
LAST_SCRAPE_STATS_PATH = LOG_DIR / "last_scrape_stats.json"

# Research pipeline
RESEARCH_SUMMARY_MAX_WORDS = 300
RESEARCH_MAX_RESULTS = 8

# Output directories for the new pipelines
RESEARCH_OUTPUT_DIR = WORKSPACE_DIR / "research"
JOBSEARCH_OUTPUT_DIR = WORKSPACE_DIR / "job_searches"

# ---------------------------------------------------------------------
# Phase 11 -- scheduler, health, hardening
# ---------------------------------------------------------------------

# Scheduler. EST-anchored: cron expressions and HH:MM active-hours
# windows are interpreted in this zone, but next_run_at is stored UTC.
SCHEDULER_TIMEZONE = "America/New_York"
SCHEDULER_POLL_INTERVAL = 30           # seconds between due-job sweeps
SCHEDULER_STARTUP_SPREAD_SECONDS = 300 # spread overdue jobs across this
SCHEDULER_TASK_TIMEOUT = 600           # per scheduled-job ceiling
SCHEDULER_MAX_CONCURRENT = 1           # hard cap; worker GPU lock is the
                                       # real serializer, this is a sanity belt

BACKUP_DIR = PROJECT_ROOT / "backups"
BACKUP_KEEP_DAYS = 7

# Hardening
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5
DISK_FREE_ALERT_BYTES = 1_000_000_000   # 1 GB
LOG_DIR_ALERT_BYTES = 100_000_000       # 100 MB
HEALTH_PORT = 18700
HEALTH_BIND_HOST = "127.0.0.1"

# ---------------------------------------------------------------------
# Phase 16 -- Claude-skip path feature flags (Batch C, prep)
# ---------------------------------------------------------------------

# Master switch for the Claude-skip path. Default OFF on first ship --
# Batch C lands in TELEMETRY-ONLY mode where eligibility is computed
# and logged but skipping is NOT performed. Flip to True only after
# observing enough log evidence that skip-eligibility predicts safe
# replays. See PHASES.md Phase 16 Batch C section.
SKIP_PATH_ENABLED = True

# Skip-eligibility thresholds (Batch C). A pattern is skip-eligible
# when:  pinned = 1  OR  (solo_passes >= MIN_PASSES AND
# solo_attempts == solo_passes AND last_verified within FRESHNESS days
# AND (qwen_plan_agreement IS NULL OR >= AGREEMENT_FLOOR)). Strict by
# design: skipping Claude is irreversible per attempt, so the gate
# must err on the side of safety.
SKIP_PATH_MIN_PASSES = 3
SKIP_PATH_FRESHNESS_DAYS = 30
SKIP_PATH_AGREEMENT_FLOOR = 0.5

# Diff-match acceptance threshold for replay (Batch B is the gate).
# A replay diff that scores >= this against the stored solution_code
# passes; below this falls through to the full Claude pipeline.
SKIP_PATH_DIFF_MATCH_THRESHOLD = 0.7

# ---------------------------------------------------------------------
# Phase 17b -- /code auto-decompose chain runner
# ---------------------------------------------------------------------

# When True, a /code that produces a DECOMPOSE response will queue
# child tasks back into the worker queue automatically (one per
# subtask). Each child runs as its own /code with its own pre-teach,
# its own KB retrieval, its own small recipe. The user sees N
# separate /code completion messages -- one per subtask -- in chat.
# Default OFF so first ship is opt-in; flip to True after live
# validation. The DECOMPOSE branch from Phase 17 Batch 1 still runs
# unchanged when this is False (just shows the user the suggested
# subtask list and stops).
CODE_CHAIN_ENABLED = True

# Hard depth cap. Top-level /code is depth=0; child tasks queued by
# the chain runner are depth=parent+1. When pre-teach emits DECOMPOSE
# at depth >= MAX_DEPTH, the chain runner refuses to queue further
# children and surfaces the suggested subtasks back to the user
# (graceful fallback to Phase 17 Batch 1 behavior). Default 1: a
# top-level /code can decompose ONCE; children cannot decompose
# further. Prevents runaway chains.
CODE_CHAIN_MAX_DEPTH = 1

LOG_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
PERSONA_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JOBSEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
