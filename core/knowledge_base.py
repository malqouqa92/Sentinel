"""SQLite + FTS5 knowledge base.

Stores patterns (working solutions) and limitations (known failures of
the local model) along with tags. FTS5 indexes tags + summary + pattern
+ explanation for fast keyword lookup. Phase 15a added local embeddings
for hybrid retrieval; FTS5 still does the keyword pass.

Phase 15a: lifecycle states replace destructive prune. Every row has
``state`` (active|stale|archived), ``pinned`` (orthogonal protection
flag), and ``archived_at`` (NULL until archived). prune() now ARCHIVES
the lowest-usage active rows; nothing is ever truly deleted by the
self-learning loop. ``auto_transition_lifecycle`` walks active->stale
on age + low usage, and stale->archived on further age. Pinned rows
are immune to all automatic transitions.
"""
import sqlite3
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from core import config
from core.database import _utcnow_iso  # reuse the iso8601 helper
from core.logger import log_event


class KnowledgeEntry(BaseModel):
    id: int
    category: str
    tags: str
    problem_summary: str
    solution_code: str | None = None
    solution_pattern: str | None = None
    explanation: str
    source_trace_id: str
    created_at: str
    usage_count: int
    # Phase 14a -- graduation test transfer-verification fields.
    # solo_attempts/passes count Qwen-only verification runs (no Claude
    # in the loop). needs_reteach=1 means the pattern's solo pass rate
    # dropped below threshold and the next match should escalate to
    # Claude rather than be used as a few-shot example.
    solo_attempts: int = 0
    solo_passes: int = 0
    last_verified_at: str | None = None
    needs_reteach: bool = False
    # Phase 14b (Solution A) -- the git SHA the working tree was on
    # BEFORE the /code attempt that produced this pattern. Graduation
    # resets the tree to this SHA, replays the stored recipe through
    # Qwen's stepfed agent, then runs Claude review for verdict --
    # the actual production skill, not blank-page text generation.
    # Old patterns from before Phase 14b have base_sha=NULL and fall
    # back to the legacy text-gen graduation.
    base_sha: str | None = None
    # Phase 15a -- lifecycle states (archive-not-delete). state values:
    # active (default; eligible for retrieval + auto-transitions),
    # stale (aged, low usage; still retrievable but a candidate for
    # archival), archived (excluded from retrieval but never deleted).
    # ``pinned`` is orthogonal: a pinned row is immune to every
    # automatic transition (prune + auto_transition_lifecycle skip
    # it). ``archived_at`` records when state flipped to archived,
    # NULL otherwise.
    state: str = "active"
    pinned: bool = False
    archived_at: str | None = None
    # Phase 15b -- provenance. Read from the contextvar at write
    # time. Default 'foreground' (user-driven) for back-compat with
    # rows written before the migration.
    created_by_origin: str = "foreground"
    # Phase 15c -- shadow planning. The recipe Qwen wrote (in shadow
    # mode, parallel to Claude's recipe) and the structural agreement
    # score in [0.0, 1.0]. NULL on patterns written before 15c or on
    # /code attempts where the shadow call timed out / crashed (the
    # shadow path is best-effort and never blocks /code).
    qwen_plan_recipe: str | None = None
    qwen_plan_agreement: float | None = None


CATEGORIES = {"pattern", "limitation"}


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    # Phase 8 perf tuning -- same as core/database.py
    conn.execute("PRAGMA cache_size = -64000;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA mmap_size = 268435456;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def _row_to_entry(r: sqlite3.Row) -> KnowledgeEntry:
    # Defensive get for Phase 14a/14b/15a columns -- old rows may
    # pre-date the migration on the read path.
    keys = r.keys() if hasattr(r, "keys") else []
    return KnowledgeEntry(
        id=r["id"], category=r["category"], tags=r["tags"],
        problem_summary=r["problem_summary"],
        solution_code=r["solution_code"],
        solution_pattern=r["solution_pattern"],
        explanation=r["explanation"],
        source_trace_id=r["source_trace_id"],
        created_at=r["created_at"], usage_count=r["usage_count"],
        solo_attempts=int(r["solo_attempts"]) if "solo_attempts" in keys else 0,
        solo_passes=int(r["solo_passes"]) if "solo_passes" in keys else 0,
        last_verified_at=(
            r["last_verified_at"] if "last_verified_at" in keys else None
        ),
        needs_reteach=bool(
            r["needs_reteach"] if "needs_reteach" in keys else 0
        ),
        base_sha=r["base_sha"] if "base_sha" in keys else None,
        state=(r["state"] if "state" in keys else "active") or "active",
        pinned=bool(r["pinned"] if "pinned" in keys else 0),
        archived_at=r["archived_at"] if "archived_at" in keys else None,
        created_by_origin=(
            r["created_by_origin"]
            if "created_by_origin" in keys else "foreground"
        ) or "foreground",
        qwen_plan_recipe=(
            r["qwen_plan_recipe"]
            if "qwen_plan_recipe" in keys else None
        ),
        qwen_plan_agreement=(
            float(r["qwen_plan_agreement"])
            if ("qwen_plan_agreement" in keys
                and r["qwen_plan_agreement"] is not None)
            else None
        ),
    )


class KnowledgeBase:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or config.KNOWLEDGE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        conn = _connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    category         TEXT NOT NULL,
                    tags             TEXT NOT NULL,
                    problem_summary  TEXT NOT NULL,
                    solution_code    TEXT,
                    solution_pattern TEXT,
                    explanation      TEXT NOT NULL,
                    source_trace_id  TEXT NOT NULL,
                    created_at       TEXT NOT NULL,
                    usage_count      INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # FTS5 indexes the searchable text. content='knowledge' makes
            # this a contentless-external-content table; sync via triggers.
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
                USING fts5(
                    tags, problem_summary, solution_pattern, explanation,
                    content='knowledge', content_rowid='id'
                )
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS knowledge_ai
                AFTER INSERT ON knowledge BEGIN
                    INSERT INTO knowledge_fts(
                        rowid, tags, problem_summary, solution_pattern,
                        explanation
                    ) VALUES (
                        new.id, new.tags, new.problem_summary,
                        COALESCE(new.solution_pattern, ''),
                        new.explanation
                    );
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS knowledge_ad
                AFTER DELETE ON knowledge BEGIN
                    INSERT INTO knowledge_fts(
                        knowledge_fts, rowid, tags, problem_summary,
                        solution_pattern, explanation
                    ) VALUES (
                        'delete', old.id, old.tags, old.problem_summary,
                        COALESCE(old.solution_pattern, ''),
                        old.explanation
                    );
                END
                """
            )
            # Phase 14a -- graduation test columns. SQLite ALTER TABLE
            # ADD COLUMN errors if the column already exists, so we
            # check the column list first. Idempotent; cheap on every
            # boot.
            existing_cols = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(knowledge)"
                ).fetchall()
            }
            if "solo_attempts" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge "
                    "ADD COLUMN solo_attempts INTEGER NOT NULL DEFAULT 0"
                )
            if "solo_passes" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge "
                    "ADD COLUMN solo_passes INTEGER NOT NULL DEFAULT 0"
                )
            if "last_verified_at" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge "
                    "ADD COLUMN last_verified_at TEXT"
                )
            if "needs_reteach" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge "
                    "ADD COLUMN needs_reteach INTEGER NOT NULL DEFAULT 0"
                )
            # Phase 14b: base_sha for agentic-pipeline graduation.
            if "base_sha" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge "
                    "ADD COLUMN base_sha TEXT"
                )
            # Pre-Phase-15: embedding for hybrid retrieval. Stored as
            # raw float32 BLOB (1.5KB per row at 384-dim). NULL on rows
            # written before this migration; backfill is a separate
            # one-shot job (KB.backfill_embeddings()). Search degrades
            # gracefully to FTS5-only for NULL rows.
            if "embedding" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge "
                    "ADD COLUMN embedding BLOB"
                )
            # Phase 15a -- lifecycle states. ALTER TABLE ADD COLUMN
            # with a NOT NULL DEFAULT backfills existing rows in a
            # single statement, so old DBs come out with state='active'
            # and pinned=0 on every existing row.
            if "state" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge "
                    "ADD COLUMN state TEXT NOT NULL DEFAULT 'active'"
                )
            if "pinned" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge "
                    "ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
                )
            if "archived_at" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge ADD COLUMN archived_at TEXT"
                )
            # Phase 15b -- write-origin provenance column. NOT NULL
            # DEFAULT backfills every pre-15b row to 'foreground'.
            if "created_by_origin" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge ADD COLUMN "
                    "created_by_origin TEXT NOT NULL "
                    "DEFAULT 'foreground'"
                )
            # Phase 15c -- shadow planning columns. Both nullable;
            # old rows just don't have shadow data, and a /code that
            # hits the shadow timeout also leaves them NULL.
            if "qwen_plan_recipe" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge "
                    "ADD COLUMN qwen_plan_recipe TEXT"
                )
            if "qwen_plan_agreement" not in existing_cols:
                conn.execute(
                    "ALTER TABLE knowledge "
                    "ADD COLUMN qwen_plan_agreement REAL"
                )
            # Partial indexes -- per the project's hardware constraint
            # discipline. Each index covers ONLY the rows we actually
            # query (needs_reteach=1 is a small subset; pinned=1 is
            # the small protected set; archived rows are excluded from
            # default search and live behind their own selective idx).
            # On a 50K-row KB these stay tiny.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_needs_reteach "
                "ON knowledge(needs_reteach) WHERE needs_reteach = 1"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_last_verified "
                "ON knowledge(last_verified_at) "
                "WHERE last_verified_at IS NOT NULL"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_pinned "
                "ON knowledge(pinned) WHERE pinned = 1"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_archived "
                "ON knowledge(state) WHERE state = 'archived'"
            )
        finally:
            conn.close()

    def _bump_usage(
        self, conn: sqlite3.Connection, ids: Iterable[int]
    ) -> None:
        ids_list = list(ids)
        if not ids_list:
            return
        placeholders = ",".join("?" * len(ids_list))
        conn.execute(
            f"UPDATE knowledge SET usage_count = usage_count + 1 "
            f"WHERE id IN ({placeholders})",
            ids_list,
        )

    @staticmethod
    def _normalize_tags(tags: list[str] | str) -> str:
        if isinstance(tags, str):
            parts = [t.strip() for t in tags.split(",") if t.strip()]
        else:
            parts = [t.strip() for t in tags if t and t.strip()]
        return ",".join(parts)

    @staticmethod
    def _escape_fts_query(query: str) -> str:
        """FTS5 query syntax treats some chars specially. Wrap each
        whitespace-delimited token in double quotes so quotes/dashes/etc.
        in user input don't break the query."""
        tokens = [t for t in query.replace('"', " ").split() if t]
        if not tokens:
            return '""'
        return " OR ".join(f'"{t}"' for t in tokens)

    # Pre-Phase-15 hybrid retrieval: FTS5 returns a wider net of
    # candidates (default 4× max_results), then we rerank by blending
    # BM25 rank with cosine similarity to a local embedding of the
    # query. Caller still gets exactly max_results entries; they're
    # just SMARTER. Rows without embeddings fall back to BM25-only
    # ranking and still appear, just without the semantic boost --
    # never excluded.
    HYBRID_CANDIDATE_MULTIPLIER = 4
    HYBRID_BM25_WEIGHT = 0.4  # 0.0=pure semantic, 1.0=pure keyword

    def search(
        self, query: str, max_results: int = 5,
        hybrid: bool = True,
        include_archived: bool = False,
        exclude_pattern_ids: list[int] | tuple[int, ...] | None = None,
    ) -> list[KnowledgeEntry]:
        """FTS5 + embedding hybrid search. Set ``hybrid=False`` to get
        the pure-FTS5 behaviour (used by tests that need deterministic
        BM25-only ordering). ``include_archived=False`` (default) hides
        rows whose state is 'archived' -- Phase 15a's archive-not-delete
        path keeps them on disk but out of retrieval. Set True to
        surface archived rows for /kb restore workflows.

        Phase 16 Batch C: ``exclude_pattern_ids`` removes specific row
        ids from results. Used by the skip-path fallback so Claude
        doesn't re-derive a recipe that just failed replay."""
        if not query.strip():
            return []
        fts_query = self._escape_fts_query(query)
        candidate_limit = (
            max_results * self.HYBRID_CANDIDATE_MULTIPLIER
            if hybrid else max_results
        )
        archived_filter = (
            "" if include_archived else " AND k.state != 'archived'"
        )
        exclude_filter = ""
        if exclude_pattern_ids:
            try:
                ids_csv = ",".join(
                    str(int(x)) for x in exclude_pattern_ids
                )
                exclude_filter = f" AND k.id NOT IN ({ids_csv})"
            except (TypeError, ValueError):
                log_event(
                    "SEN-system", "WARNING", "knowledge_base",
                    f"search: ignoring malformed exclude_pattern_ids="
                    f"{exclude_pattern_ids!r}",
                )
        conn = _connect(self.db_path)
        try:
            try:
                rows = conn.execute(
                    f"""
                    SELECT k.* FROM knowledge_fts f
                    JOIN knowledge k ON k.id = f.rowid
                    WHERE knowledge_fts MATCH ?{archived_filter}{exclude_filter}
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, candidate_limit),
                ).fetchall()
            except sqlite3.OperationalError as e:
                # Malformed FTS query -> empty result rather than crash.
                log_event(
                    "SEN-system", "WARNING", "knowledge_base",
                    f"FTS query failed: {e}; returning empty",
                )
                return []
            entries = [_row_to_entry(r) for r in rows]

            if hybrid and len(entries) > 1:
                # Pre-Phase-15: rerank candidates by hybrid score.
                # Embeddings are loaded directly off the rows (no
                # extra query). If the embed call fails (Ollama
                # down, model missing) the blend degrades to BM25
                # rank only -- behaviour identical to old code path.
                from core.embeddings import (
                    cosine_similarity, embed_text, hybrid_score,
                )
                query_blob = embed_text(query, "SEN-system")
                if query_blob is not None:
                    scored: list[tuple[float, KnowledgeEntry]] = []
                    for rank_idx, e in enumerate(entries):
                        # row.embedding may be None (pre-migration row)
                        # or bytes; cosine_similarity handles None.
                        cand_blob = (
                            rows[rank_idx]["embedding"]
                            if "embedding" in rows[rank_idx].keys()
                            else None
                        )
                        cosine = cosine_similarity(query_blob, cand_blob)
                        score = hybrid_score(
                            rank_idx, len(entries), cosine,
                            bm25_weight=self.HYBRID_BM25_WEIGHT,
                        )
                        scored.append((score, e))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    entries = [e for _, e in scored]

            entries = entries[:max_results]
            self._bump_usage(conn, (e.id for e in entries))
            # Re-fetch usage_count for accurate return.
            for entry in entries:
                entry.usage_count += 1
        finally:
            conn.close()
        return entries

    def _add(
        self, category: str, tags: list[str] | str,
        problem_summary: str, explanation: str,
        trace_id: str,
        solution_code: str | None = None,
        solution_pattern: str | None = None,
        base_sha: str | None = None,
        qwen_plan_recipe: str | None = None,
        qwen_plan_agreement: float | None = None,
    ) -> int:
        if category not in CATEGORIES:
            raise ValueError(
                f"category must be in {CATEGORIES}, got {category!r}"
            )
        tags_str = self._normalize_tags(tags)
        now = _utcnow_iso()
        # Phase 16 token-leak fix (2026-05-06): scrub credentials from
        # every text column BEFORE persistence. Caught after Qwen's
        # shadow plans started writing `.env.bot` edits with literal
        # token values into solution_code / qwen_plan_recipe, which then
        # contaminated future few-shot retrievals. Conservative env-var
        # patterns only (Q1=A). Pure-function, no I/O, never raises.
        from core.secrets_scrub import scrub as _scrub_secrets
        problem_summary = _scrub_secrets(problem_summary) or ""
        explanation = _scrub_secrets(explanation) or ""
        if solution_code is not None:
            solution_code = _scrub_secrets(solution_code)
        if solution_pattern is not None:
            solution_pattern = _scrub_secrets(solution_pattern)
        if qwen_plan_recipe is not None:
            qwen_plan_recipe = _scrub_secrets(qwen_plan_recipe)
        # Pre-Phase-15: compute embedding from the searchable surface
        # (tags + summary + pattern). Failure returns None and the row
        # is still written -- search just falls back to FTS5-only for
        # this row until a backfill picks it up.
        from core.embeddings import embed_text
        embed_text_input = " ".join(filter(None, [
            tags_str, problem_summary, solution_pattern or "",
        ]))
        embedding = embed_text(embed_text_input, trace_id)
        # Phase 15b -- read provenance off the ContextVar. No
        # threading through call sites; the contextvar carries.
        from core.write_origin import get_current_write_origin
        origin = get_current_write_origin()
        # Phase 15c -- clamp the agreement score defensively (the
        # caller already does this but a NULL/garbage value here
        # would cascade through stats math). None passes through
        # untouched so "no shadow data" stays distinguishable from
        # "shadow ran and scored 0.0".
        if qwen_plan_agreement is not None:
            try:
                qwen_plan_agreement = max(
                    0.0, min(1.0, float(qwen_plan_agreement)),
                )
            except (TypeError, ValueError):
                qwen_plan_agreement = None
        conn = _connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO knowledge (
                    category, tags, problem_summary, solution_code,
                    solution_pattern, explanation, source_trace_id,
                    created_at, usage_count, base_sha, embedding,
                    created_by_origin, qwen_plan_recipe,
                    qwen_plan_agreement
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                """,
                (category, tags_str, problem_summary, solution_code,
                 solution_pattern, explanation, trace_id, now,
                 base_sha, embedding, origin,
                 qwen_plan_recipe, qwen_plan_agreement),
            )
            new_id = cursor.lastrowid
        finally:
            conn.close()
        log_event(
            trace_id, "INFO", "knowledge_base",
            f"added {category} id={new_id} tags={tags_str!r} "
            f"summary={problem_summary[:80]!r}",
        )
        # Auto-prune if over cap.
        if self._count() > config.KNOWLEDGE_MAX_ENTRIES:
            self.prune(config.KNOWLEDGE_MAX_ENTRIES)
        return new_id

    def find_active_pattern_by_problem(
        self, problem_summary: str,
    ) -> int | None:
        """Phase 16 Option A (2026-05-06): exact problem_summary match
        against active patterns. Used by ``add_pattern`` to dedup
        repeated successful /code attempts on the same prompt.

        Scope by design:
          - category='pattern' only (not limitation -- failure rows
            represent distinct fail modes, dedup'ing them would lose
            information)
          - state='active' only (archived rows stay archived; revival
            requires explicit /kb restore <id>)

        Match is byte-exact on the scrubbed-form of the prompt (matches
        the same scrub the storage path applies in ``_add``). Returns
        the most recently created matching id, or None.
        """
        from core.secrets_scrub import scrub as _scrub_secrets
        normalized = _scrub_secrets(problem_summary) or ""
        conn = _connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT id FROM knowledge "
                "WHERE category = 'pattern' "
                "AND state = 'active' "
                "AND problem_summary = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (normalized,),
            ).fetchone()
        finally:
            conn.close()
        return row["id"] if row else None

    def _update_pattern_in_place(
        self,
        pattern_id: int,
        tags: list[str] | str,
        problem_summary: str,
        solution_code: str,
        solution_pattern: str,
        explanation: str,
        trace_id: str,
        base_sha: str | None,
        qwen_plan_recipe: str | None,
        qwen_plan_agreement: float | None,
    ) -> None:
        """Phase 16 Option A: update an existing active pattern's
        recipe/diff/embedding fields in place when ``add_pattern``
        dedups against it. Counters (solo_attempts/passes), pinned,
        needs_reteach, state, created_at, created_by_origin, and
        usage_count are NOT touched -- those are owned by other code
        paths (graduation, /kb pin, lifecycle walker, retrieval).

        The stored recipe is REPLACED with the new attempt's recipe
        because each /code may target a different file state (the
        emoji-bar prompt picks a new emoji each run); the LATEST
        verified-good recipe is more useful as a few-shot example
        than the original. base_sha is also refreshed so the next
        graduation replays against the most recent known-clean state."""
        from core.embeddings import embed_text
        from core.secrets_scrub import scrub as _scrub_secrets
        tags_str = self._normalize_tags(tags)
        problem_summary = _scrub_secrets(problem_summary) or ""
        explanation = _scrub_secrets(explanation) or ""
        solution_code = _scrub_secrets(solution_code)
        solution_pattern = _scrub_secrets(solution_pattern)
        if qwen_plan_recipe is not None:
            qwen_plan_recipe = _scrub_secrets(qwen_plan_recipe)
        if qwen_plan_agreement is not None:
            try:
                qwen_plan_agreement = max(
                    0.0, min(1.0, float(qwen_plan_agreement)),
                )
            except (TypeError, ValueError):
                qwen_plan_agreement = None
        # Re-embed because solution_pattern (the searchable surface)
        # changed. embed_text is best-effort; on failure embedding
        # stays None and search falls back to FTS5.
        embed_input = " ".join(filter(None, [
            tags_str, problem_summary, solution_pattern or "",
        ]))
        embedding = embed_text(embed_input, trace_id)
        conn = _connect(self.db_path)
        try:
            conn.execute(
                "UPDATE knowledge SET "
                "tags = ?, "
                "solution_code = ?, "
                "solution_pattern = ?, "
                "explanation = ?, "
                "base_sha = COALESCE(?, base_sha), "
                "qwen_plan_recipe = ?, "
                "qwen_plan_agreement = ?, "
                "embedding = ? "
                "WHERE id = ?",
                (tags_str, solution_code, solution_pattern, explanation,
                 base_sha, qwen_plan_recipe, qwen_plan_agreement,
                 embedding, pattern_id),
            )
            conn.commit()
        finally:
            conn.close()

    def add_pattern(
        self, tags: list[str], problem_summary: str,
        solution_code: str, solution_pattern: str,
        explanation: str, trace_id: str,
        base_sha: str | None = None,
        qwen_plan_recipe: str | None = None,
        qwen_plan_agreement: float | None = None,
    ) -> int:
        """Phase 14b: ``base_sha`` is the git SHA from BEFORE this
        pattern's /code attempt ran. Lets graduation reset the tree
        to a known-clean state and replay the recipe through the
        full agentic pipeline. Optional for back-compat with old
        patterns; when absent, graduation falls back to text-gen.

        Phase 15c: ``qwen_plan_recipe`` + ``qwen_plan_agreement``
        record what Qwen would have planned for this same problem
        (shadow mode, run in parallel with Claude's pre-teach) and
        the structural agreement score Claude vs Qwen got. Both
        optional and default to NULL -- /code attempts where the
        shadow call timed out / crashed leave them None.

        Phase 16 Option A (2026-05-06): if an active pattern with the
        same problem_summary already exists, this UPDATEs it in place
        and returns the existing id rather than inserting a near-
        duplicate. The subsequent graduation step (called by /code's
        agentic pipeline) increments the existing pattern's
        solo_attempts/solo_passes counters, eventually crossing the
        Batch D auto-pin threshold (5/5). Without this dedup, every
        /code on a recurring prompt creates a fresh row whose counter
        is permanently capped at 1, making auto-pin (and future Batch
        C skip-eligibility) mathematically unreachable."""
        existing_id = self.find_active_pattern_by_problem(problem_summary)
        if existing_id is not None:
            self._update_pattern_in_place(
                existing_id, tags, problem_summary, solution_code,
                solution_pattern, explanation, trace_id, base_sha,
                qwen_plan_recipe, qwen_plan_agreement,
            )
            log_event(
                trace_id, "INFO", "knowledge_base",
                f"DEDUP pattern_id={existing_id}: same problem as "
                f"existing active pattern; recipe/diff/embedding "
                f"refreshed. Counter will increment via graduation.",
            )
            return existing_id
        return self._add(
            "pattern", tags, problem_summary, explanation, trace_id,
            solution_code=solution_code,
            solution_pattern=solution_pattern,
            base_sha=base_sha,
            qwen_plan_recipe=qwen_plan_recipe,
            qwen_plan_agreement=qwen_plan_agreement,
        )

    def add_limitation(
        self, tags: list[str], problem_summary: str,
        explanation: str, trace_id: str,
        qwen_plan_recipe: str | None = None,
        qwen_plan_agreement: float | None = None,
    ) -> int:
        """Phase 15d: limitations now also carry shadow planning data
        when available. The signal is arguably *more* interesting on
        a limitation than on a pattern -- "could Qwen have planned
        this hopeless task?" tells us whether the failure was a
        Claude-quality issue or a Qwen-capacity issue. Both kwargs
        default to None (NULL on the row) for back-compat with old
        call sites + paths that didn't capture shadow data."""
        return self._add(
            "limitation", tags, problem_summary, explanation, trace_id,
            qwen_plan_recipe=qwen_plan_recipe,
            qwen_plan_agreement=qwen_plan_agreement,
        )

    def get_context_for_prompt(
        self, query: str, max_chars: int | None = None,
        exclude_pattern_ids: list[int] | tuple[int, ...] | None = None,
    ) -> str:
        """Phase 16 Batch C: ``exclude_pattern_ids`` is forwarded to
        ``search`` so the few-shot block built for Claude's pre-teach
        omits patterns the caller wants withheld (e.g. one that just
        failed skip-replay)."""
        if max_chars is None:
            max_chars = config.KNOWLEDGE_CONTEXT_MAX_CHARS
        entries = self.search(
            query, max_results=5,
            exclude_pattern_ids=exclude_pattern_ids,
        )
        if not entries:
            return ""
        chunks: list[str] = []
        running = 0
        for e in entries:
            block = (
                f"--- KNOWN PATTERN: {e.problem_summary} ---\n"
                f"Tags: {e.tags}\n"
                f"Solution approach: {e.solution_pattern or '(n/a)'}\n"
                f"Code:\n{e.solution_code or '(no code -- limitation)'}\n"
                f"Why it works: {e.explanation}\n"
                f"---\n"
            )
            if running + len(block) > max_chars:
                if not chunks:
                    # First block alone exceeds the cap -- truncate.
                    chunks.append(block[: max_chars])
                    running = max_chars
                break
            chunks.append(block)
            running += len(block)
        return "".join(chunks)

    def _count(self) -> int:
        conn = _connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge"
            ).fetchone()
        finally:
            conn.close()
        return int(row["n"])

    def prune(self, max_entries: int | None = None) -> int:
        """Phase 15a: prune ARCHIVES the lowest-usage non-archived,
        non-pinned rows -- it never deletes. The caller still gets a
        count back (number of rows archived). Pinned rows are skipped
        regardless of usage_count. Already-archived rows don't count
        against the active-row cap and are not re-archived."""
        if max_entries is None:
            max_entries = config.KNOWLEDGE_MAX_ENTRIES
        conn = _connect(self.db_path)
        try:
            active_count = int(conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge "
                "WHERE state != 'archived'"
            ).fetchone()["n"])
            if active_count <= max_entries:
                return 0
            to_archive = active_count - max_entries
            ids = [
                r["id"] for r in conn.execute(
                    """
                    SELECT id FROM knowledge
                    WHERE state != 'archived' AND pinned = 0
                    ORDER BY usage_count ASC, created_at ASC
                    LIMIT ?
                    """,
                    (to_archive,),
                ).fetchall()
            ]
            if not ids:
                return 0
            now = _utcnow_iso()
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE knowledge SET state = 'archived', "
                f"archived_at = ? WHERE id IN ({placeholders})",
                [now, *ids],
            )
        finally:
            conn.close()
        log_event(
            "SEN-system", "INFO", "knowledge_base",
            f"archived {len(ids)} entries (kept {max_entries} active); "
            f"ids={ids[:20]}{'...' if len(ids) > 20 else ''}",
        )
        return len(ids)

    # ─────────────────────────────────────────────────────────────
    # Phase 15a -- lifecycle (pin / unpin / restore / auto-transition)
    # ─────────────────────────────────────────────────────────────

    def pin_pattern(self, pattern_id: int) -> bool:
        """Mark a pattern pinned. Pinned rows are immune to all
        automatic transitions (prune skips them, auto-transition
        skips them). Idempotent. Returns True if the row exists."""
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE knowledge SET pinned = 1 WHERE id = ?",
                (pattern_id,),
            )
            ok = cur.rowcount > 0
        finally:
            conn.close()
        if ok:
            log_event(
                "SEN-system", "INFO", "knowledge_base",
                f"pinned pattern_id={pattern_id}",
            )
        return ok

    def unpin_pattern(self, pattern_id: int) -> bool:
        """Clear the pinned flag. Idempotent. Does NOT change state
        (an archived pin stays archived; an active pin stays active).
        Returns True if the row exists."""
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE knowledge SET pinned = 0 WHERE id = ?",
                (pattern_id,),
            )
            ok = cur.rowcount > 0
        finally:
            conn.close()
        if ok:
            log_event(
                "SEN-system", "INFO", "knowledge_base",
                f"unpinned pattern_id={pattern_id}",
            )
        return ok

    def restore_pattern(self, pattern_id: int) -> bool:
        """Bring an archived row back to state='active' and clear
        archived_at. The inverse of prune-time auto-archival; the
        Telegram /kb restore command + future curation flows call
        this. Returns True if the row exists."""
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE knowledge SET state = 'active', "
                "archived_at = NULL WHERE id = ?",
                (pattern_id,),
            )
            ok = cur.rowcount > 0
        finally:
            conn.close()
        if ok:
            log_event(
                "SEN-system", "INFO", "knowledge_base",
                f"restored pattern_id={pattern_id} -> active",
            )
        return ok

    # ─────────────────────────────────────────────────────────────────
    # Phase 16 Batch C -- skip-eligibility gate
    # ─────────────────────────────────────────────────────────────────

    SKIP_REASON_OK_PINNED = "OK_PINNED"
    SKIP_REASON_OK_TRUSTED = "OK_TRUSTED"
    SKIP_REASON_MISSING = "MISSING"
    SKIP_REASON_NOT_PATTERN = "LIMITATION"
    SKIP_REASON_ARCHIVED = "ARCHIVED"
    SKIP_REASON_NEEDS_RETEACH = "NEEDS_RETEACH"
    SKIP_REASON_LOW_PASSES = "LOW_PASSES"
    SKIP_REASON_IMPERFECT_RATE = "IMPERFECT_RATE"
    SKIP_REASON_STALE = "STALE"
    SKIP_REASON_LOW_AGREEMENT = "LOW_AGREEMENT"

    def is_skip_eligible(
        self,
        pattern_id: int | None = None,
        *,
        row: dict | None = None,
    ) -> tuple[bool, str]:
        """Skip-eligibility gate. Returns (eligible, reason_token).
        Reason tokens are class constants for stable log/grep filters.

        Eligibility:
          * exists, category='pattern', state='active', not needs_reteach
          * (pinned=1) OR (
                solo_passes >= config.SKIP_PATH_MIN_PASSES
                AND solo_attempts == solo_passes
                AND last_verified_at within FRESHNESS_DAYS
                AND (qwen_plan_agreement IS NULL
                     OR qwen_plan_agreement >= AGREEMENT_FLOOR)
            )
        """
        if row is None:
            if pattern_id is None:
                return False, self.SKIP_REASON_MISSING
            entry = self.get_pattern(pattern_id)
            if entry is None:
                return False, self.SKIP_REASON_MISSING
            row = {
                "id": entry.id,
                "category": entry.category,
                "state": entry.state,
                "pinned": entry.pinned,
                "needs_reteach": entry.needs_reteach,
                "solo_attempts": entry.solo_attempts,
                "solo_passes": entry.solo_passes,
                "last_verified_at": entry.last_verified_at,
                "qwen_plan_agreement": entry.qwen_plan_agreement,
            }
        if (row.get("category") or "").lower() != "pattern":
            return False, self.SKIP_REASON_NOT_PATTERN
        if (row.get("state") or "active") != "active":
            return False, self.SKIP_REASON_ARCHIVED
        if int(row.get("needs_reteach") or 0) == 1:
            return False, self.SKIP_REASON_NEEDS_RETEACH
        if int(row.get("pinned") or 0) == 1:
            return True, self.SKIP_REASON_OK_PINNED
        passes = int(row.get("solo_passes") or 0)
        attempts = int(row.get("solo_attempts") or 0)
        if passes < config.SKIP_PATH_MIN_PASSES:
            return False, self.SKIP_REASON_LOW_PASSES
        if passes != attempts:
            return False, self.SKIP_REASON_IMPERFECT_RATE
        last_verified = row.get("last_verified_at")
        if not last_verified:
            return False, self.SKIP_REASON_STALE
        if not self._within_freshness_window(
            last_verified, config.SKIP_PATH_FRESHNESS_DAYS,
        ):
            return False, self.SKIP_REASON_STALE
        agreement = row.get("qwen_plan_agreement")
        if agreement is not None:
            try:
                af = float(agreement)
            except (TypeError, ValueError):
                af = 0.0
            if af < config.SKIP_PATH_AGREEMENT_FLOOR:
                return False, self.SKIP_REASON_LOW_AGREEMENT
        return True, self.SKIP_REASON_OK_TRUSTED

    @staticmethod
    def _within_freshness_window(
        last_verified_at: str, days: int,
    ) -> bool:
        from datetime import datetime, timedelta, timezone
        try:
            ts = last_verified_at.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return dt >= cutoff

    def auto_transition_lifecycle(
        self,
        stale_after_days: int | None = None,
        archive_after_days: int | None = None,
    ) -> dict:
        """Phase 15a nightly walker. Active rows older than
        ``stale_after_days`` with low usage transition to 'stale';
        stale rows older than ``archive_after_days`` transition to
        'archived' with archived_at stamped. Pinned rows are NEVER
        touched. Returns ``{'stale': N, 'archived': M}`` for the
        Telegram /dashboard or scheduler logging.

        ``low usage`` is defined as usage_count <= 1 -- a pattern
        that's been retrieved at most once. Heavy usage means the
        pattern is still earning its keep regardless of age.
        """
        from datetime import datetime, timedelta, timezone
        if stale_after_days is None:
            stale_after_days = getattr(
                config, "KB_STALE_AFTER_DAYS", 30,
            )
        if archive_after_days is None:
            archive_after_days = getattr(
                config, "KB_ARCHIVE_AFTER_DAYS", 90,
            )
        now = datetime.now(timezone.utc)
        stale_cutoff = (
            now - timedelta(days=max(1, int(stale_after_days)))
        ).isoformat()
        archive_cutoff = (
            now - timedelta(days=max(1, int(archive_after_days)))
        ).isoformat()
        conn = _connect(self.db_path)
        try:
            cur1 = conn.execute(
                "UPDATE knowledge SET state = 'stale' "
                "WHERE state = 'active' AND pinned = 0 "
                "AND created_at < ? AND usage_count <= 1",
                (stale_cutoff,),
            )
            n_stale = cur1.rowcount
            cur2 = conn.execute(
                "UPDATE knowledge SET state = 'archived', "
                "archived_at = ? "
                "WHERE state = 'stale' AND pinned = 0 "
                "AND created_at < ?",
                (now.isoformat(), archive_cutoff),
            )
            n_archived = cur2.rowcount
        finally:
            conn.close()
        if n_stale or n_archived:
            log_event(
                "SEN-system", "INFO", "knowledge_base",
                f"auto_transition_lifecycle: stale={n_stale} "
                f"archived={n_archived} "
                f"(stale_after={stale_after_days}d, "
                f"archive_after={archive_after_days}d)",
            )
        return {"stale": n_stale, "archived": n_archived}

    # ─────────────────────────────────────────────────────────────
    # Pre-Phase-15 -- embedding backfill for legacy rows
    # ─────────────────────────────────────────────────────────────

    def backfill_embeddings(
        self, batch_size: int = 50, trace_id: str = "SEN-system",
    ) -> dict:
        """Compute embeddings for any rows where embedding IS NULL.
        Run on startup once, or on demand. Cheap (~50ms per row) but
        blocks on Ollama availability so call from a background thread
        for big batches.

        Returns a counts dict: {scanned, embedded, failed}.
        """
        from core.embeddings import embed_text
        scanned = embedded = failed = 0
        conn = _connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT id, tags, problem_summary, solution_pattern "
                "FROM knowledge WHERE embedding IS NULL LIMIT ?",
                (batch_size,),
            ).fetchall()
            for r in rows:
                scanned += 1
                blob = embed_text(" ".join(filter(None, [
                    r["tags"], r["problem_summary"],
                    r["solution_pattern"] or "",
                ])), trace_id)
                if blob is None:
                    failed += 1
                    continue
                conn.execute(
                    "UPDATE knowledge SET embedding = ? WHERE id = ?",
                    (blob, r["id"]),
                )
                embedded += 1
        finally:
            conn.close()
        if scanned:
            log_event(
                trace_id, "INFO", "knowledge_base",
                f"backfill_embeddings: scanned={scanned} "
                f"embedded={embedded} failed={failed}",
            )
        return {"scanned": scanned, "embedded": embedded, "failed": failed}

    # ─────────────────────────────────────────────────────────────
    # Phase 14a -- graduation test API
    # ─────────────────────────────────────────────────────────────

    # When solo_attempts >= MIN_TRIES and pass_rate < FAIL_THRESHOLD,
    # the pattern is auto-flagged needs_reteach=1.
    GRAD_MIN_TRIES = 3
    GRAD_FAIL_THRESHOLD = 0.5

    # Phase 16 Batch D -- auto-pin proven graduators.
    # When solo_passes >= MIN_PASSES AND pass_rate == 1.0 (zero
    # failures), the pattern auto-pins so the kb_lifecycle walker
    # can never archive it. STRICT thresholds chosen for safety:
    # pinning is permanent foundation, want zero ambiguity. A
    # pattern that's failed even once doesn't qualify -- no
    # 4/5=0.8 partial trust. Reversible via /kb unpin <id> if
    # needed.
    AUTO_PIN_MIN_PASSES = 5
    AUTO_PIN_REQUIRED_RATE = 1.0

    def get_pattern(self, pattern_id: int) -> KnowledgeEntry | None:
        """Single-row lookup by id. Used by the graduation runner +
        manual `/kb verify <id>`. Indexed (PRIMARY KEY)."""
        conn = _connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM knowledge WHERE id = ?", (pattern_id,),
            ).fetchone()
        finally:
            conn.close()
        return _row_to_entry(row) if row else None

    def record_solo_attempt(
        self, pattern_id: int, passed: bool, trace_id: str,
    ) -> tuple[int, int, bool]:
        """Increment solo_attempts (always) and solo_passes (if passed),
        stamp last_verified_at, and -- if attempts >= MIN_TRIES -- flip
        needs_reteach to 1 when pass_rate < FAIL_THRESHOLD.

        Phase 16 Batch D: ALSO auto-pin the pattern when it crosses
        the trust threshold (solo_passes >= AUTO_PIN_MIN_PASSES AND
        solo_pass_rate == AUTO_PIN_REQUIRED_RATE). Atomic with the
        UPDATE so a row that just earned its 5th pass at 100% rate
        gets pinned in the same transaction. Idempotent on already-
        pinned rows (no double-log, no extra UPDATE).

        Returns ``(solo_attempts, solo_passes, needs_reteach_after)`` so
        the caller can decide whether to surface a Telegram notification.
        Returns ``(0, 0, False)`` if the pattern doesn't exist.
        """
        now = _utcnow_iso()
        conn = _connect(self.db_path)
        auto_pinned = False
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT solo_attempts, solo_passes, pinned "
                "FROM knowledge WHERE id = ?",
                (pattern_id,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return (0, 0, False)
            new_attempts = int(row["solo_attempts"]) + 1
            new_passes = int(row["solo_passes"]) + (1 if passed else 0)
            # Decide whether to flip needs_reteach. Only flip ONCE we
            # have enough data; once flipped we DON'T auto-unflip --
            # the caller (or `/kb verify`) explicitly clears it.
            new_flag = 0
            if new_attempts >= self.GRAD_MIN_TRIES:
                pass_rate = new_passes / new_attempts
                if pass_rate < self.GRAD_FAIL_THRESHOLD:
                    new_flag = 1
            # Phase 16 Batch D: auto-pin proven graduators.
            # Threshold: solo_passes >= 5 AND solo_pass_rate == 1.0
            # (i.e. new_passes == new_attempts -- zero failures).
            # Skip if already pinned to avoid double-log churn.
            already_pinned = bool(row["pinned"])
            should_auto_pin = (
                not already_pinned
                and new_passes >= self.AUTO_PIN_MIN_PASSES
                and new_passes == new_attempts
            )
            if should_auto_pin:
                conn.execute(
                    "UPDATE knowledge SET "
                    "solo_attempts = ?, solo_passes = ?, "
                    "last_verified_at = ?, "
                    "needs_reteach = MAX(needs_reteach, ?), "
                    "pinned = 1 "
                    "WHERE id = ?",
                    (new_attempts, new_passes, now, new_flag, pattern_id),
                )
                auto_pinned = True
            else:
                conn.execute(
                    "UPDATE knowledge SET "
                    "solo_attempts = ?, solo_passes = ?, "
                    "last_verified_at = ?, "
                    "needs_reteach = MAX(needs_reteach, ?) "
                    "WHERE id = ?",
                    (new_attempts, new_passes, now, new_flag, pattern_id),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        log_event(
            trace_id, "INFO", "knowledge_base",
            f"graduation pattern_id={pattern_id} passed={passed} "
            f"-> {new_passes}/{new_attempts} "
            f"needs_reteach={bool(new_flag)}",
        )
        if auto_pinned:
            log_event(
                trace_id, "INFO", "knowledge_base",
                f"AUTO-PIN pattern_id={pattern_id} "
                f"({new_passes}/{new_attempts} solo passes, "
                f"perfect rate) -- now permanent foundation, "
                f"immune to kb_lifecycle archival",
            )
        return (new_attempts, new_passes, bool(new_flag))

    def clear_needs_reteach(self, pattern_id: int) -> bool:
        """Manual reset (e.g. after the user re-teaches a flagged
        pattern). Indexed; cheap. Returns True if a row was updated."""
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE knowledge SET needs_reteach = 0 WHERE id = ?",
                (pattern_id,),
            )
            return cur.rowcount > 0
        finally:
            conn.close()

    def list_needs_reteach(self, limit: int = 50) -> list[KnowledgeEntry]:
        """Patterns currently flagged needs_reteach=1. Uses the partial
        index `idx_knowledge_needs_reteach`."""
        conn = _connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM knowledge WHERE needs_reteach = 1 "
                "ORDER BY last_verified_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [_row_to_entry(r) for r in rows]

    def list_stale(
        self, days: int = 30, limit: int = 50,
    ) -> list[KnowledgeEntry]:
        """Patterns whose last_verified_at is older than ``days``.
        NEVER-verified rows (last_verified_at IS NULL) are included,
        so this also surfaces fresh patterns that haven't graduated
        yet. Uses the partial index `idx_knowledge_last_verified` for
        the date-range half; the NULL half is a small post-filter."""
        from datetime import datetime, timedelta, timezone
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
        ).isoformat()
        conn = _connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT * FROM knowledge
                 WHERE category = 'pattern'
                   AND (last_verified_at IS NULL
                        OR last_verified_at < ?)
                 ORDER BY (last_verified_at IS NULL) DESC,
                          last_verified_at ASC
                 LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
        finally:
            conn.close()
        return [_row_to_entry(r) for r in rows]

    def graduation_stats(self) -> dict:
        """Aggregate counts for `/kb` dashboard. One indexed COUNT per
        bucket -- no full table scans."""
        conn = _connect(self.db_path)
        try:
            total_patterns = int(conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge "
                "WHERE category = 'pattern'"
            ).fetchone()["n"])
            verified = int(conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge "
                "WHERE category = 'pattern' AND solo_attempts > 0"
            ).fetchone()["n"])
            # needs_reteach uses the partial index (fast even if total
            # grows to the 500-row cap).
            needs_reteach = int(conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge "
                "WHERE needs_reteach = 1"
            ).fetchone()["n"])
            agg = conn.execute(
                "SELECT SUM(solo_attempts) AS a, SUM(solo_passes) AS p "
                "FROM knowledge WHERE category = 'pattern'"
            ).fetchone()
            attempts = int(agg["a"] or 0)
            passes = int(agg["p"] or 0)
        finally:
            conn.close()
        rate = (passes / attempts) if attempts else 0.0
        return {
            "total_patterns": total_patterns,
            "verified_patterns": verified,
            "needs_reteach": needs_reteach,
            "solo_attempts": attempts,
            "solo_passes": passes,
            "solo_pass_rate": round(rate, 3),
        }

    def cleanup_low_quality_patterns(self) -> int:
        """Phase 15a alignment: ARCHIVE patterns whose solution_code
        fails the quality gate, never DELETE.

        Pre-15d-bugfix this method actually deleted rows on every bot
        startup -- and combined with the Phase 14b change that put
        diff TEXT into solution_code (instead of diff stat), the gate
        misclassified every successful /code teach as low-quality and
        silently destroyed 8 KB patterns over several days. Even with
        the gate now recognising diff text, archiving is the safer
        posture: a misjudgement is recoverable via /kb restore <id>;
        a delete is not. Skips already-archived rows so this is
        cheap to re-run.

        Already-active rows that fail the gate move to state='archived'
        with archived_at stamped (same shape as auto_transition_lifecycle's
        late stage). Limitation rows are untouched (they intentionally
        have no solution_code)."""
        try:
            from skills.code_assist import _is_real_solution
        except Exception:
            return 0
        conn = _connect(self.db_path)
        archived_ids: list[int] = []
        try:
            rows = conn.execute(
                "SELECT id, solution_code FROM knowledge "
                "WHERE category = 'pattern' "
                "AND state != 'archived'"
            ).fetchall()
            bad_ids = [
                r["id"] for r in rows
                if not _is_real_solution(r["solution_code"] or "")
            ]
            if bad_ids:
                now = _utcnow_iso()
                placeholders = ",".join("?" * len(bad_ids))
                conn.execute(
                    f"UPDATE knowledge SET state = 'archived', "
                    f"archived_at = ? WHERE id IN ({placeholders})",
                    [now, *bad_ids],
                )
                archived_ids = bad_ids
        finally:
            conn.close()
        if archived_ids:
            log_event(
                "SEN-system", "WARNING", "knowledge_base",
                f"archived {len(archived_ids)} low-quality pattern "
                f"entries (failed quality gate, NOT deleted -- "
                f"recoverable via /kb restore <id>); ids="
                f"{archived_ids[:20]}"
                f"{'...' if len(archived_ids) > 20 else ''}",
            )
        return len(archived_ids)

    def planning_stats(self) -> dict:
        """Phase 15c -- shadow-planning aggregates for /kb planning.

        Returns:
          {
            'patterns_with_shadow':  int,
            'patterns_total':        int,  (pattern category only)
            'mean_agreement':        float | None,
            'p25': float | None, 'p50': float | None, 'p75': float | None,
            'by_archetype':         [{tag, n, mean_agreement}, ...],
          }

        Pure SQLite aggregates -- no Python-side scan even at 50K
        cap. Percentiles use NTILE for cheap bucketing; output is
        the bucket boundary, not a textbook quantile, but it's the
        right shape for the Telegram readout.
        """
        conn = _connect(self.db_path)
        try:
            tot = int(conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge "
                "WHERE category = 'pattern'"
            ).fetchone()["n"])
            shadow_n = int(conn.execute(
                "SELECT COUNT(*) AS n FROM knowledge "
                "WHERE category = 'pattern' "
                "AND qwen_plan_agreement IS NOT NULL"
            ).fetchone()["n"])
            mean = None
            p25 = p50 = p75 = None
            if shadow_n > 0:
                mean_row = conn.execute(
                    "SELECT AVG(qwen_plan_agreement) AS m "
                    "FROM knowledge WHERE category = 'pattern' "
                    "AND qwen_plan_agreement IS NOT NULL"
                ).fetchone()
                mean = (
                    float(mean_row["m"]) if mean_row["m"] is not None
                    else None
                )
                # Sorted ascending -- pick at fractional positions.
                rows = conn.execute(
                    "SELECT qwen_plan_agreement AS s FROM knowledge "
                    "WHERE category = 'pattern' "
                    "AND qwen_plan_agreement IS NOT NULL "
                    "ORDER BY qwen_plan_agreement ASC"
                ).fetchall()
                vals = [float(r["s"]) for r in rows]
                if vals:
                    def _at(frac: float) -> float:
                        idx = max(
                            0, min(
                                len(vals) - 1,
                                int(round(frac * (len(vals) - 1))),
                            ),
                        )
                        return vals[idx]
                    p25, p50, p75 = _at(0.25), _at(0.50), _at(0.75)
            # Per-tag rollup. Each pattern row has a comma-separated
            # tag list; we GROUP BY the FIRST tag (an archetype-ish
            # bucket -- we don't try to be clever about multi-tag
            # rows) so the readout stays compact at the cost of some
            # under-counting on multi-archetype patterns.
            tag_rows = conn.execute(
                """
                SELECT
                  CASE
                    WHEN instr(tags, ',') > 0
                      THEN substr(tags, 1, instr(tags, ',') - 1)
                    ELSE tags
                  END AS first_tag,
                  COUNT(*) AS n,
                  AVG(qwen_plan_agreement) AS mean_agreement
                FROM knowledge
                WHERE category = 'pattern'
                  AND qwen_plan_agreement IS NOT NULL
                GROUP BY first_tag
                ORDER BY n DESC
                """
            ).fetchall()
            by_archetype = [
                {
                    "tag": (r["first_tag"] or "(no tag)"),
                    "n": int(r["n"]),
                    "mean_agreement": (
                        float(r["mean_agreement"])
                        if r["mean_agreement"] is not None else None
                    ),
                }
                for r in tag_rows
            ]
        finally:
            conn.close()
        return {
            "patterns_total": tot,
            "patterns_with_shadow": shadow_n,
            "mean_agreement": mean,
            "p25": p25, "p50": p50, "p75": p75,
            "by_archetype": by_archetype,
        }

    def origin_breakdown(self) -> dict[str, int]:
        """Phase 15b -- count rows per ``created_by_origin``. Single
        GROUP BY query, not three separate counts. Output is a plain
        dict (origin -> count). Empty dict on a fresh DB."""
        conn = _connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT created_by_origin AS origin, "
                "COUNT(*) AS n FROM knowledge "
                "GROUP BY created_by_origin "
                "ORDER BY n DESC"
            ).fetchall()
        finally:
            conn.close()
        return {r["origin"] or "foreground": int(r["n"]) for r in rows}

    def stats(self) -> dict:
        conn = _connect(self.db_path)
        try:
            total = conn.execute(
                "SELECT COUNT(*) n FROM knowledge"
            ).fetchone()["n"]
            patterns = conn.execute(
                "SELECT COUNT(*) n FROM knowledge "
                "WHERE category = 'pattern'"
            ).fetchone()["n"]
            lims = conn.execute(
                "SELECT COUNT(*) n FROM knowledge "
                "WHERE category = 'limitation'"
            ).fetchone()["n"]
            avg_row = conn.execute(
                "SELECT AVG(usage_count) avg FROM knowledge"
            ).fetchone()
            avg = float(avg_row["avg"]) if avg_row["avg"] is not None \
                else 0.0
        finally:
            conn.close()
        return {
            "total_entries": int(total),
            "patterns_count": int(patterns),
            "limitations_count": int(lims),
            "avg_usage_count": round(avg, 2),
        }
