"""Three-tier persistent memory.

Tier 1 (working): in-process per-session ring buffer (deque). NOT persisted.
Tier 2 (episodic): SQLite + FTS5 over (summary, detail, tags). Per-agent
    scope ("global" or an agent name). Decay + prune.
Tier 3 (semantic): SQLite + FTS5 over (key, value). Global facts, upsert
    by key with confidence-aware merge.

All persistence in ``memory.db`` -- separate from ``sentinel.db``
(task queue) and ``knowledge.db`` (code-pattern KB). Don't merge:
different lifecycles + the FTS triggers would cross-contaminate.
"""
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core import config
from core.logger import log_event


# -----------------------------------------------------------------
# helpers
# -----------------------------------------------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA cache_size = -64000;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA mmap_size = 268435456;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def _escape_fts_query(query: str) -> str:
    """FTS5 treats some chars specially. Wrap each whitespace-delimited
    token in double quotes so quotes/dashes/etc. in user input don't
    blow up the query.
    """
    tokens = [t for t in query.replace('"', " ").split() if t]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


def _normalize_tags(tags: list[str] | str | None) -> str:
    if tags is None:
        return ""
    if isinstance(tags, str):
        parts = [t.strip() for t in tags.split(",") if t.strip()]
    else:
        parts = [str(t).strip() for t in tags if t and str(t).strip()]
    return ",".join(parts)


# -----------------------------------------------------------------
# pydantic rows
# -----------------------------------------------------------------

class Episode(BaseModel):
    id: int
    scope: str
    trace_id: str
    event_type: str
    summary: str
    detail: str
    tags: str
    created_at: str
    relevance_score: float
    # Phase 15a: lifecycle states. Defaults make this back-compat for
    # rows persisted before the migration (Episode(**dict(r)) just
    # doesn't see the keys and falls through to defaults).
    state: str = "active"
    pinned: bool = False
    archived_at: str | None = None
    created_by_origin: str = "foreground"


class Fact(BaseModel):
    id: int
    category: str
    key: str
    value: str
    source: str
    confidence: float
    created_at: str
    updated_at: str
    state: str = "active"
    pinned: bool = False
    archived_at: str | None = None
    created_by_origin: str = "foreground"


# -----------------------------------------------------------------
# tier 1 -- working memory
# -----------------------------------------------------------------

class WorkingMemory:
    """In-process per-session ring buffer. Reset on bot restart.
    Good for short-term Telegram chat continuity.
    """

    def __init__(self, max_messages: int | None = None) -> None:
        self.max = max_messages or config.WORKING_MEMORY_MAX_MESSAGES
        self._buf: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.max),
        )

    def add(self, session_id: str, role: str, message: str) -> None:
        self._buf[session_id].append({
            "role": role,
            "message": message,
            "ts": _utcnow_iso(),
        })

    def get_recent(
        self, session_id: str, n: int | None = None,
    ) -> list[dict[str, Any]]:
        msgs = list(self._buf[session_id])
        if n is None or n >= len(msgs):
            return msgs
        return msgs[-n:]

    def clear(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._buf.clear()
        else:
            self._buf.pop(session_id, None)

    def session_count(self) -> int:
        return sum(1 for d in self._buf.values() if d)


# -----------------------------------------------------------------
# tier 2 + 3 -- persistent
# -----------------------------------------------------------------

class MemoryManager:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or config.MEMORY_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        conn = _connect(self.db_path)
        try:
            # ----- episodic -----
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope           TEXT NOT NULL,
                    trace_id        TEXT NOT NULL,
                    event_type      TEXT NOT NULL,
                    summary         TEXT NOT NULL,
                    detail          TEXT NOT NULL DEFAULT '',
                    tags            TEXT NOT NULL DEFAULT '',
                    created_at      TEXT NOT NULL,
                    relevance_score REAL NOT NULL DEFAULT 1.0
                )
                """,
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodic_scope_rel "
                "ON episodic_memory("
                "scope, relevance_score DESC, created_at DESC)",
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(
                    summary, detail, tags,
                    content='episodic_memory', content_rowid='id'
                )
                """,
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS episodic_ai
                AFTER INSERT ON episodic_memory BEGIN
                    INSERT INTO episodic_fts(
                        rowid, summary, detail, tags
                    ) VALUES (new.id, new.summary, new.detail, new.tags);
                END
                """,
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS episodic_ad
                AFTER DELETE ON episodic_memory BEGIN
                    INSERT INTO episodic_fts(
                        episodic_fts, rowid, summary, detail, tags
                    ) VALUES (
                        'delete', old.id, old.summary, old.detail, old.tags
                    );
                END
                """,
            )

            # ----- semantic -----
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    category    TEXT NOT NULL DEFAULT 'fact',
                    key         TEXT UNIQUE NOT NULL,
                    value       TEXT NOT NULL,
                    source      TEXT NOT NULL,
                    confidence  REAL NOT NULL DEFAULT 1.0,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """,
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
                    key, value,
                    content='semantic_memory', content_rowid='id'
                )
                """,
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS semantic_ai
                AFTER INSERT ON semantic_memory BEGIN
                    INSERT INTO semantic_fts(rowid, key, value)
                    VALUES (new.id, new.key, new.value);
                END
                """,
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS semantic_au
                AFTER UPDATE ON semantic_memory BEGIN
                    INSERT INTO semantic_fts(
                        semantic_fts, rowid, key, value
                    ) VALUES ('delete', old.id, old.key, old.value);
                    INSERT INTO semantic_fts(rowid, key, value)
                    VALUES (new.id, new.key, new.value);
                END
                """,
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS semantic_ad
                AFTER DELETE ON semantic_memory BEGIN
                    INSERT INTO semantic_fts(
                        semantic_fts, rowid, key, value
                    ) VALUES ('delete', old.id, old.key, old.value);
                END
                """,
            )

            # Phase 15a -- lifecycle columns + partial indexes for
            # both episodic and semantic. ALTER TABLE ADD COLUMN with
            # NOT NULL DEFAULT backfills existing rows in a single
            # statement, so old DBs come out with state='active' and
            # pinned=0 on every existing row. Idempotent (PRAGMA
            # check before each ALTER).
            for tbl in ("episodic_memory", "semantic_memory"):
                cols = {
                    row["name"]
                    for row in conn.execute(
                        f"PRAGMA table_info({tbl})"
                    ).fetchall()
                }
                if "state" not in cols:
                    conn.execute(
                        f"ALTER TABLE {tbl} "
                        f"ADD COLUMN state TEXT NOT NULL "
                        f"DEFAULT 'active'"
                    )
                if "pinned" not in cols:
                    conn.execute(
                        f"ALTER TABLE {tbl} "
                        f"ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
                    )
                if "archived_at" not in cols:
                    conn.execute(
                        f"ALTER TABLE {tbl} ADD COLUMN archived_at TEXT"
                    )
                # Phase 15b -- write-origin provenance.
                if "created_by_origin" not in cols:
                    conn.execute(
                        f"ALTER TABLE {tbl} ADD COLUMN "
                        f"created_by_origin TEXT NOT NULL "
                        f"DEFAULT 'foreground'"
                    )
                # Partial indexes -- match the project hardware
                # discipline: cover ONLY the small subset we query.
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS "
                    f"idx_{tbl}_pinned ON {tbl}(pinned) "
                    f"WHERE pinned = 1"
                )
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS "
                    f"idx_{tbl}_archived ON {tbl}(state) "
                    f"WHERE state = 'archived'"
                )
        finally:
            conn.close()

    # ---------- episodic ----------

    def store_episode(
        self,
        scope: str,
        trace_id: str,
        event_type: str,
        summary: str,
        detail: str = "",
        tags: list[str] | str | None = None,
        relevance_score: float = 1.0,
    ) -> int:
        now = _utcnow_iso()
        tags_str = _normalize_tags(tags)
        rs = max(0.0, min(1.0, float(relevance_score)))
        # Phase 15b -- read origin off the ContextVar.
        from core.write_origin import get_current_write_origin
        origin = get_current_write_origin()
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                """
                INSERT INTO episodic_memory
                (scope, trace_id, event_type, summary, detail, tags,
                 created_at, relevance_score, created_by_origin)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (scope, trace_id, event_type, summary, detail or "",
                 tags_str, now, rs, origin),
            )
            new_id = cur.lastrowid
        finally:
            conn.close()
        log_event(
            trace_id, "DEBUG", "memory",
            f"stored episode id={new_id} scope={scope} "
            f"type={event_type} summary={summary[:80]!r}",
        )
        # auto-prune per scope
        self.prune_episodes(scope=scope)
        return new_id

    def search_episodes(
        self, query: str, scope: str | None = None, limit: int = 10,
        include_archived: bool = False,
    ) -> list[Episode]:
        """Phase 15a: ``include_archived=False`` (default) filters out
        rows whose state is 'archived'."""
        if not query.strip():
            return []
        fts_q = _escape_fts_query(query)
        conn = _connect(self.db_path)
        try:
            params: list[Any] = [fts_q]
            sql = (
                "SELECT e.* FROM episodic_fts f "
                "JOIN episodic_memory e ON e.id = f.rowid "
                "WHERE episodic_fts MATCH ? "
            )
            if not include_archived:
                sql += "AND e.state != 'archived' "
            if scope is not None:
                sql += "AND e.scope = ? "
                params.append(scope)
            sql += (
                "ORDER BY e.relevance_score DESC, e.created_at DESC "
                "LIMIT ?"
            )
            params.append(limit)
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as e:
                log_event(
                    "SEN-system", "WARNING", "memory",
                    f"episodic FTS query failed: {e}",
                )
                return []
        finally:
            conn.close()
        return [Episode(**dict(r)) for r in rows]

    def get_recent_episodes(
        self, scope: str | None = None, limit: int = 10,
        include_archived: bool = False,
    ) -> list[Episode]:
        """Phase 15a: archived rows hidden unless ``include_archived``."""
        archived_filter = (
            "" if include_archived else " AND state != 'archived'"
        )
        conn = _connect(self.db_path)
        try:
            if scope is None:
                rows = conn.execute(
                    f"SELECT * FROM episodic_memory "
                    f"WHERE 1=1{archived_filter} "
                    f"ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM episodic_memory WHERE scope = ?"
                    f"{archived_filter} "
                    f"ORDER BY created_at DESC LIMIT ?",
                    (scope, limit),
                ).fetchall()
        finally:
            conn.close()
        return [Episode(**dict(r)) for r in rows]

    def get_agent_context(
        self, scope: str, query: str = "",
        max_chars: int | None = None,
    ) -> str:
        """Context block for an agent: top FTS-matching episodes (if
        query) plus most-recent episodes for the scope. Capped at
        max_chars (default config.EPISODIC_CONTEXT_MAX_CHARS).
        """
        if max_chars is None:
            max_chars = config.EPISODIC_CONTEXT_MAX_CHARS
        episodes: list[Episode] = []
        seen: set[int] = set()
        if query.strip():
            for e in self.search_episodes(query, scope=scope, limit=5):
                if e.id not in seen:
                    episodes.append(e)
                    seen.add(e.id)
        for e in self.get_recent_episodes(scope=scope, limit=5):
            if e.id not in seen:
                episodes.append(e)
                seen.add(e.id)
        if not episodes:
            return ""
        chunks: list[str] = []
        running = 0
        for e in episodes:
            block = (
                f"- [{e.event_type}] {e.summary} "
                f"(rel={e.relevance_score:.2f}, "
                f"{e.created_at[:10]})\n"
            )
            if running + len(block) > max_chars:
                break
            chunks.append(block)
            running += len(block)
        return "".join(chunks)

    def decay_relevance(
        self, days_old: int | None = None,
        factor: float | None = None,
    ) -> int:
        days_old = (
            days_old if days_old is not None
            else config.MEMORY_DECAY_DAYS
        )
        factor = factor if factor is not None else config.MEMORY_DECAY_FACTOR
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days_old)
        ).isoformat()
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE episodic_memory "
                "SET relevance_score = relevance_score * ? "
                "WHERE created_at < ?",
                (factor, cutoff),
            )
            n = cur.rowcount
        finally:
            conn.close()
        if n:
            log_event(
                "SEN-system", "INFO", "memory",
                f"decay applied to {n} episodes "
                f"(older than {days_old}d, factor={factor})",
            )
        return n

    def prune_episodes(
        self, scope: str | None = None,
        max_per_scope: int | None = None,
    ) -> int:
        """Phase 15a: archive-not-delete. Counts non-archived rows
        per scope; if over the cap, archives the lowest-relevance
        ones. Pinned rows are skipped. Returns the number archived."""
        cap = max_per_scope or config.EPISODIC_MAX_PER_SCOPE
        now = _utcnow_iso()
        conn = _connect(self.db_path)
        try:
            if scope is None:
                scopes = [
                    r["scope"] for r in conn.execute(
                        "SELECT DISTINCT scope FROM episodic_memory "
                        "WHERE state != 'archived'"
                    ).fetchall()
                ]
            else:
                scopes = [scope]
            total_archived = 0
            for s in scopes:
                count = conn.execute(
                    "SELECT COUNT(*) AS n FROM episodic_memory "
                    "WHERE scope = ? AND state != 'archived'",
                    (s,),
                ).fetchone()["n"]
                if count <= cap:
                    continue
                ids = [
                    r["id"] for r in conn.execute(
                        "SELECT id FROM episodic_memory "
                        "WHERE scope = ? AND state != 'archived' "
                        "AND pinned = 0 "
                        "ORDER BY relevance_score ASC, created_at ASC "
                        "LIMIT ?",
                        (s, count - cap),
                    ).fetchall()
                ]
                if ids:
                    placeholders = ",".join("?" * len(ids))
                    conn.execute(
                        f"UPDATE episodic_memory SET "
                        f"state = 'archived', archived_at = ? "
                        f"WHERE id IN ({placeholders})",
                        [now, *ids],
                    )
                    total_archived += len(ids)
        finally:
            conn.close()
        if total_archived:
            log_event(
                "SEN-system", "INFO", "memory",
                f"archived {total_archived} low-relevance episodes "
                f"(cap={cap})",
            )
        return total_archived

    # ---------- semantic ----------

    def store_fact(
        self,
        key: str,
        value: str,
        source: str = "user_explicit",
        confidence: float | None = None,
        category: str = "fact",
    ) -> int:
        if confidence is None:
            confidence = config.MEMORY_SOURCE_CONFIDENCE.get(source, 0.6)
        confidence = max(0.0, min(1.0, float(confidence)))
        now = _utcnow_iso()
        # Phase 15b -- origin is recorded on INSERT only; on UPDATE
        # (upsert path) we leave the original origin in place because
        # it represents who first asserted the key. A foreground
        # re-assertion of an auto-extracted fact still revives state
        # to active, but the provenance trail stays accurate.
        from core.write_origin import get_current_write_origin
        origin = get_current_write_origin()
        conn = _connect(self.db_path)
        try:
            existing = conn.execute(
                "SELECT id, confidence FROM semantic_memory WHERE key = ?",
                (key,),
            ).fetchone()
            if existing is None:
                cur = conn.execute(
                    """
                    INSERT INTO semantic_memory
                    (category, key, value, source, confidence,
                     created_at, updated_at, created_by_origin)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (category, key, value, source, confidence, now,
                     now, origin),
                )
                new_id = cur.lastrowid
            else:
                # Confidence-aware merge: higher wins; equal -> newer wins.
                # Phase 15a: any qualifying upsert also revives state to
                # 'active' (auto-transition could have moved it to
                # stale/archived; an explicit re-store is the user
                # asserting the fact still matters).
                if confidence >= existing["confidence"]:
                    conn.execute(
                        """
                        UPDATE semantic_memory SET
                            category = ?, value = ?, source = ?,
                            confidence = ?, updated_at = ?,
                            state = 'active', archived_at = NULL
                        WHERE id = ?
                        """,
                        (category, value, source, confidence, now,
                         existing["id"]),
                    )
                new_id = existing["id"]
        finally:
            conn.close()
        # auto-prune
        if self._semantic_count() > config.SEMANTIC_MAX_ENTRIES:
            self._prune_semantic()
        return new_id

    def get_fact(self, key: str) -> Fact | None:
        conn = _connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM semantic_memory WHERE key = ?", (key,),
            ).fetchone()
        finally:
            conn.close()
        return Fact(**dict(row)) if row else None

    def search_facts(
        self, query: str, limit: int = 10,
        include_archived: bool = False,
    ) -> list[Fact]:
        if not query.strip():
            return []
        fts_q = _escape_fts_query(query)
        archived_filter = (
            "" if include_archived else " AND s.state != 'archived'"
        )
        conn = _connect(self.db_path)
        try:
            try:
                rows = conn.execute(
                    f"SELECT s.* FROM semantic_fts f "
                    f"JOIN semantic_memory s ON s.id = f.rowid "
                    f"WHERE semantic_fts MATCH ?{archived_filter} "
                    f"ORDER BY s.confidence DESC, s.updated_at DESC "
                    f"LIMIT ?",
                    (fts_q, limit),
                ).fetchall()
            except sqlite3.OperationalError as e:
                log_event(
                    "SEN-system", "WARNING", "memory",
                    f"semantic FTS query failed: {e}",
                )
                return []
        finally:
            conn.close()
        return [Fact(**dict(r)) for r in rows]

    def list_facts(
        self, category: str | None = None,
        include_archived: bool = False,
    ) -> list[Fact]:
        archived_filter = (
            "" if include_archived else " AND state != 'archived'"
        )
        conn = _connect(self.db_path)
        try:
            if category is None:
                rows = conn.execute(
                    f"SELECT * FROM semantic_memory "
                    f"WHERE 1=1{archived_filter} "
                    f"ORDER BY confidence DESC, updated_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM semantic_memory WHERE category = ?"
                    f"{archived_filter} "
                    f"ORDER BY confidence DESC, updated_at DESC",
                    (category,),
                ).fetchall()
        finally:
            conn.close()
        return [Fact(**dict(r)) for r in rows]

    def delete_fact(self, key: str) -> bool:
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "DELETE FROM semantic_memory WHERE key = ?", (key,),
            )
            ok = cur.rowcount > 0
        finally:
            conn.close()
        return ok

    def get_profile_context(
        self, max_chars: int | None = None,
    ) -> str:
        max_chars = max_chars or config.PROFILE_CONTEXT_MAX_CHARS
        # Persona-file mirrors are huge; skip them in the profile
        # context (the brain gets persona separately).
        facts = [
            f for f in self.list_facts()
            if f.source != "persona_file"
        ]
        if not facts:
            return ""
        chunks: list[str] = []
        running = 0
        for f in facts:
            unconfirmed = " UNCONFIRMED" if f.confidence < 0.8 else ""
            tag = (
                f"[{f.source}, conf={f.confidence:.2f}{unconfirmed}]"
            )
            block = f"- {f.key}: {f.value} {tag}\n"
            if running + len(block) > max_chars:
                break
            chunks.append(block)
            running += len(block)
        return "".join(chunks)

    def _semantic_count(self, include_archived: bool = False) -> int:
        archived_filter = (
            "" if include_archived else " WHERE state != 'archived'"
        )
        conn = _connect(self.db_path)
        try:
            row = conn.execute(
                f"SELECT COUNT(*) n FROM semantic_memory{archived_filter}"
            ).fetchone()
        finally:
            conn.close()
        return int(row["n"])

    def _prune_semantic(self) -> int:
        """Phase 15a: archive-not-delete. Pinned rows are skipped."""
        cap = config.SEMANTIC_MAX_ENTRIES
        total = self._semantic_count()
        if total <= cap:
            return 0
        to_archive = total - cap
        now = _utcnow_iso()
        conn = _connect(self.db_path)
        try:
            ids = [
                r["id"] for r in conn.execute(
                    "SELECT id FROM semantic_memory "
                    "WHERE state != 'archived' AND pinned = 0 "
                    "ORDER BY confidence ASC, updated_at ASC "
                    "LIMIT ?",
                    (to_archive,),
                ).fetchall()
            ]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE semantic_memory SET "
                    f"state = 'archived', archived_at = ? "
                    f"WHERE id IN ({placeholders})",
                    [now, *ids],
                )
        finally:
            conn.close()
        if ids:
            log_event(
                "SEN-system", "INFO", "memory",
                f"archived {len(ids)} low-confidence facts (cap={cap})",
            )
        return len(ids)

    # ---------- Phase 15a lifecycle (pin / unpin / restore) ----------

    def pin_episode(self, episode_id: int) -> bool:
        """Pin an episode -- protects it from prune + auto-transition.
        Idempotent. Returns True if the row exists."""
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE episodic_memory SET pinned = 1 WHERE id = ?",
                (episode_id,),
            )
            return cur.rowcount > 0
        finally:
            conn.close()

    def unpin_episode(self, episode_id: int) -> bool:
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE episodic_memory SET pinned = 0 WHERE id = ?",
                (episode_id,),
            )
            return cur.rowcount > 0
        finally:
            conn.close()

    def restore_episode(self, episode_id: int) -> bool:
        """Bring an archived episode back to state='active'."""
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE episodic_memory SET state = 'active', "
                "archived_at = NULL WHERE id = ?",
                (episode_id,),
            )
            return cur.rowcount > 0
        finally:
            conn.close()

    def pin_fact(self, key: str) -> bool:
        """Pin a semantic fact by its natural key."""
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE semantic_memory SET pinned = 1 WHERE key = ?",
                (key,),
            )
            return cur.rowcount > 0
        finally:
            conn.close()

    def unpin_fact(self, key: str) -> bool:
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE semantic_memory SET pinned = 0 WHERE key = ?",
                (key,),
            )
            return cur.rowcount > 0
        finally:
            conn.close()

    def restore_fact(self, key: str) -> bool:
        """Bring an archived fact back to state='active'."""
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE semantic_memory SET state = 'active', "
                "archived_at = NULL WHERE key = ?",
                (key,),
            )
            return cur.rowcount > 0
        finally:
            conn.close()

    def auto_transition_lifecycle(
        self,
        stale_after_days: int | None = None,
        archive_after_days: int | None = None,
    ) -> dict:
        """Walk both episodic + semantic on the same age windows used
        by the KB. ``low usage`` for episodes is relevance_score <= 0.5
        (decayed at least once); for facts it's confidence < 0.6
        (auto-extracted, never user-confirmed). Pinned rows are NEVER
        touched. Returns ``{'episodic': {'stale': N, 'archived': M},
        'semantic': {...}}``."""
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
            ep_stale = conn.execute(
                "UPDATE episodic_memory SET state = 'stale' "
                "WHERE state = 'active' AND pinned = 0 "
                "AND created_at < ? AND relevance_score <= 0.5",
                (stale_cutoff,),
            ).rowcount
            ep_archived = conn.execute(
                "UPDATE episodic_memory SET state = 'archived', "
                "archived_at = ? "
                "WHERE state = 'stale' AND pinned = 0 "
                "AND created_at < ?",
                (now.isoformat(), archive_cutoff),
            ).rowcount
            sem_stale = conn.execute(
                "UPDATE semantic_memory SET state = 'stale' "
                "WHERE state = 'active' AND pinned = 0 "
                "AND created_at < ? AND confidence < 0.6",
                (stale_cutoff,),
            ).rowcount
            sem_archived = conn.execute(
                "UPDATE semantic_memory SET state = 'archived', "
                "archived_at = ? "
                "WHERE state = 'stale' AND pinned = 0 "
                "AND created_at < ?",
                (now.isoformat(), archive_cutoff),
            ).rowcount
        finally:
            conn.close()
        result = {
            "episodic": {"stale": ep_stale, "archived": ep_archived},
            "semantic": {"stale": sem_stale, "archived": sem_archived},
        }
        if any([ep_stale, ep_archived, sem_stale, sem_archived]):
            log_event(
                "SEN-system", "INFO", "memory",
                f"auto_transition_lifecycle: {result} "
                f"(stale_after={stale_after_days}d, "
                f"archive_after={archive_after_days}d)",
            )
        return result

    # ---------- persona sync ----------

    def sync_persona_files(
        self, persona_dir: Path | None = None,
    ) -> int:
        """Mirror each persona MD file's content into semantic_memory
        with key=``persona:<filename>`` and source=``persona_file``.
        Idempotent (upsert).
        """
        d = persona_dir or config.PERSONA_DIR
        n = 0
        for name in sorted(config.PROTECTED_FILES):
            p = d / name
            if not p.exists():
                continue
            try:
                content = p.read_text(encoding="utf-8")
            except Exception as e:
                log_event(
                    "SEN-system", "WARNING", "memory",
                    f"sync_persona_files: failed to read {name}: {e}",
                )
                continue
            self.store_fact(
                key=f"persona:{name}",
                value=content,
                source="persona_file",
                confidence=1.0,
                category="persona",
            )
            n += 1
        log_event(
            "SEN-system", "INFO", "memory",
            f"synced {n} persona files into semantic_memory",
        )
        return n

    # ---------- auto-extraction ----------

    async def extract_facts_from_conversation(
        self, messages: list[dict[str, Any]], trace_id: str,
        brain: Any | None = None,
    ) -> int:
        """Use the brain to identify durable facts in a recent
        conversation slice (>= AUTO_EXTRACT_THRESHOLD messages).
        Stores them at confidence 0.6 (auto_extracted). Returns the
        count stored.
        """
        # Late import: brain.py imports memory.py via the persona
        # context path -- importing at module load creates a cycle.
        from core.brain import (  # noqa: PLC0415
            BRAIN, _strip_think_block,
        )

        b = brain or BRAIN
        if len(messages) < config.AUTO_EXTRACT_THRESHOLD:
            return 0
        convo = "\n".join(
            f"{m.get('role', 'user')}: {m.get('message', '')}"
            for m in messages
        )
        prompt = (
            "Identify DURABLE facts about the user from this "
            "conversation. Durable means: preferences, decisions, "
            "identifiers, constraints that will be true in future "
            "sessions. Skip transient remarks (weather, mood, "
            "single-task asks).\n"
            "Respond with ONLY a JSON array. Each item: "
            "{\"key\": \"<short_snake_case>\", "
            "\"value\": \"<verbatim>\"}. "
            "Empty array if nothing durable.\n\n"
            f"Conversation:\n{convo[:3000]}\n/no_think"
        )
        try:
            res = await b.inference.generate(
                prompt=prompt,
                model=b.model,
                system=(
                    "You extract durable user facts. "
                    "Output JSON only. /no_think"
                ),
                temperature=0.0,
                trace_id=trace_id,
            )
        except Exception as e:
            log_event(
                trace_id, "WARNING", "memory",
                f"auto-extract inference failed: "
                f"{type(e).__name__}: {e}",
            )
            return 0

        import json as _json  # noqa: PLC0415
        raw = _strip_think_block(res.text or "")
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return 0
        try:
            items = _json.loads(raw[start:end + 1])
        except _json.JSONDecodeError:
            return 0
        if not isinstance(items, list):
            return 0
        n = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            key = str(it.get("key", "")).strip()
            value = str(it.get("value", "")).strip()
            if not key or not value:
                continue
            self.store_fact(
                key=key, value=value,
                source="auto_extracted", confidence=0.6,
            )
            n += 1
        log_event(
            trace_id, "INFO", "memory",
            f"auto-extracted {n} facts from conversation",
        )
        return n

    # ---------- stats ----------

    def stats(self) -> dict[str, Any]:
        conn = _connect(self.db_path)
        try:
            ep_total = conn.execute(
                "SELECT COUNT(*) n FROM episodic_memory"
            ).fetchone()["n"]
            ep_scopes = [
                {"scope": r["scope"], "count": r["count"]}
                for r in conn.execute(
                    "SELECT scope, COUNT(*) AS count "
                    "FROM episodic_memory GROUP BY scope "
                    "ORDER BY count DESC"
                ).fetchall()
            ]
            sm_total = conn.execute(
                "SELECT COUNT(*) n FROM semantic_memory"
            ).fetchone()["n"]
            sm_by_source = [
                {"source": r["source"], "count": r["count"]}
                for r in conn.execute(
                    "SELECT source, COUNT(*) AS count "
                    "FROM semantic_memory GROUP BY source "
                    "ORDER BY count DESC"
                ).fetchall()
            ]
        finally:
            conn.close()
        return {
            "episodic_total": int(ep_total),
            "episodic_by_scope": ep_scopes,
            "semantic_total": int(sm_total),
            "semantic_by_source": sm_by_source,
        }


# -----------------------------------------------------------------
# module-level singletons
# -----------------------------------------------------------------

WORKING_MEMORY = WorkingMemory()
_MEMORY: MemoryManager | None = None


def get_memory() -> MemoryManager:
    """Lazy singleton -- avoids opening memory.db at import time so
    tests can monkeypatch ``config.MEMORY_DB_PATH`` first.
    """
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = MemoryManager()
    return _MEMORY


def reset_memory_singleton() -> None:
    """Tests call this to force MemoryManager rebuild against the
    patched DB path.
    """
    global _MEMORY
    _MEMORY = None
