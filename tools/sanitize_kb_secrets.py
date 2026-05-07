"""Phase 16 token-leak fix migration -- one-shot sanitize of leaked rows.

Walks every row in ``knowledge.db`` and:
  - Re-computes ``scrub(text)`` for every text column that can leak.
  - If any column changed, UPDATE in place + flip ``state='archived'``
    (per Q2=B owner directive 2026-05-06: redact AND archive, so
    polluted shapes can't poison future few-shot retrievals).
  - Reports what changed.

Idempotent on re-runs: rows already redacted have nothing left to
redact, so the regex pass is a no-op and the row keeps its current
state.

Run BEFORE bot restart that picks up the new ``KnowledgeBase._add``
scrub layer, OR after, doesn't matter -- this script only touches
existing rows. New rows will be sanitized by the live code path.

Usage: ``python tools/sanitize_kb_secrets.py``  (dry-run by default)
       ``python tools/sanitize_kb_secrets.py --apply``
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Make `core` importable when run from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config  # noqa: E402
from core.secrets_scrub import REDACTED_MARKER, contains_secret, scrub  # noqa: E402

# Columns that historically caught leaks. solution_code = the actual
# diff body; solution_pattern = stored recipe; qwen_plan_recipe = Qwen
# shadow recipe; explanation = Claude pre-teach explanation;
# problem_summary = user prompt (low-risk but conservative-included
# since the same scrubber would catch a leak there too).
TEXT_COLUMNS = (
    "problem_summary",
    "solution_code",
    "solution_pattern",
    "explanation",
    "qwen_plan_recipe",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write changes. Default is dry-run.",
    )
    parser.add_argument(
        "--db", default=str(config.KNOWLEDGE_DB_PATH),
        help="Path to knowledge.db",
    )
    args = parser.parse_args()
    apply = args.apply

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cols_csv = ", ".join(TEXT_COLUMNS)
    rows = cur.execute(
        f"SELECT id, state, {cols_csv} FROM knowledge"
    ).fetchall()

    affected: list[tuple[int, list[str]]] = []
    for row in rows:
        changed_cols: list[str] = []
        new_values: dict[str, str | None] = {}
        for col in TEXT_COLUMNS:
            old = row[col]
            if old is None:
                continue
            if not contains_secret(old):
                continue
            new = scrub(old)
            if new != old:
                changed_cols.append(col)
                new_values[col] = new
        if changed_cols:
            affected.append((row["id"], changed_cols))
            if apply:
                set_clause = ", ".join(f"{c} = ?" for c in changed_cols)
                params: list[object] = [new_values[c] for c in changed_cols]
                # Per Q2=B: also archive these rows so they can't poison
                # future few-shot retrievals. Pinned rows are an edge
                # case -- archive them too here (the leak takes priority
                # over the pinning intent; user can /kb restore later).
                conn.execute(
                    f"UPDATE knowledge SET {set_clause}, "
                    f"state = 'archived', "
                    f"archived_at = COALESCE(archived_at, datetime('now')) "
                    f"WHERE id = ?",
                    (*params, row["id"]),
                )
    if apply:
        conn.commit()

    conn.close()

    print(f"=== KB secret-scrub migration ({'APPLIED' if apply else 'dry-run'}) ===")
    print(f"db: {args.db}")
    print(f"rows scanned: {len(rows)}")
    print(f"rows with leaks: {len(affected)}")
    for pid, cols in affected:
        print(f"  id={pid:<4} columns={cols}")
    if not affected:
        print("  (no leaks found)")
    if not apply and affected:
        print("\nRun again with --apply to perform the redaction + archive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
