"""Emergency: clear stale GPU lock + fail in-flight tasks.

Used when a Sentinel process was killed mid-task, leaving the DB
holding a stale GPU lock that newer bots see as 'busy' and requeue
forever. Run after killing all bot processes.
"""
import sqlite3
import sys
from pathlib import Path

# Resolve sentinel.db relative to this script's project root so this
# works on any machine (was hardcoded to a single dev path pre-prep).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config as _cfg
DB = _cfg.DB_PATH
conn = sqlite3.connect(str(DB))
print("=== locks BEFORE ===")
for row in conn.execute("SELECT resource, locked_by, locked_at FROM locks"):
    print(" ", row)
conn.execute(
    "UPDATE locks SET locked_by = NULL, locked_at = NULL "
    "WHERE resource = 'gpu'"
)
conn.commit()
print("=== locks AFTER ===")
for row in conn.execute("SELECT resource, locked_by, locked_at FROM locks"):
    print(" ", row)
cur = conn.execute(
    "UPDATE tasks SET status = 'failed', "
    "error = 'killed during runaway requeue loop' "
    "WHERE status IN ('processing', 'pending')"
)
print(f"=== tasks marked failed: {cur.rowcount} ===")
conn.commit()
conn.close()
