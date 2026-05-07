"""20-scenario stress test runner for Phase 16.

Submits 20 progressively harder /code prompts directly into the
sentinel.db tasks queue, waits for each to complete (in-process
polling, no shell loops), captures results + KB pattern info,
appends a per-scenario JSON line to results.jsonl, and prints a
summary at the end.

Usage:
    python tools/stress_test_runner.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DB = PROJECT / "sentinel.db"
KB = PROJECT / "knowledge.db"
RESULTS = PROJECT / "logs" / "stress_test_results.jsonl"
PER_SCENARIO_TIMEOUT_S = 480

# Each: (id, name, prompt). Prompts are intentionally explicit + low-risk
# (each creates a NEW file in workspace/stress_test/). Progressively
# harder: simple constant -> validation -> regex -> async -> abstract
# class hierarchy -> stats -> multi-fn modules.
SCENARIOS = [
    (1, "constant",
     'create a NEW file workspace/stress_test/s01_const.py containing exactly:\n'
     'PHASE = "phase 16 stress"\n'
     'use write_file. verify by running: python -c "from workspace.stress_test.s01_const import PHASE; assert PHASE == \'phase 16 stress\'; print(\'ok\')"'),

    (2, "add-fn",
     'create a NEW file workspace/stress_test/s02_add.py containing only:\n'
     'def add(a, b):\n    return a + b\n'
     'use write_file. verify: python -c "from workspace.stress_test.s02_add import add; assert add(2, 3) == 5; assert add(-1, 1) == 0; print(\'ok\')"'),

    (3, "two-fn",
     'create a NEW file workspace/stress_test/s03_pair.py with two top-level functions:\n'
     'def square(x): return x * x\n'
     'def cube(x): return x * x * x\n'
     'use write_file. verify: python -c "from workspace.stress_test.s03_pair import square, cube; assert square(4) == 16; assert cube(3) == 27; print(\'ok\')"'),

    (4, "validation",
     'create a NEW file workspace/stress_test/s04_clamp.py with one function:\n'
     'def clamp(x, lo, hi):\n'
     '    raises ValueError if lo > hi, otherwise returns max(lo, min(hi, x))\n'
     'use write_file. verify: python -c "from workspace.stress_test.s04_clamp import clamp; assert clamp(5, 0, 10) == 5; assert clamp(-1, 0, 10) == 0; assert clamp(11, 0, 10) == 10; \\nimport sys\\ntry: clamp(0, 5, 1); sys.exit(1)\\nexcept ValueError: print(\'ok\')"'),

    (5, "regex",
     'create a NEW file workspace/stress_test/s05_digits.py:\n'
     'import re\n'
     'def extract_digits(s): return re.findall(r"\\d+", s)\n'
     'use write_file. verify: python -c "from workspace.stress_test.s05_digits import extract_digits; assert extract_digits(\'abc123def45\') == [\'123\', \'45\']; assert extract_digits(\'none\') == []; print(\'ok\')"'),

    (6, "type-hints",
     'create a NEW file workspace/stress_test/s06_typed.py:\n'
     'def safe_divide(a: float, b: float) -> float | None: return a/b if b != 0 else None\n'
     'with full type hints. use write_file. verify: python -c "from workspace.stress_test.s06_typed import safe_divide; assert safe_divide(10, 2) == 5.0; assert safe_divide(1, 0) is None; print(\'ok\')"'),

    (7, "simple-class",
     'create a NEW file workspace/stress_test/s07_counter.py with class Counter:\n'
     '    __init__(self, start: int = 0) sets self._n = start\n'
     '    increment(self, by: int = 1) adds by to self._n\n'
     '    get(self) returns self._n\n'
     'use write_file. verify: python -c "from workspace.stress_test.s07_counter import Counter; c = Counter(); c.increment(); c.increment(2); assert c.get() == 3; print(\'ok\')"'),

    (8, "dataclass",
     'create a NEW file workspace/stress_test/s08_point.py with:\n'
     'from dataclasses import dataclass\n'
     'import math\n'
     '@dataclass\n'
     'class Point:\n'
     '    x: float\n'
     '    y: float\n'
     '    def distance_to(self, other: "Point") -> float: return math.hypot(self.x - other.x, self.y - other.y)\n'
     'use write_file. verify: python -c "from workspace.stress_test.s08_point import Point; assert abs(Point(0,0).distance_to(Point(3,4)) - 5.0) < 1e-9; print(\'ok\')"'),

    (9, "generator",
     'create a NEW file workspace/stress_test/s09_chunks.py:\n'
     'def chunks(items, n):\n'
     '    buf = []\n'
     '    for x in items:\n'
     '        buf.append(x)\n'
     '        if len(buf) == n:\n'
     '            yield buf\n'
     '            buf = []\n'
     '    if buf:\n'
     '        yield buf\n'
     'use write_file. verify: python -c "from workspace.stress_test.s09_chunks import chunks; assert list(chunks([1,2,3,4,5], 2)) == [[1,2],[3,4],[5]]; print(\'ok\')"'),

    (10, "decorator-flag",
     'create a NEW file workspace/stress_test/s10_called.py with a decorator track_called that wraps any function and adds a was_called attribute on the WRAPPER (initially False, set True after first call). implement using functools.wraps and a closure flag. use write_file. verify: python -c "from workspace.stress_test.s10_called import track_called\n@track_called\ndef foo(): return 1\nassert foo.was_called is False\nfoo()\nassert foo.was_called is True\nprint(\'ok\')"'),

    (11, "context-mgr",
     'create a NEW file workspace/stress_test/s11_timer.py with class Timer that supports `with Timer() as t:` and after exit sets t.elapsed (float seconds since enter). use time.monotonic. use write_file. verify: python -c "from workspace.stress_test.s11_timer import Timer; import time; t = Timer()\nwith t:\n    time.sleep(0.02)\nassert t.elapsed >= 0.02\nprint(\'ok\')"'),

    (12, "async",
     'create a NEW file workspace/stress_test/s12_async.py:\n'
     'import asyncio\n'
     'async def double_async(x, delay=0.001):\n'
     '    await asyncio.sleep(delay)\n'
     '    return x * 2\n'
     'use write_file. verify: python -c "import asyncio; from workspace.stress_test.s12_async import double_async; assert asyncio.run(double_async(7)) == 14; print(\'ok\')"'),

    (13, "memoize",
     'create a NEW file workspace/stress_test/s13_memo.py with a decorator memoize that caches results in a dict keyed by the args tuple. use functools.wraps. use write_file. verify: python -c "from workspace.stress_test.s13_memo import memoize\ncalls = []\n@memoize\ndef f(x):\n    calls.append(x)\n    return x*x\nassert f(3) == 9; assert f(3) == 9; assert calls == [3]; print(\'ok\')"'),

    (14, "abstract",
     'create a NEW file workspace/stress_test/s14_shape.py with:\n'
     'from abc import ABC, abstractmethod\n'
     'import math\n'
     'class Shape(ABC):\n'
     '    @abstractmethod\n'
     '    def area(self) -> float: ...\n'
     'class Circle(Shape):\n'
     '    def __init__(self, radius): self.radius = radius\n'
     '    def area(self): return math.pi * self.radius ** 2\n'
     'class Rectangle(Shape):\n'
     '    def __init__(self, width, height): self.width, self.height = width, height\n'
     '    def area(self): return self.width * self.height\n'
     'use write_file. verify: python -c "import math; from workspace.stress_test.s14_shape import Circle, Rectangle; assert abs(Circle(5).area() - math.pi*25) < 1e-9; assert Rectangle(3, 4).area() == 12; print(\'ok\')"'),

    (15, "enum",
     'create a NEW file workspace/stress_test/s15_status.py:\n'
     'from enum import Enum\n'
     'class Status(Enum):\n'
     '    OK = "ok"\n'
     '    WARN = "warn"\n'
     '    ERROR = "error"\n'
     'use write_file. verify: python -c "from workspace.stress_test.s15_status import Status; assert Status.OK.value == \'ok\'; assert Status.WARN != Status.ERROR; assert len(list(Status)) == 3; print(\'ok\')"'),

    (16, "json-io",
     'create a NEW file workspace/stress_test/s16_jsonio.py:\n'
     'import json\n'
     'from pathlib import Path\n'
     'def dump_json(path, data): Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")\n'
     'def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))\n'
     'use write_file. verify: python -c "import tempfile, os; from workspace.stress_test.s16_jsonio import dump_json, load_json; p = tempfile.mktemp(suffix=\'.json\'); dump_json(p, {\'a\': 1, \'b\': [1,2]}); assert load_json(p) == {\'a\': 1, \'b\': [1,2]}; os.unlink(p); print(\'ok\')"'),

    (17, "regex-groups",
     'create a NEW file workspace/stress_test/s17_email.py:\n'
     'import re\n'
     '_EMAIL = re.compile(r"^(?P<user>[^@\\s]+)@(?P<domain>[^@\\s]+)$")\n'
     'def parse_email(s):\n'
     '    m = _EMAIL.match(s)\n'
     '    return {"user": m.group("user"), "domain": m.group("domain")} if m else None\n'
     'use write_file. verify: python -c "from workspace.stress_test.s17_email import parse_email; assert parse_email(\'a@b.c\') == {\'user\': \'a\', \'domain\': \'b.c\'}; assert parse_email(\'not an email\') is None; print(\'ok\')"'),

    (18, "property-class",
     'create a NEW file workspace/stress_test/s18_temp.py with class Temperature:\n'
     '    __init__(self, celsius: float) raises ValueError if celsius < -273.15, stores self._c\n'
     '    @property celsius returns self._c\n'
     '    @property fahrenheit returns self._c * 9/5 + 32\n'
     '    @property kelvin returns self._c + 273.15\n'
     'use write_file. verify: python -c "from workspace.stress_test.s18_temp import Temperature\nt = Temperature(25)\nassert t.celsius == 25\nassert t.fahrenheit == 77.0\nassert abs(t.kelvin - 298.15) < 1e-9\nimport sys\ntry: Temperature(-300); sys.exit(1)\nexcept ValueError: print(\'ok\')"'),

    (19, "stats",
     'create a NEW file workspace/stress_test/s19_stats.py with three functions:\n'
     'def mean(xs): raises ValueError on empty, else sum(xs)/len(xs)\n'
     'def median(xs): raises ValueError on empty, else middle of sorted (or avg of two middles for even-length)\n'
     'def mode(xs): raises ValueError on empty, else most-frequent value, ties broken by smallest\n'
     'use write_file. verify: python -c "from workspace.stress_test.s19_stats import mean, median, mode\nassert mean([1,2,3]) == 2.0\nassert median([1,2,3,4]) == 2.5\nassert mode([1,2,2,3,3]) == 2\nimport sys\ntry: mean([]); sys.exit(1)\nexcept ValueError: print(\'ok\')"'),

    (20, "multi-fn",
     'create a NEW file workspace/stress_test/s20_text.py with three functions:\n'
     'def count_words(s): returns len(s.split())\n'
     'def count_chars(s): returns len(s)\n'
     'def count_lines(s): returns 0 if s == "" else s.count("\\n") + 1\n'
     'use write_file. verify: python -c "from workspace.stress_test.s20_text import count_words, count_chars, count_lines\nassert count_words(\'hello world\') == 2\nassert count_chars(\'hi\') == 2\nassert count_lines(\'\') == 0\nassert count_lines(\'a\') == 1\nassert count_lines(\'a\\\\nb\\\\nc\') == 3\nprint(\'ok\')"'),
]


def submit_code(prompt: str) -> tuple[str, str]:
    task_id = uuid.uuid4().hex
    trace_id = f"SEN-{task_id[:8]}"
    args = json.dumps({"text": prompt})
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(DB))
    conn.execute(
        "INSERT INTO tasks (task_id, trace_id, command, args, status, "
        "priority, retry_count, max_retries, recovery_count, max_recoveries, "
        "created_at, updated_at) VALUES (?, ?, 'code', ?, 'pending', 5, 0, 5, 0, 3, ?, ?)",
        (task_id, trace_id, args, now, now),
    )
    conn.commit()
    conn.close()
    return task_id, trace_id


def wait_for_task(task_id: str, timeout: int) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        conn = sqlite3.connect(str(DB))
        conn.row_factory = sqlite3.Row
        try:
            r = conn.execute(
                "SELECT status, result, error FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        finally:
            conn.close()
        if r and r["status"] in ("completed", "failed"):
            return dict(r)
        time.sleep(2)
    return None


def latest_kb_for_trace(trace_id: str) -> dict | None:
    conn = sqlite3.connect(str(KB))
    conn.row_factory = sqlite3.Row
    try:
        r = conn.execute(
            "SELECT id, category, problem_summary, solo_attempts, "
            "solo_passes, pinned, qwen_plan_agreement, state "
            "FROM knowledge WHERE source_trace_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (trace_id,),
        ).fetchone()
        # Also handle dedup case: if no row matches trace, the most
        # recently updated active pattern is what got bumped.
        if r is None:
            r = conn.execute(
                "SELECT id, category, problem_summary, solo_attempts, "
                "solo_passes, pinned, qwen_plan_agreement, state "
                "FROM knowledge WHERE category='pattern' AND state='active' "
                "ORDER BY last_verified_at DESC LIMIT 1",
            ).fetchone()
    finally:
        conn.close()
    return dict(r) if r else None


def append_result(record: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def emit(line: str) -> None:
    """Print to stdout with flush so the Monitor sees each event live."""
    print(line, flush=True)


def main() -> int:
    emit(f"# stress test starting at {datetime.now().isoformat()}")
    emit(f"# {len(SCENARIOS)} scenarios queued")
    if RESULTS.exists():
        RESULTS.unlink()

    results = []
    for sid, name, prompt in SCENARIOS:
        t0 = time.time()
        emit(f"\n[{sid}/20] {name} -- submitting")
        task_id, trace_id = submit_code(prompt)
        task = wait_for_task(task_id, PER_SCENARIO_TIMEOUT_S)
        elapsed = time.time() - t0

        if task is None:
            rec = {
                "id": sid, "name": name, "trace_id": trace_id,
                "status": "TIMEOUT", "verdict": "TIMEOUT",
                "elapsed_s": round(elapsed, 1),
            }
            results.append(rec)
            append_result(rec)
            emit(f"[{sid}/20] {name} -- TIMEOUT after {elapsed:.0f}s")
            continue

        result_str = task.get("result") or "{}"
        try:
            result = json.loads(result_str)
        except json.JSONDecodeError:
            result = {"raw": result_str}
        sol = (
            result.get("solution") if isinstance(result, dict) else None
        ) or str(result)

        if "Solved" in sol:
            verdict = "PASS"
        elif "Could not complete" in sol:
            verdict = "FAIL"
        else:
            verdict = "?"

        kb = latest_kb_for_trace(trace_id)
        rec = {
            "id": sid, "name": name, "trace_id": trace_id,
            "status": task["status"], "verdict": verdict,
            "elapsed_s": round(elapsed, 1),
            "pattern_id": kb["id"] if kb else None,
            "category": kb["category"] if kb else None,
            "solo_passes": kb["solo_passes"] if kb else None,
            "solo_attempts": kb["solo_attempts"] if kb else None,
            "pinned": kb["pinned"] if kb else None,
            "agreement": kb["qwen_plan_agreement"] if kb else None,
        }
        results.append(rec)
        append_result(rec)

        pid = rec["pattern_id"]
        cat = rec["category"]
        solo = (f"{rec['solo_passes']}/{rec['solo_attempts']}"
                if rec["solo_passes"] is not None else "-")
        agree = rec["agreement"]
        agree_s = f"{agree:.3f}" if isinstance(agree, (int, float)) else "-"
        pin = "PIN" if rec["pinned"] else "  "
        emit(
            f"[{sid}/20] {name} -- {verdict} {elapsed:>4.0f}s "
            f"pid={pid} cat={cat} solo={solo} agree={agree_s} {pin}"
        )

    # Summary
    emit("\n# === SUMMARY ===")
    n_pass = sum(1 for r in results if r.get("verdict") == "PASS")
    n_fail = sum(1 for r in results if r.get("verdict") == "FAIL")
    n_to = sum(1 for r in results if r.get("verdict") == "TIMEOUT")
    n_other = len(results) - n_pass - n_fail - n_to
    total = sum(r["elapsed_s"] for r in results)
    emit(f"# PASS:    {n_pass}/{len(results)}")
    emit(f"# FAIL:    {n_fail}/{len(results)}")
    emit(f"# TIMEOUT: {n_to}/{len(results)}")
    emit(f"# OTHER:   {n_other}/{len(results)}")
    emit(f"# total wall time: {total/60:.1f} min")
    emit(f"# results jsonl: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
