"""20-test ECC driver for /code. Submits via the same path Telegram uses
(route() -> DB queue -> worker). Waits for each to complete sequentially.
Bails early if 3 in a row fail with the same verdict text."""
import json
import sqlite3
import sys
import time

from core.database import get_task
from core.router import route

PROMPTS = [
    # Trivial (1-5): single new file with simple function
    "/code create core/strings_helpers.py with a function reverse(s) that returns the reversed string",
    "/code add a function square(x) that returns x*x to core/util.py",
    "/code create core/temperature.py with celsius_to_fahrenheit(c) returning c*9/5+32",
    "/code create core/booleans.py with is_even(n) that returns True if n is even",
    "/code add factorial(n) to core/util.py using recursion",
    # Easy (6-10): slightly more involved single-function
    "/code add merge_dicts(a, b) to core/util.py returning a merged dict",
    "/code create core/textutil.py with count_vowels(s) returning the count",
    "/code add min_max(lst) to core/util.py returning a (min, max) tuple",
    "/code add slugify(text) to core/util.py returning a URL-friendly slug",
    "/code add chunked(lst, n) to core/util.py yielding n-sized sub-lists",
    # Medium (11-15): more logic / decorators / classes
    "/code add flatten(lst) to core/util.py that recursively flattens nested lists",
    "/code create core/cache_util.py with a memoize decorator using functools",
    "/code add partition(pred, lst) to core/util.py returning (yes, no) split lists",
    "/code create core/retry_util.py with a retry decorator (3 attempts, exponential backoff)",
    "/code add a Stack class to core/util.py with push/pop/peek methods",
    # Harder (16-20): cross-file / multi-edit
    "/code add a multiply(a, b) function to core/util.py",
    "/code add a subtract(a, b) function to core/util.py",
    "/code add a module-level docstring to core/util.py describing its purpose",
    "/code create core/exports.py that imports add and multiply from core.util",
    "/code add a divide(a, b) function to core/util.py that raises ValueError on div-by-zero",
]


def wait_for_task(task_id: str, timeout: int = 600) -> dict | None:
    """Poll DB until task is completed/failed or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        task = get_task(task_id)
        if task and task.status in ("completed", "failed"):
            return task.model_dump()
        time.sleep(2)
    return None


def short_verdict(result_str: str) -> str:
    """Pull a short verdict + solved_by from the result blob."""
    try:
        data = json.loads(result_str) if isinstance(result_str, str) \
                                       else result_str
    except Exception:
        return "(unparsed)"
    if isinstance(data, dict) and data.get("_error"):
        return f"ERROR: {str(data.get('error', ''))[:80]}"
    solved_by = data.get("solved_by", "?")
    validated = data.get("validated", False)
    attempts = data.get("attempts", "?")
    note = (data.get("teaching_note") or "")[:80]
    return f"solved_by={solved_by} validated={validated} attempts={attempts} note={note!r}"


def main():
    results: list[dict] = []
    consecutive_fails = 0
    last_failure_signature = ""
    overall_start = time.time()

    # Skip prompt #1 (already submitted manually)
    skip_first = "--skip-first" in sys.argv
    start_idx = 1 if skip_first else 0

    for i, prompt in enumerate(PROMPTS[start_idx:], start=start_idx + 1):
        print(f"\n{'=' * 70}")
        print(f"TEST {i}/20: {prompt[:80]}")
        print('=' * 70)
        rr = route(prompt)
        if rr.status != "ok":
            results.append({"i": i, "prompt": prompt,
                            "outcome": f"ROUTE_FAIL: {rr.message}"})
            print(f"  routing failed: {rr.message}")
            continue
        print(f"  enqueued task_id={rr.task_id} trace_id={rr.trace_id}")
        t0 = time.time()
        task = wait_for_task(rr.task_id, timeout=600)
        elapsed = int(time.time() - t0)
        if task is None:
            outcome = "TIMEOUT"
            print(f"  TIMEOUT after {elapsed}s")
        elif task["status"] == "failed":
            outcome = f"FAILED: {task.get('error', '')[:80]}"
            print(f"  failed in {elapsed}s: {outcome}")
        else:
            verdict = short_verdict(task["result"])
            outcome = verdict
            print(f"  done in {elapsed}s: {verdict}")
        results.append({"i": i, "prompt": prompt,
                        "elapsed": elapsed, "outcome": outcome,
                        "trace_id": rr.trace_id})

        # Early-bail: 3 in a row with same failure signature
        is_failure = ("ERROR" in outcome
                      or "TIMEOUT" in outcome
                      or "qwen_failed" in outcome
                      or "FAILED" in outcome)
        sig = outcome[:40]
        if is_failure and sig == last_failure_signature:
            consecutive_fails += 1
        elif is_failure:
            consecutive_fails = 1
            last_failure_signature = sig
        else:
            consecutive_fails = 0
        if consecutive_fails >= 3:
            print(f"\n!!! 3 consecutive same-signature failures -- BAILING")
            results.append({"BAILED_AT": i,
                            "reason": last_failure_signature})
            break

        # Total time guard
        if time.time() - overall_start > 3600:
            print(f"\n!!! 60-min total budget hit -- BAILING")
            break

    print("\n\n" + "=" * 70)
    print(f"FINAL RESULTS ({len(results)} tests recorded):")
    print("=" * 70)
    for r in results:
        if "BAILED_AT" in r:
            print(f"  BAILED at test {r['BAILED_AT']}: {r['reason']}")
        else:
            i = r["i"]
            outcome = r["outcome"]
            elapsed = r.get("elapsed", "?")
            print(f"  {i:2d}. ({elapsed}s) {outcome[:90]}")
    return results


if __name__ == "__main__":
    main()
