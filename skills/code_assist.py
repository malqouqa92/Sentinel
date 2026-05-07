"""Self-improving code assistant: Claude teaches Qwen, executor validates.

Flow per request:
  1. KB lookup: gather "you've seen this before" patterns from knowledge.db
  2. Qwen attempt: build prompt with KB context, ask for JSON {code, explanation}
  3. Validate: actually run the code through the code_execute skill
     - PASS  -> store pattern, return as solved_by="qwen"
     - FAIL  -> proceed
  4. Claude teaches: solution + teaching_note + reusable pattern
  5. Qwen re-attempts with the teaching context
  6. Validate again:
     PASS  -> store as pattern, solved_by="qwen_taught"
     FAIL  -> store Claude's solution as pattern (not limitation -- it
              still works), solved_by="claude_direct"

Graceful degradations:
  - No SENTINEL_CLAUDE_KEY     -> skip steps 4-6, return Qwen attempt with warning
  - Claude API error/timeout   -> same: Qwen attempt with warning
  - Ollama down                -> SkillError (the skill is meaningless w/o it)
"""
import asyncio
import time
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from core import config
from core.knowledge_base import KnowledgeBase
from core.llm import LLMError, OllamaClient
from core.logger import log_event
from core.registry import SKILL_REGISTRY
from core.skills import BaseSkill, SkillError


# ----------------------- I/O schemas ---------------------------------

class CodeAssistInput(BaseModel):
    problem: str
    code_context: str | None = None
    language: str = "python"


class CodeAssistOutput(BaseModel):
    solution: str
    explanation: str
    solved_by: str   # "qwen" | "qwen_agent" | "qwen_taught" | "qwen_failed" | "claude_direct"
    teaching_note: str | None = None
    knowledge_entries_used: list[int]
    attempts: int
    validated: bool
    execution_result: str | None = None


# ----------------------- Helpers -------------------------------------

QWEN_SYSTEM_BASE = (
    "You are a Python developer. Respond with ONLY a JSON object "
    "containing:\n"
    "- \"code\": the complete solution code -- valid Python with all "
    "  necessary imports. Define functions/classes that solve the "
    "  problem. DO NOT include an `if __name__ == \"__main__\":` "
    "  test block; the validator will call your functions directly.\n"
    "- \"explanation\": one short conversational paragraph describing "
    "  what the code does (no implementation details).\n"
    "Do not include markdown fences. JSON only."
)

QWEN_TAUGHT_SYSTEM = (
    "You are a Python developer. A senior engineer reviewed your "
    "previous attempt and provided corrections. Study the teaching "
    "note and pattern carefully, then write the corrected solution. "
    "Define functions/classes only -- DO NOT include an `if __name__"
    "` block. Respond with ONLY a JSON object: "
    "{\"code\": ..., \"explanation\": ...}"
)

CLAUDE_SYSTEM = (
    "You are a senior engineer teaching a JUNIOR (a small 3B model) "
    "to solve coding problems. The junior just attempted this and "
    "failed. Your goal is to make the junior able to solve this "
    "class of problem on its own next time -- not just to ship code.\n"
    "\n"
    "Output a JSON object with FOUR fields:\n"
    "1. \"solution\" -- the complete working Python code. Define "
    "functions/classes only. DO NOT include an if __name__ block "
    "-- the validator calls functions directly.\n"
    "2. \"teaching_note\" -- one paragraph (<=300 chars) describing "
    "the EXACT mistake the junior made and why.\n"
    "3. \"pattern\" -- a step-by-step recipe a 3B model can follow. "
    "Format MUST be:\n"
    "STEP 1: <first concrete operation>\n"
    "STEP 2: <next>\n"
    "STEP 3: <next>\n"
    "KEY INSIGHT: <the one thing the small model usually misses>\n"
    "Each STEP must be one short sentence the junior can execute.\n"
    "4. \"worked_example\" -- a concrete demonstration of the recipe "
    "applied to a sample. Format: 'Inputs: <sample inputs>\\nCall: "
    "<function call>\\nExpected output: <what it should return or "
    "print>'. This is what the junior pattern-matches against in "
    "future runs.\n"
    "\n"
    "Respond with ONLY the JSON object. No markdown fences, no "
    "preamble, no explanation outside the JSON."
)


def _coerce_str(value) -> str:
    """LLMs occasionally return arrays of lines or dicts where a string
    was expected. Coerce defensively so we don't blow up at the
    Pydantic validation boundary."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_coerce_str(v) for v in value)
    if isinstance(value, dict):
        # Sometimes the model nests: {"code": {"value": "..."}}
        for key in ("code", "value", "text", "content"):
            if key in value:
                return _coerce_str(value[key])
        # Fallback: serialize
        return json.dumps(value)
    return str(value)


def _extract_json(text: str) -> dict | None:
    """Recover a parsed JSON object from messy LLM output. Tries several
    progressively-more-lenient strategies; returns the first dict that
    passes."""
    if not text:
        return None

    # Strategy 1: non-greedy fenced block then json.loads
    fenced = re.search(
        r"```(?:json)?\s*(.+?)\s*```",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 2: GREEDY fenced block (handles nested fences in content)
    greedy = re.search(
        r"```(?:json)?\s*(.+)\s*```",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    if greedy:
        try:
            return json.loads(greedy.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: largest {...} span anywhere
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 4: regex-extract the three expected fields individually.
    # Useful when Claude's JSON has subtle escaping issues but each
    # field is recoverable on its own. We unescape \n and \" so the
    # resulting code is runnable.
    def _grab(field: str) -> str | None:
        m = re.search(
            rf'"{field}"\s*:\s*"((?:\\.|[^"\\])*)"',
            text, flags=re.DOTALL,
        )
        if not m:
            return None
        raw = m.group(1)
        # JSON string -> Python string (newlines, quotes, etc.)
        try:
            return json.loads(f'"{raw}"')
        except Exception:
            return raw.replace(r"\n", "\n").replace(r'\"', '"')
    sol = _grab("solution")
    if sol:
        return {
            "solution": sol,
            "teaching_note": _grab("teaching_note") or "",
            "pattern": _grab("pattern") or "",
        }

    # Strategy 5: pull a python code fence out as the solution
    code_block = re.search(
        r"```(?:python|py)?\s*(.+?)\s*```",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    if code_block:
        return {
            "solution": code_block.group(1),
            "teaching_note": "(unparsed JSON wrapper -- raw code recovered)",
            "pattern": "",
        }
    return None


def _clean_solution_text(text: str) -> str:
    """Normalize LLM code output: strip markdown fences, unwrap
    stringly-encoded JSON envelopes, decode escape sequences. Used
    both before quality-gating and before storage so the KB is
    always clean."""
    if not text:
        return ""
    text = text.strip()
    # Pass 1: markdown fence
    fence = re.search(
        r"```(?:python|py)?\s*(.+?)\s*```",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        text = fence.group(1).strip()
    # Pass 2: full JSON object with "code" / "solution" key
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("code", "solution"):
                if key in parsed and isinstance(parsed[key], str):
                    text = parsed[key]
                    break
        elif isinstance(parsed, str):
            text = parsed
    except (json.JSONDecodeError, ValueError):
        pass
    # Pass 3: lone JSON-encoded string starting with {"
    # Sometimes qwen-coder emits {"import x\nfoo..."} (no key, just the
    # stringified code wrapped in object braces). Recover by lifting
    # the largest "..." span when valid Python is inside.
    if text.startswith('{"') and ("\\n" in text or "\\\"" in text):
        inner_match = re.match(r'^\s*\{\s*"(.+?)"\s*[:,}]?',
                               text, flags=re.DOTALL)
        if inner_match:
            cand = inner_match.group(1)
            try:
                cand = json.loads(f'"{cand}"')
            except (json.JSONDecodeError, ValueError):
                cand = (
                    cand.replace("\\n", "\n")
                        .replace("\\t", "\t")
                        .replace('\\"', '"')
                        .replace("\\\\", "\\")
                )
            if "import " in cand or "def " in cand or "print(" in cand:
                text = cand
    # Pass 4: bare escape sequences in plain text (no real newlines)
    if "\\n" in text and "\n" not in text:
        text = (
            text.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
        )
    return text.strip()


def _is_real_solution(code: str) -> bool:
    """Quality gate: a 'solution' must be substantive, not a trivial
    literal. Used both at validation time and when seeding the KB.

    Recognizes three valid shapes:
      1. Python source with structural markers (def/class/import/etc)
      2. `git diff --stat` output (legacy pre-Phase-14b agentic /code
         stored the stat as solution_code)
      3. `git diff` body (Phase 14b+ stores the actual diff text --
         markers `diff --git`, hunk headers, +/- lines)

    Phase 15d-bugfix: shape 3 added because Phase 14b changed
    solution_code from stat to body, but the gate was never updated.
    Result: 8 successful KB patterns silently deleted by
    cleanup_low_quality_patterns on bot restarts before this fix.
    """
    if not code:
        return False
    stripped = code.strip()
    if len(stripped) < 30:
        return False
    # Shape 3 (Phase 14b+): git diff body. Strongest single marker
    # is the literal `diff --git ` header; hunk headers `@@ ` and
    # +/- lines are corroborating signals. We require diff --git
    # OR (hunk header + add/remove lines) because diff --git alone
    # is rare-but-possible in unrelated text.
    if "diff --git " in stripped:
        return True
    has_hunk = "@@ " in stripped or stripped.count("\n@@") > 0
    has_pm = (
        any(line.startswith("+") and not line.startswith("+++")
            for line in stripped.splitlines())
        and any(line.startswith("-") and not line.startswith("---")
                for line in stripped.splitlines())
    )
    if has_hunk and has_pm:
        return True
    # Shape 2: git diff --stat fingerprint
    diff_stat_markers = (" file changed", " files changed",
                         " insertion", " deletion", " | ")
    diff_marker_hits = sum(1 for m in diff_stat_markers if m in stripped)
    if diff_marker_hits >= 2:
        return True
    # Shape 1: Python source
    structural = ("def ", "class ", "import ", "from ",
                  "return", "for ", "while ", "with ",
                  "print(")
    if not any(marker in stripped for marker in structural):
        return False
    try:
        import ast as _ast
        tree = _ast.parse(stripped)
    except SyntaxError:
        return False
    if len(tree.body) <= 1:
        if tree.body and isinstance(tree.body[0], _ast.Expr) and \
           isinstance(tree.body[0].value, _ast.Constant):
            return False
    return True


def _format_kb_examples(entries) -> str:
    """Few-shot framing of KB hits. Each entry becomes a worked
    'EXAMPLE N' block with the full working code, the recipe pattern,
    and the why. This is what Qwen actually pattern-matches against."""
    if not entries:
        return ""
    chunks = []
    for i, e in enumerate(entries[:3], 1):
        chunks.append(
            f"EXAMPLE {i}\n"
            f"Problem: {e.problem_summary}\n"
            f"Solution:\n```python\n{(e.solution_code or '')[:1500]}"
            f"\n```\n"
            f"Recipe:\n{e.solution_pattern or '(no recipe)'}\n"
            f"Why it works: {e.explanation}\n"
        )
    return "\n".join(chunks)


_STEP_LINE_RE = re.compile(
    r"^STEP\s+\d+\s*:.*?(?=\n\s*STEP\s+\d+\s*:|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def _extract_recipe_steps_from_text(text: str) -> str:
    """Pull just the STEP N: blocks out of an LLM response, dropping
    any narration/exploration/markdown noise around them.

    When teacher Claude has tools enabled it tends to write thousands
    of chars of analysis and Read-output before the actual recipe;
    naive truncation chops the recipe off the end. This recovers the
    actionable bits regardless of where they are in the response."""
    if not text:
        return ""
    matches = _STEP_LINE_RE.findall(text)
    if not matches:
        return text  # no STEPs found -- pass through, let parser try
    return "\n".join(m.strip() for m in matches if m.strip())


# Phase 15d -- recipe length cap for stepfed mode. The Phase 9
# rationale for the original 4000-char cap was "longer recipes push
# Qwen into prose-mode (no tool calls)". Stepfed parses each step
# INDIVIDUALLY -- prose drift across step boundaries doesn't apply.
# Bumping to 8000 covers the multi-file edits that were failing
# before because the last step's `new=` arg got cut mid-string.
RECIPE_MAX_CHARS_STEPFED = 8000


_PROJECT_PATH_RE = re.compile(
    # Project-relative paths only (one or more dirs + file w/ ext).
    # Bounded by non-path chars or string boundaries. Excludes
    # absolute Windows / Unix prefixes -- we want PORTABLE paths
    # the recipe writer would also accept.
    r"(?<![\w./\\])"
    r"((?:[a-zA-Z_][a-zA-Z0-9_-]*/)+[a-zA-Z_][a-zA-Z0-9_.-]*"
    r"\.(?:py|md|yml|yaml|json|toml|sh|ps1|sql))"
    r"(?![\w/])",
)


def _extract_project_paths(*texts: str) -> set[str]:
    """Phase 15d: pull project-relative-looking paths out of arbitrary
    text (recipe steps, review reasoning). Used to build the "files
    Claude has already read in prior attempts" hint for corrective
    teach. Conservative -- only matches paths with a recognised
    extension and at least one directory segment, so we don't
    falsely seed the hint with bare module names."""
    out: set[str] = set()
    for t in texts:
        if not t:
            continue
        for m in _PROJECT_PATH_RE.finditer(t):
            p = m.group(1).strip()
            # Drop windows-y absolute remnants if they slipped past.
            if ":" in p or p.startswith("/"):
                continue
            out.add(p)
    return out


def _shape_repetition_phrase(
    prior: str, current: str, n: int = 5,
) -> str | None:
    """Phase 15d: detect "we're attempting variants of the same
    Claude-recipe quality issue" by looking for an n-gram of
    NON-stopword tokens that appears verbatim in two consecutive
    review reasonings.

    Returns the matching phrase (joined with spaces) or None.

    The threshold is intentionally strict: we only bail when the
    SAME concrete failure mode shows up twice. Generic shared phrases
    like "Read confirms that the" don't qualify because they're
    composed mostly of stopwords."""
    if not prior or not current:
        return None
    stop = {
        "the", "a", "an", "and", "or", "but", "if", "of", "to", "in",
        "on", "at", "for", "with", "is", "are", "was", "were", "be",
        "been", "being", "has", "have", "had", "do", "does", "did",
        "this", "that", "these", "those", "it", "its", "as", "by",
        "from", "not", "no", "so", "into", "than", "then", "when",
        "where", "which", "who", "whom", "what", "while", "still",
        "also", "only", "any", "all", "some", "such", "now", "yet",
        "i", "you", "he", "she", "we", "they", "them", "us", "his",
        "her", "their", "your", "my", "me",
    }

    def _tokens(s: str) -> list[str]:
        # Lowercase, keep word chars + dots (so file.py stays one
        # token), strip everything else.
        cleaned = re.sub(r"[^a-zA-Z0-9_.]+", " ", s.lower())
        return [t for t in cleaned.split() if t]

    def _ngrams(toks: list[str], n: int) -> set[str]:
        out: set[str] = set()
        for i in range(len(toks) - n + 1):
            window = toks[i:i + n]
            # Skip phrases that are >50% stopwords -- those are
            # boilerplate, not the actual failure shape.
            non_stop = sum(1 for w in window if w not in stop)
            if non_stop * 2 < n:
                continue
            out.add(" ".join(window))
        return out

    p_grams = _ngrams(_tokens(prior), n)
    c_grams = _ngrams(_tokens(current), n)
    common = p_grams & c_grams
    if not common:
        return None
    # Pick the longest concrete-looking phrase deterministically.
    return sorted(common, key=lambda s: (-len(s), s))[0]


def _truncate_recipe_to_steps(recipe: str, max_chars: int) -> str:
    """Phase 15d: drop trailing partial step if the recipe exceeds the
    cap, instead of hard-cutting mid-string and leaving a malformed
    edit_file path=... old="..." with no new= arg.

    Walks STEP blocks in order, accumulating until adding the next one
    would overflow. Always returns at least one complete step (or the
    original recipe if no STEPs found / it's already under the cap).
    """
    if not recipe or len(recipe) <= max_chars:
        return recipe
    matches = _STEP_LINE_RE.findall(recipe)
    if not matches:
        # No parseable STEPs -- preserve the original cut behavior so
        # we don't lose data on a recipe shape we don't recognise.
        return recipe[:max_chars]
    kept: list[str] = []
    running = 0
    for raw_step in matches:
        step = raw_step.strip()
        if not step:
            continue
        # +1 for the joining newline once we have multiple steps.
        cost = len(step) + (1 if kept else 0)
        if running + cost > max_chars:
            break
        kept.append(step)
        running += cost
    if not kept:
        # First step alone exceeds the cap -- best we can do is keep
        # it (truncated); a downstream parser will catch any damage.
        return matches[0].strip()[:max_chars]
    return "\n".join(kept)


def _normalize_recipe_paths(recipe: str) -> str:
    """Rewrite absolute paths under PROJECT_ROOT to POSIX-relative.
    Stored patterns become portable across machines."""
    if not recipe:
        return recipe
    root = str(config.PROJECT_ROOT)
    # Match the root with either / or \ separators (Windows mixes them
    # in Claude's output)
    for sep in ("\\\\", "\\", "/"):
        prefix = root.replace("\\", sep) + sep
        recipe = recipe.replace(prefix, "")
    # Catch edge case: bare PROJECT_ROOT without trailing sep
    recipe = recipe.replace(root + "\\", "").replace(root + "/", "")
    return recipe


def _format_kb_for_claude(patterns, limitations) -> str:
    """Frame KB hits FOR CLAUDE during pre/corrective teach.

    Patterns are prior winning recipes Claude generated -- showing them
    back lets Claude ADAPT a proven tool sequence rather than re-derive
    one each time. Limitations are prior failure modes -- showing them
    back lets Claude AVOID known dead ends. Bounded ~3000 chars."""
    if not patterns and not limitations:
        return ""
    parts = []
    if patterns:
        parts.append("PRIOR SUCCESSFUL FIXES (adapt these tool sequences):")
        for e in patterns[:3]:
            recipe = (e.solution_pattern or "(no recipe stored)")[:600]
            files = (e.solution_code or "")[:200]
            parts.append(
                f"[Pattern #{e.id}] {e.problem_summary}\n"
                f"  Tags: {e.tags}\n"
                f"  Files touched: {files}\n"
                f"  Winning recipe:\n{recipe}\n"
                f"  Used {e.usage_count}x. {e.explanation[:160]}"
            )
    if limitations:
        parts.append("\nPRIOR FAILURES (do NOT repeat these mistakes):")
        for e in limitations[:2]:
            parts.append(
                f"[Limit #{e.id}] {e.problem_summary}\n"
                f"  Why it failed: {e.explanation[:240]}"
            )
    return "\n\n".join(parts)[:3000]


def _extract_code_fallback(text: str) -> str | None:
    """If JSON parse failed, try to pull a python code block."""
    if not text:
        return None
    fenced = re.search(
        r"```(?:python|py)?\s*(.+?)\s*```",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1).strip()
    # Last resort: if it looks like Python, return as-is
    if "def " in text or "import " in text or "print(" in text:
        return text.strip()
    return None


def _qwen_user_prompt(
    problem: str, code_context: str | None, kb_examples: str,
) -> str:
    parts: list[str] = []
    if kb_examples:
        parts.append(
            "Here are worked examples of similar problems you have "
            "solved before. Study the recipes and adapt the same "
            "patterns to the new problem below.\n\n" + kb_examples
        )
    parts.append(f"PROBLEM:\n{problem}")
    if code_context:
        parts.append(f"EXISTING CODE:\n{code_context}")
    if kb_examples:
        parts.append(
            "Now apply the same recipe pattern to the problem above. "
            "Output JSON only."
        )
    return "\n\n".join(parts)


def _qwen_taught_user_prompt(
    problem: str, code_context: str | None,
    teaching_note: str, pattern: str, claude_solution: str,
) -> str:
    parts = [
        f"TEACHING NOTE: {teaching_note}",
        f"PATTERN TO FOLLOW: {pattern}",
        f"CORRECT APPROACH (from senior):\n{claude_solution}",
        f"PROBLEM:\n{problem}",
    ]
    if code_context:
        parts.append(f"EXISTING CODE:\n{code_context}")
    return "\n\n".join(parts)


def _claude_user_prompt(
    problem: str, code_context: str | None, qwen_code: str,
    stdout: str, stderr: str, return_code: int,
) -> str:
    parts = [f"PROBLEM:\n{problem}"]
    if code_context:
        parts.append(f"EXISTING CODE:\n{code_context}")
    parts.append(f"JUNIOR'S ATTEMPT:\n{qwen_code}")
    parts.append(
        f"EXECUTION ERROR:\nstdout: {stdout[:1000]}\n"
        f"stderr: {stderr[:1000]}\n"
        f"return_code: {return_code}"
    )
    return "\n\n".join(parts)


def _summarize_problem(problem: str, max_len: int = 120) -> str:
    p = problem.strip().replace("\n", " ")
    return p if len(p) <= max_len else p[:max_len - 3] + "..."


def _extract_tags(problem: str, code: str) -> list[str]:
    """Cheap tag extraction: pick out a handful of programming-domain
    keywords that show up in the problem or code. Good enough for FTS
    retrieval; not meant to be exhaustive."""
    text = (problem + " " + (code or "")).lower()
    candidates = [
        "asyncio", "subprocess", "thread", "lock", "lru", "ttl", "cache",
        "regex", "json", "csv", "yaml", "http", "websocket", "sqlite",
        "pydantic", "decorator", "generator", "iterator", "context manager",
        "dataclass", "list", "dict", "set", "string", "palindrome",
        "fibonacci", "recursion", "loop", "sort", "binary search",
        "datetime", "timezone", "file", "path", "encoding",
        "exception", "timeout", "retry", "queue", "stack", "tree",
        "graph", "hash", "function", "class",
    ]
    found = []
    for kw in candidates:
        if kw in text:
            found.append(kw.replace(" ", "_"))
    return found[:8]


# ----------------------- Validation -----------------------------------
# AST-based smoke validation as the PRIMARY check. We parse the code,
# look for top-level functions/classes, exec() the module to catch
# import/name errors, then synthesize a simple call to a non-private
# function. This avoids the false-negative trap where Qwen produces
# a perfectly fine function but its `if __name__` test block crashes
# the subprocess validator over something irrelevant (curses, network,
# sleep loops, etc.).

import ast as _ast
import inspect as _inspect


def _synthesize_args(func) -> tuple[tuple, dict]:
    """Build a (args, kwargs) tuple that's likely to be acceptable to
    the function based on its annotations. Falls back to empty for
    untypable params -- the function will raise TypeError, which we
    treat as a soft pass (function exists; signature mismatch is OK)."""
    try:
        sig = _inspect.signature(func)
    except (TypeError, ValueError):
        return (), {}
    args: list = []
    kwargs: dict = {}
    SAMPLE = {
        int: 5, float: 50.0, str: "test", bool: True,
        list: [1, 2, 3], dict: {"a": 1}, tuple: (1, 2), set: {1, 2},
        bytes: b"x",
    }
    for p in sig.parameters.values():
        if p.kind == _inspect.Parameter.VAR_POSITIONAL or \
           p.kind == _inspect.Parameter.VAR_KEYWORD:
            continue
        if p.default is not _inspect.Parameter.empty:
            continue  # has a default; let it use that
        ann = p.annotation
        sample = None
        if ann is not _inspect.Parameter.empty:
            sample = SAMPLE.get(ann)
        if sample is None:
            sample = "test"
        if p.kind == _inspect.Parameter.KEYWORD_ONLY:
            kwargs[p.name] = sample
        else:
            args.append(sample)
    return tuple(args), kwargs


async def _validate_code(
    code: str, trace_id: str,
) -> tuple[bool, dict]:
    """Pass criteria (in priority order):
      1. AST parses (no SyntaxError)
      2. Module exec() runs without raising (imports + module-level OK)
      3. At least one top-level function or class is defined
      4. Smoke-call a public function:
         - succeeds -> PASS
         - raises TypeError on bad args -> PASS (function exists)
         - raises any other exception -> still PASS if function ran
           past the call boundary; FAIL if it never started
    If 1-3 fail OR no function exists, fall back to the legacy
    subprocess execution check (for scripts with module-level prints)."""

    # --- 1. AST parse ---
    code = code or ""
    try:
        tree = _ast.parse(code)
    except SyntaxError as e:
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"validate: SyntaxError -> fail ({e})")
        return False, {"return_code": 1, "stderr": f"SyntaxError: {e}",
                       "stdout": ""}

    # --- 2. Module exec ---
    sandbox: dict = {"__name__": "__sandbox__"}
    try:
        exec(compile(tree, "<solution>", "exec"), sandbox)
    except Exception as e:
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"validate: module exec failed ({type(e).__name__}: "
                  f"{str(e)[:120]}) -> falling back to subprocess")
        return await _subprocess_validate(code, trace_id)

    # --- 3. Find a public function/class ---
    callables = [
        n for n in tree.body
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                          _ast.ClassDef))
        and not n.name.startswith("_")
    ]
    if not callables:
        log_event(trace_id, "INFO", "skill.code_assist",
                  "validate: no public function/class -> fallback")
        return await _subprocess_validate(code, trace_id)

    # --- 4. Smoke call ---
    target = callables[0]
    obj = sandbox.get(target.name)
    if obj is None:
        return await _subprocess_validate(code, trace_id)
    if isinstance(target, _ast.ClassDef):
        # Just instantiating without args is enough proof it's defined
        try:
            obj()
        except TypeError:
            pass  # required init args; class exists, that's the win
        except Exception as e:
            log_event(trace_id, "INFO", "skill.code_assist",
                      f"validate: class {target.name}() raised "
                      f"{type(e).__name__}: {str(e)[:80]}; accepting")
        return True, {"return_code": 0, "stdout": "",
                      "stderr": f"class {target.name} validated"}
    args, kwargs = _synthesize_args(obj)
    try:
        result = obj(*args, **kwargs)
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"validate: {target.name}{args} -> ok")
    except TypeError as e:
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"validate: {target.name} TypeError on smoke args "
                  f"({e}) -- function exists, accepting")
    except Exception as e:
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"validate: {target.name} raised "
                  f"{type(e).__name__}: {str(e)[:80]} on smoke args "
                  f"-- function ran, accepting")
    return True, {"return_code": 0, "stdout": str(result)[:500]
                  if 'result' in locals() else "",
                  "stderr": ""}


async def _subprocess_validate(
    code: str, trace_id: str,
) -> tuple[bool, dict]:
    """Legacy subprocess execution. Used as a fallback when AST checks
    can't find a function (e.g., the code is a top-level script)."""
    code_skill = SKILL_REGISTRY.get("code_execute")
    if code_skill is None:
        raise SkillError(
            "code_assist",
            "code_execute skill is not registered (cannot validate)",
            trace_id,
        )
    inp = code_skill.input_schema(
        code=code, timeout=config.TEACHING_EXECUTOR_TIMEOUT,
    )
    result = await code_skill.execute(inp, trace_id)
    out = result.model_dump()
    passed = (out["return_code"] == 0)
    if passed and "Traceback (most recent call last)" in (
        out.get("stderr") or ""
    ):
        passed = False
    return passed, out


# ----------------------- Qwen call -----------------------------------

def _qwen_generate(
    system: str, user: str, trace_id: str, model: str,
    timeout: int = 900,
    format_json: bool = True,
    num_predict: int | None = None,
) -> str:
    """Sync helper. Caller wraps in asyncio.to_thread.
    timeout default 15 min -- some Qwen calls on a 4GB GPU with cold
    model load + KB-augmented prompts genuinely take that long.

    Phase 15d-bugfix: ``format_json`` defaults to True for back-compat
    with the production stepfed transcription path (which asks Qwen
    to emit a tool-call dict). The shadow-plan path passes False so
    Ollama doesn't force Qwen into JSON mode and override the
    system prompt's "STEP 1: tool key=value" format directive --
    that bug suppressed every shadow agreement score to 0.0 between
    Phase 15c ship and this fix."""
    return OllamaClient().generate(
        model=model, prompt=user, system=system,
        format_json=format_json, trace_id=trace_id,
        timeout=timeout, num_predict=num_predict,
    )


# Phase 15c -- shadow planning. Hard cap so a stuck Qwen never
# blocks /code (the OllamaClient default is 900s).
# Phase 16 Batch A: bumped 30 -> 90 because tools-enabled shadow
# does multi-turn LLM calls.
# Phase 16 Batch A bugfix: bumped 90 -> 180 after first production
# trace (SEN-bbef3a4f) timed out at 90s on a cold qwen2.5-coder:3b
# load. Real Ollama chat-call latency on a 4GB GPU during cold-load
# is 15-30s per call; with up to 5 read_file/list_dir tool turns
# plus the final recipe-output turn, the 90s budget couldn't
# accommodate. 180s gives cold-load runs room to finish; warm-model
# runs still complete in ~30s well under the cap.
QWEN_SHADOW_TIMEOUT_S = 180

# Phase 16 Batch A -- when True, shadow planning uses the agentic
# read-only loop (run_shadow_planner) instead of one-shot text-out.
#
# Phase 16 Batch A revert (2026-05-06): default flipped True -> False
# after stress test (S1 SEN-9e28f685 + S2 SEN-a95e89b2) showed
# consistent 180s+ shadow timeouts. Root cause is NOT cold-load
# (a 90->180 timeout bump didn't help) but single-call inference
# time on this 4GB GPU with KB-augmented prompts: one Qwen chat
# call with 5-10K chars of context takes 180-210 seconds, larger
# than any reasonable shadow budget. The agentic loop multiplies
# this -- 2-3 chat calls = 6-10 minutes per /code, unacceptable.
#
# The architectural work (SHADOW_TOOLS_SCHEMA, SHADOW_PLANNER_SYSTEM,
# run_shadow_planner) is preserved behind this flag. Flip back to
# True once we have num_predict caps or a faster GPU.
#
# In the meantime, the legacy one-shot path with format_json=False
# (Phase 15d-bugfix) gives reliable mid-quality shadow data in
# ~10-15s without blocking /code.
SHADOW_PLAN_USE_TOOLS = False

QWEN_SHADOW_SYSTEM_BASE = (
    "You are a junior engineer. Given a problem and KB context, "
    "produce a recipe of tool calls in EXACTLY the format below. "
    "No prose, no preamble, no markdown -- just numbered STEP lines.\n"
    "\n"
    "Tools you can use:\n"
    "  read_file(path)\n"
    "  list_dir(path)\n"
    "  write_file(path, content)\n"
    "  edit_file(path, old, new)\n"
    "  run_bash(command)\n"
    "  done(summary)\n"
    "\n"
    "Output exactly this shape:\n"
    "STEP 1: <tool> <key>=\"<value>\" ...\n"
    "STEP 2: <tool> ...\n"
    "STEP 3: done summary=\"<one-line summary>\"\n"
)


def _load_qwencoder_memo() -> str:
    """Phase 15d: read the QWENCODER.md teaching memo fresh on every
    call. Edits via /curate qwencoder (or hand-edits) take effect on
    the NEXT shadow plan, no bot restart needed. Capped at the
    PERSONA_INJECT_MAX_CHARS limit to avoid blowing past Qwen's
    context budget when KB context + project_map + memo all stack.

    Returns "" if the file is missing -- the system prompt still
    works without it (the BASE prompt has the contract); the memo
    just adds the failure-pattern knowledge."""
    try:
        path = config.PERSONA_DIR / "QWENCODER.md"
        if not path.exists():
            return ""
        cap = config.PERSONA_INJECT_MAX_CHARS.get("QWENCODER.md", 6000)
        text = path.read_text(encoding="utf-8")
        if len(text) > cap:
            text = text[:cap]
        return text
    except Exception:
        # Best-effort: the shadow path must never raise from here.
        return ""


def _qwen_shadow_system_prompt() -> str:
    """Compose the BASE prompt + the QWENCODER.md memo. Memo is
    appended (not prepended) so the strict contract instructions
    are the first thing the model reads."""
    memo = _load_qwencoder_memo()
    if not memo:
        return QWEN_SHADOW_SYSTEM_BASE
    return (
        QWEN_SHADOW_SYSTEM_BASE
        + "\n\n"
        + "==== QWENCODER MEMO (curated coding playbook) ====\n"
        + memo
    )


# Back-compat alias: existing tests / older imports may reference
# the constant name. Resolves to the dynamic composer.
QWEN_SHADOW_SYSTEM = QWEN_SHADOW_SYSTEM_BASE


async def _qwen_shadow_plan(
    problem: str, code_context: str | None,
    kb_patterns_block: str, project_map: str,
    backend_model: str, trace_id: str,
) -> str | None:
    """Phase 15c: ask Qwen for the same recipe Claude just produced
    (using the same KB context) so we can score structural agreement.
    Best-effort: any failure (timeout, Ollama down, garbage output)
    returns None and the caller continues with Claude's recipe.

    Phase 16 Batch A: when SHADOW_PLAN_USE_TOOLS is True (default),
    dispatches to qwen_agent.run_shadow_planner -- a multi-turn
    agentic loop where Qwen has read_file + list_dir (NO write,
    NO edit, NO bash). This raises agreement scores significantly
    by letting Qwen actually look at the code before writing the
    recipe instead of guessing.

    NEVER raise -- the shadow path must not block /code."""
    if SHADOW_PLAN_USE_TOOLS:
        # Phase 16 Batch A: agentic read-only path. Qwen explores,
        # then outputs a recipe. Hard 90s timeout cap (6x the cost
        # of one-shot but typically lands at 0.6+ agreement vs the
        # 0.3-0.5 ceiling of one-shot blind).
        try:
            from core.qwen_agent import run_shadow_planner
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    run_shadow_planner,
                    problem, kb_patterns_block, project_map,
                    trace_id, backend_model,
                ),
                timeout=QWEN_SHADOW_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log_event(
                trace_id, "INFO", "skill.code_assist",
                f"shadow plan: agentic Qwen timed out at "
                f"{QWEN_SHADOW_TIMEOUT_S}s -- continuing without "
                f"shadow data",
            )
            return None
        except Exception as e:
            log_event(
                trace_id, "INFO", "skill.code_assist",
                f"shadow plan: agentic Qwen failed "
                f"({type(e).__name__}: {e}) -- continuing",
            )
            return None
        recipe = result.get("recipe") or ""
        tool_calls = result.get("tool_calls", 0)
        if not recipe:
            log_event(
                trace_id, "INFO", "skill.code_assist",
                f"shadow plan: agentic Qwen produced no recipe "
                f"after {tool_calls} tool calls",
            )
            return None
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"shadow plan (agentic): {tool_calls} read-only tool "
            f"calls, recipe {len(recipe)} chars",
        )
        return recipe

    # Phase 15c legacy path -- one-shot, no tools. Kept for A/B
    # comparison and fallback when SHADOW_PLAN_USE_TOOLS=False.
    user = (
        f"Problem:\n{problem}\n\n"
        f"Project map:\n{project_map[:1500]}\n\n"
        + (
            f"Code context:\n{code_context[:1500]}\n\n"
            if code_context else ""
        )
        + (
            f"KB context (prior patterns + limitations):\n"
            f"{kb_patterns_block[:2500]}\n\n"
            if kb_patterns_block else ""
        )
        + "Now produce the recipe.\n"
    )
    # Phase 15d: load QWENCODER.md fresh per call so curated edits
    # take effect on the NEXT shadow plan without a bot restart.
    system_prompt = _qwen_shadow_system_prompt()
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(
                _qwen_generate,
                system_prompt, user, trace_id, backend_model,
                # Phase 15d-bugfix: format_json=False so Ollama
                # doesn't lock Qwen into JSON-output mode. The shadow
                # plan needs plain-text "STEP 1: ..." -- JSON mode
                # was forcing every score to 0.0 because the parser
                # found no STEP blocks. Discovered live on pattern
                # id=64 right after Phase 15d shipped.
                900,    # timeout positional (matches helper sig)
                False,  # format_json=False
            ),
            timeout=QWEN_SHADOW_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"shadow plan: Qwen timed out at {QWEN_SHADOW_TIMEOUT_S}s "
            f"-- continuing without shadow data",
        )
        return None
    except Exception as e:
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"shadow plan: Qwen failed "
            f"({type(e).__name__}: {e}) -- continuing",
        )
        return None
    if not raw:
        return None
    # Reuse the existing extractor so prose around STEP lines is
    # stripped exactly the way the production parser expects it.
    extracted = _extract_recipe_steps_from_text(raw)
    return extracted or None


# ----------------------- Claude CLI subprocess ------------------------
# Phase 8: extracted to core.claude_cli. Kept here as a thin alias so
# existing tests that monkeypatch this symbol still work.

from core.claude_cli import find_claude_cli as _find_claude_cli  # noqa: E402, F401


PRE_TEACH_SYSTEM = (
    "You are a senior engineer producing a recipe for a junior agent "
    "(an executor) to run. YOU HAVE TOOLS: Read, Grep, Glob. USE THEM "
    "before writing any recipe step that references a file -- the "
    "executor cannot afford guesses.\n"
    "\n"
    "Workflow:\n"
    "1. USE Read/Grep/Glob to inspect every file you reference.\n"
    "2. DECIDE: is this task small enough for ONE recipe?\n"
    "   - YES (3-7 STEPs, <=2 files of substantial work): emit "
    "STEP-N format (default path).\n"
    "   - NO (>~8 STEPs, >2 files, OR adding a brand-new command lane "
    "/ pipeline / agent): emit DECOMPOSE format instead. The user will "
    "run each subtask as its own /code. Decomposing a big task is "
    "BETTER than producing a 12-step recipe that hits the 8000-char "
    "cap and gets truncated.\n"
    "\n"
    "Executor's tools (it has NO others):\n"
    "  read_file(path)\n"
    "  list_dir(path)\n"
    "  write_file(path, content)\n"
    "  edit_file(path, old, new)\n"
    "  run_bash(command)\n"
    "  done(summary)\n"
    "\n"
    "RULES for STEP-N output:\n"
    "1. THE FIRST CHARACTERS OF YOUR OUTPUT MUST BE `STEP 1:`. "
    "No preamble, no markdown heading, no narrative, no \"I'll do X\" "
    "intro -- just the STEP lines, beginning with `STEP 1:` on the "
    "FIRST line.\n"
    "2. Every step is exactly ONE tool call on its own line. NO prose "
    "between STEP lines. NO summary text after the final STEP. The "
    "done() summary IS the summary -- nothing follows it.\n"
    "3. Aim for 3-7 STEPs. If you need more than ~8 STEPs, the task "
    "is too big -- use DECOMPOSE instead (see below).\n"
    "4. PREFER write_file (whole-file rewrite) over edit_file when a "
    "file is new OR when more than ~30%% of it changes. With "
    "write_file you give the executor the COMPLETE content -- no "
    "string-matching needed -- this is the most reliable path.\n"
    "5. If you use edit_file, the `old` arg MUST be a literal "
    "substring you copy-pasted from a Read tool result. Include 3-5 "
    "lines of unique surrounding context. NEVER invent `old` text.\n"
    "6. The final step is ALWAYS done(summary=\"...\").\n"
    "7. NO triple quotes inside content. Use \\n for newlines, \\\\ "
    "for backslashes.\n"
    "8. If PRIOR SUCCESSFUL FIXES are shown, ADAPT one of them.\n"
    "9. Use PROJECT-ROOT-RELATIVE paths (e.g. `core/foo.py`), NOT "
    "absolute Windows paths. The executor expands them.\n"
    "\n"
    "RULES for DECOMPOSE output:\n"
    "1. First line: `DECOMPOSE` (literal, alone on the line, no "
    "punctuation after).\n"
    "2. Then 2-5 lines, each starting with `- /code ` followed by a "
    "concrete subtask phrased as a /code prompt. Each subtask must "
    "be small enough to land in <=8 STEPs on its own.\n"
    "3. NO additional text after the bullet list. NO recipe STEPs. "
    "NO summary.\n"
    "4. Subtasks should be ordered: each later one assumes earlier "
    "ones have shipped and been /commit'd.\n"
    "5. Decomposition is for SCOPE, not for retries. Don't decompose "
    "a 5-step recipe into 3 sub-prompts.\n"
    "\n"
    "GOOD STEP-N output (small task -- emit ONLY this, nothing else):\n"
    "STEP 1: write_file path=\"core/util.py\" "
    "content=\"def add(a, b):\\n    return a + b\\n\"\n"
    "STEP 2: run_bash command=\"python -c 'from core.util import add; "
    "print(add(2, 3))'\"\n"
    "STEP 3: done summary=\"created core/util.py with add(a, b)\"\n"
    "\n"
    "WRONG STEP-N output (will be rejected by the parser):\n"
    "  Sure! I'll start by reading the file, then add the function.\n"
    "  STEP 1: read_file path=\"core/util.py\"\n"
    "  Now let me write the function:\n"
    "  STEP 2: edit_file ...\n"
    "[REJECTED: prose before STEP 1, prose between STEPs.]\n"
    "\n"
    "GOOD DECOMPOSE output (big task -- emit ONLY this, nothing else):\n"
    "DECOMPOSE\n"
    "- /code add empty /qcode handler stub in interfaces/telegram_bot.py "
    "that just replies 'not implemented'\n"
    "- /code wire /qcode handler to call core.qwen_agent planner with "
    "KB context\n"
    "- /code wire /qcode planner output to run_agent_stepfed and "
    "report pass/fail\n"
    "\n"
    "**PHASE 17h -- SIZE-FORCED DECOMPOSE (load-bearing).**\n"
    "BEFORE writing the recipe, do this checklist out loud (in your "
    "head, not in the response):\n"
    "  Q1. How many distinct FILES will my recipe touch?\n"
    "  Q2. How many STEPs do I anticipate?\n"
    "  Q3. Am I adding a new command lane / agent / pipeline / skill?\n"
    "If ANY of (Q1 > 2 files) OR (Q2 > 7 STEPs) OR (Q3 = yes), you "
    "MUST emit DECOMPOSE -- regardless of the complexity tier the "
    "system told you. The 8000-char parser cap is a HARD CEILING; "
    "STEPs past it get DROPPED SILENTLY (live failure 2026-05-07 "
    "~02:23Z: Claude wrote 9077 chars, parser truncated to 6160 at "
    "step boundary, dropped the `done` step, run was incomplete). "
    "DECOMPOSE has no such cap because each subtask gets its own "
    "fresh /code with its own pre-teach + 8000-char budget. Two "
    "small recipes always beat one big truncated recipe.\n"
    "\n"
    "If you're unsure whether a task is big or small, COUNT THE FILES. "
    "More than 2 distinct file paths in your recipe -> DECOMPOSE.\n"
    "\n"
    "**SUBTASK SIZING (live observation, 2026-05-07):** smaller "
    "subtasks reliably outperform larger ones. When you DECOMPOSE, "
    "prefer 4-6 small subtasks (each touching 1 file, 3-5 STEPs) "
    "over 2-3 large ones (each touching multiple files). The chain "
    "runner has no per-subtask overhead beyond the Claude pre-teach "
    "call (~30-60s each), so 5 small subtasks at ~3 min each is "
    "FASTER than 2 large subtasks where one needs corrective_teach "
    "loops. Each smaller subtask also produces a sharper KB pattern.\n"
)


# Phase 16 Option 1 -- recipe reformat retry. Used when stepfed's
# parser extracts <2 STEP blocks from a Claude recipe; rather than
# letting stepfed fall back to the legacy free-form run_agent path
# (which doesn't pin Qwen to literal recipe commands and frequently
# does over-engineered or wrong edits), we re-ask Claude with
# explicit format-only directives.
#
# 2026-05-06 hot-fix (degeneracy guard): the original reformat prompt
# coerced Claude into recipe shape EVEN WHEN the original response
# was a legitimate refusal/clarification request. Trace SEN-b46a27cf
# (pattern id=96): Claude said "provide the emoji pair you want" with
# no STEPs; reformat then squeezed that question into a 2-STEP
# read_file+done recipe whose `done` summary was the question, which
# stepfed dutifully executed and reviewer-Claude passed because the
# file was already in target state. Result: a degenerate 1/1 PASS
# pattern with no actual edit. Fix: the reformat prompt now
# explicitly tells Claude that LEGITIMATE REFUSAL is preferred to
# fabricating a recipe, and that recipes MUST contain at least one
# edit_file or write_file step (no edit-less recipes). Both the
# prompt AND a post-reformat parsed-step check enforce this.
_REFORMAT_SYSTEM = (
    "You are reformatting a code-edit recipe to match an exact format. "
    "Your previous response did not contain parseable STEP blocks.\n"
    "\n"
    "*** IMPORTANT: legitimate refusal is BETTER than a fake recipe. "
    "If your previous response was asking for clarification, refusing "
    "due to ambiguity, or noting that the file already meets the "
    "request -- DO NOT INVENT a recipe to satisfy the format. Return "
    "your previous response UNCHANGED. The system will treat that as "
    "'no actionable recipe' and surface your reasoning to the user. "
    "Producing a degenerate read-and-done recipe with the question as "
    "the done summary is the WORST outcome -- it gets stored as a "
    "fake successful pattern.\n"
    "\n"
    "If the request IS actionable, output the recipe ONLY in this "
    "exact format. NO markdown, NO prose between STEP lines, NO "
    "numbered lists, NO commentary:\n"
    "\n"
    "STEP 1: <tool> <key>=\"<value>\" [<key2>=\"<value2>\"]\n"
    "STEP 2: <tool> <key>=\"<value>\" ...\n"
    "STEP N: done summary=\"<one-line>\"\n"
    "\n"
    "Each STEP is exactly ONE tool call. Tools available: read_file, "
    "list_dir, write_file, edit_file, run_bash, done. Use double-quoted "
    "string args; \\n for newlines, \\\\ for backslashes inside values. "
    "The final STEP is ALWAYS done(summary=\"...\"). The recipe MUST "
    "include at least one edit_file or write_file step -- a recipe of "
    "ONLY read_file + done is degenerate and will be rejected."
)


def _load_code_tiers_memo() -> str:
    """Phase 17c: read the CODE_TIERS.md tier playbook fresh on every
    /code call so curated edits take effect on the NEXT pre-teach
    without bot restart. Capped at the PERSONA_INJECT_MAX_CHARS limit
    (default 3000) to stay under Claude's prompt budget. Returns ""
    when the file is missing -- the existing PRE_TEACH_SYSTEM rules
    still cover STEP-N + DECOMPOSE; the playbook adds tier-specific
    nudges on top."""
    try:
        path = config.PERSONA_DIR / "CODE_TIERS.md"
        if not path.exists():
            return ""
        cap = config.PERSONA_INJECT_MAX_CHARS.get("CODE_TIERS.md", 3000)
        text = path.read_text(encoding="utf-8")
        if len(text) > cap:
            text = text[:cap]
        return text
    except Exception:
        # Best-effort: pre-teach must never raise from here.
        return ""


def _classify_complexity_tier(problem: str) -> tuple[str, float]:
    """Phase 17c: deterministic pre-teach complexity hint via
    `core.complexity.classify_complexity`. Returns ``(tier, score)``
    -- tier is one of {'basic', 'standard', 'advanced'}. Best-effort:
    on any failure returns ``('standard', 0.5)`` so the playbook
    still loads and Claude gets a neutral hint."""
    try:
        from core.complexity import classify_complexity
        result = classify_complexity(
            command="code",
            args={"text": problem},
            skill_name="code_assist",
        )
        return result.tier, result.score
    except Exception:
        return "standard", 0.5


async def _claude_pre_teach(
    problem: str, code_context: str | None,
    kb_patterns_block: str, project_map: str,
    trace_id: str,
) -> str:
    """Step 1 of agentic /code: Claude generates a STEP-by-STEP recipe
    BEFORE Qwen attempts. KB-augmented: prior winning recipes and known
    failure modes are injected so Claude adapts rather than re-derives.

    Phase 17c: also loads `workspace/persona/CODE_TIERS.md` and the
    complexity classifier's tier verdict, appending both to the
    PRE_TEACH_SYSTEM prompt. The tier hint nudges Claude toward
    DECOMPOSE on tier-3 (advanced) tasks instead of one-shotting them.
    """
    from core.claude_cli import ClaudeCliClient, ClaudeCliError
    # Phase 17h hot-fix: reset truncation flag at function START
    # (not after Claude returns). If we exit early via the
    # `not client.available` or `except ClaudeCliError` paths
    # below, the flag retains its value from the PRIOR call --
    # which can incorrectly trigger force-DECOMPOSE on a fresh
    # /code with stale state. Live trigger: 2026-05-07 ~02:29Z
    # log showed `FORCE-DECOMPOSE retry: prior_recipe=5 chars`,
    # which is impossible if the flag was reset properly.
    global _LAST_PRE_TEACH_TRUNCATED
    _LAST_PRE_TEACH_TRUNCATED = False
    client = ClaudeCliClient()
    if not client.available:
        return ""
    user_parts = [f"USER REQUEST:\n{problem}"]
    if project_map:
        user_parts.append(f"PROJECT FILES:\n{project_map[:1500]}")
    if code_context:
        user_parts.append(f"CODE CONTEXT:\n{code_context[:1500]}")
    if kb_patterns_block:
        user_parts.append(kb_patterns_block)
    user = "\n\n".join(user_parts)

    # Phase 17c: append CODE_TIERS.md playbook + tier hint to
    # PRE_TEACH_SYSTEM. Memo is APPENDED so the strict format rules
    # in PRE_TEACH_SYSTEM are read first; the tier guidance refines
    # those rules with task-size-aware nudges.
    tier_memo = _load_code_tiers_memo()
    tier_label, tier_score = _classify_complexity_tier(problem)
    if tier_memo:
        system_prompt = (
            PRE_TEACH_SYSTEM
            + "\n\n==== CODE_TIERS playbook (curated tier guidance) ====\n"
            + tier_memo
            + f"\n==== complexity verdict ====\n"
            + f"This task scored complexity={tier_score:.2f} which "
            + f"maps to tier='{tier_label}'. Read the matching tier "
            + f"section above and follow its action + anti-pattern "
            + f"guidance.\n"
        )
    else:
        system_prompt = PRE_TEACH_SYSTEM
    log_event(trace_id, "INFO", "skill.code_assist",
              f"claude pre-teach starting; prompt_chars={len(user)} "
              f"kb_block_chars={len(kb_patterns_block)} "
              f"tier={tier_label} score={tier_score:.2f} "
              f"tier_memo_chars={len(tier_memo)}")
    try:
        raw = await client.generate(
            prompt=user, system=system_prompt, trace_id=trace_id,
            tools=["Read", "Grep", "Glob"],
        )
    except ClaudeCliError as e:
        log_event(trace_id, "WARNING", "skill.code_assist",
                  f"pre-teach claude call failed: {e}")
        return ""
    # Extract STEP lines BEFORE truncating: tools-enabled Claude often
    # writes long narration around the actual recipe; naive truncation
    # chops the STEP block off the end.
    extracted = _extract_recipe_steps_from_text(raw)
    if extracted != raw:
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"pre-teach: extracted recipe steps from "
                  f"{len(raw)}-char response, kept {len(extracted)} chars")
    recipe = extracted.strip()
    # Phase 17h: signal truncation back to the pipeline (via
    # module-level _LAST_PRE_TEACH_TRUNCATED). Flag was already
    # reset to False at function start (above); only set True
    # below if truncation actually fires.
    if len(recipe) > RECIPE_MAX_CHARS_STEPFED:
        before = len(recipe)
        recipe = _truncate_recipe_to_steps(
            recipe, RECIPE_MAX_CHARS_STEPFED,
        )
        _LAST_PRE_TEACH_TRUNCATED = True
        log_event(trace_id, "WARNING", "skill.code_assist",
                  f"pre-teach recipe too long ({before} chars); "
                  f"truncated to {len(recipe)} chars at step boundary "
                  f"-- pipeline will trigger force-DECOMPOSE retry")
    return recipe


# Phase 17h: module-level truncation flag, set by _claude_pre_teach,
# read by _run_agentic_pipeline. Module-level so existing test mocks
# of _claude_pre_teach don't have to change their return type.
_LAST_PRE_TEACH_TRUNCATED: bool = False


_FORCE_DECOMPOSE_SYSTEM = (
    "You are re-attempting a recipe. Your previous STEP-N response "
    "was too big and got TRUNCATED at the parser cap (8000 chars). "
    "Trailing STEPs were silently dropped, including the `done` "
    "marker. The task is too big for one recipe.\n"
    "\n"
    "You MUST emit DECOMPOSE format this time. NO STEP-N. Decompose "
    "into 2-5 subtasks; each subtask must be small enough to land "
    "in <=7 STEPs on its own.\n"
    "\n"
    "Format (emit ONLY this, nothing else):\n"
    "DECOMPOSE\n"
    "- /code <concrete subtask 1>\n"
    "- /code <concrete subtask 2>\n"
    "- /code <concrete subtask 3>\n"
    "\n"
    "Rules:\n"
    "1. First line: literally `DECOMPOSE` (no markdown).\n"
    "2. 2-5 bullets, each starting with `- /code ` followed by a\n"
    "   specific subtask phrased as a /code prompt.\n"
    "3. Subtasks ordered: each later one assumes earlier ones shipped.\n"
    "4. NO recipe STEPs. NO summary. NO commentary.\n"
    "5. Do NOT just repeat your previous recipe in DECOMPOSE shape --\n"
    "   actually break it into independent subtasks.\n"
)


async def _claude_force_decompose(
    problem: str, prior_recipe: str, trace_id: str,
) -> str:
    """Phase 17h: re-ask Claude with DECOMPOSE-or-bail framing after
    a pre-teach truncation. Returns a (hopefully) DECOMPOSE-formatted
    response. No tools (Claude already explored the codebase in the
    truncated pre-teach call). Best-effort: any failure returns ""
    and the caller falls back to the original truncated recipe."""
    from core.claude_cli import ClaudeCliClient, ClaudeCliError
    client = ClaudeCliClient()
    if not client.available:
        return ""
    user = (
        f"USER REQUEST:\n{problem}\n\n"
        f"YOUR PREVIOUS (TRUNCATED) RECIPE:\n{prior_recipe[:3000]}\n\n"
        f"That recipe exceeded the 8000-char cap and got truncated. "
        f"Decompose into 2-5 subtasks now. DECOMPOSE format only."
    )
    log_event(trace_id, "INFO", "skill.code_assist",
              f"FORCE-DECOMPOSE retry: prior_recipe={len(prior_recipe)} chars")
    try:
        text = await client.generate(
            prompt=user, system=_FORCE_DECOMPOSE_SYSTEM,
            trace_id=trace_id, tools=None,
        )
    except ClaudeCliError as e:
        log_event(trace_id, "WARNING", "skill.code_assist",
                  f"FORCE-DECOMPOSE failed: {e}")
        return ""
    return text.strip()


def _recipe_has_edit_step(recipe: str) -> bool:
    """Phase 16 hot-fix: does this recipe contain at least one
    `edit_file` or `write_file` step?

    Used in two places:
      1. After Option 1's reformat returns a recipe -- if reformat
         produced an edit-less recipe (degenerate refusal coerced
         into shape), reject it as if reformat failed.
      2. After stepfed runs -- if the executed recipe had no
         edit/write step, fail the attempt rather than letting
         reviewer-Claude pass it on the basis of "file is already
         in target state."

    Returns False on parse failure (best-effort, never raises)."""
    try:
        from core.qwen_agent import (
            _parse_recipe_steps, _parse_step_text_to_tool_call,
        )
        for step_text in _parse_recipe_steps(recipe or ""):
            call = _parse_step_text_to_tool_call(step_text)
            if not call:
                continue
            tool = (
                (call.get("function", {}).get("name")
                 if "function" in call
                 else call.get("name", "")) or ""
            ).lower()
            if tool in ("edit_file", "write_file"):
                return True
    except Exception:
        pass
    return False


async def _claude_reformat_recipe(
    prior_recipe: str, problem: str, trace_id: str,
) -> str:
    """Phase 16 Option 1: re-ask Claude to reformat a recipe that
    failed local parse (<2 STEP blocks).

    The legacy run_agent fallback path inside run_agent_stepfed gives
    Qwen a free-form conversation and tends to do over-engineered or
    wrong edits. A targeted reformat call to Claude with strict
    format-only directives is faster (~15s) and produces a recipe the
    stepfed executor can pin Qwen to.

    No tools enabled -- this is pure text reshape, not exploration.
    On any failure (Claude unavailable, timeout, exception, empty
    response) returns "" and the caller falls through to the existing
    legacy fallback as a last-resort safety net."""
    from core.claude_cli import ClaudeCliClient, ClaudeCliError
    client = ClaudeCliClient()
    if not client.available:
        return ""
    user = (
        f"PROBLEM:\n{problem}\n"
        f"\n"
        f"YOUR PREVIOUS RESPONSE (could not be parsed into STEP blocks):\n"
        f"---\n{prior_recipe}\n---\n"
        f"\n"
        f"Reformat that into the required STEP N: format. Same intent, "
        f"different shape. Output the recipe ONLY -- nothing else."
    )
    log_event(trace_id, "INFO", "skill.code_assist",
              f"reformat: re-asking claude; "
              f"prior_recipe={len(prior_recipe)} chars")
    try:
        raw = await client.generate(
            prompt=user, system=_REFORMAT_SYSTEM, trace_id=trace_id,
            tools=[],
        )
    except ClaudeCliError as e:
        log_event(trace_id, "WARNING", "skill.code_assist",
                  f"reformat claude call failed: {e}")
        return ""
    extracted = _extract_recipe_steps_from_text(raw)
    if extracted != raw:
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"reformat: extracted recipe steps from "
                  f"{len(raw)}-char response, kept {len(extracted)} chars")
    recipe = extracted.strip()
    if len(recipe) > RECIPE_MAX_CHARS_STEPFED:
        before = len(recipe)
        recipe = _truncate_recipe_to_steps(
            recipe, RECIPE_MAX_CHARS_STEPFED,
        )
        log_event(trace_id, "WARNING", "skill.code_assist",
                  f"reformat recipe too long ({before} chars); "
                  f"truncated to {len(recipe)} chars at step boundary")
    return recipe


async def _claude_review(
    problem: str, recipe: str, qwen_summary: str,
    qwen_session: list[dict], git_diff: str, trace_id: str,
) -> dict:
    """Step 3 of agentic /code: Claude reviews Qwen's session + diff
    and produces a verdict. Returns {"verdict": "pass"|"fail",
    "reasoning": "..."}."""
    from core.claude_cli import ClaudeCliClient, ClaudeCliError
    client = ClaudeCliClient()
    if not client.available:
        return {"verdict": "unknown", "reasoning": "claude CLI unavailable"}

    # Compact the session: tool name + arg keys, drop big result blobs
    compact_trace = []
    for s in qwen_session[:30]:
        compact_trace.append(
            f"step {s.get('step')}: {s.get('tool')}({list((s.get('args') or {}).keys())}) "
            f"-> {'OK' if 'ok' in (s.get('result') or {}) else 'ERR: ' + str((s.get('result') or {}).get('error', ''))[:80]}"
        )
    trace_str = "\n".join(compact_trace)

    system = (
        "You are reviewing a junior agent's code change. YOU HAVE TOOLS: "
        "Read, Grep, Glob, Bash. USE THEM TO VERIFY -- do NOT infer "
        "from the diff alone.\n"
        "\n"
        "Required workflow:\n"
        "1. Read each file the RECIPE references -- see the FULL "
        "current state, not just the diff window.\n"
        "2. Grep for any function/symbol the diff references to confirm "
        "it actually exists somewhere in the codebase.\n"
        "3. Optionally `Bash python -c \"import <module>; ...\"` to "
        "confirm the change at least imports cleanly.\n"
        "4. **PHASE 17f -- RECIPE-PROMISE VERIFICATION (load-bearing).** "
        "For EACH `edit_file`/`write_file` step in the RECIPE GIVEN, "
        "READ or Grep the target file and confirm the claimed change "
        "is actually present. Silent no-ops (anchor mismatch, wrong "
        "path, malformed args) are SILENT FAILURES that show up only "
        "if you check. A recipe step that didn't actually land is a "
        "FAIL even if OTHER steps succeeded. Cross-reference the "
        "TOOL TRACE -- any step with ERR is a failed promise.\n"
        "5. THEN decide: pass (EVERY recipe step's claimed change is "
        "reflected in the actual file state AND the user request is "
        "satisfied) or fail (any silent no-op, partial wiring, "
        "missing entry, or unmet promise).\n"
        "\n"
        "BUDGET: cap your tool use at ~7 calls. Don't audit the whole "
        "repo -- just verify each recipe step's specific claim.\n"
        "\n"
        "Output ONE JSON object on the final line, with two fields: "
        "\"verdict\" (\"pass\" or \"fail\") and \"reasoning\" (one "
        "sentence quoting what you verified, INCLUDING which recipe "
        "step's claim you confirmed -- e.g. 'Recipe step 3 (edit "
        "core/config.py to add /qcode) verified: Grep core/config.py "
        "shows /qcode in COMMAND_AGENT_MAP at line 54'). JSON only, no "
        "markdown fences."
    )
    user = (
        f"USER REQUEST:\n{problem}\n\n"
        f"RECIPE GIVEN:\n{recipe[:1000]}\n\n"
        f"JUNIOR'S SUMMARY:\n{qwen_summary[:600]}\n\n"
        f"TOOL TRACE (compact):\n{trace_str[:2000]}\n\n"
        f"GIT DIFF:\n{git_diff[:3000]}"
    )
    try:
        text = await client.generate(
            prompt=user, system=system, trace_id=trace_id,
            tools=["Read", "Grep", "Glob", "Bash"],
        )
    except ClaudeCliError as e:
        return {"verdict": "unknown",
                "reasoning": f"claude review failed: {e}"}
    parsed = _extract_json(text)
    if not isinstance(parsed, dict) or "verdict" not in parsed:
        return {"verdict": "unknown",
                "reasoning": text.strip()[:300]}
    verdict = str(parsed.get("verdict", "unknown")).lower()
    if verdict not in ("pass", "fail"):
        verdict = "unknown"
    return {
        "verdict": verdict,
        "reasoning": str(parsed.get("reasoning", ""))[:500],
    }


CORRECTIVE_SYSTEM = (
    "You are a senior engineer. The JUNIOR just attempted a task "
    "following the recipe you previously gave it. Verdict: FAIL. "
    "YOU HAVE TOOLS: Read, Grep, Glob. USE THEM to see the current "
    "file state, then produce a sharper recipe.\n"
    "\n"
    "Workflow:\n"
    "1. USE Read on target files -- see exact current bytes.\n"
    "2. Diagnose the prior failure from session + diff.\n"
    "3. Output a NEW recipe.\n"
    "\n"
    "Junior's tools (it has NO others) -- USE THESE EXACT SIGNATURES:\n"
    "  read_file(path=...)\n"
    "  list_dir(path=...)\n"
    "  write_file(path=..., content=...)         <- TWO args\n"
    "  edit_file(path=..., old=..., new=...)     <- THREE args\n"
    "  run_bash(command=...)\n"
    "  done(summary=...)\n"
    "\n"
    "**PHASE 17f -- TOOL-SIGNATURE DISCIPLINE (load-bearing).**\n"
    "NEVER use arg names like `text=`, `body=`, `code=`, or any\n"
    "abbreviation. The arg names above are EXACT. The previous\n"
    "live failure (2026-05-07 ~01:48Z) had Claude write\n"
    "  edit_file path=\"x\" text=\"...\"\n"
    "which silently failed with 'unexpected keyword argument text'.\n"
    "Use ONLY the arg names listed above. Same on every call.\n"
    "\n"
    "RULES:\n"
    "1. Each step is exactly one tool call. No prose between steps.\n"
    "2. PREFER write_file (whole-file rewrite). If prior verdict says "
    "'old not found' or 'no diff' or 'missing 2 required positional "
    "arguments' -- the prior recipe used edit_file with hallucinated "
    "`old` OR malformed args. DO NOT repeat that. Use write_file "
    "with the COMPLETE new content.\n"
    "3. If you use edit_file, `old` MUST be copy-pasted from a Read "
    "result you ran in this turn. Include 3-5 lines of context.\n"
    "4. Final step is ALWAYS done(summary=\"...\").\n"
    "5. If the junior CREATED a new file but never WIRED it in, your "
    "recipe MUST include a write_file rewrite of the call-site file "
    "with the wiring in place.\n"
    "6. If prior session had consecutive failures, shorten and prefer "
    "write_file.\n"
    "7. NO triple quotes. Use \\n for newlines.\n"
    "\n"
    "Output the new recipe ONLY in STEP N: tool args format. No "
    "preamble, commentary, or markdown."
)


async def _claude_corrective_teach(
    problem: str, prior_recipe: str, prior_session: list[dict],
    prior_diff: str, prior_verdict_reasoning: str,
    kb_patterns_block: str, project_map: str,
    trace_id: str,
    files_already_read: set[str] | None = None,
) -> str:
    """Generate a sharper recipe after a failed attempt. Differs from
    pre_teach in that it has the prior attempt's full context.

    Phase 15d: ``files_already_read`` lists project-relative paths
    the prior attempt(s) demonstrably touched (extracted from the
    recipe + review reasoning). The tree was reset to base_sha
    between attempts so those files are unchanged -- Claude can
    skip re-reading them, cutting attempt 2-5 latency by ~50%."""
    from core.claude_cli import ClaudeCliClient, ClaudeCliError
    client = ClaudeCliClient()
    if not client.available:
        return ""
    compact_trace = []
    for s in prior_session[:30]:
        result = s.get("result") or {}
        outcome = "OK" if "ok" in result else (
            "ERR: " + str(result.get("error", ""))[:80]
        )
        compact_trace.append(
            f"step {s.get('step')}: {s.get('tool')}"
            f"({list((s.get('args') or {}).keys())}) -> {outcome}"
        )
    trace_str = "\n".join(compact_trace)
    parts = [
        f"USER REQUEST:\n{problem}",
        f"PRIOR RECIPE you gave:\n{prior_recipe[:1200]}",
        f"WHAT JUNIOR DID (compact tool trace):\n{trace_str[:1500]}",
        f"DIFF JUNIOR PRODUCED:\n{prior_diff[:2000]}",
        f"REVIEWER VERDICT: FAIL -- {prior_verdict_reasoning[:400]}",
    ]
    # Phase 15d: hint Claude that prior exploration is still valid.
    # The tree was reset between attempts, so files are exactly as
    # they were when Claude read them. Re-reading is wasted Claude
    # time on the corrective_teach call. The hint is advisory:
    # Claude can still Read again if it wants to verify.
    if files_already_read:
        sorted_paths = sorted(files_already_read)[:20]
        parts.append(
            "FILES YOU'VE ALREADY READ IN PRIOR ATTEMPTS (tree was "
            "reset to the same base SHA, so their contents are "
            "UNCHANGED -- skip re-reading unless you genuinely "
            "need to re-verify):\n"
            + "\n".join(f"- {p}" for p in sorted_paths)
        )
    if project_map:
        parts.append(f"PROJECT FILES:\n{project_map[:1200]}")
    if kb_patterns_block:
        parts.append(kb_patterns_block)
    user = "\n\n".join(parts)
    log_event(trace_id, "INFO", "skill.code_assist",
              f"corrective teach starting; prompt_chars={len(user)}")
    try:
        raw = await client.generate(
            prompt=user, system=CORRECTIVE_SYSTEM, trace_id=trace_id,
            tools=["Read", "Grep", "Glob"],
        )
    except ClaudeCliError as e:
        log_event(trace_id, "WARNING", "skill.code_assist",
                  f"corrective teach claude call failed: {e}")
        return ""
    extracted = _extract_recipe_steps_from_text(raw)
    if extracted != raw:
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"corrective: extracted recipe steps from "
                  f"{len(raw)}-char response, kept {len(extracted)} chars")
    recipe = extracted.strip()
    if len(recipe) > RECIPE_MAX_CHARS_STEPFED:
        before = len(recipe)
        recipe = _truncate_recipe_to_steps(
            recipe, RECIPE_MAX_CHARS_STEPFED,
        )
        log_event(trace_id, "WARNING", "skill.code_assist",
                  f"corrective recipe too long ({before} chars); "
                  f"truncated to {len(recipe)} chars at step boundary")
    return recipe


async def _claude_teach(
    problem: str, code_context: str | None, qwen_code: str,
    stdout: str, stderr: str, return_code: int, trace_id: str,
) -> dict | None:
    from core.claude_cli import ClaudeCliClient, ClaudeCliError
    # Probe via the local alias so test monkeypatches still work.
    if _find_claude_cli() is None:
        log_event(
            trace_id, "WARNING", "skill.code_assist",
            "Claude teaching skipped: `claude` CLI not found",
        )
        return None
    client = ClaudeCliClient()
    if not client.available:
        log_event(
            trace_id, "WARNING", "skill.code_assist",
            "Claude teaching skipped: `claude` CLI not found",
        )
        return None
    user_prompt = _claude_user_prompt(
        problem, code_context, qwen_code, stdout, stderr, return_code,
    )
    try:
        raw_text = await client.generate(
            prompt=user_prompt, system=CLAUDE_SYSTEM, trace_id=trace_id,
        )
    except ClaudeCliError as e:
        log_event(
            trace_id, "WARNING", "skill.code_assist",
            f"claude CLI call failed: {e}",
        )
        return None
    parsed = _extract_json(raw_text)
    if not parsed or "solution" not in parsed:
        log_event(
            trace_id, "WARNING", "skill.code_assist",
            f"claude response not parseable as expected JSON; "
            f"raw[:300]={raw_text[:300]!r}",
        )
        return {
            "solution": raw_text,
            "teaching_note": "(unparsed)",
            "pattern": "(unparsed)",
        }
    return parsed


# ----------------------- Skill ---------------------------------------

class CodeAssistSkill(BaseSkill):
    name: ClassVar[str] = "code_assist"
    description: ClassVar[str] = (
        "Solves coding problems using Qwen with executor validation "
        "and Claude-powered teaching fallback; accumulates patterns "
        "in the knowledge base"
    )
    version: ClassVar[str] = "1.0.0"
    requires_gpu: ClassVar[bool] = True
    input_schema: ClassVar[type[BaseModel]] = CodeAssistInput
    output_schema: ClassVar[type[BaseModel]] = CodeAssistOutput

    def validate_input(self, raw: dict) -> BaseModel:
        # Router free-text or with --context flag.
        keys = set(raw.keys())
        if keys == {"text"}:
            return CodeAssistInput(problem=raw["text"])
        if "text" in raw and "context" in raw:
            return CodeAssistInput(
                problem=raw["text"], code_context=raw["context"],
            )
        # Already structured input (direct skill invocation in tests).
        return CodeAssistInput(**raw)

    async def execute(
        self, input_data: BaseModel, trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        if not isinstance(input_data, CodeAssistInput):
            raise SkillError(
                self.name,
                f"expected CodeAssistInput, got "
                f"{type(input_data).__name__}",
                trace_id,
            )
        # ============================================================
        # AGENTIC /code (Phase 9 final architecture):
        #   1. Claude PRE-TEACHES: emits a step-by-step recipe
        #   2. Qwen-coder EXECUTES: tool-calling loop, edits files
        #   3. Claude REVIEWS: pass/fail verdict on Qwen's diff
        #   4. KB stores the session as a learnable pattern
        # Claude is teacher + reviewer, never the main player.
        # If Qwen still can't fix it after teaching, we record a
        # limitation and tell the user. Claude does NOT finish the job.
        # ============================================================
        kb = KnowledgeBase()
        kb_query = input_data.problem
        kb_entries = kb.search(kb_query, max_results=8)
        # KB filter: include all patterns (the quality gate during
        # store already filters out garbage via _is_real_solution).
        # NOTE: an earlier usage_count > 0 filter created a deadlock --
        # fresh patterns stayed at usage_count=0 forever because they
        # were never surfaced. Trust the store-time quality gate.
        # Phase 14a: needs_reteach patterns are known-bad teachers
        # (graduation tests showed Qwen can't reproduce solo). Drop
        # them from kb_patterns AND treat them as a Claude-escalation
        # signal -- Qwen would learn the wrong thing from them.
        reteach_hits = [
            e for e in kb_entries
            if e.category == "pattern" and e.needs_reteach
        ]
        kb_patterns = [
            e for e in kb_entries
            if e.category == "pattern" and not e.needs_reteach
        ][:3]
        # Cap limitations at 1 -- one warning is enough.
        kb_limitations = [
            e for e in kb_entries if e.category == "limitation"
        ][:1]
        kb_ids = [e.id for e in kb_entries]
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"AGENTIC /code starting; KB matched_ids={kb_ids} "
            f"patterns={len(kb_patterns)} "
            f"limitations={len(kb_limitations)} "
            f"needs_reteach_hits={len(reteach_hits)}",
        )

        if _find_claude_cli() is not None:
            return await _run_agentic_pipeline(
                input_data, trace_id, kb, kb_patterns, kb_limitations,
                kb_ids, context,
            )

        # No Claude CLI available -- fall back to legacy Qwen text-gen
        # teaching loop below. (Should be rare on the user's setup.)
        log_event(
            trace_id, "WARNING", "skill.code_assist",
            "Claude CLI unavailable -- falling back to legacy "
            "text-only teaching loop",
        )
        # ---- FALLBACK: original Qwen + teaching path ----
        # Agent may pin a model name (registry name like "qwen-coder").
        # Resolve to backend model_id; fall back to DEFAULT_MODEL if
        # neither agent pin nor registry lookup yields anything.
        model_name_or_id: str = (
            (context or {}).get("model") or config.DEFAULT_MODEL
        )
        try:
            from core.llm import INFERENCE_CLIENT
            cfg = INFERENCE_CLIENT.model_registry.get(model_name_or_id)
            model = cfg.model_id if cfg else model_name_or_id
        except Exception:
            model = model_name_or_id
        kb = KnowledgeBase()

        # ---- Step 1: KB lookup ----
        kb_query = input_data.problem
        kb_entries = kb.search(kb_query, max_results=5)
        # Only patterns (working solutions) make good few-shot examples;
        # limitation entries are noise here.
        good_entries = [e for e in kb_entries if e.category == "pattern"]
        kb_examples = _format_kb_examples(good_entries)
        kb_ids = [e.id for e in kb_entries]
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"KB lookup query={kb_query[:80]!r} "
            f"matched_ids={kb_ids} "
            f"few_shot_examples={len(good_entries[:3])}",
        )

        attempts = 0

        # ---- Step 2: Qwen first attempt ----
        attempts += 1
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"qwen attempt 1 model={model}",
        )
        try:
            qwen_raw = await asyncio.to_thread(
                _qwen_generate,
                QWEN_SYSTEM_BASE,
                _qwen_user_prompt(
                    input_data.problem, input_data.code_context,
                    kb_examples,
                ),
                trace_id, model,
            )
        except LLMError as e:
            raise SkillError(
                self.name, f"Qwen call failed: {e}", trace_id,
            ) from e

        parsed = _extract_json(qwen_raw)
        qwen_code = _coerce_str(
            (parsed or {}).get("code")
            if isinstance(parsed, dict) else None
        )
        qwen_explanation = _coerce_str(
            (parsed or {}).get("explanation", "")
            if isinstance(parsed, dict) else ""
        )
        if not qwen_code:
            qwen_code = _coerce_str(_extract_code_fallback(qwen_raw))
        qwen_code = _clean_solution_text(qwen_code)

        # ---- Step 2b: Validate ----
        if qwen_code:
            log_event(
                trace_id, "INFO", "skill.code_assist",
                "validating qwen attempt 1 via code_execute",
            )
            passed, exec_result = await _validate_code(
                qwen_code, trace_id,
            )
            # Quality gate: rc=0 alone isn't enough; reject degenerate
            # solutions like '200' that parse and run but solve nothing.
            if passed and not _is_real_solution(qwen_code):
                log_event(
                    trace_id, "WARNING", "skill.code_assist",
                    f"qwen attempt 1 passed execution but failed "
                    f"quality gate (len={len(qwen_code)}); "
                    f"treating as failure",
                )
                passed = False
            if passed:
                log_event(
                    trace_id, "INFO", "skill.code_assist",
                    "qwen attempt 1 passed -> storing pattern",
                )
                # Store pattern.
                tags = _extract_tags(input_data.problem, qwen_code)
                kb.add_pattern(
                    tags=tags,
                    problem_summary=_summarize_problem(input_data.problem),
                    solution_code=qwen_code,
                    solution_pattern=(
                        qwen_explanation or "qwen one-shot solution"
                    ),
                    explanation=(
                        qwen_explanation
                        or "qwen produced working code on first attempt"
                    ),
                    trace_id=trace_id,
                )
                return CodeAssistOutput(
                    solution=qwen_code,
                    explanation=qwen_explanation
                        or "Qwen solved on first attempt.",
                    solved_by="qwen",
                    teaching_note=None,
                    knowledge_entries_used=kb_ids,
                    attempts=attempts,
                    validated=True,
                    execution_result=(
                        exec_result.get("stdout", "")[:1000]
                    ),
                )
        else:
            exec_result = {
                "stdout": "", "stderr": "qwen returned no parseable code",
                "return_code": -1,
            }
            log_event(
                trace_id, "WARNING", "skill.code_assist",
                "qwen produced no parseable code",
            )

        # ---- Step 3: Claude teaches (if available) ----
        teaching = await _claude_teach(
            input_data.problem, input_data.code_context,
            qwen_code or "",
            exec_result.get("stdout", ""),
            exec_result.get("stderr", ""),
            int(exec_result.get("return_code", -1)),
            trace_id,
        )

        if teaching is None:
            # Claude CLI unavailable or call failed: graceful degradation.
            warning = (
                "Claude CLI teaching unavailable "
                "(CLI not found or call failed). "
                "This is Qwen's unvalidated attempt."
            )
            return CodeAssistOutput(
                solution=qwen_code or "",
                explanation=(qwen_explanation + "\n\n" + warning)
                    if qwen_explanation else warning,
                solved_by="qwen",
                teaching_note=None,
                knowledge_entries_used=kb_ids,
                attempts=attempts,
                validated=False,
                execution_result=None,
            )

        claude_solution = _clean_solution_text(
            _coerce_str(teaching.get("solution", ""))
        )
        teaching_note = _coerce_str(teaching.get("teaching_note", ""))
        pattern_recipe = _coerce_str(teaching.get("pattern", ""))
        worked_example = _coerce_str(
            teaching.get("worked_example", "")
        )
        # Combine recipe + worked example into a single solution_pattern
        # string so KB retrieval can show both to Qwen as one block.
        pattern = pattern_recipe + (
            f"\n\nWORKED EXAMPLE:\n{worked_example}"
            if worked_example else ""
        )

        # ---- Step 4: Qwen re-attempt with teaching ----
        attempts += 1
        log_event(
            trace_id, "INFO", "skill.code_assist",
            "qwen attempt 2 with teaching context",
        )
        try:
            qwen2_raw = await asyncio.to_thread(
                _qwen_generate,
                QWEN_TAUGHT_SYSTEM,
                _qwen_taught_user_prompt(
                    input_data.problem, input_data.code_context,
                    teaching_note, pattern, claude_solution,
                ),
                trace_id, model,
            )
            parsed2 = _extract_json(qwen2_raw)
            qwen2_code = _coerce_str(
                (parsed2 or {}).get("code")
                if isinstance(parsed2, dict) else None
            )
            qwen2_explanation = _coerce_str(
                (parsed2 or {}).get("explanation", "")
                if isinstance(parsed2, dict) else ""
            )
            if not qwen2_code:
                qwen2_code = _coerce_str(
                    _extract_code_fallback(qwen2_raw)
                )
            qwen2_code = _clean_solution_text(qwen2_code)
        except LLMError as e:
            log_event(
                trace_id, "WARNING", "skill.code_assist",
                f"qwen retry call failed: {e}",
            )
            qwen2_code = None
            qwen2_explanation = ""

        if qwen2_code:
            passed2, exec_result2 = await _validate_code(
                qwen2_code, trace_id,
            )
            if passed2 and not _is_real_solution(qwen2_code):
                log_event(
                    trace_id, "WARNING", "skill.code_assist",
                    f"qwen_taught passed execution but failed quality "
                    f"gate (len={len(qwen2_code)}); falling through "
                    f"to claude_direct",
                )
                passed2 = False
            if passed2:
                log_event(
                    trace_id, "INFO", "skill.code_assist",
                    "qwen_taught succeeded -> storing pattern",
                )
                tags = _extract_tags(input_data.problem, qwen2_code)
                kb.add_pattern(
                    tags=tags,
                    problem_summary=_summarize_problem(input_data.problem),
                    solution_code=qwen2_code,
                    solution_pattern=pattern,
                    explanation=teaching_note,
                    trace_id=trace_id,
                )
                return CodeAssistOutput(
                    solution=qwen2_code,
                    explanation=qwen2_explanation
                        or teaching_note or "Qwen learned from teaching.",
                    solved_by="qwen_taught",
                    teaching_note=teaching_note,
                    knowledge_entries_used=kb_ids,
                    attempts=attempts,
                    validated=True,
                    execution_result=(
                        exec_result2.get("stdout", "")[:1000]
                    ),
                )

        # ---- Step 4 fail: return Claude direct, store as pattern ----
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"qwen could not learn pattern for: "
            f"{_summarize_problem(input_data.problem)} -- "
            f"returning claude_direct, storing pattern",
        )
        # Validate Claude's solution if possible (for execution_result).
        claude_passed, claude_exec = (False, {"stdout": ""})
        if claude_solution.strip():
            try:
                claude_passed, claude_exec = await _validate_code(
                    claude_solution, trace_id,
                )
            except Exception as e:
                log_event(
                    trace_id, "WARNING", "skill.code_assist",
                    f"could not validate claude solution: {e}",
                )

        tags = _extract_tags(input_data.problem, claude_solution)
        kb.add_pattern(
            tags=tags,
            problem_summary=_summarize_problem(input_data.problem),
            solution_code=claude_solution,
            solution_pattern=pattern,
            explanation=teaching_note,
            trace_id=trace_id,
        )
        # Also record the limitation: Qwen couldn't learn this even
        # with teaching.
        kb.add_limitation(
            tags=tags + ["qwen_limitation"],
            problem_summary=(
                "Qwen could not learn: "
                + _summarize_problem(input_data.problem)
            ),
            explanation=teaching_note
                or "Qwen failed to apply Claude's teaching.",
            trace_id=trace_id,
        )

        return CodeAssistOutput(
            solution=claude_solution,
            explanation=teaching_note
                or "Claude provided the solution directly.",
            solved_by="claude_direct",
            teaching_note=teaching_note,
            knowledge_entries_used=kb_ids,
            attempts=attempts,
            validated=claude_passed,
            execution_result=(
                claude_exec.get("stdout", "")[:1000]
                if claude_passed else None
            ),
        )


# ----------------------------------------------------------------------
# Phase 9 final: agentic /code pipeline (Claude teacher + Qwen worker)
# ----------------------------------------------------------------------

async def _git_snapshot(trace_id: str) -> str:
    """Capture HEAD before /code runs so we can compute diff after."""
    proc = await asyncio.create_subprocess_shell(
        "git rev-parse HEAD",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(config.PROJECT_ROOT),
    )
    out, _ = await proc.communicate()
    return out.decode("utf-8", "replace").strip()


async def _git_commit_changes(trace_id: str, message: str) -> None:
    """No-op since Phase 10 -- the owner commits manually now.

    Earlier this auto-committed Qwen's working-tree changes (both
    intermediate "agentic /code attempt N" commits and a final
    "/code: <problem>" commit on success). That polluted git
    history during concurrent edits and was disabled by owner
    request after Phase 10.

    Diff helpers were updated to compare against the working tree
    (with ``git add -N`` for untracked) so attempt diffs still work
    without a commit boundary.
    """
    log_event(
        trace_id, "DEBUG", "skill.code_assist",
        f"auto-commit disabled (owner commits manually); "
        f"would have been: {message[:80]!r}",
    )
    # Stage intent-to-add for any new files so the diff helpers below
    # can see them without actually committing.
    proc = await asyncio.create_subprocess_shell(
        "git add -N -- core skills agents tests workspace",
        cwd=str(config.PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()


async def _git_commit_for_graduation(
    trace_id: str, message: str,
) -> str | None:
    """DISABLED 2026-05-06 per owner directive: NO AUTO-COMMITS.

    The only commit path in Sentinel is the manual /commit Telegram
    command. This function is preserved as a no-op so callers don't
    need conditional checks; it logs a debug line and returns None.

    Trade-off: graduation's tree-state discipline (Phase 15e) was
    designed around this commit existing. With auto-commit disabled,
    graduation falls back to the pre-15e stash-only restoration
    path -- if `git stash push -u` silently fails to capture a
    change, the user's /code edits CAN be wiped by graduation's
    `reset --hard base_sha`. This was the original Phase 15e bug.
    Phase 17 candidate: redesign graduation to never need a commit
    (e.g., serialize the dirty diff to a temp file before reset,
    apply it after).

    Original docstring (preserved for reference):
        Phase 15e: REAL commit -- distinct from _git_commit_changes
        (the Phase 10 no-op). Committed /code's working-tree changes
        before graduation so pre_grad_sha included them and the
        final reset --hard pre_grad_sha RESTORED instead of REVERTED.
    """
    log_event(
        trace_id, "DEBUG", "skill.code_assist",
        f"grad-snapshot disabled (NO AUTO-COMMITS owner directive); "
        f"would have been: {message[:80]!r}",
    )
    return None
    # ----- Original implementation kept below, unreachable.
    # If owner re-enables auto-commits, delete the early `return None`
    # above. Until then, the body is dead code preserved for context.
    scope = "core skills agents tests interfaces workspace"
    add_proc = await asyncio.create_subprocess_shell(
        f"git add -- {scope}",
        cwd=str(config.PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    add_out, add_err = await add_proc.communicate()
    if add_proc.returncode != 0:
        log_event(
            trace_id, "WARNING", "skill.code_assist",
            f"grad-snapshot git add failed: "
            f"{add_err.decode('utf-8', 'replace')[:200]}",
        )
        return None
    # Check whether anything was actually staged. `git diff --cached
    # --quiet` exits 0 when there are no staged changes, 1 when there
    # are. We want the latter to proceed.
    diff_proc = await asyncio.create_subprocess_shell(
        "git diff --cached --quiet",
        cwd=str(config.PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await diff_proc.communicate()
    if diff_proc.returncode == 0:
        log_event(
            trace_id, "DEBUG", "skill.code_assist",
            "grad-snapshot: nothing to commit (no staged changes)",
        )
        return None
    # Commit using a sentinel identity so users can `git log
    # --invert-grep --grep=sentinel-grad` if they want to filter
    # them out. -n skips pre-commit hooks (this is a system commit,
    # not a user commit) but signing/etc. are not bypassed.
    commit_proc = await asyncio.create_subprocess_exec(
        "git",
        "-c", "user.email=sentinel-grad@sentinel.local",
        "-c", "user.name=Sentinel-Grad-Snapshot",
        "commit", "-m", f"sentinel-grad: {message[:80]}",
        cwd=str(config.PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    c_out, c_err = await commit_proc.communicate()
    if commit_proc.returncode != 0:
        log_event(
            trace_id, "WARNING", "skill.code_assist",
            f"grad-snapshot git commit failed (rc="
            f"{commit_proc.returncode}): "
            f"{c_err.decode('utf-8', 'replace')[:240]}",
        )
        return None
    sha = await _git_snapshot(trace_id)
    log_event(
        trace_id, "INFO", "skill.code_assist",
        f"grad-snapshot committed at {sha[:8]} -- "
        f"graduation can now safely tree-reset",
    )
    return sha


async def _git_diff_stat(
    since_sha: str, paths: list[str] | None = None,
) -> str:
    """Working-tree diff (no HEAD ref) so Qwen's uncommitted changes
    show up. _git_commit_changes does git add -N first so new files
    appear here too. Owner-manual-commit-only flow.

    Phase 16 Batch C: ``paths`` scopes the stat to specific files so
    the user-visible "Files changed" output isn't polluted by
    unrelated dirty work in the tree (e.g. leftover workspace/
    artifacts from a prior /code or stress run that aren't in
    `_COMMIT_INCLUDE` scope yet)."""
    if paths:
        cmd_args = ["git", "diff", "--stat", since_sha, "--", *paths]
        proc = await asyncio.create_subprocess_exec(
            *cmd_args,
            cwd=str(config.PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        proc = await asyncio.create_subprocess_shell(
            f"git diff --stat {since_sha}",
            cwd=str(config.PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    out, _ = await proc.communicate()
    return out.decode("utf-8", "replace").strip()


async def _git_diff_full(
    since_sha: str, paths: list[str] | None = None,
) -> str:
    """Phase 16 Batch C: optional ``paths`` scopes the diff to recipe-
    touched files so the stored solution_code isn't polluted with
    unrelated dirty work in the tree (which then tanks future
    skip-path diff-match against this pattern via 1/N Jaccard)."""
    if paths:
        cmd_args = ["git", "diff", since_sha, "--", *paths]
        proc = await asyncio.create_subprocess_exec(
            *cmd_args,
            cwd=str(config.PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        proc = await asyncio.create_subprocess_shell(
            f"git diff {since_sha}",
            cwd=str(config.PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    out, _ = await proc.communicate()
    return out.decode("utf-8", "replace")[:8000].strip()


async def _chain_child_auto_commit(
    recipe_paths: list[str],
    child_task_id: str,
    parent_task_id: str,
    problem: str,
    trace_id: str,
) -> bool:
    """Phase 17d -- scoped exception to NO-AUTO-COMMITS for chain
    runner subtasks. Commits ONLY the recipe-touched paths so
    unrelated dirty work in the tree isn't swept up. Returns True
    on success, False otherwise (always best-effort, never raises).

    Why scoped to recipe_paths instead of _COMMIT_INCLUDE wide
    scope: the wide scope (used by manual /commit) sweeps EVERY
    file under core/, skills/, agents/, etc. -- if the user has
    in-progress edits anywhere, they'd get committed too. Recipe
    paths are exactly what the child task touched -- nothing more.

    Commit identity: chain-child@sentinel.local + message prefix
    'sentinel-chain:' so users can `git log --grep=sentinel-chain`
    to see chain-runner commits, and `git revert` / `/revert chain`
    can target them precisely.
    """
    import subprocess

    if not recipe_paths:
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"CHAIN-COMMIT skipped (no recipe paths) "
            f"child={child_task_id[:12]} parent={parent_task_id[:12]}",
        )
        return False

    cwd = str(config.PROJECT_ROOT)
    summary = problem[:60].replace("\n", " ").strip()
    msg = (
        f"sentinel-chain: child {child_task_id[:8]} of "
        f"parent {parent_task_id[:8]} -- {summary}"
    )

    async def _run(*args) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            "git", *args, cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return (
            proc.returncode or 0,
            (out.decode("utf-8", "replace")
             + err.decode("utf-8", "replace")),
        )

    # Stage ONLY recipe paths (intent-to-add for new files).
    # Each path is staged individually so a single bad path
    # (e.g., file no longer exists) doesn't abort the rest.
    staged_count = 0
    for p in recipe_paths:
        rc, _ = await _run("add", "--", p)
        if rc == 0:
            staged_count += 1
    if staged_count == 0:
        log_event(
            trace_id, "WARNING", "skill.code_assist",
            f"CHAIN-COMMIT no paths staged child="
            f"{child_task_id[:12]}",
        )
        return False

    # Check if anything to commit (the staged paths might have
    # been already-tracked + unmodified, so no actual diff).
    rc, stat = await _run("diff", "--cached", "--stat")
    if not stat.strip():
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"CHAIN-COMMIT nothing-to-commit child="
            f"{child_task_id[:12]} (recipe paths already in HEAD)",
        )
        return False

    rc, out = await _run(
        "-c", "user.email=chain-child@sentinel.local",
        "-c", "user.name=Sentinel-Chain",
        "commit", "-q", "-m", msg,
    )
    if rc != 0:
        log_event(
            trace_id, "WARNING", "skill.code_assist",
            f"CHAIN-COMMIT failed rc={rc} "
            f"child={child_task_id[:12]} stderr={out[:200]!r}",
        )
        return False

    rc, sha = await _run("rev-parse", "--short", "HEAD")
    sha = sha.strip()
    log_event(
        trace_id, "INFO", "skill.code_assist",
        f"CHAIN-COMMIT ok sha={sha} "
        f"child={child_task_id[:12]} "
        f"parent={parent_task_id[:12]} "
        f"paths={recipe_paths}",
    )
    return True


async def _verify_syntax_of_changed_files(
    diff_stat: str, trace_id: str,
) -> tuple[bool, str]:
    """Run `python -m py_compile` on every .py file mentioned in the
    diff stat. This is a server-side gate after Claude's review verdict:
    Claude says PASS based on Read+Grep inspection, but it has been
    known to miss syntax errors (e.g. unterminated f-strings) that the
    diff "looks right" but won't actually load.

    Returns (is_clean, error_summary). is_clean=True means all changed
    .py files compile. is_clean=False overrides Claude's verdict."""
    if not diff_stat:
        return True, ""
    py_files: list[str] = []
    for line in diff_stat.splitlines():
        # Lines look like:  core/foo.py | 5 +++++
        # Or:  .../__pycache__/x.cpython-312.pyc | Bin ...
        line = line.strip()
        if "|" not in line:
            continue
        path = line.split("|", 1)[0].strip()
        if not path.endswith(".py"):
            continue
        # Strip git's "..." prefix on truncated paths
        if path.startswith(".../"):
            continue
        py_files.append(path)
    if not py_files:
        return True, ""
    log_event(trace_id, "INFO", "skill.code_assist",
              f"syntax check: verifying {len(py_files)} .py file(s) "
              f"compile: {py_files[:5]}")
    errors: list[str] = []
    for path in py_files:
        full = config.PROJECT_ROOT / path
        if not full.exists():
            continue
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "py_compile", str(full),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", "replace").strip()
            errors.append(f"{path}: {err_text[:300]}")
    if errors:
        summary = " | ".join(errors)[:500]
        log_event(trace_id, "WARNING", "skill.code_assist",
                  f"syntax check FAILED: {summary}")
        return False, summary
    log_event(trace_id, "INFO", "skill.code_assist",
              f"syntax check passed for {len(py_files)} file(s)")
    return True, ""


async def _git_reset_hard(
    base_sha: str, trace_id: str,
    exclude_paths: list[str] | None = None,
) -> bool:
    """Restore tracked code dirs to base_sha and remove untracked files
    in those dirs. Scoped (NOT a global git reset --hard) so the live
    log file (logs/sentinel.jsonl), SQLite DBs, and workspace/ runtime
    artifacts are protected from accidental wipes during /code retries
    or graduation runs.

    Phase 14b bugfix: ``interfaces`` was missing from the scope --
    /code freely edits ``interfaces/telegram_bot.py`` (per
    SentinelTelegramBot._COMMIT_INCLUDE) but the reset never undid
    those edits. Caused a false-positive graduation pass for
    pattern #54 (the agentic graduation reviewed stale tree state
    from a prior /code attempt instead of THIS run's output).

    Phase 17j (2026-05-07): ``exclude_paths`` lets callers SKIP
    specific paths from the reset. Critical for graduation's finally
    block: after replay recreates the /code outcome in recipe paths,
    the final reset would wipe those edits AGAIN. Excluding recipe
    paths preserves the user's /code work. Live trigger 2026-05-07
    ~04:09Z: /code added /prompt handler to telegram_bot.py + created
    PROMPT_BRIEF.md; reviewer PASSED; graduation's finally
    `_git_reset_hard(pre_grad_sha)` wiped the handler edit (file
    PROMPT_BRIEF.md survived because it's outside the reset scope).
    """
    scoped = "core skills agents tests interfaces"
    # Phase 17j: read-aside-then-restore approach. Pathspec
    # `:(exclude)` syntax fights cmd.exe shell parsing on Windows
    # (live trigger: 17j initial impl had `pathspec 'core' did not
    # match any file(s) known to git` errors). Simpler: snapshot
    # the excluded files to memory BEFORE the broad reset, run the
    # normal reset, write the snapshotted content back. Works
    # cross-platform regardless of shell quoting quirks.
    saved: dict[str, bytes | None] = {}
    if exclude_paths:
        for p in exclude_paths:
            abs_p = config.PROJECT_ROOT / p
            try:
                saved[p] = abs_p.read_bytes() if abs_p.exists() else None
            except Exception:
                saved[p] = None
    cmds = [
        f"git checkout {base_sha} -- {scoped}",
        f"git clean -fd -- {scoped}",
    ]
    for cmd in cmds:
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=str(config.PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            log_event(trace_id, "ERROR", "skill.code_assist",
                      f"reset cmd failed: {cmd!r} rc={proc.returncode} "
                      f"err={err.decode('utf-8', 'replace')[:200]!r}")
            return False
    # Phase 17j: write back any excluded paths' content (saved before
    # the reset wiped them). For paths that didn't exist pre-reset
    # (saved value None), do nothing -- reset's git clean -fd would
    # have removed an untracked-new file, leaving "doesn't exist"
    # which matches the saved state.
    if exclude_paths:
        restored_count = 0
        for p, content in saved.items():
            abs_p = config.PROJECT_ROOT / p
            try:
                if content is not None:
                    abs_p.parent.mkdir(parents=True, exist_ok=True)
                    abs_p.write_bytes(content)
                    restored_count += 1
            except Exception as e:
                log_event(trace_id, "WARNING", "skill.code_assist",
                          f"reset exclude_paths restore failed for "
                          f"{p}: {type(e).__name__}: {e}")
        if restored_count:
            log_event(trace_id, "INFO", "skill.code_assist",
                      f"reset preserved {restored_count} excluded "
                      f"path(s): {list(saved.keys())}")
    log_event(trace_id, "INFO", "skill.code_assist",
              f"reset to base_sha={base_sha[:8]} (scoped to {scoped}"
              f"{f', excluded {len(exclude_paths)} paths' if exclude_paths else ''})")
    return True


# Max iterations of (pre-teach -> qwen -> review -> corrective-teach).
MAX_TEACH_ATTEMPTS = 5


# ─────────────────────────────────────────────────────────────────────
# Phase 17 Batch 1 -- decomposition mode + STEP-N format tightening.
#
# Live failure mode (trace SEN-b203948c, 2026-05-06 22:09Z): /qcode
# prompt produced 14692-char Claude response, truncated to 8000 chars
# at step boundary, parsed to 1 STEP. Reformat retry would have spent
# another Claude CLI call. Root cause: Claude tried to one-shot a
# multi-component change. Decomposition is the structural fix --
# Claude rates task size; if too big, returns a list of /code-shaped
# subtasks instead of a giant recipe. The user runs each subtask
# separately. Each lands cleanly in a small recipe.
#
# This module DOES NOT change Qwen's behavior. Qwen still sees one
# small recipe at a time. Claude is doing the batching, on Qwen's
# behalf.
# ─────────────────────────────────────────────────────────────────────

_DECOMP_SUBTASK_RE = re.compile(
    r'^\s*[-*]\s*(/code\s+\S.*?)\s*$', re.MULTILINE,
)


# Phase 17e v2 -- the strict regex was still too narrow. Now: accept
# ANY first line that contains the word "DECOMPOSE" (case-insensitive)
# AND the recipe contains >=1 '- /code ...' bullet. The bullet list
# is the real safety net -- false positives without /code bullets
# never reach the chain runner. This trades a tiny false-positive
# risk (Claude saying "I'll DECOMPOSE this" + happens to have a
# /code bullet) for robustness against Claude's formatting variation.
_DECOMP_KEYWORD_RE = re.compile(r'\bDECOMPOSE\b', re.IGNORECASE)


def _extract_decomposition(recipe: str) -> list[str] | None:
    """Detect Claude's DECOMPOSE response and return the subtask list.

    Returns ``None`` if recipe is not a decomposition block (caller
    falls through to the STEP-N path). Returns ``list[str]`` of
    ``/code <subtask>`` strings (always >=1) if recipe is a valid
    decomposition.

    Phase 17e: first-line match relaxed. Originally required exact
    ``DECOMPOSE`` or ``DECOMPOSE:`` -- too brittle for Claude's
    formatting variation. Now accepts any first non-blank line
    matching the regex ``[markdown_prefix]?[**]?DECOMPOSE[**]?:?``
    case-insensitive. Examples that match:
        DECOMPOSE
        DECOMPOSE:
        **DECOMPOSE**
        ## DECOMPOSE
        ### DECOMPOSE:
        - DECOMPOSE
        decompose:
    Examples that still don't match (defensive against narrative use):
        I'll decompose this task into:
        DECOMPOSE THIS:  (extra word)
        Some prose, then DECOMPOSE on next line  (not first line)

    The bullet list of subtasks must still appear -- even if the first
    line matches, no ``- /code ...`` bullets means ``None`` is
    returned (treated as STEP-N attempt with weird prose).
    """
    if not recipe:
        return None
    stripped = recipe.lstrip()
    if not stripped:
        return None
    first_line = stripped.split("\n", 1)[0]
    # Phase 17e v2: accept ANY first line containing the word
    # DECOMPOSE (case-insensitive). Claude's format varies wildly
    # (markdown headers, bold, prose like 'I'll DECOMPOSE this:').
    # The bullet list below is the real gating signal.
    if not _DECOMP_KEYWORD_RE.search(first_line):
        return None
    subtasks = [
        m.group(1).strip()
        for m in _DECOMP_SUBTASK_RE.finditer(recipe)
    ]
    if not subtasks:
        return None
    return subtasks


def _format_decomposition_response(subtasks: list[str]) -> str:
    """Render Claude's decomposition as user-facing markdown."""
    lines = [
        "\U0001F4CB *Task too big for one recipe.* "
        "Suggested split:",
        "",
    ]
    for i, sub in enumerate(subtasks, 1):
        lines.append(f"{i}. `{sub}`")
    lines.extend([
        "",
        "_Run them in order. /commit between each. "
        "Each lands cleanly with a small recipe._",
    ])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Phase 16 Batch C -- Claude-skip path (rebuild 2026-05-06 v2 with
# surgical cleanup; v1 self-wiped uncommitted source via _git_reset_hard
# scope blast).
# ─────────────────────────────────────────────────────────────────────


_RECIPE_PATH_RE = re.compile(r'\bpath\s*=\s*"([^"]+)"')


def _extract_recipe_paths(recipe: str) -> list[str]:
    """Pull every ``path="..."`` arg from a recipe. Skip-replay's
    failure cleanup uses this to revert ONLY files the recipe
    touched."""
    if not recipe:
        return []
    seen = []
    for m in _RECIPE_PATH_RE.finditer(recipe):
        p = m.group(1).strip()
        if p and p not in seen:
            seen.append(p)
    return seen


async def _maybe_skip_path(
    *,
    input_data,
    trace_id: str,
    kb,
    kb_patterns,
    backend_model: str,
    base_sha: str,
) -> dict:
    """Returns dict with status in {'success','failed','ineligible',
    'telemetry_only'} plus pattern_id, reason, and (on success)
    solution + diff_stat."""
    if not kb_patterns:
        return {
            "status": "ineligible",
            "pattern_id": None,
            "reason": "no_kb_match",
        }
    top = kb_patterns[0]

    eligible, elig_reason = kb.is_skip_eligible(pattern_id=top.id)
    log_event(
        trace_id, "INFO", "skill.code_assist",
        f"skip-eligibility: pattern_id={top.id} "
        f"eligible={eligible} reason={elig_reason}",
    )
    if not eligible:
        return {
            "status": "ineligible",
            "pattern_id": top.id,
            "reason": f"eligibility:{elig_reason}",
        }

    from core.recipe_linter import lint_recipe_for_skip
    lint = lint_recipe_for_skip(top.solution_pattern or "")
    log_event(
        trace_id, "INFO", "skill.code_assist",
        f"skip-lint: pattern_id={top.id} safe={lint.safe} "
        f"reason={lint.reason}",
    )
    if not lint.safe:
        return {
            "status": "ineligible",
            "pattern_id": top.id,
            "reason": f"lint:{lint.reason}",
        }

    if not config.SKIP_PATH_ENABLED:
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"SKIP-WOULD-FIRE pattern_id={top.id} -- all gates passed, "
            f"telemetry-only mode.",
        )
        return {
            "status": "telemetry_only",
            "pattern_id": top.id,
            "reason": "telemetry_only_flag_off",
        }

    log_event(
        trace_id, "INFO", "skill.code_assist",
        f"SKIP-FIRING pattern_id={top.id} -- replaying stored recipe, "
        f"NO Claude calls",
    )
    return await _execute_skip_replay(
        pattern=top, input_data=input_data, trace_id=trace_id,
        kb=kb, backend_model=backend_model, base_sha=base_sha,
    )


async def _execute_skip_replay(
    *, pattern, input_data, trace_id: str, kb,
    backend_model: str, base_sha: str,
) -> dict:
    """Run stepfed against stored recipe, verify with diff-match,
    update counter. SURGICAL cleanup on failure (NEVER scoped reset)
    -- see _RECIPE_PATH_RE / _extract_recipe_paths /
    core.tree_state.surgical_revert."""
    from core.diff_match import evaluate_diff_match
    from core.qwen_agent import run_agent_stepfed
    from core.tree_state import (
        restore_dirty_tree, snapshot_dirty_tree, surgical_revert,
    )

    recipe = pattern.solution_pattern or ""
    pattern_id = pattern.id
    recipe_paths = _extract_recipe_paths(recipe)

    # Phase 16 Batch C drift-detect (bail-no-penalty). Pre-flight
    # check: edit_file old= literal must exist in target file. If
    # not, skip-replay would silently no-op and tank diff-match,
    # unfairly penalizing the pattern's counter. Bail without
    # incrementing on environmental state drift.
    from core.skip_drift import detect_recipe_drift
    _drift, _drift_reason = detect_recipe_drift(recipe, config.PROJECT_ROOT)
    if _drift:
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"SKIP-DRIFT pattern_id={pattern_id} {_drift_reason} -- "
            f"bailing before replay; counter NOT incremented",
        )
        return {
            "status": "failed",
            "pattern_id": pattern_id,
            "reason": f"environmental:state_drift:{_drift_reason}",
        }

    try:
        handle = await snapshot_dirty_tree(config.PROJECT_ROOT)
    except Exception as e:
        log_event(
            trace_id, "WARNING", "skill.code_assist",
            f"SKIP-ERROR pattern_id={pattern_id}: snapshot failed "
            f"({type(e).__name__}: {e})",
        )
        return {
            "status": "failed",
            "pattern_id": pattern_id,
            "reason": "environmental:snapshot_failed",
        }

    replay_diff_stat = ""
    try:
        agent_result = await asyncio.to_thread(
            run_agent_stepfed,
            input_data.problem, recipe, trace_id, backend_model,
        )
        completed = bool(agent_result.get("completed_via_done", False))
        steps = int(agent_result.get("steps", 0))

        # Phase 16 Batch C diff-match fixes (2026-05-06):
        # 1. `git add -N` recipe paths so untracked write_file outputs
        #    appear in `git diff <sha>` (otherwise new files are
        #    invisible and replay diff comes back empty).
        # 2. Scope diff capture to recipe paths only so unrelated
        #    dirty work in the tree doesn't inflate file count and
        #    tank Jaccard via 1/N-where-N=total-dirty-files.
        from core.tree_state import _git as _git_helper
        for _rp in recipe_paths:
            _abs_p = config.PROJECT_ROOT / _rp
            if _abs_p.exists():
                try:
                    await _git_helper(
                        "add", "-N", "--", _rp,
                        cwd=config.PROJECT_ROOT,
                    )
                except Exception:
                    pass

        if recipe_paths:
            _rc1, replay_diff, _ = await _git_helper(
                "diff", base_sha, "--", *recipe_paths,
                cwd=config.PROJECT_ROOT,
            )
            replay_diff = replay_diff[:8000].strip() if _rc1 == 0 else ""
            _rc2, replay_diff_stat, _ = await _git_helper(
                "diff", "--stat", base_sha, "--", *recipe_paths,
                cwd=config.PROJECT_ROOT,
            )
            replay_diff_stat = replay_diff_stat.strip() if _rc2 == 0 else ""
        else:
            replay_diff = await _git_diff_full(base_sha)
            replay_diff_stat = await _git_diff_stat(base_sha)

        match_result = evaluate_diff_match(
            pattern.solution_code or "",
            replay_diff,
            threshold=config.SKIP_PATH_DIFF_MATCH_THRESHOLD,
        )

        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"SKIP-REPLAY pattern_id={pattern_id} steps={steps} "
            f"done={completed} diff_match={match_result.score:.3f} "
            f"accept={match_result.accept}",
        )

        if completed and steps > 0 and match_result.accept:
            new_attempts, new_passes, _flag = await asyncio.to_thread(
                kb.record_solo_attempt, pattern_id, True, trace_id,
            )
            log_event(
                trace_id, "INFO", "skill.code_assist",
                f"SKIP-SUCCESS pattern_id={pattern_id} now "
                f"{new_passes}/{new_attempts} solo",
            )
            r = await restore_dirty_tree(handle)
            if not r.restored:
                log_event(
                    trace_id, "WARNING", "skill.code_assist",
                    f"SKIP-RESTORE-REJECT pattern_id={pattern_id}: "
                    f"{r.reason}",
                )
            solution = (
                f"**Solved via skip-path** on attempt 1 -- replayed "
                f"pattern #{pattern_id} alone, NO Claude calls.\n\n"
                f"**Recipe:** {steps} steps, completed_via_done={completed}\n\n"
                f"**Diff-match:** {match_result.score:.3f} "
                f"(threshold {config.SKIP_PATH_DIFF_MATCH_THRESHOLD}) "
                f"{match_result.reason}\n\n"
                f"**Pattern:** now at {new_passes}/{new_attempts} solo passes.\n\n"
                f"**Files changed:**\n```\n{replay_diff_stat[:1500]}\n```"
            )
            return {
                "status": "success",
                "pattern_id": pattern_id,
                "reason": (
                    f"replay_passed_diff_match={match_result.score:.3f}"
                ),
                "solution": solution,
                "diff_stat": replay_diff_stat,
            }

        if not completed:
            failure_reason = "recipe_broken:no_done"
        elif steps == 0:
            failure_reason = "recipe_broken:no_steps"
        else:
            failure_reason = (
                f"runtime_check_failed:diff_match="
                f"{match_result.score:.3f}<"
                f"{config.SKIP_PATH_DIFF_MATCH_THRESHOLD}"
            )

        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"SKIP-FAILED pattern_id={pattern_id} {failure_reason}",
        )

        # SURGICAL cleanup -- only files the recipe mentioned.
        try:
            rc, reverted, removed = await surgical_revert(
                config.PROJECT_ROOT, recipe_paths,
            )
            log_event(
                trace_id, "INFO", "skill.code_assist",
                f"SKIP-CLEANUP surgical_revert rc={rc} "
                f"reverted={reverted} removed_new={removed}",
            )
        except Exception as e:
            log_event(
                trace_id, "WARNING", "skill.code_assist",
                f"SKIP-CLEANUP surgical_revert failed: "
                f"{type(e).__name__}: {e}",
            )

        try:
            await restore_dirty_tree(handle)
        except Exception as e:
            log_event(
                trace_id, "WARNING", "skill.code_assist",
                f"SKIP-CLEANUP restore failed: {type(e).__name__}: {e}",
            )

        try:
            await asyncio.to_thread(
                kb.record_solo_attempt, pattern_id, False, trace_id,
            )
        except Exception as e:
            log_event(
                trace_id, "WARNING", "skill.code_assist",
                f"SKIP-COUNTER record_solo_attempt failed: "
                f"{type(e).__name__}: {e}",
            )

        return {
            "status": "failed",
            "pattern_id": pattern_id,
            "reason": failure_reason,
        }

    except Exception as e:
        log_event(
            trace_id, "WARNING", "skill.code_assist",
            f"SKIP-ENV-ERROR pattern_id={pattern_id} "
            f"{type(e).__name__}: {e}",
        )
        try:
            await surgical_revert(config.PROJECT_ROOT, recipe_paths)
        except Exception:
            pass
        try:
            await restore_dirty_tree(handle)
        except Exception:
            pass
        return {
            "status": "failed",
            "pattern_id": pattern_id,
            "reason": f"environmental:{type(e).__name__}",
        }


async def _run_agentic_pipeline(
    input_data, trace_id, kb, kb_patterns, kb_limitations, kb_ids,
    context,
):
    """Full agentic /code with iterative teaching.

    Loop (up to MAX_TEACH_ATTEMPTS):
      attempt 1:  pre_teach (KB-augmented) -> qwen -> review
      attempt 2+: corrective_teach -> qwen -> review
      between attempts: git checkout <base> + git clean -fd

    Exit conditions:
      pass     -> store winning recipe as KB pattern
      exhausted -> reset tree, store limitation w/ last recipe tried
    """
    from core.qwen_agent import (
        run_agent_stepfed, _project_map, _parse_recipe_steps,
    )

    model_name_or_id: str = (
        (context or {}).get("model") or "qwen-coder"
    )
    try:
        from core.llm import INFERENCE_CLIENT
        cfg = INFERENCE_CLIENT.model_registry.get(model_name_or_id)
        backend_model = cfg.model_id if cfg else "qwen2.5-coder:3b"
    except Exception:
        backend_model = "qwen2.5-coder:3b"

    base_sha = await _git_snapshot(trace_id)
    log_event(trace_id, "INFO", "skill.code_assist",
              f"agentic /code: base_sha={base_sha[:8]} "
              f"max_attempts={MAX_TEACH_ATTEMPTS}")

    project_map = _project_map()
    kb_block = _format_kb_for_claude(kb_patterns, kb_limitations)

    # Phase 16 Batch C -- skip-path attempt BEFORE Claude pre-teach.
    # Telemetry-only by default (config.SKIP_PATH_ENABLED=False).
    skip_outcome = await _maybe_skip_path(
        input_data=input_data, trace_id=trace_id, kb=kb,
        kb_patterns=kb_patterns, backend_model=backend_model,
        base_sha=base_sha,
    )
    if skip_outcome["status"] == "success":
        return CodeAssistOutput(
            solution=skip_outcome["solution"],
            explanation="",
            solved_by="qwen_skip_path",
            teaching_note=skip_outcome["reason"],
            knowledge_entries_used=[skip_outcome["pattern_id"]],
            attempts=1,
            validated=True,
            execution_result=skip_outcome.get("diff_stat", "")[:500],
        )
    excluded_skip_id: int | None = None
    if (
        skip_outcome["status"] == "failed"
        and skip_outcome.get("pattern_id") is not None
    ):
        excluded_skip_id = skip_outcome["pattern_id"]
        kb_patterns = [
            p for p in kb_patterns if p.id != excluded_skip_id
        ]
        kb_block = _format_kb_for_claude(kb_patterns, kb_limitations)
        log_event(
            trace_id, "INFO", "skill.code_assist",
            f"skip-fallback: excluding pattern_id={excluded_skip_id} "
            f"from few-shot block",
        )

    started_at = time.monotonic()
    last_recipe = ""
    last_review = {"verdict": "unknown", "reasoning": ""}
    last_agent_result = {
        "summary": "", "session": [], "steps": 0,
        "completed_via_done": False,
    }
    last_diff_stat = ""
    last_diff_full = ""
    review = last_review
    attempt = 0
    # Phase 15c -- shadow planning state. Captured once on attempt 1
    # alongside Claude's pre_teach; survives across retries so the
    # final add_pattern call can stamp it on the row even if the
    # winning recipe came from a corrective attempt.
    shadow_recipe: str | None = None
    shadow_agreement: float | None = None
    # Phase 15d -- track review reasonings so we can detect
    # "we're attempting variants of the same root cause" and bail
    # without grinding through all 5 attempts. Also accumulate the
    # set of files Claude has read so corrective_teach can hint at
    # the prior exploration and skip re-reading.
    review_reasonings: list[str] = []
    bailed_on_repetition = False
    bail_phrase: str | None = None
    files_already_read: set[str] = set()

    # Phase 17a -- /kill polling. Look up our own task_id once via the
    # trace_id so we can poll `is_kill_requested` between attempts.
    # Best-effort: if the lookup fails (or the trace_id doesn't map to
    # a task row), kill polling is skipped and the pipeline runs as
    # before. This shouldn't happen in practice -- /code always comes
    # in via the worker's task queue -- but defensive nonetheless.
    # Phase 17b also reuses this row for chain-runner depth + parenting.
    kill_task_id: str | None = None
    self_chain_depth: int = 0
    try:
        from core import database as _db
        _row = _db.get_task_by_trace_id(trace_id)
        if _row is not None:
            kill_task_id = _row.task_id
            self_chain_depth = _row.chain_depth
    except Exception as _e:
        log_event(
            trace_id, "DEBUG", "skill.code_assist",
            f"kill-poll: trace lookup failed ({type(_e).__name__}: "
            f"{_e}); /kill will be a no-op for this run",
        )
    bailed_on_kill = False

    for attempt in range(1, MAX_TEACH_ATTEMPTS + 1):
        # Phase 17a -- poll kill flag at the top of each attempt.
        # Cannot interrupt mid-Claude-CLI (subprocess is a black box)
        # but we can refuse to start the next attempt. Worst-case
        # latency between /kill and bail = current attempt duration.
        if kill_task_id is not None:
            try:
                from core import database as _db_kill
                if _db_kill.is_kill_requested(kill_task_id):
                    log_event(
                        trace_id, "WARNING", "skill.code_assist",
                        f"KILL-REQUESTED task_id={kill_task_id} -- "
                        f"bailing before attempt {attempt}",
                    )
                    bailed_on_kill = True
                    break
            except Exception:
                pass
        elapsed = int(time.monotonic() - started_at)
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"=== attempt {attempt}/{MAX_TEACH_ATTEMPTS} "
                  f"(elapsed {elapsed}s) ===")

        if attempt == 1:
            recipe = await _claude_pre_teach(
                input_data.problem, input_data.code_context,
                kb_block, project_map, trace_id,
            )
            # Phase 17h -- if pre-teach got truncated (recipe was
            # over 8000-char cap), force a DECOMPOSE retry rather
            # than running a partial recipe with dropped trailing
            # STEPs. Trailing STEPs include the `done` marker, so
            # truncated runs always have done=False which propagates
            # to fail. Live trigger 2026-05-07 ~02:23Z. Best-effort:
            # if force-decompose returns empty, fall through to the
            # truncated recipe (existing behavior).
            try:
                if _LAST_PRE_TEACH_TRUNCATED:
                    log_event(
                        trace_id, "WARNING", "skill.code_assist",
                        "PRE_TEACH-OVERSIZE detected; triggering "
                        "force-DECOMPOSE retry",
                    )
                    forced = await _claude_force_decompose(
                        input_data.problem, recipe, trace_id,
                    )
                    if forced and _extract_decomposition(forced):
                        log_event(
                            trace_id, "INFO", "skill.code_assist",
                            f"PRE_TEACH-OVERSIZE recovered: forced "
                            f"DECOMPOSE returned {len(forced)} chars, "
                            f"replacing truncated STEP-N recipe",
                        )
                        recipe = forced
                    else:
                        log_event(
                            trace_id, "WARNING", "skill.code_assist",
                            "PRE_TEACH-OVERSIZE force-decompose "
                            "did not return a valid DECOMPOSE block; "
                            "falling back to truncated STEP-N",
                        )
            except Exception as e:
                log_event(
                    trace_id, "WARNING", "skill.code_assist",
                    f"PRE_TEACH-OVERSIZE retry exception "
                    f"({type(e).__name__}: {e}); falling back",
                )
            # Phase 17 Batch 1 -- decomposition short-circuit.
            # If Claude judged the task too big and emitted a
            # DECOMPOSE block (literal first line + bullet list of
            # /code-shaped subtasks), surface those to the user and
            # exit cleanly. NO stepfed, NO KB write, NO graduation.
            # User runs each subtask as its own /code; each lands
            # cleanly in a small recipe instead of one giant 14000-
            # char recipe getting truncated and degenerating.
            decomp_subtasks = _extract_decomposition(recipe)
            # Phase 17e v2 diagnostic: when matcher rejects but the
            # recipe is short (<=2000 chars) AND not a STEP-N response,
            # log the first line so future runs can see Claude's
            # actual format if our matcher fails again.
            if decomp_subtasks is None and recipe and len(recipe) <= 2000:
                _first = recipe.lstrip().split("\n", 1)[0][:200]
                if "STEP " not in _first.upper():
                    log_event(
                        trace_id, "INFO", "skill.code_assist",
                        f"DECOMPOSE-MISS: matcher rejected first_line="
                        f"{_first!r} recipe_chars={len(recipe)}",
                    )
            if decomp_subtasks:
                # Phase 17b -- chain runner. If enabled AND we're not
                # already at the depth cap, queue the subtasks as
                # child tasks and return a chain-started outcome.
                # Otherwise (flag off OR cap hit), fall through to the
                # Phase 17 Batch 1 behavior: surface the markdown list
                # to the user for manual execution.
                chain_can_fire = (
                    config.CODE_CHAIN_ENABLED
                    and kill_task_id is not None
                    and self_chain_depth < config.CODE_CHAIN_MAX_DEPTH
                )
                log_event(
                    trace_id, "INFO", "skill.code_assist",
                    f"pre-teach: decomposition suggested "
                    f"({len(decomp_subtasks)} subtasks); "
                    f"chain_enabled={config.CODE_CHAIN_ENABLED} "
                    f"depth={self_chain_depth}/"
                    f"{config.CODE_CHAIN_MAX_DEPTH} "
                    f"chain_will_fire={chain_can_fire}",
                )
                if chain_can_fire:
                    # Queue children sequentially (worker drains them
                    # via existing GPU lock + queue). Each child gets
                    # parent_task_id + chain_depth = self+1 so it
                    # cannot decompose further (caps depth at
                    # MAX_DEPTH). Best-effort enqueue: any failure
                    # logs + falls through to the manual surface.
                    try:
                        from core import database as _db_chain
                        from core.telemetry import generate_trace_id
                        queued_ids: list[str] = []
                        for sub in decomp_subtasks:
                            sub_text = sub
                            if sub_text.startswith("/code "):
                                sub_text = sub_text[len("/code "):].strip()
                            child_trace = generate_trace_id()
                            child_id = _db_chain.add_task(
                                trace_id=child_trace,
                                command="code",
                                args={"text": sub_text},
                                parent_task_id=kill_task_id,
                                chain_depth=self_chain_depth + 1,
                            )
                            queued_ids.append(child_id)
                            log_event(
                                trace_id, "INFO", "skill.code_assist",
                                f"CHAIN-QUEUED child task_id="
                                f"{child_id[:12]} "
                                f"trace_id={child_trace} "
                                f"sub={sub_text[:80]!r}",
                            )
                    except Exception as e:
                        log_event(
                            trace_id, "WARNING", "skill.code_assist",
                            f"CHAIN-ERROR enqueue failed "
                            f"({type(e).__name__}: {e}); "
                            f"falling through to manual surface",
                        )
                        queued_ids = []
                    if queued_ids:
                        # Build a chain-started body. Worker will
                        # process each child sequentially; each
                        # child reports its own /code completion
                        # in chat. User /commits at the end.
                        lines = [
                            "🔗 *Chain started.* "
                            f"Queued {len(queued_ids)} subtasks; "
                            "you'll see one /code completion message "
                            "per subtask as they finish.",
                            "",
                        ]
                        for i, sub in enumerate(decomp_subtasks, 1):
                            lines.append(f"{i}. `{sub}`")
                        lines.extend([
                            "",
                            "_Subtasks run sequentially. Working "
                            "tree carries each subtask's edits "
                            "into the next. /commit at the end "
                            "(after all subtasks finish) to lock "
                            "in the result. /kill to abort the "
                            "current subtask._",
                        ])
                        return CodeAssistOutput(
                            solution="\n".join(lines),
                            explanation="",
                            solved_by="chain_started",
                            teaching_note=(
                                f"Chain runner queued "
                                f"{len(queued_ids)} child /code "
                                f"tasks (depth "
                                f"{self_chain_depth + 1})."
                            ),
                            knowledge_entries_used=kb_ids,
                            attempts=1,
                            validated=False,
                            execution_result=(
                                "queued_task_ids=" + ",".join(
                                    t[:12] for t in queued_ids
                                )
                            ),
                        )
                # Either chain disabled, depth-capped, or enqueue
                # failed -- fall through to manual surface.
                return CodeAssistOutput(
                    solution=_format_decomposition_response(
                        decomp_subtasks,
                    ),
                    explanation="",
                    solved_by="decompose_suggested",
                    teaching_note=(
                        "Claude judged the task too big for one "
                        "recipe; suggested decomposition into "
                        f"{len(decomp_subtasks)} subtasks."
                    ),
                    knowledge_entries_used=kb_ids,
                    attempts=1,
                    validated=False,
                    execution_result="",
                )
            # Phase 15c -- shadow plan call. After Claude has
            # written the canonical recipe, ask Qwen for the same
            # thing using identical KB context. Score the structural
            # agreement and stash for the eventual add_pattern call.
            # Wrapped in a broad try/except: ANY failure here must
            # not affect the production path.
            try:
                shadow_recipe = await _qwen_shadow_plan(
                    input_data.problem, input_data.code_context,
                    kb_block, project_map, backend_model, trace_id,
                )
                if shadow_recipe and recipe:
                    from core.plan_agreement import score_plan_agreement
                    shadow_agreement = score_plan_agreement(
                        recipe, shadow_recipe,
                    )
                    log_event(
                        trace_id, "INFO", "skill.code_assist",
                        f"shadow plan: qwen recipe "
                        f"{len(shadow_recipe)} chars, agreement="
                        f"{shadow_agreement:.3f}",
                    )
            except Exception as e:
                log_event(
                    trace_id, "INFO", "skill.code_assist",
                    f"shadow plan: skipped "
                    f"({type(e).__name__}: {e})",
                )
                shadow_recipe = None
                shadow_agreement = None
        else:
            # Phase 15d: extract project paths from prior recipe +
            # last review reasoning so corrective_teach can hint
            # Claude not to re-read them. Tree was reset to base_sha
            # so the files are byte-for-byte what Claude saw last.
            files_already_read |= _extract_project_paths(
                last_recipe, last_review.get("reasoning") or "",
            )
            recipe = await _claude_corrective_teach(
                input_data.problem, last_recipe,
                last_agent_result["session"], last_diff_full,
                last_review["reasoning"], kb_block, project_map,
                trace_id,
                files_already_read=files_already_read,
            )
        if not recipe:
            log_event(trace_id, "WARNING", "skill.code_assist",
                      f"attempt {attempt}: empty recipe; bailing")
            break
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"attempt {attempt}: recipe {len(recipe)} chars")

        # Phase 16 Option 1 -- parse-check before stepfed. If Claude's
        # recipe doesn't have >=2 parseable STEP blocks, stepfed would
        # fall back to the legacy free-form run_agent path (which
        # doesn't pin Qwen to literal recipe commands -- the bug
        # surface seen in SEN-ca56136c attempt 1: 66s of legacy run
        # producing a wrong-direction edit). Re-ask Claude once with
        # strict format-only directives; if reformat ALSO fails,
        # fall through and let stepfed's legacy fallback fire as the
        # last-resort safety net.
        #
        # 2026-05-06 hot-fix (degeneracy guard): also reject reformat
        # results that have no edit_file/write_file step. Trace
        # SEN-b46a27cf showed Option 1's reformat coercing a Claude
        # refusal ("provide the emoji pair you want") into a
        # read_file+done recipe that subsequently passed review on
        # the basis of "file is already in target state." Detecting
        # the edit-less shape catches that.
        parsed_steps = _parse_recipe_steps(recipe)
        if len(parsed_steps) < 2:
            log_event(trace_id, "WARNING", "skill.code_assist",
                      f"attempt {attempt}: recipe parsed "
                      f"{len(parsed_steps)} STEP(s); "
                      f"requesting reformat to avoid legacy fallback")
            reformatted = await _claude_reformat_recipe(
                recipe, input_data.problem, trace_id,
            )
            parsed_retry = (
                _parse_recipe_steps(reformatted) if reformatted else []
            )
            reformat_has_edit = (
                _recipe_has_edit_step(reformatted) if reformatted else False
            )
            if len(parsed_retry) >= 2 and reformat_has_edit:
                log_event(trace_id, "INFO", "skill.code_assist",
                          f"attempt {attempt}: reformat succeeded "
                          f"({len(parsed_retry)} steps, "
                          f"{len(reformatted)} chars)")
                recipe = reformatted
            elif len(parsed_retry) >= 2 and not reformat_has_edit:
                log_event(
                    trace_id, "WARNING", "skill.code_assist",
                    f"attempt {attempt}: reformat produced edit-less "
                    f"recipe ({len(parsed_retry)} steps but no "
                    f"edit_file/write_file); treating as legitimate "
                    f"refusal -- bailing attempt without storing pattern"
                )
                # Capture the rejected recipe + a clear verdict so
                # the post-loop limitation-store path has informative
                # context (rather than empty defaults).
                last_recipe = reformatted
                last_review = {
                    "verdict": "fail",
                    "reasoning": (
                        "Claude's recipe had no edit_file or "
                        "write_file step (recipe was likely a "
                        "clarification request or refusal coerced "
                        "into recipe shape). Bailing -- no pattern "
                        "stored. Try a more specific prompt or "
                        "verify the file isn't already in the "
                        "desired state."
                    ),
                }
                review = last_review
                break
            else:
                log_event(trace_id, "WARNING", "skill.code_assist",
                          f"attempt {attempt}: reformat retry "
                          f"exhausted ({len(parsed_retry)} steps "
                          f"after retry); falling through to legacy "
                          f"run_agent fallback")
        last_recipe = recipe

        # 2026-05-06 hot-fix (post-pre-teach degeneracy guard): even
        # without going through reformat, Claude can produce a recipe
        # that parses cleanly to 2+ steps but has no edit. A
        # read_file + run_bash + done shape, for example. Reject
        # before stepfed runs to save the wasted Qwen time + the
        # potential false-pass from reviewer-Claude.
        if not _recipe_has_edit_step(recipe):
            log_event(
                trace_id, "WARNING", "skill.code_assist",
                f"attempt {attempt}: recipe has no edit_file/write_file "
                f"step -- not a real solution. Bailing attempt."
            )
            last_recipe = recipe
            last_review = {
                "verdict": "fail",
                "reasoning": (
                    "Recipe had no edit_file or write_file step "
                    "(likely a clarification request or refusal). "
                    "Bailing -- no pattern stored. Try a more "
                    "specific prompt or verify the file isn't "
                    "already in the desired state."
                ),
            }
            review = last_review
            break

        agent_result = await asyncio.to_thread(
            run_agent_stepfed, input_data.problem, recipe, trace_id,
            backend_model,
        )
        last_agent_result = agent_result
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"attempt {attempt}: qwen finished steps="
                  f"{agent_result['steps']} done="
                  f"{agent_result['completed_via_done']}")

        await _git_commit_changes(
            trace_id,
            f"agentic /code attempt {attempt}: "
            f"{input_data.problem[:60]}",
        )
        # Phase 16 Batch C: scope BOTH diff_stat AND diff_full to
        # recipe-touched paths. diff_stat scoping = clean user-visible
        # "Files changed". diff_full scoping = clean solution_code at
        # storage time, so future skip-path diff-match against this
        # pattern compares apples-to-apples (without scoping, stored
        # = wide diff with leftover workspace/ pollution, and replay
        # = narrow diff scoped to recipe paths -- they'd never match,
        # which is exactly the 0.167 Jaccard bug seen on pattern #125).
        attempt_paths = _extract_recipe_paths(recipe)
        diff_stat = await _git_diff_stat(
            base_sha,
            paths=attempt_paths if attempt_paths else None,
        )
        diff_full = await _git_diff_full(
            base_sha,
            paths=attempt_paths if attempt_paths else None,
        )
        last_diff_stat = diff_stat
        last_diff_full = diff_full

        review = await _claude_review(
            input_data.problem, recipe,
            agent_result["summary"], agent_result["session"],
            diff_full, trace_id,
        )
        # Server-side syntax-check gate: Claude can read+grep but
        # has been observed to miss things like unterminated f-strings.
        # If Claude says PASS but a touched .py file doesn't compile,
        # override to FAIL with the syntax error.
        if review["verdict"] == "pass":
            syntax_ok, syntax_err = await _verify_syntax_of_changed_files(
                diff_stat, trace_id,
            )
            if not syntax_ok:
                review = {
                    "verdict": "fail",
                    "reasoning": (
                        f"Reviewer said pass but syntax check failed: "
                        f"{syntax_err}"
                    ),
                }
                log_event(trace_id, "WARNING", "skill.code_assist",
                          f"attempt {attempt}: PASS overridden to FAIL "
                          f"by syntax check")

        last_review = review
        log_event(trace_id, "INFO", "skill.code_assist",
                  f"attempt {attempt}: verdict={review['verdict']} "
                  f"reasoning={review['reasoning'][:100]!r}")

        if review["verdict"] == "pass":
            log_event(trace_id, "INFO", "skill.code_assist",
                      f"attempt {attempt}: PASS -- storing pattern")
            break

        # Phase 15d -- bail-on-shape-repetition. After attempt 2+,
        # if the latest review reasoning shares a substantive 5-gram
        # with any prior review's reasoning, we're chasing variants
        # of the same Claude-recipe quality issue. Skip remaining
        # retries and store a limitation. The 9 minutes saved on a
        # hopeless 5-attempt run is worth the rare false-positive
        # bail (which the user can /code again to retry from scratch).
        review_reasonings.append(review.get("reasoning") or "")
        if attempt >= 2:
            for prior_idx, prior_reasoning in enumerate(
                review_reasonings[:-1],
            ):
                phrase = _shape_repetition_phrase(
                    prior_reasoning, review_reasonings[-1],
                )
                if phrase:
                    bailed_on_repetition = True
                    bail_phrase = phrase
                    log_event(
                        trace_id, "WARNING", "skill.code_assist",
                        f"attempt {attempt}: BAIL -- shape repetition "
                        f"with attempt {prior_idx + 1} review on "
                        f"phrase {phrase!r}; skipping remaining "
                        f"{MAX_TEACH_ATTEMPTS - attempt} attempts",
                    )
                    break
            if bailed_on_repetition:
                break

        if attempt < MAX_TEACH_ATTEMPTS:
            log_event(trace_id, "INFO", "skill.code_assist",
                      f"attempt {attempt} FAILED -- resetting tree to "
                      f"{base_sha[:8]} for next attempt")
            ok = await _git_reset_hard(base_sha, trace_id)
            if not ok:
                log_event(trace_id, "ERROR", "skill.code_assist",
                          "reset failed; aborting iteration")
                break

    # ---- Final storage ----
    final_diff_stat = last_diff_stat
    grad_result: dict | None = None
    if review["verdict"] == "pass":
        # Phase 10 no-op call (kept for the diff-helper side effect of
        # `git add -N`).
        await _git_commit_changes(
            trace_id,
            f"/code: {input_data.problem[:80]}",
        )
        # Phase 15e: REAL commit BEFORE graduation. Pre-15e the
        # working-tree changes from the winning attempt survived
        # into graduation uncommitted, and graduation's stash dance
        # was silently failing to capture them -- pattern #66 was
        # the smoking gun. Snapshotting now means pre_grad_sha
        # includes the change, so graduation's final reset RESTORES
        # the change instead of reverting it. Best-effort -- if the
        # commit fails (no changes, hook crash, signing required),
        # graduation's old behaviour kicks in and we just lose the
        # tree-restore-on-success guarantee for this one /code.
        await _git_commit_for_graduation(
            trace_id,
            f"/code: {input_data.problem[:80]}",
        )
        # Strip absolute paths so the stored pattern is portable
        # across machines and different file targets.
        portable_recipe = _normalize_recipe_paths(last_recipe)
        # Phase 14b: store the actual diff text (capped) instead of
        # the diff stat. The stat is human-readable but useless as a
        # few-shot KB example for Qwen -- Qwen needs to SEE the
        # before/after, not a list of byte counts. Cap at 2000 chars
        # so KB context budget (4000 chars) can still fit two patterns.
        portable_diff = (last_diff_full or last_agent_result["summary"])[:2000]
        new_pattern_id = kb.add_pattern(
            tags=_extract_tags(
                input_data.problem, last_agent_result["summary"],
            ),
            problem_summary=_summarize_problem(input_data.problem),
            solution_code=portable_diff,
            solution_pattern=portable_recipe,
            explanation=(
                f"Solved on attempt {attempt}/{MAX_TEACH_ATTEMPTS}. "
                f"Qwen executed {last_agent_result['steps']} steps. "
                f"Verdict: {review['reasoning'][:160]}"
            ),
            trace_id=trace_id,
            # Phase 14b: capture the pre-/code SHA so graduation can
            # reset the tree and replay the recipe through the full
            # pipeline (the actual production skill).
            base_sha=base_sha,
            # Phase 15c: shadow planning -- stamp Qwen's parallel
            # recipe + agreement score so we can track over time
            # whether Qwen could plan this on its own. Both None on
            # shadow timeout / failure (best-effort).
            qwen_plan_recipe=shadow_recipe,
            qwen_plan_agreement=shadow_agreement,
        )
        solved_by = "qwen_agent"
        # Phase 14a: graduation test. Re-run the same problem with
        # Qwen-only (with KB context, no Claude in loop) and update
        # solo_attempts/passes on the new row. Adds ~15-30s but is
        # the only ground truth we have for "did Qwen actually learn
        # this". Best-effort -- failure here doesn't fail the /code.
        try:
            from core.kb_graduation import graduate_pattern
            # Resolve worker model id (same logic as the legacy path).
            from core.llm import INFERENCE_CLIENT
            grad_model_name = (
                (context or {}).get("model") or config.WORKER_MODEL
            )
            grad_cfg = INFERENCE_CLIENT.model_registry.get(grad_model_name)
            grad_model = grad_cfg.model_id if grad_cfg else grad_model_name
            grad_result = await graduate_pattern(
                pattern_id=new_pattern_id,
                problem=input_data.problem,
                code_context=input_data.code_context,
                kb=kb, model_id=grad_model, trace_id=trace_id,
            )
        except Exception as e:
            log_event(
                trace_id, "WARNING", "skill.code_assist",
                f"graduation skipped: {type(e).__name__}: {e}",
            )
        # Phase 17d -- chain-child auto-commit. SCOPED EXCEPTION to
        # the NO-AUTO-COMMITS owner directive: chain-runner subtasks
        # MUST commit between siblings to preserve state. Without
        # this, sibling /code's _git_reset_hard wipes uncommitted
        # work between chain hops (Phase 17b's known failure mode
        # observed live 2026-05-06 ~00:54 -- subtask 1's qcode_assist
        # files survived but core/config.py + telegram edits got
        # blasted by subtask 2's reset). Standalone /code (no parent)
        # still respects NO-AUTO-COMMITS; only chain children commit.
        # /revert in chat (or /revert chain N) undoes any chain
        # commit. Scoped to recipe paths only -- never sweeps up
        # unrelated dirty work.
        try:
            from core import database as _db_chain
            if kill_task_id is not None:
                _self_row = _db_chain.get_task(kill_task_id)
                if (
                    _self_row is not None
                    and _self_row.parent_task_id is not None
                ):
                    await _chain_child_auto_commit(
                        recipe_paths=_extract_recipe_paths(recipe),
                        child_task_id=kill_task_id,
                        parent_task_id=_self_row.parent_task_id,
                        problem=input_data.problem,
                        trace_id=trace_id,
                    )
        except Exception as e:
            log_event(
                trace_id, "WARNING", "skill.code_assist",
                f"chain auto-commit skipped: "
                f"{type(e).__name__}: {e}",
            )
    else:
        await _git_reset_hard(base_sha, trace_id)
        # Phase 15d: explanation is now attempt-count-aware (bail-on-
        # repetition exits early so "after 5 attempts" would lie),
        # and shadow planning data is preserved on the limitation row
        # alongside it on the pattern row. The shadow signal is
        # arguably MORE interesting on a limitation -- it tells us
        # whether the failure was Claude-quality (recipe issues) or
        # Qwen-capacity (couldn't have planned it either).
        if bailed_on_kill:
            # Phase 17a -- /kill abort path. Don't store a
            # limitation since "user manually aborted" is not a
            # signal that the task is impossible. Just reset
            # tree + return a clean qwen_killed outcome.
            outcome_line = (
                "Aborted by /kill before "
                f"attempt {attempt}. No limitation stored "
                "(user-initiated abort, not a failure)."
            )
        elif bailed_on_repetition:
            outcome_line = (
                f"Bailed after {attempt} attempts on shape repetition "
                f"(phrase {bail_phrase!r}). Last verdict: "
                f"{review['reasoning'][:240]}."
            )
        else:
            outcome_line = (
                f"Failed after {attempt} attempts with iterative "
                f"teaching. Last verdict: {review['reasoning'][:240]}."
            )
        if bailed_on_kill:
            # Phase 17a: skip kb.add_limitation -- user-initiated
            # aborts are not capability signals. Set solved_by to
            # "qwen_killed" so the bot's render branch can surface
            # a friendly "aborted" message.
            solved_by = "qwen_killed"
        else:
            kb.add_limitation(
                tags=_extract_tags(
                    input_data.problem, last_agent_result["summary"],
                ),
                problem_summary=(
                    "Qwen could not complete: "
                    + _summarize_problem(input_data.problem)
                ),
                explanation=(
                    f"{outcome_line} "
                    f"Last recipe Claude tried:\n{last_recipe[:600]}"
                ),
                trace_id=trace_id,
                qwen_plan_recipe=shadow_recipe,
                qwen_plan_agreement=shadow_agreement,
            )
            solved_by = "qwen_failed"
        final_diff_stat = ""

    if solved_by == "qwen_agent":
        body_parts = [
            f"**Solved** on attempt {attempt}/{MAX_TEACH_ATTEMPTS}.",
            f"**Qwen:** {last_agent_result['summary'][:300]}",
            f"**Verdict:** PASS -- {review['reasoning'][:200]}",
        ]
        if final_diff_stat:
            body_parts.append(
                f"**Files changed:**\n```\n{final_diff_stat[:1200]}\n```"
            )
        # Phase 14a: surface the graduation result so the user sees
        # whether Qwen could solve solo right after Claude taught it.
        if grad_result is not None:
            mark = "✓" if grad_result["passed"] else "✗"
            body_parts.append(
                f"**Graduation:** {mark} {grad_result['solo_passes']}/"
                f"{grad_result['solo_attempts']} solo "
                f"(KB pattern #{grad_result['pattern_id']}, "
                f"{grad_result['duration_s']}s)"
            )
            if grad_result["needs_reteach"]:
                body_parts.append(
                    "⚠️ This pattern dropped below the graduation pass "
                    "rate threshold. Future matches will route to Claude "
                    "instead of being used as a few-shot example. "
                    "Use `/kb verify <id>` to retry, or `/kb retake <id>` "
                    "to clear the flag after re-teaching."
                )
        body_parts.append(
            "Restart the bot to apply (Ctrl+C, re-run main.py)."
        )
    elif solved_by == "qwen_killed":
        # Phase 17a -- /kill abort path. Friendly message, no
        # "limitation stored" framing (it's not a failure).
        attempts_run = max(attempt - 1, 0)
        body_parts = [
            "🛑 **Aborted by /kill.**",
            (
                f"Stopped before attempt {attempt}/{MAX_TEACH_ATTEMPTS}. "
                f"{attempts_run} attempt"
                f"{'s' if attempts_run != 1 else ''} ran before abort."
            ),
            "Working tree was reset to clean state. No files changed.",
            "_No limitation stored -- /kill is user-initiated, not a "
            "capability signal._",
        ]
    else:
        # 2026-05-06 hot-fix: report ACTUAL attempt count, not the
        # MAX. Early bail (degeneracy guard, shape repetition, etc.)
        # may exit on attempt 1 -- saying "after 5 attempts" then
        # is misleading. `attempt` is set by the loop variable to
        # whichever iteration we exited on.
        attempts_run = max(attempt, 1)
        body_parts = [
            f"**Could not complete** after {attempts_run} "
            f"teaching attempt{'s' if attempts_run != 1 else ''}. "
            f"Limitation stored in KB.",
            f"**Last verdict:** {review['reasoning'][:300]}",
            "Tree was reset to clean state. No files changed.",
        ]
    final_text = "\n\n".join(body_parts)

    return CodeAssistOutput(
        solution=final_text,
        explanation="",
        solved_by=solved_by,
        teaching_note=review["reasoning"],
        knowledge_entries_used=kb_ids,
        attempts=attempt,
        validated=(review["verdict"] == "pass"),
        execution_result=final_diff_stat[:500],
    )
