# QWENCODER.md — Teaching memo for the Sentinel worker model

> This file is the **worker model's coding playbook**. It's loaded into the system prompt
> for every shadow-plan call (Phase 15c) and is the seed for human + Claude-curated
> learnings about what trips up the 3B coder on this codebase.
>
> **Edit it directly** to add a rule. The file_guard service watches for tampering;
> sanctioned writes go through `core.file_guard.authorize_update`.
> When `/curate qwencoder` lands, it will read recent low-agreement traces from
> `logs/sentinel.jsonl` and propose deltas.

---

## 1. Recipe contract — the only output shape that works

A recipe is a numbered list of `STEP N:` lines. Each line is **exactly one tool call**
in `tool_name key="value" key="value"` format. **No prose between steps. No markdown
bullets. No backticks around args. No code fences around the recipe.**

The Sentinel parser is strict, deterministic, and Python-side — it does NOT speak
natural-language workflow. If you can't write a step in the contract format, the
executor will refuse the step.

### Canonical example — short, one file, has a verifier

```
STEP 1: write_file path="math_utils.py" content="def add(a, b):\n    return a + b\n"
STEP 2: run_bash command="python -c 'from math_utils import add; assert add(2, 3) == 5'"
STEP 3: done summary="created math_utils.add"
```

That's the shape. Always.

---

## 2. Tool reference (the executor has no others)

| Tool | Required args | Purpose | Notes |
|------|---------------|---------|-------|
| `read_file` | `path` | Read a file before editing | Cheap; do this BEFORE `edit_file` |
| `list_dir` | `path` | List a directory | Use to confirm a path exists |
| `write_file` | `path`, `content` | Whole-file rewrite | **Prefer this** — most reliable |
| `edit_file` | `path`, `old`, `new` | Substring replace | `old` MUST be unique + verbatim |
| `run_bash` | `command` | Run a shell command | Use to verify behavior |
| `done` | `summary` | Final step, always last | Required to mark completion |

**Every step has all required args. Every time.** A missing `new=` on `edit_file` is
the single most common production failure (Phase 15c traces show it on 3 of 5 attempts
in the May 2026 multi-handler edit). Don't ship a step that's missing args even when
the rest of the recipe makes the intent obvious.

---

## 3. Argument quoting rules

- Values use **double quotes**: `path="core/util.py"` — never backticks, never
  single quotes.
- Inside a value, use JSON-style escapes: `\n` for newline, `\\` for backslash,
  `\"` for an internal double quote.
- **No triple quotes** anywhere in `content=`. Use literal `\n` instead.
- Paths are **project-relative POSIX**: `core/foo.py`, not `C:\Users\...\core\foo.py`
  and not `/abs/path/...`. The executor sandbox rejects paths that escape the
  project root.

### Bad shapes the parser will refuse

```
STEP 1: edit_file
  - path: `core/foo.py`
  - old: ...
  - new: ...
```
↑ markdown bullets. Won't parse.

```
STEP 1: edit_file path=`core/foo.py` old=`x` new=`y`
```
↑ backticks. Won't parse.

```
STEP 1: edit_file path="core/foo.py" old="abc"
```
↑ missing `new=` arg. Step will run and crash.

---

## 4. `edit_file` rules (where most failures live)

**`old=` must be a verbatim copy-paste from a `read_file` result you ran in the same
recipe.** Don't paraphrase. Don't summarize. Don't fix typos. Don't drop trailing
whitespace. The executor does an exact string match — close-enough fails with
`old string not found in file`.

**`old=` must be unique in the file** — include 3–5 lines of surrounding context if
the substring would otherwise repeat. The executor refuses ambiguous matches with
`old string appears N times; include more context to make it unique`.

**Prefer `write_file` over `edit_file`** when:
- the file is new (no existing content to match against)
- more than ~30% of the file changes
- the prior attempt failed with "old not found" or "old appears N times"

`write_file` takes the COMPLETE new content and skips the brittle string-matching
step entirely. It's the reliable path.

---

## 5. Verification step

Every recipe SHOULD include a `run_bash` step that exercises the new behavior, before
the final `done`. Examples:

```
STEP N: run_bash command="python -c 'from math_utils import gcd; assert gcd(12, 8) == 4'"
STEP N: run_bash command="python -m pytest tests/test_phase15d_resilience.py -q"
STEP N: run_bash command="python -c 'import importlib, core.foo; importlib.reload(core.foo)'"
```

The verifier proves the change is wired correctly. Skipping it is how recipes
"appear successful" but leave the codebase in a state where imports break or
behavior is unchanged.

---

## 6. Cross-file wiring (the integration footgun)

When a recipe creates a NEW file, the recipe MUST also include a step that wires the
new file into the call site(s) that use it. Creating `core/util.py` and never
importing it anywhere is a "successful" recipe with zero effect on production.

Pattern:
```
STEP 1: write_file path="core/new_module.py" content="..."
STEP 2: edit_file path="core/existing.py" old="from core.foo import bar" new="from core.foo import bar\nfrom core.new_module import baz"
STEP 3: run_bash command="python -c 'from core.existing import baz; print(\"wired\")'"
STEP 4: done summary="added core.new_module and wired into core.existing"
```

If you can't name a call site, the new file shouldn't exist — fold the
functionality into an existing module instead.

---

## 7. Failure-mode cheat sheet

| Symptom in a review | Most likely cause | What to do next time |
|---|---|---|
| `old string not found in file` | Paraphrased `old=` arg | Run `read_file` first; copy-paste verbatim |
| `old string appears N times` | Ambiguous match | Include 3–5 lines of unique context in `old=` |
| `path escapes sandbox` | Absolute / `..` path | Use project-relative POSIX paths |
| `tool_edit_file() missing 'new'` | Truncated or malformed step | Always include `new=` arg, every time |
| Reviewer: "wired but never called" | New file, no integration step | Add an `edit_file` step on the call site |
| Reviewer: "syntax error" / "unterminated" | Missed escapes in `content=` | Double-check `\n` / `\\` / `\"` in writes |

---

## 8. Recipe length

The Sentinel cap for stepfed mode is **8000 characters** total recipe length, after
extracting STEP blocks. The truncator drops trailing partial steps at the boundary,
so an over-budget recipe loses the LAST steps cleanly — but still loses them.

Rule: **be concise**. Write the minimum number of steps that solves the problem.
Each step should accomplish one concrete change. If a recipe is heading past 6000
chars, split the change into two `/code` calls or fold edits together with
`write_file`.

---

## 9. KB context — when prior patterns are injected

The Sentinel KB injects up to ~3000 chars of relevant prior patterns into the prompt
before this memo is read. When KB context shows a working pattern for a similar
problem, **adapt that pattern's recipe shape** rather than rederiving from scratch.
The KB's job is to make the worker model faster on familiar work — use it.

---

## 10. Curator-added learnings

> *Section reserved for `/curate qwencoder` to append concrete deltas extracted from
> recent low-agreement traces. Each entry should be: a 1-line failure description,
> the recipe shape that caused it, and the fix.*

<!-- CURATOR-ENTRIES-BEGIN -->
<!-- CURATOR-ENTRIES-END -->
