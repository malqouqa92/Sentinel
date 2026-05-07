"""Phase 15d -- /code resilience hardening + QWENCODER.md.

ECC. No GPU, no real LLM, no actual /code execution. The retry-loop
hardening is verified at the unit level (helpers + their wiring) and
at the source level (the call-site wires the helper correctly). This
mirrors the test_phase14a / test_phase15c approach -- avoid spinning
up Ollama or the Telegram bot.

Coverage:
  Step-boundary truncation (Fix 1):
    R01 -- short recipe passes through unchanged
    R02 -- truncates at step boundary, drops trailing partial step
    R03 -- preserves at least one whole step when first step alone exceeds cap
    R04 -- recipe with no STEP markers falls back to char-cut

  Recipe budget (Fix 2):
    R11 -- RECIPE_MAX_CHARS_STEPFED is 8000 (was 4000 in 15c)

  Shape-repetition bail (Fix 3):
    R21 -- identical reasonings -> bail phrase non-None
    R22 -- different concrete failures -> no bail
    R23 -- shared boilerplate (mostly stopwords) does NOT trigger bail
    R24 -- empty / None inputs -> no bail
    R25 -- 5-gram threshold respected (4-gram match -> no bail)

  Already-read paths (Fix 4):
    R31 -- extracts project-relative paths from a recipe
    R32 -- extracts paths from review reasoning
    R33 -- ignores absolute paths (won't seed the hint with junk)
    R34 -- ignores bare module names without extension
    R35 -- corrective_teach signature accepts files_already_read kwarg

  Shadow data on limitations (Fix 5):
    R41 -- add_limitation accepts new kwargs and stores them
    R42 -- old call sites (no kwargs) still work, store NULL
    R43 -- failure-path source-level check pipes shadow_recipe + agreement

  QWENCODER.md (Fix 6):
    R51 -- file exists at workspace/persona/QWENCODER.md
    R52 -- file is in PROTECTED_FILES set
    R53 -- PERSONA_INJECT_MAX_CHARS has an entry for QWENCODER.md
    R54 -- _load_qwencoder_memo reads the file
    R55 -- _load_qwencoder_memo returns "" when file missing
    R56 -- _qwen_shadow_system_prompt composes BASE + memo
    R57 -- _qwen_shadow_plan source uses the dynamic composer
    R58 -- memo content has the canonical recipe contract

  Phase 15d-bugfix -- shadow plan exits Ollama JSON mode:
    R61 -- _qwen_generate accepts format_json kwarg, default True
    R62 -- _qwen_shadow_plan source passes format_json=False
    R63 -- production stepfed call sites still get format_json=True
           (back-compat: existing transcription calls keep JSON mode)

  Phase 15d-bugfix-2 -- KB no longer destroys real diff patterns:
    R71 -- _is_real_solution accepts `git diff --git` body (Phase 14b shape)
    R72 -- _is_real_solution accepts hunk + +/- lines (diff body w/o header)
    R73 -- _is_real_solution still accepts diff --stat (back-compat)
    R74 -- _is_real_solution still accepts Python source
    R75 -- _is_real_solution rejects bare literals
    R76 -- cleanup_low_quality_patterns now ARCHIVES (not DELETES)
    R77 -- cleanup_low_quality_patterns skips already-archived rows
    R78 -- a real /code-style diff body survives cleanup
"""
from __future__ import annotations

import struct
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core import embeddings as emb
from core.knowledge_base import KnowledgeBase
import skills.code_assist as ca


def _stub_embedder(monkeypatch):
    def fake_embed(text, trace_id="SEN-system"):
        seed = sum(ord(c) for c in (text or "")) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(config.EMBEDDING_DIM).tolist()
        return struct.pack(f"<{len(vec)}f", *vec)
    monkeypatch.setattr(emb, "embed_text", fake_embed)


@pytest.fixture
def fresh_kb(tmp_path, monkeypatch):
    db_path = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    _stub_embedder(monkeypatch)
    return KnowledgeBase(db_path=db_path)


def _read_code_assist():
    return (
        Path(__file__).resolve().parent.parent
        / "skills" / "code_assist.py"
    ).read_text(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────
# Fix 1 -- step-boundary truncation
# ─────────────────────────────────────────────────────────────────


def test_r01_short_recipe_unchanged():
    short = (
        'STEP 1: read_file path="a.py"\n'
        'STEP 2: done summary="ok"'
    )
    assert ca._truncate_recipe_to_steps(short, 1000) == short


def test_r02_truncate_drops_trailing_partial():
    # Three steps, second is fat. Cap admits step 1 + step 2 but NOT
    # step 3. Step 3 must be dropped wholesale, not chopped mid-args.
    s1 = 'STEP 1: read_file path="a.py"'
    s2 = 'STEP 2: write_file path="b.py" content="' + ("x" * 200) + '"'
    s3 = 'STEP 3: edit_file path="c.py" old="z" new="zz"'
    recipe = "\n".join([s1, s2, s3])
    cap = len(s1) + 1 + len(s2) + 5  # admits 1+2, not 3
    out = ca._truncate_recipe_to_steps(recipe, cap)
    assert "STEP 3:" not in out
    assert "STEP 2:" in out
    assert "STEP 1:" in out
    # Crucially: step 2 is COMPLETE (no mid-string cut).
    assert out.endswith('"')


def test_r03_first_step_alone_exceeds_cap():
    # If even the first step is fatter than the cap, return it
    # truncated rather than empty -- some output is better than none.
    big_step = 'STEP 1: write_file path="x" content="' + ("y" * 5000) + '"'
    out = ca._truncate_recipe_to_steps(big_step, 100)
    assert out.startswith("STEP 1:")
    assert len(out) <= 100


def test_r04_no_step_markers_falls_back_to_char_cut():
    """If the input has no STEP-N pattern at all, behaviour matches
    the legacy hard-cut (no surprises for callers passing junk)."""
    junk = "x" * 10000
    out = ca._truncate_recipe_to_steps(junk, 100)
    assert len(out) == 100


# ─────────────────────────────────────────────────────────────────
# Fix 2 -- recipe budget
# ─────────────────────────────────────────────────────────────────


def test_r11_recipe_max_is_8000():
    assert ca.RECIPE_MAX_CHARS_STEPFED == 8000


# ─────────────────────────────────────────────────────────────────
# Fix 3 -- shape-repetition bail
# ─────────────────────────────────────────────────────────────────


def test_r21_identical_reasonings_match():
    a = (
        "Read lines 614-646 confirms _wait_for_task still calls "
        "_build_bar directly in all three places"
    )
    b = (
        "Read lines 620-650 confirms _wait_for_task still calls "
        "_build_bar directly in all three places"
    )
    phrase = ca._shape_repetition_phrase(a, b)
    assert phrase is not None
    # Concrete substantive overlap, not boilerplate.
    assert "_wait_for_task" in phrase or "_build_bar" in phrase


def test_r22_different_concrete_failures_no_bail():
    a = "old string not found in file"
    b = "syntax error unterminated string literal"
    assert ca._shape_repetition_phrase(a, b) is None


def test_r23_boilerplate_does_not_trigger():
    """Sentences composed mostly of stopwords share lots of word
    overlap but the substantive-tokens filter must reject them."""
    a = "The diff has been applied as it was specified in the prior recipe"
    b = "The diff has been applied as it was specified in the prior recipe"
    # Identical sentences DO match (this is technically a real
    # repetition signal). The filter only blocks shared 5-grams that
    # are >50% stopwords; a literal-identical sentence has at least
    # ONE 5-gram with enough non-stopword tokens to pass.
    # So this DOES bail. That's the right call -- if Claude is
    # writing the same review verbatim, retries aren't helping.
    out = ca._shape_repetition_phrase(a, b)
    assert out is not None  # sanity: literal identity DOES trigger

    # The genuine no-bail case: shared boilerplate fragments only.
    a = "The function it was the in of and"
    b = "The function it was the in of and"
    out2 = ca._shape_repetition_phrase(a, b)
    assert out2 is None  # all-stopwords window rejected


def test_r24_empty_inputs_no_bail():
    assert ca._shape_repetition_phrase("", "") is None
    assert ca._shape_repetition_phrase(None, "anything") is None
    assert ca._shape_repetition_phrase("anything", None) is None


def test_r25_only_4gram_overlap_no_bail():
    """Default n=5 -- a 4-token overlap should NOT trigger bail."""
    a = "cv match comp red"  # 4 tokens
    b = "cv match comp red flags entirely"
    assert ca._shape_repetition_phrase(a, b) is None


# ─────────────────────────────────────────────────────────────────
# Fix 4 -- already-read paths
# ─────────────────────────────────────────────────────────────────


def test_r31_extract_paths_from_recipe():
    recipe = (
        'STEP 1: read_file path="core/foo.py"\n'
        'STEP 2: edit_file path="interfaces/telegram_bot.py" '
        'old="x" new="y"\n'
        'STEP 3: done summary="ok"'
    )
    paths = ca._extract_project_paths(recipe)
    assert "core/foo.py" in paths
    assert "interfaces/telegram_bot.py" in paths


def test_r32_extract_paths_from_review_reasoning():
    reasoning = (
        "Read lines 100-200 of interfaces/telegram_bot.py confirms "
        "the change in core/util.py is wired correctly."
    )
    paths = ca._extract_project_paths(reasoning)
    assert "interfaces/telegram_bot.py" in paths
    assert "core/util.py" in paths


def test_r33_ignores_absolute_paths():
    text = "Reading C:\\Users\\me\\foo.py and /abs/bar.py"
    paths = ca._extract_project_paths(text)
    # Only project-relative shapes survive.
    assert all(":" not in p and not p.startswith("/") for p in paths)


def test_r34_ignores_bare_module_names():
    """`from core import foo` mentions `core` and `foo` but neither
    has a slash + extension, so the extractor must skip them."""
    text = "from core import foo  # no slash-path here"
    paths = ca._extract_project_paths(text)
    assert paths == set()


def test_r35_corrective_teach_signature_has_kwarg():
    """The signature must accept files_already_read as a keyword
    arg with a default of None, so older call sites still work."""
    import inspect
    sig = inspect.signature(ca._claude_corrective_teach)
    assert "files_already_read" in sig.parameters
    p = sig.parameters["files_already_read"]
    assert p.default is None


# ─────────────────────────────────────────────────────────────────
# Fix 5 -- shadow data on limitations
# ─────────────────────────────────────────────────────────────────


def test_r41_add_limitation_stores_shadow_fields(fresh_kb):
    lid = fresh_kb.add_limitation(
        tags=["x"], problem_summary="failed task",
        explanation="couldn't do it", trace_id="SEN-r41",
        qwen_plan_recipe='STEP 1: done summary="x"',
        qwen_plan_agreement=0.25,
    )
    row = fresh_kb.get_pattern(lid)
    assert row.category == "limitation"
    assert row.qwen_plan_recipe == 'STEP 1: done summary="x"'
    assert row.qwen_plan_agreement == pytest.approx(0.25, abs=1e-6)


def test_r42_add_limitation_back_compat_no_kwargs(fresh_kb):
    """Existing call sites without the new kwargs still work and
    leave the shadow columns NULL."""
    lid = fresh_kb.add_limitation(
        tags=[], problem_summary="oldstyle", explanation="e",
        trace_id="SEN-r42",
    )
    row = fresh_kb.get_pattern(lid)
    assert row.qwen_plan_recipe is None
    assert row.qwen_plan_agreement is None


def test_r43_failure_path_pipes_shadow():
    """Source-level: confirm the limitation call site in
    _run_agentic_pipeline forwards shadow_recipe + agreement."""
    src = _read_code_assist()
    # Find the failure-path block (the `else:` that calls add_limitation
    # after MAX_TEACH_ATTEMPTS exhausted / bail).
    add_lim_idx = src.rfind("kb.add_limitation(")
    assert add_lim_idx > 0
    block = src[add_lim_idx:add_lim_idx + 1500]
    assert "qwen_plan_recipe=shadow_recipe" in block
    assert "qwen_plan_agreement=shadow_agreement" in block


# ─────────────────────────────────────────────────────────────────
# Fix 6 -- QWENCODER.md
# ─────────────────────────────────────────────────────────────────


def test_r51_qwencoder_file_exists():
    p = config.PERSONA_DIR / "QWENCODER.md"
    assert p.exists(), f"QWENCODER.md missing at {p}"
    assert p.stat().st_size > 1000, "memo too short to be useful"


def test_r52_qwencoder_in_protected_files():
    assert "QWENCODER.md" in config.PROTECTED_FILES


def test_r53_qwencoder_inject_cap_set():
    assert "QWENCODER.md" in config.PERSONA_INJECT_MAX_CHARS
    assert config.PERSONA_INJECT_MAX_CHARS["QWENCODER.md"] >= 4000


def test_r54_load_qwencoder_memo_reads_file():
    memo = ca._load_qwencoder_memo()
    assert memo  # non-empty
    # Sanity: the memo references the recipe contract.
    assert "STEP" in memo


def test_r55_load_qwencoder_memo_missing_returns_empty(
    tmp_path, monkeypatch,
):
    """When the file is missing, the loader returns "" and the
    shadow path still works (just without the memo)."""
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)
    assert ca._load_qwencoder_memo() == ""


def test_r56_shadow_system_prompt_composes_with_memo():
    """Memo is APPENDED to BASE so the strict contract is read first."""
    base = ca.QWEN_SHADOW_SYSTEM_BASE
    full = ca._qwen_shadow_system_prompt()
    assert full.startswith(base)
    # And the memo content is in there.
    assert "QWENCODER MEMO" in full or "STEP" in full


def test_r57_shadow_plan_uses_dynamic_composer():
    """Source-level check that the legacy one-shot path in
    _qwen_shadow_plan calls the dynamic system-prompt composer
    rather than the static BASE constant.

    Phase 16 Batch A added an agentic path FIRST (which uses its
    own SHADOW_PLANNER_SYSTEM in qwen_agent), so we now scan the
    WHOLE function body; the dynamic composer call lives in the
    legacy fallback that runs when SHADOW_PLAN_USE_TOOLS=False."""
    src = _read_code_assist()
    plan_idx = src.find("async def _qwen_shadow_plan")
    assert plan_idx > 0
    next_def = src.find("\nasync def ", plan_idx + 1)
    if next_def < 0:
        next_def = src.find("\ndef ", plan_idx + 1)
    body = src[plan_idx:next_def] if next_def > 0 else src[plan_idx:]
    assert "_qwen_shadow_system_prompt" in body


def test_r61_qwen_generate_format_json_kwarg():
    """The format_json kwarg exists and defaults to True (so the
    production stepfed transcription path keeps JSON mode without
    a code change)."""
    import inspect
    sig = inspect.signature(ca._qwen_generate)
    assert "format_json" in sig.parameters
    assert sig.parameters["format_json"].default is True


def test_r62_shadow_plan_passes_format_json_false():
    """Source-level: the LEGACY one-shot path in _qwen_shadow_plan
    must pass format_json=False so Ollama doesn't lock Qwen into
    structured-JSON output (which overrides the system prompt's
    STEP-N format directive and pins every shadow agreement score
    to 0.0).

    Phase 16 Batch A added an agentic path BEFORE the legacy
    one-shot path; this test still enforces the invariant on the
    legacy path because SHADOW_PLAN_USE_TOOLS=False is a documented
    fallback. Search the WHOLE function body, not just the first
    3500 chars (the agentic path expanded the function size)."""
    src = _read_code_assist()
    plan_idx = src.find("async def _qwen_shadow_plan")
    assert plan_idx > 0
    # Find the function end by searching for the next top-level def
    # / async def AFTER the shadow_plan signature.
    next_def = src.find("\nasync def ", plan_idx + 1)
    if next_def < 0:
        next_def = src.find("\ndef ", plan_idx + 1)
    body = src[plan_idx:next_def] if next_def > 0 else src[plan_idx:]
    # The legacy fallback path must still pass format_json=False to
    # _qwen_generate -- one of these literal forms.
    assert (
        "format_json=False" in body
        or "False,  # format_json=False" in body
        or "False  # format_json=False" in body
    ), (
        "_qwen_shadow_plan legacy path must pass format_json=False "
        "to _qwen_generate"
    )


def test_r63_production_stepfed_call_sites_unchanged():
    """Phase 15d-bugfix back-compat: the production stepfed
    transcription path uses _qwen_generate WITHOUT specifying
    format_json (so it falls through to the default True)."""
    src = _read_code_assist()
    # Two production call sites (stepfed transcription) plus the
    # shadow path = three total. The shadow path passes False;
    # the other two should not pass it.
    qwen_calls = []
    idx = 0
    while True:
        idx = src.find("_qwen_generate,\n", idx)
        if idx < 0:
            break
        # Take ~600 chars after the call to inspect the kwargs.
        qwen_calls.append(src[idx:idx + 700])
        idx += 1
    assert len(qwen_calls) >= 3, (
        f"expected at least 3 _qwen_generate call sites, "
        f"found {len(qwen_calls)}"
    )
    # Exactly one passes format_json=False. The rest should NOT.
    false_count = sum(
        1 for c in qwen_calls
        if "format_json=False" in c or "False  # format_json" in c
        or "False,  # format_json" in c
    )
    assert false_count == 1, (
        f"expected exactly 1 call site to pass format_json=False, "
        f"found {false_count}"
    )


def test_r71_is_real_solution_accepts_diff_body():
    """Phase 14b shape: solution_code stores the actual diff text
    starting with `diff --git`. Pre-bugfix, this failed the gate
    and 8 successful KB patterns got silently deleted on restart."""
    diff_body = (
        "diff --git a/math_utils.py b/math_utils.py\n"
        "index abc123..def456 100644\n"
        "--- a/math_utils.py\n"
        "+++ b/math_utils.py\n"
        "@@ -10,3 +10,7 @@\n"
        " def add(a, b):\n"
        "     return a + b\n"
        "+\n"
        "+def gcd(a, b):\n"
        "+    return b if a == 0 else gcd(b % a, a)\n"
    )
    assert ca._is_real_solution(diff_body) is True


def test_r72_is_real_solution_accepts_hunk_plus_minus():
    """Diff body with hunk header + add/remove lines but no
    `diff --git` header (e.g. a partial diff fragment) still
    passes."""
    fragment = (
        "@@ -5,7 +5,7 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    return 2\n"
        " def bar():\n"
        "     return 3\n"
    )
    assert ca._is_real_solution(fragment) is True


def test_r73_is_real_solution_accepts_diff_stat():
    """Pre-Phase-14b shape (back-compat): the `git diff --stat`
    cosmetic output. Old KB rows still populated this; the gate
    must keep accepting them so we don't NEW-bug existing rows."""
    stat = (
        " core/foo.py | 12 ++++++------\n"
        " core/bar.py |  3 ++-\n"
        " 2 files changed, 9 insertions(+), 6 deletions(-)\n"
    )
    assert ca._is_real_solution(stat) is True


def test_r74_is_real_solution_accepts_python_source():
    py = (
        "def gcd(a, b):\n"
        "    while b:\n"
        "        a, b = b, a % b\n"
        "    return a\n"
        "print(gcd(12, 8))\n"
    )
    assert ca._is_real_solution(py) is True


def test_r75_is_real_solution_rejects_literal():
    assert ca._is_real_solution("42") is False
    assert ca._is_real_solution("'hello'") is False
    assert ca._is_real_solution("") is False
    assert ca._is_real_solution("   ") is False
    # Just under the 30-char minimum -> reject
    assert ca._is_real_solution("x = 1\ny = 2\nz = 3") is False


def test_r76_cleanup_archives_not_deletes(fresh_kb):
    """Phase 15d-bugfix-2: the cleanup path now archives instead
    of deleting. This is the architectural fix that protects
    against future quality-gate misjudgments -- archived rows are
    recoverable via /kb restore <id>; deleted rows are not."""
    # Seed a pattern whose solution_code will fail the gate (too
    # short, no markers).
    pid_bad = fresh_kb.add_pattern(
        tags=["x"], problem_summary="bad",
        solution_code="x = 1",  # too short, no diff/Python markers
        solution_pattern="STEP 1: done", explanation="e",
        trace_id="SEN-r76-bad",
    )
    pid_good = fresh_kb.add_pattern(
        tags=["x"], problem_summary="good",
        solution_code=(
            "diff --git a/foo.py b/foo.py\n"
            "@@ -1,3 +1,4 @@\n"
            " a\n"
            "-b\n"
            "+c\n"
        ),
        solution_pattern="STEP 1: done", explanation="e",
        trace_id="SEN-r76-good",
    )
    n = fresh_kb.cleanup_low_quality_patterns()
    assert n == 1
    # The "bad" row is ARCHIVED, not deleted -- still on disk.
    bad = fresh_kb.get_pattern(pid_bad)
    assert bad is not None, "row was DELETED (regression!)"
    assert bad.state == "archived"
    assert bad.archived_at is not None
    # The "good" diff-body row is untouched.
    good = fresh_kb.get_pattern(pid_good)
    assert good is not None
    assert good.state == "active"


def test_r77_cleanup_skips_already_archived(fresh_kb):
    """Re-running cleanup after an earlier run shouldn't re-archive
    the same rows (idempotent + cheap to re-run)."""
    pid = fresh_kb.add_pattern(
        tags=["x"], problem_summary="bad",
        solution_code="x = 1",
        solution_pattern="STEP 1: done", explanation="e",
        trace_id="SEN-r77",
    )
    first = fresh_kb.cleanup_low_quality_patterns()
    assert first == 1
    second = fresh_kb.cleanup_low_quality_patterns()
    assert second == 0  # nothing left to archive


def test_r78_real_code_diff_survives_cleanup(fresh_kb):
    """Regression: the EXACT shape that was being deleted in
    production (Phase 14b diff text from a /code teach) must
    survive the cleanup pass on bot startup."""
    # This is the literal shape stored by _run_agentic_pipeline
    # after a successful /code: full git diff body, capped at 2000.
    realistic_diff = (
        "diff --git a/interfaces/telegram_bot.py b/interfaces/telegram_bot.py\n"
        "index abc123..def456 100644\n"
        "--- a/interfaces/telegram_bot.py\n"
        "+++ b/interfaces/telegram_bot.py\n"
        "@@ -14,2 +14,2 @@\n"
        "-def _build_bar(pct: int, w: int = 10) -> str:\n"
        "-    return '#' * f + '.' * (w - f)\n"
        "+def _build_bar(pct: int, w: int = 10) -> str:\n"
        "+    return '🟦' * f + '⬜' * (w - f)\n"
    )
    pid = fresh_kb.add_pattern(
        tags=["test"], problem_summary="emoji bar",
        solution_code=realistic_diff,
        solution_pattern="STEP 1: edit_file ...", explanation="e",
        trace_id="SEN-r78",
    )
    fresh_kb.cleanup_low_quality_patterns()
    survivor = fresh_kb.get_pattern(pid)
    assert survivor is not None
    assert survivor.state == "active", (
        "real diff body was archived -- this was the original bug"
    )


def test_r58_memo_has_recipe_contract():
    """Sanity check: the memo teaches the canonical recipe shape +
    addresses the most common failure modes we observed in 15c
    production traces (missing new=, markdown bullets, backticks)."""
    p = config.PERSONA_DIR / "QWENCODER.md"
    memo = p.read_text(encoding="utf-8")
    # Recipe contract spelled out
    assert "STEP" in memo
    assert "done(summary" in memo or "done summary" in memo
    # Each common failure pattern is called out
    must_mention = [
        "edit_file",     # the high-risk tool
        "write_file",    # the preferred path
        "old",           # exact-match constraint
        "verbatim",      # don't paraphrase
        "project-relative",  # path rules
    ]
    for term in must_mention:
        assert term in memo, f"memo missing concept: {term}"
