"""Phase 18 -- /gwen v2 ECC tests.

Replaces the old `core.qwen_agent.run_agent` (native tool-calling)
path with `core.gwen_agent.run_gwen_open`:
  - Unsandboxed open tools (read/write/edit_file, list_dir, run_bash)
  - Literal-recipe fast path (no LLM if input starts with STEP N:)
  - English path via Qwen text-gen + same executor

Test groups:
  P -- _open_resolve path expansion (~, abs, rel, backslash)
  T -- open tool dispatch (read/write/edit/list_dir/run_bash)
  L -- literal-recipe detection + /gwen-prefix stripping
  E -- _execute_recipe parser + per-step error handling
  R -- run_gwen_open routing (literal vs english)
  S -- GwenAssistSkill integration
  B -- PROMPT_BRIEF.md content guarantees
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Project root for source-level brief checks
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


# =====================================================================
# Group P: _open_resolve
# =====================================================================


def test_p01_expanduser_tilde():
    from core.gwen_agent import _open_resolve
    home = Path.home()
    out = _open_resolve("~/Downloads/test_p01.txt")
    assert str(out).startswith(str(home))
    assert out.name == "test_p01.txt"


def test_p02_absolute_path_passthrough(tmp_path):
    from core.gwen_agent import _open_resolve
    abs_path = str(tmp_path / "abs_target.txt")
    out = _open_resolve(abs_path)
    assert out == (tmp_path / "abs_target.txt").resolve()


def test_p03_relative_resolves_against_project_root():
    from core import config
    from core.gwen_agent import _open_resolve
    out = _open_resolve("core/config.py")
    assert out == (config.PROJECT_ROOT / "core" / "config.py").resolve()


def test_p04_backslash_normalized():
    from core.gwen_agent import _open_resolve
    a = _open_resolve("core\\config.py")
    b = _open_resolve("core/config.py")
    assert a == b


def test_p05_no_sandbox_check_on_escape(tmp_path):
    """Phase 18: NO ValueError on path escaping PROJECT_ROOT."""
    from core.gwen_agent import _open_resolve
    # tmp_path is almost certainly outside PROJECT_ROOT; should resolve
    # without raising.
    out = _open_resolve(str(tmp_path / "escape.txt"))
    assert isinstance(out, Path)


def test_p06_dotdot_collapses_via_resolve(tmp_path):
    from core.gwen_agent import _open_resolve
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    out = _open_resolve(str(nested / ".." / "b" / "x.txt"))
    assert out == (nested / "x.txt").resolve()


# =====================================================================
# Group T: open tool dispatch
# =====================================================================


def test_t01_open_write_then_read_outside_project(tmp_path):
    from core.gwen_agent import open_read_file, open_write_file
    target = tmp_path / "outside.txt"
    w = open_write_file(str(target), "hello\n")
    assert w.get("ok") is True
    assert w["bytes_written"] == len("hello\n")
    r = open_read_file(str(target))
    assert r["ok"] is True
    assert r["content"] == "hello\n"
    assert r["truncated"] is False


def test_t02_read_missing_file(tmp_path):
    from core.gwen_agent import open_read_file
    r = open_read_file(str(tmp_path / "nope.txt"))
    assert "error" in r and "not found" in r["error"]


def test_t03_read_directory_returns_error(tmp_path):
    from core.gwen_agent import open_read_file
    r = open_read_file(str(tmp_path))
    assert "error" in r and "not a file" in r["error"]


def test_t04_write_creates_parent_dirs(tmp_path):
    from core.gwen_agent import open_write_file
    target = tmp_path / "deep" / "deeper" / "x.txt"
    w = open_write_file(str(target), "data")
    assert w["ok"] is True
    assert target.exists()


def test_t05_edit_file_unique_anchor(tmp_path):
    from core.gwen_agent import open_edit_file, open_write_file
    target = tmp_path / "edit.txt"
    open_write_file(str(target), "alpha\nbeta\ngamma\n")
    e = open_edit_file(str(target), old="beta", new="BETA")
    assert e["ok"] is True
    assert target.read_text() == "alpha\nBETA\ngamma\n"


def test_t06_edit_file_missing_anchor(tmp_path):
    from core.gwen_agent import open_edit_file, open_write_file
    target = tmp_path / "edit.txt"
    open_write_file(str(target), "alpha\n")
    e = open_edit_file(str(target), old="MISSING", new="x")
    assert "error" in e and "not found" in e["error"]


def test_t07_edit_file_ambiguous_anchor(tmp_path):
    from core.gwen_agent import open_edit_file, open_write_file
    target = tmp_path / "edit.txt"
    open_write_file(str(target), "x\nx\nx\n")
    e = open_edit_file(str(target), old="x", new="y")
    assert "error" in e and "appears 3 times" in e["error"]


def test_t08_edit_file_missing(tmp_path):
    from core.gwen_agent import open_edit_file
    e = open_edit_file(str(tmp_path / "noexist.txt"), old="a", new="b")
    assert "error" in e and "not found" in e["error"]


def test_t09_list_dir(tmp_path):
    from core.gwen_agent import open_list_dir
    (tmp_path / "f1.txt").write_text("")
    (tmp_path / "sub").mkdir()
    r = open_list_dir(str(tmp_path))
    assert r["ok"] is True
    items = r["items"]
    # Order is sorted, so file < dir? Actually sorted by name; both
    # appear regardless.
    joined = " ".join(items)
    assert "f1.txt" in joined
    assert "sub" in joined


def test_t10_list_dir_not_a_dir(tmp_path):
    from core.gwen_agent import open_list_dir, open_write_file
    target = tmp_path / "f.txt"
    open_write_file(str(target), "")
    r = open_list_dir(str(target))
    assert "error" in r and "not a directory" in r["error"]


def test_t11_run_bash_basic():
    from core.gwen_agent import open_run_bash
    # Cross-platform: python -c is universally available
    r = open_run_bash(command='python -c "print(42)"')
    assert r["ok"] is True
    assert r["return_code"] == 0
    assert "42" in r["stdout"]


def test_t12_run_bash_alias_cmd():
    """Recipes from external AI sometimes use `cmd` instead of `command`."""
    from core.gwen_agent import open_run_bash
    r = open_run_bash(cmd='python -c "print(7)"')
    assert r["ok"] is True
    assert "7" in r["stdout"]


def test_t13_run_bash_missing_command():
    from core.gwen_agent import open_run_bash
    r = open_run_bash()
    assert "error" in r and "missing" in r["error"]


def test_t14_run_bash_with_cwd(tmp_path):
    from core.gwen_agent import open_run_bash
    (tmp_path / "marker.txt").write_text("ok")
    if sys.platform == "win32":
        cmd = "dir marker.txt"
    else:
        cmd = "ls marker.txt"
    r = open_run_bash(command=cmd, cwd=str(tmp_path))
    assert r["ok"] is True
    assert r["return_code"] == 0


def test_t15_run_bash_bad_cwd():
    from core.gwen_agent import open_run_bash
    r = open_run_bash(command="echo x", cwd="C:/this/path/does/not/exist/anywhere")
    assert "error" in r and "not a directory" in r["error"]


def test_t16_run_bash_nonzero_exit_is_ok():
    """Non-zero return code is NOT a tool error -- it's reported."""
    from core.gwen_agent import open_run_bash
    r = open_run_bash(command='python -c "import sys; sys.exit(3)"')
    assert r["ok"] is True
    assert r["return_code"] == 3


# =====================================================================
# Group L: literal-recipe detection + /gwen prefix strip
# =====================================================================


def test_l01_detects_step1():
    from core.gwen_agent import _is_literal_recipe
    assert _is_literal_recipe("STEP 1: read_file path=\"a\"") is True


def test_l02_detects_with_gwen_prefix():
    from core.gwen_agent import _is_literal_recipe
    assert _is_literal_recipe("/gwen STEP 1: done summary=\"x\"") is True


def test_l03_detects_with_leading_whitespace():
    from core.gwen_agent import _is_literal_recipe
    assert _is_literal_recipe("   STEP 1: done summary=\"x\"") is True


def test_l04_case_insensitive():
    from core.gwen_agent import _is_literal_recipe
    assert _is_literal_recipe("step 1: done summary=\"x\"") is True


def test_l05_rejects_english():
    from core.gwen_agent import _is_literal_recipe
    assert _is_literal_recipe("create a file on my desktop") is False


def test_l06_rejects_empty_and_none():
    from core.gwen_agent import _is_literal_recipe
    assert _is_literal_recipe("") is False
    assert _is_literal_recipe(None) is False  # type: ignore[arg-type]


def test_l07_rejects_step_in_middle_of_prose():
    from core.gwen_agent import _is_literal_recipe
    # "STEP 1:" must be at the START -- prose before it doesn't count
    assert _is_literal_recipe("Here is a recipe. STEP 1: do x") is False


def test_l08_strip_gwen_prefix_per_line():
    from core.gwen_agent import _strip_gwen_prefixes
    inp = "/gwen STEP 1: read_file path=\"a\"\n/gwen STEP 2: done summary=\"x\""
    out = _strip_gwen_prefixes(inp)
    assert out == "STEP 1: read_file path=\"a\"\nSTEP 2: done summary=\"x\""


def test_l09_strip_gwen_prefix_preserves_indent():
    from core.gwen_agent import _strip_gwen_prefixes
    inp = "  /gwen STEP 1: done summary=\"x\""
    out = _strip_gwen_prefixes(inp)
    assert out == "  STEP 1: done summary=\"x\""


def test_l10_strip_gwen_prefix_drops_bare_gwen_line():
    from core.gwen_agent import _strip_gwen_prefixes
    inp = "/gwen\nSTEP 1: done summary=\"x\""
    out = _strip_gwen_prefixes(inp)
    assert out == "STEP 1: done summary=\"x\""


def test_l11_strip_gwen_prefix_leaves_non_gwen_lines():
    from core.gwen_agent import _strip_gwen_prefixes
    inp = "STEP 1: read_file path=\"a\"\nSTEP 2: done summary=\"x\""
    out = _strip_gwen_prefixes(inp)
    assert out == inp


def test_l12_strip_inline_gwen_before_step_marker():
    """Phase 18d-gz polish: when Telegram collapses newlines, mid-line
    `/gwen STEP N:` patterns should still be normalized away."""
    from core.gwen_agent import _strip_gwen_prefixes
    inp = "STEP 1: write_file path=\"x\" content=\"hi\" /gwen STEP 2: done summary=\"ok\""
    out = _strip_gwen_prefixes(inp)
    assert "/gwen" not in out, f"got: {out!r}"
    assert out == 'STEP 1: write_file path="x" content="hi" STEP 2: done summary="ok"'


def test_l13_inline_gwen_strip_preserves_legitimate_gwen_in_values():
    """Only strip `/gwen ` when followed by STEP marker. Don't corrupt
    `/gwen` mentions inside legitimate arg values."""
    from core.gwen_agent import _strip_gwen_prefixes
    inp = 'STEP 1: done summary="rerun via /gwen later"'
    out = _strip_gwen_prefixes(inp)
    assert out == inp


def test_l14_collapsed_paste_with_inline_gwen_parses_full_steps(tmp_path):
    """End-to-end: AI emitted multi-line with /gwen on every step;
    Telegram collapsed newlines into spaces. Parser still produces
    clean per-step bodies."""
    from core.gwen_agent import _execute_recipe
    target_a = (tmp_path / "a.txt").as_posix()
    target_b = (tmp_path / "b.txt").as_posix()
    recipe = (
        f'STEP 1: write_file path="{target_a}" content="A" '
        f'/gwen STEP 2: write_file path="{target_b}" content="B" '
        f'/gwen STEP 3: done summary="ok"'
    )
    out = _execute_recipe(recipe, "SEN-test-l14")
    assert out["completed_via_done"], str(out["session"])
    assert Path(target_a).read_text() == "A"
    assert Path(target_b).read_text() == "B"


# =====================================================================
# Group E: _execute_recipe (literal-path executor)
# =====================================================================


def test_e01_simple_done(tmp_path, monkeypatch):
    from core.gwen_agent import _execute_recipe
    out = _execute_recipe(
        'STEP 1: done summary="hello"',
        trace_id="SEN-test-e01",
    )
    assert out["completed_via_done"] is True
    assert out["steps"] == 1
    assert out["summary"] == "hello"
    assert out["error"] is None


def test_e02_write_then_done(tmp_path):
    from core.gwen_agent import _execute_recipe
    target = tmp_path / "e02.txt"
    # Use as_posix() -- recipes from external AIs always use forward
    # slashes (PROMPT_BRIEF.md mandates it). Backslashed paths in the
    # recipe trigger \t/\U JSON-escape decoding bugs in the parser
    # before _open_resolve gets a chance to normalize.
    recipe = (
        f'STEP 1: write_file path="{target.as_posix()}" content="hi"\n'
        f'STEP 2: done summary="wrote e02"'
    )
    out = _execute_recipe(recipe, trace_id="SEN-test-e02")
    assert out["completed_via_done"] is True
    assert out["steps"] == 2
    assert target.read_text() == "hi"


def test_e03_zero_steps(tmp_path):
    from core.gwen_agent import _execute_recipe
    out = _execute_recipe("nothing here", trace_id="SEN-test-e03")
    assert out["steps"] == 0
    assert out["error"] == "recipe parser returned 0 steps"
    assert out["completed_via_done"] is False


def test_e04_error_mid_recipe_continues(tmp_path):
    """An errored step does NOT abort the recipe; later steps run."""
    from core.gwen_agent import _execute_recipe
    target = tmp_path / "e04.txt"
    p = target.as_posix()
    recipe = (
        f'STEP 1: edit_file path="{p}" old="x" new="y"\n'  # missing file
        f'STEP 2: write_file path="{p}" content="recovered"\n'
        f'STEP 3: done summary="ok"'
    )
    out = _execute_recipe(recipe, trace_id="SEN-test-e04")
    assert out["steps"] == 3
    assert out["completed_via_done"] is True
    assert target.read_text() == "recovered"
    assert "error" in out["session"][0]["result"]
    assert out["session"][1]["result"].get("ok") is True


def test_e05_unknown_tool_errors_then_continues(tmp_path):
    from core.gwen_agent import _execute_recipe
    recipe = (
        'STEP 1: nuke_database path="prod"\n'
        'STEP 2: done summary="x"'
    )
    out = _execute_recipe(recipe, trace_id="SEN-test-e05")
    # Unknown tool name -- _parse_step_text_to_tool_call returns None
    # so _execute_recipe records "unparseable" rather than dispatching.
    assert out["steps"] == 2
    assert "did not parse" in out["session"][0]["result"]["error"]


def test_e06_strip_gwen_prefix_inside_executor():
    """Recipe with `/gwen ` prefixes should still execute."""
    from core.gwen_agent import _execute_recipe
    out = _execute_recipe(
        '/gwen STEP 1: done summary="prefixed"',
        trace_id="SEN-test-e06",
    )
    assert out["completed_via_done"] is True
    assert out["summary"] == "prefixed"


def test_e07_session_trace_shape(tmp_path):
    from core.gwen_agent import _execute_recipe
    target = tmp_path / "e07.txt"
    recipe = (
        f'STEP 1: write_file path="{target.as_posix()}" content="x"\n'
        f'STEP 2: done summary="ok"'
    )
    out = _execute_recipe(recipe, trace_id="SEN-test-e07")
    s0 = out["session"][0]
    assert set(s0.keys()) == {"step", "tool", "args", "result"}
    assert s0["tool"] == "write_file"
    assert s0["result"]["ok"] is True


# =====================================================================
# Group R: run_gwen_open routing
# =====================================================================


def test_r01_literal_path_no_llm_call(tmp_path):
    """Literal recipe must NOT invoke _qwen_generate."""
    from core import gwen_agent

    # Patch the lazy import target; if we hit english_to_recipe it'll
    # call this and the test fails.
    called = {"hit": False}

    def boom(*a, **kw):
        called["hit"] = True
        raise AssertionError("LLM was called on literal-recipe path")

    with patch("skills.code_assist._qwen_generate", side_effect=boom):
        out = gwen_agent.run_gwen_open(
            'STEP 1: done summary="hi"',
            trace_id="SEN-test-r01",
            model="qwen2.5-coder:3b",
        )

    assert called["hit"] is False
    assert out["mode"] == "literal"
    assert out["completed_via_done"] is True


def test_r02_english_path_calls_llm():
    from core import gwen_agent

    fake_recipe = 'STEP 1: done summary="from-english"'

    with patch(
        "skills.code_assist._qwen_generate",
        return_value=fake_recipe,
    ) as mock_gen:
        out = gwen_agent.run_gwen_open(
            "create a desktop note please",
            trace_id="SEN-test-r02",
            model="qwen2.5-coder:3b",
        )

    assert mock_gen.called
    assert out["mode"] == "english"
    assert out["recipe"] == fake_recipe
    assert out["summary"] == "from-english"


def test_r03_english_path_llm_failure(tmp_path):
    from core import gwen_agent

    with patch(
        "skills.code_assist._qwen_generate",
        side_effect=RuntimeError("ollama down"),
    ):
        out = gwen_agent.run_gwen_open(
            "do something",
            trace_id="SEN-test-r03",
            model="qwen2.5-coder:3b",
        )
    assert out["mode"] == "english"
    assert out["completed_via_done"] is False
    assert "ollama down" in out["error"]


def test_r04_literal_with_gwen_prefix_routes_to_literal():
    from core import gwen_agent

    def boom(*a, **kw):
        raise AssertionError("LLM was called on prefixed literal recipe")

    with patch("skills.code_assist._qwen_generate", side_effect=boom):
        out = gwen_agent.run_gwen_open(
            '/gwen STEP 1: done summary="ok"',
            trace_id="SEN-test-r04",
            model="qwen2.5-coder:3b",
        )
    assert out["mode"] == "literal"
    assert out["completed_via_done"] is True


# =====================================================================
# Group S: GwenAssistSkill integration
# =====================================================================


def test_s01_skill_io_schemas():
    from skills.gwen_assist import (
        GwenAssistInput,
        GwenAssistOutput,
        GwenAssistSkill,
    )
    assert GwenAssistSkill.input_schema is GwenAssistInput
    assert GwenAssistSkill.output_schema is GwenAssistOutput


def test_s02_skill_executes_literal_recipe(tmp_path):
    from skills.gwen_assist import GwenAssistInput, GwenAssistSkill

    target = tmp_path / "s02.txt"
    skill = GwenAssistSkill()
    inp = GwenAssistInput(
        text=f'STEP 1: write_file path="{target.as_posix()}" content="ok"\n'
             f'STEP 2: done summary="wrote s02"',
    )

    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-s02"))

    assert out.solved_by == "gwen_ok"
    assert out.completed is True
    assert out.steps_executed == 2
    assert out.mode == "literal"
    assert "wrote s02" in out.solution or "literal" in out.solution
    assert target.read_text() == "ok"


def test_s03_skill_failure_marks_gwen_failed():
    """Recipe that never calls done() -> solved_by='gwen_failed'."""
    from skills.gwen_assist import GwenAssistInput, GwenAssistSkill

    skill = GwenAssistSkill()
    inp = GwenAssistInput(text='STEP 1: list_dir path="."')

    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-s03"))
    assert out.solved_by == "gwen_failed"
    assert out.completed is False


def test_s04_skill_solution_contains_trace_header():
    from skills.gwen_assist import GwenAssistInput, GwenAssistSkill

    skill = GwenAssistSkill()
    inp = GwenAssistInput(text='STEP 1: done summary="hi"')
    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-s04"))
    assert "/gwen" in out.solution
    assert "literal" in out.solution


def test_s05_skill_english_path_with_mocked_llm(tmp_path):
    from skills.gwen_assist import GwenAssistInput, GwenAssistSkill

    target = tmp_path / "s05.txt"
    fake_recipe = (
        f'STEP 1: write_file path="{target.as_posix()}" content="english-ok"\n'
        f'STEP 2: done summary="english path worked"'
    )
    with patch(
        "skills.code_assist._qwen_generate",
        return_value=fake_recipe,
    ):
        skill = GwenAssistSkill()
        inp = GwenAssistInput(text="please make file s05")
        out = asyncio.run(skill.execute(inp, trace_id="SEN-test-s05"))

    assert out.mode == "english"
    assert out.completed is True
    assert target.read_text() == "english-ok"
    # The skill renders the Qwen recipe so the user can audit
    assert "english-ok" in out.solution or "Qwen recipe" in out.solution


# =====================================================================
# Group B: PROMPT_BRIEF.md content guarantees
# =====================================================================


BRIEF_PATH = PROJECT / "workspace" / "persona" / "PROMPT_BRIEF.md"


def test_b01_brief_exists_and_nontrivial():
    assert BRIEF_PATH.exists()
    text = BRIEF_PATH.read_text(encoding="utf-8")
    assert len(text) > 1500


def test_b02_brief_mentions_gwen_command():
    text = BRIEF_PATH.read_text(encoding="utf-8")
    assert "/gwen" in text


def test_b03_brief_mentions_qcode_command():
    text = BRIEF_PATH.read_text(encoding="utf-8")
    assert "/qcode" in text


def test_b04_brief_instructs_gwen_prefix_only_on_step_1():
    """Phase 18d-gz polish (2026-05-07): brief now mandates `/gwen `
    on STEP 1 ONLY, not every step. Pre-polish wording was
    'Prefix EVERY step with /gwen' which caused Gemini and other
    AIs to emit `/gwen ` on every line -- when Telegram collapses
    newlines, the inline /gwen markers smush into prior step values."""
    text = BRIEF_PATH.read_text(encoding="utf-8")
    lower = text.lower()
    # Must explicitly tell the AI to prefix STEP 1 only
    assert "prefix only step 1" in lower or "/gwen ` only on step 1" in lower or "step 1 with `/gwen `" in lower, (
        "brief should explicitly say /gwen goes on STEP 1 only"
    )
    # Must still mention `/gwen ` somewhere (the canonical example uses it)
    assert "/gwen " in text


def test_b05_brief_has_gwen_prefixed_example():
    text = BRIEF_PATH.read_text(encoding="utf-8")
    # At least one canonical example with `/gwen STEP 1:`
    assert "/gwen STEP 1:" in text


def test_b06_brief_documents_unsandboxed_paths():
    text = BRIEF_PATH.read_text(encoding="utf-8")
    # Either ~/Desktop or C:/Users absolute path (real-world targets)
    assert "~/Desktop" in text or "C:/Users" in text


def test_b07_brief_calls_out_anti_patterns():
    text = BRIEF_PATH.read_text(encoding="utf-8")
    assert "Anti-Pattern" in text or "ANTI-PATTERN" in text or "anti-pattern" in text.lower()


def test_b08_brief_calls_out_backslash_path_antipattern():
    text = BRIEF_PATH.read_text(encoding="utf-8")
    # Forward-slash rule must appear somewhere
    lower = text.lower()
    assert "forward slash" in lower or "backslash" in lower


def test_b09_brief_documents_done_as_final_step():
    text = BRIEF_PATH.read_text(encoding="utf-8")
    lower = text.lower()
    assert "done" in lower
    assert "summary" in lower


def test_b10_brief_distinguishes_gwen_vs_qcode_sandbox():
    text = BRIEF_PATH.read_text(encoding="utf-8")
    # Must explain when to pick which command
    lower = text.lower()
    assert "sandbox" in lower
    assert "project" in lower


# =====================================================================
# Group I: imports + wiring sanity
# =====================================================================


def test_i01_skill_registered_in_command_map():
    from core import config
    assert config.COMMAND_AGENT_MAP.get("/gwen") == "gwen_assistant"


def test_i02_agent_yaml_pipeline_uses_gwen_assist():
    import yaml
    yaml_path = PROJECT / "agents" / "gwen_assistant.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["name"] == "gwen_assistant"
    assert "gwen_assist" in data["skill_pipeline"]


def test_i03_open_dispatch_has_all_tools():
    from core.gwen_agent import OPEN_TOOL_DISPATCH
    expected = {"read_file", "write_file", "edit_file",
                "list_dir", "run_bash"}
    assert set(OPEN_TOOL_DISPATCH.keys()) == expected


def test_i04_run_gwen_open_returns_required_keys(tmp_path):
    """Public contract: every call returns these keys."""
    from core.gwen_agent import run_gwen_open
    out = run_gwen_open(
        'STEP 1: done summary="contract-test"',
        trace_id="SEN-test-i04",
        model="qwen2.5-coder:3b",
    )
    required = {"summary", "session", "steps",
                "completed_via_done", "error", "mode"}
    assert required <= set(out.keys())


def test_i05_no_safe_resolve_in_gwen_agent():
    """gwen_agent must NOT use the sandboxed _safe_resolve from
    core.qwen_agent. That would defeat full system access. We check
    that the *call* form does not appear (`_safe_resolve(`); a bare
    name in a comment would not actually wire the sandbox in."""
    src = (PROJECT / "core" / "gwen_agent.py").read_text(encoding="utf-8")
    assert "_safe_resolve(" not in src
    # Also defensively check: not imported from core.qwen_agent
    assert "from core.qwen_agent import _safe_resolve" not in src


def test_i06_handle_gwen_present_in_bot():
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    assert "handle_gwen" in src
    assert 'CommandHandler("gwen"' in src


# =====================================================================
# Group M: multi-line input survives router + bot handler intact
# =====================================================================


def test_m01_router_preserves_newlines_in_text_arg():
    """Router fix: rest-of-input keeps internal whitespace incl \\n."""
    from core.router import route
    body = (
        'STEP 1: write_file path="x" content="a"\n'
        'STEP 2: done summary="ok"'
    )
    rr = route(f"/gwen {body}")
    assert rr.status == "ok"
    assert rr.args.get("text") == body
    # Newline must literally be present (not collapsed to space)
    assert "\n" in rr.args["text"]


def test_m02_router_preserves_multiline_for_qcode():
    """Same fix benefits /qcode too (any text-arg command)."""
    from core.router import route
    body = 'STEP 1: read_file path="a"\nSTEP 2: done summary="x"'
    rr = route(f"/qcode {body}")
    assert rr.status == "ok"
    assert rr.args.get("text") == body
    assert "\n" in rr.args["text"]


def test_m03_router_preserves_multispace_in_text():
    """Internal multi-space runs preserved (not collapsed)."""
    from core.router import route
    rr = route("/gwen hello   world")
    assert rr.status == "ok"
    assert rr.args.get("text") == "hello   world"


def test_m04_router_single_line_unchanged():
    """No-flag single-line input still routes to the same {text: ...}
    shape callers expect."""
    from core.router import route
    rr = route("/gwen hello world")
    assert rr.status == "ok"
    assert rr.args == {"text": "hello world"}


def test_m05_router_flags_path_unaffected():
    """Flag-parsing branch unchanged (/jobsearch etc. still work)."""
    from core.router import route
    rr = route("/jobsearch sales --location remote --results 5")
    assert rr.status == "ok"
    # Existing semantic: text + flag values
    assert rr.args.get("location") == "remote"
    assert rr.args.get("results") == "5"
    assert rr.args.get("text") == "sales"


def test_m06_router_empty_text_no_args():
    from core.router import route
    rr = route("/gwen")
    assert rr.status == "ok"
    assert rr.args == {}


def test_m07_handle_gwen_reads_message_text_not_args_join():
    """Source-level guard: handle_gwen MUST NOT use the
    `' '.join(context.args)` pattern that collapses newlines.
    It must read update.message.text and split with maxsplit=1.

    Live trigger: 2026-05-07 09:16Z, multi-line /gwen recipe lost
    its newlines when handler joined args, _parse_recipe_steps saw
    one giant step, _KV_RE greedily grabbed STEP-2's args into
    STEP-1 -> write_file got a stray `command=` kwarg -> bad args
    error.
    """
    src = (PROJECT / "interfaces" / "telegram_bot.py").read_text(
        encoding="utf-8",
    )
    handler_idx = src.find("async def handle_gwen")
    assert handler_idx > 0
    body = src[handler_idx:handler_idx + 3000]
    # NEW path -- read update.message.text and split(maxsplit=1)
    assert "update.message.text" in body
    assert "split(maxsplit=1)" in body
    # OLD bug pattern -- " ".join(context.args) -- must be GONE
    assert '" ".join(context.args)' not in body


# =====================================================================
# Group W: paste-wrap collapse inside arg values
# =====================================================================
# Telegram (and other chat clients) soft-wrap long lines on display.
# When the user copies and pastes, the wrap becomes a real newline.
# That real newline lands inside a content="..." or command="..."
# value, json.loads fails, the fallback path keeps the newline, and
# the downstream shell or Python interpreter sees a broken multi-line
# command/file. Phase 18 fix: collapse \s*\n\s* runs in raw_value to
# a single space inside the parser fallback. Intended newlines (via
# the escape sequence \n) are unaffected.


def test_w01_paste_wrap_in_command_collapses_to_space():
    """Real newline + indent inside command="..." -> single space."""
    from core.qwen_agent import _parse_step_text_to_tool_call

    step = (
        'run_bash command="python -c \\"import sys;\n'
        '  print(42)\\""'
    )
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed is not None
    cmd = parsed["function"]["arguments"]["command"]
    # Real newline collapsed -- no embedded newline in the command
    assert "\n" not in cmd
    # And the content joined with a single space (not concatenated)
    assert "import sys; print(42)" in cmd


def test_w02_paste_wrap_in_content_collapses_to_space():
    """Real newline + indent inside content="..." -> single space."""
    from core.qwen_agent import _parse_step_text_to_tool_call

    step = (
        'write_file path="x" content="import platform\n'
        '  print(platform.platform())"'
    )
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed is not None
    content = parsed["function"]["arguments"]["content"]
    assert "\n" not in content
    assert "import platform print(platform.platform())" in content


def test_w03_intended_backslash_n_still_decodes_to_newline():
    """The escape sequence \\n in the recipe text MUST still produce
    a real newline in the decoded arg value -- this is the documented
    way for users / external AI to write multi-line content."""
    from core.qwen_agent import _parse_step_text_to_tool_call

    # raw recipe (as a Python string): write_file path="x" content="line1\nline2"
    # The \n in the source is two characters: backslash + n.
    step = r'write_file path="x" content="line1\nline2"'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed is not None
    content = parsed["function"]["arguments"]["content"]
    assert content == "line1\nline2"


def test_w04_mixed_real_newline_and_escape_n():
    """If a value has BOTH a paste-wrap real newline AND an intended
    escape \\n, real one collapses to space and escape decodes to
    newline."""
    from core.qwen_agent import _parse_step_text_to_tool_call

    # raw_value has: "real_nl\n  escape_nl\\n"
    step = (
        'write_file path="x" content="real_nl\n'
        '  escape_nl\\n"'
    )
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed is not None
    content = parsed["function"]["arguments"]["content"]
    # exactly one real newline (from the escape), one space (from the wrap)
    assert content.count("\n") == 1
    assert "real_nl escape_nl" in content


def test_w05_three_line_wrap_collapses_to_two_spaces_total():
    """A long command wrapped onto 3 visual lines -> 2 wrap points
    -> 2 spaces (not preserved newlines)."""
    from core.qwen_agent import _parse_step_text_to_tool_call

    step = (
        'run_bash command="python -c \\"a=1;\n'
        '  b=2;\n'
        '  c=3;\\""'
    )
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed is not None
    cmd = parsed["function"]["arguments"]["command"]
    assert "\n" not in cmd
    assert "a=1; b=2; c=3;" in cmd


def test_w06_no_wrap_no_change():
    """Single-line value with no real newlines -- unchanged."""
    from core.qwen_agent import _parse_step_text_to_tool_call

    step = 'run_bash command="echo hello world"'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed["function"]["arguments"]["command"] == "echo hello world"


def test_w07_end_to_end_pasted_run_bash_with_wrap_runs(tmp_path):
    """Integration: a recipe whose run_bash command was wrapped onto
    multiple visual lines should now execute successfully (the actual
    failure mode caught at 13:28Z)."""
    import asyncio
    from skills.gwen_assist import GwenAssistInput, GwenAssistSkill

    out_path = (tmp_path / "out.txt").as_posix()
    # Simulate a Telegram-wrapped run_bash command -- real newlines
    # in the middle of the python -c argument
    text = (
        f'STEP 1: run_bash command="python -c \\"import sys;\n'
        f'  print(sys.version.split()[0])\\" > {out_path}"\n'
        f'STEP 2: done summary="wrote python version to {out_path}"'
    )

    skill = GwenAssistSkill()
    out = asyncio.run(
        skill.execute(GwenAssistInput(text=text), trace_id="SEN-test-w07"),
    )
    assert out.completed is True
    assert out.solved_by == "gwen_ok"
    # File should exist with non-empty content (Python version line)
    written = Path(out_path).read_text(encoding="utf-8").strip()
    # Python version is something like "3.12.0"; sanity-check format
    assert written, f"out.txt is empty -- command did not run cleanly: {written!r}"
    assert written[0].isdigit(), f"unexpected output: {written!r}"


def test_m08_end_to_end_multiline_via_router_to_skill(tmp_path):
    """Integration: route a multi-line /gwen recipe and verify the
    skill receives newlines and executes 3 steps (not 1 giant step).

    We bypass the worker dispatch and call the skill directly with
    the routed args, mirroring what the orchestrator does.
    """
    import asyncio
    from core.router import route
    from skills.gwen_assist import GwenAssistInput, GwenAssistSkill

    target = tmp_path / "m08.txt"
    body = (
        f'STEP 1: write_file path="{target.as_posix()}" content="ok"\n'
        f'STEP 2: read_file path="{target.as_posix()}"\n'
        f'STEP 3: done summary="m08 e2e"'
    )
    rr = route(f"/gwen {body}")
    assert rr.status == "ok"
    text = rr.args["text"]
    assert text.count("STEP") == 3  # all 3 STEPs present, not collapsed

    inp = GwenAssistInput(text=text)
    out = asyncio.run(GwenAssistSkill().execute(inp, trace_id="SEN-test-m08"))
    assert out.completed is True
    assert out.steps_executed == 3
    assert out.solved_by == "gwen_ok"
    assert target.read_text() == "ok"
