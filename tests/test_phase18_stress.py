"""Phase 18 stress harness -- 10 progressive /gwen scenarios.

Each test runs through the REAL recipe parser + open-tool dispatcher
(`core.gwen_agent.run_gwen_open` literal-recipe path). No Telegram,
no LLM, no mocks of the executor -- only the bot/network layer is
elided, so anything that breaks here would also break in production
under the same input.

T01 - trivial write+done
T02 - multi-step run_bash chain
T03 - single-line collapsed paste (the live production failure)
T04 - Unicode smart-quote substitution (Telegram auto-format)
T05 - long content with many \\n escapes (real script)
T06 - edit_file with unique multi-line anchor
T07 - multi-file mini-app + integration run
T08 - shell pipe + redirect
T09 - SQLite-backed CLI: write tool, exercise multiple subcommands
T10 - daily-ops dashboard (the ambitious build)

Tmpdir-isolated: every test writes under tmp_path, never the real
Desktop or PROJECT_ROOT.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from core.gwen_agent import _execute_recipe, run_gwen_open  # noqa: E402


def _run(recipe: str, trace_id: str = "SEN-stress") -> dict:
    """Run a recipe through the literal-path executor and return
    the result dict. Does not raise; failures live in result['error']
    and per-step result['error'] entries."""
    return _execute_recipe(recipe, trace_id=trace_id)


def _summary(out: dict) -> str:
    parts = [
        f"steps={out['steps']}",
        f"completed={out['completed_via_done']}",
    ]
    if out.get("error"):
        parts.append(f"err={out['error']}")
    for s in out.get("session", []):
        i = s.get("step")
        tool = s.get("tool")
        res = s.get("result", {}) or {}
        if "error" in res:
            parts.append(f"S{i}.{tool}=ERR:{res['error'][:80]}")
        else:
            parts.append(f"S{i}.{tool}=ok")
    return " | ".join(parts)


# =====================================================================
# T01 - trivial
# =====================================================================


def test_t01_trivial_write_done(tmp_path):
    """Warmup: write a tiny file and call done."""
    target = tmp_path / "t01.txt"
    recipe = (
        f'STEP 1: write_file path="{target.as_posix()}" content="hello"\n'
        f'STEP 2: done summary="t01 ok"'
    )
    out = _run(recipe, "SEN-T01")
    assert out["completed_via_done"], _summary(out)
    assert out["steps"] == 2, _summary(out)
    assert target.read_text() == "hello", _summary(out)


# =====================================================================
# T02 - multi-step bash chain
# =====================================================================


def test_t02_bash_chain(tmp_path):
    """Three run_bash steps: create file, append, type back.

    Path is unquoted in the shell command -- the outer recipe wrapper
    is `command="..."` and an inner `"` would terminate the recipe
    arg early. Tmp paths don't have spaces so unquoted is safe.
    Real /gwen recipes against paths with spaces should use the
    escape form: command="cmd \\"path with space\\""
    """
    target = tmp_path / "t02.txt"
    p = target.as_posix()
    cmd_create = f'echo first > {p}'
    cmd_append = f'echo second >> {p}'
    # Cross-platform read via python -c -- avoids cmd.exe `type` syntax
    # error on forward-slash paths and bash `cat` differences.
    cmd_read = f"python -c \\\"print(open(r'{p}').read())\\\""
    recipe = (
        f'STEP 1: run_bash command="{cmd_create}"\n'
        f'STEP 2: run_bash command="{cmd_append}"\n'
        f'STEP 3: run_bash command="{cmd_read}"\n'
        f'STEP 4: done summary="t02 chain ok"'
    )
    out = _run(recipe, "SEN-T02")
    assert out["completed_via_done"], _summary(out)
    body = target.read_text()
    assert "first" in body and "second" in body, _summary(out)
    # Step 3 read the file back via python -c; stdout should match.
    s3 = out["session"][2]["result"]
    assert s3.get("return_code") == 0, _summary(out) + f" stderr={s3.get('stderr')!r}"
    assert "first" in s3.get("stdout", "") and "second" in s3.get("stdout", ""), (
        _summary(out) + " | stdout=" + s3.get("stdout", "")[:200]
    )


# =====================================================================
# T03 - single-line collapsed paste (LIVE BUG REPRO)
# =====================================================================


def test_t03_single_line_collapsed_paste(tmp_path):
    """Some Telegram clients collapse newlines on paste. The strict
    parser sees one giant step; relaxed-fallback should recover.

    This is the reproducer for the 2026-05-07 09:47Z dashboard
    failure (trace SEN-14231d4b)."""
    target = tmp_path / "t03.txt"
    p = target.as_posix()
    # Note: NO newlines between STEPs -- they're space-separated
    recipe = (
        f'STEP 1: write_file path="{p}" content="line-1" '
        f'STEP 2: write_file path="{p}" content="line-2-overwrite" '
        f'STEP 3: done summary="t03 collapsed paste ok"'
    )
    out = _run(recipe, "SEN-T03")
    assert out["steps"] == 3, _summary(out)
    assert out["completed_via_done"], _summary(out)
    assert target.read_text() == "line-2-overwrite", _summary(out)


# =====================================================================
# T04 - Unicode smart quotes
# =====================================================================


def test_t04_smart_quotes(tmp_path):
    """Telegram on some platforms converts " to curly quotes. The
    Phase 18b parser accepts both straight and curly."""
    target = tmp_path / "t04.txt"
    p = target.as_posix()
    # Use Unicode curly quotes directly
    recipe = (
        f"STEP 1: write_file path=“{p}” content=“smart-quoted”\n"
        f"STEP 2: done summary=“t04 smart quotes ok”"
    )
    out = _run(recipe, "SEN-T04")
    assert out["completed_via_done"], _summary(out)
    assert target.read_text() == "smart-quoted", _summary(out)


# =====================================================================
# T05 - long content with many \n escapes (real Python script)
# =====================================================================


def test_t05_long_content_real_script(tmp_path):
    """Write a real-ish Python module via write_file. ~600 chars of
    content with many \\n escapes. Then run it and verify output."""
    script = tmp_path / "t05_calc.py"
    p = script.as_posix()
    # Module-level only (no indented blocks) -- safest for paste-wrap
    content = (
        "import sys\\n"
        "if len(sys.argv) < 2: print('usage: t05_calc.py <expr>'); sys.exit(0)\\n"
        "expr = sys.argv[1]\\n"
        "allowed = set('0123456789+-*/(). ')\\n"
        "if not all(c in allowed for c in expr): print('bad chars'); sys.exit(1)\\n"
        "result = eval(expr, {'__builtins__': {}}, {})\\n"
        "print(result)\\n"
    )
    recipe = (
        f'STEP 1: write_file path="{p}" content="{content}"\n'
        f'STEP 2: run_bash command="python {p} 2+3*4"\n'
        f'STEP 3: done summary="t05 calc script ok"'
    )
    out = _run(recipe, "SEN-T05")
    assert out["completed_via_done"], _summary(out)
    assert script.exists(), _summary(out)
    s2 = out["session"][1]["result"]
    assert "14" in s2.get("stdout", ""), (
        _summary(out) + " stdout=" + s2.get("stdout", "")[:200]
        + " stderr=" + s2.get("stderr", "")[:200]
    )


# =====================================================================
# T06 - edit_file with multi-line anchor
# =====================================================================


def test_t06_edit_file_unique_anchor(tmp_path):
    """edit_file with a 3-line anchor that must match byte-for-byte."""
    target = tmp_path / "t06.py"
    target.write_text(
        "def foo():\n    return 1\n\n"
        "def bar():\n    return 2\n\n"
        "def baz():\n    return 3\n"
    )
    # Anchor: foo's body lines (multi-line via \n escape)
    recipe = (
        f'STEP 1: edit_file path="{target.as_posix()}" '
        f'old="def foo():\\n    return 1" '
        f'new="def foo():\\n    return 100"\n'
        f'STEP 2: run_bash command="python -c \\"import importlib.util as u; '
        f's=u.spec_from_file_location(chr(116),\'{target.as_posix()}\'); '
        f'm=u.module_from_spec(s); s.loader.exec_module(m); print(m.foo())\\""\n'
        f'STEP 3: done summary="t06 edit ok"'
    )
    out = _run(recipe, "SEN-T06")
    assert out["completed_via_done"], _summary(out)
    s2 = out["session"][1]["result"]
    assert "100" in s2.get("stdout", ""), (
        _summary(out) + " stdout=" + s2.get("stdout", "")[:200]
        + " stderr=" + s2.get("stderr", "")[:200]
    )


# =====================================================================
# T07 - multi-file mini app
# =====================================================================


def test_t07_multifile_mini_app(tmp_path):
    """Build a 2-file Python app and run an integration test:
       greet.py -> greet(name)
       run.py   -> imports greet, prints greet('world')
    """
    appdir = tmp_path / "t07app"
    appdir.mkdir()
    greet = (appdir / "greet.py").as_posix()
    run = (appdir / "run.py").as_posix()
    recipe = (
        f'STEP 1: write_file path="{greet}" '
        f'content="def greet(name):\\n    return f\\"hello, {{name}}!\\"\\n"\n'
        f'STEP 2: write_file path="{run}" '
        f'content="import sys; sys.path.insert(0, r\'{appdir.as_posix()}\')\\n'
        f'from greet import greet\\nprint(greet(\'world\'))\\n"\n'
        f'STEP 3: run_bash command="python {run}"\n'
        f'STEP 4: done summary="t07 multi-file app ok"'
    )
    out = _run(recipe, "SEN-T07")
    assert out["completed_via_done"], _summary(out)
    s3 = out["session"][2]["result"]
    assert "hello, world!" in s3.get("stdout", ""), (
        _summary(out) + " stdout=" + s3.get("stdout", "")[:200]
        + " stderr=" + s3.get("stderr", "")[:200]
    )


# =====================================================================
# T08 - shell pipe + redirect
# =====================================================================


def test_t08_shell_pipe_redirect(tmp_path):
    """Test that pipes and redirects survive parser+shell. Path
    unquoted (see T02 docstring for rationale)."""
    out_file = (tmp_path / "t08.txt").as_posix()
    if sys.platform == "win32":
        cmd = f'echo alpha beta gamma | findstr beta > {out_file}'
    else:
        cmd = f'echo alpha beta gamma | grep beta > {out_file}'
    recipe = (
        f'STEP 1: run_bash command="{cmd}"\n'
        f'STEP 2: done summary="t08 pipe redirect ok"'
    )
    out = _run(recipe, "SEN-T08")
    assert out["completed_via_done"], _summary(out)
    body = Path(out_file).read_text(encoding="utf-8", errors="replace")
    assert "beta" in body, _summary(out) + f" body={body!r}"


# =====================================================================
# T09 - SQLite-backed CLI tool
# =====================================================================


def test_t09_sqlite_cli(tmp_path):
    """Write a small SQLite-backed bookmark CLI, then exercise:
       - add a row
       - add another
       - list (verify both present)
       - find (verify search works)
    """
    appdir = tmp_path / "t09app"
    appdir.mkdir()
    db = (appdir / "bm.db").as_posix()
    cli = (appdir / "bm.py").as_posix()
    # Module-level Python (no def blocks needed -- top-level branch on argv)
    content = (
        "import sqlite3, sys\\n"
        f"DB = r'{db}'\\n"
        "c = sqlite3.connect(DB)\\n"
        "c.execute('CREATE TABLE IF NOT EXISTS b(id INTEGER PRIMARY KEY, url TEXT, tag TEXT)')\\n"
        "c.commit()\\n"
        "if len(sys.argv) < 2: print('usage'); sys.exit(0)\\n"
        "cmd = sys.argv[1]\\n"
        "if cmd == 'add': c.execute('INSERT INTO b(url, tag) VALUES (?, ?)', (sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')); c.commit(); print('saved')\\n"
        "elif cmd == 'list':\\n"
        " [print(r[0], r[1], r[2]) for r in c.execute('SELECT id, url, tag FROM b ORDER BY id')]\\n"
        "elif cmd == 'find':\\n"
        " q = sys.argv[2]; [print(r[0], r[1], r[2]) for r in c.execute('SELECT id, url, tag FROM b WHERE url LIKE ? OR tag LIKE ?', (f'%{q}%', f'%{q}%'))]\\n"
    )
    recipe = (
        f'STEP 1: write_file path="{cli}" content="{content}"\n'
        f'STEP 2: run_bash command="python {cli} add https://example.com demo"\n'
        f'STEP 3: run_bash command="python {cli} add https://github.com code"\n'
        f'STEP 4: run_bash command="python {cli} list"\n'
        f'STEP 5: run_bash command="python {cli} find example"\n'
        f'STEP 6: done summary="t09 sqlite CLI ok"'
    )
    out = _run(recipe, "SEN-T09")
    assert out["completed_via_done"], _summary(out)
    s4 = out["session"][3]["result"]  # list
    assert "example.com" in s4.get("stdout", ""), (
        _summary(out) + " s4stdout=" + s4.get("stdout", "")[:300]
        + " s4stderr=" + s4.get("stderr", "")[:300]
    )
    assert "github.com" in s4.get("stdout", ""), _summary(out)
    s5 = out["session"][4]["result"]  # find
    assert "example.com" in s5.get("stdout", ""), _summary(out)
    assert "github.com" not in s5.get("stdout", ""), _summary(out)
    # Verify DB persisted
    rows = list(sqlite3.connect(db).execute("SELECT url FROM b"))
    assert len(rows) == 2, _summary(out)


# =====================================================================
# T10 - daily-ops dashboard
# =====================================================================


def test_t10_dashboard(tmp_path):
    """The ambitious build: read disk + git + write HTML report.

    Adapted from the user's failed 9:47Z attempt: pulls disk usage
    and a synthetic 'KB' (a tmp SQLite we create here) then renders
    HTML to the dashboard dir."""
    dash = tmp_path / "dashboard"
    dash.mkdir()
    # Seed a fake KB to query
    fake_kb = (tmp_path / "kb.db").as_posix()
    c = sqlite3.connect(fake_kb)
    c.execute("CREATE TABLE knowledge(id INTEGER PRIMARY KEY, pinned INTEGER, state TEXT)")
    c.execute("INSERT INTO knowledge(pinned, state) VALUES (1, 'active'), (0, 'active'), (0, 'archived')")
    c.commit()
    c.close()

    build_py = (dash / "build.py").as_posix()
    index_html = (dash / "index.html").as_posix()

    # Module-level only; HTML built via concatenation (no f-string with
    # CSS braces, no nested f-strings) to keep the source dead-simple.
    content = (
        "import shutil, sqlite3\\n"
        f"KB = r'{fake_kb}'\\n"
        "t, u, f = shutil.disk_usage('/')\\n"
        "c = sqlite3.connect(KB)\\n"
        "row = c.execute('SELECT COUNT(*), SUM(CASE WHEN pinned=1 THEN 1 ELSE 0 END), SUM(CASE WHEN state=\\\"archived\\\" THEN 1 ELSE 0 END) FROM knowledge').fetchone()\\n"
        "html_parts = ['<html><body>', '<h1>Sentinel Daily Ops</h1>', f'<p>Disk: {f//1000000} MB free of {t//1000000} MB</p>', f'<p>KB: total={row[0]} pinned={row[1]} archived={row[2]}</p>', '</body></html>']\\n"
        f"open(r'{index_html}', 'w', encoding='utf-8').write(''.join(html_parts))\\n"
        f"print('wrote', r'{index_html}')\\n"
    )
    recipe = (
        f'STEP 1: write_file path="{build_py}" content="{content}"\n'
        f'STEP 2: run_bash command="python {build_py}"\n'
        f'STEP 3: read_file path="{index_html}"\n'
        f'STEP 4: done summary="t10 dashboard ok"'
    )
    out = _run(recipe, "SEN-T10")
    assert out["completed_via_done"], _summary(out)
    body = Path(index_html).read_text(encoding="utf-8")
    assert "<h1>Sentinel Daily Ops</h1>" in body, _summary(out) + f" body={body!r}"
    assert "total=3" in body, _summary(out) + f" body={body!r}"
    assert "pinned=1" in body, _summary(out) + f" body={body!r}"
    assert "archived=1" in body, _summary(out) + f" body={body!r}"
    s3 = out["session"][2]["result"]
    assert s3.get("ok") is True, _summary(out)


# =====================================================================
# T11 - production failure mode: single-line + unquoted values
# =====================================================================


def test_t12_unquoted_run_bash_permissive_recovery(tmp_path):
    """Phase 18c: when Telegram strips quotes from a recipe, single-arg
    tools (run_bash, read_file, list_dir, done) recover via permissive
    parse: take everything after `<arg>=` to end-of-step as the value."""
    out_file = (tmp_path / "t12.txt").as_posix()
    cmd_create = f'echo permissive-ok > {out_file}'
    # NO quotes around the command value -- simulates Telegram strip
    recipe = (
        f'STEP 1: run_bash command={cmd_create}\n'
        f'STEP 2: done summary=t12 unquoted ok'
    )
    out = _run(recipe, "SEN-T12")
    assert out["completed_via_done"], _summary(out)
    body = Path(out_file).read_text(encoding="utf-8")
    assert "permissive-ok" in body, _summary(out) + f" body={body!r}"
    s1 = out["session"][0]["result"]
    assert s1.get("ok") is True, _summary(out)


def test_t13_unquoted_done_summary_permissive(tmp_path):
    """done() with unquoted summary recovers via permissive parse."""
    recipe = "STEP 1: done summary=this is an unquoted summary"
    out = _run(recipe, "SEN-T13")
    assert out["completed_via_done"], _summary(out)
    assert out["summary"] == "this is an unquoted summary", _summary(out)


def test_t14_unquoted_read_file_permissive(tmp_path):
    """read_file with unquoted path recovers."""
    target = tmp_path / "t14.txt"
    target.write_text("readback")
    recipe = (
        f"STEP 1: read_file path={target.as_posix()}\n"
        f"STEP 2: done summary=read t14"
    )
    out = _run(recipe, "SEN-T14")
    assert out["completed_via_done"], _summary(out)
    s1 = out["session"][0]["result"]
    assert s1.get("ok") is True, _summary(out)
    assert "readback" in s1.get("content", ""), _summary(out)


def test_t15_unquoted_write_file_safely_rejects(tmp_path):
    """Multi-arg tools (write_file) MUST NOT use permissive parse --
    `\\w+=` patterns inside `content=` would be misparsed as kwargs.
    The parser correctly returns 'did not parse' rather than producing
    a broken kwarg dispatch."""
    recipe = (
        f"STEP 1: write_file path={tmp_path.as_posix()}/t15.txt content=hello\n"
        f"STEP 2: done summary=t15"
    )
    out = _run(recipe, "SEN-T15")
    # done still fires
    assert out["completed_via_done"], _summary(out)
    # STEP 1 surfaces a clear parse error (NOT a silent bad-args dispatch)
    s1 = out["session"][0]["result"]
    assert "error" in s1, _summary(out)
    assert "did not parse" in s1["error"], _summary(out)


def test_t11_production_failure_input_partial_recovery(tmp_path):
    """The exact malformed input the bot received at 2026-05-07 09:47Z
    (trace SEN-14231d4b): multi-step recipe where Telegram (or user
    paste) collapsed all newlines into spaces AND stripped every
    quote character.

    With the Phase 18b strict->relaxed STEP boundary fallback, the
    parser now correctly identifies all 5 STEP markers (was: 1).
    Individual steps still fail to parse arg key=values (no quotes
    means _KV_RE matches nothing), but each step's failure surfaces
    as a clear per-step error rather than the whole recipe being one
    giant unparseable blob -- meaningful progress for diagnostics.

    The remaining gap (no-quote values) is recoverable by the user
    via /restart of the bot to pick up Phase 18 handler/router fixes
    that preserve raw newlines AND quotes from update.message.text.
    """
    # Verbatim production-input shape (from gwen_assist dispatching log)
    recipe = (
        "STEP 1: run_bash command=mkdir C:/Users/test/Desktop/dashboard 2>nul "
        "STEP 2: write_file path=C:/Users/test/Desktop/dashboard/build.py content=import json "
        "STEP 3: run_bash command=python C:/Users/test/Desktop/dashboard/build.py "
        "STEP 4: run_bash command=start C:/Users/test/Desktop/dashboard/index.html "
        "STEP 5: done summary=built dashboard"
    )
    out = _run(recipe, "SEN-T11")
    # CRITICAL: relaxed-fallback recovers all 5 steps (was 1 in pre-Phase-18b)
    assert out["steps"] == 5, _summary(out)
    # With Phase 18c permissive-fallback for single-arg tools:
    # STEPs 1, 3, 4 (run_bash) and STEP 5 (done) should now have args
    # populated. Only STEP 2 (write_file, multi-arg) still fails.
    parse_errors = [
        s for s in out["session"]
        if "did not parse" in str(s.get("result", {}).get("error", ""))
    ]
    assert len(parse_errors) == 1, (
        _summary(out) + f" expected only STEP 2 to fail; got {len(parse_errors)}"
    )
    # And the failing step is STEP 2 (write_file)
    assert parse_errors[0]["step"] == 2, _summary(out)
    # done() succeeded
    assert out["completed_via_done"] is True, _summary(out)
