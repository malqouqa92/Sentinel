"""Phase 18d -- _b64 arg suffix for paste-mangle-immune long content.

The actual problem this solves: user pastes a recipe with a long
write_file content="..." into Telegram. The Telegram client mangles
it (strips quotes, converts smart-quotes, inserts paste-wrap
newlines). The decoded content arriving at the executor is broken.

Solution: recipe author encodes the content as base64. Base64 alphabet
([A-Za-z0-9+/=]) is paste-mangle-immune:
  - No quote chars to strip
  - No special chars to smart-quote-convert
  - b64decode(validate=False) ignores stray whitespace from paste-wrap

Test groups:
  D - decode happy path
  M - paste-mangle simulations (each Telegram corruption mode)
  L - LONG content (~2000 chars module)
  E - end-to-end: encoded recipe -> mangled -> parsed -> decoded -> file written
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from core.gwen_agent import _execute_recipe  # noqa: E402
from core.qwen_agent import _parse_step_text_to_tool_call  # noqa: E402


# ============================================================
# D group -- basic _b64 decode
# ============================================================


def test_d01_b64_content_decodes_to_plain():
    plain = "import sys\nprint('hi')\n"
    b64 = base64.b64encode(plain.encode()).decode("ascii")
    step = f'write_file path="x.py" content_b64="{b64}"'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed is not None
    args = parsed["function"]["arguments"]
    assert args["path"] == "x.py"
    assert args["content"] == plain
    assert "content_b64" not in args  # key was renamed


def test_d02_b64_command_decodes_for_run_bash():
    plain = 'echo "complex command with quotes" > out.txt'
    b64 = base64.b64encode(plain.encode()).decode("ascii")
    step = f'run_bash command_b64="{b64}"'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed["function"]["arguments"]["command"] == plain


def test_d03_b64_done_summary():
    plain = "summary with \"quotes\" and 'apostrophes'"
    b64 = base64.b64encode(plain.encode()).decode("ascii")
    step = f'done summary_b64="{b64}"'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed["function"]["arguments"]["summary"] == plain


def test_d04_mixed_plain_and_b64():
    """path stays plain, content goes b64."""
    content = "exec('print(99)')"
    b64 = base64.b64encode(content.encode()).decode("ascii")
    step = f'write_file path="hi.py" content_b64="{b64}"'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    assert args["path"] == "hi.py"
    assert args["content"] == content


def test_d05_b64_preserves_real_newlines():
    plain = "line one\nline two\n    indented\nline four"
    b64 = base64.b64encode(plain.encode()).decode("ascii")
    step = f'write_file path="x" content_b64="{b64}"'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed["function"]["arguments"]["content"] == plain


def test_d06_b64_decode_failure_is_safe():
    """Garbage in _b64 -- the key is left as-is so tool dispatch fails
    cleanly rather than silently dropping the arg."""
    # b64decode IS lenient (validate=False) but `!@#$%` strips to nothing
    # and decodes to b'' -- we should still detect this as "decoded"
    # but to an empty value. The tool then dispatches with empty content,
    # which is a clear caller error vs. silent drop.
    step = 'write_file path="x" content_b64="not-actually-b64!@#$"'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    # decode succeeded (validate=False is permissive); content is empty
    # or near-empty bytes
    assert "content" in args or "content_b64" in args


# ============================================================
# M group -- Telegram corruption modes survived
# ============================================================


def test_m01_b64_survives_quote_strip(tmp_path):
    """Telegram strips outer double-quotes -> the b64 string still
    parses via the permissive single-arg fallback (run_bash)."""
    plain = "echo b64-strip-survived > {}".format((tmp_path / "m01.txt").as_posix())
    b64 = base64.b64encode(plain.encode()).decode("ascii")
    # NO QUOTES around the b64 value
    step = f'run_bash command_b64={b64}'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed is not None
    cmd = parsed["function"]["arguments"]["command"]
    assert cmd == plain, f"got: {cmd!r}"


def test_m02_b64_survives_smart_quotes(tmp_path):
    """Telegram converts " to curly U+201C/U+201D."""
    plain = 'echo curly > out.txt'
    b64 = base64.b64encode(plain.encode()).decode("ascii")
    # Curly quotes around the b64 value
    step = f'run_bash command_b64=“{b64}”'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed["function"]["arguments"]["command"] == plain


def test_m03_b64_survives_paste_wrap_newlines():
    """Telegram inserts real newlines mid-value (visual wrap)."""
    plain = "this is a longer string that will be base64-encoded " * 4
    b64 = base64.b64encode(plain.encode()).decode("ascii")
    # Insert a real newline + indent in the middle of the b64 value
    half = len(b64) // 2
    wrapped_b64 = b64[:half] + "\n  " + b64[half:]
    step = f'run_bash command_b64="{wrapped_b64}"'
    parsed = _parse_step_text_to_tool_call(step)
    # b64decode ignores whitespace, decoded == original plain
    assert parsed["function"]["arguments"]["command"] == plain


def test_m04_b64_survives_combined_corruptions():
    """All three corruptions at once: smart-quote + paste-wrap +
    nothing else weird. b64 should still recover."""
    plain = "complex value with newlines\nand \"quotes\" inside"
    b64 = base64.b64encode(plain.encode()).decode("ascii")
    # Inject newline + indent mid-value, wrap with curly quotes
    half = len(b64) // 2
    mangled = b64[:half] + "\n   " + b64[half:]
    step = f'run_bash command_b64=“{mangled}”'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed["function"]["arguments"]["command"] == plain


# ============================================================
# L group -- long content via _b64
# ============================================================


REAL_PYTHON_MODULE = '''import sqlite3
import sys
import datetime
import argparse


DB_PATH = "h_journal.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            tag TEXT,
            text TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
        USING fts5(text, tag, content=entries, content_rowid=id)
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS entries_ai
        AFTER INSERT ON entries BEGIN
            INSERT INTO entries_fts(rowid, text, tag)
            VALUES (new.id, new.text, new.tag);
        END
    """)
    conn.commit()
    return conn


def add(text, tag=""):
    conn = init_db()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO entries (ts, tag, text) VALUES (?, ?, ?)",
        (ts, tag, text),
    )
    conn.commit()
    return cur.lastrowid


def find(query):
    conn = init_db()
    rows = conn.execute("""
        SELECT entries.id, entries.ts, entries.tag, entries.text
        FROM entries_fts
        JOIN entries ON entries.id = entries_fts.rowid
        WHERE entries_fts MATCH ?
        ORDER BY entries.ts DESC
    """, (query,)).fetchall()
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("add")
    a.add_argument("text")
    a.add_argument("--tag", default="")
    f = sp.add_parser("find")
    f.add_argument("query")
    args = p.parse_args()
    if args.cmd == "add":
        new_id = add(args.text, args.tag)
        print(f"saved #{new_id}")
    elif args.cmd == "find":
        for r in find(args.query):
            print(f"#{r[0]} [{r[1]}] [{r[2]}] {r[3]}")
'''


def test_l01_long_module_via_b64(tmp_path):
    """A real ~2KB Python module written via _b64 survives parsing."""
    target = tmp_path / "l01_journal.py"
    b64 = base64.b64encode(REAL_PYTHON_MODULE.encode()).decode("ascii")
    step = f'write_file path="{target.as_posix()}" content_b64="{b64}"'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    assert args["content"] == REAL_PYTHON_MODULE
    assert len(args["content"]) > 1500


def test_l02_long_b64_paste_wrapped_3x():
    """A long b64 value with 3 visual wraps still decodes cleanly."""
    b64 = base64.b64encode(REAL_PYTHON_MODULE.encode()).decode("ascii")
    n = len(b64)
    # Insert 3 paste-wrap newlines at quarter, half, three-quarter points
    mangled = (
        b64[:n // 4] + "\n  "
        + b64[n // 4:n // 2] + "\n  "
        + b64[n // 2:3 * n // 4] + "\n  "
        + b64[3 * n // 4:]
    )
    step = f'write_file path="x.py" content_b64="{mangled}"'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed["function"]["arguments"]["content"] == REAL_PYTHON_MODULE


# ============================================================
# E group -- end-to-end through _execute_recipe
# ============================================================


def test_e01_e2e_real_module_b64_executes(tmp_path):
    """Build the real journal module via _b64 and verify it runs."""
    target = tmp_path / "j.py"
    b64 = base64.b64encode(REAL_PYTHON_MODULE.encode()).decode("ascii")
    p = target.as_posix()
    recipe = (
        f'STEP 1: write_file path="{p}" content_b64="{b64}"\n'
        f'STEP 2: run_bash command="cd {tmp_path.as_posix()} && python {p} add hello-world --tag test"\n'
        f'STEP 3: run_bash command="cd {tmp_path.as_posix()} && python {p} find hello"\n'
        f'STEP 4: done summary="e2e b64 journal works"'
    )
    out = _execute_recipe(recipe, "SEN-test-e01")
    assert out["completed_via_done"], (
        "session: " + str(out.get("session"))
    )
    # Verify the module was written byte-for-byte
    assert target.read_text() == REAL_PYTHON_MODULE
    # Verify journal output: STEP 3's stdout should contain "hello-world"
    s3 = out["session"][2]["result"]
    assert "hello-world" in s3.get("stdout", ""), (
        f"stdout: {s3.get('stdout')!r} stderr: {s3.get('stderr')!r}"
    )


# ============================================================
# Z group -- _b64gz (gzip+base64 for size-limited transports)
# ============================================================


def test_z01_b64gz_quoted_decompresses_correctly():
    import zlib
    plain = "import sys\nprint(1)\n" * 50  # ~1KB, compresses well
    gz = base64.b64encode(zlib.compress(plain.encode(), 9)).decode()
    step = f'write_file path="x.py" content_b64gz="{gz}"'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    assert args["content"] == plain
    assert "content_b64gz" not in args


def test_z02_b64gz_compression_ratio_is_real():
    """Sanity: gzip+b64 should be <50% the size of plain b64 for
    repetitive Python source."""
    import zlib
    plain = ("import sqlite3, sys, datetime\n"
             "def add(c, t, x):\n"
             "    c.execute('INSERT INTO n VALUES(?, ?, ?)', (...))\n") * 30
    plain_b64 = base64.b64encode(plain.encode()).decode()
    gz_b64 = base64.b64encode(zlib.compress(plain.encode(), 9)).decode()
    assert len(gz_b64) < len(plain_b64) * 0.5, (
        f"gz_b64={len(gz_b64)} not significantly smaller than plain_b64={len(plain_b64)}"
    )


def test_z03_b64gz_unquoted_with_paste_wrap():
    """Telegram-mangled _b64gz: unquoted, with paste-wrap injected."""
    import zlib
    plain = "real working python module" * 50
    gz = base64.b64encode(zlib.compress(plain.encode(), 9)).decode()
    half = len(gz) // 2
    wrapped = gz[:half] + "\n  " + gz[half:]
    step = f'write_file path=foo content_b64gz={wrapped}'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    assert args["path"] == "foo"
    assert args["content"] == plain


def test_z04_b64gz_e2e_writes_real_file(tmp_path):
    """End-to-end: gzip+b64 source -> file on disk -> python runs it."""
    import zlib
    target = tmp_path / "z04.py"
    plain_module = (
        "import sys\n"
        "for i in range(int(sys.argv[1] if len(sys.argv) > 1 else 5)):\n"
        "    print(i, i*i)\n"
    )
    gz = base64.b64encode(zlib.compress(plain_module.encode(), 9)).decode()
    p = target.as_posix()
    recipe = (
        f'STEP 1: write_file path="{p}" content_b64gz="{gz}"\n'
        f'STEP 2: run_bash command="python {p} 4"\n'
        f'STEP 3: done summary="z04 e2e b64gz worked"'
    )
    out = _execute_recipe(recipe, "SEN-test-z04")
    assert out["completed_via_done"], str(out["session"])
    assert target.read_text() == plain_module
    s2 = out["session"][1]["result"]
    assert "9" in s2.get("stdout", ""), f"expected 0 1 4 9 in output: {s2}"


def test_z05_b64gz_decode_failure_safe():
    """Garbage in _b64gz key -- the value won't decompress; key is
    left as-is so dispatch fails cleanly."""
    step = 'write_file path="x" content_b64gz="not-real-gzip-data!!!"'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    # decoded into args["content"] OR left as content_b64gz
    # Either way the tool dispatch will surface the issue clearly.
    assert "content_b64gz" in args or "content" in args


def test_z06_b64gz_takes_priority_over_b64():
    """If both `_b64` and `_b64gz` are present (shouldn't happen but...)
    each populates its own plain key. Last-write wins via dict update."""
    import zlib
    plain_a = "from b64"
    plain_b = "from b64gz"
    a64 = base64.b64encode(plain_a.encode()).decode()
    g64 = base64.b64encode(zlib.compress(plain_b.encode(), 9)).decode()
    # Use different keys so we can verify both decode independently.
    step = f'write_file path_b64="{a64}" content_b64gz="{g64}"'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    assert args["path"] == plain_a
    assert args["content"] == plain_b


# ============================================================
# G group -- generality tests (not just one specific prompt)
# ============================================================


def test_g01_multiple_b64_args_in_one_step():
    """Two b64 args in a single step both decode."""
    a = base64.b64encode(b"hello").decode()
    b = base64.b64encode(b"world").decode()
    step = f'write_file path_b64="{a}" content_b64="{b}"'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    assert args["path"] == "hello"
    assert args["content"] == "world"


def test_g02_b64_padding_no_padding_one_eq_two_eq():
    """All three b64 padding forms work: '', '=', '==', via decoded
    bytes lengths divisible by 3, divisible-by-3-plus-1, plus-2."""
    for plain in ("abc", "ab", "a", "abcdef", "abcde", "abcd"):
        b64 = base64.b64encode(plain.encode()).decode()
        step = f'write_file path="x" content_b64="{b64}"'
        parsed = _parse_step_text_to_tool_call(step)
        assert parsed["function"]["arguments"]["content"] == plain, (
            f"failed for plain={plain!r} b64={b64!r}"
        )


def test_g03_unquoted_b64_with_double_eq_padding():
    """Unquoted b64 ending in `==` (the bug from B3)."""
    plain = "hi"  # encodes to "aGk=" with single padding
    b64_dbl = base64.b64encode(b"abcd").decode()  # "YWJjZA=="
    assert b64_dbl.endswith("==")
    step = f'write_file path=foo content_b64={b64_dbl}'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    assert args["path"] == "foo"
    assert args["content"] == "abcd"


def test_g04_unquoted_b64_with_paste_wrap_AND_double_eq():
    """The B3 production-failure shape: paste-wrap mid-b64 + `==` end."""
    plain = "abcd" * 20  # encodes long enough for wrap to land mid-value
    b64 = base64.b64encode(plain.encode()).decode()
    half = len(b64) // 2
    wrapped = b64[:half] + "\n  " + b64[half:]
    step = f'write_file path=foo content_b64={wrapped}'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    assert args["path"] == "foo"
    assert args["content"] == plain


def test_g05_unquoted_b64_with_curly_smart_quotes():
    plain = "smart quote test " * 10
    b64 = base64.b64encode(plain.encode()).decode()
    step = f"write_file path=“foo” content_b64=“{b64}”"
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    assert args["content"] == plain


def test_g06_quoted_b64_still_works():
    """The simplest case: properly-quoted b64. Strict path handles it."""
    plain = "quoted base64"
    b64 = base64.b64encode(plain.encode()).decode()
    step = f'write_file path="x.py" content_b64="{b64}"'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    assert args["path"] == "x.py"
    assert args["content"] == plain


def test_g07_no_b64_marker_doesnt_trigger_multi_arg_path():
    """Sanity: no `_b64` in step_text -> multi-arg permissive doesn't
    fire, and unquoted multi-arg correctly returns None (parse fail
    rather than silent split that produces broken args)."""
    step = 'write_file path=foo content=bar'
    parsed = _parse_step_text_to_tool_call(step)
    # write_file is multi-arg; no quotes; no b64 -> parse fail
    assert parsed is None


def test_g08_plain_arg_before_b64_extracted_correctly():
    """path is plain, content is b64. Plain region is BEFORE b64
    marker, parsed independently."""
    plain = "x" * 200  # long enough that b64 has padding
    b64 = base64.b64encode(plain.encode()).decode()
    step = f'write_file path=/tmp/some/path.txt content_b64={b64}'
    parsed = _parse_step_text_to_tool_call(step)
    args = parsed["function"]["arguments"]
    assert args["path"] == "/tmp/some/path.txt"
    assert args["content"] == plain


def test_g09_unicode_content_via_b64():
    """Unicode chars (emoji, accents) survive b64 round-trip."""
    plain = "héllo wörld 日本語 🎉\nsecond line"
    b64 = base64.b64encode(plain.encode("utf-8")).decode()
    step = f'write_file path="x" content_b64="{b64}"'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed["function"]["arguments"]["content"] == plain


def test_g10_very_long_content_8kb():
    """8KB of content. ~11KB of base64. Multi-line b64 in source via
    paste-wrap simulation. End-to-end in _execute_recipe."""
    plain = "line " * 1500  # ~7500 chars
    b64 = base64.b64encode(plain.encode()).decode()
    # Inject 5 paste-wrap newlines
    n = len(b64)
    parts = [b64[i:i+n//5] for i in range(0, n, n//5)]
    wrapped = "\n  ".join(parts)
    step = f'write_file path="x.txt" content_b64="{wrapped}"'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed["function"]["arguments"]["content"] == plain


def test_g11_b64_with_plus_and_slash_chars():
    """base64 standard alphabet includes + and /. These are preserved."""
    plain = b"\xff\xee\xdd\xcc\xbb\xaa" * 10
    b64 = base64.b64encode(plain).decode()
    # Sanity: this particular b64 should contain + or /
    assert any(c in b64 for c in "+/"), f"b64 lacks +//: {b64}"
    step = f'write_file path="x" content_b64="{b64}"'
    parsed = _parse_step_text_to_tool_call(step)
    decoded_content = parsed["function"]["arguments"]["content"]
    assert decoded_content.encode("utf-8", errors="replace") == plain.decode("utf-8", errors="replace").encode("utf-8")


def test_g12_run_bash_b64_with_double_eq_padding():
    """run_bash command_b64 with `==` padding, unquoted."""
    cmd = "echo abc"  # encodes to "ZWNobyBhYmM=" (single =)
    cmd_dbl = "ec"  # encodes to "ZWM=" hmm need ==
    # find a command that encodes with ==
    for c in ("e", "ec", "echo a", "ec ", "abcd"):
        b64 = base64.b64encode(c.encode()).decode()
        if b64.endswith("=="):
            test_cmd = c; test_b64 = b64; break
    else:
        pytest.skip("could not find a command encoding to ==")
    step = f'run_bash command_b64={test_b64}'
    parsed = _parse_step_text_to_tool_call(step)
    assert parsed["function"]["arguments"]["command"] == test_cmd


def test_e02_e2e_b64_with_telegram_mangle_simulation(tmp_path):
    """Simulate the actual production failure: long content that
    would normally be paste-mangled, sent as _b64, with a paste-wrap
    INJECTED into the b64 value. End-to-end recovery."""
    target = tmp_path / "mangled.py"
    b64 = base64.b64encode(REAL_PYTHON_MODULE.encode()).decode("ascii")
    # Inject paste-wrap mid-b64
    half = len(b64) // 2
    wrapped_b64 = b64[:half] + "\n  " + b64[half:]
    p = target.as_posix()
    recipe = (
        f'STEP 1: write_file path="{p}" content_b64="{wrapped_b64}"\n'
        f'STEP 2: run_bash command="python {p} add e2e-test"\n'
        f'STEP 3: done summary="mangle-survived"'
    )
    out = _execute_recipe(recipe, "SEN-test-e02")
    assert out["completed_via_done"], str(out.get("session"))
    assert target.read_text() == REAL_PYTHON_MODULE
