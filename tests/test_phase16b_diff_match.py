"""Phase 16 Batch B -- diff-match deterministic verification.

The diff-match scorer is the cheap deterministic gate Batch C uses to
decide "did the replayed recipe produce a structurally similar diff
to the stored pattern?". Without it, the only verification is Claude
review, which defeats the speed gain of skipping Claude in the first
place.

Coverage:
  Hunk parsing (boundary cases):
    M01 -- empty diff -> empty hunk set
    M02 -- whitespace-only diff -> empty hunk set
    M03 -- single-file single-hunk diff parses to 1 tuple
    M04 -- multi-file multi-hunk diff parses to N tuples
    M05 -- pure-context hunk (no +/- lines) is dropped
    M06 -- malformed diff (no +++ header) returns empty set safely

  Score correctness:
    M11 -- identical diffs -> score 1.0
    M12 -- completely different diffs -> score 0.0
    M13 -- 50% overlap -> score in (0.0, 1.0)
    M14 -- both empty -> score 0.0 (caller's decision)
    M15 -- one empty, one populated -> score 0.0
    M16 -- whitespace difference in body alone -> still equal (norm)
    M17 -- different file paths, same content -> score 0.0
    M18 -- same file, different content -> score 0.0
    M19 -- same file, same content, line numbers shifted -> 1.0
           (line numbers are NOT in the signature)

  Evaluate (verdict + reason):
    M21 -- empty replay diff -> reject with reason
    M22 -- empty stored diff -> reject with reason
    M23 -- identical diffs -> accept with score 1.0 in reason
    M24 -- score below threshold -> reject with reason
    M25 -- threshold is configurable via parameter
    M26 -- DIFF_MATCH_ACCEPT_THRESHOLD module constant equals 0.7

  Robustness:
    M31 -- never raises on garbage input
    M32 -- handles unicode in diff bodies
    M33 -- handles binary-flagged diffs gracefully (returns 0.0)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.diff_match import (
    DIFF_MATCH_ACCEPT_THRESHOLD,
    DiffMatchResult,
    _parse_hunks,
    evaluate_diff_match,
    score_diff_match,
)


# Convenient sample diff fragments used across tests.
_SIMPLE_DIFF_A = """\
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
 def hello():
-    return "old"
+    return "new"
"""

_SIMPLE_DIFF_B = """\
--- a/bar.py
+++ b/bar.py
@@ -10,2 +10,2 @@
-    x = 1
+    x = 2
"""

_TWO_FILE_DIFF = _SIMPLE_DIFF_A + _SIMPLE_DIFF_B


# ─────────────────────────────────────────────────────────────────
# Hunk parsing
# ─────────────────────────────────────────────────────────────────


def test_m01_empty_diff_empty_set():
    assert _parse_hunks("") == set()


def test_m02_whitespace_only_diff_empty_set():
    assert _parse_hunks("   \n\t\n  ") == set()


def test_m03_single_file_single_hunk():
    s = _parse_hunks(_SIMPLE_DIFF_A)
    assert len(s) == 1
    (path, sig), = s
    assert path == "foo.py"
    assert sig[0] == 1  # 1 added line
    assert sig[1] == 1  # 1 removed line


def test_m04_multi_file_multi_hunk():
    s = _parse_hunks(_TWO_FILE_DIFF)
    assert len(s) == 2
    paths = {tup[0] for tup in s}
    assert paths == {"foo.py", "bar.py"}


def test_m05_pure_context_hunk_dropped():
    """Hunk with only context lines (no +/- markers) shouldn't
    appear in the set."""
    diff = """\
--- a/x.py
+++ b/x.py
@@ -1,3 +1,3 @@
 line1
 line2
 line3
"""
    assert _parse_hunks(diff) == set()


def test_m06_malformed_diff_no_header_safe():
    """No +++ header -> no path captured -> empty set, no crash."""
    diff = "@@ random text @@\nsome stuff\n+more"
    assert _parse_hunks(diff) == set()


# ─────────────────────────────────────────────────────────────────
# Score correctness
# ─────────────────────────────────────────────────────────────────


def test_m11_identical_diffs_score_one():
    assert score_diff_match(_SIMPLE_DIFF_A, _SIMPLE_DIFF_A) == 1.0


def test_m12_completely_different_score_zero():
    assert score_diff_match(_SIMPLE_DIFF_A, _SIMPLE_DIFF_B) == 0.0


def test_m13_partial_overlap_in_range():
    """One file shared between stored and replay -> 1/2 overlap."""
    s = score_diff_match(_TWO_FILE_DIFF, _SIMPLE_DIFF_A)
    assert 0.0 < s < 1.0
    assert s == pytest.approx(0.5)  # 1 shared / 2 total


def test_m14_both_empty_score_zero():
    assert score_diff_match("", "") == 0.0


def test_m15_one_empty_score_zero():
    assert score_diff_match(_SIMPLE_DIFF_A, "") == 0.0
    assert score_diff_match("", _SIMPLE_DIFF_A) == 0.0


def test_m16_trailing_whitespace_normalized():
    """Trailing whitespace difference shouldn't break the match.
    The signature normalizes via rstrip()."""
    a = """\
--- a/x.py
+++ b/x.py
@@ -1 +1 @@
-old
+new
"""
    b = """\
--- a/x.py
+++ b/x.py
@@ -1 +1 @@
-old
+new\t
"""
    assert score_diff_match(a, b) == 1.0


def test_m17_different_paths_same_content_zero():
    """Same +/- content but different file paths -> not a match."""
    a = """\
--- a/x.py
+++ b/x.py
@@ -1 +1 @@
-old
+new
"""
    b = """\
--- a/y.py
+++ b/y.py
@@ -1 +1 @@
-old
+new
"""
    assert score_diff_match(a, b) == 0.0


def test_m18_same_path_different_content_zero():
    a = """\
--- a/x.py
+++ b/x.py
@@ -1 +1 @@
-foo
+bar
"""
    b = """\
--- a/x.py
+++ b/x.py
@@ -1 +1 @@
-baz
+qux
"""
    assert score_diff_match(a, b) == 0.0


def test_m19_line_numbers_drift_still_match():
    """The signature does NOT include line numbers, so a hunk that
    moves to a different position in the file but has the same
    content matches."""
    a = """\
--- a/x.py
+++ b/x.py
@@ -1,3 +1,3 @@
 ctx
-old
+new
"""
    b = """\
--- a/x.py
+++ b/x.py
@@ -50,3 +50,3 @@
 ctx
-old
+new
"""
    assert score_diff_match(a, b) == 1.0


# ─────────────────────────────────────────────────────────────────
# evaluate_diff_match (verdict + reason)
# ─────────────────────────────────────────────────────────────────


def test_m21_empty_replay_rejects_with_reason():
    r = evaluate_diff_match(_SIMPLE_DIFF_A, "")
    assert isinstance(r, DiffMatchResult)
    assert r.accept is False
    assert "empty" in r.reason.lower()


def test_m22_empty_stored_rejects_with_reason():
    r = evaluate_diff_match("", _SIMPLE_DIFF_A)
    assert r.accept is False
    assert "stored" in r.reason.lower() or "no diff" in r.reason.lower()


def test_m23_identical_accepts_with_score():
    r = evaluate_diff_match(_SIMPLE_DIFF_A, _SIMPLE_DIFF_A)
    assert r.accept is True
    assert r.score == 1.0
    assert "1.000" in r.reason or "1.0" in r.reason


def test_m24_below_threshold_rejects():
    """Two completely different diffs -> score 0.0 -> below 0.7 ->
    reject."""
    r = evaluate_diff_match(_SIMPLE_DIFF_A, _SIMPLE_DIFF_B)
    assert r.accept is False
    assert r.score == 0.0


def test_m25_threshold_parameter_overrides():
    """The default threshold rejects 0.5 overlap. Pass a lower
    threshold and the same input passes."""
    # 1 shared / 2 total = 0.5 overlap
    r_default = evaluate_diff_match(_TWO_FILE_DIFF, _SIMPLE_DIFF_A)
    assert r_default.accept is False  # 0.5 < 0.7
    r_loose = evaluate_diff_match(_TWO_FILE_DIFF, _SIMPLE_DIFF_A,
                                  threshold=0.4)
    assert r_loose.accept is True  # 0.5 >= 0.4


def test_m26_module_constant_threshold():
    assert DIFF_MATCH_ACCEPT_THRESHOLD == 0.7


# ─────────────────────────────────────────────────────────────────
# Robustness
# ─────────────────────────────────────────────────────────────────


def test_m31_never_raises_on_garbage():
    """Random non-diff strings must return 0.0, not crash."""
    for garbage in [
        "not a diff at all",
        "@@@ malformed @@@",
        "\x00\x01binary",
        "+++ ",  # path-less header
        "@@",
        None,
    ]:
        try:
            score = score_diff_match(garbage or "", _SIMPLE_DIFF_A)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
        except Exception as e:
            pytest.fail(f"score_diff_match raised on {garbage!r}: {e}")


def test_m32_unicode_in_diff_body():
    """Emoji + non-ASCII content shouldn't break sha256 hashing or
    line splitting."""
    a = """\
--- a/x.py
+++ b/x.py
@@ -1 +1 @@
-old 🟠
+new 🟦
"""
    s = score_diff_match(a, a)
    assert s == 1.0


def test_m33_evaluate_handles_garbage_gracefully():
    """evaluate_diff_match must never raise -- worst case is a
    reject verdict with a low score."""
    r = evaluate_diff_match("\x00garbage\x01", _SIMPLE_DIFF_A)
    assert isinstance(r, DiffMatchResult)
    assert isinstance(r.accept, bool)
    assert isinstance(r.score, float)
