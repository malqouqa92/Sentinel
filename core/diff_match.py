"""Phase 16 Batch B -- deterministic diff-match verification.

When Batch C ships and a /code skip-eligible pattern is replayed, we
need a CHEAP deterministic gate that decides "did the replay produce
a diff structurally similar to the stored recipe's diff?". The whole
point of skipping Claude review is speed; using Claude as the safety
net would defeat that. This module is the safety net.

The algorithm: hunk-set Jaccard.

  1. Parse both diffs into a set of (file_path, hunk_signature) tuples.
     - file_path: the project-relative path the hunk targets.
     - hunk_signature: a content-derived fingerprint that's stable
       under minor line-number drift but changes when the actual edit
       changes. Signature is a sorted, normalized tuple of:
         - the count of added lines
         - the count of removed lines
         - sha256 of the concatenated added lines (no surrounding ctx)
         - sha256 of the concatenated removed lines

  2. Jaccard similarity: |stored ∩ replay| / |stored ∪ replay|.

  3. Three special cases gate the ACCEPT verdict on top of the score:
     - empty diffs match each other (1.0) trivially -- REJECT as a
       safety guard. Empty replay diff means nothing happened; we
       don't want a silent "the file was already in the desired
       state" pass.
     - diff parse failure -> 0.0 (treat as "no agreement").
     - one-side-empty -> 0.0 (one diff produced changes, the other
       didn't -- not a match).

  4. Threshold for ACCEPT: score >= DIFF_MATCH_ACCEPT_THRESHOLD (0.7
     by default). 0.7 picked to allow same-file same-sized-edit drift
     while rejecting wrong-file or wrong-content edits.

This module is pure deterministic Python -- no LLM, no GPU, no
network. Best-effort: any internal exception returns 0.0 (treated as
no agreement). Never raises.

Wired into /code's pattern-store path as TELEMETRY-ONLY in this
batch: every successful /code records the score against the closest
stored pattern, but the score does not gate any behavior yet. Batch
C consumes the score for the skip-eligibility decision.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


DIFF_MATCH_ACCEPT_THRESHOLD = 0.7


@dataclass(frozen=True)
class DiffMatchResult:
    """Result of a diff-match comparison.

    score: in [0.0, 1.0]; hunk-set Jaccard.
    accept: bool; True iff score >= threshold AND not a special-case
            rejection.
    reason: short human-readable string explaining accept/reject.
    """
    score: float
    accept: bool
    reason: str


# Matches `--- a/path` then `+++ b/path` then `@@ -X,Y +A,B @@`
# blocks. Tolerates GNU diff and `git diff` output. Captures the
# b-side path (the post-image) since that's the canonical target.
_FILE_HEADER_RE = re.compile(
    r"^\+\+\+\s+(?:b/)?(?P<path>[^\s\n]+)$",
    re.MULTILINE,
)
_HUNK_HEADER_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@",
                             re.MULTILINE)


def _normalize_line(line: str) -> str:
    """Drop trailing whitespace + the leading +/- marker. Used to
    build the content fingerprint so whitespace drift doesn't change
    the hash."""
    return line.rstrip()


def _hunk_signature(hunk_body: str) -> tuple[int, int, str, str]:
    """Compute a content-derived fingerprint of a single hunk body.

    Hunk body = the lines AFTER the @@ marker, before the next @@ or
    file-header. Lines are categorized by their first character:
      '+' -> added line
      '-' -> removed line
      ' ' -> context (ignored for the signature)
    Empty / malformed lines are skipped.

    Returns (n_added, n_removed, sha256_added, sha256_removed)."""
    added: list[str] = []
    removed: list[str] = []
    for ln in hunk_body.splitlines():
        if not ln:
            continue
        marker = ln[0]
        body = ln[1:]
        if marker == "+":
            added.append(_normalize_line(body))
        elif marker == "-":
            removed.append(_normalize_line(body))
        # context (' '), '\\' (no-newline marker), and stray noise: skip
    add_blob = "\n".join(added)
    rm_blob = "\n".join(removed)
    return (
        len(added),
        len(removed),
        hashlib.sha256(add_blob.encode("utf-8", "replace")).hexdigest()[:16],
        hashlib.sha256(rm_blob.encode("utf-8", "replace")).hexdigest()[:16],
    )


def _parse_hunks(diff_text: str) -> set[tuple]:
    """Parse a diff into a set of (file_path, hunk_signature) tuples.

    File-path detection uses the +++ b/ header. Hunks within a file
    are split on @@ markers. Returns empty set on parse failure or
    empty input."""
    if not diff_text or not diff_text.strip():
        return set()
    hunks: set[tuple] = set()
    # Walk through the diff line-by-line tracking the current file
    # and accumulating hunk bodies.
    current_path: str | None = None
    current_body: list[str] = []
    in_hunk = False

    def flush() -> None:
        nonlocal current_body
        if in_hunk and current_path and current_body:
            sig = _hunk_signature("\n".join(current_body))
            # Skip pure-context hunks (no add or remove)
            if sig[0] > 0 or sig[1] > 0:
                hunks.add((current_path, sig))
        current_body = []

    for line in diff_text.splitlines():
        # `--- a/path` is the FROM-image file header. Multiple files
        # in one diff are separated by `--- a/...` then `+++ b/...`
        # pairs. Treat `--- a/` as a hunk boundary so the next file's
        # header lines don't get absorbed into the prior hunk's body
        # (which would corrupt the +/- signature counts and hashes).
        if re.match(r"^---\s+(?:a/|/dev/null|[^+])", line):
            flush()
            in_hunk = False
            continue
        # File header on the post-image side (+++ b/path).
        m = re.match(r"^\+\+\+\s+(?:b/)?(.+)$", line)
        if m:
            flush()
            current_path = m.group(1).strip()
            in_hunk = False
            continue
        # Hunk header (@@ -X,Y +A,B @@)
        if line.startswith("@@") and "@@" in line[2:]:
            flush()
            in_hunk = True
            current_body = []
            continue
        if in_hunk:
            current_body.append(line)
    flush()
    return hunks


def score_diff_match(stored_diff: str, replay_diff: str) -> float:
    """Pure score in [0.0, 1.0]. Hunk-set Jaccard.

    Best-effort: any internal exception returns 0.0. Never raises."""
    try:
        stored_set = _parse_hunks(stored_diff or "")
        replay_set = _parse_hunks(replay_diff or "")
        if not stored_set and not replay_set:
            return 0.0  # both empty -- not "match", caller decides
        if not stored_set or not replay_set:
            return 0.0
        intersection = stored_set & replay_set
        union = stored_set | replay_set
        if not union:
            return 0.0
        return len(intersection) / len(union)
    except Exception:
        return 0.0


def evaluate_diff_match(
    stored_diff: str, replay_diff: str,
    threshold: float = DIFF_MATCH_ACCEPT_THRESHOLD,
) -> DiffMatchResult:
    """Score + accept verdict + reason.

    Special-case rejections (regardless of score):
      - replay_diff is empty -> reject ("replay produced no changes;
        empty diff can't be a positive verification")
      - stored_diff is empty -> reject ("nothing to compare against")
      - score < threshold -> reject ("hunk-set Jaccard X.XX below
        threshold Y.YY")
    """
    if not (replay_diff or "").strip():
        return DiffMatchResult(
            score=0.0, accept=False,
            reason="replay produced no diff (empty); cannot verify",
        )
    if not (stored_diff or "").strip():
        return DiffMatchResult(
            score=0.0, accept=False,
            reason="stored pattern has no diff to compare against",
        )
    score = score_diff_match(stored_diff, replay_diff)
    if score >= threshold:
        return DiffMatchResult(
            score=score, accept=True,
            reason=f"hunk-set Jaccard {score:.3f} >= {threshold:.3f}",
        )
    return DiffMatchResult(
        score=score, accept=False,
        reason=f"hunk-set Jaccard {score:.3f} below {threshold:.3f}",
    )
