"""Phase 16 token-leak fix -- redact credentials before KB persistence.

Caught 2026-05-06 in trace SEN-b46a27cf and earlier: Qwen's shadow plans
on emoji-progress-bar prompts started producing recipes that edited
``.env.bot`` (a file Qwen had no business modifying) and copied the
literal ``SENTINEL_TELEGRAM_TOKEN=...`` value into the new content. Those
recipe strings + their resulting diffs were stored verbatim in
``knowledge.db`` (columns ``solution_code``, ``qwen_plan_recipe``).

Result: 4 leaked rows (id=90, 91, 96, 97) at discovery. The KB then
fed those rows back into Qwen's few-shot retrieval pool, contaminating
future shadow plans -- a self-amplifying leak.

This module is the redaction layer. It runs at INSERT time
(``KnowledgeBase._add``) so no token-shaped string ever lands in the KB
in the first place. Pure-regex, pure-function, no I/O.

Conservative pattern set (env-var-style only) -- matches the actual
leak shape on this project; low false-positive risk on legitimate code.
Wider patterns (Bearer / ghp_ / sk_ / AKIA) deliberately deferred --
chosen by the owner on 2026-05-06.
"""
from __future__ import annotations

import re

# ─── Configuration ──────────────────────────────────────────────────

# Sentinel marker. All redactions emit this exact string so logs +
# downstream consumers can detect "this row was sanitized" with a single
# substring check.
REDACTED_MARKER = "<REDACTED>"

# The credential-key suffixes we treat as sensitive. Conservative
# (env-var style) per owner directive 2026-05-06 (Q1=A). Match is
# case-insensitive on the suffix; the prefix can be any \w* chars
# (letters, digits, underscore).
_SECRET_KEY_SUFFIXES = (
    "TOKEN",
    "KEY",
    "SECRET",
    "PASSWORD",
    "PASSPHRASE",
    "API_KEY",  # explicit -- matches even when prefix is empty
)

# Regex assembled once at import.
#
# Pattern logic:
#   - `(?P<key>...)` captures the credential-name part. Word chars then
#     one of the sensitive suffixes (case-insensitive). Boundary `\b`
#     before the prefix prevents matching mid-word noise.
#   - `=` literal (env-var assignment shape).
#   - `(?P<value>...)` captures the value. Stops at whitespace,
#     newline, comma, semicolon, or end-of-string. Quoted values
#     (single or double) are captured up to the matching quote.
#
# Two alternatives merged into one regex via | so we make a single
# pass. Quoted values are captured INCLUDING quotes so the redaction
# preserves shape (TOKEN="<REDACTED>" stays a quoted string).

_SUFFIX_GROUP = "|".join(
    re.escape(s) for s in sorted(_SECRET_KEY_SUFFIXES, key=len, reverse=True)
)

# ALL-CAPS env-var-style key matcher only. Deliberately strict to
# avoid matching Python code (`token = ...`, `key=key`, `password ==
# x`, `WHERE key = ?`). Real leaks on this project are all
# ENV_VAR_NAME=value shape per audit 2026-05-06; a wider pattern
# previously flagged 5 false positives in legitimate code.
#
# Key shape: optional all-caps prefix + `_` + sensitive suffix.
# - `TOKEN=...`           matches (no prefix, suffix only)
# - `TELEGRAM_TOKEN=...`  matches (caps prefix + suffix)
# - `MY_API_KEY=...`      matches
# - `token=...`           DOES NOT match (lowercase)
# - `key = value`         DOES NOT match (whitespace around =)
# - `kwargs[key]=v`       DOES NOT match (lowercase)
# - `WHERE key = ?`       DOES NOT match (lowercase + spaces)
_KEY_PATTERN = (
    rf"\b(?P<key>(?:[A-Z][A-Z0-9_]*_)?(?:{_SUFFIX_GROUP}))="
)

# Two value shapes: quoted "..." / '...', or unquoted-up-to-delimiter.
_VAL_QUOTED = r'(?P<vq>"[^"\n]{1,512}"|\'[^\'\n]{1,512}\')'
_VAL_UNQUOTED = r"(?P<vu>[^\s,;&<>\"'\\\n]{1,512})"

# Combined: KEY= followed by either quoted or unquoted value.
# NO re.IGNORECASE -- the strictness is the point.
_SECRET_RE = re.compile(
    _KEY_PATTERN + rf"(?:{_VAL_QUOTED}|{_VAL_UNQUOTED})",
)


# ─── Public API ─────────────────────────────────────────────────────

def scrub(text: str | None) -> str | None:
    """Return ``text`` with every credential value replaced by
    ``<REDACTED>``. Idempotent (running scrub on already-scrubbed text
    is a no-op). Preserves the credential KEY so logs/diffs still show
    which env-var was edited; only the VALUE is removed.

    Returns the input unchanged on:
      - None (passthrough)
      - empty string
      - no match (most rows -- cheap fast path)

    Never raises."""
    if not text:
        return text
    if REDACTED_MARKER in text and not _SECRET_RE.search(text):
        # Already sanitized AND no fresh leaks -- skip work.
        return text
    try:
        return _SECRET_RE.sub(_replace_match, text)
    except Exception:
        # Pathological input (giant string, regex backtrack worst-case).
        # Returning the text unchanged would re-leak; returning a fully
        # blank string discards data. Compromise: return marker-only.
        return REDACTED_MARKER


def contains_secret(text: str | None) -> bool:
    """Quick check: does this text contain at least one credential
    value the scrubber would redact? Cheap; same regex used by scrub.
    """
    if not text:
        return False
    return bool(_SECRET_RE.search(text))


def _replace_match(m: re.Match) -> str:
    """Build the replacement string for a single match. Preserves the
    key + `=` + (if quoted) the surrounding quote characters so shape
    is intact for human readers and for downstream parsers that may
    still walk the env-var-style line."""
    key = m.group("key")
    quoted = m.group("vq")
    if quoted:
        # Preserve the quote character so the redacted line reads as a
        # quoted string still (e.g. TOKEN="<REDACTED>"). Choose quote
        # by inspecting the matched value's first char.
        q = quoted[0]
        return f"{key}={q}{REDACTED_MARKER}{q}"
    return f"{key}={REDACTED_MARKER}"
