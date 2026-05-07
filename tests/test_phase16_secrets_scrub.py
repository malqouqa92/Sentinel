"""Phase 16 token-leak fix -- ECC tests for secrets_scrub + KB wiring.

Two scopes:
  1. Pure-function correctness of `core.secrets_scrub.scrub`:
     credential patterns it MUST catch, shapes it MUST preserve, and
     non-credentials it MUST NOT touch.
  2. KB integration: `KnowledgeBase._add` scrubs all four risky text
     columns BEFORE persistence, so a row inserted with a leaked token
     comes back clean from the DB.

Owner directive 2026-05-06 (Q1=A): conservative env-var-style patterns
only -- TOKEN/KEY/SECRET/PASSWORD/PASSPHRASE. Wider patterns
(Bearer/ghp_/sk_/AKIA) deferred. Tests are scoped accordingly.
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from core import config, embeddings as emb, secrets_scrub
from core.knowledge_base import KnowledgeBase
from core.secrets_scrub import (
    REDACTED_MARKER,
    contains_secret,
    scrub,
)


def _stub_embedder(monkeypatch):
    """Don't hit Ollama from tests -- deterministic fake embedding."""
    def fake_embed(text, trace_id="SEN-system"):
        seed = sum(ord(c) for c in (text or "")) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(config.EMBEDDING_DIM).tolist()
        return struct.pack(f"<{len(vec)}f", *vec)
    monkeypatch.setattr(emb, "embed_text", fake_embed)

# ─────────────────────────────────────────────────────────────────────
# scrub() -- pure-function pattern coverage
# ─────────────────────────────────────────────────────────────────────


def test_t01_redacts_telegram_token():
    """The exact leak shape from trace SEN-b46a27cf -- env-var with
    `<NAME>_TOKEN=<value>` containing a colon-and-base64 token."""
    txt = "SENTINEL_TELEGRAM_TOKEN=1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef-g"
    out = scrub(txt)
    assert out is not None
    assert "1234567890" not in out
    assert "ABCDEFGHIJ" not in out
    assert REDACTED_MARKER in out
    assert "SENTINEL_TELEGRAM_TOKEN=" in out  # key preserved


def test_t02_redacts_bare_token():
    """A naked `TOKEN=value` (no project prefix) must still match --
    `_KEY_PATTERN` allows zero-width prefix."""
    out = scrub("TOKEN=abc123def456")
    assert "abc123def456" not in out
    assert REDACTED_MARKER in out


@pytest.mark.parametrize("suffix", ["TOKEN", "KEY", "SECRET", "PASSWORD", "PASSPHRASE", "API_KEY"])
def test_t03_each_suffix_redacts(suffix):
    """Every credential suffix in `_SECRET_KEY_SUFFIXES` redacts. Names
    that didn't match would silently leak in production."""
    txt = f"SOME_{suffix}=secret_value_xyz_789"
    out = scrub(txt)
    assert "secret_value_xyz_789" not in out, f"{suffix} did not redact"


def test_t04_lowercase_keys_intentionally_skipped():
    """Lowercase keys (`token=`, `password=`, `key = value`) are
    intentionally NOT redacted -- they're Python variable assignment
    shape, not env-var leaks. The conservative regex (Q1=A) targets
    ENV_VAR_NAME=value only. Verified against the historical
    false-positive set: id=9/31/48/54/70 had lowercase `token=` /
    `key=` from legitimate Python code that the original (looser)
    regex was flagging."""
    out = scrub("my_token=lowercase_value_abc")
    assert out == "my_token=lowercase_value_abc", (
        "lowercase env-var-shaped strings must NOT be redacted "
        "(they're indistinguishable from Python kwargs)"
    )
    out2 = scrub("password = some_value")
    assert out2 == "password = some_value"
    out3 = scrub("key=foo, value=bar")
    assert out3 == "key=foo, value=bar"


def test_t05_quoted_value_preserves_quotes():
    """`TOKEN="..."` redaction must keep the quote characters intact so
    config-file shape is preserved for human readers."""
    out = scrub('TOKEN="abc-secret-xyz"')
    assert 'TOKEN="' in out
    assert f'"{REDACTED_MARKER}"' in out
    assert "abc-secret-xyz" not in out


def test_t06_single_quoted_value_preserves_quotes():
    out = scrub("API_KEY='abc-secret-xyz'")
    assert "API_KEY='" in out
    assert f"'{REDACTED_MARKER}'" in out
    assert "abc-secret-xyz" not in out


def test_t07_value_stops_at_whitespace():
    """Unquoted values stop at whitespace -- so a recipe step like
    `TOKEN=xyz123 some other text` redacts only `xyz123`."""
    out = scrub("TOKEN=xyz123 leftover_word")
    assert "xyz123" not in out
    assert "leftover_word" in out, "non-credential trailing text must survive"


def test_t08_value_stops_at_newline():
    out = scrub("TOKEN=secret_a\nTOKEN=secret_b")
    assert "secret_a" not in out
    assert "secret_b" not in out
    # Both lines redacted independently
    assert out.count(REDACTED_MARKER) == 2


def test_t09_multiple_credentials_in_same_string():
    """Several creds in one block -- each is independently redacted."""
    txt = (
        "TELEGRAM_TOKEN=tok1\n"
        "ANTHROPIC_API_KEY=tok2\n"
        "DB_PASSWORD=tok3"
    )
    out = scrub(txt)
    assert "tok1" not in out
    assert "tok2" not in out
    assert "tok3" not in out
    assert out.count(REDACTED_MARKER) == 3


def test_t10_idempotent_on_redacted_text():
    """Running scrub on already-redacted text produces the same string,
    not double-redaction or corruption."""
    once = scrub("TOKEN=abc123")
    twice = scrub(once)
    assert once == twice


def test_t11_passes_through_text_with_no_credentials():
    """Plain code/prose with no env-var-shaped credentials is returned
    unchanged. False-positive guard."""
    txt = (
        'STEP 1: edit_file path="interfaces/telegram_bot.py" '
        'old="🟦" new="🟩"'
    )
    out = scrub(txt)
    assert out == txt


def test_t12_none_passthrough():
    assert scrub(None) is None


def test_t13_empty_string_passthrough():
    assert scrub("") == ""


def test_t14_does_not_match_word_boundaries():
    """`mockey=value` must NOT match -- no `_KEY` boundary; the regex
    requires a sensitive-suffix terminal match. False-positive on
    legitimate-looking variable names is the failure mode here."""
    out = scrub("# this is not a token: mongo_keystore=open")
    # `keystore` is not a key suffix; the substring `_KEY` inside
    # `keystore` does NOT terminate at the `=`.
    assert "open" in out
    # And the original substring should be intact:
    assert "mongo_keystore=open" in out


def test_t15_lowercase_url_token_intentionally_skipped():
    """URL-style `?token=value` is lowercase, so the strict regex does
    NOT match. Acceptable: lowercase is genuinely ambiguous (could be
    a Python kwarg, a code variable, OR a URL token). The conservative
    cut-off chosen 2026-05-06 favors zero false-positives in code over
    catching every plausible URL leak. If wider URL coverage is ever
    needed, add an explicit `\\?\\w*token=` pattern."""
    txt = "https://example.com/api?token=abc123&other=keep"
    out = scrub(txt)
    assert out == txt, "lowercase URL token should pass through (intentional)"


def test_t15b_uppercase_url_token_does_redact():
    """An UPPERCASE env-var-style key (`?TOKEN=...`) WOULD match,
    consistent with the all-caps env-var rule."""
    out = scrub("?TELEGRAM_TOKEN=abc123&other=keep")
    assert "abc123" not in out
    assert "other=keep" in out


def test_t16_value_length_caps_at_512():
    """Pathological 10MB input shouldn't hang the regex via excessive
    backtracking. The {1,512} bound enforces an upper limit per match
    so worst-case is bounded."""
    huge = "x" * 10_000
    out = scrub(f"TOKEN={huge} done")
    # The first 512 chars get redacted; remaining `x`s stay.
    assert REDACTED_MARKER in out
    assert "done" in out


def test_t17_real_recipe_from_id96_redacts():
    """The actual leaked recipe text from pattern id=96 -- regression
    guard so this exact byte sequence can never re-leak."""
    leaked = (
        'STEP 1: read_file path=".env.bot"\n'
        'STEP 2: edit_file path=".env.bot" '
        'old="SENTINEL_TELEGRAM_TOKEN=1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef-g" '
        'new="SENTINEL_TELEGRAM_TOKEN=1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef-g\\nTELEGRAM_BOT_EMOJI=🟦"'
    )
    out = scrub(leaked)
    assert "1234567890" not in out
    assert "ABCDEFGHIJ" not in out
    # Recipe shape (STEP/path) preserved
    assert "STEP 1:" in out
    assert ".env.bot" in out


def test_t18_contains_secret_positive():
    assert contains_secret("PASSWORD=hello") is True


def test_t19_contains_secret_negative():
    assert contains_secret("def hello(): return 42") is False


def test_t20_contains_secret_none_safe():
    assert contains_secret(None) is False
    assert contains_secret("") is False


# ─────────────────────────────────────────────────────────────────────
# KnowledgeBase wiring -- INSERT path scrubs every text column
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_kb(tmp_path: Path, monkeypatch) -> KnowledgeBase:
    """Isolated KB on a tmp_path. Mirror the Phase 15b fixture pattern
    (Path object, not str; embedder stubbed to avoid Ollama)."""
    db_path = tmp_path / "test_knowledge.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    _stub_embedder(monkeypatch)
    return KnowledgeBase(db_path=db_path)


def _row(kb: KnowledgeBase, pid: int) -> sqlite3.Row:
    """Pull the raw row to inspect every text column directly."""
    conn = sqlite3.connect(kb.db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM knowledge WHERE id = ?", (pid,)
        ).fetchone()
    finally:
        conn.close()


def test_t30_add_pattern_scrubs_solution_code(fresh_kb):
    """A pattern inserted with a leaked token in `solution_code` must
    come back from the DB redacted."""
    pid = fresh_kb.add_pattern(
        tags=["leak", "test"],
        problem_summary="add an env var",
        solution_code=(
            "diff --git a/.env b/.env\n"
            "--- a/.env\n"
            "+++ b/.env\n"
            "@@ -1,1 +1,2 @@\n"
            "+API_TOKEN=plaintext_secret_xyz\n"
        ),
        solution_pattern="STEP 1: done",
        explanation="adds an api token to dotenv",
        trace_id="SEN-test-t30",
    )
    row = _row(fresh_kb, pid)
    assert "plaintext_secret_xyz" not in row["solution_code"]
    assert REDACTED_MARKER in row["solution_code"]


def test_t31_add_pattern_scrubs_solution_pattern(fresh_kb):
    """The `solution_pattern` column (Claude's stored recipe) must
    also be scrubbed -- recipes can include literal tokens via
    edit_file old=/new= args."""
    pid = fresh_kb.add_pattern(
        tags=["leak"],
        problem_summary="rotate token",
        solution_code="(diff)",
        solution_pattern=(
            'STEP 1: edit_file path=".env" '
            'old="SECRET_KEY=oldvalue" '
            'new="SECRET_KEY=newsecret_xyz"\n'
            'STEP 2: done summary="rotated"'
        ),
        explanation="rotates a secret",
        trace_id="SEN-test-t31",
    )
    row = _row(fresh_kb, pid)
    assert "oldvalue" not in row["solution_pattern"]
    assert "newsecret_xyz" not in row["solution_pattern"]
    assert "STEP 1:" in row["solution_pattern"]


def test_t32_add_pattern_scrubs_qwen_plan_recipe(fresh_kb):
    """The Qwen shadow recipe column was the actual leak vector for
    pattern id=96 -- this is the regression guard."""
    leaked_shadow = (
        'STEP 1: read_file path=".env.bot"\n'
        'STEP 2: edit_file path=".env.bot" '
        'old="SENTINEL_TELEGRAM_TOKEN=abc123" '
        'new="SENTINEL_TELEGRAM_TOKEN=def456\\nFOO=🟦"'
    )
    pid = fresh_kb.add_pattern(
        tags=["leak"],
        problem_summary="emoji bar",
        solution_code="(diff)",
        solution_pattern="STEP 1: done",
        explanation="emoji change",
        trace_id="SEN-test-t32",
        qwen_plan_recipe=leaked_shadow,
        qwen_plan_agreement=0.5,
    )
    row = _row(fresh_kb, pid)
    assert "abc123" not in row["qwen_plan_recipe"]
    assert "def456" not in row["qwen_plan_recipe"]
    assert "STEP 1:" in row["qwen_plan_recipe"]


def test_t33_add_pattern_scrubs_explanation(fresh_kb):
    """Even explanation text gets scrubbed -- defense in depth in case
    Claude's prose mentions an actual credential value."""
    pid = fresh_kb.add_pattern(
        tags=["leak"],
        problem_summary="prose leak",
        solution_code="(diff)",
        solution_pattern="STEP 1: done",
        explanation=(
            "We rotated the token. The old value was OLD_TOKEN=stale_val "
            "and we replaced it with NEW_TOKEN=fresh_val per security policy."
        ),
        trace_id="SEN-test-t33",
    )
    row = _row(fresh_kb, pid)
    assert "stale_val" not in row["explanation"]
    assert "fresh_val" not in row["explanation"]
    # Prose around the redactions intact
    assert "rotated" in row["explanation"]


def test_t34_add_limitation_scrubs_qwen_plan_recipe(fresh_kb):
    """Limitations carry shadow data too (Phase 15d). Same wiring,
    same scrub behavior."""
    pid = fresh_kb.add_limitation(
        tags=["limitation"],
        problem_summary="hopeless",
        explanation="qwen could not handle",
        trace_id="SEN-test-t34",
        qwen_plan_recipe='edit_file old="TOKEN=hopeless_val" new="x"',
        qwen_plan_agreement=0.0,
    )
    row = _row(fresh_kb, pid)
    assert "hopeless_val" not in row["qwen_plan_recipe"]


def test_t35_clean_pattern_passes_through_unchanged(fresh_kb):
    """Plain code with NO credentials must reach the DB byte-for-byte
    -- the scrub must not corrupt legitimate code via false positives."""
    clean_recipe = (
        'STEP 1: edit_file path="interfaces/telegram_bot.py" '
        'old="🟦" new="🟩"\n'
        'STEP 2: run_bash command="python -c \\"print(1)\\""\n'
        'STEP 3: done summary="emoji change"'
    )
    pid = fresh_kb.add_pattern(
        tags=["clean"],
        problem_summary="emoji bar change",
        solution_code="(diff)",
        solution_pattern=clean_recipe,
        explanation="straightforward emoji swap",
        trace_id="SEN-test-t35",
    )
    row = _row(fresh_kb, pid)
    assert row["solution_pattern"] == clean_recipe, (
        "clean recipe must round-trip byte-for-byte"
    )


def test_t36_problem_summary_scrubbed(fresh_kb):
    """If a user-supplied prompt has a credential, it gets scrubbed
    too (defense in depth at the boundary)."""
    pid = fresh_kb.add_pattern(
        tags=["edge"],
        problem_summary="add MY_TOKEN=hardcoded_val to config",
        solution_code="(diff)",
        solution_pattern="STEP 1: done",
        explanation="x",
        trace_id="SEN-test-t36",
    )
    row = _row(fresh_kb, pid)
    assert "hardcoded_val" not in row["problem_summary"]
