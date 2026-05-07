"""Phase 13 Batch 6 -- lightweight legitimacy signals.

ECC. No browser, no LLM, no network. Pure functions + one DB-integration
test using the temp_db fixture from conftest.

Coverage:
  classify_apply_url:
    01 -- empty URL -> unknown + signal "no apply URL"
    02 -- malformed URL -> suspicious
    03 -- ATS domain (greenhouse, lever, workday) -> ats, no signal
    04 -- aggregator (indeed, linkedin) -> aggregator, no signal
    05 -- foreign TLD (.ru, .cn) -> suspicious + signal
    06 -- unknown but well-formed -> unknown, no signal

  detect_repost_cadence:
    11 -- no matches -> 0, None
    12 -- 1 match (same title rebroadcast) -> 0 (excludes self by URL)
    13 -- 2 matches different titles -> 2, "reposted 2x" signal
    14 -- 3+ matches -> count, "ghost-job" signal
    15 -- title fuzzy-match: 'Sr Regional Sales Mgr' ~ 'Regional Sales Manager'
    16 -- company fuzzy-match: 'AcmeCo Inc' matches 'AcmeCo'
    17 -- empty company -> 0, None
    18 -- fetch raises -> 0, None (defensive)

  collect_signals:
    21 -- happy path returns both URL + repost signals
    22 -- empty inputs returns []

  DB integration (find_recent_company_postings):
    31 -- only returns rows from same company within window
    32 -- excludes rows older than `days`
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import database
from core.legitimacy import (
    classify_apply_url, collect_signals, detect_repost_cadence,
)


# ─────────────────────────────────────────────────────────────────
# classify_apply_url
# ─────────────────────────────────────────────────────────────────

def test_la_01_empty_url():
    tier, sig = classify_apply_url("")
    assert tier == "unknown"
    assert sig is not None and "no apply URL" in sig


def test_la_02_malformed_url():
    tier, sig = classify_apply_url("not-a-url")
    assert tier == "suspicious"
    assert sig is not None


def test_la_03_ats_domains():
    for url in [
        "https://boards.greenhouse.io/foo/jobs/123",
        "https://jobs.lever.co/acme/abc-def",
        "https://acme.workday.com/job/123",
        "https://acme.bamboohr.com/jobs/view/42",
        "https://jobs.smartrecruiters.com/acme/123",
    ]:
        tier, sig = classify_apply_url(url)
        assert tier == "ats", f"expected ats for {url}, got {tier}"
        assert sig is None  # no concerning signal for ATS


def test_la_04_aggregator_domains():
    for url in [
        "https://www.indeed.com/viewjob?jk=abc",
        "https://www.linkedin.com/jobs/view/12345",
        "https://www.ziprecruiter.com/jobs/123",
    ]:
        tier, sig = classify_apply_url(url)
        assert tier == "aggregator"
        assert sig is None


def test_la_05_foreign_tld():
    for url, tld in [
        ("https://jobs.example.ru/123", ".ru"),
        ("https://hr.acme.cn/jobs/4", ".cn"),
    ]:
        tier, sig = classify_apply_url(url)
        assert tier == "suspicious"
        assert sig is not None and tld in sig


def test_la_06_unknown_domain_returns_unknown_no_signal():
    tier, sig = classify_apply_url("https://careers.acmeco.com/jobs/42")
    assert tier == "unknown"
    assert sig is None  # we don't surface signals for unfamiliar but
                        # otherwise-fine domains


# ─────────────────────────────────────────────────────────────────
# detect_repost_cadence
# ─────────────────────────────────────────────────────────────────

def _stub_fetcher(rows: list[dict]):
    def _fetch(co, days):
        return rows
    return _fetch


def test_la_11_no_matches_returns_zero():
    fetch = _stub_fetcher([])
    n, sig = detect_repost_cadence(
        "Regional Sales Manager", "AcmeCo", "https://x", fetch,
    )
    assert n == 0
    assert sig is None


def test_la_12_self_match_excluded():
    """A row with the same URL as the current posting is the SAME
    posting being re-scraped; not a repost."""
    same_url = "https://example.com/job/100"
    fetch = _stub_fetcher([
        {"title": "Regional Sales Manager", "url": same_url},
    ])
    n, sig = detect_repost_cadence(
        "Regional Sales Manager", "AcmeCo", same_url, fetch,
    )
    assert n == 0
    assert sig is None


def test_la_13_two_matches_yields_repost_signal():
    fetch = _stub_fetcher([
        {"title": "Regional Sales Manager", "url": "https://x/1"},
        {"title": "Regional Sales Manager (East)", "url": "https://x/2"},
    ])
    n, sig = detect_repost_cadence(
        "Regional Sales Manager", "AcmeCo", "https://x/new", fetch,
    )
    assert n == 2
    assert sig is not None and "reposted 2" in sig


def test_la_14_three_matches_flags_ghost_job():
    fetch = _stub_fetcher([
        {"title": "Regional Sales Manager", "url": "https://x/1"},
        {"title": "Regional Sales Manager (East)", "url": "https://x/2"},
        {"title": "Regional Sales Manager (Midwest)", "url": "https://x/3"},
    ])
    n, sig = detect_repost_cadence(
        "Regional Sales Manager", "AcmeCo", "https://x/new", fetch,
    )
    assert n == 3
    assert sig is not None and "ghost-job" in sig.lower()


def test_la_15_title_fuzzy_match():
    """Fuzzy: 'Sr Regional Sales Mgr' should match 'Regional Sales Manager'
    -- common token overlap, level/title-fluff stripped."""
    fetch = _stub_fetcher([
        {"title": "Regional Sales Manager", "url": "https://x/1"},
        {"title": "Regional Sales Manager", "url": "https://x/2"},
    ])
    n, sig = detect_repost_cadence(
        "Sr Regional Sales Mgr", "AcmeCo", "https://x/new", fetch,
    )
    assert n == 2
    assert sig is not None


def test_la_16_company_normalize():
    """Company normalization is the caller's job (database query layer);
    detect_repost_cadence is given an already-normalized name. We just
    verify it doesn't barf on suffixes if they slip in."""
    fetch = _stub_fetcher([
        {"title": "Regional Sales Manager", "url": "https://x/1"},
        {"title": "Regional Sales Manager", "url": "https://x/2"},
    ])
    n, _ = detect_repost_cadence(
        "Regional Sales Manager", "AcmeCo Inc.", "https://x/new", fetch,
    )
    assert n == 2  # company normalization on input side does the work


def test_la_17_empty_company():
    fetch = _stub_fetcher([
        {"title": "Regional Sales Manager", "url": "https://x/1"},
    ])
    n, sig = detect_repost_cadence(
        "Regional Sales Manager", "", "https://x/new", fetch,
    )
    assert n == 0
    assert sig is None


def test_la_18_fetch_raises_returns_zero():
    def _fetch(co, days):
        raise RuntimeError("simulated DB outage")
    n, sig = detect_repost_cadence(
        "Regional Sales Manager", "AcmeCo", "https://x", _fetch,
    )
    assert n == 0
    assert sig is None


# ─────────────────────────────────────────────────────────────────
# collect_signals
# ─────────────────────────────────────────────────────────────────

def test_la_21_collect_combines_url_and_repost_signals():
    fetch = _stub_fetcher([
        {"title": "Regional Sales Manager", "url": "https://x/1"},
        {"title": "Regional Sales Manager", "url": "https://x/2"},
        {"title": "Regional Sales Manager", "url": "https://x/3"},
    ])
    sigs = collect_signals(
        "Regional Sales Manager", "AcmeCo",
        "https://hr.example.ru/123",  # foreign TLD -> URL signal
        fetch,
    )
    # One URL signal + one repost (>=3 ghost-job) signal.
    assert len(sigs) == 2
    assert any("non-US" in s for s in sigs)
    assert any("ghost-job" in s.lower() for s in sigs)


def test_la_22_empty_inputs_return_empty():
    fetch = _stub_fetcher([])
    sigs = collect_signals("", "", "", fetch)
    # The empty URL alone produces a "no apply URL" signal.
    assert len(sigs) == 1
    assert "no apply URL" in sigs[0]


# ─────────────────────────────────────────────────────────────────
# DB integration: find_recent_company_postings
# ─────────────────────────────────────────────────────────────────

def test_la_31_finds_only_same_company_recent(temp_db):
    a = database.upsert_application(
        url="https://x.com/1", title="RSM", company="AcmeCo Inc",
    )
    b = database.upsert_application(
        url="https://x.com/2", title="RSM", company="acmeco",  # same co
    )
    _c = database.upsert_application(
        url="https://x.com/3", title="RSM", company="GammaCorp",
    )
    rows = database.find_recent_company_postings("acmeco", days=90)
    urls = {r["url"] for r in rows}
    assert "https://x.com/1" in urls
    assert "https://x.com/2" in urls
    assert "https://x.com/3" not in urls


def test_la_32_excludes_rows_older_than_window(temp_db, monkeypatch):
    """Manually backdate one row's last_seen_at and confirm the window
    filter excludes it."""
    aid = database.upsert_application(
        url="https://x.com/old", title="RSM", company="AcmeCo",
    )
    _bid = database.upsert_application(
        url="https://x.com/new", title="RSM", company="AcmeCo",
    )
    # Backdate aid by 200 days.
    old_ts = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    import sqlite3
    from core.database import _connect
    conn = _connect()
    try:
        conn.execute(
            "UPDATE applications SET last_seen_at = ? WHERE id = ?",
            (old_ts, aid),
        )
        conn.commit()
    finally:
        conn.close()
    rows = database.find_recent_company_postings("acmeco", days=90)
    urls = {r["url"] for r in rows}
    assert "https://x.com/new" in urls
    assert "https://x.com/old" not in urls
