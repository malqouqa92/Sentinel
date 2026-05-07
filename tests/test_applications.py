"""Phase 12 Batch 3 -- applications table + dedup + report integration.

ECC: schema, URL canonicalization, upsert idempotency, state-keep on
re-upsert, transition history, alias normalization, report writes
applications, top_n_telegram round-trips through Telegram handler.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, database
from skills.job_report import (
    JobReportInput, JobReportSkill, _build_summary_md, top_n_for_telegram,
)


# ---------------------------------------------------------------------
# URL canonicalization
# ---------------------------------------------------------------------

def test_hash_url_strips_tracking_params():
    """utm_*, gclid, fbclid drop out of the hash."""
    a = database._hash_url("https://example.com/job/1?utm_source=foo&id=1")
    b = database._hash_url("https://example.com/job/1?id=1")
    assert a == b


def test_hash_url_case_and_whitespace_insensitive():
    a = database._hash_url("  https://EXAMPLE.com/Job/1 ")
    b = database._hash_url("https://example.com/job/1")
    # Path case-sensitive on most servers; lowercased here for dedup
    # because we accept both /job/1 and /Job/1 as the same listing.
    assert a == b


def test_hash_url_empty_returns_empty():
    assert database._hash_url("") == ""
    assert database._hash_url(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Upsert + state semantics
# ---------------------------------------------------------------------

def test_upsert_application_inserts_new_row():
    aid = database.upsert_application(
        url="https://example.com/job/1",
        title="RSM", company="Acme", location="Detroit",
        archetype="Regional Sales Manager",
        score=4.6, recommendation="apply_now",
    )
    assert aid >= 1
    rows = database.list_applications()
    assert len(rows) == 1
    assert rows[0]["title"] == "RSM"
    assert rows[0]["state"] == "evaluated"
    assert rows[0]["score"] == 4.6
    history = json.loads(rows[0]["history"])
    assert history[0]["to"] == "evaluated"


def test_upsert_application_dedupes_on_canonical_url():
    aid1 = database.upsert_application(
        url="https://example.com/job/1?utm_source=foo",
        title="RSM", company="Acme",
    )
    aid2 = database.upsert_application(
        url="https://example.com/job/1?gclid=xyz&fbclid=abc",
        title="RSM (refreshed)", company="Acme", score=4.7,
    )
    assert aid1 == aid2
    rows = database.list_applications()
    assert len(rows) == 1
    assert rows[0]["title"] == "RSM (refreshed)"
    assert rows[0]["score"] == 4.7


def test_upsert_application_keeps_advanced_state_on_rescrape():
    """If the user has already moved an application to 'interview',
    a subsequent re-scrape that would set state='evaluated' must NOT
    regress the state."""
    aid = database.upsert_application(
        url="https://example.com/job/2", title="X", company="X",
    )
    database.transition_application(aid, "interview")
    # Re-scrape comes through with default state='evaluated'.
    database.upsert_application(
        url="https://example.com/job/2", title="X", company="X",
        score=4.0,
    )
    rows = database.list_applications()
    assert rows[0]["state"] == "interview"
    assert rows[0]["score"] == 4.0  # score still refreshes


def test_upsert_application_rejects_empty_url():
    with pytest.raises(ValueError):
        database.upsert_application(url="", title="x", company="x")


# ---------------------------------------------------------------------
# Transitions + history
# ---------------------------------------------------------------------

def test_transition_application_appends_history():
    aid = database.upsert_application(
        url="https://example.com/job/3", title="x", company="x",
    )
    upd = database.transition_application(aid, "applied", note="via portal")
    assert upd is not None
    assert upd["state"] == "applied"
    assert upd["applied_at"] is not None
    history = json.loads(upd["history"])
    assert len(history) == 2
    assert history[1]["from"] == "evaluated"
    assert history[1]["to"] == "applied"
    assert history[1]["note"] == "via portal"


def test_transition_application_returns_none_for_missing():
    assert database.transition_application(99999, "applied") is None


def test_transition_normalizes_state_aliases():
    aid = database.upsert_application(
        url="https://example.com/job/4", title="x", company="x",
    )
    # Spanish alias -> 'applied'
    upd = database.transition_application(aid, "aplicado")
    assert upd["state"] == "applied"


def test_transition_rejects_unknown_state():
    aid = database.upsert_application(
        url="https://example.com/job/5", title="x", company="x",
    )
    with pytest.raises(ValueError):
        database.transition_application(aid, "lunar")


# ---------------------------------------------------------------------
# list_applications + filtering
# ---------------------------------------------------------------------

def test_list_applications_filter_by_state():
    a = database.upsert_application(
        url="https://example.com/job/6", title="A", company="X",
    )
    b = database.upsert_application(
        url="https://example.com/job/7", title="B", company="Y",
    )
    database.transition_application(a, "applied")
    applied_only = database.list_applications(state="applied")
    assert {r["id"] for r in applied_only} == {a}
    eval_only = database.list_applications(state="evaluated")
    assert {r["id"] for r in eval_only} == {b}


def test_list_applications_unknown_state_raises():
    with pytest.raises(ValueError):
        database.list_applications(state="moonshot")


def test_application_exists():
    database.upsert_application(
        url="https://example.com/job/8", title="x", company="x",
    )
    assert database.application_exists("https://example.com/job/8")
    # canonicalization works for exists() too
    assert database.application_exists(
        "https://example.com/job/8?utm_campaign=z",
    )
    assert not database.application_exists("https://example.com/never")
    assert not database.application_exists("")


# ---------------------------------------------------------------------
# JobReport writes applications
# ---------------------------------------------------------------------

def test_job_report_persists_each_scored_to_applications(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        config, "JOBSEARCH_OUTPUT_DIR", tmp_path / "job_searches",
    )
    config.JOBSEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skill = JobReportSkill()
    scored = [
        {"title": "RSM", "company": "Acme", "location": "Detroit",
         "score": 4.6, "recommendation": "apply_now",
         "archetype": "Regional Sales Manager",
         "url": "https://example.com/job/A",
         "reasons": ["good"]},
        {"title": "TSM", "company": "Beta", "location": "Ann Arbor",
         "score": 4.1, "recommendation": "worth_applying",
         "archetype": "Territory Sales Manager",
         "url": "https://example.com/job/B",
         "reasons": ["ok"]},
        {"title": "skip-me", "company": "Z", "location": "Remote",
         "score": 2.0, "recommendation": "skip", "archetype": "X",
         "url": "",      # empty url -> skipped, not written
         "reasons": ["ghost"]},
    ]
    inp = JobReportInput(scored=scored)
    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-apps"))
    assert out.total_postings == 3
    apps = database.list_applications()
    # 2 written (skipped one had empty url).
    assert len(apps) == 2
    titles = {r["title"] for r in apps}
    assert titles == {"RSM", "TSM"}
    rsm = next(r for r in apps if r["title"] == "RSM")
    assert rsm["recommendation"] == "apply_now"
    assert rsm["score"] == 4.6
    assert rsm["state"] == "evaluated"


def test_job_report_top_n_telegram_in_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config, "JOBSEARCH_OUTPUT_DIR", tmp_path / "job_searches",
    )
    config.JOBSEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skill = JobReportSkill()
    inp = JobReportInput(scored=[
        {"title": "RSM", "company": "Acme", "location": "X",
         "score": 4.6, "recommendation": "apply_now",
         "url": "https://example.com/A", "reasons": ["good"]},
    ])
    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-tg"))
    assert out.top_n_telegram
    assert "RSM" in out.top_n_telegram
    assert "Acme" in out.top_n_telegram
    assert "https://example.com/A" in out.top_n_telegram
