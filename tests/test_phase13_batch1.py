"""Phase 13 Batch 1 -- scrape-time dedup, region preference, --avoid verification.

ECC. Mocked LLM, real SQLite via conftest's temp_db fixture.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, database
from core.job_profile import (
    Profile, LocationPref, TitleFilter,
    load_profile, region_score_adjustment, state_in_whitelist,
    title_passes,
)


# ---------------------------------------------------------------------
# #2 -- scrape-time dedup against applications table
# ---------------------------------------------------------------------

def test_scrape_dedups_urls_already_in_applications(monkeypatch, tmp_path):
    """A scraped posting whose url_hash matches an existing applications
    row should be dropped before extract/score."""
    from skills.job_scrape import JobScrapeSkill
    # Empty profile so title_passes lets everything through.
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)

    seen_url = "https://example.com/job/seen"
    new_url = "https://example.com/job/fresh"
    database.upsert_application(
        url=seen_url, title="Old", company="X",
    )

    skill = JobScrapeSkill()
    fake_rows = [
        {"title": "RSM Detroit", "company": "X", "location": "Detroit, MI",
         "description": "blah", "job_url": seen_url, "site": "indeed"},
        {"title": "RSM Detroit 2", "company": "Y", "location": "Detroit, MI",
         "description": "blah", "job_url": new_url, "site": "indeed"},
    ]

    async def _run():
        with patch("skills.job_scrape._scrape_blocking",
                   return_value=fake_rows):
            inp = skill.validate_input({"query": "RSM"})
            return await skill.execute(inp, trace_id="SEN-test-dedup")
    out = asyncio.run(_run())

    assert len(out.postings) == 1
    assert out.postings[0].url == new_url


def test_scrape_dedup_with_url_canonicalization(monkeypatch, tmp_path):
    """Two URLs that canonicalize to the same hash (one with utm params)
    should be treated as the same posting."""
    from skills.job_scrape import JobScrapeSkill
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)

    base = "https://example.com/job/canonical"
    database.upsert_application(url=base, title="Seen", company="X")
    skill = JobScrapeSkill()
    fake_rows = [
        # Same canonical URL, with tracking params -- should still dedup
        {"title": "RSM", "company": "X", "location": "Detroit, MI",
         "description": "blah",
         "job_url": f"{base}?utm_source=foo&gclid=abc",
         "site": "indeed"},
    ]
    async def _run():
        with patch("skills.job_scrape._scrape_blocking",
                   return_value=fake_rows):
            inp = skill.validate_input({"query": "RSM"})
            return await skill.execute(inp, trace_id="SEN-test-canon")
    out = asyncio.run(_run())
    assert len(out.postings) == 0


def test_scrape_dedup_does_not_drop_url_less_postings(monkeypatch, tmp_path):
    """A scraped row with no URL should not be dedup-skipped (we have
    no key to compare). Phase 13 Batch 5b would multiply this row by
    the number of expanded queries, so we disable expansion here."""
    from skills.job_scrape import JobScrapeSkill
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)
    monkeypatch.setattr(config, "JOB_QUERY_EXPANSION_ENABLED", False)
    skill = JobScrapeSkill()
    fake_rows = [
        {"title": "RSM", "company": "X", "location": "Detroit, MI",
         "description": "blah", "job_url": "", "site": "indeed"},
    ]
    async def _run():
        with patch("skills.job_scrape._scrape_blocking",
                   return_value=fake_rows):
            inp = skill.validate_input({"query": "RSM"})
            return await skill.execute(inp, trace_id="SEN-test-no-url")
    out = asyncio.run(_run())
    assert len(out.postings) == 1


# ---------------------------------------------------------------------
# #3a -- region keyword boost / avoid
# ---------------------------------------------------------------------

def _profile_with_regions(boost=None, avoid=None):
    return Profile(
        title_filter=TitleFilter(
            region_boost=boost or [],
            region_avoid=avoid or [],
        ).normalize(),
    )


def test_region_boost_adds_half_per_hit_in_title_or_location():
    p = _profile_with_regions(
        boost=["midwest", "northeast", "michigan"],
    )
    # 1 hit (midwest in title)
    assert region_score_adjustment(
        "RSM Midwest", "Detroit, MI", p,
    ) == 0.5
    # 2 hits (northeast + michigan would be 1 from each)
    assert region_score_adjustment(
        "Northeast Director", "Michigan, US", p,
    ) == 1.0
    # 3 hits clamped to 1.0
    assert region_score_adjustment(
        "Midwest Northeast", "Michigan, MI", p,
    ) == 1.0


def test_region_avoid_subtracts_and_caps():
    p = _profile_with_regions(avoid=["southeast", "florida"])
    assert region_score_adjustment("RSM Southeast", "Miami, FL", p) == -0.5
    assert region_score_adjustment(
        "Southeast Director", "Florida, US", p,
    ) == -1.0


def test_region_boost_and_avoid_combine_directionally():
    p = _profile_with_regions(
        boost=["midwest"], avoid=["southeast"],
    )
    # +0.5 boost - 0.5 avoid = 0
    assert region_score_adjustment(
        "Midwest Southeast Hybrid", "X", p,
    ) == 0.0


def test_region_no_match_returns_zero():
    p = _profile_with_regions(
        boost=["midwest"], avoid=["southeast"],
    )
    assert region_score_adjustment("RSM", "Detroit, MI", p) == 0.0


# ---------------------------------------------------------------------
# #3b -- state whitelist
# ---------------------------------------------------------------------

def _profile_with_states(states):
    return Profile(location=LocationPref(accepted_states=states))


def test_state_whitelist_accepts_listed_state():
    p = _profile_with_states(["MI", "OH"])
    assert state_in_whitelist("Detroit, MI, US", p) is True
    assert state_in_whitelist("Cleveland, OH", p) is True


def test_state_whitelist_rejects_unlisted_state():
    p = _profile_with_states(["MI", "OH"])
    assert state_in_whitelist("Atlanta, GA", p) is False
    assert state_in_whitelist("San Francisco, CA, US", p) is False


def test_state_whitelist_returns_none_when_no_states_set():
    p = _profile_with_states([])
    assert state_in_whitelist("Detroit, MI", p) is None


def test_state_whitelist_returns_none_on_unparseable_location():
    """Foreign cities, 'Remote', '(unknown)' all return None -- no
    penalty applied to noise."""
    p = _profile_with_states(["MI"])
    assert state_in_whitelist("Toronto, Ontario, Canada", p) is None
    assert state_in_whitelist("(unknown)", p) is None
    assert state_in_whitelist("Remote", p) is None
    assert state_in_whitelist("", p) is None


def test_state_whitelist_normalizes_case():
    p = _profile_with_states(["mi", "ny"])  # lowercase in profile
    assert state_in_whitelist("Detroit, MI", p) is True


# ---------------------------------------------------------------------
# #3 integration with job_score: nudges affect global score
# ---------------------------------------------------------------------

def test_job_score_applies_region_boost(monkeypatch, tmp_path):
    """A posting in a boost region should score HIGHER than the same
    posting in a neutral location."""
    import yaml
    from skills.job_score import JobScoreInput, JobScoreSkill, ScoredPosting
    from skills import job_score

    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "PROFILE.yml").write_text(yaml.safe_dump({
        "title_filter": {
            "region_boost": ["midwest"],
            "region_avoid": [],
        },
        "location": {"accepted_states": []},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "PERSONA_DIR", persona)

    canned_dims = '{"dimensions":{"cv_match":4,"north_star":4,"comp":3,"cultural_signals":4,"red_flags":5},"reasons":["x"],"legitimacy_signals":[]}'

    class StubLLM:
        def generate(self, **kw):
            return canned_dims

    # Capture the real function BEFORE monkey-patching to avoid recursion.
    real_score_blocking = job_score._score_blocking

    def fake_score(inp, profile_summary, archetype, boost, trace_id, model_id,
                   client=None, region_adjust=0.0, state_ok=None, weights=None):
        return real_score_blocking(
            inp, profile_summary, archetype, boost, trace_id,
            model_id, StubLLM(), region_adjust, state_ok, weights,
        )
    monkeypatch.setattr(job_score, "_score_blocking", fake_score)

    skill = JobScoreSkill()
    base_inp = skill.validate_input({
        "title": "RSM", "company": "X", "location": "(unknown)",
        "location_type": "remote", "url": "http://x", "seniority": "senior",
    })
    boosted_inp = skill.validate_input({
        "title": "RSM Midwest", "company": "X",
        "location": "Detroit, MI",
        "location_type": "remote", "url": "http://y", "seniority": "senior",
    })
    base = asyncio.run(skill.execute(base_inp, trace_id="SEN-test-r1"))
    boosted = asyncio.run(skill.execute(boosted_inp, trace_id="SEN-test-r2"))
    assert boosted.score > base.score, (
        f"region boost should raise score: base={base.score} "
        f"boosted={boosted.score}"
    )


# ---------------------------------------------------------------------
# #11 -- --avoid flag still works (regression check)
# ---------------------------------------------------------------------

def test_jobsearch_avoid_flag_drops_company_postings(monkeypatch, tmp_path):
    """Passing --avoid 'BadCo,Spam' from the CLI should drop matching
    postings at scrape-time. Already implemented in Phase 12; test is
    a regression check."""
    from skills.job_scrape import JobScrapeSkill
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)
    skill = JobScrapeSkill()
    fake_rows = [
        {"title": "RSM", "company": "BadCo Inc", "location": "Detroit, MI",
         "description": "x", "job_url": "http://1", "site": "indeed"},
        {"title": "RSM", "company": "GoodCo", "location": "Detroit, MI",
         "description": "x", "job_url": "http://2", "site": "indeed"},
        {"title": "RSM", "company": "Spam Corp", "location": "Detroit, MI",
         "description": "x", "job_url": "http://3", "site": "indeed"},
    ]
    async def _run():
        with patch("skills.job_scrape._scrape_blocking",
                   return_value=fake_rows):
            inp = skill.validate_input({
                "query": "RSM", "avoid": "BadCo,Spam",
            })
            return await skill.execute(inp, trace_id="SEN-test-avoid")
    out = asyncio.run(_run())
    assert len(out.postings) == 1
    assert out.postings[0].company == "GoodCo"
