"""Phase 12 Batch 1 -- PROFILE.yml loader + title/avoid/workplace filters.

ECC: every input class for the loader and filter helpers.
No real jobspy, no real LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.job_profile import (
    Profile, Archetype, TitleFilter, has_seniority_boost,
    load_profile, title_passes,
)
from skills.job_scrape import (
    JobScrapeInput, _row_to_posting, _workplace_to_is_remote,
)


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------

def test_load_profile_returns_default_when_file_missing(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)
    p = load_profile()
    assert isinstance(p, Profile)
    assert p.target_roles.archetypes == []
    assert p.title_filter.positive == []
    assert p.location.workplace_preference == "all"


def test_load_profile_full_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)
    src = {
        "candidate": {"full_name": "X", "location": "Detroit, MI"},
        "target_roles": {
            "primary": ["Regional Sales Manager"],
            "archetypes": [
                {"name": "Regional Sales Manager", "fit": "primary",
                 "keywords": ["regional sales"]},
                {"name": "SDR", "fit": "adjacent"},
            ],
        },
        "title_filter": {
            "positive": ["Sales", "ACCOUNT"],   # case mixed
            "negative": [" intern ", "junior"],  # whitespace
            "seniority_boost": ["senior"],
        },
        "avoid_companies": ["BadCorp", "Spammers Inc"],
        "location": {"workplace_preference": "hybrid"},
    }
    (tmp_path / "PROFILE.yml").write_text(
        yaml.safe_dump(src), encoding="utf-8",
    )
    p = load_profile()
    assert p.candidate.location == "Detroit, MI"
    assert len(p.target_roles.archetypes) == 2
    # Normalization: lowercased, deduped, stripped.
    assert p.title_filter.positive == ["account", "sales"]
    assert p.title_filter.negative == ["intern", "junior"]
    assert p.title_filter.seniority_boost == ["senior"]
    assert "BadCorp" in p.avoid_companies
    assert p.location.workplace_preference == "hybrid"


def test_load_profile_malformed_yaml_returns_default(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)
    (tmp_path / "PROFILE.yml").write_text(
        "candidate: [unclosed\n", encoding="utf-8",
    )
    p = load_profile()
    assert isinstance(p, Profile)
    assert p.target_roles.archetypes == []


def test_load_profile_top_level_not_mapping_returns_default(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)
    (tmp_path / "PROFILE.yml").write_text(
        "- list-instead-of-dict\n", encoding="utf-8",
    )
    p = load_profile()
    assert p.target_roles.archetypes == []


def test_load_profile_invalid_archetype_fit_returns_default(
    tmp_path, monkeypatch,
):
    """A bad enum value (fit='ultimate') breaks validation; loader logs
    + returns defaults, never raises."""
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)
    (tmp_path / "PROFILE.yml").write_text(yaml.safe_dump({
        "target_roles": {
            "archetypes": [{"name": "X", "fit": "ultimate"}],
        },
    }), encoding="utf-8")
    p = load_profile()
    assert p.target_roles.archetypes == []


# ---------------------------------------------------------------------
# title_passes -- the pre-LLM filter
# ---------------------------------------------------------------------

def _profile_with(positive=None, negative=None, avoid=None, boost=None):
    return Profile(
        title_filter=TitleFilter(
            positive=positive or [], negative=negative or [],
            seniority_boost=boost or [],
        ).normalize(),
        avoid_companies=avoid or [],
    )


def test_title_passes_empty_filter_lets_everything_through():
    p = _profile_with()
    assert title_passes("Anything", p, company="Anyone")


def test_title_passes_requires_positive_match():
    p = _profile_with(positive=["sales"])
    assert title_passes("Senior Sales Manager", p)
    assert not title_passes("Software Engineer", p)


def test_title_passes_rejects_on_negative_match():
    p = _profile_with(positive=["sales"], negative=["intern", "junior"])
    assert not title_passes("Sales Intern", p)
    assert not title_passes("Junior Sales Rep", p)
    assert title_passes("Senior Sales Manager", p)


def test_title_passes_negative_takes_precedence_over_positive():
    p = _profile_with(positive=["sales", "engineer"], negative=["sales"])
    # 'Sales Engineer' has positive 'engineer' AND negative 'sales' --
    # negative wins.
    assert not title_passes("Sales Engineer", p)


def test_title_passes_avoid_company_substring_match():
    p = _profile_with(positive=["sales"], avoid=["BadCorp"])
    assert not title_passes("Sales Mgr", p, company="BadCorp Inc")
    assert not title_passes("Sales Mgr", p, company="badcorp inc")  # case
    assert title_passes("Sales Mgr", p, company="GoodCo")


def test_title_passes_extra_avoid_per_call_merges():
    p = _profile_with(positive=["sales"], avoid=["BadCorp"])
    # /jobsearch --avoid "Foo,Bar" comes in as extra_avoid
    assert not title_passes(
        "Sales Mgr", p, extra_avoid=["Foo"], company="Foo Industries",
    )
    assert not title_passes(
        "Sales Mgr", p, extra_avoid=["Bar"], company="Acme Bar",
    )
    assert title_passes(
        "Sales Mgr", p, extra_avoid=["Foo"], company="Acme Inc",
    )


def test_title_passes_avoid_matches_in_title_too():
    """An avoid keyword that appears in the title (not just the company)
    also rejects -- useful for blacklisting role keywords."""
    p = _profile_with(positive=["sales"])
    assert not title_passes(
        "Sales Manager - Cannabis", p,
        extra_avoid=["cannabis"], company="GoodCo",
    )


def test_has_seniority_boost():
    p = _profile_with(boost=["senior", "principal"])
    assert has_seniority_boost("Senior Sales Manager", p)
    assert has_seniority_boost("Principal Account Director", p)
    assert not has_seniority_boost("Entry Sales Rep", p)


# ---------------------------------------------------------------------
# JobScrapeInput field validators
# ---------------------------------------------------------------------

def test_jobscrape_input_workplace_aliases():
    cases = {
        "remote": "remote", "wfh": "remote",
        "on-site": "on-site", "onsite": "on-site", "in-person": "on-site",
        "hybrid": "hybrid",
        "all": "all", "any": "all", "": "all", None: "all",
    }
    for raw, expected in cases.items():
        m = JobScrapeInput(query="x", workplace=raw)
        assert m.workplace == expected, f"{raw!r} -> {m.workplace}"


def test_jobscrape_input_rejects_invalid_workplace():
    with pytest.raises(Exception):
        JobScrapeInput(query="x", workplace="lunar")


def test_jobscrape_input_avoid_split_string():
    m = JobScrapeInput(query="x", avoid="Foo,Bar Baz, ,Quux")
    assert m.avoid == ["Foo", "Bar Baz", "Quux"]


def test_jobscrape_input_avoid_list_passthrough():
    m = JobScrapeInput(query="x", avoid=["A ", " B"])
    assert m.avoid == ["A", "B"]


def test_workplace_to_is_remote_mapping():
    assert _workplace_to_is_remote("remote") is True
    assert _workplace_to_is_remote("on-site") is False
    assert _workplace_to_is_remote("hybrid") is None
    assert _workplace_to_is_remote("all") is None


# ---------------------------------------------------------------------
# _row_to_posting (jobspy row -> ScrapedPosting)
# ---------------------------------------------------------------------

def test_row_to_posting_carries_workplace_pref():
    row = {
        "title": "Sales Mgr", "company": "Acme",
        "location": "Detroit, MI", "description": "blah",
        "job_url": "https://example/x", "site": "indeed",
    }
    p = _row_to_posting(row, workplace_pref="hybrid")
    assert p.workplace_pref == "hybrid"
    assert p.title == "Sales Mgr"
    assert p.url == "https://example/x"


def test_row_to_posting_handles_missing_fields_gracefully():
    p = _row_to_posting({}, workplace_pref="all")
    assert p.title == "" and p.company == "" and p.url == ""
    assert p.workplace_pref == "all"
