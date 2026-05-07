"""Phase 12.5 -- core/geo: zip lookup, distance, commute gate.

ECC. No network (pgeocode is offline). No LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.geo import (
    _looks_foreign, city_state_to_latlong,
    distance_miles_from_zip, haversine_miles, outside_commute,
    parse_city_state, zip_to_latlong,
)
from core.job_profile import LocationPref, Profile


# ---------------------------------------------------------------------
# zip_to_latlong
# ---------------------------------------------------------------------

def test_zip_to_latlong_known_us_zip():
    r = zip_to_latlong("48125")  # Dearborn Heights
    assert r is not None
    lat, lng = r
    assert 42.0 < lat < 42.6
    assert -83.5 < lng < -83.0


def test_zip_to_latlong_accepts_extended_zip():
    """Five-digit + four-digit suffix should still resolve."""
    r = zip_to_latlong("48125-1234")
    assert r is not None


@pytest.mark.parametrize("bad", ["", "abc", "123", "1234567890", "ZZZZZ", None])
def test_zip_to_latlong_rejects_garbage(bad):
    assert zip_to_latlong(bad) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# parse_city_state
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Detroit, MI, US", ("Detroit", "MI")),
    ("Detroit, MI", ("Detroit", "MI")),
    ("Detroit, MI 48125", ("Detroit", "MI")),
    ("Cincinnati, OH, US", ("Cincinnati", "OH")),
    ("Chicago, IL, USA", ("Chicago", "IL")),
])
def test_parse_city_state_us_shapes(text, expected):
    assert parse_city_state(text) == expected


@pytest.mark.parametrize("text", [
    "Toronto, Ontario, Canada",
    "Mississauga, Ontario, Canada",
    "Indiana, United States",  # state-only, no city
    "Remote",
    "(unknown)",
    "",
])
def test_parse_city_state_returns_none_on_unparseable(text):
    assert parse_city_state(text) is None


# ---------------------------------------------------------------------
# haversine_miles
# ---------------------------------------------------------------------

def test_haversine_dearborn_to_chicago_in_expected_range():
    # Dearborn Heights ~ (42.27, -83.26), Chicago ~ (41.88, -87.62)
    miles = haversine_miles(42.2768, -83.2606, 41.8858, -87.6181)
    assert 220 < miles < 240, f"got {miles}"


def test_haversine_zero_distance():
    assert haversine_miles(42, -83, 42, -83) == 0


# ---------------------------------------------------------------------
# distance_miles_from_zip end-to-end
# ---------------------------------------------------------------------

@pytest.mark.parametrize("loc,low,high", [
    ("Detroit, MI, US",       0, 30),
    ("Cincinnati, OH, US",  200, 250),
    ("Chicago, IL, US",     220, 250),
    ("Hudsonville, MI, US", 130, 160),
    ("Milwaukee, WI, US",   230, 260),
])
def test_distance_from_48125_us_cities(loc, low, high):
    d = distance_miles_from_zip("48125", loc)
    assert d is not None and low <= d <= high, f"{loc}: got {d}"


@pytest.mark.parametrize("loc", [
    "Toronto, Ontario, Canada",
    "Mississauga, Ontario, Canada",
    "Indiana, United States",          # state-only -- can't geocode
    "(unknown)",
    "",
    "Some company HQ",
])
def test_distance_returns_none_on_unresolvable(loc):
    assert distance_miles_from_zip("48125", loc) is None


def test_distance_with_embedded_zip_in_text():
    """If location text contains a zip, prefer that to city/state lookup."""
    assert distance_miles_from_zip("48125", "Cincinnati, OH 45202") is not None


def test_distance_returns_none_when_src_zip_invalid():
    assert distance_miles_from_zip("ZZZZZ", "Detroit, MI, US") is None


# ---------------------------------------------------------------------
# _looks_foreign (word-boundary)
# ---------------------------------------------------------------------

def test_looks_foreign_matches_country_words():
    """Multiple tokens may match; assert the result is in a known
    set rather than a specific value (set iteration is unordered)."""
    assert _looks_foreign("Toronto, Ontario, Canada") in {"canada", "ontario"}
    assert _looks_foreign("London, England") == "england"
    assert _looks_foreign("Berlin, Germany") == "germany"


def test_looks_foreign_uses_word_boundary_not_substring():
    """The substring 'india' is in 'Indiana' -- the word-boundary check
    must NOT flag Indiana as foreign."""
    assert _looks_foreign("Indianapolis, IN, US") is None
    assert _looks_foreign("Indiana, United States") is None
    # And the real 'India' still matches:
    assert _looks_foreign("Mumbai, India") == "india"


def test_looks_foreign_returns_none_on_us_text():
    assert _looks_foreign("Detroit, MI, US") is None
    assert _looks_foreign("(unknown)") is None
    assert _looks_foreign("") is None


# ---------------------------------------------------------------------
# outside_commute -- the policy
# ---------------------------------------------------------------------

def _profile_can_commute(zip_, max_miles, willing=False):
    return Profile(
        location=LocationPref(
            primary_zip=zip_, onsite_max_miles=max_miles,
            willing_to_relocate=willing,
        ),
    )


def test_outside_commute_distant_us_onsite_is_gated():
    p = _profile_can_commute("48125", 20)
    is_out, miles, reason = outside_commute(
        p, "onsite", "Cincinnati, OH, US",
    )
    assert is_out
    assert miles is not None and 200 < miles < 250
    assert "20-mi commute cap" in reason


def test_outside_commute_local_onsite_passes():
    p = _profile_can_commute("48125", 20)
    is_out, miles, reason = outside_commute(
        p, "onsite", "Detroit, MI, US",
    )
    assert not is_out
    assert miles is not None and miles < 20


def test_outside_commute_remote_bypasses_distance():
    p = _profile_can_commute("48125", 20)
    is_out, miles, reason = outside_commute(
        p, "remote", "Schaumburg, IL, US",
    )
    assert not is_out
    assert "remote" in reason


def test_outside_commute_foreign_country_is_gated():
    p = _profile_can_commute("48125", 20)
    is_out, miles, reason = outside_commute(
        p, "onsite", "Toronto, Ontario, Canada",
    )
    assert is_out
    assert miles is None
    # Either province ("ontario") or country ("canada") may be the
    # first match -- both are valid foreign signals.
    assert ("canada" in reason.lower()) or ("ontario" in reason.lower())


def test_outside_commute_willing_to_relocate_disables_gate():
    p = _profile_can_commute("48125", 20, willing=True)
    is_out, _, reason = outside_commute(
        p, "onsite", "Cincinnati, OH, US",
    )
    assert not is_out
    assert "willing to relocate" in reason


def test_outside_commute_no_zip_or_max_miles_disables_gate():
    p = Profile(location=LocationPref())  # all defaults: zip='', max_miles=None
    is_out, _, _ = outside_commute(p, "onsite", "Cincinnati, OH, US")
    assert not is_out


def test_outside_commute_unresolvable_us_location_falls_through():
    """An unresolvable US location (e.g. just a state) doesn't gate --
    the LLM still gets to score it. Conservative on purpose."""
    p = _profile_can_commute("48125", 20)
    is_out, miles, reason = outside_commute(
        p, "onsite", "Indiana, United States",
    )
    assert not is_out
    assert miles is None


def test_outside_commute_hybrid_treated_like_onsite():
    p = _profile_can_commute("48125", 20)
    is_out, _, _ = outside_commute(p, "hybrid", "Chicago, IL, US")
    assert is_out


# ---------------------------------------------------------------------
# job_score gate integration -- short-circuits BEFORE the LLM
# ---------------------------------------------------------------------

def test_job_score_commute_gate_skips_llm(monkeypatch, tmp_path):
    """An out-of-commute on-site posting must produce a skip-band
    ScoredPosting WITHOUT the LLM ever being called."""
    import asyncio
    import yaml
    from skills.job_score import JobScoreInput, JobScoreSkill
    # Seed PROFILE so the gate has data to fire on
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "PROFILE.yml").write_text(yaml.safe_dump({
        "location": {
            "primary_zip": "48125", "onsite_max_miles": 20,
            "willing_to_relocate": False,
        },
    }), encoding="utf-8")
    monkeypatch.setattr("core.config.PERSONA_DIR", persona)

    # If the gate fires the LLM should NOT be called -- assert that.
    from skills import job_score
    def fail_llm(*a, **kw):
        raise AssertionError("LLM must not be called when commute gate fires")
    monkeypatch.setattr(job_score, "_score_blocking", fail_llm)

    skill = JobScoreSkill()
    inp = skill.validate_input({
        "title": "Regional Sales Manager (East)",
        "company": "EDP",
        "location": "Chicago, IL, US",
        "location_type": "onsite",
        "url": "https://example.com/x",
        "seniority": "senior",
    })
    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-gate"))
    assert out.recommendation == "skip"
    assert out.score == 2.0
    assert out.dimensions["red_flags"] == 1.0
    assert any("commute" in r.lower() for r in out.reasons), out.reasons


def test_job_score_local_onsite_does_not_gate(monkeypatch, tmp_path):
    """A local onsite posting (within the cap) MUST fall through to
    the LLM. Test by replacing the LLM with a stub that returns a
    canned high score and asserting we see it."""
    import asyncio
    import yaml
    from skills.job_score import JobScoreInput, JobScoreSkill, ScoredPosting
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "PROFILE.yml").write_text(yaml.safe_dump({
        "location": {
            "primary_zip": "48125", "onsite_max_miles": 20,
            "willing_to_relocate": False,
        },
    }), encoding="utf-8")
    monkeypatch.setattr("core.config.PERSONA_DIR", persona)

    from skills import job_score
    def fake_llm(inp, profile_summary, archetype, boost, trace_id, model_id,
                 client=None, region_adjust=0.0, state_ok=None, weights=None):
        return ScoredPosting(
            title=inp.title, company=inp.company, location=inp.location,
            location_type=inp.location_type, seniority=inp.seniority or "mid",
            url=inp.url, archetype=archetype, score=4.5,
            dimensions={"cv_match": 5, "north_star": 5, "comp": 4,
                        "cultural_signals": 4, "red_flags": 5},
            recommendation="apply_now", reasons=["good fit"],
        )
    monkeypatch.setattr(job_score, "_score_blocking", fake_llm)

    skill = JobScoreSkill()
    inp = skill.validate_input({
        "title": "RSM Detroit",
        "company": "Acme", "location": "Detroit, MI, US",
        "location_type": "onsite", "url": "https://example.com/y",
        "seniority": "senior",
    })
    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-pass"))
    # LLM ran -- the canned response made it through.
    assert out.recommendation == "apply_now"
    assert out.score == 4.5
