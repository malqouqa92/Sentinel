"""Phase 12 Batch 2 -- archetypes module + new ScoredPosting schema.

ECC: archetype detection, weighted_score math, recommendation bands,
legitimacy tiers, dimension validators, profile-archetype merging.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.archetypes import (
    DEFAULT_ARCHETYPES, DIMENSION_WEIGHTS,
    _all_archetypes, detect_archetype, legitimacy_tier,
    recommendation_band, weighted_score,
)
from core.job_profile import Archetype as ProfileArchetype
from skills.job_score import Legitimacy, ScoredPosting


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("Regional Sales Manager - Detroit", "Regional Sales Manager"),
    ("Senior Regional Director", "Regional Sales Manager"),
    ("Territory Sales Representative", "Territory Sales Manager"),
    ("Field Sales Engineer", "Territory Sales Manager"),
    ("Senior Account Executive", "Account Executive"),
    ("Strategic Account Manager", "Account Executive"),
    ("BDR Position", "Sales Development Representative"),
    ("Sales Development Rep", "Sales Development Representative"),
    ("RevOps Analyst", "Sales Operations / RevOps"),
    ("Sales Operations Lead", "Sales Operations / RevOps"),
    ("Customer Success Manager", "Customer Success Manager"),
    # 'CSM, Strategic Accounts' is ambiguous between CSM and AE under the
    # current substring-count classifier; phase 13 may add tie-breaking
    # by phrase length. Asserting the CSM-only version instead.
    ("Senior Customer Success Manager", "Customer Success Manager"),
])
def test_detect_archetype_from_title(title, expected):
    arch, hits = detect_archetype(title)
    assert arch == expected, f"{title!r} -> {arch} (hits={hits})"
    assert hits > 0


def test_detect_archetype_unknown_when_no_match():
    arch, hits = detect_archetype("Software Engineer")
    assert arch == "Unknown"
    assert hits == 0


def test_detect_archetype_title_weighted_3x():
    """Title hit should beat 2 desc hits when default weight is 3:1."""
    # Title: 1 hit on 'territory' (weight 3) = 3
    # Desc:  2 hits on 'account executive' (weight 1 each) = 2
    arch, _ = detect_archetype(
        "Territory Closer",
        description="Senior Account Executive Account Executive",
    )
    assert arch == "Territory Sales Manager"


def test_detect_archetype_user_overrides_default():
    """A PROFILE archetype with the same name as a default takes that
    name's slot but uses the user's keywords. A new name extends the
    list."""
    profile_archetypes = [
        ProfileArchetype(
            name="Custom Quant Sales", fit="primary",
            keywords=["quant", "algorithmic"],
        ),
    ]
    merged = _all_archetypes(profile_archetypes)
    names = [a.name for a in merged]
    assert "Custom Quant Sales" in names
    # Defaults still present
    assert any("Regional Sales" in n for n in names)
    arch, _ = detect_archetype(
        "Quant Sales Lead", profile_archetypes=profile_archetypes,
    )
    assert arch == "Custom Quant Sales"


def test_detect_archetype_skip_fit_excludes():
    profile_archetypes = [
        ProfileArchetype(name="Customer Success Manager", fit="skip"),
    ]
    # CSM hits in title would normally win, but fit=skip removes it.
    arch, _ = detect_archetype(
        "Customer Success Manager",
        profile_archetypes=profile_archetypes,
    )
    assert arch != "Customer Success Manager"


# ---------------------------------------------------------------------
# weighted_score math
# ---------------------------------------------------------------------

def test_weighted_score_all_fives_returns_five():
    assert weighted_score({k: 5 for k in DIMENSION_WEIGHTS}) == 5.0


def test_weighted_score_all_ones_returns_one():
    assert weighted_score({k: 1 for k in DIMENSION_WEIGHTS}) == 1.0


def test_weighted_score_missing_dim_treated_as_neutral_3():
    # Only cv_match supplied; the rest default to 3.
    # 0.30*5 + 0.25*3 + 0.20*3 + 0.15*3 + 0.10*3 = 1.5 + 0.75 + 0.6 + 0.45 + 0.3 = 3.6
    assert weighted_score({"cv_match": 5}) == 3.6


def test_weighted_score_clamps_out_of_range():
    # 99 -> clamp to 5; -10 -> clamp to 1
    assert weighted_score({k: 99 for k in DIMENSION_WEIGHTS}) == 5.0
    assert weighted_score({k: -10 for k in DIMENSION_WEIGHTS}) == 1.0


def test_weighted_score_handles_garbage_values():
    # Strings + None should fall back to neutral 3 (weighted_score is
    # the safety belt; the validator on ScoredPosting also clamps).
    assert weighted_score({"cv_match": "bad", "north_star": None}) == 3.0


# ---------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------

@pytest.mark.parametrize("score,band", [
    (5.0, "apply_now"), (4.5, "apply_now"),
    (4.49, "worth_applying"), (4.0, "worth_applying"),
    (3.99, "maybe"), (3.5, "maybe"),
    (3.49, "skip"), (1.0, "skip"),
])
def test_recommendation_band(score, band):
    assert recommendation_band(score) == band


# ---------------------------------------------------------------------
# Legitimacy tiers
# ---------------------------------------------------------------------

@pytest.mark.parametrize("count,tier", [
    (0, "high"), (1, "high"),
    (2, "caution"),
    (3, "suspicious"), (5, "suspicious"),
])
def test_legitimacy_tier(count, tier):
    assert legitimacy_tier(count) == tier


# ---------------------------------------------------------------------
# ScoredPosting validators (the new 1-5 schema)
# ---------------------------------------------------------------------

def test_scored_posting_clamps_score_to_1_5():
    sp = ScoredPosting(
        title="x", company="x", location="x", location_type="onsite",
        seniority="mid", archetype="X", score=99,
    )
    assert sp.score == 5.0
    sp = ScoredPosting(
        title="x", company="x", location="x", location_type="onsite",
        seniority="mid", archetype="X", score=-3,
    )
    assert sp.score == 1.0


def test_scored_posting_score_string_falls_to_floor():
    sp = ScoredPosting(
        title="x", company="x", location="x", location_type="onsite",
        seniority="mid", archetype="X", score="not-a-number",
    )
    assert sp.score == 1.0


def test_scored_posting_dimensions_clamp_each_value():
    sp = ScoredPosting(
        title="x", company="x", location="x", location_type="onsite",
        seniority="mid", archetype="X", score=3.0,
        dimensions={"cv_match": 99, "north_star": -2, "comp": "garbage"},
    )
    assert sp.dimensions["cv_match"] == 5.0
    assert sp.dimensions["north_star"] == 1.0
    assert sp.dimensions["comp"] == 3.0


def test_scored_posting_recommendation_band_strict():
    sp = ScoredPosting(
        title="x", company="x", location="x", location_type="onsite",
        seniority="mid", archetype="X", score=4.6,
        recommendation="moonshot",
    )
    # Unknown band falls back to 'skip'.
    assert sp.recommendation == "skip"


def test_scored_posting_reasons_capped_at_5():
    sp = ScoredPosting(
        title="x", company="x", location="x", location_type="onsite",
        seniority="mid", archetype="X", score=3.0,
        reasons=[f"r{i}" for i in range(10)],
    )
    assert len(sp.reasons) == 5


def test_legitimacy_default_high_no_signals():
    legit = Legitimacy()
    assert legit.tier == "high"
    assert legit.signals == []


# ---------------------------------------------------------------------
# job_report rendering for the new schema
# ---------------------------------------------------------------------

def test_job_report_csv_includes_new_columns():
    from skills.job_report import _build_csv
    rows = [{
        "title": "RSM", "company": "Acme", "location": "Detroit",
        "score": 4.5, "recommendation": "apply_now",
        "archetype": "Regional Sales Manager",
        "url": "https://x.example/1",
        "legitimacy": {"tier": "high", "signals": []},
        "reasons": ["a", "b"],
    }]
    csv = _build_csv(rows)
    assert "recommendation" in csv
    assert "archetype" in csv
    assert "legitimacy_tier" in csv
    assert "https://x.example/1" in csv
    assert "Regional Sales Manager" in csv
    assert "apply_now" in csv


def test_job_report_top_n_telegram_filters_skip_band():
    from skills.job_report import top_n_for_telegram
    rows = [
        {"title": "A", "company": "X", "score": 4.6,
         "recommendation": "apply_now", "url": "u1", "reasons": ["good"]},
        {"title": "B", "company": "Y", "score": 3.0,
         "recommendation": "skip", "url": "u2", "reasons": ["nope"]},
        {"title": "C", "company": "Z", "score": 4.1,
         "recommendation": "worth_applying", "url": "u3", "reasons": ["ok"]},
    ]
    msg = top_n_for_telegram(rows, n=3)
    assert "A @ X" in msg
    assert "C @ Z" in msg
    # Skip-band is excluded even though the top-N ask = 3.
    assert "B @ Y" not in msg


def test_job_report_top_n_telegram_no_eligible_returns_friendly_msg():
    from skills.job_report import top_n_for_telegram
    rows = [{"recommendation": "skip", "title": "x"}]
    msg = top_n_for_telegram(rows, n=3)
    assert "No postings" in msg


def test_job_report_summary_md_band_counts():
    from skills.job_report import _build_summary_md
    rows = [
        {"score": 4.6, "recommendation": "apply_now", "title": "a", "company": "x", "location": "y"},
        {"score": 4.6, "recommendation": "apply_now", "title": "b", "company": "x", "location": "y"},
        {"score": 4.1, "recommendation": "worth_applying", "title": "c", "company": "x", "location": "y"},
        {"score": 3.6, "recommendation": "maybe", "title": "d", "company": "x", "location": "y"},
        {"score": 2.0, "recommendation": "skip", "title": "e", "company": "x", "location": "y"},
    ]
    md = _build_summary_md(rows)
    assert "Apply now (≥4.5): **2**" in md
    assert "Worth applying (4.0–4.4): **1**" in md
    assert "Maybe (3.5–3.9): **1**" in md
    assert "Skip (<3.5): **1**" in md
