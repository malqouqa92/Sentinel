"""Phase 13 Batch 5 -- smarter rescan: query expansion + adaptive filter.

ECC. No network, no real LLM calls.

Coverage:
  Query expansion (a):
    01 -- empty query returns []
    02 -- expansion disabled -> single-element list
    03 -- query with no abbrev match -> single-element list
    04 -- "RSM Detroit" expands to include "Regional Sales Manager Detroit"
    05 -- max_variants cap is honored
    06 -- bidirectional: full title -> abbrev variant present
    07 -- duplicate variants are deduped
    08 -- case-insensitive matching but original case preserved

  Adaptive filter (b):
    11 -- should_act: below min_drops returns False
    12 -- should_act: below ratio threshold returns False
    13 -- should_act: feature flag off returns False
    14 -- should_act: above both thresholds returns True
    15 -- extract_candidates: brain returns valid -> filtered list
    16 -- extract_candidates: refuses keywords colliding with positives
    17 -- extract_candidates: refuses keywords already in negatives
    18 -- extract_candidates: capped at MAX_NEW
    19 -- extract_candidates: brain failure returns []
    20 -- apply_to_profile: appends to YAML, no duplicates
    21 -- apply_to_profile: missing PROFILE returns 0
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import adaptive_filter, config, query_expansion
from core.job_profile import (
    LocationPref, Profile, TitleFilter,
)


# ─────────────────────────────────────────────────────────────────
# Query expansion
# ─────────────────────────────────────────────────────────────────

def test_qe_01_empty_returns_empty():
    assert query_expansion.expand_query("") == []
    assert query_expansion.expand_query("   ") == []


def test_qe_02_disabled_returns_single(monkeypatch):
    monkeypatch.setattr(config, "JOB_QUERY_EXPANSION_ENABLED", False)
    assert query_expansion.expand_query("RSM Detroit") == ["RSM Detroit"]


def test_qe_03_no_abbrev_match_returns_single():
    out = query_expansion.expand_query("Quantum Coffee Roaster")
    assert out == ["Quantum Coffee Roaster"]


def test_qe_04_rsm_expands_to_full_title():
    out = query_expansion.expand_query("RSM Detroit", max_variants=3)
    assert out[0] == "RSM Detroit"
    assert any("regional sales manager" in v.lower() for v in out)
    assert len(out) >= 2


def test_qe_05_max_variants_cap_honored():
    out = query_expansion.expand_query("RSM Detroit", max_variants=2)
    assert len(out) == 2
    out1 = query_expansion.expand_query("RSM Detroit", max_variants=1)
    assert out1 == ["RSM Detroit"]


def test_qe_06_full_title_expands_to_abbrev():
    out = query_expansion.expand_query(
        "Regional Sales Manager Midwest", max_variants=3,
    )
    # Should include 'rsm' variant
    assert any("rsm" in v.lower() for v in out)


def test_qe_07_duplicate_variants_deduped():
    # 'sales operations' maps to 'revops' etc. Make sure we don't
    # emit the same string twice if multiple keys produce it.
    out = query_expansion.expand_query("Sales Operations", max_variants=5)
    seen = set()
    for v in out:
        assert v.lower() not in seen, f"dup variant: {v!r}"
        seen.add(v.lower())


def test_qe_08_case_insensitive_match_preserves_surrounding_case():
    out = query_expansion.expand_query("rsm DETROIT", max_variants=3)
    assert out[0] == "rsm DETROIT"
    # The expanded query keeps the surrounding " DETROIT" verbatim.
    assert any("DETROIT" in v for v in out[1:])


# ─────────────────────────────────────────────────────────────────
# Adaptive filter -- should_act
# ─────────────────────────────────────────────────────────────────

def test_af_11_should_act_below_min_drops(monkeypatch):
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_MIN_DROPS", 10)
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_DROP_RATIO", 0.4)
    assert not adaptive_filter.should_act(dropped_count=5, total_scraped=10)


def test_af_12_should_act_below_ratio(monkeypatch):
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_MIN_DROPS", 10)
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_DROP_RATIO", 0.4)
    # 12/100 = 12% below 40%
    assert not adaptive_filter.should_act(dropped_count=12, total_scraped=100)


def test_af_13_should_act_disabled(monkeypatch):
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_ENABLED", False)
    assert not adaptive_filter.should_act(dropped_count=100, total_scraped=100)


def test_af_14_should_act_passes_both_thresholds(monkeypatch):
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_ENABLED", True)
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_MIN_DROPS", 10)
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_DROP_RATIO", 0.4)
    assert adaptive_filter.should_act(dropped_count=20, total_scraped=40)


# ─────────────────────────────────────────────────────────────────
# Adaptive filter -- extract_candidates
# ─────────────────────────────────────────────────────────────────

def _stub_brain(response: str):
    """Returns a callable matching the brain_generate contract."""
    def _gen(prompt, system, format_json, trace_id):
        return response
    return _gen


def _profile_with_filters(
    positive=None, negative=None, seniority=None,
):
    return Profile(
        title_filter=TitleFilter(
            positive=positive or [],
            negative=negative or [],
            seniority_boost=seniority or [],
        ).normalize(),
        location=LocationPref(),
    )


def test_af_15_extract_returns_validated_list(monkeypatch):
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_MAX_NEW", 5)
    profile = _profile_with_filters()
    brain = _stub_brain('{"negatives": ["insurance", "mortgage", "loan"]}')
    out = adaptive_filter.extract_candidates(
        ["Insurance Sales Agent", "Mortgage Loan Officer"],
        profile, brain, "SEN-test-15",
    )
    assert out == ["insurance", "mortgage", "loan"]


def test_af_16_extract_refuses_collisions_with_positives(monkeypatch):
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_MAX_NEW", 5)
    # 'sales' is positive -- brain should not be allowed to add it.
    profile = _profile_with_filters(positive=["sales"])
    brain = _stub_brain('{"negatives": ["sales", "insurance"]}')
    out = adaptive_filter.extract_candidates(
        ["Insurance Sales Agent"], profile, brain, "SEN-test-16",
    )
    assert "sales" not in out
    assert "insurance" in out


def test_af_17_extract_refuses_already_present_negatives(monkeypatch):
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_MAX_NEW", 5)
    profile = _profile_with_filters(negative=["insurance"])
    brain = _stub_brain('{"negatives": ["insurance", "mortgage"]}')
    out = adaptive_filter.extract_candidates(
        ["X"], profile, brain, "SEN-test-17",
    )
    assert out == ["mortgage"]


def test_af_18_extract_capped_at_max_new(monkeypatch):
    monkeypatch.setattr(config, "JOB_ADAPTIVE_FILTER_MAX_NEW", 2)
    profile = _profile_with_filters()
    brain = _stub_brain(
        '{"negatives": ["a1", "b2", "c3", "d4", "e5"]}'
    )
    out = adaptive_filter.extract_candidates(
        ["X"], profile, brain, "SEN-test-18",
    )
    assert len(out) == 2
    assert out == ["a1", "b2"]


def test_af_19_extract_brain_failure_returns_empty():
    profile = _profile_with_filters()

    def boom(prompt, system, format_json, trace_id):
        raise RuntimeError("simulated brain crash")

    out = adaptive_filter.extract_candidates(
        ["X"], profile, boom, "SEN-test-19",
    )
    assert out == []


def test_af_19b_extract_garbage_response_returns_empty():
    profile = _profile_with_filters()
    brain = _stub_brain("not json at all")
    out = adaptive_filter.extract_candidates(
        ["X"], profile, brain, "SEN-test-19b",
    )
    assert out == []


# ─────────────────────────────────────────────────────────────────
# Adaptive filter -- apply_to_profile
# ─────────────────────────────────────────────────────────────────

def test_af_20_apply_appends_no_duplicates(monkeypatch, tmp_path):
    persona = tmp_path / "persona"
    persona.mkdir()
    profile_path = persona / "PROFILE.yml"
    profile_path.write_text(yaml.safe_dump({
        "title_filter": {
            "negative": ["insurance"],
            "positive": [],
        },
    }), encoding="utf-8")
    monkeypatch.setattr(config, "PERSONA_DIR", persona)

    # 'insurance' already there, 'mortgage' is new, 'loan' is new
    n = adaptive_filter.apply_to_profile(
        ["insurance", "mortgage", "loan"], "SEN-test-20",
    )
    assert n == 2
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    negatives = data["title_filter"]["negative"]
    assert "insurance" in negatives
    assert "mortgage" in negatives
    assert "loan" in negatives
    # No duplicate of 'insurance'
    assert sum(1 for k in negatives if k == "insurance") == 1


def test_af_21_apply_missing_profile_returns_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)  # no PROFILE.yml here
    assert adaptive_filter.apply_to_profile(["x"], "SEN-test-21") == 0
