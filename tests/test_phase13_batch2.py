"""Phase 13 Batch 2 -- per-archetype dimension weights.

ECC. No network, no LLM (one integration test stubs the LLM).

What we're verifying:
  - ARCHETYPE_DEFAULT_WEIGHTS presets exist for all 6 default archetypes
  - Each preset sums to 1.0 (the math relies on it)
  - weights_for_archetype() lookup chain: profile override > preset > global
  - PROFILE-supplied weights are normalized to sum=1.0
  - weighted_score() uses the passed-in weights, falls back to global default
  - End-to-end: a posting scored as RSM gets a different score than the
    same posting scored as RevOps (because their weights differ)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.archetypes import (
    ARCHETYPE_DEFAULT_WEIGHTS, DIMENSION_KEYS, DIMENSION_WEIGHTS,
    _normalize_weights, weighted_score, weights_for_archetype,
)
from core.job_profile import Archetype


# ---------------------------------------------------------------------
# Preset sanity
# ---------------------------------------------------------------------

def test_all_default_archetypes_have_a_preset():
    """Every name in DEFAULT_ARCHETYPES should have a weight preset."""
    from core.archetypes import DEFAULT_ARCHETYPES
    for arch in DEFAULT_ARCHETYPES:
        assert arch.name in ARCHETYPE_DEFAULT_WEIGHTS, (
            f"missing preset for archetype: {arch.name}"
        )


@pytest.mark.parametrize("name,preset", list(ARCHETYPE_DEFAULT_WEIGHTS.items()))
def test_preset_weights_sum_to_one(name, preset):
    """Math correctness gate: each row sums to ~1.0."""
    total = sum(preset.values())
    assert abs(total - 1.0) < 1e-9, f"{name} weights sum to {total}, not 1.0"


@pytest.mark.parametrize("name,preset", list(ARCHETYPE_DEFAULT_WEIGHTS.items()))
def test_preset_weights_cover_all_dimensions(name, preset):
    """Every preset must define every dimension; missing ones default to
    0 in weighted_score and silently distort results."""
    assert set(preset.keys()) == set(DIMENSION_KEYS), (
        f"{name} missing keys: {set(DIMENSION_KEYS) - set(preset.keys())}"
    )


# ---------------------------------------------------------------------
# weights_for_archetype lookup chain
# ---------------------------------------------------------------------

def test_lookup_returns_preset_for_known_archetype():
    w = weights_for_archetype("Regional Sales Manager", profile_archetypes=[])
    assert w == ARCHETYPE_DEFAULT_WEIGHTS["Regional Sales Manager"]


def test_lookup_falls_back_to_global_for_unknown_archetype():
    w = weights_for_archetype("Astronaut Closer", profile_archetypes=[])
    assert w == DIMENSION_WEIGHTS


def test_lookup_profile_override_wins_over_preset():
    """A PROFILE archetype with explicit weights overrides the preset."""
    custom = {"cv_match": 0.5, "north_star": 0.3, "comp": 0.1,
              "cultural_signals": 0.05, "red_flags": 0.05}
    profile_arch = [Archetype(name="Regional Sales Manager", weights=custom)]
    w = weights_for_archetype("Regional Sales Manager", profile_arch)
    # Should match custom (already sums to 1.0).
    for k in DIMENSION_KEYS:
        assert abs(w[k] - custom[k]) < 1e-9


def test_lookup_profile_archetype_without_weights_falls_through_to_preset():
    """Profile names the archetype but doesn't override weights -> preset."""
    profile_arch = [Archetype(name="Regional Sales Manager", weights={})]
    w = weights_for_archetype("Regional Sales Manager", profile_arch)
    assert w == ARCHETYPE_DEFAULT_WEIGHTS["Regional Sales Manager"]


def test_lookup_normalizes_user_weights_summing_to_two():
    """If user supplies weights summing to 2.0, normalize to 1.0."""
    custom = {"cv_match": 1.0, "north_star": 0.5, "comp": 0.3,
              "cultural_signals": 0.1, "red_flags": 0.1}  # sums to 2.0
    profile_arch = [Archetype(name="X", weights=custom)]
    w = weights_for_archetype("X", profile_arch)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # Ratios preserved.
    assert abs(w["cv_match"] - 0.5) < 1e-9  # 1.0/2.0


def test_lookup_handles_partial_user_weights():
    """User supplies only 2 of 5 dims -- the rest default to 0 and the
    given two are normalized to sum=1.0. (User's job to supply all 5
    if they want a balanced override; this is a graceful-degradation
    test, not an endorsement.)"""
    profile_arch = [Archetype(
        name="Regional Sales Manager",
        weights={"cv_match": 1.0, "north_star": 1.0},
    )]
    w = weights_for_archetype("Regional Sales Manager", profile_arch)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w["cv_match"] - 0.5) < 1e-9
    assert abs(w["north_star"] - 0.5) < 1e-9
    assert w["comp"] == 0.0


def test_normalize_returns_none_on_empty_or_zero():
    assert _normalize_weights(None) is None
    assert _normalize_weights({}) is None
    assert _normalize_weights({"cv_match": 0.0, "north_star": 0.0}) is None


def test_normalize_drops_negative_values():
    out = _normalize_weights({"cv_match": 1.0, "north_star": -1.0})
    assert out["cv_match"] == 1.0
    assert out["north_star"] == 0.0


def test_lookup_case_insensitive_match():
    """Profile archetype name matched case-insensitively."""
    custom = {"cv_match": 0.4, "north_star": 0.3, "comp": 0.1,
              "cultural_signals": 0.1, "red_flags": 0.1}
    profile_arch = [Archetype(name="REGIONAL SALES MANAGER", weights=custom)]
    w = weights_for_archetype("Regional Sales Manager", profile_arch)
    assert abs(w["cv_match"] - 0.4) < 1e-9


# ---------------------------------------------------------------------
# weighted_score with explicit weights
# ---------------------------------------------------------------------

def test_weighted_score_uses_passed_weights():
    """A 5-on-cv_match scoring 1 elsewhere with cv_match-heavy weights
    should score higher than the same dims with the global default."""
    dims = {"cv_match": 5, "north_star": 1, "comp": 1,
            "cultural_signals": 1, "red_flags": 1}
    cv_heavy = {"cv_match": 0.80, "north_star": 0.05, "comp": 0.05,
                "cultural_signals": 0.05, "red_flags": 0.05}
    high = weighted_score(dims, cv_heavy)
    default = weighted_score(dims)  # global default cv_match=0.30
    # 0.80*5 + 0.20*1 = 4.20 vs 0.30*5 + 0.70*1 = 2.20
    assert high == pytest.approx(4.20, abs=0.01)
    assert default == pytest.approx(2.20, abs=0.01)


def test_weighted_score_with_no_weights_uses_global_default():
    dims = {k: 5 for k in DIMENSION_KEYS}
    assert weighted_score(dims, None) == pytest.approx(5.0, abs=0.01)
    assert weighted_score(dims) == pytest.approx(5.0, abs=0.01)


def test_weighted_score_clamps_dim_values_in_1_to_5():
    """A 99 in cv_match should be clamped to 5."""
    dims = {"cv_match": 99, "north_star": 5, "comp": 5,
            "cultural_signals": 5, "red_flags": 5}
    assert weighted_score(dims) == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------
# End-to-end: same posting, different archetype, different score
# ---------------------------------------------------------------------

def test_per_archetype_weights_actually_change_global_score(
    monkeypatch, tmp_path,
):
    """A posting where the LLM returns identical dims but the detected
    archetype differs should produce a different global score, because
    each archetype weights the dims differently."""
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "PROFILE.yml").write_text(
        yaml.safe_dump({"location": {"willing_to_relocate": True}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PERSONA_DIR", persona)

    from skills import job_score
    from skills.job_score import JobScoreSkill, ScoredPosting

    # Stub LLM returns dims that ARE asymmetric across the dimensions
    # (cv_match=5, north_star=1) -- so weight differences matter.
    def fake_llm(inp, profile_summary, archetype, boost, trace_id, model_id,
                 client=None, region_adjust=0.0, state_ok=None, weights=None):
        # Simulate _score_blocking's real math path: build dims, then
        # call weighted_score with the weights the caller resolved.
        dims = {"cv_match": 5, "north_star": 1, "comp": 3,
                "cultural_signals": 3, "red_flags": 3}
        return ScoredPosting(
            title=inp.title, company=inp.company, location=inp.location,
            location_type=inp.location_type, seniority=inp.seniority or "mid",
            url=inp.url, archetype=archetype,
            score=__import__("core.archetypes",
                             fromlist=["weighted_score"]).weighted_score(
                                 dims, weights),
            dimensions=dims,
            recommendation="maybe", reasons=["test"],
        )
    monkeypatch.setattr(job_score, "_score_blocking", fake_llm)

    skill = JobScoreSkill()
    # RSM in title -> detect_archetype returns "Regional Sales Manager"
    # which weights north_star high; cv_match=5,north_star=1 should
    # score relatively low under RSM weights vs RevOps weights
    # (RevOps is cv_match-heavy at 0.40).
    rsm_inp = skill.validate_input({
        "title": "Regional Sales Manager Detroit", "company": "X",
        "location": "Detroit, MI", "location_type": "remote",
        "url": "http://1", "seniority": "senior",
    })
    revops_inp = skill.validate_input({
        "title": "RevOps Manager (Sales Operations)", "company": "X",
        "location": "Detroit, MI", "location_type": "remote",
        "url": "http://2", "seniority": "senior",
    })
    rsm = asyncio.run(skill.execute(rsm_inp, trace_id="SEN-test-arch1"))
    revops = asyncio.run(skill.execute(revops_inp, trace_id="SEN-test-arch2"))

    assert rsm.archetype == "Regional Sales Manager"
    assert revops.archetype == "Sales Operations / RevOps"
    # Same dims, different weights -> different scores.
    # RSM:    0.25*5 + 0.30*1 + 0.20*3 + 0.15*3 + 0.10*3 = 2.90
    # RevOps: 0.40*5 + 0.20*1 + 0.20*3 + 0.10*3 + 0.10*3 = 3.40
    assert revops.score > rsm.score, (
        f"RevOps weights (cv-heavy) should score higher when cv=5, "
        f"north_star=1: rsm={rsm.score} revops={revops.score}"
    )
    assert rsm.score == pytest.approx(2.90, abs=0.05)
    assert revops.score == pytest.approx(3.40, abs=0.05)
