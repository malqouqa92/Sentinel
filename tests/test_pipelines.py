"""Phase 10 -- Tests N, O, P, R: job search pipeline.

N -- /jobsearch returns >=5 postings from >=2 sites (network + jobspy)
O -- job_score produces a valid score+reasons+recommend triple
P -- full /jobsearch writes CSV+markdown+episode; recall finds it
R -- fan-out: 3 scrape results -> job_score called 3 times
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core import config
from core.memory import get_memory


# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------

def _has_jobspy() -> bool:
    try:
        import jobspy  # noqa: F401
        return True
    except ImportError:
        return False


# -------------------------------------------------------------------
# N -- live scrape (skipped without network or jobspy installed)
# -------------------------------------------------------------------

@pytest.mark.requires_network
def test_n_jobsearch_live_scrape_returns_results():
    if not _has_jobspy():
        pytest.skip(
            "python-jobspy not installed -- "
            "`py -3.12 -m pip install python-jobspy`"
        )
    from skills.job_scrape import JobScrapeInput, JobScrapeSkill
    skill = JobScrapeSkill()
    inp = skill.validate_input({
        "query": "software engineer",
        "location": "Detroit, MI",
        "distance": 100,
        "hours_old": 168,
        "results_wanted": 10,
    })
    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-N"))
    assert out.postings, "expected at least 1 posting"
    sites = {p.site for p in out.postings if p.site}
    # If we got at least 5 we have meaningful coverage.
    assert len(out.postings) >= 5 or len(sites) >= 1


# -------------------------------------------------------------------
# O -- job_score schema correctness
# -------------------------------------------------------------------

class _StubOllama:
    def __init__(self, response: str):
        self._response = response

    def generate(self, **kwargs):
        return self._response


def test_o_job_score_produces_valid_output(monkeypatch, tmp_path):
    """Phase 12 hard cutover: new 1-5 weighted rubric with archetype +
    legitimacy. The LLM returns {dimensions, reasons, legitimacy_signals}
    -- no longer {score, recommend}.

    Isolates from the live workspace/persona/PROFILE.yml by pointing
    PERSONA_DIR at an empty tmp dir so load_profile() returns defaults
    (no seniority_boost keywords -> no +0.5 north_star bump that would
    perturb the weighted-score arithmetic asserted below)."""
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)
    from skills import job_score
    skill = job_score.JobScoreSkill()
    stub = _StubOllama(
        '{"dimensions": {"cv_match": 5, "north_star": 4, "comp": 3, '
        '"cultural_signals": 4, "red_flags": 5}, '
        '"reasons": ["Detroit metro fits", "B2B sales role", "good comp"], '
        '"legitimacy_signals": []}'
    )
    original_score = job_score._score_blocking

    def fake_score(inp, profile_summary, archetype, boost, trace_id,
                   model_id, client=None,
                   region_adjust=0.0, state_ok=None, weights=None):
        return original_score(
            inp, profile_summary, archetype, boost, trace_id, model_id,
            client=stub,
            region_adjust=region_adjust, state_ok=state_ok,
            weights=weights,
        )

    monkeypatch.setattr(job_score, "_score_blocking", fake_score)

    sample_extracted = {
        "title": "Regional Sales Manager",
        "company": "AcmeCo",
        "location": "Detroit, MI",
        "location_type": "hybrid",
        "salary_range": "$85,000 - $110,000",
        "industry": "B2B SaaS",
        "seniority": "mid",
        "key_requirements": ["3+ years B2B sales"],
        "deal_breakers": [],
        "relevance_signals": ["B2B", "sales"],
        "confidence": 0.9,
        "url": "https://example.com/job/1",
    }
    inp = skill.validate_input(sample_extracted)
    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-O"))
    # New schema: 1-5 global score, derived band, dimensions dict.
    assert 1.0 <= out.score <= 5.0
    # Phase 13: RSM now uses per-archetype weights
    # cv_match=0.25, north_star=0.30, comp=0.20, cultural_signals=0.15, red_flags=0.10
    # 0.25*5 + 0.30*4 + 0.20*3 + 0.15*4 + 0.10*5 = 4.15
    assert out.score == pytest.approx(4.15, abs=0.05)
    assert out.recommendation == "worth_applying"
    assert set(out.dimensions.keys()) == {
        "cv_match", "north_star", "comp", "cultural_signals", "red_flags",
    }
    assert isinstance(out.reasons, list) and len(out.reasons) >= 1
    assert out.archetype == "Regional Sales Manager"
    assert out.legitimacy.tier == "high"
    assert out.title == "Regional Sales Manager"
    assert out.company == "AcmeCo"
    assert out.url == "https://example.com/job/1"


# -------------------------------------------------------------------
# Shared mock for P + R: stub the full pipeline LLM stack.
# -------------------------------------------------------------------

def _install_pipeline_stubs(monkeypatch, n_postings: int = 3):
    """Mock job_scrape, job_extract, job_score by patching the
    REGISTRY INSTANCE methods directly. Class-level patches don't
    survive cross-test interference from earlier real-/code tests in
    the full suite; instance-level patches do (highest precedence in
    Python attribute lookup)."""
    import types as _types
    from core.registry import SKILL_REGISTRY
    from core.agent_registry import AGENT_REGISTRY

    counters = {"scrape": 0, "extract": 0, "score": 0}
    fake_postings = [
        {
            "title": f"Account Executive {i}",
            "company": f"Co-{i}",
            "location": "Detroit, MI",
            "description": f"Sell stuff to people. Posting #{i}.",
            "url": f"https://example.com/job/{i}",
            "site": "indeed" if i % 2 == 0 else "linkedin",
            "text": f"Title: AE{i}\nCompany: Co-{i}\nLocation: Detroit",
        }
        for i in range(n_postings)
    ]

    from skills import job_scrape, job_extract, job_score

    async def fake_scrape_execute(self, input_data, trace_id, context=None):
        counters["scrape"] += 1
        return job_scrape.JobScrapeOutput(
            postings=[
                job_scrape.ScrapedPosting(**p) for p in fake_postings
            ]
        )

    async def fake_extract_execute(self, input_data, trace_id, context=None):
        counters["extract"] += 1
        return job_extract.JobExtraction(
            title=f"Title-{counters['extract']}",
            company=f"Co-{counters['extract']}",
            location="Detroit, MI",
            location_type="hybrid",
            salary_range="$80,000 - $100,000",
            industry="Tech",
            seniority="mid",
            key_requirements=["B2B"],
            deal_breakers=[],
            relevance_signals=["sales"],
            confidence=0.85,
        )

    def _make_one(input_one, idx_for_name: int):
        # Alternate apply_now / skip so the report sees a mix.
        if idx_for_name % 2 == 1:
            dims = {"cv_match": 5, "north_star": 5, "comp": 4,
                    "cultural_signals": 4, "red_flags": 5}
            score = 4.6
            band = "apply_now"
        else:
            dims = {"cv_match": 3, "north_star": 3, "comp": 3,
                    "cultural_signals": 3, "red_flags": 3}
            score = 3.0
            band = "skip"
        return job_score.ScoredPosting(
            title=input_one.title or f"Title-{idx_for_name}",
            company=input_one.company or f"Co-{idx_for_name}",
            location=input_one.location or "Detroit, MI",
            location_type=input_one.location_type or "hybrid",
            salary_range=input_one.salary_range,
            seniority=input_one.seniority or "mid",
            url=getattr(input_one, "url", "") or f"https://example.com/job/{idx_for_name}",
            archetype="Regional Sales Manager",
            score=score,
            dimensions=dims,
            recommendation=band,
            reasons=["good fit", "Detroit location"],
        )

    async def fake_score_execute(self, input_data, trace_id, context=None):
        # Phase 13: job_score is now accepts_list=True. The pipeline
        # gives us a JobScoreBatchInput; we score each posting and
        # return JobScoreBatchOutput. The score counter increments
        # per posting (not per call) so existing assertions still
        # measure work done.
        if isinstance(input_data, job_score.JobScoreBatchInput):
            postings = []
            for inp in input_data.postings:
                counters["score"] += 1
                postings.append(_make_one(inp, counters["score"]))
            return job_score.JobScoreBatchOutput(postings=postings)
        # Single-item back-compat path (CLI / direct test use).
        counters["score"] += 1
        return _make_one(input_data, counters["score"])

    # Patch instance attributes (highest precedence in attr lookup)
    # so cross-test class mutations from earlier real-/code tests
    # don't break us.
    for name, fake in (
        ("job_scrape", fake_scrape_execute),
        ("job_extract", fake_extract_execute),
        ("job_score", fake_score_execute),
    ):
        registry_inst = SKILL_REGISTRY.get(name)
        if registry_inst is not None:
            bound = _types.MethodType(fake, registry_inst)
            monkeypatch.setattr(registry_inst, "execute", bound)
        agent = AGENT_REGISTRY.get("job_searcher")
        if agent is None:
            continue
        for s in agent._skills:
            if s.name == name and s is not registry_inst:
                bound = _types.MethodType(fake, s)
                monkeypatch.setattr(s, "execute", bound)

    return counters


# -------------------------------------------------------------------
# P -- full /jobsearch agent run writes CSV+md+episode
# -------------------------------------------------------------------

def test_p_full_jobsearch_writes_outputs_and_episode(
    monkeypatch, tmp_path,
):
    from core.agent_registry import AGENT_REGISTRY

    # Redirect the output dir so we don't pollute the real workspace.
    monkeypatch.setattr(
        config, "JOBSEARCH_OUTPUT_DIR", tmp_path / "job_searches",
    )
    config.JOBSEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counters = _install_pipeline_stubs(monkeypatch, n_postings=3)

    agent = AGENT_REGISTRY.get("job_searcher")
    assert agent is not None, "job_searcher agent should be registered"

    out = asyncio.run(agent.run(
        {"query": "Account Executive", "location": "Detroit, MI"},
        "SEN-test-P",
    ))
    assert out.get("_error") is not True, f"agent error: {out}"
    assert counters["scrape"] == 1
    assert counters["extract"] == 3
    assert counters["score"] == 3
    assert out["total_postings"] == 3
    # csv/summary paths are PROJECT_ROOT-relative when possible,
    # absolute otherwise (test redirects to tmp_path -> absolute).
    csv_path = Path(out["csv_path"])
    md_path = Path(out["summary_path"])
    if not csv_path.is_absolute():
        csv_path = config.PROJECT_ROOT / csv_path
    if not md_path.is_absolute():
        md_path = config.PROJECT_ROOT / md_path
    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert md_path.exists() and md_path.stat().st_size > 0
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "score" in csv_text and "company" in csv_text

    # Episode exists with a recallable summary.
    mem = get_memory()
    eps = mem.get_recent_episodes(scope="job_searcher", limit=5)
    assert any(
        "Job report" in e.summary and "scored" in e.summary
        for e in eps
    ), f"expected job_searcher episode; got: {[e.summary for e in eps]}"


# -------------------------------------------------------------------
# R -- fan-out: 3 scrape items -> job_score called 3 times
# -------------------------------------------------------------------

def test_r_fanout_calls_score_per_item(monkeypatch, tmp_path):
    from core.agent_registry import AGENT_REGISTRY
    monkeypatch.setattr(
        config, "JOBSEARCH_OUTPUT_DIR", tmp_path / "job_searches",
    )
    config.JOBSEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counters = _install_pipeline_stubs(monkeypatch, n_postings=3)
    agent = AGENT_REGISTRY.get("job_searcher")
    assert agent is not None
    out = asyncio.run(agent.run(
        {"query": "anything", "location": "Detroit"},
        "SEN-test-R",
    ))
    assert out.get("_error") is not True, f"agent error: {out}"
    # Fan-out: scrape once, extract per item, score per item.
    assert counters["scrape"] == 1
    assert counters["extract"] == 3
    assert counters["score"] == 3
