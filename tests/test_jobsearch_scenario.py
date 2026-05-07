"""Phase 12 -- end-to-end "Regional Sales Manager" scenario.

Custom integration test the user explicitly asked for. Runs the FULL
job_searcher agent pipeline end-to-end against REAL python-jobspy
(Indeed only, tiny sample) with the LLM steps mocked so it stays
deterministic + fast.

Verifies:
  - real jobspy returns postings for "Regional Sales Manager" in
    Detroit-MI within the last week
  - title-filter pre-pass keeps the matches
  - extract + score (mocked) produce ScoredPosting rows
  - job_report writes CSV + summary.md
  - applications table receives new rows (one per posting with a URL)
  - top-3 Telegram message is non-empty and includes apply URLs

Marked `requires_network` so a sandboxed CI without outbound access
auto-skips. Marked `slow` because the real Indeed call adds ~10-20s.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types as _types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, database
from core.agent_registry import AGENT_REGISTRY
from core.registry import SKILL_REGISTRY
from skills import job_extract, job_score


@pytest.mark.requires_network
@pytest.mark.slow
def test_regional_sales_manager_scenario(monkeypatch, tmp_path):
    """End-to-end: scrape (real) -> filter -> extract (mock) -> score
    (mock) -> report -> applications. Owner's canonical scenario."""
    # Redirect output so the real workspace stays clean.
    monkeypatch.setattr(
        config, "JOBSEARCH_OUTPUT_DIR", tmp_path / "job_searches",
    )
    config.JOBSEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Seed a PROFILE.yml that recognizes Regional Sales Manager so the
    # title filter actually fires (default empty profile lets all
    # through, which would still pass but doesn't exercise the filter).
    profile_yaml = """
candidate:
  full_name: "Test"
  location: "Detroit, MI"
target_roles:
  primary:
    - "Regional Sales Manager"
title_filter:
  positive:
    - "sales"
    - "regional"
    - "account"
  negative:
    - "intern"
    - "engineer"
    - "developer"
location:
  workplace_preference: "all"
"""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "PROFILE.yml").write_text(profile_yaml, encoding="utf-8")
    monkeypatch.setattr(config, "PERSONA_DIR", persona_dir)

    # Mock job_extract so each scraped posting gets a synthetic
    # extraction without hitting Qwen. Carry through url + title +
    # company + location so the downstream scorer + applications
    # writer have real data.
    counters = {"extract": 0, "score": 0}

    async def fake_extract_execute(self, input_data, trace_id, context=None):
        counters["extract"] += 1
        # input_data is a JobExtractInput with .text containing the
        # scrape's combined text. We pull the title/company/location
        # back out of the text header lines for fidelity.
        lines = (input_data.text or "").splitlines()
        title = next((l[7:].strip() for l in lines
                      if l.startswith("Title: ")), "Sales Position")
        company = next((l[9:].strip() for l in lines
                        if l.startswith("Company: ")), "SomeCo")
        location = next((l[10:].strip() for l in lines
                         if l.startswith("Location: ")), "Detroit, MI")
        url = next((l[5:].strip() for l in lines
                    if l.startswith("URL: ")), "")
        return job_extract.JobExtraction(
            title=title, company=company, location=location,
            location_type="hybrid",
            salary_range="$100,000 - $130,000",
            industry="B2B",
            seniority="senior",
            key_requirements=["B2B sales", "5+ years"],
            deal_breakers=[],
            relevance_signals=["sales", "regional"],
            confidence=0.85,
            # Phase 12 carry-throughs (extension fields):
            url=url,
        )

    def _make_one_score(input_one, idx_for_name: int):
        if idx_for_name % 2 == 1:
            dims = {"cv_match": 5, "north_star": 5, "comp": 4,
                    "cultural_signals": 4, "red_flags": 5}
            score = 4.6
            band = "apply_now"
        else:
            dims = {"cv_match": 4, "north_star": 4, "comp": 4,
                    "cultural_signals": 4, "red_flags": 4}
            score = 4.0
            band = "worth_applying"
        return job_score.ScoredPosting(
            title=input_one.title or f"Job-{idx_for_name}",
            company=input_one.company or f"Co-{idx_for_name}",
            location=input_one.location or "Detroit, MI",
            location_type=input_one.location_type or "hybrid",
            salary_range=input_one.salary_range,
            seniority=input_one.seniority or "senior",
            url=getattr(input_one, "url", "") or "",
            archetype="Regional Sales Manager",
            score=score,
            dimensions=dims,
            recommendation=band,
            reasons=["good fit", "Detroit metro"],
        )

    async def fake_score_execute(self, input_data, trace_id, context=None):
        # Phase 13: job_score is now accepts_list=True. Counters track
        # postings (not calls) so existing assertions still measure work.
        if isinstance(input_data, job_score.JobScoreBatchInput):
            postings = []
            for inp in input_data.postings:
                counters["score"] += 1
                postings.append(_make_one_score(inp, counters["score"]))
            return job_score.JobScoreBatchOutput(postings=postings)
        counters["score"] += 1
        return _make_one_score(input_data, counters["score"])

    # Patch instance attrs (matches the pattern used in test_pipelines).
    SKILL_REGISTRY.discover()
    AGENT_REGISTRY.discover()
    for name, fake in (
        ("job_extract", fake_extract_execute),
        ("job_score", fake_score_execute),
    ):
        registry_inst = SKILL_REGISTRY.get(name)
        if registry_inst is not None:
            monkeypatch.setattr(
                registry_inst, "execute",
                _types.MethodType(fake, registry_inst),
            )
        agent = AGENT_REGISTRY.get("job_searcher")
        if agent is None:
            continue
        for s in agent._skills:
            if s.name == name and s is not registry_inst:
                monkeypatch.setattr(
                    s, "execute", _types.MethodType(fake, s),
                )

    # ---- The owner's canonical query ----
    agent = AGENT_REGISTRY.get("job_searcher")
    assert agent is not None
    out = asyncio.run(agent.run(
        {
            "query": "Regional Sales Manager",
            "location": "Detroit, Michigan",
            "sites": ["indeed"],         # one board to keep the test fast
            "distance": 50,
            "hours_old": 168,            # last week
            "results_wanted": 5,         # tiny sample
            "workplace": "all",
        },
        "SEN-test-RSM-scenario",
    ))

    # If jobspy returned 0 postings (bad rate-limit day, network glitch),
    # the test should skip rather than fail loudly -- we're testing the
    # PIPELINE, not Indeed's uptime.
    if out.get("_error") or out.get("total_postings", 0) == 0:
        pytest.skip(
            f"jobspy returned no postings for 'Regional Sales Manager' "
            f"in Detroit; can't exercise the pipeline. raw={out!r}"
        )

    # Pipeline output sanity
    assert out["total_postings"] >= 1
    assert counters["extract"] == out["total_postings"]
    assert counters["score"] == out["total_postings"]
    # Band counts add up to total
    assert (
        out["apply_now_count"] + out["worth_applying_count"]
        + out["maybe_count"] + out["skip_count"]
    ) == out["total_postings"]
    # CSV + summary written
    csv_path = Path(out["csv_path"])
    md_path = Path(out["summary_path"])
    if not csv_path.is_absolute():
        csv_path = config.PROJECT_ROOT / csv_path
    if not md_path.is_absolute():
        md_path = config.PROJECT_ROOT / md_path
    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert md_path.exists() and md_path.stat().st_size > 0
    # Applications table populated
    apps = database.list_applications()
    # At least 1 (some real postings may have empty URLs that we skip).
    assert len(apps) >= 1, "expected at least one application written"
    for a in apps:
        assert a["state"] == "evaluated"
        assert a["archetype"] == "Regional Sales Manager"
        assert a["score"] is not None
        assert a["url"]
    # Top-3 Telegram message is non-empty + includes apply URL
    top_msg = out.get("top_n_telegram") or ""
    assert top_msg, f"top_n_telegram should be non-empty, got: {out!r}"
    # Either one of the synthetic companies OR a real Indeed company
    # name shows up; the URL field is the safer assertion.
    assert "https://" in top_msg or "http://" in top_msg, top_msg
