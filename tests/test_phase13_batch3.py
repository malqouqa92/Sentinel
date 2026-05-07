"""Phase 13 Batch 3 -- LLM batching for job_score.

ECC. Mock OllamaClient.generate so we can assert call shape +
return-payload handling. No network, no GPU.

What we're verifying:
  - batch path: N postings -> 1 LLM call returning a results array
  - chunking: N > batch_size -> ceil(N / batch_size) calls
  - partial parse failure: one bad item in the array -> per-item
    fallback for THAT item only; other items use the batch result
  - whole-batch parse failure: results missing / wrong type / wrong
    length -> per-item fallback for the entire batch
  - LLM call failure (LLMError raised): per-item fallback
  - commute-gated postings short-circuit BEFORE the LLM and don't
    count toward batch size
  - single-item back-compat: scoring one JobScoreInput still works
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.llm import LLMError
from skills import job_score
from skills.job_score import (
    JobScoreInput, JobScoreBatchInput, JobScoreBatchOutput,
    JobScoreSkill, ScoredPosting,
)


def _empty_persona(monkeypatch, tmp_path):
    """Point PERSONA_DIR at an empty tmp dir so load_profile() returns
    defaults -- no commute gate, no seniority boost, no region/state
    nudges -- so we can assert batch math directly."""
    monkeypatch.setattr(config, "PERSONA_DIR", tmp_path)


def _stub_inputs(n: int) -> list[dict]:
    return [
        {
            "title": f"Regional Sales Manager {i}",
            "company": f"Co-{i}",
            "location": "Detroit, MI",
            "location_type": "remote",  # avoid commute gate
            "salary_range": "$110,000-$150,000",
            "industry": "B2B SaaS",
            "seniority": "senior",
            "key_requirements": ["B2B sales"],
            "deal_breakers": [],
            "relevance_signals": ["sales"],
            "url": f"https://example.com/job/{i}",
        }
        for i in range(n)
    ]


def _good_batch_response(n: int) -> str:
    """A JSON payload with N well-formed result entries."""
    return json.dumps({
        "results": [
            {
                "dimensions": {
                    "cv_match": 5, "north_star": 4, "comp": 3,
                    "cultural_signals": 4, "red_flags": 5,
                },
                "reasons": [f"reason for posting {i}"],
                "legitimacy_signals": [],
            }
            for i in range(n)
        ]
    })


class _StubLLM:
    """Captures generate() calls; replays a queue of responses."""

    def __init__(self, responses: list[str | Exception]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("more LLM calls than responses queued")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


# ---------------------------------------------------------------------
# Batch path -- happy path
# ---------------------------------------------------------------------

def test_batch_n_postings_uses_one_llm_call(monkeypatch, tmp_path):
    """5 postings + batch_size=5 -> 1 LLM call, 5 ScoredPostings out."""
    _empty_persona(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "JOB_SCORE_BATCH_SIZE", 5)

    stub = _StubLLM([_good_batch_response(5)])
    monkeypatch.setattr(
        "skills.job_score.OllamaClient", lambda: stub,
    )

    skill = JobScoreSkill()
    inp = skill.validate_input(_stub_inputs(5))
    assert isinstance(inp, JobScoreBatchInput)
    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-batch-5"))

    assert isinstance(out, JobScoreBatchOutput)
    assert len(out.postings) == 5
    assert len(stub.calls) == 1
    # Every call gets the BATCH system prompt, not the single one.
    assert "score all" in stub.calls[0]["prompt"].lower()
    assert "[posting_1]" in stub.calls[0]["prompt"]
    assert "[posting_5]" in stub.calls[0]["prompt"]


def test_batch_chunks_when_n_exceeds_batch_size(monkeypatch, tmp_path):
    """12 postings, batch_size=5 -> 3 chunks (5+5+2) -> 3 LLM calls."""
    _empty_persona(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "JOB_SCORE_BATCH_SIZE", 5)

    stub = _StubLLM([
        _good_batch_response(5),
        _good_batch_response(5),
        _good_batch_response(2),
    ])
    monkeypatch.setattr(
        "skills.job_score.OllamaClient", lambda: stub,
    )

    skill = JobScoreSkill()
    inp = skill.validate_input(_stub_inputs(12))
    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-batch-12"))

    assert len(out.postings) == 12
    assert len(stub.calls) == 3
    # Last chunk only had 2 postings -- prompt should reflect that.
    assert "[posting_2]" in stub.calls[2]["prompt"]
    assert "[posting_3]" not in stub.calls[2]["prompt"]


def test_batch_preserves_per_item_url_and_title(monkeypatch, tmp_path):
    """The N output ScoredPostings must carry their input title/url
    in the same order -- this is the contract the report relies on."""
    _empty_persona(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "JOB_SCORE_BATCH_SIZE", 10)

    stub = _StubLLM([_good_batch_response(3)])
    monkeypatch.setattr(
        "skills.job_score.OllamaClient", lambda: stub,
    )
    skill = JobScoreSkill()
    inp = skill.validate_input(_stub_inputs(3))
    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-order"))

    for i, scored in enumerate(out.postings):
        assert scored.url == f"https://example.com/job/{i}", (
            f"row {i}: url mismatch -- got {scored.url!r}"
        )
        assert f"{i}" in scored.title


# ---------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------

def test_batch_whole_payload_unparseable_falls_back_per_item(
    monkeypatch, tmp_path,
):
    """Garbage JSON -> per-item LLM call for each posting in the batch.
    With 3 postings: 1 batch attempt + 3 per-item retries = 4 total."""
    _empty_persona(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "JOB_SCORE_BATCH_SIZE", 5)

    single_ok = json.dumps({
        "dimensions": {"cv_match": 4, "north_star": 4, "comp": 3,
                       "cultural_signals": 3, "red_flags": 5},
        "reasons": ["per-item fallback"],
        "legitimacy_signals": [],
    })
    stub = _StubLLM(["not valid json", single_ok, single_ok, single_ok])
    monkeypatch.setattr(
        "skills.job_score.OllamaClient", lambda: stub,
    )

    skill = JobScoreSkill()
    out = asyncio.run(skill.execute(
        skill.validate_input(_stub_inputs(3)),
        trace_id="SEN-test-bad-batch",
    ))
    assert len(out.postings) == 3
    assert len(stub.calls) == 4
    # Per-item retries use the SINGLE-item system prompt
    # (different from BATCH_SYSTEM_PROMPT).
    assert "in input order" in stub.calls[0]["system"].lower()  # batch
    assert "in input order" not in stub.calls[1]["system"].lower()  # single


def test_batch_results_wrong_length_falls_back_per_item(
    monkeypatch, tmp_path,
):
    """Batch returned results=[only 2 items] for n=3 -> per-item fallback."""
    _empty_persona(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "JOB_SCORE_BATCH_SIZE", 5)

    short_payload = json.dumps({
        "results": [
            {"dimensions": {"cv_match": 5, "north_star": 5, "comp": 3,
                            "cultural_signals": 3, "red_flags": 5},
             "reasons": ["x"], "legitimacy_signals": []},
            {"dimensions": {"cv_match": 5, "north_star": 5, "comp": 3,
                            "cultural_signals": 3, "red_flags": 5},
             "reasons": ["x"], "legitimacy_signals": []},
        ]
    })
    single_ok = json.dumps({
        "dimensions": {"cv_match": 3, "north_star": 3, "comp": 3,
                       "cultural_signals": 3, "red_flags": 3},
        "reasons": ["per-item"], "legitimacy_signals": [],
    })
    stub = _StubLLM([short_payload, single_ok, single_ok, single_ok])
    monkeypatch.setattr(
        "skills.job_score.OllamaClient", lambda: stub,
    )

    skill = JobScoreSkill()
    out = asyncio.run(skill.execute(
        skill.validate_input(_stub_inputs(3)),
        trace_id="SEN-test-short",
    ))
    assert len(out.postings) == 3
    assert len(stub.calls) == 4


def test_batch_one_bad_item_uses_per_item_for_that_one_only(
    monkeypatch, tmp_path,
):
    """Batch returned 3 results, but result[1] is malformed (not a dict).
    -> 1 batch call + 1 per-item retry for index 1 = 2 calls total.
    Other 2 postings use the batch result directly."""
    _empty_persona(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "JOB_SCORE_BATCH_SIZE", 5)

    mixed_payload = json.dumps({
        "results": [
            {"dimensions": {"cv_match": 5, "north_star": 5, "comp": 3,
                            "cultural_signals": 3, "red_flags": 5},
             "reasons": ["good"], "legitimacy_signals": []},
            "garbage string instead of dict",
            {"dimensions": {"cv_match": 4, "north_star": 4, "comp": 4,
                            "cultural_signals": 4, "red_flags": 4},
             "reasons": ["also good"], "legitimacy_signals": []},
        ]
    })
    single_ok = json.dumps({
        "dimensions": {"cv_match": 3, "north_star": 3, "comp": 3,
                       "cultural_signals": 3, "red_flags": 3},
        "reasons": ["per-item retry"], "legitimacy_signals": [],
    })
    stub = _StubLLM([mixed_payload, single_ok])
    monkeypatch.setattr(
        "skills.job_score.OllamaClient", lambda: stub,
    )

    skill = JobScoreSkill()
    out = asyncio.run(skill.execute(
        skill.validate_input(_stub_inputs(3)),
        trace_id="SEN-test-mixed",
    ))
    assert len(out.postings) == 3
    assert len(stub.calls) == 2
    assert out.postings[1].reasons == ["per-item retry"]
    assert out.postings[0].reasons == ["good"]
    assert out.postings[2].reasons == ["also good"]


def test_batch_llm_raises_falls_back_per_item(monkeypatch, tmp_path):
    """Ollama raises LLMError on the batch call -> per-item fallback."""
    _empty_persona(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "JOB_SCORE_BATCH_SIZE", 5)

    single_ok = json.dumps({
        "dimensions": {"cv_match": 3, "north_star": 3, "comp": 3,
                       "cultural_signals": 3, "red_flags": 3},
        "reasons": ["per-item ok"], "legitimacy_signals": [],
    })
    stub = _StubLLM([
        LLMError("simulated ollama crash"),
        single_ok, single_ok,
    ])
    monkeypatch.setattr(
        "skills.job_score.OllamaClient", lambda: stub,
    )

    skill = JobScoreSkill()
    out = asyncio.run(skill.execute(
        skill.validate_input(_stub_inputs(2)),
        trace_id="SEN-test-llm-crash",
    ))
    assert len(out.postings) == 2
    assert len(stub.calls) == 3  # 1 batch + 2 per-item


# ---------------------------------------------------------------------
# Commute gate interaction
# ---------------------------------------------------------------------

def test_commute_gated_postings_skip_llm_in_batch_mode(
    monkeypatch, tmp_path,
):
    """Profile says 20-mile cap from 48125, will not relocate. Mix of
    one local-remote (LLM-scored) + one onsite-Cincinnati (gated).
    -> only 1 LLM call (for the 1 ungated posting), 2 ScoredPostings out."""
    persona = tmp_path / "persona"
    persona.mkdir()
    (persona / "PROFILE.yml").write_text(yaml.safe_dump({
        "location": {
            "primary_zip": "48125", "onsite_max_miles": 20,
            "willing_to_relocate": False,
        },
    }), encoding="utf-8")
    monkeypatch.setattr(config, "PERSONA_DIR", persona)
    monkeypatch.setattr(config, "JOB_SCORE_BATCH_SIZE", 5)

    stub = _StubLLM([_good_batch_response(1)])
    monkeypatch.setattr(
        "skills.job_score.OllamaClient", lambda: stub,
    )

    inputs = [
        {"title": "RSM Detroit Remote", "company": "X",
         "location": "(unknown)",  # remote, no commute gate
         "location_type": "remote",
         "url": "http://1", "seniority": "senior"},
        {"title": "RSM Cincinnati", "company": "Y",
         "location": "Cincinnati, OH, US",  # 200+ miles, gated
         "location_type": "onsite",
         "url": "http://2", "seniority": "senior"},
    ]
    skill = JobScoreSkill()
    out = asyncio.run(skill.execute(
        skill.validate_input(inputs),
        trace_id="SEN-test-mixed-gate",
    ))
    assert len(out.postings) == 2
    assert len(stub.calls) == 1, (
        f"only the ungated posting should hit the LLM; got {len(stub.calls)} "
        "calls"
    )
    # Order preserved: gated posting still in slot 1 (skip band).
    assert out.postings[1].recommendation == "skip"
    assert any("commute" in r.lower() for r in out.postings[1].reasons), (
        out.postings[1].reasons
    )


# ---------------------------------------------------------------------
# Single-item back-compat
# ---------------------------------------------------------------------

def test_single_item_input_still_works(monkeypatch, tmp_path):
    """Passing a single JobScoreInput (CLI / direct test path) must
    still produce ONE ScoredPosting (not a batch wrapper)."""
    _empty_persona(monkeypatch, tmp_path)

    single_ok = json.dumps({
        "dimensions": {"cv_match": 5, "north_star": 5, "comp": 3,
                       "cultural_signals": 3, "red_flags": 5},
        "reasons": ["singleton"], "legitimacy_signals": [],
    })
    stub = _StubLLM([single_ok])
    monkeypatch.setattr(
        "skills.job_score.OllamaClient", lambda: stub,
    )

    skill = JobScoreSkill()
    inp = skill.validate_input(_stub_inputs(1)[0])  # one dict
    assert isinstance(inp, JobScoreInput)  # NOT JobScoreBatchInput
    out = asyncio.run(skill.execute(inp, trace_id="SEN-test-single"))
    assert isinstance(out, ScoredPosting)
    assert out.reasons == ["singleton"]
    assert len(stub.calls) == 1
    # Single path uses the SINGLE system prompt, not batch.
    assert "score all" not in stub.calls[0]["system"].lower()


# ---------------------------------------------------------------------
# Skill class semantics
# ---------------------------------------------------------------------

def test_skill_declares_accepts_list_and_output_is_list():
    """The agent runtime checks these flags to avoid fan-out and to
    unwrap the list. A regression here breaks the whole pipeline."""
    skill = JobScoreSkill()
    assert skill.accepts_list is True
    assert skill.output_is_list is True


def test_validate_input_dispatches_on_shape():
    skill = JobScoreSkill()
    # list[dict] -> JobScoreBatchInput
    batch = skill.validate_input(_stub_inputs(2))
    assert isinstance(batch, JobScoreBatchInput)
    assert len(batch.postings) == 2
    # dict -> JobScoreInput
    single = skill.validate_input(_stub_inputs(1)[0])
    assert isinstance(single, JobScoreInput)
    # already a JobScoreInput -> returned as-is
    again = skill.validate_input(single)
    assert again is single
