import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import database, worker
from core.router import route
from skills import job_extract  # noqa: F401  (kept for symmetry)
from skills.job_extract import (
    ALLOWED_LOCATION_TYPES,
    ALLOWED_SENIORITY,
    JobExtraction,
    extract,
)


SAMPLE_JOB = """\
Regional Sales Manager - Midwest Territory
Acme Industrial Solutions | Chicago, IL (Hybrid - 2 days/week in office)

We're looking for an experienced B2B sales leader to own our Midwest territory
($5M annual quota). You'll manage a team of 4 AEs and drive growth across
manufacturing and distribution verticals.

Requirements:
- 7+ years B2B sales experience, 3+ in management
- Experience selling into manufacturing or industrial accounts
- CRM proficiency (Salesforce preferred)
- Willingness to travel 40% within territory

Compensation: $120K-$145K base + commission
"""


MESSY_JOB = """\
SR ACCT MGR -- enterprise SaaS // remote (US only)
quota $1.8m ARR, mgmt 2 SDRs, MEDDIC required
must have 5+ yrs closing exp w/ $50k+ deals, salesforce + outreach.io a +
no comp listed. travel ~10%
"""


VALID_JSON_RESPONSE = json.dumps({
    "title": "Regional Sales Manager",
    "company": "Acme Industrial Solutions",
    "location": "Chicago, IL",
    "location_type": "hybrid",
    "salary_range": "$120K-$145K",
    "industry": "Industrial",
    "seniority": "senior",
    "key_requirements": [
        "7+ years B2B sales experience",
        "3+ years in management",
        "Manufacturing/industrial sales experience",
        "Salesforce proficiency",
        "40% travel within territory",
    ],
    "deal_breakers": ["40% travel"],
    "relevance_signals": [
        "B2B sales leader", "RSM", "manage team of 4 AEs",
    ],
    "confidence": 0.85,
})


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, model, prompt, system=None, temperature=None,
                 timeout=None, format_json=False, trace_id="SEN-test"):
        self.calls.append({
            "model": model, "prompt": prompt, "system": system,
            "format_json": format_json,
        })
        if not self.responses:
            raise RuntimeError("FakeClient: no more canned responses")
        return self.responses.pop(0)


def test_g_parse_failure_recovery():
    """First attempt returns invalid JSON; second attempt returns valid.
    Retry fires; result is the valid extraction."""
    client = FakeClient([
        "Sure! Here's the extraction:\n```\nthis is not json at all\n```",
        VALID_JSON_RESPONSE,
    ])
    result = extract(SAMPLE_JOB, trace_id="SEN-test-G", client=client)
    assert isinstance(result, JobExtraction)
    assert result.title == "Regional Sales Manager"
    assert result.company == "Acme Industrial Solutions"
    assert result.location_type == "hybrid"
    assert result.seniority == "senior"
    assert result.confidence == 0.85
    assert len(client.calls) == 2  # retry actually fired


def test_h_total_failure_returns_empty():
    """Both attempts return garbage. Returns empty JobExtraction with
    confidence=0.0 (not raises)."""
    client = FakeClient([
        "garbage output not json",
        "still garbage on retry too",
    ])
    result = extract(SAMPLE_JOB, trace_id="SEN-test-H", client=client)
    assert isinstance(result, JobExtraction)
    assert result.confidence == 0.0
    assert result.title == ""
    assert result.location_type in ALLOWED_LOCATION_TYPES
    assert result.seniority in ALLOWED_SENIORITY
    assert len(client.calls) == 2  # both attempts made


def test_validator_snaps_capitalization():
    """LLM might return 'Senior' or 'Hybrid' — validators normalize."""
    client = FakeClient([
        json.dumps({
            "title": "Test", "company": "X", "location": "Y",
            "location_type": "Hybrid",
            "salary_range": None, "industry": None,
            "seniority": "Senior",
            "key_requirements": ["a"] * 10,  # over the cap
            "deal_breakers": [], "relevance_signals": [],
            "confidence": 1.5,  # over the clamp
        }),
    ])
    result = extract(SAMPLE_JOB, trace_id="SEN-test-norm", client=client)
    assert result.location_type == "hybrid"
    assert result.seniority == "senior"
    assert len(result.key_requirements) == 5  # capped
    assert result.confidence == 1.0  # clamped


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_e_clean_extraction():
    """Real LLM call on a clean job posting. Lenient assertions: shape +
    membership, not exact strings."""
    result = extract(SAMPLE_JOB, trace_id="SEN-test-E")
    assert isinstance(result, JobExtraction)
    assert result.title.strip() != ""
    assert result.company.strip() != ""
    assert result.location.strip() != ""
    assert result.location_type in ALLOWED_LOCATION_TYPES
    assert result.seniority in ALLOWED_SENIORITY
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence > 0.5
    assert len(result.key_requirements) <= 5


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_f_messy_input():
    """Real LLM call on a messy posting with no salary. salary_range
    should be None; required fields populated."""
    result = extract(MESSY_JOB, trace_id="SEN-test-F")
    assert isinstance(result, JobExtraction)
    assert result.salary_range is None
    assert result.title.strip() != ""
    assert result.location_type in ALLOWED_LOCATION_TYPES
    assert result.seniority in ALLOWED_SENIORITY


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_i_full_pipeline():
    """End-to-end: route → enqueue → worker → real LLM → DB.
    Uses the bonus '/extract' command registered at module import.
    Asserts task completes with valid JobExtraction stored as result;
    GPU lock is released after."""
    raw = SAMPLE_JOB.replace("\n", " ")  # router collapses whitespace
    result = route(f"/extract {raw}")
    assert result.status == "ok"
    assert result.task_id is not None

    async def drain():
        shutdown = asyncio.Event()
        await asyncio.wait_for(
            worker.worker_loop(shutdown, idle_exit=True), timeout=180.0
        )

    asyncio.run(drain())

    task = database.get_task(result.task_id)
    assert task is not None
    assert task.status == "completed", \
        f"task did not complete: status={task.status} error={task.error}"
    extracted = JobExtraction(**task.result)
    assert extracted.title.strip() != ""
    assert extracted.location_type in ALLOWED_LOCATION_TYPES

    # GPU lock should be released (not held by this task_id).
    assert database.acquire_lock("gpu", "post-test-check") is True
    database.release_lock("gpu", "post-test-check")
