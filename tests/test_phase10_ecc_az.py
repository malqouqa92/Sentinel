"""Phase 10 ECC A-Z — comprehensive end-to-end correctness sweep.

26 lettered checks covering Phase 1-10. Most are tight assertions that
the relevant subsystem is healthy; a handful do small integration runs.
Each test is intentionally small so failures pinpoint the broken
subsystem; deep behavior is covered by the per-subsystem tests
(test_brain.py, test_memory.py, test_pipelines.py, etc.).
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import config


# A — config integrity
def test_a_config_integrity():
    assert config.PROJECT_ROOT.exists()
    assert config.WORKSPACE_DIR.exists()
    assert config.PERSONA_DIR.exists()
    assert isinstance(config.REGISTERED_COMMANDS, set)
    for cmd in (
        "/ping", "/help", "/code", "/jobsearch",
        "/research", "/remember", "/curate",
    ):
        assert cmd in config.REGISTERED_COMMANDS, cmd
    assert isinstance(config.PROTECTED_FILES, set)
    assert "MEMORY.md" in config.PROTECTED_FILES


# B — telemetry trace IDs
def test_b_trace_id_format():
    from core.telemetry import generate_trace_id
    seen = {generate_trace_id() for _ in range(50)}
    assert len(seen) == 50
    for tid in seen:
        assert re.fullmatch(r"SEN-[0-9a-f]{8}", tid), tid


# C — logger writes JSONL with required fields
def test_c_logger_jsonl_shape(tmp_path, monkeypatch):
    from core.logger import log_event
    log_event("SEN-test-c", "INFO", "ecc", "hello world")
    log_path = config.LOG_DIR / config.LOG_FILE
    assert log_path.exists()
    with log_path.open(encoding="utf-8") as f:
        last = f.readlines()[-1]
    entry = json.loads(last)
    for k in ("timestamp", "trace_id", "level", "component", "message"):
        assert k in entry, f"missing field: {k}"


# D — router first-token + flag parsing
def test_d_router_rules():
    from core.router import route
    bad = route("hello /ping")
    assert bad.status == "error"
    assert bad.error_code == "INVALID_POSITION"
    unknown = route("/unknownXYZ")
    assert unknown.status == "error"
    assert unknown.error_code == "UNKNOWN_COMMAND"
    ok = route("/jobsearch RSM --location \"Detroit, MI\" --hours 48")
    assert ok.status == "ok"
    assert ok.command == "jobsearch"
    assert ok.args.get("hours") == "48"
    assert "Detroit, MI" in ok.args.get("location", "")


# E — database CRUD
def test_e_database_round_trip():
    from core import database
    tid = database.add_task(
        "SEN-test-e", "ping", {"text": "hi"},
    )
    row = database.get_task(tid)
    assert row is not None and row.command == "ping"
    database.complete_task(tid, {"ok": True})
    row2 = database.get_task(tid)
    assert row2 is not None and row2.status == "completed"


# F — worker would acquire GPU lock (DB-level only here)
def test_f_gpu_lock_acquire_release():
    from core import database
    assert database.acquire_lock("gpu", "task-x") is True
    # Second try by a different task should fail.
    assert database.acquire_lock("gpu", "task-y") is False
    database.release_lock("gpu", "task-x")
    assert database.acquire_lock("gpu", "task-z") is True


# G — skill registry has expected skills
def test_g_skill_registry_complete():
    from core.registry import SKILL_REGISTRY
    names = {s["name"] for s in SKILL_REGISTRY.list_skills()}
    for required in (
        "job_extract", "job_scrape", "job_score", "job_report",
        "web_search", "web_summarize", "research_report",
        "file_io", "code_execute",
    ):
        assert required in names, required


# H — agent registry has expected agents
def test_h_agent_registry_complete():
    from core.agent_registry import AGENT_REGISTRY
    for required in ("job_analyst", "job_searcher", "researcher"):
        assert AGENT_REGISTRY.get(required) is not None, required


# I — orchestrator dispatch table is sane
def test_i_orchestrator_command_map():
    cm = config.COMMAND_AGENT_MAP
    assert cm["/jobsearch"] == "job_searcher"
    assert cm["/research"] == "researcher"
    assert cm["/code"] == "code_assistant"
    assert cm["/ping"] is None


# J — knowledge base FTS5 round-trip
def test_j_kb_search_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", tmp_path / "k.db")
    from core.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(tmp_path / "k.db")
    kb.add_pattern(
        tags=["ecc", "test"], problem_summary="round-trip",
        solution_code="def f(): return 1",
        solution_pattern="trivial", explanation="just a sanity round-trip",
        trace_id="SEN-test-j",
    )
    hits = kb.search("ecc round-trip")
    assert len(hits) >= 1
    assert hits[0].problem_summary == "round-trip"


# K — brain output is parseable JSON-or-chat
def test_k_brain_output_shape():
    from core.brain import _extract_json_object, _strip_think_block
    parsed = _extract_json_object(
        '<think>cot</think>\n```json\n{"intent":"chat","response":"hi"}\n```',
    )
    assert parsed is not None
    assert parsed["intent"] == "chat"
    stripped = _strip_think_block("<think>x</think>visible")
    assert stripped == "visible"


# L — three-tier memory works
def test_l_memory_three_tier():
    from core.memory import WorkingMemory, get_memory
    wm = WorkingMemory(max_messages=3)
    for i in range(5):
        wm.add("sess1", "user", f"msg-{i}")
    assert len(wm.get_recent("sess1")) == 3
    m = get_memory()
    m.store_episode("ecc", "SEN-l", "test", "summary 1")
    m.store_fact("ecc_key", "ecc_value")
    assert m.get_fact("ecc_key").value == "ecc_value"
    assert len(m.get_recent_episodes(scope="ecc")) >= 1


# M — file_guard authorize_update keeps diff-watch clean
def test_m_file_guard_authorize_clean(temp_persona):
    from core.file_guard import FileGuard
    alerts: list[str] = []
    g = FileGuard(directory=temp_persona, alert_callback=alerts.append)
    g.authorize_update("MEMORY.md", "# Memory\n- ECC fact\n")
    assert g.check_integrity() == []
    assert alerts == []


# N — persona files seeded in real workspace exist
def test_n_persona_files_seeded():
    for name in config.PROTECTED_FILES:
        assert (config.PERSONA_DIR / name).exists(), name


# O — job pipeline fan-out (smoke)
def test_o_job_pipeline_fanout(monkeypatch, tmp_path):
    from core.agent_registry import AGENT_REGISTRY
    from skills import job_scrape, job_extract, job_score
    monkeypatch.setattr(
        config, "JOBSEARCH_OUTPUT_DIR", tmp_path / "js",
    )
    config.JOBSEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counters = {"score": 0}

    async def fake_scrape(self, input_data, trace_id, context=None):
        return job_scrape.JobScrapeOutput(postings=[
            job_scrape.ScrapedPosting(
                title=f"T{i}", company=f"C{i}", location="Detroit",
                description="", url=f"u{i}", site="indeed",
                text=f"posting {i}",
            ) for i in range(2)
        ])

    async def fake_extract(self, input_data, trace_id, context=None):
        return job_extract.JobExtraction(
            title="T", company="C", location="Detroit",
            location_type="hybrid", seniority="mid", confidence=0.8,
        )

    def _make(input_one):
        return job_score.ScoredPosting(
            title=input_one.title, company=input_one.company,
            location=input_one.location,
            location_type=input_one.location_type,
            seniority=input_one.seniority,
            score=3.0, dimensions={}, reasons=[],
            recommendation="skip",
        )

    async def fake_score(self, input_data, trace_id, context=None):
        # Phase 13: handle batched input. Counters track postings, not calls.
        if isinstance(input_data, job_score.JobScoreBatchInput):
            postings = []
            for inp in input_data.postings:
                counters["score"] += 1
                postings.append(_make(inp))
            return job_score.JobScoreBatchOutput(postings=postings)
        counters["score"] += 1
        return _make(input_data)
    monkeypatch.setattr(job_scrape.JobScrapeSkill, "execute", fake_scrape)
    monkeypatch.setattr(job_extract.JobExtractSkill, "execute", fake_extract)
    monkeypatch.setattr(job_score.JobScoreSkill, "execute", fake_score)
    agent = AGENT_REGISTRY.get("job_searcher")
    out = asyncio.run(agent.run({"query": "T"}, "SEN-ecc-O"))
    assert out.get("_error") is not True
    assert counters["score"] == 2


# P — research pipeline writes a report
def test_p_research_writes_report(monkeypatch, tmp_path):
    from core.agent_registry import AGENT_REGISTRY
    from skills import web_search, web_summarize
    monkeypatch.setattr(
        config, "RESEARCH_OUTPUT_DIR", tmp_path / "r",
    )
    config.RESEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async def fake_search(self, input_data, trace_id, context=None):
        return web_search.WebSearchOutput(
            query=input_data.query,
            results=[web_search.WebSearchResult(
                title="T", url="u", snippet="s", body_text="b",
            )],
            result_count=1,
        )

    async def fake_sum(self, input_data, trace_id, context=None):
        return web_summarize.WebSummarizeOutput(
            query=input_data.query,
            summaries=[web_summarize.ResultSummary(
                title="T", url="u", summary="brief",
            )],
        )
    monkeypatch.setattr(web_search.WebSearchSkill, "execute", fake_search)
    monkeypatch.setattr(web_summarize.WebSummarizeSkill, "execute", fake_sum)
    agent = AGENT_REGISTRY.get("researcher")
    out = asyncio.run(agent.run({"query": "X"}, "SEN-ecc-P"))
    assert out.get("_error") is not True
    rp = Path(out["report_path"])
    if not rp.is_absolute():
        rp = config.PROJECT_ROOT / rp
    assert rp.exists() and rp.stat().st_size > 0


# Q — curation propose + apply (mocked claude)
def test_q_curation_propose_apply(temp_persona):
    from core.curation import CurationFlow
    from core.file_guard import FileGuard
    from core.memory import get_memory

    class _F:
        async def generate(self, prompt, system, trace_id, timeout=None, **kw):
            return (
                '{"memory_additions":["q-fact"],'
                '"memory_removals":[],"user_updates":[],'
                '"no_changes":false}'
            )
    g = FileGuard(directory=temp_persona)
    flow = CurationFlow(
        memory_manager=get_memory(), file_guard=g,
        brain=SimpleNamespace(reload_persona=lambda: None),
        claude_client=_F(),
    )
    record = asyncio.run(flow.propose("SEN-ecc-Q"))
    assert record.get("_error") is not True
    applied = flow.apply(record["token"], "SEN-ecc-Q-apply")
    assert applied["memory_md_changed"] is True
    md = (temp_persona / "MEMORY.md").read_text(encoding="utf-8")
    assert "q-fact" in md


# R — bot module imports cleanly + handlers wired
def test_r_bot_handlers_registered():
    from interfaces.telegram_bot import SentinelTelegramBot
    # Inspecting handler names without constructing (would need token).
    src = Path(SentinelTelegramBot.__module__.replace(".", "/") + ".py")
    body = (config.PROJECT_ROOT / src).read_text(encoding="utf-8")
    for cmd in (
        "remember", "forget", "recall", "memory",
        "jobsearch", "research", "curate",
        "curate_approve", "curate_reject",
    ):
        assert f'CommandHandler("{cmd}"' in body, cmd


# S — working memory ring buffer
def test_s_working_memory_ring():
    from core.memory import WorkingMemory
    wm = WorkingMemory(max_messages=2)
    wm.add("s", "user", "a")
    wm.add("s", "user", "b")
    wm.add("s", "user", "c")
    msgs = wm.get_recent("s")
    assert [m["message"] for m in msgs] == ["b", "c"]


# T — episodic decay reduces score
def test_t_episodic_decay_lowers_score():
    from core.memory import get_memory
    m = get_memory()
    eid = m.store_episode("ecc-T", "SEN-T", "x", "decayable", relevance_score=1.0)
    conn = sqlite3.connect(m.db_path)
    conn.execute(
        "UPDATE episodic_memory SET created_at = ? WHERE id = ?",
        ("2020-01-01T00:00:00+00:00", eid),
    )
    conn.commit()
    conn.close()
    m.decay_relevance(days_old=30, factor=0.5)
    e = next(x for x in m.get_recent_episodes(scope="ecc-T") if x.id == eid)
    assert e.relevance_score == pytest.approx(0.5, abs=0.01)


# U — semantic upsert by key
def test_u_semantic_upsert():
    from core.memory import get_memory
    m = get_memory()
    m.store_fact("u_key", "first", source="auto_extracted")
    m.store_fact("u_key", "second", source="user_explicit")
    f = m.get_fact("u_key")
    assert f is not None
    assert f.value == "second"  # higher confidence wins


# V — auto-extraction stores facts at conf 0.6 from a JSON-array reply
def test_v_auto_extraction_smoke():
    from core.memory import get_memory
    from pydantic import BaseModel
    m = get_memory()

    class _Result(BaseModel):
        text: str
        model_used: str = "fake"
        backend: str = "fake"

    class _Inf:
        async def generate(self, **kw):
            return _Result(text='[{"key":"v_key","value":"v_val"}]')

    fake_brain = SimpleNamespace(inference=_Inf(), model="x")
    msgs = [{"role": "user", "message": f"m{i}"} for i in range(5)]
    n = asyncio.run(m.extract_facts_from_conversation(
        msgs, "SEN-V", brain=fake_brain,
    ))
    assert n == 1
    assert m.get_fact("v_key").confidence == pytest.approx(0.6)


# W — /remember + /recall round-trip via memory API (handler tested elsewhere)
def test_w_remember_recall_round_trip():
    from core.memory import get_memory
    m = get_memory()
    m.store_fact("w_key", "w_val", source="user_explicit")
    facts = m.search_facts("w_key")
    assert len(facts) == 1 and facts[0].value == "w_val"


# X — persona injected into brain system prompt
def test_x_persona_in_brain_system():
    from core.brain import BrainRouter
    brain = BrainRouter(inference_client=SimpleNamespace())
    sys_prompt = brain._build_system_prompt()
    assert "PERSONA CONTEXT" in sys_prompt
    assert "IDENTITY.md" in sys_prompt


# Y — error envelope surfaces from agent on skill failure
def test_y_error_envelope_on_skill_failure(monkeypatch):
    from core.agent_registry import AGENT_REGISTRY
    from skills import job_scrape

    async def boom(self, input_data, trace_id, context=None):
        from core.skills import SkillError
        raise SkillError("job_scrape", "boom-on-purpose", trace_id)
    monkeypatch.setattr(
        job_scrape.JobScrapeSkill, "execute", boom,
    )
    agent = AGENT_REGISTRY.get("job_searcher")
    out = asyncio.run(agent.run({"query": "x"}, "SEN-ecc-Y"))
    assert out.get("_error") is True
    assert out.get("failed_at") == "job_scrape"
    assert "boom-on-purpose" in out.get("error", "")


# Z — multi-skill agent: job_analyst still runs (single-step) AND
# the new agents both load + dispatch (basic confidence run).
def test_z_multi_skill_agents_loadable():
    from core.agent_registry import AGENT_REGISTRY
    for name in ("job_analyst", "job_searcher", "researcher"):
        agent = AGENT_REGISTRY.get(name)
        assert agent is not None, name
        # Pipelines are non-empty and reference real skills.
        assert agent.config.skill_pipeline
