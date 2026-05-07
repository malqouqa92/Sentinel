"""Phase 10 -- Test Q: full /research pipeline.

Mocks web_search (no network) + web_summarize (no LLM) and runs the
researcher agent end-to-end. Verifies markdown report written and
episodic entry stored.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core import config
from core.memory import get_memory


def test_q_research_pipeline_writes_report_and_episode(monkeypatch, tmp_path):
    from core.agent_registry import AGENT_REGISTRY
    monkeypatch.setattr(
        config, "RESEARCH_OUTPUT_DIR", tmp_path / "research",
    )
    config.RESEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Mock web_search ---
    from skills import web_search

    fake_results = [
        {
            "title": "Source 1: AI Coding Agents",
            "url": "https://example.com/1",
            "snippet": "Discussion of LLM-based coding agents.",
            "body_text": (
                "Recent benchmarks show coding agents are getting "
                "more reliable. SWE-bench scores jumped Q1 2026."
            ),
        },
        {
            "title": "Source 2: Agent reliability",
            "url": "https://example.com/2",
            "snippet": "Reliability metrics for autonomous agents.",
            "body_text": (
                "Three reliability axes: retry, fallback, recovery."
            ),
        },
    ]

    async def fake_search_execute(self, input_data, trace_id, context=None):
        return web_search.WebSearchOutput(
            query=input_data.query,
            results=[web_search.WebSearchResult(**r) for r in fake_results],
            result_count=len(fake_results),
        )
    monkeypatch.setattr(
        web_search.WebSearchSkill, "execute", fake_search_execute,
    )

    # --- Mock web_summarize ---
    from skills import web_summarize

    async def fake_summarize_execute(
        self, input_data, trace_id, context=None,
    ):
        return web_summarize.WebSummarizeOutput(
            query=input_data.query,
            summaries=[
                web_summarize.ResultSummary(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    summary=(
                        f"Stub summary of {r.get('title', '?')}: "
                        f"{(r.get('body_text') or '')[:80]}"
                    ),
                )
                for r in input_data.results
            ],
        )
    monkeypatch.setattr(
        web_summarize.WebSummarizeSkill, "execute", fake_summarize_execute,
    )

    agent = AGENT_REGISTRY.get("researcher")
    assert agent is not None, "researcher agent should be registered"
    out = asyncio.run(agent.run(
        {"query": "AI coding agent reliability"},
        "SEN-test-Q",
    ))
    assert out.get("_error") is not True, f"agent error: {out}"
    assert out["summary_count"] == 2
    report_path = Path(out["report_path"])
    if not report_path.is_absolute():
        report_path = config.PROJECT_ROOT / report_path
    assert report_path.exists()
    body = report_path.read_text(encoding="utf-8")
    assert "AI coding agent reliability" in body
    assert "Source 1: AI Coding Agents" in body

    mem = get_memory()
    eps = mem.get_recent_episodes(scope="researcher", limit=5)
    assert any(
        "Research brief" in e.summary and "2 sources" in e.summary
        for e in eps
    ), f"expected researcher episode; got: {[e.summary for e in eps]}"
