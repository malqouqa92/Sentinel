"""Per-result summarizer for the research pipeline.

Consumes a WebSearchOutput-shaped dict (``{query, results, result_count}``)
and emits ``{query, summaries: [...]}`` where each summary is a short
brief of a single search hit.

Iterates internally over results -- no fan-out needed because
``web_search`` returns a multi-field dict. accepts_list=False.
"""
from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from core import config
from core.llm import INFERENCE_CLIENT, LLMError, OllamaClient
from core.logger import log_event
from core.skills import BaseSkill, SkillError


class WebSummarizeInput(BaseModel):
    query: str = ""
    results: list[dict[str, Any]] = Field(default_factory=list)
    # web_search puts its hit count here; we ignore it but tolerate
    # the field so model-dump-then-validate doesn't reject.
    result_count: int | None = None

    model_config = {"extra": "allow"}


class ResultSummary(BaseModel):
    title: str
    url: str
    summary: str


class WebSummarizeOutput(BaseModel):
    query: str
    summaries: list[ResultSummary] = Field(default_factory=list)


SYSTEM_PROMPT = (
    "You are a research analyst writing a one-paragraph factual brief "
    "of a single search result. Stick to claims supported by the "
    "provided text. Do not speculate. Cap your reply at "
    "{max_words} words. No preamble, no signoff, no bullets. /no_think"
)


def _summarize_one(
    title: str, url: str, snippet: str, body: str | None,
    query: str, model_id: str, trace_id: str,
    client: OllamaClient,
    max_words: int,
) -> str:
    src = body or snippet or ""
    src = src[:1500]  # keep prompt cheap; the brain ctx is small
    if not src.strip():
        return f"(no extractable content for {url})"
    user_prompt = (
        f"Search query: {query}\n"
        f"Result title: {title}\n"
        f"Result URL: {url}\n"
        f"Result content (truncated):\n{src}\n\n"
        f"Write a {max_words}-word factual summary of THIS result that "
        f"answers the query."
    )
    try:
        return client.generate(
            model=model_id,
            prompt=user_prompt,
            system=SYSTEM_PROMPT.format(max_words=max_words),
            trace_id=trace_id,
        ).strip()
    except LLMError as e:
        log_event(
            trace_id, "WARNING", "skill.web_summarize",
            f"summary failed for {url}: {type(e).__name__}: {e}",
        )
        return f"(summary unavailable: {type(e).__name__})"


def _summarize_blocking(
    inp: WebSummarizeInput, model_id: str, trace_id: str,
    client: OllamaClient | None = None,
) -> WebSummarizeOutput:
    client = client or OllamaClient()
    summaries: list[ResultSummary] = []
    max_words = config.RESEARCH_SUMMARY_MAX_WORDS
    log_event(
        trace_id, "INFO", "skill.web_summarize",
        f"summarizing {len(inp.results)} results model={model_id}",
    )
    for r in inp.results:
        title = str(r.get("title") or "").strip() or "(no title)"
        url = str(r.get("url") or "").strip()
        snippet = str(r.get("snippet") or "").strip()
        body = r.get("body_text")
        summary = _summarize_one(
            title, url, snippet, body, inp.query,
            model_id, trace_id, client, max_words,
        )
        summaries.append(
            ResultSummary(title=title, url=url, summary=summary),
        )
    return WebSummarizeOutput(query=inp.query, summaries=summaries)


class WebSummarizeSkill(BaseSkill):
    name: ClassVar[str] = "web_summarize"
    description: ClassVar[str] = (
        "Summarize each web_search result with the local LLM"
    )
    version: ClassVar[str] = "1.0.0"
    requires_gpu: ClassVar[bool] = True
    input_schema: ClassVar[type[BaseModel]] = WebSummarizeInput
    output_schema: ClassVar[type[BaseModel]] = WebSummarizeOutput

    async def execute(
        self, input_data: BaseModel, trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        if not isinstance(input_data, WebSummarizeInput):
            raise SkillError(
                self.name,
                f"expected WebSummarizeInput, got "
                f"{type(input_data).__name__}",
                trace_id,
            )
        model_name_or_id: str = (
            (context or {}).get("model") or config.BRAIN_MODEL
        )
        cfg = INFERENCE_CLIENT.model_registry.get(model_name_or_id)
        backend_model_id = cfg.model_id if cfg else model_name_or_id
        return await asyncio.to_thread(
            _summarize_blocking, input_data, backend_model_id, trace_id,
        )
