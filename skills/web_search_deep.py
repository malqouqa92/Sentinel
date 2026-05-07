"""Deep web-search skill for the research pipeline.

Thin subclass of WebSearchSkill that forces ``scrape_content=True`` and
``max_results=config.RESEARCH_MAX_RESULTS`` so the researcher agent gets
full page text rather than short snippets.  The /search command continues
to use the base WebSearchSkill unchanged.
"""
from typing import ClassVar

from pydantic import BaseModel

from core import config
from skills.web_search import WebSearchInput, WebSearchOutput, WebSearchSkill


class DeepWebSearchSkill(WebSearchSkill):
    name: ClassVar[str] = "web_search_deep"
    description: ClassVar[str] = (
        "Deep web search with full-page scraping for research pipelines"
    )
    version: ClassVar[str] = "1.0.0"
    requires_gpu: ClassVar[bool] = False
    input_schema: ClassVar[type[BaseModel]] = WebSearchInput
    output_schema: ClassVar[type[BaseModel]] = WebSearchOutput

    def validate_input(self, raw: dict) -> BaseModel:
        if set(raw.keys()) == {"text"}:
            return WebSearchInput(
                query=raw["text"],
                max_results=config.RESEARCH_MAX_RESULTS,
                scrape_content=True,
            )
        data = dict(raw)
        data.setdefault("scrape_content", True)
        data.setdefault("max_results", config.RESEARCH_MAX_RESULTS)
        return WebSearchInput(**data)
