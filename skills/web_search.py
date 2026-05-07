"""Web search + optional content scraping via DuckDuckGo HTML.

No API keys, no SDKs. httpx for async HTTP, BeautifulSoup for parsing.
Honors robots.txt before scraping any page. Rate-limited via
config.SCRAPE_DELAY between outbound requests. Rotates User-Agent
per request to look less robotic.
"""
import asyncio
import random
from typing import ClassVar
from urllib.parse import parse_qs, unquote, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

from core import config
from core.logger import log_event
from core.skills import BaseSkill, SkillError


class WebSearchInput(BaseModel):
    query: str
    max_results: int = 5
    scrape_content: bool = False


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    body_text: str | None = None


class WebSearchOutput(BaseModel):
    query: str
    results: list[WebSearchResult]
    result_count: int


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
    "Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 "
    "Mobile/15E148 Safari/604.1",
]

DDG_URL = "https://html.duckduckgo.com/html/"

# robots.txt parser cache: hostname -> RobotFileParser (or None if fetch failed)
_robots_cache: dict[str, RobotFileParser | None] = {}


def _pick_ua() -> str:
    return random.choice(USER_AGENTS)


def _unwrap_ddg_url(href: str) -> str:
    """DDG wraps result URLs as //duckduckgo.com/l/?uddg=ENCODED. Extract."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l"):
        qs = parse_qs(parsed.query)
        target = qs.get("uddg", [None])[0]
        if target:
            return unquote(target)
    return href


def _is_anomaly_page(html: str) -> bool:
    """DDG serves a CAPTCHA-style 'anomaly modal' to suspected bots.
    Detect it so callers know the empty result list is a block, not
    a real no-results query."""
    return "anomaly-modal" in html or "anomaly_modal" in html


def _parse_ddg_html(html: str) -> list[WebSearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[WebSearchResult] = []
    # Most common result block on html.duckduckgo.com
    for div in soup.select("div.result, div.web-result"):
        a = div.select_one("a.result__a")
        if a is None:
            a = div.select_one("h2 a")
        if a is None:
            continue
        title = a.get_text(strip=True)
        url = _unwrap_ddg_url(a.get("href", ""))
        snippet_el = (
            div.select_one("a.result__snippet")
            or div.select_one(".result__snippet")
            or div.select_one(".snippet")
        )
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        if title and url:
            out.append(WebSearchResult(
                title=title, url=url, snippet=snippet,
            ))
    # Lite endpoint fallback: <a> rows in a flat table
    if not out:
        for a in soup.select("a.result-link, a[href]"):
            href = a.get("href", "")
            if not href.startswith(("http://", "https://", "//")):
                continue
            url = _unwrap_ddg_url(href)
            if "duckduckgo.com" in url:
                continue
            title = a.get_text(strip=True)
            if title and url:
                out.append(WebSearchResult(
                    title=title, url=url, snippet="",
                ))
    return out


async def _fetch_ddg(
    query: str, trace_id: str, timeout: float
) -> str:
    headers = {"User-Agent": _pick_ua(), "Accept": "text/html"}
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers=headers,
    ) as client:
        # POST is the canonical form; GET also works as fallback.
        resp = await client.post(DDG_URL, data={"q": query})
        if resp.status_code != 200:
            log_event(
                trace_id, "WARNING", "skill.web_search",
                f"DDG returned status={resp.status_code}; trying GET",
            )
            resp = await client.get(DDG_URL, params={"q": query})
        resp.raise_for_status()
        return resp.text


async def _check_robots(url: str, trace_id: str) -> bool:
    """Return True if we may scrape this URL (robots-allowed). Cached
    per host. On any robots.txt fetch failure, default to allow with a
    DEBUG note (best-effort policy)."""
    parsed = urlparse(url)
    host = parsed.netloc
    if not host:
        return False
    if host not in _robots_cache:
        rp = RobotFileParser()
        rp.set_url(f"{parsed.scheme}://{host}/robots.txt")
        try:
            await asyncio.to_thread(rp.read)
            _robots_cache[host] = rp
        except Exception as e:
            log_event(
                trace_id, "DEBUG", "skill.web_search",
                f"robots.txt fetch failed for {host}: {e}; allowing",
            )
            _robots_cache[host] = None
    parser = _robots_cache[host]
    if parser is None:
        return True  # No info -> allow.
    return parser.can_fetch("*", url)


def _extract_main_text(html: str, cap: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer",
                     "header", "aside", "form"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())
    if len(text) > cap:
        return text[:cap]
    return text


async def _scrape_one(
    result: WebSearchResult, trace_id: str,
) -> WebSearchResult:
    try:
        if not await _check_robots(result.url, trace_id):
            log_event(
                trace_id, "WARNING", "skill.web_search",
                f"robots.txt disallows scraping {result.url}; skipping body",
            )
            return result
        headers = {"User-Agent": _pick_ua(),
                   "Accept": "text/html,application/xhtml+xml"}
        async with httpx.AsyncClient(
            timeout=config.SCRAPE_TIMEOUT, follow_redirects=True,
            headers=headers,
        ) as client:
            resp = await client.get(result.url)
            if resp.status_code != 200:
                log_event(
                    trace_id, "WARNING", "skill.web_search",
                    f"page status={resp.status_code} for {result.url}; "
                    f"skipping body",
                )
                return result
            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype.lower():
                log_event(
                    trace_id, "DEBUG", "skill.web_search",
                    f"non-html content-type={ctype} for {result.url}; "
                    f"skipping body",
                )
                return result
            body = _extract_main_text(
                resp.text, config.SCRAPE_MAX_BODY_CHARS,
            )
            return result.model_copy(update={"body_text": body})
    except Exception as e:
        log_event(
            trace_id, "WARNING", "skill.web_search",
            f"scrape failed for {result.url}: {type(e).__name__}: {e}",
        )
        return result


async def _scrape_results(
    results: list[WebSearchResult], trace_id: str,
) -> list[WebSearchResult]:
    enriched: list[WebSearchResult] = []
    for i, r in enumerate(results):
        if i > 0:
            await asyncio.sleep(config.SCRAPE_DELAY)
        enriched.append(await _scrape_one(r, trace_id))
    return enriched


class WebSearchSkill(BaseSkill):
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Searches the web via DuckDuckGo HTML; optionally scrapes "
        "top results' body text"
    )
    version: ClassVar[str] = "1.0.0"
    requires_gpu: ClassVar[bool] = False
    input_schema: ClassVar[type[BaseModel]] = WebSearchInput
    output_schema: ClassVar[type[BaseModel]] = WebSearchOutput

    def validate_input(self, raw: dict) -> BaseModel:
        # Router-text mode: "/search foo bar baz" -> {"text": "foo bar baz"}
        if set(raw.keys()) == {"text"}:
            return WebSearchInput(query=raw["text"])
        return WebSearchInput(**raw)

    async def execute(
        self, input_data: BaseModel, trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        if not isinstance(input_data, WebSearchInput):
            raise SkillError(
                self.name,
                f"expected WebSearchInput, got {type(input_data).__name__}",
                trace_id,
            )
        log_event(
            trace_id, "INFO", "skill.web_search",
            f"search starting query={input_data.query!r} "
            f"max_results={input_data.max_results} "
            f"scrape={input_data.scrape_content}",
        )
        try:
            html = await _fetch_ddg(
                input_data.query, trace_id, config.SCRAPE_TIMEOUT,
            )
        except Exception as e:
            raise SkillError(
                self.name,
                f"DDG search failed: {type(e).__name__}: {e}",
                trace_id,
            ) from e

        if _is_anomaly_page(html):
            log_event(
                trace_id, "WARNING", "skill.web_search",
                "DDG returned an anomaly/CAPTCHA page (bot detected); "
                "returning empty result set",
            )
            results = []
        else:
            results = _parse_ddg_html(html)[: input_data.max_results]
        log_event(
            trace_id, "INFO", "skill.web_search",
            f"got {len(results)} raw results from DDG",
        )

        if input_data.scrape_content and results:
            results = await _scrape_results(results, trace_id)

        return WebSearchOutput(
            query=input_data.query,
            results=results,
            result_count=len(results),
        )
