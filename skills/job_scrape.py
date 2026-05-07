"""Job scraping skill -- python-jobspy wrapper.

Uses ``jobspy.scrape_jobs`` (sync) wrapped in ``asyncio.to_thread``.
Lazy-imported so the bot still boots if python-jobspy is not installed
-- ``/jobsearch`` will fail with a clear install-needed error instead.

Phase 12: post-scrape title + avoid + workplace pre-filter using
PROFILE.yml + per-call /jobsearch flags. Filtering BEFORE the
fan-out to job_extract drops postings that would otherwise eat
LLM tokens for a guaranteed-rejection.

Output is wrapped in a single-field model (``postings``) and marked
``output_is_list = True`` so the agent unwraps the list and the next
skill (``job_extract``) fans out per posting.
"""
from __future__ import annotations

import asyncio
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from core import config, database
from core.job_profile import load_profile, title_passes
from core.logger import log_event
from core.skills import BaseSkill, SkillError


class JobScrapeInput(BaseModel):
    query: str
    location: str | None = None
    sites: list[str] | None = None
    distance: int = Field(
        default_factory=lambda: config.JOBSPY_DEFAULT_DISTANCE,
    )
    hours_old: int = Field(
        default_factory=lambda: config.JOBSPY_DEFAULT_HOURS_OLD,
    )
    results_wanted: int = Field(
        default_factory=lambda: config.JOBSPY_DEFAULT_RESULTS,
    )
    # Phase 12: per-call avoid list, merged with PROFILE.avoid_companies.
    # Comma-separated string from /jobsearch --avoid "Foo,Bar".
    avoid: list[str] = Field(default_factory=list)
    # Phase 12: workplace preference. on-site|hybrid|remote|all.
    # `remote` and `on-site` get server-side filtering via jobspy is_remote;
    # `hybrid` falls through to job_extract for post-filter (best-effort
    # since jobspy does not expose a hybrid flag).
    workplace: str = "all"

    @field_validator("avoid", mode="before")
    @classmethod
    def _split_avoid(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(s).strip() for s in v if str(s).strip()]
        return v

    @field_validator("workplace", mode="before")
    @classmethod
    def _normalize_workplace(cls, v):
        if v is None or v == "":
            return "all"
        if not isinstance(v, str):
            v = str(v)
        v = v.strip().lower()
        v = {"onsite": "on-site", "in-person": "on-site",
             "wfh": "remote", "any": "all"}.get(v, v)
        if v not in ("on-site", "hybrid", "remote", "all"):
            raise ValueError(
                f"workplace must be on-site|hybrid|remote|all, got {v!r}"
            )
        return v

    @field_validator("sites", mode="before")
    @classmethod
    def _split_sites(cls, v):
        # Router gives flag values as strings; let users pass
        # `--sites indeed,linkedin` or `--sites indeed`.
        if v is None:
            return None
        if isinstance(v, str):
            return [s.strip().lower() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(s).strip().lower() for s in v if str(s).strip()]
        return v


class ScrapedPosting(BaseModel):
    title: str
    company: str
    location: str
    description: str
    url: str
    site: str
    text: str  # combined for downstream job_extract input
    # Phase 12: workplace preference passed through so job_extract /
    # job_score can post-filter for `hybrid` (jobspy can't filter
    # server-side for it). For `on-site`/`remote` jobspy does the
    # server-side filter, but we still propagate so downstream can
    # double-check after extraction.
    workplace_pref: str = "all"


class JobScrapeOutput(BaseModel):
    """Phase 12: SINGLE list field is intentional -- the agent's
    output_is_list=True path requires exactly one list-typed field
    so it can unwrap to a bare list for the next skill's fan-out.

    Phase 13 telemetry (dropped titles sample, query expansion list,
    etc.) is written to LAST_SCRAPE_STATS_PATH instead of being added
    here. The bot reads that sidecar to drive the adaptive filter."""
    postings: list[ScrapedPosting]


def _row_to_posting(row: dict, workplace_pref: str = "all") -> ScrapedPosting:
    """Coerce a jobspy DataFrame row (dict) into ScrapedPosting.
    Field names vary slightly between jobspy versions; we normalize
    here so the downstream contract is stable."""
    title = str(row.get("title") or "").strip()
    company = str(row.get("company") or "").strip()
    location_parts = []
    for k in ("location", "city", "state", "country"):
        v = row.get(k)
        if v and str(v).strip():
            location_parts.append(str(v).strip())
            break  # location field is usually preferred; fall through if blank
    location = location_parts[0] if location_parts else ""
    desc_raw = str(row.get("description") or "").strip()
    desc = desc_raw[: config.JOB_DESCRIPTION_MAX_CHARS]
    url = str(
        row.get("job_url") or row.get("url") or row.get("apply_url") or "",
    ).strip()
    site = str(row.get("site") or row.get("source") or "").strip()
    text_parts = [
        f"Title: {title}",
        f"Company: {company}",
        f"Location: {location}",
        f"Source: {site}",
        f"URL: {url}",
        "",
        desc,
    ]
    text = "\n".join(p for p in text_parts if p is not None)
    return ScrapedPosting(
        title=title, company=company, location=location,
        description=desc, url=url, site=site, text=text,
        workplace_pref=workplace_pref,
    )


def _workplace_to_is_remote(workplace: str) -> bool | None:
    """Phase 12: translate our 4-value preference into jobspy's
    is_remote tri-state. None means "let everything through"."""
    if workplace == "remote":
        return True
    if workplace == "on-site":
        return False
    # hybrid + all -> no server-side filter; downstream may post-filter.
    return None


def _scrape_blocking(
    query: str, location: str | None, sites: list[str],
    distance: int, hours_old: int, results_wanted: int,
    workplace: str, trace_id: str,
) -> list[dict]:
    """Sync helper -- imported lazily; raises SkillError if the
    python-jobspy package isn't installed."""
    try:
        from jobspy import scrape_jobs  # type: ignore
    except ImportError as e:
        raise SkillError(
            "job_scrape",
            "python-jobspy is not installed. Run "
            "`py -3.12 -m pip install python-jobspy` and retry. "
            f"underlying error: {e}",
            trace_id,
        )
    is_remote = _workplace_to_is_remote(workplace)
    log_event(
        trace_id, "INFO", "skill.job_scrape",
        f"scraping query={query!r} location={location!r} "
        f"sites={sites} distance={distance} hours_old={hours_old} "
        f"results_wanted={results_wanted} workplace={workplace} "
        f"is_remote={is_remote}",
    )
    kwargs: dict = dict(
        site_name=sites,
        search_term=query,
        location=location,
        distance=distance,
        hours_old=hours_old,
        results_wanted=results_wanted,
        # Phase 12.5: tells jobspy to fetch the full description page
        # for each LinkedIn job instead of returning the truncated
        # search-result snippet (~150 chars). Adds ~1s/posting but
        # gives the scorer the same depth of body it gets from Indeed.
        linkedin_fetch_description=True,
    )
    if is_remote is not None:
        kwargs["is_remote"] = is_remote
    df = scrape_jobs(**kwargs)
    if df is None or len(df) == 0:
        return []
    # python-jobspy returns a pandas DataFrame; convert to list of
    # dicts without forcing a hard pandas import in this module.
    rows = df.to_dict(orient="records")
    return rows


class JobScrapeSkill(BaseSkill):
    name: ClassVar[str] = "job_scrape"
    description: ClassVar[str] = (
        "Scrape job postings from indeed/linkedin/google/zip via "
        "python-jobspy"
    )
    version: ClassVar[str] = "1.1.0"
    requires_gpu: ClassVar[bool] = False
    input_schema: ClassVar[type[BaseModel]] = JobScrapeInput
    output_schema: ClassVar[type[BaseModel]] = JobScrapeOutput
    output_is_list: ClassVar[bool] = True

    def validate_input(self, raw):  # noqa: ANN001 -- BaseSkill API
        # Translate user-friendly flag aliases into the schema's
        # canonical field names. The router puts free text under
        # "text", and the user types "--hours" / "--results" rather
        # than the verbose Pydantic field names.
        if not isinstance(raw, dict):
            return self.input_schema(**raw) if raw else self.input_schema()
        raw = dict(raw)
        if "text" in raw and "query" not in raw:
            raw["query"] = raw.pop("text")
        else:
            raw.pop("text", None)
        if "hours" in raw and "hours_old" not in raw:
            raw["hours_old"] = raw.pop("hours")
        if "results" in raw and "results_wanted" not in raw:
            raw["results_wanted"] = raw.pop("results")
        return self.input_schema(**raw)

    async def execute(
        self,
        input_data: BaseModel,
        trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        if not isinstance(input_data, JobScrapeInput):
            raise SkillError(
                self.name,
                f"expected JobScrapeInput, got {type(input_data).__name__}",
                trace_id,
            )
        # Phase 12: load profile for title/avoid filter.
        # Falls back to defaults if PROFILE.yml is missing -- in which
        # case nothing is filtered out.
        profile = load_profile(trace_id)
        # Workplace from input takes precedence over profile default,
        # but if input is the default ("all") and profile has a
        # preference, use the profile's.
        workplace = input_data.workplace
        if workplace == "all" and profile.location.workplace_preference != "all":
            workplace = profile.location.workplace_preference
            log_event(trace_id, "INFO", "skill.job_scrape",
                      f"workplace defaulted from PROFILE.yml -> {workplace}")
        # Phase 12.5: when the candidate cannot relocate AND the
        # request is for an on-site/hybrid role, narrow jobspy's
        # search radius to PROFILE.location.onsite_max_miles so we
        # don't pay LLM cost on Cincinnati/Chicago/Toronto postings
        # that the commute gate will skip anyway. Remote/all bypass.
        distance = input_data.distance
        if (workplace in ("on-site", "hybrid")
                and not profile.location.willing_to_relocate
                and profile.location.onsite_max_miles
                and profile.location.primary_zip):
            if distance == config.JOBSPY_DEFAULT_DISTANCE:
                # Only override the DEFAULT distance -- if the caller
                # explicitly asked for --distance N, honor it.
                old, distance = distance, profile.location.onsite_max_miles
                log_event(
                    trace_id, "INFO", "skill.job_scrape",
                    f"distance narrowed {old} -> {distance} mi via "
                    f"PROFILE.location.onsite_max_miles + "
                    f"willing_to_relocate=False",
                )
        sites = input_data.sites or list(config.JOBSPY_SITES)
        # Phase 13 (Batch 5b): expand the query into N variants. Each
        # variant runs jobspy independently; rows are deduped by URL
        # below. Disabled / no matches => single-query unchanged path.
        from core.query_expansion import expand_query, log_expansion
        queries = expand_query(input_data.query) or [input_data.query]
        log_expansion(input_data.query, queries, trace_id)
        all_rows: list[dict] = []
        seen_urls: set[str] = set()
        for q in queries:
            qrows = await asyncio.to_thread(
                _scrape_blocking,
                q, input_data.location, sites,
                distance, input_data.hours_old,
                input_data.results_wanted, workplace, trace_id,
            )
            for r in qrows:
                url = str(
                    r.get("job_url") or r.get("url")
                    or r.get("apply_url") or "",
                ).strip().lower()
                # Dedup across query variants by URL. URL-less rows
                # always pass through (we have no key to compare).
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                all_rows.append(r)
        all_postings = [_row_to_posting(r, workplace) for r in all_rows]
        # Phase 12: pre-LLM title + avoid filter.
        # Phase 13: also pre-LLM dedup against the applications table
        # (drop URLs we've already evaluated this account, regardless
        # of state). Saves the entire extract+score chain on dupes.
        kept: list[ScrapedPosting] = []
        dropped_title = 0
        dropped_avoid = 0
        dropped_dupe = 0
        # Phase 13 (Batch 5a): keep a sample of dropped titles so the
        # bot's adaptive filter can ask the brain for new negatives.
        dropped_title_titles: list[str] = []
        sample_cap = getattr(config, "JOB_ADAPTIVE_FILTER_SAMPLE_SIZE", 20)
        avoid_set = list(input_data.avoid) + list(profile.avoid_companies)
        for p in all_postings:
            # 13.1 dedup -- cheapest check first (one indexed SELECT).
            if p.url and database.application_exists(p.url):
                dropped_dupe += 1
                continue
            if not title_passes(
                p.title, profile, extra_avoid=avoid_set, company=p.company,
            ):
                # Bucketize the drop reason for logs (cheap).
                lower_avoid = {a.strip().lower() for a in avoid_set if a.strip()}
                hit_avoid = any(
                    a in p.company.lower() or a in p.title.lower()
                    for a in lower_avoid
                )
                if hit_avoid:
                    dropped_avoid += 1
                else:
                    dropped_title += 1
                    if (len(dropped_title_titles) < sample_cap
                            and p.title.strip()):
                        dropped_title_titles.append(p.title.strip())
                continue
            kept.append(p)
        log_event(
            trace_id, "INFO", "skill.job_scrape",
            f"scraped={len(all_postings)} kept={len(kept)} "
            f"dropped_title={dropped_title} dropped_avoid={dropped_avoid} "
            f"dropped_dupe={dropped_dupe} "
            f"queries={len(queries)} workplace={workplace}",
        )
        # Phase 13: write telemetry to a sidecar file the bot reads
        # post-pipeline to drive the adaptive filter. Best-effort --
        # any I/O error here must NOT break the scrape itself.
        try:
            import json as _json
            stats = {
                "trace_id": trace_id,
                "query": input_data.query,
                "queries_used": queries,
                "scraped_total": len(all_postings),
                "kept": len(kept),
                "dropped_title": dropped_title,
                "dropped_avoid": dropped_avoid,
                "dropped_dupe": dropped_dupe,
                "dropped_title_sample": dropped_title_titles,
            }
            stats_path = getattr(
                config, "LAST_SCRAPE_STATS_PATH",
                config.LOG_DIR / "last_scrape_stats.json",
            )
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(
                _json.dumps(stats, indent=2), encoding="utf-8",
            )
        except Exception as e:
            log_event(
                trace_id, "DEBUG", "skill.job_scrape",
                f"stats sidecar write skipped: {type(e).__name__}: {e}",
            )
        return JobScrapeOutput(postings=kept)
