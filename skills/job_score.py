"""Job scoring skill -- 5-dimension 1-5 weighted rubric (Phase 12).
Phase 13 -- batched LLM calls (N postings per Ollama request).

Hard cutover from the Phase 10 0-1 fit score:

  Old: ScoredPosting{score: float 0..1, recommend: bool}
  New: ScoredPosting{score: float 1..5, dimensions: dict, archetype: str,
                     legitimacy: dict, recommendation: str (band)}

Reads PROFILE.yml for archetype list + narrative + comp range. The
archetype is detected deterministically (substring keyword match) and
passed to the LLM as framing. The LLM returns the 5 dimensions as
1-5 ints; archetypes.weighted_score combines them; bands derive the
recommendation label (apply_now / worth_applying / maybe / skip).

Posting-legitimacy is a separate signal (does NOT affect global score).
Phase 12 only uses HTML-text signals (LLM-judged JD specificity,
salary transparency, contradiction).

Phase 13 batching: the skill is now ``accepts_list=True`` /
``output_is_list=True``. Pipeline hands it the full list of extracted
postings; the skill chunks them into ``JOB_SCORE_BATCH_SIZE`` groups
and asks Ollama to score each chunk in one call. On any per-batch
JSON parse failure we fall back to per-item scoring for that batch
(slower but matches Phase 12 behavior). Single-item input still works
for direct test/CLI use -- ``execute()`` dispatches on input type.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator

from core import config
from core.archetypes import (
    DIMENSION_KEYS,
    detect_archetype,
    legitimacy_tier,
    recommendation_band,
    weighted_score,
    weights_for_archetype,
)
from core.geo import outside_commute
from core.job_profile import (
    has_seniority_boost, load_profile,
    region_score_adjustment, state_in_whitelist,
)
from core.llm import INFERENCE_CLIENT, LLMError, OllamaClient
from core.logger import log_event
from core.skills import BaseSkill, SkillError


class JobScoreInput(BaseModel):
    """Mirrors JobExtraction shape so it can sit downstream of
    job_extract in a pipeline. Permissive on extras for forward-compat."""
    title: str = ""
    company: str = ""
    location: str = ""
    location_type: str = ""
    salary_range: str | None = None
    industry: str | None = None
    seniority: str = ""
    key_requirements: list[str] = Field(default_factory=list)
    deal_breakers: list[str] = Field(default_factory=list)
    relevance_signals: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    # Phase 12 carry-throughs from job_scrape -> job_extract:
    url: str = ""
    description: str = ""
    workplace_pref: str = "all"

    model_config = {"extra": "allow"}


class Legitimacy(BaseModel):
    tier: str = "high"          # high | caution | suspicious
    signals: list[str] = Field(default_factory=list)


class ScoredPosting(BaseModel):
    """Phase 12 scoring output. score is 1.0-5.0 weighted across
    the 5 dimensions (cv_match, north_star, comp, cultural_signals,
    red_flags). recommendation is one of: apply_now, worth_applying,
    maybe, skip."""
    title: str
    company: str
    location: str
    location_type: str
    salary_range: str | None = None
    seniority: str
    url: str = ""
    archetype: str = "Unknown"
    score: float                 # global 1.0-5.0
    dimensions: dict[str, float] = Field(default_factory=dict)
    recommendation: str = "skip"  # band label
    reasons: list[str] = Field(default_factory=list)
    legitimacy: Legitimacy = Field(default_factory=Legitimacy)

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 1.0
        return round(max(1.0, min(5.0, v)), 2)

    @field_validator("dimensions", mode="before")
    @classmethod
    def _clamp_dims(cls, v: Any) -> dict[str, float]:
        out = {}
        for k, val in (v or {}).items():
            try:
                fv = float(val)
            except (TypeError, ValueError):
                fv = 3.0
            out[k] = round(max(1.0, min(5.0, fv)), 2)
        return out

    @field_validator("reasons")
    @classmethod
    def _cap_reasons(cls, v: list[str]) -> list[str]:
        return [str(r).strip() for r in (v or []) if str(r).strip()][:5]

    @field_validator("recommendation")
    @classmethod
    def _check_band(cls, v: str) -> str:
        v = (v or "skip").strip().lower()
        if v not in ("apply_now", "worth_applying", "maybe", "skip"):
            return "skip"
        return v


class JobScoreBatchInput(BaseModel):
    """Phase 13 batch input. Wraps a list of JobScoreInput so the
    skill can accept the whole pipeline list at once and chunk it
    internally. Used when pipeline hands the skill a list of extracted
    postings (the common path)."""
    postings: list[JobScoreInput]


class JobScoreBatchOutput(BaseModel):
    """Phase 13 list-wrapped output. Agent unwraps via output_is_list=True
    so downstream job_report receives a bare list."""
    postings: list[ScoredPosting]


SYSTEM_PROMPT = """You are a job-fit scorer for a B2B sales professional.

Output ONLY a JSON object with this exact shape:

{
  "dimensions": {
    "cv_match": <int 1..5>,
    "north_star": <int 1..5>,
    "comp": <int 1..5>,
    "cultural_signals": <int 1..5>,
    "red_flags": <int 1..5>
  },
  "reasons": [<short reason>, ...],
  "legitimacy_signals": [<short signal>, ...]
}

Dimension scoring rules (1=worst, 5=best for ALL of them):
- cv_match: skill + experience overlap with USER profile
- north_star: alignment with the candidate's PROFILE.target_roles primary list
- comp: salary vs market for this role + level (5=top quartile, 1=well below)
- cultural_signals: company quality, growth trajectory, remote policy, team
- red_flags: 5 = NO concerning flags, 1 = many concerning flags
  (deal-breakers, contradictions, vague comp, ghost-job patterns)

Reasons:
- 3-5 short phrases. Cite specific JD terms.
- Be honest, not optimistic.

Legitimacy signals (ghost-job indicators, separate from score):
- Generic JD with no specifics about the team or product
- Vague salary or "competitive" instead of a range
- Contradictions ("entry-level role requires 8+ years experience")
- Excessive nice-to-haves with no clear must-haves
- Re-posted multiple times with no changes (caller will check)
- Empty list = no concerning signals.

HARD COMMUTE RULE:
- If the USER PROFILE notes "will not relocate" AND the role is on-site
  or hybrid AND the posting's location is outside the candidate's
  commute radius (or in a different country), this is a HARD
  DEAL-BREAKER. Set red_flags=1, cv_match<=2, cultural_signals<=2,
  and add 'outside commute radius' to legitimacy_signals.
  (Note: in most cases the caller will short-circuit these BEFORE
  reaching you, but this rule exists as a defense in depth.)

Title-suffix tells: 'East', 'West', 'North', 'South', '-Region',
'(Region)' suffixes commonly mean the role covers a non-local
territory or requires extensive travel from a non-Detroit office.
If the company HQ / posting location is outside the commute radius,
treat as HARD DEAL-BREAKER above.

Output ONLY the JSON object. No markdown fence, no prose."""


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    fenced = re.search(
        r"```(?:json)?\s*(.+?)\s*```", text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first:last + 1])
        except json.JSONDecodeError:
            return None
    return None


def _profile_summary(profile) -> str:
    """Compact text summary of the candidate profile for the LLM
    prompt. Avoids dumping the whole YAML; keeps prompt small."""
    primary_roles = profile.target_roles.primary or []
    superpowers = profile.narrative.superpowers or []
    arch_lines = []
    for a in profile.target_roles.archetypes[:6]:
        if a.fit == "skip":
            continue
        arch_lines.append(f"  - {a.name} (fit={a.fit})")
    return "\n".join([
        f"Candidate: {profile.candidate.full_name or '(unset)'} "
        f"in {profile.candidate.location or '(unset)'}",
        f"Headline: {profile.narrative.headline or '(unset)'}",
        f"Primary target roles: {', '.join(primary_roles) or '(unset)'}",
        f"Archetypes:\n{chr(10).join(arch_lines) or '  (none)'}",
        f"Superpowers: {', '.join(superpowers) or '(unset)'}",
        f"Comp target: {profile.compensation.target_range_usd or '(unset)'} "
        f"(walk-away: {profile.compensation.minimum_usd or '(unset)'})",
        f"Location pref: {profile.location.workplace_preference} "
        f"({profile.location.primary_city or 'no city'}"
        + (f", zip {profile.location.primary_zip}"
           if profile.location.primary_zip else "")
        + (f", on-site/hybrid must be within "
           f"{profile.location.onsite_max_miles} miles"
           if profile.location.onsite_max_miles else "")
        + (", will not relocate"
           if not profile.location.willing_to_relocate else ", open to relocation")
        + ")",
    ])


BATCH_SYSTEM_PROMPT = """You are a job-fit scorer for a B2B sales professional.

You will receive N job postings labeled [posting_1], [posting_2], etc.
Output ONLY a JSON object with this exact shape:

{
  "results": [
    {"dimensions": {...}, "reasons": [...], "legitimacy_signals": [...]},
    ...one entry per posting, IN THE SAME ORDER...
  ]
}

Each result's dimensions object MUST contain: cv_match, north_star,
comp, cultural_signals, red_flags -- all integers 1..5. The same
scoring rules and HARD COMMUTE RULE from the single-posting prompt
apply to each posting independently.

Dimension scoring rules (1=worst, 5=best):
- cv_match: skill + experience overlap with USER profile
- north_star: alignment with PROFILE.target_roles primary list
- comp: salary vs market for this role + level
- cultural_signals: company quality, growth, team, remote policy
- red_flags: 5 = NO concerning flags, 1 = many flags
  (deal-breakers, contradictions, vague comp, ghost-job patterns)

Reasons: 3-5 short phrases per posting. Cite specific JD terms.
Legitimacy signals: ghost-job indicators (generic JD, vague comp,
contradictions, excessive nice-to-haves). Empty list = none.

Output ONLY the JSON object. No markdown fence, no prose. The
"results" list MUST have exactly N entries in input order."""


def _score_one_from_parsed(
    inp: JobScoreInput,
    archetype: str,
    seniority_boost: bool,
    parsed: dict | None,
    region_adjust: float,
    state_ok: bool | None,
    weights: dict[str, float] | None,
    raw_excerpt: str,
    trace_id: str,
) -> ScoredPosting | None:
    """Build a ScoredPosting from a single parsed result dict (one entry
    of a batch's "results" list, OR the whole single-call output).
    Returns None if the parsed dict is structurally unusable so the
    caller can decide to retry / fall back. Applies the same dim-fill,
    seniority/region/state nudges, and weighting as _score_blocking."""
    if not isinstance(parsed, dict):
        return None
    dims_raw = parsed.get("dimensions") or {}
    if not isinstance(dims_raw, dict):
        dims_raw = {}
    dims = {k: dims_raw.get(k, 3) for k in DIMENSION_KEYS}
    if seniority_boost:
        dims["north_star"] = min(5.0, float(dims["north_star"]) + 0.5)
    if region_adjust:
        dims["north_star"] = max(
            1.0, min(5.0, float(dims["north_star"]) + region_adjust),
        )
    if state_ok is False:
        dims["cultural_signals"] = max(
            1.0, float(dims["cultural_signals"]) - 0.5,
        )
    global_score = weighted_score(dims, weights)
    legit_signals = parsed.get("legitimacy_signals") or []
    if not isinstance(legit_signals, list):
        legit_signals = [str(legit_signals)]
    legit_signals = [str(s).strip() for s in legit_signals if str(s).strip()]
    # Phase 13 (Batch 6): merge in deterministic legitimacy signals
    # (URL classification + repost cadence). Best-effort -- failure
    # here just means the LLM-only signals stand.
    try:
        from core import database
        from core.legitimacy import collect_signals
        deterministic = collect_signals(
            inp.title, inp.company, inp.url,
            database.find_recent_company_postings,
        )
        for s in deterministic:
            if s and s not in legit_signals:
                legit_signals.append(s)
    except Exception as e:
        log_event(
            trace_id, "DEBUG", "skill.job_score",
            f"deterministic legitimacy signals skipped: "
            f"{type(e).__name__}: {e}",
        )
    legit = Legitimacy(
        tier=legitimacy_tier(len(legit_signals)),
        signals=legit_signals[:6],
    )
    reasons = parsed.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    try:
        return ScoredPosting(
            title=inp.title or "(unknown)",
            company=inp.company or "(unknown)",
            location=inp.location or "(unknown)",
            location_type=inp.location_type or "onsite",
            salary_range=inp.salary_range,
            seniority=inp.seniority or "mid",
            url=inp.url or "",
            archetype=archetype,
            score=global_score,
            dimensions=dims,
            recommendation=recommendation_band(global_score),
            reasons=reasons,
            legitimacy=legit,
        )
    except Exception as e:
        log_event(
            trace_id, "WARNING", "skill.job_score",
            f"output validation failed: {type(e).__name__}: {e}; "
            f"raw[:200]={raw_excerpt[:200]!r}",
        )
        return None


def _score_blocking(
    inp: JobScoreInput,
    profile_summary_text: str,
    archetype: str,
    seniority_boost: bool,
    trace_id: str,
    model_id: str,
    client: OllamaClient | None = None,
    region_adjust: float = 0.0,
    state_ok: bool | None = None,
    weights: dict[str, float] | None = None,
) -> ScoredPosting:
    client = client or OllamaClient()
    prompt_parts = [
        "USER PROFILE:",
        profile_summary_text,
        "",
        f"DETECTED ARCHETYPE: {archetype}",
        f"TITLE SENIORITY BOOST: {'yes' if seniority_boost else 'no'}",
        "",
        "JOB POSTING:",
        json.dumps({
            "title": inp.title,
            "company": inp.company,
            "location": inp.location,
            "location_type": inp.location_type,
            "salary_range": inp.salary_range,
            "industry": inp.industry,
            "seniority": inp.seniority,
            "key_requirements": inp.key_requirements,
            "deal_breakers": inp.deal_breakers,
            "relevance_signals": inp.relevance_signals,
            "description_excerpt": (inp.description or "")[:1500],
        }, indent=2),
    ]
    if (inp.workplace_pref and inp.workplace_pref != "all"
            and inp.location_type
            and inp.workplace_pref != inp.location_type):
        prompt_parts.append(
            f"\nNOTE: candidate prefers workplace='{inp.workplace_pref}' "
            f"but this posting is location_type='{inp.location_type}'. "
            f"Penalize cv_match and cultural_signals accordingly."
        )
    prompt = "\n".join(prompt_parts)
    log_event(
        trace_id, "INFO", "skill.job_score",
        f"scoring title={inp.title[:60]!r} archetype={archetype} "
        f"model={model_id}",
    )
    try:
        raw = client.generate(
            model=model_id, prompt=prompt, system=SYSTEM_PROMPT,
            format_json=True, trace_id=trace_id,
        )
    except LLMError as e:
        log_event(
            trace_id, "WARNING", "skill.job_score",
            f"LLM call failed: {type(e).__name__}: {e}; returning floor",
        )
        return _floor_scored(inp, archetype,
                             reason=f"LLM call failed: {type(e).__name__}")
    parsed = _extract_json(raw)
    scored = _score_one_from_parsed(
        inp, archetype, seniority_boost, parsed,
        region_adjust, state_ok, weights, raw, trace_id,
    )
    if scored is None:
        return _floor_scored(
            inp, archetype, reason="output validation failed",
        )
    return scored


def _score_batch_blocking(
    items: list[tuple[JobScoreInput, str, bool, float, bool | None,
                      dict[str, float] | None]],
    profile_summary_text: str,
    trace_id: str,
    model_id: str,
    client: OllamaClient | None = None,
) -> list[ScoredPosting]:
    """Phase 13: score N postings in a single LLM call.

    `items` is a list of (input, archetype, seniority_boost, region_adjust,
    state_ok, weights) tuples -- everything the per-item math needs --
    pre-computed by the caller (which has the profile in scope).

    Returns a list[ScoredPosting] of length len(items), in the same
    order. On any batch parse failure we fall back to per-item
    `_score_blocking` for the affected items so the caller never sees
    a length mismatch.
    """
    if not items:
        return []
    client = client or OllamaClient()
    n = len(items)

    # Build the multi-posting prompt. Each posting gets its own labeled
    # block so the LLM can preserve order.
    posting_lines: list[str] = []
    archetype_lines: list[str] = []
    boost_lines: list[str] = []
    for i, (inp, archetype, boost, _ra, _so, _w) in enumerate(items, start=1):
        posting_lines.append(
            f"[posting_{i}]\n"
            + json.dumps({
                "title": inp.title,
                "company": inp.company,
                "location": inp.location,
                "location_type": inp.location_type,
                "salary_range": inp.salary_range,
                "industry": inp.industry,
                "seniority": inp.seniority,
                "key_requirements": inp.key_requirements,
                "deal_breakers": inp.deal_breakers,
                "relevance_signals": inp.relevance_signals,
                "description_excerpt": (inp.description or "")[:1500],
            }, indent=2)
        )
        archetype_lines.append(f"  posting_{i}: {archetype}")
        boost_lines.append(f"  posting_{i}: {'yes' if boost else 'no'}")

    prompt = "\n".join([
        "USER PROFILE:",
        profile_summary_text,
        "",
        f"DETECTED ARCHETYPES per posting (n={n}):",
        *archetype_lines,
        "",
        "TITLE SENIORITY BOOST per posting:",
        *boost_lines,
        "",
        f"POSTINGS (score all {n} in order):",
        *posting_lines,
    ])
    log_event(
        trace_id, "INFO", "skill.job_score",
        f"batch scoring n={n} model={model_id}",
    )

    # Try the batch call. Any LLM failure -> per-item fallback for the
    # whole batch (~N×slower but correct).
    try:
        raw = client.generate(
            model=model_id, prompt=prompt, system=BATCH_SYSTEM_PROMPT,
            format_json=True, trace_id=trace_id,
        )
    except LLMError as e:
        log_event(
            trace_id, "WARNING", "skill.job_score",
            f"batch LLM call failed: {type(e).__name__}: {e}; "
            f"falling back to per-item",
        )
        return [
            _score_blocking(
                inp, profile_summary_text, archetype, boost,
                trace_id, model_id, client, ra, so, w,
            )
            for (inp, archetype, boost, ra, so, w) in items
        ]

    parsed = _extract_json(raw) or {}
    results = parsed.get("results")
    if not isinstance(results, list) or len(results) != n:
        log_event(
            trace_id, "WARNING", "skill.job_score",
            f"batch parse failed (got {type(results).__name__} "
            f"len={len(results) if isinstance(results, list) else 'n/a'} "
            f"expected list of {n}); falling back to per-item",
        )
        return [
            _score_blocking(
                inp, profile_summary_text, archetype, boost,
                trace_id, model_id, client, ra, so, w,
            )
            for (inp, archetype, boost, ra, so, w) in items
        ]

    # Per-item conversion. If a single entry is structurally bad,
    # fall back to per-item LLM call for THAT entry only.
    out: list[ScoredPosting] = []
    for i, (inp, archetype, boost, ra, so, w) in enumerate(items):
        scored = _score_one_from_parsed(
            inp, archetype, boost, results[i],
            ra, so, w, raw, trace_id,
        )
        if scored is None:
            log_event(
                trace_id, "WARNING", "skill.job_score",
                f"batch item {i} malformed; per-item retry "
                f"title={inp.title[:60]!r}",
            )
            scored = _score_blocking(
                inp, profile_summary_text, archetype, boost,
                trace_id, model_id, client, ra, so, w,
            )
        out.append(scored)
    return out


def _floor_scored(inp: JobScoreInput, archetype: str,
                  reason: str) -> ScoredPosting:
    """Sentinel value when scoring fails. Always 'skip' band."""
    floor_dims = {k: 1.0 for k in DIMENSION_KEYS}
    return ScoredPosting(
        title=inp.title or "(unknown)",
        company=inp.company or "(unknown)",
        location=inp.location or "(unknown)",
        location_type=inp.location_type or "onsite",
        salary_range=inp.salary_range,
        seniority=inp.seniority or "mid",
        url=inp.url or "",
        archetype=archetype,
        score=1.0,
        dimensions=floor_dims,
        recommendation="skip",
        reasons=[reason],
        legitimacy=Legitimacy(tier="caution", signals=[reason]),
    )


def _commute_skipped(
    inp: JobScoreInput, archetype: str, reason: str,
    miles: float | None,
) -> ScoredPosting:
    """Phase 12.5: hard-commute-fail short-circuit. Score=2.0 (in skip
    band but not absolute floor), red_flags=1, all other dims=2 to
    signal 'wasn't even evaluated for fit -- pure deal-breaker'."""
    dims = {k: 2.0 for k in DIMENSION_KEYS}
    dims["red_flags"] = 1.0
    return ScoredPosting(
        title=inp.title or "(unknown)",
        company=inp.company or "(unknown)",
        location=inp.location or "(unknown)",
        location_type=inp.location_type or "onsite",
        salary_range=inp.salary_range,
        seniority=inp.seniority or "mid",
        url=inp.url or "",
        archetype=archetype,
        score=2.0,
        dimensions=dims,
        recommendation="skip",
        reasons=[
            f"⛔ {reason}",
            "(commute-gated; not LLM-scored)",
        ],
        legitimacy=Legitimacy(
            tier="high", signals=["outside commute radius"],
        ),
    )


class JobScoreSkill(BaseSkill):
    name: ClassVar[str] = "job_score"
    description: ClassVar[str] = (
        "Score job postings via 5-dim 1-5 weighted rubric "
        "with archetype detection and posting-legitimacy assessment. "
        "Phase 13: batches N postings per LLM call."
    )
    version: ClassVar[str] = "2.1.0"
    requires_gpu: ClassVar[bool] = True
    input_schema: ClassVar[type[BaseModel]] = JobScoreInput
    output_schema: ClassVar[type[BaseModel]] = JobScoreBatchOutput
    accepts_list: ClassVar[bool] = True
    output_is_list: ClassVar[bool] = True

    def validate_input(self, raw):  # noqa: ANN001
        """Phase 13: pipeline gives us a list of dicts (from job_extract
        fan-out). Wrap into JobScoreBatchInput. Single-dict input still
        works for direct test/CLI use -- returns a JobScoreInput so the
        existing per-item code path executes."""
        if isinstance(raw, list):
            return JobScoreBatchInput(postings=[
                item if isinstance(item, JobScoreInput)
                else JobScoreInput(**item)
                for item in raw
            ])
        if isinstance(raw, JobScoreInput):
            return raw
        if isinstance(raw, dict):
            return JobScoreInput(**raw)
        raise SkillError(
            self.name,
            f"expected list or dict, got {type(raw).__name__}",
            "SEN-system",
        )

    def _resolve_per_item(self, inp: JobScoreInput, profile, trace_id: str):
        """Compute everything _score_blocking / _score_batch_blocking
        need beyond the LLM call: archetype, seniority boost, region
        nudge, state whitelist, per-archetype weights, AND the commute
        gate result. Returns (archetype, boost, region_adjust, state_ok,
        weights, commute_skip_or_None).

        commute_skip_or_None is a pre-built ScoredPosting if the gate
        fired, else None (caller must score via LLM)."""
        archetype, _hits = detect_archetype(
            inp.title, inp.description,
            profile.target_roles.archetypes,
        )
        is_out, miles, reason = outside_commute(
            profile, inp.location_type, inp.location,
        )
        if is_out:
            log_event(
                trace_id, "INFO", "skill.job_score",
                f"COMMUTE GATE skip title={inp.title[:60]!r} "
                f"loc={inp.location[:40]!r} miles={miles}",
            )
            return (archetype, False, 0.0, None, None,
                    _commute_skipped(inp, archetype, reason, miles))
        boost = has_seniority_boost(inp.title, profile)
        region_adjust = region_score_adjustment(
            inp.title, inp.location, profile,
        )
        state_ok = state_in_whitelist(inp.location, profile)
        weights = weights_for_archetype(
            archetype, profile.target_roles.archetypes,
        )
        return (archetype, boost, region_adjust, state_ok, weights, None)

    async def execute(
        self,
        input_data: BaseModel,
        trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        model_name_or_id: str = (
            (context or {}).get("model") or config.WORKER_MODEL
        )
        cfg = INFERENCE_CLIENT.model_registry.get(model_name_or_id)
        backend_model_id = cfg.model_id if cfg else model_name_or_id
        profile = load_profile(trace_id)
        profile_summary = _profile_summary(profile)

        # Single-item path (back-compat for tests + CLI).
        if isinstance(input_data, JobScoreInput):
            arch, boost, ra, so, w, gate = self._resolve_per_item(
                input_data, profile, trace_id,
            )
            if gate is not None:
                return gate
            return await asyncio.to_thread(
                _score_blocking, input_data, profile_summary,
                arch, boost, trace_id, backend_model_id,
                None, ra, so, w,
            )

        # Batch path.
        if not isinstance(input_data, JobScoreBatchInput):
            raise SkillError(
                self.name,
                f"expected JobScoreInput or JobScoreBatchInput, "
                f"got {type(input_data).__name__}",
                trace_id,
            )

        # Resolve per-item context AND short-circuit commute-gated
        # postings before any LLM call.
        results: list[ScoredPosting | None] = [None] * len(input_data.postings)
        to_llm: list[tuple[int, JobScoreInput, str, bool, float,
                           bool | None, dict[str, float] | None]] = []
        gated = 0
        for i, inp in enumerate(input_data.postings):
            arch, boost, ra, so, w, gate = self._resolve_per_item(
                inp, profile, trace_id,
            )
            if gate is not None:
                results[i] = gate
                gated += 1
            else:
                to_llm.append((i, inp, arch, boost, ra, so, w))

        # Chunk what's left into batches and run them.
        batch_size = max(1, int(getattr(config, "JOB_SCORE_BATCH_SIZE", 5)))
        log_event(
            trace_id, "INFO", "skill.job_score",
            f"batch dispatch total={len(input_data.postings)} "
            f"commute_gated={gated} llm_items={len(to_llm)} "
            f"batch_size={batch_size}",
        )
        for chunk_start in range(0, len(to_llm), batch_size):
            chunk = to_llm[chunk_start:chunk_start + batch_size]
            chunk_items = [
                (inp, arch, boost, ra, so, w)
                for (_idx, inp, arch, boost, ra, so, w) in chunk
            ]
            scored_chunk = await asyncio.to_thread(
                _score_batch_blocking,
                chunk_items, profile_summary, trace_id, backend_model_id,
                None,
            )
            for (idx, *_rest), scored in zip(chunk, scored_chunk):
                results[idx] = scored

        # All slots filled (gate paths + LLM paths).
        return JobScoreBatchOutput(postings=[
            r for r in results if r is not None
        ])
