"""Phase 12 -- candidate profile loader.

Reads workspace/persona/PROFILE.yml into a typed Pydantic model. Used
by job_scrape (title filter + avoid list + workplace preference) and
by job_score (Batch 2: archetype weights + comp + narrative).

Loader is lazy + cheap: the YAML is small. Re-read on every call so
the user can edit PROFILE.yml mid-session and have the next /jobsearch
pick it up without restarting the bot.

Missing or malformed PROFILE.yml never crashes the pipeline -- the
loader returns a default instance and logs a WARNING. Empty / partial
profiles are accepted: the title filter just becomes "let everything
through".
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from core import config
from core.logger import log_event


_FIT_VALUES = ("primary", "secondary", "adjacent", "skip")
_WORKPLACE_VALUES = ("on-site", "hybrid", "remote", "all")


class Candidate(BaseModel):
    full_name: str = ""
    email: str = ""
    location: str = ""
    linkedin: str = ""


class Archetype(BaseModel):
    name: str
    level: str = ""
    fit: Literal["primary", "secondary", "adjacent", "skip"] = "primary"
    keywords: list[str] = Field(default_factory=list)
    # Phase 13: per-archetype dimension weights (optional override).
    # Keys must match archetypes.DIMENSION_KEYS; values are normalized
    # to sum=1.0 at lookup time. Empty dict -> use the preset for this
    # archetype name (or the global default if no preset exists).
    weights: dict[str, float] = Field(default_factory=dict)


class TitleFilter(BaseModel):
    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)
    seniority_boost: list[str] = Field(default_factory=list)
    # Phase 13: region preference. Keywords appearing in title or location
    # text. region_boost adds +0.5 to north_star (like seniority_boost);
    # region_avoid subtracts 0.5 (capped). Symmetric, intentionally light --
    # the hard commute gate already handles distance; this just nudges
    # ambiguous remote/multi-state postings.
    region_boost: list[str] = Field(default_factory=list)
    region_avoid: list[str] = Field(default_factory=list)

    def normalize(self) -> "TitleFilter":
        """Lowercase + dedupe all keyword lists."""
        return TitleFilter(
            positive=sorted({k.strip().lower() for k in self.positive
                             if k.strip()}),
            negative=sorted({k.strip().lower() for k in self.negative
                             if k.strip()}),
            seniority_boost=sorted({k.strip().lower()
                                    for k in self.seniority_boost
                                    if k.strip()}),
            region_boost=sorted({k.strip().lower()
                                 for k in self.region_boost
                                 if k.strip()}),
            region_avoid=sorted({k.strip().lower()
                                 for k in self.region_avoid
                                 if k.strip()}),
        )


class TargetRoles(BaseModel):
    primary: list[str] = Field(default_factory=list)
    archetypes: list[Archetype] = Field(default_factory=list)


class Narrative(BaseModel):
    headline: str = ""
    exit_story: str = ""
    superpowers: list[str] = Field(default_factory=list)
    proof_points: list[dict] = Field(default_factory=list)


class Compensation(BaseModel):
    target_range_usd: str = ""
    minimum_usd: str = ""


class LocationPref(BaseModel):
    primary_city: str = ""
    primary_zip: str = ""              # for on-site distance gating
    workplace_preference: Literal["on-site", "hybrid", "remote", "all"] = "all"
    onsite_max_miles: int | None = None  # max commute for on-site/hybrid
    willing_to_relocate: bool = False
    visa_status: str = ""
    # Phase 13: state-level whitelist for remote / hybrid roles where
    # the city is unknown / unresolvable. Two-letter US state codes.
    # Empty list = no whitelist (accept any US state). Posting whose
    # location text resolves to a state outside this list will get
    # a small score penalty (NOT a hard skip -- the commute gate
    # handles hard skips for on-site).
    accepted_states: list[str] = Field(default_factory=list)


class Profile(BaseModel):
    candidate: Candidate = Field(default_factory=Candidate)
    target_roles: TargetRoles = Field(default_factory=TargetRoles)
    title_filter: TitleFilter = Field(default_factory=TitleFilter)
    avoid_companies: list[str] = Field(default_factory=list)
    narrative: Narrative = Field(default_factory=Narrative)
    compensation: Compensation = Field(default_factory=Compensation)
    location: LocationPref = Field(default_factory=LocationPref)

    @field_validator("avoid_companies", mode="before")
    @classmethod
    def _coerce_avoid(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return [str(s).strip() for s in v if str(s).strip()]


def profile_path() -> Path:
    """Canonical PROFILE.yml location. Test code can monkeypatch
    config.PERSONA_DIR to redirect."""
    return config.PERSONA_DIR / "PROFILE.yml"


def example_path() -> Path:
    return config.PERSONA_DIR / "PROFILE.example.yml"


def load_profile(trace_id: str = "SEN-system") -> Profile:
    """Read + validate PROFILE.yml. Returns a default Profile() on any
    read/parse/validation failure (logged at WARNING).

    Re-reads on every call -- callers can rely on freshness without
    needing to invalidate caches when the user edits the file."""
    p = profile_path()
    if not p.exists():
        return Profile()  # empty defaults; everything passes filter
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        log_event(trace_id, "WARNING", "job_profile",
                  f"PROFILE.yml YAML parse failed -- using defaults. err={e}")
        return Profile()
    if not isinstance(raw, dict):
        log_event(trace_id, "WARNING", "job_profile",
                  "PROFILE.yml top-level is not a mapping -- using defaults")
        return Profile()
    try:
        prof = Profile(**raw)
    except ValidationError as e:
        log_event(trace_id, "WARNING", "job_profile",
                  f"PROFILE.yml schema rejected -- using defaults. err={e}")
        return Profile()
    # Normalize the title filter once on the way out so callers don't
    # have to think about case.
    prof.title_filter = prof.title_filter.normalize()
    return prof


# ---------------------------------------------------------------------
# Title-filter evaluation -- pure, no I/O.
# ---------------------------------------------------------------------

def title_passes(
    title: str, profile: Profile, extra_avoid: list[str] | None = None,
    company: str = "",
) -> bool:
    """Decide whether a posting's title (and company) should survive the
    pre-LLM filter.

    Rules (all case-insensitive substring matches):
      1. If profile.title_filter.positive is non-empty, at least one of
         its keywords MUST appear in the title.
      2. If any profile.title_filter.negative keyword appears in the
         title, the posting is rejected.
      3. If `company` matches any of profile.avoid_companies (or
         extra_avoid passed in from /jobsearch --avoid), reject.
    """
    t = (title or "").lower()
    c = (company or "").lower()
    avoids = list(profile.avoid_companies) + list(extra_avoid or [])
    avoids = [a.strip().lower() for a in avoids if a.strip()]
    for a in avoids:
        if a and (a in c or a in t):
            return False
    pos = profile.title_filter.positive
    neg = profile.title_filter.negative
    for n in neg:
        if n and n in t:
            return False
    if pos:
        if not any(p and p in t for p in pos):
            return False
    return True


def has_seniority_boost(title: str, profile: Profile) -> bool:
    """Whether any seniority_boost keyword appears in the title.
    Used in Batch 2 by the scorer to nudge the level dimension up."""
    t = (title or "").lower()
    return any(k and k in t for k in profile.title_filter.seniority_boost)


# ---------------------------------------------------------------------
# Phase 13: region preference + state whitelist
# ---------------------------------------------------------------------

def region_score_adjustment(
    title: str, location_text: str, profile: Profile,
) -> float:
    """Sum the region nudges. +0.5 per region_boost keyword hit (in
    title OR location), -0.5 per region_avoid hit. Capped at +/-1.0
    so a roundup of buzzwords can't dominate the global score."""
    blob = ((title or "") + " " + (location_text or "")).lower()
    boost_hits = sum(
        1 for k in profile.title_filter.region_boost
        if k and k in blob
    )
    avoid_hits = sum(
        1 for k in profile.title_filter.region_avoid
        if k and k in blob
    )
    raw = 0.5 * boost_hits - 0.5 * avoid_hits
    return max(-1.0, min(1.0, raw))


def state_in_whitelist(
    location_text: str, profile: Profile,
) -> bool | None:
    """True if the posting's state is in profile.location.accepted_states.
    Returns None when the whitelist is empty (no preference) OR when
    the state can't be resolved from location_text. Used by the scorer
    to apply a small penalty when a remote/multi-state posting is in
    a state the candidate doesn't want."""
    accepted = [s.strip().upper() for s in profile.location.accepted_states
                if s.strip()]
    if not accepted:
        return None  # no preference set
    # Lazy import to avoid cycle if geo grows.
    from core.geo import parse_city_state
    cs = parse_city_state(location_text)
    if cs is None:
        return None  # unresolvable; don't penalize on noise
    return cs[1].upper() in accepted
