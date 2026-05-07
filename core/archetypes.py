"""Phase 12 -- archetype detection + scoring weights.
Phase 13 -- per-archetype weights.

Six B2B-sales archetypes preset for Sentinel's first user. Each
archetype has:
  - keywords: substrings the title/JD scanner uses for classification
  - weights: how the 5-dimension score gets weighted into the global
    (Phase 13: each archetype now has its own preset; PROFILE.yml may
     override per-archetype.)

The classifier is deterministic + cheap (substring count). The scorer
LLM is told which archetype was detected so it can frame its judgement,
but the math (weights -> global) lives here.

Custom archetypes can be defined in PROFILE.yml under
target_roles.archetypes -- those override / extend these defaults.
"""
from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Default per-dimension weights (sum to 1.0). Used as fallback when an
# archetype has no preset and PROFILE supplies no override.
# ---------------------------------------------------------------------

DIMENSION_WEIGHTS: Final[dict[str, float]] = {
    "cv_match": 0.30,        # skills + experience overlap
    "north_star": 0.25,      # alignment with PROFILE.target_roles
    "comp": 0.20,            # salary vs market
    "cultural_signals": 0.15,  # team, remote policy, growth
    "red_flags": 0.10,       # higher = fewer flags (positive sense)
}

DIMENSION_KEYS: Final[tuple[str, ...]] = tuple(DIMENSION_WEIGHTS.keys())


# ---------------------------------------------------------------------
# Phase 13 -- per-archetype weight presets. Each row sums to 1.0.
# ---------------------------------------------------------------------

ARCHETYPE_DEFAULT_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    # Primary target -- north_star matters most (this IS the role).
    "Regional Sales Manager": {
        "cv_match": 0.25, "north_star": 0.30, "comp": 0.20,
        "cultural_signals": 0.15, "red_flags": 0.10,
    },
    # Primary target, similar shape to RSM.
    "Territory Sales Manager": {
        "cv_match": 0.25, "north_star": 0.30, "comp": 0.20,
        "cultural_signals": 0.15, "red_flags": 0.10,
    },
    # Secondary -- not the target role, so comp matters more (avoid
    # taking a step down without paying for it).
    "Account Executive": {
        "cv_match": 0.30, "north_star": 0.20, "comp": 0.25,
        "cultural_signals": 0.15, "red_flags": 0.10,
    },
    # Adjacent -- weight skills high (it's a different toolset entirely)
    # and red_flags high (mill jobs are common in the SDR layer).
    "Sales Development Representative": {
        "cv_match": 0.25, "north_star": 0.10, "comp": 0.25,
        "cultural_signals": 0.15, "red_flags": 0.25,
    },
    # Skills-driven role -- cv_match dominates.
    "Sales Operations / RevOps": {
        "cv_match": 0.40, "north_star": 0.20, "comp": 0.20,
        "cultural_signals": 0.10, "red_flags": 0.10,
    },
    # Relationship role -- cultural_signals matters most.
    "Customer Success Manager": {
        "cv_match": 0.25, "north_star": 0.15, "comp": 0.20,
        "cultural_signals": 0.30, "red_flags": 0.10,
    },
}


def _normalize_weights(w: dict[str, float] | None) -> dict[str, float] | None:
    """Coerce a user-supplied weights dict to {dim: float} summing to 1.0.
    Returns None if the input is None, empty, or unusable -- callers fall
    back to DIMENSION_WEIGHTS in that case."""
    if not w:
        return None
    cleaned: dict[str, float] = {}
    for k in DIMENSION_KEYS:
        v = w.get(k)
        try:
            f = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            f = 0.0
        if f < 0:
            f = 0.0
        cleaned[k] = f
    total = sum(cleaned.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in cleaned.items()}


def weights_for_archetype(
    archetype_name: str,
    profile_archetypes: list | None = None,
) -> dict[str, float]:
    """Return the dimension weights for a given archetype.

    Lookup order:
      1. PROFILE.yml archetype with matching name AND non-empty weights.
      2. ARCHETYPE_DEFAULT_WEIGHTS preset for the archetype name.
      3. DIMENSION_WEIGHTS (the global default).
    """
    nl = (archetype_name or "").strip().lower()
    for arch in (profile_archetypes or []):
        try:
            if (arch.name or "").strip().lower() == nl:
                norm = _normalize_weights(getattr(arch, "weights", None))
                if norm is not None:
                    return norm
                break
        except Exception:
            continue
    preset = ARCHETYPE_DEFAULT_WEIGHTS.get(archetype_name)
    if preset is not None:
        return dict(preset)
    return dict(DIMENSION_WEIGHTS)


# ---------------------------------------------------------------------
# Default archetypes (6, sales-flavored). Mirrors PROFILE.example.yml.
# ---------------------------------------------------------------------

class ArchetypeSpec(BaseModel):
    name: str
    keywords: list[str] = Field(default_factory=list)


DEFAULT_ARCHETYPES: Final[list[ArchetypeSpec]] = [
    ArchetypeSpec(name="Regional Sales Manager",
                  keywords=["regional sales", "regional manager",
                            "area sales", "regional director"]),
    ArchetypeSpec(name="Territory Sales Manager",
                  keywords=["territory", "field sales", "outside sales",
                            "territory manager"]),
    ArchetypeSpec(name="Account Executive",
                  keywords=["account executive", "account manager",
                            "named accounts", "enterprise account",
                            "strategic account"]),
    ArchetypeSpec(name="Sales Development Representative",
                  keywords=["sdr", "bdr", "sales development",
                            "business development representative"]),
    ArchetypeSpec(name="Sales Operations / RevOps",
                  keywords=["sales operations", "revops",
                            "revenue operations", "sales ops",
                            "sales analytics"]),
    ArchetypeSpec(name="Customer Success Manager",
                  keywords=["customer success", "csm",
                            "account renewal", "post-sales"]),
]


def _all_archetypes(profile_archetypes: list | None = None) -> list[ArchetypeSpec]:
    """Merge PROFILE.target_roles.archetypes (if any) over the defaults.

    Rules:
      - A PROFILE archetype with `fit='skip'` suppresses any default of
        the same name.
      - A PROFILE archetype with any other fit replaces the default
        entry of the same name AND extends the list with new names.
      - Defaults whose name was not mentioned come through unchanged.
    """
    out: list[ArchetypeSpec] = []
    seen_names: set[str] = set()
    skip_names: set[str] = set()
    for arch in (profile_archetypes or []):
        try:
            fit = getattr(arch, "fit", "primary")
            name = arch.name
            kws = list(getattr(arch, "keywords", None) or [])
        except Exception:
            continue
        if not name:
            continue
        nl = name.lower()
        if fit == "skip":
            skip_names.add(nl)
            continue
        if nl in seen_names:
            continue
        seen_names.add(nl)
        out.append(ArchetypeSpec(name=name, keywords=kws))
    for d in DEFAULT_ARCHETYPES:
        nl = d.name.lower()
        if nl in skip_names or nl in seen_names:
            continue
        out.append(d)
    return out


def detect_archetype(
    title: str,
    description: str = "",
    profile_archetypes: list | None = None,
) -> tuple[str, int]:
    """Classify a posting into one archetype by keyword-hit count.

    Returns ``(archetype_name, hit_count)``. If no keyword matches,
    returns ``("Unknown", 0)``. Title is weighted 3x relative to the
    description body.
    """
    title_l = (title or "").lower()
    desc_l = (description or "").lower()
    best_name = "Unknown"
    best_score = 0
    for arch in _all_archetypes(profile_archetypes):
        score = 0
        for kw in arch.keywords:
            kw_l = kw.lower()
            if not kw_l:
                continue
            score += 3 * title_l.count(kw_l)
            score += desc_l.count(kw_l)
        if score > best_score:
            best_score = score
            best_name = arch.name
    return (best_name, best_score)


# ---------------------------------------------------------------------
# Score math
# ---------------------------------------------------------------------

def weighted_score(
    dimensions: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Combine per-dimension 1-5 scores into a single 1-5 global.
    Missing dimensions are treated as 3 (neutral). Result clamped to
    [1.0, 5.0].

    Phase 13: weights are now per-archetype. Pass an explicit weights
    dict (typically from weights_for_archetype()); falls back to
    DIMENSION_WEIGHTS when None.
    """
    use_weights = weights or DIMENSION_WEIGHTS
    total = 0.0
    for k in DIMENSION_KEYS:
        w = use_weights.get(k, 0.0)
        v = dimensions.get(k)
        try:
            v = float(v) if v is not None else 3.0
        except (TypeError, ValueError):
            v = 3.0
        v = max(1.0, min(5.0, v))
        total += w * v
    return round(max(1.0, min(5.0, total)), 2)


def recommendation_band(global_score: float) -> str:
    """Map a 1-5 global score to one of four action labels:
      apply_now    -- >= 4.5
      worth_applying -- 4.0 .. 4.4
      maybe        -- 3.5 .. 3.9
      skip         -- < 3.5
    """
    if global_score >= 4.5:
        return "apply_now"
    if global_score >= 4.0:
        return "worth_applying"
    if global_score >= 3.5:
        return "maybe"
    return "skip"


# ---------------------------------------------------------------------
# Posting legitimacy -- HTML-only signals for Phase 12.
# ---------------------------------------------------------------------

def legitimacy_tier(concerning_signal_count: int) -> str:
    """Translate the LLM's concerning-signal count into a 3-tier band.
      0-1 -> high
      2   -> caution
      3+  -> suspicious
    """
    if concerning_signal_count >= 3:
        return "suspicious"
    if concerning_signal_count == 2:
        return "caution"
    return "high"
