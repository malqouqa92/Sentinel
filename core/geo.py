"""Phase 12.5 -- offline geo helpers for the commute-distance gate.

Uses ``pgeocode`` (offline US-zip dataset, no API key, no rate limit)
to convert ZIP codes and "City, ST" strings into lat/long, then runs
the haversine formula for great-circle miles.

The commute gate (``outside_commute``) is the policy used by
``skills/job_score.py`` to short-circuit the LLM when an on-site or
hybrid posting falls outside the candidate's reachable radius and
they cannot relocate. Saves Qwen GPU cost AND prevents the LLM from
generating optimistic-looking scores for hard deal-breakers.

The pgeocode lookup is cached at module level (the dataset is ~3 MB
and we only need a Nominatim instance per country). Failures are
silent: callers MUST treat ``None`` distance as "unknown -- don't
gate" so the pipeline degrades gracefully on weird location strings.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

# pgeocode is a heavy import (loads CSV) -- defer until first call.
_NOMI: Any = None


def _nomi() -> Any:
    """Return a process-singleton pgeocode.Nominatim('us'). Loaded once."""
    global _NOMI
    if _NOMI is None:
        import pgeocode  # noqa: PLC0415
        _NOMI = pgeocode.Nominatim("us")
    return _NOMI


@lru_cache(maxsize=2048)
def zip_to_latlong(zip_str: str) -> tuple[float, float] | None:
    """Look up (lat, lng) for a 5-digit US ZIP. Returns None on miss
    or if the input is not a 5-digit ZIP."""
    if not zip_str:
        return None
    z = str(zip_str).strip().split("-")[0]  # accept "48125-1234"
    if not (len(z) == 5 and z.isdigit()):
        return None
    try:
        r = _nomi().query_postal_code(z)
        lat = float(r.latitude)
        lng = float(r.longitude)
    except Exception:
        return None
    if math.isnan(lat) or math.isnan(lng):
        return None
    return (lat, lng)


# US states (full + 2-letter) for parsing "City, ST[, US]" strings.
_STATE_2 = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
}


def parse_city_state(text: str) -> tuple[str, str] | None:
    """Best-effort: 'Detroit, MI, US' or 'Detroit, MI' or 'Detroit, MI 48125'
    -> ('Detroit', 'MI'). Returns None when the shape is wrong (Canadian
    cities, 'Remote', 'Unknown', etc)."""
    if not text:
        return None
    s = text.strip().rstrip(".")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    if parts[-1].upper() in ("US", "USA", "UNITED STATES"):
        parts = parts[:-1]
    if len(parts) < 2:
        return None
    city = parts[0]
    state_token = parts[1].split()[0].upper() if parts[1].split() else ""
    if state_token in _STATE_2:
        return (city, state_token)
    return None


@lru_cache(maxsize=4096)
def city_state_to_latlong(city: str, state: str) -> tuple[float, float] | None:
    """Look up the centroid of a city+state via pgeocode's place_name
    column. Crude but adequate for a 20-mile gate. Returns None on miss."""
    if not city or not state:
        return None
    try:
        df = _nomi()._data
        mask = (
            (df["state_code"].str.upper() == state.upper())
            & (df["place_name"].str.lower() == city.lower())
        )
        rows = df.loc[mask]
        if rows.empty:
            return None
        lat = float(rows["latitude"].median())
        lng = float(rows["longitude"].median())
        if math.isnan(lat) or math.isnan(lng):
            return None
        return (lat, lng)
    except Exception:
        return None


def haversine_miles(
    lat1: float, lng1: float, lat2: float, lng2: float,
) -> float:
    """Great-circle distance in statute miles. Earth radius 3958.8 mi."""
    R = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def distance_miles_from_zip(zip_str: str, location_text: str) -> float | None:
    """Distance in miles from a known US zip to a posting's location text.
    Tries (in order):
      1. Embedded 5-digit zip in location_text
      2. (City, ST) via parse_city_state -> city_state_to_latlong
    Returns None when the location can't be resolved (e.g. 'Toronto,
    Ontario', 'Remote', 'unknown'). Callers should treat None as
    'don't gate -- fall through to LLM.'"""
    src = zip_to_latlong(zip_str)
    if src is None:
        return None
    if not location_text:
        return None
    m = re.search(r"\b(\d{5})(?:-\d{4})?\b", location_text)
    if m:
        dst = zip_to_latlong(m.group(1))
        if dst is not None:
            return round(haversine_miles(*src, *dst), 1)
    cs = parse_city_state(location_text)
    if cs is not None:
        dst = city_state_to_latlong(*cs)
        if dst is not None:
            return round(haversine_miles(*src, *dst), 1)
    return None


# Foreign-country tokens that indicate a posting is outside the US.
# When the candidate's primary_zip is US AND they cannot relocate, any
# of these in the location_text is treated as a hard commute fail
# (pgeocode's US dataset can't measure the distance, but we don't need
# to -- it's simply not commutable).
_FOREIGN_TOKENS = {
    "canada", "ontario", "quebec", "british columbia", "alberta", "manitoba",
    "saskatchewan", "nova scotia", "new brunswick", "newfoundland",
    "united kingdom", "england", "scotland", "wales", "ireland",
    "germany", "france", "spain", "italy", "netherlands", "belgium",
    "australia", "new zealand", "india", "singapore", "japan",
    "mexico", "brazil",
}


def _looks_foreign(location_text: str) -> str | None:
    """Word-boundary match so 'Indiana' doesn't trip 'india', etc."""
    if not location_text:
        return None
    low = location_text.lower()
    for tok in _FOREIGN_TOKENS:
        if re.search(r"\b" + re.escape(tok) + r"\b", low):
            return tok
    return None


def outside_commute(
    profile: Any, location_type: str, location_text: str,
) -> tuple[bool, float | None, str]:
    """Decide whether a posting falls outside the candidate's commute.

    Returns ``(is_outside, distance_miles, reason)``. ``is_outside`` is
    True when:
      * profile.location.willing_to_relocate is False
      * location_type in {'onsite', 'hybrid'}  (remote bypasses)
      * AND any of:
          (a) profile.location.primary_zip is US, posting is in a known
              foreign country -> True with miles=None
          (b) distance_miles_from_zip > onsite_max_miles
              (requires primary_zip + onsite_max_miles set)

    Unknown US locations -> ``(False, None, '...')`` so the LLM still
    gets a shot at scoring. Conservative by design.
    """
    loc = getattr(profile, "location", None)
    if loc is None:
        return (False, None, "no location preferences in profile")
    if loc.willing_to_relocate:
        return (False, None, "candidate is willing to relocate")
    if (location_type or "").lower() == "remote":
        return (False, None, "remote -- commute irrelevant")
    foreign = _looks_foreign(location_text)
    if foreign and loc.primary_zip:
        return (
            True, None,
            f"posting in {foreign} -- candidate cannot relocate "
            f"from US zip {loc.primary_zip}",
        )
    if not loc.primary_zip or not loc.onsite_max_miles:
        return (False, None, "no primary_zip or onsite_max_miles set")
    miles = distance_miles_from_zip(loc.primary_zip, location_text)
    if miles is None:
        return (False, None, f"could not geocode location: {location_text!r}")
    if miles > loc.onsite_max_miles:
        return (
            True, miles,
            f"{miles:.0f} mi from {loc.primary_zip} exceeds the "
            f"{loc.onsite_max_miles}-mi commute cap; "
            f"candidate cannot relocate",
        )
    return (False, miles, f"within {loc.onsite_max_miles}-mi commute")
