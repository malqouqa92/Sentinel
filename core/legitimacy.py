"""Phase 13 (Batch 6): lightweight legitimacy signals.

Goal: enrich the LLM-judged legitimacy_signals list with two cheap
deterministic checks that don't need a browser snapshot or any LLM
call.

  1. Repost cadence: same company + similar title appearing in the
     applications table within the last 90 days. A handful of reposts
     of the same role is normal rotation; >=3 reposts is a ghost-job
     signal (the company isn't actually hiring, just harvesting
     resumes).

  2. Apply URL classification: postings that link to a known ATS
     (greenhouse, lever, workday, etc.) are higher-trust; postings
     with NO apply link or with sketchy domains get flagged.

These outputs slot into the existing legitimacy_signals list that
``ScoredPosting.legitimacy.signals`` carries. The legitimacy tier
(high/caution/suspicious) is already a function of signal count, so
adding signals here automatically downgrades suspect postings without
touching the scoring math.

Pure functions -- no I/O beyond the optional DB query inside
``detect_repost_cadence``. Test by passing in the row list directly.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit


# Lowercase domain fragments that indicate a real ATS (high trust).
# We look for substring matches in the apply URL host.
_ATS_DOMAINS: tuple[str, ...] = (
    "greenhouse.io", "lever.co", "workday.com", "myworkdayjobs.com",
    "bamboohr.com", "smartrecruiters.com", "icims.com",
    "ashbyhq.com", "jobvite.com", "jazzhr.com",
    "applytojob.com", "rippling.com", "personio.com",
    "successfactors.com", "taleo.net", "wd5.myworkdaysite.com",
)

# Aggregators -- lower signal but not bad. We don't flag, just don't
# upgrade trust.
_AGGREGATOR_DOMAINS: tuple[str, ...] = (
    "indeed.com", "linkedin.com", "ziprecruiter.com",
    "glassdoor.com", "monster.com", "simplyhired.com",
    "google.com",  # google jobs forwards
)

# Country TLDs we don't apply to (per PROFILE: USA only). A posting
# whose apply URL is a foreign company portal is a soft red flag --
# legitimate jobs CAN have foreign HQs but it's worth surfacing.
_NON_US_CC_TLDS: tuple[str, ...] = (
    ".ru", ".cn", ".in", ".pk", ".ng",
)


def classify_apply_url(url: str) -> tuple[str, str | None]:
    """Return (tier, signal_or_none).

    tier ∈ {"ats", "aggregator", "unknown", "suspicious"}.
    signal_or_none: a short string to add to legitimacy_signals when
    the classification warrants surfacing (only for "suspicious" or
    truly missing URL).
    """
    raw = (url or "").strip().lower()
    if not raw:
        return ("unknown", "no apply URL on posting")
    try:
        host = urlsplit(raw).netloc
    except Exception:
        return ("suspicious", "malformed apply URL")
    if not host:
        return ("suspicious", "malformed apply URL")
    for d in _ATS_DOMAINS:
        if d in host:
            return ("ats", None)
    for d in _AGGREGATOR_DOMAINS:
        if d in host:
            return ("aggregator", None)
    # Foreign TLD heuristic -- last-segment match so we don't false-
    # positive on something like 'india.example.com'.
    for tld in _NON_US_CC_TLDS:
        if host.endswith(tld):
            return ("suspicious", f"non-US apply domain ({tld})")
    return ("unknown", None)


_NON_WORD = re.compile(r"[^a-z0-9]+")


def _normalize_title(t: str) -> str:
    """Lowercase + collapse non-word chars + strip common rank/level
    fluff so 'Senior Regional Sales Manager' and 'Regional Sales Mgr'
    fuzzy-match the same role."""
    s = (t or "").lower()
    s = _NON_WORD.sub(" ", s).strip()
    s = re.sub(
        r"\b(senior|sr|junior|jr|lead|principal|i{1,3}|iv|v|"
        r"associate|assistant|mgr|mngr|exec)\b",
        " ", s,
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_company(c: str) -> str:
    """Lowercase + strip common suffixes so 'AcmeCo Inc.' matches 'AcmeCo'."""
    s = (c or "").lower().strip()
    s = re.sub(
        r"[,.\s]*(inc|llc|corp|corporation|co|ltd|gmbh|company|"
        r"group|holdings)\.?\s*$",
        "", s,
    ).strip()
    return s


def _title_similar(a: str, b: str) -> bool:
    """Tokens overlap >= 50% AND length-normalized lengths are close.
    Cheap Jaccard-on-words; good enough for ghost-job detection without
    pulling in difflib."""
    ta = set(_normalize_title(a).split())
    tb = set(_normalize_title(b).split())
    if not ta or not tb:
        return False
    overlap = len(ta & tb)
    smaller = min(len(ta), len(tb))
    return overlap / smaller >= 0.5


def detect_repost_cadence(
    title: str, company: str, current_url: str,
    fetch_recent_company_postings,
    days: int = 90,
) -> tuple[int, str | None]:
    """Return (repost_count, signal_or_none).

    repost_count is the number of OTHER postings in the applications
    table from the same company with a similar title in the last
    ``days``. The current posting (matched by URL) is excluded.

    signal_or_none is a short string to add to legitimacy_signals when
    repost_count >= 2; None otherwise.

    ``fetch_recent_company_postings`` is a callable that takes
    (company_normalized, days) and returns a list of dicts with at
    least 'title' and 'url' keys. Injected so this module stays
    decoupled from the database; tests pass in a list directly.
    """
    co_norm = _normalize_company(company)
    if not co_norm:
        return (0, None)
    try:
        rows = fetch_recent_company_postings(co_norm, days) or []
    except Exception:
        return (0, None)
    cur_url = (current_url or "").strip().lower()
    matches = 0
    for row in rows:
        row_url = str(row.get("url") or "").strip().lower()
        if row_url and row_url == cur_url:
            continue  # skip self-match (same posting from prior scrape)
        if _title_similar(title, row.get("title", "")):
            matches += 1
    if matches >= 3:
        return (matches, f"reposted ≥{matches}× in last {days}d (ghost-job)")
    if matches >= 2:
        return (matches, f"reposted {matches}× in last {days}d")
    return (matches, None)


def collect_signals(
    title: str, company: str, url: str,
    fetch_recent_company_postings,
    days: int = 90,
) -> list[str]:
    """One-shot: run all lightweight checks and return the list of
    signals to APPEND to ScoredPosting.legitimacy.signals. Order is
    stable (URL classification first, repost cadence second). May be
    empty if nothing concerning surfaced."""
    out: list[str] = []
    _tier, url_sig = classify_apply_url(url)
    if url_sig:
        out.append(url_sig)
    _count, repost_sig = detect_repost_cadence(
        title, company, url, fetch_recent_company_postings, days=days,
    )
    if repost_sig:
        out.append(repost_sig)
    return out
