"""Phase 13 (Batch 5a): adaptive title filter.

After a /jobsearch where the title pre-filter dropped a high fraction
of scraped postings (config.JOB_ADAPTIVE_FILTER_DROP_RATIO), sample the
dropped titles, ask the brain to extract 1..N candidate negative
keywords, and append them to PROFILE.title_filter.negative IN PLACE so
the next /jobsearch is more selective.

Auto-apply by design (per owner request) -- the bot notifies the user
of every change via Telegram, and changes are append-only so the user
can edit /profile to remove anything later.

Safety bounds:
  - never add a keyword that already appears in negative
  - never add a keyword that overlaps with positive or seniority_boost
    (would block target archetypes)
  - cap at ``config.JOB_ADAPTIVE_FILTER_MAX_NEW`` per call
  - skip if total drops < ``config.JOB_ADAPTIVE_FILTER_MIN_DROPS``
  - skip if drop_ratio < ``config.JOB_ADAPTIVE_FILTER_DROP_RATIO``
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from core import config
from core.logger import log_event


SYSTEM_PROMPT = """You analyze rejected job titles to extract common
themes the candidate wants to filter out.

Input: a list of job titles that were dropped from a search because they
didn't match the candidate's target roles.

Output ONLY a JSON object:

{
  "negatives": ["<short keyword>", ...]
}

Rules:
- Each keyword: 1-3 words, lowercase, the most distinctive word(s) that
  the rejected titles share.
- Aim for 1-3 keywords. Quality over quantity. If the dropped titles
  are too varied to find a pattern, return [].
- Avoid generic words like "manager", "director", "senior" alone --
  they would block target roles too. Always pair them with industry/
  domain context (e.g. "insurance manager" not "manager").
- Examples of good extractions:
    Input: ["Insurance Sales Agent", "Insurance Account Manager"]
    Output: {"negatives": ["insurance"]}
    Input: ["Mortgage Loan Officer", "Loan Processor"]
    Output: {"negatives": ["loan", "mortgage"]}

Output ONLY the JSON object. No prose, no fence."""


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


def should_act(dropped_count: int, total_scraped: int) -> bool:
    """Predicate: do the drop counts justify spending a brain call?"""
    if not getattr(config, "JOB_ADAPTIVE_FILTER_ENABLED", True):
        return False
    min_drops = getattr(config, "JOB_ADAPTIVE_FILTER_MIN_DROPS", 10)
    if dropped_count < min_drops:
        return False
    if total_scraped <= 0:
        return False
    ratio = dropped_count / total_scraped
    threshold = getattr(config, "JOB_ADAPTIVE_FILTER_DROP_RATIO", 0.4)
    return ratio >= threshold


def extract_candidates(
    dropped_titles: list[str],
    profile,
    brain_generate,
    trace_id: str,
) -> list[str]:
    """Ask the brain for negative-keyword candidates from a sample of
    dropped titles. Returns a (validated, deduped, capped) list ready
    to append to PROFILE.title_filter.negative.

    ``brain_generate`` is a callable matching OllamaClient.generate's
    signature: ``brain_generate(model=..., prompt=..., system=...,
    format_json=True, trace_id=...) -> str`` (the LLM text response).
    Caller is expected to pass a closure bound to the brain client/model
    so this module stays decoupled from the LLM stack.
    """
    if not dropped_titles:
        return []
    sample_size = getattr(config, "JOB_ADAPTIVE_FILTER_SAMPLE_SIZE", 20)
    sample = dropped_titles[:sample_size]
    cap = getattr(config, "JOB_ADAPTIVE_FILTER_MAX_NEW", 3)

    # Existing keyword sets (lowercase) we must NOT collide with.
    tf = profile.title_filter
    existing_neg = {k.lower().strip() for k in tf.negative if k.strip()}
    blockers = {k.lower().strip()
                for k in (list(tf.positive) + list(tf.seniority_boost))
                if k.strip()}

    prompt = (
        "Dropped job titles (sample):\n"
        + "\n".join(f"- {t}" for t in sample)
    )
    try:
        raw = brain_generate(
            prompt=prompt, system=SYSTEM_PROMPT,
            format_json=True, trace_id=trace_id,
        )
    except Exception as e:
        log_event(
            trace_id, "WARNING", "adaptive_filter",
            f"brain call failed: {type(e).__name__}: {e}",
        )
        return []

    parsed = _extract_json(raw) or {}
    raw_list = parsed.get("negatives") or []
    if not isinstance(raw_list, list):
        return []

    out: list[str] = []
    for item in raw_list:
        kw = str(item or "").strip().lower()
        if not kw or len(kw) < 2 or len(kw) > 40:
            continue
        if kw in existing_neg:
            continue
        if kw in blockers:
            log_event(
                trace_id, "INFO", "adaptive_filter",
                f"refused to add {kw!r} -- collides with positive/boost",
            )
            continue
        # Avoid keywords that would BLOCK target-role substrings (e.g.
        # don't add 'sales' if 'sales' is in any positive / seniority).
        if any(kw in b for b in blockers):
            log_event(
                trace_id, "INFO", "adaptive_filter",
                f"refused to add {kw!r} -- substring of a blocker",
            )
            continue
        if kw not in out:
            out.append(kw)
        if len(out) >= cap:
            break
    return out


def apply_to_profile(new_negatives: list[str], trace_id: str) -> int:
    """Append the new negatives to PROFILE.yml's title_filter.negative
    in place. Atomic write via temp + rename. Returns the count actually
    appended (0 if nothing to do or file missing).

    Re-loads the YAML to avoid stomping on any concurrent edits the user
    might have made. Skips keywords already present.
    """
    if not new_negatives:
        return 0
    profile_path: Path = config.PERSONA_DIR / "PROFILE.yml"
    if not profile_path.exists():
        log_event(
            trace_id, "WARNING", "adaptive_filter",
            f"PROFILE.yml not found at {profile_path}; skipping append",
        )
        return 0
    try:
        text = profile_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
    except Exception as e:
        log_event(
            trace_id, "WARNING", "adaptive_filter",
            f"PROFILE.yml read/parse failed: {type(e).__name__}: {e}",
        )
        return 0
    if not isinstance(data, dict):
        return 0

    tf = data.setdefault("title_filter", {})
    if not isinstance(tf, dict):
        return 0
    neg_list = tf.setdefault("negative", [])
    if not isinstance(neg_list, list):
        return 0

    existing = {str(k).strip().lower() for k in neg_list if str(k).strip()}
    added = 0
    for kw in new_negatives:
        if kw.lower() not in existing:
            neg_list.append(kw)
            existing.add(kw.lower())
            added += 1

    if added == 0:
        return 0

    tmp = profile_path.with_suffix(".yml.tmp")
    try:
        tmp.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        tmp.replace(profile_path)
    except Exception as e:
        log_event(
            trace_id, "WARNING", "adaptive_filter",
            f"PROFILE.yml write failed: {type(e).__name__}: {e}",
        )
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        return 0

    log_event(
        trace_id, "INFO", "adaptive_filter",
        f"appended {added} negatives to PROFILE.yml: {new_negatives}",
    )
    return added
