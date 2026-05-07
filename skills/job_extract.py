"""Job posting extraction skill.

Migrated from subagents/job_extractor.py in Phase 5. The extraction
function (`extract`) is preserved as a sync helper for testability;
the BaseSkill subclass wraps it as the framework-facing API.
"""
import asyncio
import json
import re
from typing import ClassVar

from pydantic import BaseModel, Field, ValidationError, field_validator

from core import config
from core.llm import INFERENCE_CLIENT, LLMError, OllamaClient
from core.logger import log_event
from core.skills import BaseSkill, SkillError

ALLOWED_LOCATION_TYPES = {"onsite", "hybrid", "remote"}
ALLOWED_SENIORITY = {
    "entry", "mid", "senior", "director", "vp", "c-level"
}


class JobExtractInput(BaseModel):
    text: str
    # Phase 12: optional carry-throughs from job_scrape so url +
    # workplace_pref survive the fan-out hop. Other fields on
    # ScrapedPosting (title/company/etc) are duplicates of what the
    # extractor will derive from `text`, so we leave them off.
    url: str = ""
    workplace_pref: str = "all"

    model_config = {"extra": "ignore"}


class JobExtraction(BaseModel):
    title: str
    company: str
    location: str
    location_type: str
    salary_range: str | None = None
    industry: str | None = None
    seniority: str
    key_requirements: list[str] = Field(default_factory=list)
    deal_breakers: list[str] = Field(default_factory=list)
    relevance_signals: list[str] = Field(default_factory=list)
    confidence: float
    # Phase 12 carry-throughs (echoed from JobExtractInput so job_score
    # can post-filter on workplace and so the apps table gets the URL).
    url: str = ""
    description: str = ""
    workplace_pref: str = "all"

    @field_validator("location_type")
    @classmethod
    def _norm_location_type(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in ALLOWED_LOCATION_TYPES:
            raise ValueError(
                f"location_type must be one of "
                f"{sorted(ALLOWED_LOCATION_TYPES)}, got {v!r}"
            )
        return v

    @field_validator("seniority")
    @classmethod
    def _norm_seniority(cls, v: str) -> str:
        v = (v or "").strip().lower()
        v = v.replace("c-suite", "c-level").replace("c level", "c-level")
        v = v.replace("clevel", "c-level").replace(
            "vp/vice president", "vp"
        )
        v = v.replace("vice president", "vp")
        if v.startswith("entry") or v in {"junior", "jr"}:
            v = "entry"
        if v in {"mid-level", "midlevel", "mid level", "intermediate"}:
            v = "mid"
        if v not in ALLOWED_SENIORITY:
            raise ValueError(
                f"seniority must be one of {sorted(ALLOWED_SENIORITY)}, "
                f"got {v!r}"
            )
        return v

    @field_validator("key_requirements")
    @classmethod
    def _cap_requirements(cls, v: list[str]) -> list[str]:
        return v[:5]

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, v))


SYSTEM_PROMPT = """You extract structured data from job postings.

Return ONLY valid JSON with this exact schema. Do not include markdown, prose, or explanations.

{
  "title": string,                  // job title as posted, preserve original capitalization
  "company": string,                // company name, preserve original capitalization
  "location": string,               // city/state or "Remote", preserve original capitalization
  "location_type": string,          // exactly one of: "onsite" | "hybrid" | "remote"
  "salary_range": string | null,    // candidate's pay/compensation only, or null if not listed
  "industry": string | null,        // best guess from context, or null
  "seniority": string,              // exactly one of: "entry" | "mid" | "senior" | "director" | "vp" | "c-level"
  "key_requirements": string[],     // max 5 hard requirements
  "deal_breakers": string[],        // disqualifiers (clearance, relocation, etc.)
  "relevance_signals": string[],    // phrases hinting at sales/ops/B2B/management background
  "confidence": number              // 0.0-1.0 -- your own confidence in this extraction
}

Critical rules for salary_range:
- salary_range is ONLY the candidate's compensation: base salary, hourly rate, OTE, or total comp.
- DO NOT use deal sizes, quota amounts, ARR/MRR, account values, contract values, product prices,
  budget figures, or any other dollar amounts that are NOT the worker's pay.
- If the posting does not state the candidate's pay, salary_range MUST be null.

Other constraints:
- "location_type" and "seniority" MUST use the exact lowercase values listed above. All other
  string fields preserve the original capitalization from the posting.
- "key_requirements" MUST contain at most 5 items.
- If a field is unknown, use an empty string for strings, [] for lists, or null where the schema allows null.
- Always emit valid JSON parseable by Python's json.loads.
"""

REPAIR_SYSTEM_PROMPT = """The previous attempt produced output that failed JSON
parsing or schema validation. Re-emit the SAME extraction as valid JSON
matching the schema below. Do not include any prose or markdown -- JSON only.
Use the allowed enum values for location_type and seniority.
"""


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
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            return None
    return None


def _empty_extraction() -> JobExtraction:
    return JobExtraction(
        title="", company="", location="",
        location_type="onsite", salary_range=None, industry=None,
        seniority="mid", key_requirements=[], deal_breakers=[],
        relevance_signals=[], confidence=0.0,
    )


_default_client: OllamaClient | None = None


def _get_client() -> OllamaClient:
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client


def extract(
    raw_text: str,
    trace_id: str,
    client: OllamaClient | None = None,
    model: str | None = None,
) -> JobExtraction:
    """Sync extraction helper. Calls the LLM once, retries once with a
    repair prompt on parse/validation failure, returns an empty
    JobExtraction with confidence=0.0 if both attempts fail."""
    client = client or _get_client()
    model = model or config.DEFAULT_MODEL

    log_event(
        trace_id, "INFO", "skill.job_extract",
        f"extract starting chars={len(raw_text)} model={model}",
    )

    first_raw = ""
    try:
        first_raw = client.generate(
            model=model, prompt=raw_text, system=SYSTEM_PROMPT,
            format_json=True, trace_id=trace_id,
        )
        parsed = _extract_json(first_raw)
        if parsed is None:
            raise ValueError("JSON parse failed on first attempt")
        return JobExtraction(**parsed)
    except (LLMError, ValidationError, ValueError, TypeError) as e:
        log_event(
            trace_id, "WARNING", "skill.job_extract",
            f"first attempt failed: {type(e).__name__}: "
            f"{str(e)[:200]}; retrying",
        )

    repair_prompt = (
        f"Previous output (invalid):\n{first_raw}\n\n"
        f"Original job posting:\n{raw_text}\n\n"
        "Re-emit corrected JSON now."
    )
    second_raw = ""
    try:
        second_raw = client.generate(
            model=model, prompt=repair_prompt,
            system=REPAIR_SYSTEM_PROMPT, format_json=True,
            trace_id=trace_id,
        )
        parsed = _extract_json(second_raw)
        if parsed is None:
            raise ValueError("JSON parse failed on second attempt")
        return JobExtraction(**parsed)
    except (LLMError, ValidationError, ValueError, TypeError) as e:
        log_event(
            trace_id, "ERROR", "skill.job_extract",
            f"second attempt failed: {type(e).__name__}: "
            f"{str(e)[:200]} | raw_first={first_raw[:500]!r} "
            f"raw_second={second_raw[:500]!r}",
        )
        return _empty_extraction()


class JobExtractSkill(BaseSkill):
    name: ClassVar[str] = "job_extract"
    description: ClassVar[str] = (
        "Extracts structured data from raw job posting text "
        "using local LLM"
    )
    version: ClassVar[str] = "1.0.0"
    requires_gpu: ClassVar[bool] = True
    input_schema: ClassVar[type[BaseModel]] = JobExtractInput
    output_schema: ClassVar[type[BaseModel]] = JobExtraction

    async def execute(
        self,
        input_data: BaseModel,
        trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        if not isinstance(input_data, JobExtractInput):
            raise SkillError(
                self.name,
                f"expected JobExtractInput, got {type(input_data).__name__}",
                trace_id,
            )
        # Agent may pin a model name (registry name like "qwen-3b"); if
        # so we resolve to that backend's model_id. Otherwise fall back
        # to the legacy DEFAULT_MODEL behavior for compatibility.
        model_name_or_id: str = (
            (context or {}).get("model") or config.DEFAULT_MODEL
        )
        cfg = INFERENCE_CLIENT.model_registry.get(model_name_or_id)
        backend_model_id = cfg.model_id if cfg else model_name_or_id
        # extract() is sync (uses urllib + Ollama HTTP). Wrap for asyncio.
        out = await asyncio.to_thread(
            extract, input_data.text, trace_id, None, backend_model_id,
        )
        # Phase 12: echo carry-throughs into the output so job_score
        # gets the url + workplace_pref and applications table gets
        # populated downstream.
        if isinstance(out, JobExtraction):
            if input_data.url:
                out.url = input_data.url
            if input_data.workplace_pref:
                out.workplace_pref = input_data.workplace_pref
            # description = first ~1500 chars of the raw text body, after
            # the header lines that scrape prepended.
            text = input_data.text or ""
            body_split = text.split("\n\n", 1)
            out.description = body_split[1] if len(body_split) == 2 else text
        return out
