"""Job report skill -- final stage of /jobsearch pipeline.

Consumes a list of ``ScoredPosting`` (one per scraped posting after
fan-out through ``job_extract`` and ``job_score``) and writes:

  workspace/job_searches/<timestamp>/jobs.csv      -- all postings sorted by score
  workspace/job_searches/<timestamp>/summary.md    -- top picks + stats

Stores one episodic memory entry under scope=``job_searcher`` with a
summary the brain can recall via ``/recall``.

This is the canonical ``accepts_list=True`` skill -- it overrides
validate_input to translate the bare list (post fan-out) into the
expected ``{"scored": [...]}`` shape.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from core import config
from core.logger import log_event
from core.skills import BaseSkill, SkillError


class JobReportInput(BaseModel):
    scored: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class JobReportOutput(BaseModel):
    csv_path: str
    summary_path: str
    total_postings: int
    apply_now_count: int = 0
    worth_applying_count: int = 0
    maybe_count: int = 0
    skip_count: int = 0
    top_company: str | None = None
    top_score: float = 0.0
    top_n_telegram: str = ""


def _now_dirname() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


_BAND_ICON = {
    "apply_now": "🔥", "worth_applying": "✅",
    "maybe": "🤔", "skip": "❌",
}


def _build_csv(scored: list[dict[str, Any]]) -> str:
    if not scored:
        return ""
    field_order = [
        "score", "recommendation", "archetype", "title", "company",
        "location", "location_type", "seniority", "salary_range", "url",
        "legitimacy_tier", "reasons",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=field_order, extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in scored:
        line = {k: row.get(k, "") for k in field_order}
        # Pull tier out of the nested legitimacy dict if present.
        legit = row.get("legitimacy") or {}
        if isinstance(legit, dict):
            line["legitimacy_tier"] = legit.get("tier", "")
        if isinstance(line["reasons"], list):
            line["reasons"] = " | ".join(str(r) for r in line["reasons"])
        writer.writerow(line)
    return buf.getvalue()


def _build_summary_md(scored: list[dict[str, Any]]) -> str:
    if not scored:
        return "# Job Search\n\n(no postings scored)\n"
    total = len(scored)
    band_counts = {b: 0 for b in _BAND_ICON}
    for s in scored:
        band_counts[s.get("recommendation", "skip")] = (
            band_counts.get(s.get("recommendation", "skip"), 0) + 1
        )
    lines = [
        "# Job Search Summary",
        "",
        f"- Total postings scored: **{total}**",
        f"- Apply now (≥4.5): **{band_counts['apply_now']}** 🔥",
        f"- Worth applying (4.0–4.4): **{band_counts['worth_applying']}** ✅",
        f"- Maybe (3.5–3.9): **{band_counts['maybe']}** 🤔",
        f"- Skip (<3.5): **{band_counts['skip']}** ❌",
        "",
        "## Top picks",
        "",
    ]
    top = sorted(
        scored, key=lambda s: float(s.get("score", 0)), reverse=True,
    )[:10]
    for i, s in enumerate(top, 1):
        title = s.get("title", "?")
        company = s.get("company", "?")
        location = s.get("location", "?")
        score = float(s.get("score", 0))
        band = s.get("recommendation", "skip")
        icon = _BAND_ICON.get(band, "?")
        archetype = s.get("archetype", "Unknown")
        url = s.get("url", "")
        reasons = s.get("reasons") or []
        if isinstance(reasons, list):
            reasons_str = "; ".join(str(r) for r in reasons[:3])
        else:
            reasons_str = str(reasons)[:160]
        legit = s.get("legitimacy") or {}
        legit_tier = legit.get("tier", "high") if isinstance(legit, dict) else "?"
        legit_marker = "" if legit_tier == "high" else f" ⚠️ {legit_tier}"
        link = f" [apply]({url})" if url else ""
        lines.append(
            f"{i}. {icon} **{title}** @ {company} ({location}) "
            f"— **{score:.2f}** [{archetype}]{legit_marker}{link}",
        )
        if reasons_str:
            lines.append(f"   - {reasons_str}")
    lines.append("")
    return "\n".join(lines)


def top_n_for_telegram(
    scored: list[dict[str, Any]], n: int = 3,
) -> str:
    """Phase 12: build a compact Telegram message for the top N
    apply-worthy postings. Filters out 'skip'-band entries even if
    fewer than N remain."""
    eligible = [
        s for s in scored
        if s.get("recommendation", "skip") != "skip"
    ]
    eligible.sort(
        key=lambda s: float(s.get("score", 0)), reverse=True,
    )
    if not eligible:
        return "No postings scored above 'skip' band this run."
    lines = [f"📋 Top {min(n, len(eligible))} job picks:"]
    for i, s in enumerate(eligible[:n], 1):
        icon = _BAND_ICON.get(s.get("recommendation", "skip"), "?")
        score = float(s.get("score", 0))
        title = s.get("title", "?")
        company = s.get("company", "?")
        location = s.get("location", "?")
        url = s.get("url", "")
        reasons = s.get("reasons") or []
        if isinstance(reasons, list) and reasons:
            tagline = str(reasons[0])[:120]
        else:
            tagline = ""
        lines.append(
            f"\n{i}. {icon} {title} @ {company}\n"
            f"   📍 {location} — score {score:.2f}\n"
            f"   {tagline}"
        )
        if url:
            lines.append(f"   🔗 {url}")
    return "\n".join(lines)


class JobReportSkill(BaseSkill):
    name: ClassVar[str] = "job_report"
    description: ClassVar[str] = (
        "Aggregate scored postings into CSV + markdown report"
    )
    version: ClassVar[str] = "1.0.0"
    requires_gpu: ClassVar[bool] = False
    accepts_list: ClassVar[bool] = True
    input_schema: ClassVar[type[BaseModel]] = JobReportInput
    output_schema: ClassVar[type[BaseModel]] = JobReportOutput

    def validate_input(self, raw):  # noqa: ANN001 -- BaseSkill API
        # Fan-out aggregation hands us a bare list of ScoredPosting
        # dicts; wrap it in the schema field.
        if isinstance(raw, list):
            return self.input_schema(scored=raw)
        return self.input_schema(**raw)

    async def execute(
        self,
        input_data: BaseModel,
        trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        if not isinstance(input_data, JobReportInput):
            raise SkillError(
                self.name,
                f"expected JobReportInput, got {type(input_data).__name__}",
                trace_id,
            )
        scored = list(input_data.scored or [])
        scored.sort(
            key=lambda s: float(s.get("score", 0)), reverse=True,
        )
        out_dir = config.JOBSEARCH_OUTPUT_DIR / _now_dirname()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "jobs.csv"
        md_path = out_dir / "summary.md"
        csv_path.write_text(_build_csv(scored), encoding="utf-8")
        md_path.write_text(_build_summary_md(scored), encoding="utf-8")
        band_counts = {b: 0 for b in _BAND_ICON}
        for s in scored:
            band_counts[s.get("recommendation", "skip")] = (
                band_counts.get(s.get("recommendation", "skip"), 0) + 1
            )
        top = scored[0] if scored else None
        # PROJECT_ROOT-relative when possible; absolute otherwise so
        # tests using tmp_path don't fail.
        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(config.PROJECT_ROOT))
            except ValueError:
                return str(p)
        telegram_msg = top_n_for_telegram(scored, n=3)
        out = JobReportOutput(
            csv_path=_rel(csv_path),
            summary_path=_rel(md_path),
            total_postings=len(scored),
            apply_now_count=band_counts["apply_now"],
            worth_applying_count=band_counts["worth_applying"],
            maybe_count=band_counts["maybe"],
            skip_count=band_counts["skip"],
            top_company=(top or {}).get("company") if top else None,
            top_score=float((top or {}).get("score", 0.0)) if top else 0.0,
            top_n_telegram=telegram_msg,
        )
        log_event(
            trace_id, "INFO", "skill.job_report",
            f"wrote report total={out.total_postings} "
            f"apply_now={out.apply_now_count} "
            f"worth={out.worth_applying_count} "
            f"maybe={out.maybe_count} skip={out.skip_count} "
            f"dir={out_dir.name}",
        )
        # Phase 12: persist each scored posting into the applications
        # table so /jobs (Phase 13) can show state, dedup against
        # future runs, and track the lifecycle past "evaluated".
        # Errors here are logged but never break the report write --
        # the workspace files are the authoritative output.
        try:
            from core import database  # noqa: PLC0415
            for s in scored:
                url = (s.get("url") or "").strip()
                if not url:
                    continue
                try:
                    database.upsert_application(
                        url=url,
                        title=s.get("title", "(unknown)"),
                        company=s.get("company", "(unknown)"),
                        location=s.get("location"),
                        archetype=s.get("archetype"),
                        score=s.get("score"),
                        recommendation=s.get("recommendation"),
                    )
                except Exception as e:
                    log_event(
                        trace_id, "WARNING", "skill.job_report",
                        f"upsert_application failed for url={url[:80]!r}: "
                        f"{type(e).__name__}: {e}",
                    )
        except Exception as e:
            log_event(
                trace_id, "WARNING", "skill.job_report",
                f"applications-table write skipped: "
                f"{type(e).__name__}: {e}",
            )
        # Episodic memory entry for /recall later.
        try:
            from core.memory import get_memory  # noqa: PLC0415
            mem = get_memory()
            mem.store_episode(
                scope="job_searcher",
                trace_id=trace_id,
                event_type="report",
                summary=(
                    f"Job report: {out.total_postings} scored, "
                    f"{out.apply_now_count} apply_now + "
                    f"{out.worth_applying_count} worth; "
                    f"top={out.top_company} @ {out.top_score:.2f}"
                ),
                detail=(
                    f"csv={out.csv_path}\nsummary={out.summary_path}"
                ),
                tags=["job_search", "report"],
            )
        except Exception as e:
            log_event(
                trace_id, "DEBUG", "skill.job_report",
                f"episodic store skipped: {type(e).__name__}: {e}",
            )
        return out
