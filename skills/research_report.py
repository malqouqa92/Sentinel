"""Final stage of /research pipeline.

Compiles per-result summaries from ``web_summarize`` into one markdown
report saved to ``workspace/research/<timestamp>/report.md``. Stores
one episodic memory entry under scope=``researcher`` so the brain can
recall it via ``/recall``.

Pure CPU, no LLM calls -- the model already ran in web_summarize.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from core import config
from core.logger import log_event
from core.skills import BaseSkill, SkillError


class ResearchReportInput(BaseModel):
    query: str = ""
    summaries: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class ResearchReportOutput(BaseModel):
    query: str
    report_path: str
    summary_count: int
    summaries: list[dict[str, Any]] = Field(default_factory=list)


def _now_dirname() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_report_md(query: str, summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return (
            f"# Research Brief: {query}\n\n"
            f"(no results -- search returned 0 hits)\n"
        )
    lines = [
        f"# Research Brief: {query}",
        "",
        f"_{len(summaries)} sources reviewed_",
        "",
    ]
    for i, s in enumerate(summaries, 1):
        title = s.get("title") or "(no title)"
        url = s.get("url") or ""
        summary = s.get("summary") or ""
        lines.append(f"## {i}. {title}")
        if url:
            lines.append(f"<{url}>")
        lines.append("")
        lines.append(summary.strip())
        lines.append("")
    return "\n".join(lines)


class ResearchReportSkill(BaseSkill):
    name: ClassVar[str] = "research_report"
    description: ClassVar[str] = (
        "Compile per-result summaries into one markdown research brief"
    )
    version: ClassVar[str] = "1.0.0"
    requires_gpu: ClassVar[bool] = False
    input_schema: ClassVar[type[BaseModel]] = ResearchReportInput
    output_schema: ClassVar[type[BaseModel]] = ResearchReportOutput

    async def execute(
        self, input_data: BaseModel, trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        if not isinstance(input_data, ResearchReportInput):
            raise SkillError(
                self.name,
                f"expected ResearchReportInput, got "
                f"{type(input_data).__name__}",
                trace_id,
            )
        out_dir = config.RESEARCH_OUTPUT_DIR / _now_dirname()
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "report.md"
        report_path.write_text(
            _build_report_md(input_data.query, input_data.summaries),
            encoding="utf-8",
        )

        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(config.PROJECT_ROOT))
            except ValueError:
                return str(p)

        out = ResearchReportOutput(
            query=input_data.query,
            report_path=_rel(report_path),
            summary_count=len(input_data.summaries),
            summaries=[
                {
                    "title": str(s.get("title") or ""),
                    "url": str(s.get("url") or ""),
                    "summary": str(s.get("summary") or "")[:300],
                }
                for s in input_data.summaries
            ],
        )
        log_event(
            trace_id, "INFO", "skill.research_report",
            f"wrote report query={input_data.query[:60]!r} "
            f"sources={out.summary_count} dir={out_dir.name}",
        )
        # Episodic memory entry for /recall later.
        try:
            from core.memory import get_memory  # noqa: PLC0415
            mem = get_memory()
            mem.store_episode(
                scope="researcher", trace_id=trace_id,
                event_type="report",
                summary=(
                    f"Research brief on {input_data.query!r}: "
                    f"{out.summary_count} sources"
                ),
                detail=f"report={out.report_path}",
                tags=["research", "report"],
            )
        except Exception as e:
            log_event(
                trace_id, "DEBUG", "skill.research_report",
                f"episodic store skipped: {type(e).__name__}: {e}",
            )
        return out
