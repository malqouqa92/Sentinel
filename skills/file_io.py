"""Sandboxed file I/O within config.WORKSPACE_DIR.

All paths are resolved and verified to be inside the workspace root.
Any path traversal attempt (`../`, absolute paths outside workspace)
raises SkillError. Refuses to read files larger than 10MB.
"""
import asyncio
import csv
import io
import shlex
from typing import ClassVar

from pydantic import BaseModel

from core import config
from core.logger import log_event
from core.skills import BaseSkill, SkillError

ALLOWED_ACTIONS = {"read", "write", "append", "list"}
MAX_READ_BYTES = 10 * 1024 * 1024  # 10MB


class FileIOInput(BaseModel):
    action: str
    path: str
    content: str | None = None
    encoding: str = "utf-8"


class FileIOOutput(BaseModel):
    action: str
    path: str
    success: bool
    content: str | None = None
    files: list[str] | None = None
    bytes_written: int | None = None
    csv_preview: list[dict] | None = None
    error: str | None = None


def _resolve_safe(rel_path: str, trace_id: str):
    """Resolve a path under the workspace; reject any escape attempt.
    Returns the resolved absolute path."""
    workspace = config.WORKSPACE_DIR.resolve()
    candidate = (workspace / rel_path).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as e:
        raise SkillError(
            "file_io",
            f"path escapes workspace sandbox: {rel_path!r} -> "
            f"{candidate}",
            trace_id,
        ) from e
    return candidate


def _read(path, encoding: str, trace_id: str) -> dict:
    if not path.exists():
        raise SkillError("file_io", f"file not found: {path}", trace_id)
    if not path.is_file():
        raise SkillError("file_io", f"not a regular file: {path}", trace_id)
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        log_event(
            trace_id, "WARNING", "skill.file_io",
            f"refusing to read large file: {path} size={size} bytes",
        )
        raise SkillError(
            "file_io",
            f"file exceeds {MAX_READ_BYTES} byte read limit "
            f"({size} bytes): {path}",
            trace_id,
        )
    text = path.read_text(encoding=encoding)
    csv_preview = None
    if path.suffix.lower() == ".csv":
        try:
            reader = csv.DictReader(io.StringIO(text))
            rows = []
            for i, row in enumerate(reader):
                if i >= 10:
                    break
                rows.append(row)
            csv_preview = rows
        except Exception as e:
            log_event(
                trace_id, "WARNING", "skill.file_io",
                f"csv preview failed for {path}: {e}",
            )
    return {"content": text, "csv_preview": csv_preview}


def _write(path, content: str, encoding: str, append: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding=encoding) as f:
        n = f.write(content)
    return n


def _list(path, trace_id: str) -> list[str]:
    if not path.exists():
        raise SkillError(
            "file_io", f"directory not found: {path}", trace_id,
        )
    if not path.is_dir():
        raise SkillError(
            "file_io", f"not a directory: {path}", trace_id,
        )
    workspace = config.WORKSPACE_DIR.resolve()
    return sorted(
        str(p.relative_to(workspace)).replace("\\", "/")
        for p in path.iterdir()
    )


def _do(input_data: FileIOInput, trace_id: str) -> FileIOOutput:
    action = input_data.action.strip().lower()
    if action not in ALLOWED_ACTIONS:
        raise SkillError(
            "file_io",
            f"unknown action: {action!r}; allowed: "
            f"{sorted(ALLOWED_ACTIONS)}",
            trace_id,
        )
    target = _resolve_safe(input_data.path, trace_id)
    log_event(
        trace_id, "INFO", "skill.file_io",
        f"action={action} path={input_data.path!r}",
    )
    if action == "read":
        r = _read(target, input_data.encoding, trace_id)
        return FileIOOutput(
            action=action, path=input_data.path, success=True,
            content=r["content"], csv_preview=r["csv_preview"],
        )
    if action == "list":
        files = _list(target, trace_id)
        return FileIOOutput(
            action=action, path=input_data.path, success=True,
            files=files,
        )
    if action in ("write", "append"):
        if input_data.content is None:
            raise SkillError(
                "file_io",
                f"action={action} requires non-null 'content'",
                trace_id,
            )
        n = _write(
            target, input_data.content, input_data.encoding,
            append=(action == "append"),
        )
        return FileIOOutput(
            action=action, path=input_data.path, success=True,
            bytes_written=n,
        )
    raise SkillError("file_io", f"unhandled action: {action}", trace_id)


class FileIOSkill(BaseSkill):
    name: ClassVar[str] = "file_io"
    description: ClassVar[str] = (
        "Read, write, append, or list files within the workspace "
        "(sandboxed, path traversal blocked)"
    )
    version: ClassVar[str] = "1.0.0"
    requires_gpu: ClassVar[bool] = False
    input_schema: ClassVar[type[BaseModel]] = FileIOInput
    output_schema: ClassVar[type[BaseModel]] = FileIOOutput

    def validate_input(self, raw: dict) -> BaseModel:
        # Router free-text mode: "/file write path.txt content goes here"
        if set(raw.keys()) == {"text"}:
            try:
                parts = shlex.split(raw["text"])
            except ValueError:
                parts = raw["text"].split()
            if len(parts) >= 2:
                action, path = parts[0], parts[1]
                content = " ".join(parts[2:]) if len(parts) > 2 else None
                return FileIOInput(
                    action=action, path=path, content=content,
                )
            if len(parts) == 1:
                # `list` with no path means workspace root.
                return FileIOInput(action=parts[0], path=".")
            raise ValueError(
                "file_io text-mode input must be: "
                "'<action> <path> [content...]'"
            )
        return FileIOInput(**raw)

    async def execute(
        self, input_data: BaseModel, trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        if not isinstance(input_data, FileIOInput):
            raise SkillError(
                self.name,
                f"expected FileIOInput, got {type(input_data).__name__}",
                trace_id,
            )
        return await asyncio.to_thread(_do, input_data, trace_id)
