"""Python code execution via subprocess (isolated, timed-out, capped).

The code runs in a subprocess (NOT exec()/eval() in the parent) using
the same Python interpreter, with cwd = config.WORKSPACE_DIR. stdout
and stderr are capped at 50KB each. Process is killed on timeout.

Security note: this runs arbitrary user-supplied Python. Fine for a
personal/local tool; do NOT expose this skill over a network without
adding seccomp/jail/container isolation.
"""
import asyncio
import shlex
import sys
import tempfile
import time
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from core import config
from core.logger import log_event
from core.skills import BaseSkill, SkillError

OUTPUT_CAP = 50 * 1024  # 50KB


class CodeExecuteInput(BaseModel):
    code: str
    timeout: int = 30
    save_as: str | None = None


class CodeExecuteOutput(BaseModel):
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    saved_path: str | None = None
    truncated: bool = False


def _resolve_save_path(rel_path: str, trace_id: str) -> Path:
    """Same workspace sandbox check as file_io. Reject any path that
    resolves outside config.WORKSPACE_DIR."""
    workspace = config.WORKSPACE_DIR.resolve()
    candidate = (workspace / rel_path).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as e:
        raise SkillError(
            "code_execute",
            f"save_as path escapes workspace sandbox: {rel_path!r} "
            f"-> {candidate}",
            trace_id,
        ) from e
    return candidate


def _cap(text: bytes) -> tuple[str, bool]:
    decoded = text.decode("utf-8", errors="replace")
    if len(decoded) > OUTPUT_CAP:
        return decoded[:OUTPUT_CAP], True
    return decoded, False


async def _run(input_data: CodeExecuteInput, trace_id: str
               ) -> CodeExecuteOutput:
    workspace = config.WORKSPACE_DIR.resolve()
    saved_path: str | None = None

    # Decide where the script lives.
    if input_data.save_as:
        target = _resolve_save_path(input_data.save_as, trace_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(input_data.code, encoding="utf-8")
        script_path = target
        saved_path = str(target.relative_to(workspace)).replace("\\", "/")
        log_event(
            trace_id, "INFO", "skill.code_execute",
            f"saved code to {saved_path} ({len(input_data.code)} chars)",
        )
    else:
        # Anonymous temp script in the workspace. Cleaned up after.
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=str(workspace),
            delete=False, encoding="utf-8",
        )
        try:
            tf.write(input_data.code)
        finally:
            tf.close()
        script_path = Path(tf.name)

    log_event(
        trace_id, "INFO", "skill.code_execute",
        f"executing script={script_path.name} timeout={input_data.timeout}s",
    )

    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=input_data.timeout,
            )
            elapsed = time.monotonic() - start
            stdout, t1 = _cap(stdout_bytes)
            stderr, t2 = _cap(stderr_bytes)
            return CodeExecuteOutput(
                stdout=stdout, stderr=stderr,
                return_code=proc.returncode,
                execution_time=elapsed,
                saved_path=saved_path,
                truncated=(t1 or t2),
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            elapsed = time.monotonic() - start
            log_event(
                trace_id, "WARNING", "skill.code_execute",
                f"timeout after {input_data.timeout}s; killed pid={proc.pid}",
            )
            return CodeExecuteOutput(
                stdout="", stderr=(
                    f"Execution timed out after {input_data.timeout}s"
                ),
                return_code=-1,
                execution_time=elapsed,
                saved_path=saved_path,
            )
    finally:
        # Clean up anonymous temp scripts; preserve user-saved ones.
        if not input_data.save_as:
            try:
                script_path.unlink(missing_ok=True)
            except Exception:
                pass


class CodeExecuteSkill(BaseSkill):
    name: ClassVar[str] = "code_execute"
    description: ClassVar[str] = (
        "Executes Python code in a subprocess with timeout and "
        "output cap (cwd = workspace)"
    )
    version: ClassVar[str] = "1.0.0"
    requires_gpu: ClassVar[bool] = False
    input_schema: ClassVar[type[BaseModel]] = CodeExecuteInput
    output_schema: ClassVar[type[BaseModel]] = CodeExecuteOutput

    def validate_input(self, raw: dict) -> BaseModel:
        # Router free-text mode: "/exec print('hi')" -> {"text": "print('hi')"}
        if set(raw.keys()) == {"text"}:
            return CodeExecuteInput(code=raw["text"])
        return CodeExecuteInput(**raw)

    async def execute(
        self, input_data: BaseModel, trace_id: str,
        context: dict | None = None,
    ) -> BaseModel:
        if not isinstance(input_data, CodeExecuteInput):
            raise SkillError(
                self.name,
                f"expected CodeExecuteInput, got "
                f"{type(input_data).__name__}",
                trace_id,
            )
        return await _run(input_data, trace_id)
