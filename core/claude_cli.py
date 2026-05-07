"""Local Claude Code CLI subprocess client. Sentinel makes ZERO outbound
network calls; teaching and Claude-tier inference go through the user's
locally-authenticated `claude` binary.
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from core import config
from core.logger import log_event


class ClaudeCliError(Exception):
    """Raised when the Claude CLI subprocess fails (not found,
    timed out, returned non-zero, or produced unparseable output)."""


def find_claude_cli() -> str | None:
    """Resolve the claude CLI binary. Prefer the .exe over the .cmd
    shim on Windows so subprocess args bypass cmd.exe's mangling."""
    exe_candidates = [
        os.path.expandvars(
            r"%APPDATA%\npm\node_modules"
            r"\@anthropic-ai\claude-code\bin\claude.exe"
        ),
        os.path.expandvars(
            r"%LOCALAPPDATA%\npm\node_modules"
            r"\@anthropic-ai\claude-code\bin\claude.exe"
        ),
    ]
    for c in exe_candidates:
        if c and Path(c).exists():
            return c
    found = shutil.which("claude")
    if found:
        return found
    cmd_candidate = os.path.expandvars(r"%APPDATA%\npm\claude.cmd")
    if cmd_candidate and Path(cmd_candidate).exists():
        return cmd_candidate
    return None


class ClaudeCliClient:
    """Async client around the local `claude` CLI subprocess.

    The CLI is invoked with --print --output-format json (one-shot,
    structured output). Tools are disabled (`--tools ""`) so the CLI
    behaves as a pure text generator. We do NOT pass --bare because
    that disables OAuth/keychain reads, breaking the user's normal
    Claude Code login.
    """

    def __init__(self) -> None:
        self.cli_path = find_claude_cli()

    @property
    def available(self) -> bool:
        return self.cli_path is not None

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        timeout: int | None = None,
        model: str | None = None,
        trace_id: str = "SEN-system",
        tools: list[str] | None = None,
        cwd: str | None = None,
    ) -> str:
        """Send `prompt` to the Claude CLI and return the model's
        response text. Raises ClaudeCliError on any failure.

        `tools`: list of Claude tool names to ALLOW (e.g. ['Read',
        'Grep', 'Glob']). When None or empty, the CLI runs with all
        tools disabled (pure text generation). When a list is given,
        the CLI is invoked with `--allowedTools` so Claude can actually
        Read files / Grep etc. while producing the recipe."""
        if not self.available:
            raise ClaudeCliError(
                "claude CLI not found on PATH or known npm locations"
            )
        if timeout is None:
            timeout = config.CLAUDE_CLI_TIMEOUT
        cmd = [
            self.cli_path,
            "--print",
            "--output-format", "json",
            "--disable-slash-commands",
            "--allow-dangerously-skip-permissions",
            "--model", model or config.CLAUDE_CLI_MODEL,
        ]
        if tools:
            cmd.extend(["--allowedTools", " ".join(tools)])
        # No `else` branch: claude CLI v2.1.128 dropped `--tools`
        # alias; passing `--tools ""` eats the prompt arg.
        if system is not None:
            cmd.extend(["--system-prompt", system])
        cmd.append(prompt)

        log_event(
            trace_id, "INFO", "claude_cli",
            f"invoking claude CLI; prompt_chars={len(prompt)} "
            f"system_chars={len(system) if system else 0} "
            f"tools={tools or 'none'} timeout={timeout}s",
        )

        is_cmd_shim = (
            sys.platform == "win32"
            and self.cli_path.lower().endswith((".cmd", ".bat"))
        )
        proc_cwd = cwd or str(config.PROJECT_ROOT)
        try:
            if is_cmd_shim:
                cmd_line = subprocess.list2cmdline(cmd)
                proc = await asyncio.create_subprocess_shell(
                    cmd_line,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=proc_cwd,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=proc_cwd,
                )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                raise ClaudeCliError(
                    f"claude CLI timed out after {timeout}s"
                )
        except FileNotFoundError as e:
            raise ClaudeCliError(
                f"claude CLI subprocess failed to start: {e}"
            ) from e

        if proc.returncode != 0:
            tail = stderr_bytes[:500].decode("utf-8", "replace")
            raise ClaudeCliError(
                f"claude CLI rc={proc.returncode}; stderr={tail!r}"
            )

        try:
            payload = json.loads(stdout_bytes.decode("utf-8"))
        except json.JSONDecodeError as e:
            head = stdout_bytes[:500].decode("utf-8", "replace")
            raise ClaudeCliError(
                f"claude CLI output not valid JSON: {e}; stdout={head!r}"
            ) from e

        if payload.get("is_error"):
            raise ClaudeCliError(
                f"claude CLI reported error: "
                f"{payload.get('result', '')[:300]}"
            )

        text = payload.get("result", "") or ""
        log_event(
            trace_id, "INFO", "claude_cli",
            f"claude CLI ok chars={len(text)} cost_usd="
            f"{payload.get('total_cost_usd', 0):.4f}",
        )
        return text


CLAUDE_CLI = ClaudeCliClient()


# ============================================================
# Phase 9: thin adapter exposing a result-object API for callers
# (e.g., Telegram bot) that prefer .success/.text/.error over
# raise-on-error semantics.
# ============================================================

from pydantic import BaseModel as _BaseModel


class GenerateResult(_BaseModel):
    success: bool
    text: str = ""
    error: str | None = None


class ClaudeCLI:
    """Result-object wrapper around ClaudeCliClient. Same underlying
    subprocess; just a friendlier surface for callers that don't want
    to try/except every call."""

    def __init__(self) -> None:
        self._client = ClaudeCliClient()

    @property
    def available(self) -> bool:
        return self._client.available

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        timeout: int | None = None,
        model: str | None = None,
        trace_id: str = "SEN-system",
    ) -> GenerateResult:
        try:
            text = await self._client.generate(
                prompt=prompt, system=system, timeout=timeout,
                model=model, trace_id=trace_id,
            )
            return GenerateResult(success=True, text=text)
        except ClaudeCliError as e:
            return GenerateResult(success=False, error=str(e))
        except Exception as e:
            return GenerateResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
            )
