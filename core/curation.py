"""Nightly curation flow.

Reviews the last 24h of episodic activity, asks the local Claude CLI
to propose updates to ``MEMORY.md`` and ``USER.md``, sanity-checks the
proposal, and surfaces it to Telegram for owner approval. Approved
changes are written through ``file_guard.authorize_update`` so the
diff-watch baseline stays in sync.

Sentinel makes ZERO outbound API calls -- the Claude integration is
the local ``claude.exe`` subprocess via ``core/claude_cli.py``.

State (the pending-proposal dict) is in-memory only. If the bot
restarts before approval, the proposal is lost; the user just runs
``/curate`` again.
"""
from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from core import config
from core.claude_cli import ClaudeCliClient, ClaudeCliError
from core.file_guard import FileGuard
from core.logger import log_event


CURATOR_SYSTEM = """You are Sentinel's memory curator. Review the last 24 hours of activity
and the current MEMORY.md and USER.md. Propose updates as ONLY a JSON object:

{
  "memory_additions": ["new fact line", ...],
  "memory_removals": ["old fact line to delete", ...],
  "user_updates": [{"section": "...", "change": "...", "reason": "..."}],
  "no_changes": false
}

Rules:
- Only durable facts (preferences, decisions, outcomes) -- not transient remarks.
- If a new fact contradicts an existing one, propose removing the old in memory_removals.
- USER.md changes must include a "reason" field.
- If nothing worth updating, set no_changes:true and leave the lists empty.
- Do NOT include code, commands, secrets, or session tokens in any proposal.
- Output ONLY the JSON object. No prose, no markdown fences."""


def _parse_json_object(text: str) -> dict | None:
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


def _short_token() -> str:
    return secrets.token_hex(2).upper()


def _within_lookback(iso_ts: str, hours: int) -> bool:
    try:
        ts = datetime.fromisoformat(iso_ts)
    except Exception:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts >= cutoff


def _format_episodes(episodes: list[Any]) -> str:
    if not episodes:
        return "(no episodes in lookback window)"
    lines = []
    for e in episodes:
        lines.append(
            f"- [{e.scope}/{e.event_type}] {e.summary} "
            f"({e.created_at[:16]})"
        )
        if e.detail:
            lines.append(f"  detail: {e.detail[:200]}")
    return "\n".join(lines)


def _apply_proposal_to_memory_md(
    current_md: str,
    additions: list[str],
    removals: list[str],
) -> str:
    """Apply additions + removals to MEMORY.md content. Removals are
    matched by exact substring; additions go under '## Facts'."""
    out = current_md
    for r in removals:
        if not r:
            continue
        for prefix in ("", "- ", "* "):
            target = (prefix + r).strip()
            if target in out:
                out = out.replace(target, "").rstrip() + "\n"
                break
    if additions:
        block = "\n".join(f"- {a}" for a in additions if a)
        if "## Facts" in out:
            head, _, rest = out.partition("## Facts")
            out = head + "## Facts\n" + block + "\n" + rest.lstrip("\n")
        else:
            out = out.rstrip() + "\n\n## Facts\n" + block + "\n"
    return out


class CurationFlow:
    def __init__(
        self,
        memory_manager: Any,
        file_guard: FileGuard,
        brain: Any | None = None,
        claude_client: ClaudeCliClient | None = None,
    ) -> None:
        self.memory = memory_manager
        self.file_guard = file_guard
        self.brain = brain
        self.claude = claude_client or ClaudeCliClient()
        self._pending: dict[str, dict[str, Any]] = {}

    def list_pending(self) -> list[str]:
        return list(self._pending.keys())

    def get_pending(self, token: str) -> dict[str, Any] | None:
        return self._pending.get(token)

    def discard_pending(self, token: str) -> bool:
        return self._pending.pop(token, None) is not None

    async def propose(
        self, trace_id: str,
        lookback_hours: int | None = None,
    ) -> dict[str, Any]:
        """Run the propose pass. Returns the proposal record (with
        token) or a {"_error": True, ...} envelope on failure. The
        proposal is stored under self._pending only on success."""
        lookback_hours = lookback_hours or config.CURATION_LOOKBACK_HOURS
        recent_eps = [
            e for e in self.memory.get_recent_episodes(limit=200)
            if _within_lookback(e.created_at, lookback_hours)
        ]
        memory_path = config.PERSONA_DIR / "MEMORY.md"
        user_path = config.PERSONA_DIR / "USER.md"
        memory_md = (
            memory_path.read_text(encoding="utf-8")
            if memory_path.exists() else ""
        )
        user_md = (
            user_path.read_text(encoding="utf-8")
            if user_path.exists() else ""
        )

        prompt = (
            f"Recent activity ({lookback_hours}h):\n"
            f"{_format_episodes(recent_eps)}\n\n"
            f"Current MEMORY.md:\n```markdown\n{memory_md}\n```\n\n"
            f"Current USER.md:\n```markdown\n{user_md}\n```\n\n"
            "Propose updates per the system prompt. /no_think"
        )
        log_event(
            trace_id, "INFO", "curation",
            f"propose start episodes={len(recent_eps)}",
        )
        try:
            curator_resp = await self.claude.generate(
                prompt=prompt, system=CURATOR_SYSTEM,
                trace_id=trace_id,
                timeout=config.CLAUDE_CLI_TIMEOUT,
            )
        except ClaudeCliError as e:
            log_event(
                trace_id, "ERROR", "curation",
                f"curator claude call failed: {e}",
            )
            return {
                "_error": True,
                "error": f"Claude CLI failed: {e}",
                "trace_id": trace_id,
            }
        proposal = _parse_json_object(curator_resp) or {}
        if not isinstance(proposal, dict):
            return {
                "_error": True,
                "error": "curator returned non-JSON",
                "trace_id": trace_id,
            }

        ok, issues = self._sanity_check(proposal)
        if not ok:
            return {
                "_error": True,
                "error": "sanity-check rejected: " + "; ".join(issues),
                "trace_id": trace_id,
            }

        token = _short_token()
        record = {
            "token": token,
            "trace_id": trace_id,
            "proposal": proposal,
            "memory_md_before": memory_md,
            "user_md_before": user_md,
            "lookback_hours": lookback_hours,
            "episodes_reviewed": len(recent_eps),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._pending[token] = record
        log_event(
            trace_id, "INFO", "curation",
            f"proposal stored token={token} "
            f"additions={len(proposal.get('memory_additions') or [])} "
            f"removals={len(proposal.get('memory_removals') or [])} "
            f"no_changes={proposal.get('no_changes')}",
        )
        return record

    @staticmethod
    def _sanity_check(proposal: dict) -> tuple[bool, list[str]]:
        issues: list[str] = []
        forbidden = [
            r"\beval\s*\(", r"\bexec\s*\(", r"subprocess",
            r"os\.system", r"importlib", r"__import__",
            r"shutil\s*\.\s*rmtree", r"os\.remove",
            r"BEGIN\s+(?:RSA|EC|OPENSSH)\s+PRIVATE",
            r"AKIA[0-9A-Z]{16}",
            r"sk-[A-Za-z0-9]{20,}",
            r"ghp_[A-Za-z0-9]{20,}",
        ]
        flat = json.dumps(proposal)
        for pat in forbidden:
            if re.search(pat, flat, flags=re.IGNORECASE):
                issues.append(f"forbidden pattern: {pat}")
        for k in ("memory_additions", "memory_removals"):
            v = proposal.get(k)
            if v is not None and not isinstance(v, list):
                issues.append(f"{k} must be a list, got {type(v).__name__}")
        return (not issues, issues)

    def apply(
        self, token: str, trace_id: str,
    ) -> dict[str, Any]:
        """Apply a previously-proposed change. Writes MEMORY.md +
        USER.md via file_guard.authorize_update. Reloads brain persona
        and re-syncs persona_files into semantic memory."""
        record = self._pending.pop(token, None)
        if record is None:
            return {
                "_error": True,
                "error": f"no pending proposal for token {token!r}",
                "trace_id": trace_id,
            }
        proposal = record["proposal"]
        applied: dict[str, Any] = {
            "token": token,
            "memory_md_changed": False,
            "user_md_changed": False,
        }
        if not proposal.get("no_changes"):
            additions = proposal.get("memory_additions") or []
            removals = proposal.get("memory_removals") or []
            if additions or removals:
                new_md = _apply_proposal_to_memory_md(
                    record["memory_md_before"], additions, removals,
                )
                if new_md != record["memory_md_before"]:
                    self.file_guard.authorize_update("MEMORY.md", new_md)
                    applied["memory_md_changed"] = True
            user_updates = proposal.get("user_updates") or []
            if user_updates:
                appended = (
                    record["user_md_before"].rstrip()
                    + "\n\n## Curator notes ("
                    + datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    + ")\n"
                )
                for u in user_updates:
                    section = u.get("section", "?")
                    change = u.get("change", "?")
                    reason = u.get("reason", "?")
                    appended += (
                        f"- ({section}) {change}\n  reason: {reason}\n"
                    )
                self.file_guard.authorize_update("USER.md", appended)
                applied["user_md_changed"] = True
        if self.brain is not None:
            try:
                self.brain.reload_persona()
            except Exception as e:
                log_event(
                    trace_id, "WARNING", "curation",
                    f"brain.reload_persona failed: {e}",
                )
        try:
            self.memory.sync_persona_files()
        except Exception as e:
            log_event(
                trace_id, "WARNING", "curation",
                f"sync_persona_files failed: {e}",
            )
        log_event(
            trace_id, "INFO", "curation",
            f"applied token={token} "
            f"memory_changed={applied['memory_md_changed']} "
            f"user_changed={applied['user_md_changed']}",
        )
        return applied


CURATION: CurationFlow | None = None


def install_curation_flow(flow: CurationFlow) -> None:
    global CURATION
    CURATION = flow


def get_curation_flow() -> CurationFlow | None:
    return CURATION
