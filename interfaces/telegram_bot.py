"""Sentinel's Telegram interface.

Three lanes per user message:
  /code           -> route through full skill pipeline (code_assist + teaching)
  /claude         -> direct passthrough to local Claude CLI subprocess
  free text       -> brain classifies intent -> dispatch or chat

Every handler authorizes via Telegram user_id (config.TELEGRAM_AUTHORIZED_USERS).
Long replies are split at line boundaries to fit Telegram's 4096-char limit.
"""
import asyncio
import re


BAR_FILLED_EMOJI = "🟠"
BAR_EMPTY_EMOJI = "❄️"


def _build_bar(pct: int, w: int = 10) -> str:
    pct = max(0, min(100, pct))
    f = int(w * pct / 100)
    return BAR_FILLED_EMOJI * f + BAR_EMPTY_EMOJI * (w - f) + " " + str(pct) + "%"

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar

from telegram import BotCommand, Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters,
)

from string import Template as _string_Template

from core import config, database
from core.brain import BrainRouter
from core.curation import CurationFlow, install_curation_flow
from core.file_guard import FileGuard, install_file_guard
from core.memory import WORKING_MEMORY, get_memory


# --- /schedule subcommand parsing -------------------------------------

class _ScheduleArgError(ValueError):
    """Raised by _parse_schedule_add for user-fixable input errors."""


_PROFILE_USAGE = (
    "Usage:\n"
    "  /profile             -- show current PROFILE.yml\n"
    "  /profile init        -- create PROFILE.yml from the example template\n"
    "  /profile show        -- pretty-print parsed values\n"
    "  /profile set <dotted.path> <value>\n"
    "                       -- e.g. /profile set candidate.location \"Detroit, MI\"\n"
    "  /profile edit        -- show full file text + path for manual edit"
)


_SCHEDULE_USAGE = (
    "Usage:\n"
    "  /schedule add \"<name>\" --cron \"<expr>\" --command \"<cmd>\" "
    "[--hours HH-HH] [--isolated]\n"
    "  /schedule add \"<name>\" --interval <30m|2h|45s|1d> "
    "--command \"<cmd>\"\n"
    "  /schedule add \"<name>\" --once <ISO datetime> "
    "--command \"<cmd>\" [--delete-after]\n"
    "  /schedule list\n"
    "  /schedule pause <id> | resume <id> | delete <id> | runs <id>\n"
    "Times in EST (cron + active-hours window)."
)


def _format_curation_proposal(record: dict) -> str:
    """Render a CurationFlow pending record for display in chat. Used by
    /curate (initial propose response) and /curate review (re-display)."""
    proposal = record.get("proposal") or {}
    token = record.get("token", "?")
    if proposal.get("no_changes"):
        return (
            f"📋 [{token}] No durable changes worth proposing "
            f"(reviewed {record.get('episodes_reviewed', '?')} episodes "
            f"over {record.get('lookback_hours', '?')}h)."
        )
    additions = proposal.get("memory_additions") or []
    removals = proposal.get("memory_removals") or []
    user_updates = proposal.get("user_updates") or []
    lines = [
        f"📋 Curation proposal [{token}]",
        f"_(reviewed {record.get('episodes_reviewed', '?')} episodes "
        f"over {record.get('lookback_hours', '?')}h, "
        f"created {record.get('created_at', '?')})_",
        "",
    ]
    if additions:
        lines.append("**MEMORY.md additions:**")
        for a in additions:
            lines.append(f"  + {a}")
        lines.append("")
    if removals:
        lines.append("**MEMORY.md removals:**")
        for r in removals:
            lines.append(f"  - {r}")
        lines.append("")
    if user_updates:
        lines.append("**USER.md updates (appended as notes):**")
        for u in user_updates:
            section = u.get("section", "?")
            change = u.get("change", "?")
            reason = u.get("reason", "?")
            lines.append(f"  • ({section}) {change}")
            lines.append(f"    _reason: {reason}_")
        lines.append("")
    lines.append(f"Approve: /curate_approve {token}")
    lines.append(f"Reject:  /curate_reject {token}")
    return "\n".join(lines)


def _parse_schedule_add(args: list[str]) -> dict:
    """Parse the /schedule add subcommand into a dict. Strict; raises
    _ScheduleArgError on any malformed input.

    First positional token (or quoted phrase, but Telegram already
    splits on spaces -- callers should quote their name as a single
    token at the bot prompt) is the job name. Remaining tokens are
    --flag value pairs."""
    if not args:
        raise _ScheduleArgError("missing job name")
    name = args[0]
    rest = args[1:]
    out = {
        "name": name,
        "schedule_type": None,
        "schedule_value": None,
        "command": None,
        "session_type": "main",
        "active_hours_start": None,
        "active_hours_end": None,
        "delete_after_run": False,
    }
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok in ("--cron", "--interval", "--once"):
            if i + 1 >= len(rest):
                raise _ScheduleArgError(f"{tok} needs a value")
            out["schedule_type"] = tok[2:]
            out["schedule_value"] = rest[i + 1]
            i += 2
        elif tok == "--command":
            if i + 1 >= len(rest):
                raise _ScheduleArgError("--command needs a value")
            # Greedy: consume the rest of args as the command body so
            # commands with spaces survive Telegram's tokenization.
            out["command"] = " ".join(rest[i + 1:])
            i = len(rest)
        elif tok == "--hours":
            if i + 1 >= len(rest):
                raise _ScheduleArgError("--hours needs HH-HH")
            window = rest[i + 1]
            if "-" not in window:
                raise _ScheduleArgError(
                    f"--hours expects HH-HH, got {window!r}"
                )
            start, end = window.split("-", 1)
            out["active_hours_start"] = (
                start if ":" in start else f"{int(start):02d}:00"
            )
            out["active_hours_end"] = (
                end if ":" in end else f"{int(end):02d}:00"
            )
            i += 2
        elif tok == "--isolated":
            out["session_type"] = "isolated"
            i += 1
        elif tok == "--delete-after":
            out["delete_after_run"] = True
            i += 1
        else:
            raise _ScheduleArgError(f"unknown flag {tok!r}")
    if out["schedule_type"] is None:
        raise _ScheduleArgError(
            "must specify exactly one of --cron / --interval / --once"
        )
    if not out["command"]:
        raise _ScheduleArgError("--command is required")
    if not out["command"].startswith("/"):
        raise _ScheduleArgError(
            f"--command must be a slash-command, got {out['command']!r}"
        )
    return out


# ----------------------------------------------------------------------
# Progress tracking: read the JSONL log tail, find the latest event for
# this trace_id, map it to a user-friendly stage label.
# ----------------------------------------------------------------------

_STAGE_RULES: list[tuple[str, str, str]] = [
    # (component, substring_in_message, label)
    ("knowledge_base", "added pattern",        "💾 Saved pattern"),
    ("skill.code_assist", "qwen_taught",       "✅ Solution found"),
    ("skill.code_assist", "qwen could not learn", "📦 Using Claude's code"),
    ("skill.code_assist", "qwen attempt 2",    "🤖 Qwen retry"),
    ("claude_cli", "claude CLI ok",            "🎓 Claude responded"),
    ("claude_cli", "invoking claude CLI",      "🎓 Asking Claude..."),
    ("skill.code_assist", "validating qwen",   "🧪 Testing solution"),
    ("skill.code_assist", "qwen attempt 1",    "🤖 Qwen drafting"),
    ("skill.code_assist", "KB lookup",         "📚 Checking memory"),
    ("skill.web_search", "got",                "🌐 Got results"),
    ("skill.web_search", "search starting",    "🌐 Searching..."),
    ("skill.code_execute", "executing script", "🧪 Running code"),
    ("orchestrator", "dispatching",            "🔀 Dispatching"),
    ("agent.code_assistant", "pipeline",       "🧠 Pipeline running"),
    ("agent.job_analyst", "pipeline",          "🧠 Pipeline running"),
    ("worker", "executing task",               "⚙️  Worker started"),
    ("database", "task claimed",               "📥 Task claimed"),
    ("database", "task added",                 "📥 Queued"),
        ("router", "received",                     "🔧 Routed"),
]

_STAGE_PCT: dict[str, int] = {
    "🔧 Routed":               5,
    "📥 Queued":              10,
    "📥 Task claimed":        15,
    "⚙️  Worker started":     20,
    "🔀 Dispatching":         30,
    "🧠 Pipeline running":    40,
    "📚 Checking memory":     45,
    "🌐 Searching...":        50,
    "🌐 Got results":         55,
    "🧪 Running code":        55,
    "🤖 Qwen drafting":       55,
    "🧪 Testing solution":    65,
    "🎓 Asking Claude...":    70,
    "🤖 Qwen retry":          75,
    "🎓 Claude responded":    80,
    "📦 Using Claude's code": 85,
    "✅ Solution found":      90,
    "💾 Saved pattern":       95,
    "🔧 Working...":           5,
}


def _latest_stage_for_trace(trace_id: str) -> str:
    """Scan the JSONL log file for the latest event matching this
    trace_id and return a human-readable stage label."""
    log_path = config.LOG_DIR / config.LOG_FILE
    if not log_path.exists():
        return "🔧 Working..."
    # Read the last ~16KB which is plenty for recent events
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16384))
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return "🔧 Working..."
    import json as _json
    latest_match: tuple[int, str] | None = None
    for line in tail.splitlines():
        if not line.strip() or trace_id not in line:
            continue
        try:
            entry = _json.loads(line)
        except Exception:
            continue
        if entry.get("trace_id") != trace_id:
            continue
        comp = entry.get("component", "")
        msg = entry.get("message", "")
        for i, (rule_comp, rule_substr, label) in enumerate(_STAGE_RULES):
            if comp == rule_comp and rule_substr in msg:
                # Earlier rules in the list = later stages -> higher priority
                priority = len(_STAGE_RULES) - i
                if latest_match is None or priority > latest_match[0]:
                    latest_match = (priority, label)
                break
    return latest_match[1] if latest_match else "🔧 Working..."
from core.claude_cli import ClaudeCLI
from core.knowledge_base import KnowledgeBase
from core.llm import InferenceClient
from core.logger import log_event
from core.router import route
from core.telemetry import generate_trace_id


class SentinelTelegramBot:
    def __init__(
        self,
        token: str,
        brain: BrainRouter,
        claude_cli: ClaudeCLI,
        inference_client: InferenceClient,
        knowledge_base: KnowledgeBase | None = None,
        file_guard: FileGuard | None = None,
    ) -> None:
        self.app = Application.builder().token(token).build()
        self.brain = brain
        self.claude_cli = claude_cli
        self.inference = inference_client
        self.kb = knowledge_base or KnowledgeBase()
        self._started_at = time.monotonic()
        # Phase 10: file_guard alerts route to Telegram via send_alert.
        # Constructed lazily so the alert callback closes over self.
        self.file_guard = file_guard or FileGuard(
            alert_callback=self.send_alert_sync,
        )
        install_file_guard(self.file_guard)
        # Curation flow shares the file_guard + brain. Lazy-builds
        # ClaudeCliClient internally.
        self.curation = CurationFlow(
            memory_manager=get_memory(),
            file_guard=self.file_guard,
            brain=self.brain,
        )
        install_curation_flow(self.curation)
        self._heartbeat_task: asyncio.Task | None = None
        self._curation_scheduler_task: asyncio.Task | None = None

        self.app.add_handler(CommandHandler("start", self.handle_start))
        self.app.add_handler(CommandHandler("code", self.handle_code))
        self.app.add_handler(CommandHandler("qcode", self.handle_qcode))
        self.app.add_handler(CommandHandler("gwen", self.handle_gwen))
        self.app.add_handler(CommandHandler("prompt", self.handle_prompt))
        self.app.add_handler(CommandHandler("encode", self.handle_encode))
        self.app.add_handler(CommandHandler("gwenask", self.handle_gwenask))
        self.app.add_handler(CommandHandler("claude", self.handle_claude))
        self.app.add_handler(CommandHandler("search", self.handle_search))
        self.app.add_handler(CommandHandler("extract", self.handle_extract))
        # Phase 10 pipelines.
        self.app.add_handler(
            CommandHandler("jobsearch", self.handle_jobsearch),
        )
        self.app.add_handler(
            CommandHandler("research", self.handle_research),
        )
        self.app.add_handler(
            CommandHandler("curate_approve", self.handle_curate_approve),
        )
        self.app.add_handler(
            CommandHandler("curate_reject", self.handle_curate_reject),
        )
        self.app.add_handler(CommandHandler("models", self.handle_models))
        self.app.add_handler(CommandHandler("status", self.handle_status))
        self.app.add_handler(CommandHandler("kb", self.handle_kb))
        self.app.add_handler(CommandHandler("help", self.handle_help))
        # Phase 10: persistent memory commands.
        self.app.add_handler(
            CommandHandler("remember", self.handle_remember),
        )
        self.app.add_handler(
            CommandHandler("forget", self.handle_forget),
        )
        self.app.add_handler(
            CommandHandler("recall", self.handle_recall),
        )
        self.app.add_handler(
            CommandHandler("memory", self.handle_memory),
        )
        self.app.add_handler(
            CommandHandler("curate", self.handle_curate),
        )
        self.app.add_handler(CommandHandler("commit", self.handle_commit))
        self.app.add_handler(CommandHandler("revert", self.handle_revert))
        self.app.add_handler(CommandHandler("restart", self.handle_restart))
        self.app.add_handler(CommandHandler("kill", self.handle_kill))
        self.app.add_handler(CommandHandler("schedule", self.handle_schedule))
        self.app.add_handler(CommandHandler("dashboard", self.handle_dashboard))
        self.app.add_handler(CommandHandler("profile", self.handle_profile))
        self.app.add_handler(CommandHandler("jobs", self.handle_jobs))
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, self.handle_message,
        ))

    # ---------- lifecycle ----------

    async def start(self) -> None:
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        log_event("SEN-system", "INFO", "telegram",
                  "Telegram bot polling started")
        await self._sync_bot_commands()
        # Phase 10: heartbeat loop runs file_guard.check_integrity()
        # every FILE_GUARD_CHECK_INTERVAL seconds.
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
        )
        # Phase 10: nightly curation scheduler.
        self._curation_scheduler_task = asyncio.create_task(
            self._curation_scheduler_loop(),
        )
        # Mirror MEMORY.md (and all persona files) into the semantic store on
        # startup so /recall can find them from day one, before curation runs.
        try:
            _n = await asyncio.to_thread(get_memory().sync_persona_files)
            log_event(
                "SEN-system", "INFO", "telegram",
                f"startup persona sync: {_n} file(s) mirrored to memory.db",
            )
        except Exception as _exc:
            log_event(
                "SEN-system", "WARNING", "telegram",
                f"startup persona sync failed: {type(_exc).__name__}: {_exc}",
            )
        # Welcome broadcast on every restart.
        _est = timezone(timedelta(hours=-5))
        _boot_ts = datetime.now(_est).strftime("%d/%m/%Y %H:%M EST")
        await self.send_alert(
            f"🟢 Sentinel OS online — {_boot_ts}\n"
            f"Type /help for available commands."
        )

    async def stop(self) -> None:
        for attr in ("_heartbeat_task", "_curation_scheduler_task"):
            t = getattr(self, attr, None)
            if t is not None:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
                setattr(self, attr, None)
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
        log_event("SEN-system", "INFO", "telegram",
                  "Telegram bot stopped")

    async def _heartbeat_loop(self) -> None:
        """Periodic file-guard integrity check. Cancelled on stop()."""
        interval = config.FILE_GUARD_CHECK_INTERVAL
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    tampered = await asyncio.to_thread(
                        self.file_guard.check_integrity,
                    )
                    if tampered:
                        log_event(
                            "SEN-system", "WARNING", "telegram",
                            f"heartbeat: {len(tampered)} tampered "
                            f"persona files: {tampered}",
                        )
                except Exception as e:
                    log_event(
                        "SEN-system", "ERROR", "telegram",
                        f"heartbeat check_integrity raised: "
                        f"{type(e).__name__}: {e}",
                    )
        except asyncio.CancelledError:
            return

    async def _curation_scheduler_loop(self) -> None:
        """Sleep until the next curation_hour:curation_minute local
        time and trigger a curation proposal that gets pushed to all
        authorized users via send_alert. Repeats daily.
        """
        try:
            while True:
                now = datetime.now()
                target = now.replace(
                    hour=config.CURATION_HOUR_LOCAL,
                    minute=config.CURATION_MINUTE_LOCAL,
                    second=0, microsecond=0,
                )
                if target <= now:
                    target = target + timedelta(days=1)
                wait_s = (target - now).total_seconds()
                await asyncio.sleep(wait_s)
                trace_id = generate_trace_id()
                log_event(
                    "SEN-system", "INFO", "telegram",
                    f"nightly curation triggered trace={trace_id}",
                )
                try:
                    record = await self.curation.propose(trace_id)
                    if record.get("_error"):
                        await self.send_alert(
                            f"🌙 Nightly curation failed: "
                            f"{record.get('error', '?')[:300]}",
                        )
                    elif record.get("proposal", {}).get("no_changes"):
                        await self.send_alert(
                            "🌙 Nightly curation: no durable changes "
                            "worth proposing.",
                        )
                    else:
                        token = record["token"]
                        prop = record["proposal"]
                        adds = len(prop.get("memory_additions") or [])
                        rems = len(prop.get("memory_removals") or [])
                        usr = len(prop.get("user_updates") or [])
                        await self.send_alert(
                            f"🌙 Nightly curation [{token}]: "
                            f"{adds} additions, {rems} removals, "
                            f"{usr} user updates. "
                            f"Review with /curate "
                            f"(no, this is the proposal already -- "
                            f"send /curate_approve {token} to apply or "
                            f"/curate_reject {token} to discard).",
                        )
                except Exception as e:
                    log_event(
                        "SEN-system", "ERROR", "telegram",
                        f"nightly curation error: "
                        f"{type(e).__name__}: {e}",
                    )
        except asyncio.CancelledError:
            return

    # ---------- Phase 10: alert channel ----------

    async def send_alert(self, text: str) -> None:
        """Push a message to every authorized user. Used by file_guard
        and the curation flow.
        """
        for uid in config.TELEGRAM_AUTHORIZED_USERS:
            try:
                await self.app.bot.send_message(
                    chat_id=uid, text=text,
                )
            except Exception as e:
                log_event(
                    "SEN-system", "WARNING", "telegram",
                    f"send_alert to {uid} failed: "
                    f"{type(e).__name__}: {e}",
                )

    def send_alert_sync(self, text: str) -> None:
        """Sync wrapper for callbacks that aren't already in an event
        loop (e.g., file_guard called from a sync code path). Schedules
        the async send onto the running loop if there is one;
        otherwise logs and drops the alert.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.create_task(self.send_alert(text))
            return
        log_event(
            "SEN-system", "WARNING", "telegram",
            f"send_alert_sync: no running loop, alert dropped: "
            f"{text[:200]!r}",
        )

    # ---------- auth ----------

    async def _check_auth(self, update: Update) -> bool:
        user = update.effective_user
        if user is None:
            return False
        user_id = user.id
        if user_id not in config.TELEGRAM_AUTHORIZED_USERS:
            try:
                await update.message.reply_text("Unauthorized.")
            except Exception:
                pass
            log_event(
                generate_trace_id(), "WARNING", "telegram",
                f"unauthorized access from user_id={user_id} "
                f"username={user.username!r}",
            )
            return False
        return True

    # ---------- helpers ----------

    async def _send_long(self, update: Update, text: str) -> None:
        """Split at line boundaries to respect Telegram's 4096-char cap."""
        max_len = config.TELEGRAM_MAX_MESSAGE_LENGTH
        if not text:
            await update.message.reply_text("(empty response)")
            return
        if len(text) <= max_len:
            await update.message.reply_text(text)
            return
        chunks: list[str] = []
        current = ""
        for line in text.split("\n"):
            # If a single line is bigger than the cap, hard-wrap it.
            while len(line) > max_len:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(line[:max_len])
                line = line[max_len:]
            if len(current) + len(line) + 1 > max_len:
                chunks.append(current)
                current = line
            else:
                current = (current + "\n" + line) if current else line
        if current:
            chunks.append(current)
        for chunk in chunks:
            await update.message.reply_text(chunk)
            await asyncio.sleep(0.4)

    async def _wait_for_task(
        self, task_id: str, timeout: int,
        progress_message=None, trace_id: str | None = None,
    ) -> dict | None:
        """Poll DB for task completion. Backs off 2s -> 10s.

        If `progress_message` (a Telegram Message) and `trace_id` are
        provided, the message text is edited every poll cycle to show
        the current pipeline stage + elapsed time. On completion the
        progress message is edited one final time to show
        '✅ Done in Xs' (or '❌ Failed in Xs') so the user sees the
        actual total. Returns the task as a dict (model_dump)."""
        start = time.time()
        poll = 2.0
        last_progress_text = ""
        while time.time() - start < timeout:
            task = await asyncio.to_thread(database.get_task, task_id)
            if task and task.status in ("completed", "failed"):
                if progress_message is not None:
                    elapsed = int(time.time() - start)
                    final = (
                        f"✅ Done in {elapsed}s\n{_build_bar(100)}"
                        if task.status == "completed"
                        else f"❌ Failed in {elapsed}s\n{_build_bar(100)}"
                    )
                    try:
                        await progress_message.edit_text(final)
                    except Exception:
                        pass
                return task.model_dump()
            if progress_message is not None and trace_id is not None:
                stage = await asyncio.to_thread(
                    _latest_stage_for_trace, trace_id,
                )
                elapsed = int(time.time() - start)
                pct = _STAGE_PCT.get(stage, 5)
                txt = f"{stage}  ({elapsed}s)\n{_build_bar(pct)}"
                if txt != last_progress_text:
                    try:
                        await progress_message.edit_text(txt)
                        last_progress_text = txt
                    except Exception:
                        # Telegram rate-limits edits; ignore
                        pass
            await asyncio.sleep(poll)
            poll = min(poll * 1.3, 10.0)
        return None

    async def _relay_chain_children(
        self, update: Update, parent_task_id: str,
        timeout: int | None = None,
    ) -> None:
        """Phase 17d -- poll for chain children of a parent task and
        relay each child's completion to chat as it lands. Plus a
        rolling 'Subtask N/M' status message that edits in place.

        Runs after a parent's chain_started result was sent to chat.
        Returns when all children have terminal status (completed
        or failed). Bounded by `timeout` (default = full task
        timeout, since each child is itself a /code).
        """
        if timeout is None:
            timeout = config.TELEGRAM_TASK_TIMEOUT
        start = time.time()
        children = await asyncio.to_thread(
            database.list_children, parent_task_id,
        )
        if not children:
            return
        total = len(children)
        already_relayed: set[str] = set()
        # Rolling status message we keep editing.
        status_msg = await update.message.reply_text(
            f"⏳ chain progress 0/{total} done"
        )
        poll = 2.0
        while time.time() - start < timeout:
            children = await asyncio.to_thread(
                database.list_children, parent_task_id,
            )
            done_ids = [
                c.task_id for c in children
                if c.status in ("completed", "failed")
            ]
            # Relay any newly-completed children we haven't
            # surfaced yet. Iterate in creation order so chat
            # shows them sequentially.
            for c in children:
                if (
                    c.task_id in done_ids
                    and c.task_id not in already_relayed
                ):
                    already_relayed.add(c.task_id)
                    await self._send_chain_child_result(update, c)
            done_count = len(done_ids)
            try:
                await status_msg.edit_text(
                    f"⏳ chain progress {done_count}/{total} done"
                )
            except Exception:
                pass
            if done_count >= total:
                # Final edit of status message.
                try:
                    await status_msg.edit_text(
                        f"🔗 chain complete — {done_count}/{total} subtasks done"
                    )
                except Exception:
                    pass
                return
            await asyncio.sleep(poll)
            poll = min(poll * 1.3, 10.0)
        # Timeout
        try:
            await status_msg.edit_text(
                f"⚠️ chain timed out after {int(time.time() - start)}s "
                f"({len(already_relayed)}/{total} subtasks done)"
            )
        except Exception:
            pass

    async def _send_chain_child_result(
        self, update: Update, child_task,
    ) -> None:
        """Phase 17d -- format + send a single child task's result
        to chat. Mirrors the result-rendering branch in handle_code
        but prefixed with a 'Subtask' marker so users can see which
        subtask of the chain they're reading."""
        idx_marker = f"📦 *Subtask `{child_task.task_id[:8]}`*"
        if child_task.status == "failed":
            err = (child_task.error or "(no error)")[:1500]
            await self._send_long(
                update,
                f"{idx_marker} ❌ failed\n\n```\n{err}\n```",
            )
            return
        result = child_task.result or {}
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                result = {"solution": result}
        if (
            isinstance(result, dict) and result.get("_error")
        ):
            err = result.get("error", "(no error)")[:1500]
            await self._send_long(
                update,
                f"{idx_marker} ❌ failed\n\n```\n{err}\n```",
            )
            return
        solved_by = result.get("solved_by", "")
        solution = (result.get("solution") or "").strip()
        # Reuse the same render branch logic as handle_code.
        if solved_by in (
            "qwen_agent", "qwen_failed", "qwen_skip_path",
            "decompose_suggested", "qwen_killed", "chain_started",
        ):
            body = solution[:3500]
        else:
            body = solution[:3500] or "(no solution body)"
        await self._send_long(update, f"{idx_marker}\n\n{body}")

    async def _route_and_wait(
        self, update: Update, command_string: str,
    ) -> None:
        """Common path: route a /command, await its task, summarize via
        brain, deliver."""
        progress_msg = await update.message.reply_text(f"🔧 On it...\n{_build_bar(0)}")
        rr = await asyncio.to_thread(route, command_string)
        if rr.status == "error":
            await update.message.reply_text(
                f"Routing error: {rr.message}"
            )
            return
        result = await self._wait_for_task(
            rr.task_id, timeout=config.TELEGRAM_TASK_TIMEOUT,
            progress_message=progress_msg, trace_id=rr.trace_id,
        )
        if result and result["status"] == "completed":
            summary = await self.brain.summarize_result(
                original_request=command_string,
                raw_result=result["result"],
                trace_id=rr.trace_id,
            )
            await self._send_long(update, summary)
        elif result and result["status"] == "failed":
            await update.message.reply_text(
                f"Failed: {result.get('error', 'Unknown error')}"
            )
        else:
            await update.message.reply_text(
                "Task timed out. /status for queue state."
            )

    # ---------- handlers ----------

    async def handle_start(self, update: Update,
                           context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        await update.message.reply_text(
            "Sentinel OS online. Type /help for commands, or just talk "
            "to me."
        )

    async def handle_help(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        await update.message.reply_text(
            "Sentinel OS Commands:\n\n"
            "── Code & AI ──\n"
            "/code <problem> -- Qwen solves; Claude teaches if needed; skip-path on proven patterns\n"
            "/claude <msg> -- Direct Claude CLI passthrough\n\n"
            "── Research & Search ──\n"
            "/search <query> -- Web search\n"
            "/research <topic> -- Compile a research brief\n\n"
            "── Jobs ──\n"
            "/jobsearch [role] -- Full pipeline (no args = PROFILE defaults)\n"
            "/extract <url> -- Extract structured job data\n"
            "/profile [init|show|set|edit] -- Manage PROFILE.yml\n"
            "/jobs [state|id] [newstate] -- Browse & advance applications\n\n"
            "── Memory ──\n"
            "/remember <key>: <value> -- Store a fact\n"
            "/forget <key> -- Delete a fact\n"
            "/recall <query> -- Search facts + episodes\n"
            "/memory -- Memory stats\n"
            "/curate [review] -- Trigger memory curation\n\n"
            "── Knowledge Base ──\n"
            "/kb -- Stats + origin breakdown\n"
            "/kb show <id> -- Full pattern record + agreement breakdown\n"
            "/kb planning -- Shadow-plan agreement stats\n"
            "/kb pin <id> / unpin <id> / restore <id> -- Lifecycle control\n\n"
            "── System ──\n"
            "/status -- Queue, GPU lock, worker\n"
            "/dashboard -- Full system snapshot\n"
            "/models -- List available models\n"
            "/schedule add|list|pause|resume|delete|runs -- Scheduled jobs\n"
            "/commit [message] -- Commit source changes\n"
            "/revert -- Undo last commit (keeps staged)\n"
            "/restart -- Graceful bot restart\n"
            "/help -- This message\n\n"
            "Or just type normally -- I'll figure out what you need."
        )

    BOT_COMMAND_MENU: list[tuple[str, str]] = [
        ("start", "Wake Sentinel / confirm it's alive"),
        ("code", "Qwen solves; Claude teaches if needed; skip-path on proven patterns"),
        ("qcode", "Qwen-only attempt — no Claude, no retries"),
        ("gwen", "Talk to Gwen directly — full run_bash access, no Claude, no retries"),
        ("prompt", "Get a brief to teach ChatGPT/Claude/Gemini how to write /qcode prompts"),
        ("encode", "Paste Python source -> get b64gz string ready for /gwen content_b64gz= arg"),
        ("gwenask", "Describe an app -> local Qwen writes a /gwen recipe (no external AI; simple apps only)"),
        ("claude", "Talk directly to Claude CLI"),
        ("search", "Web search"),
        ("extract", "Extract structured data from a job posting URL"),
        ("jobsearch", "Full job pipeline (no args=PROFILE defaults; 'view profile' to preview)"),
        ("research", "Compile a research brief on any topic"),
        ("remember", "Store a fact  /remember key: value"),
        ("forget", "Delete a stored fact  /forget key"),
        ("recall", "Search memory facts and episodes"),
        ("memory", "Memory stats (episodic + semantic counts)"),
        ("curate", "Trigger memory curation; 'review' to re-show pending"),
        ("curate_approve", "Approve a pending curation proposal"),
        ("curate_reject", "Reject a pending curation proposal"),
        ("commit", "Commit current source changes (optional message)"),
        ("revert", "Undo the last git commit (keeps changes staged)"),
        ("models", "List available local + CLI models"),
        ("status", "Queue depth, GPU lock, worker health"),
        ("kb", "KB stats; show <id>, pin/unpin, planning, stale, reteach"),
        ("restart", "Graceful restart of the Sentinel bot"),
        ("kill", "Soft-abort the in-flight /code (bails at next attempt boundary)"),
        ("schedule", "Scheduled jobs: add/list/pause/resume/delete/runs"),
        ("dashboard", "Full system snapshot: queue, GPU, scheduler, memory, disk"),
        ("profile", "Manage PROFILE.yml: init/show/set/edit"),
        ("jobs", "Browse applications; /jobs <id> <state> to advance lifecycle"),
        ("help", "List all commands"),
    ]

    async def _sync_bot_commands(self) -> None:
        """Push the canonical command menu to Telegram so the in-chat
        slash autocomplete reflects what the bot actually serves. Runs
        once on startup; idempotent on Telegram's side."""
        commands = [BotCommand(c, d) for c, d in self.BOT_COMMAND_MENU]
        try:
            await self.app.bot.set_my_commands(commands)
            log_event("SEN-system", "INFO", "telegram",
                      f"set_my_commands ok ({len(commands)} commands)")
        except Exception as e:
            log_event("SEN-system", "WARNING", "telegram",
                      f"set_my_commands failed: {e}")

    async def handle_code(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        text = " ".join(context.args) if context.args else ""
        if not text:
            await update.message.reply_text(
                "Usage: /code <describe your problem or paste code>"
            )
            return
        trace_id = generate_trace_id()
        progress_msg = await update.message.reply_text(f"🔧 Starting...\n{_build_bar(0)}")
        rr = await asyncio.to_thread(route, f"/code {text}")
        if rr.status == "error":
            await update.message.reply_text(
                f"Routing error: {rr.message}"
            )
            return
        result = await self._wait_for_task(
            rr.task_id, timeout=config.TELEGRAM_TASK_TIMEOUT,
            progress_message=progress_msg, trace_id=rr.trace_id,
        )
        if result and result["status"] == "completed":
            output = result["result"]
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except Exception:
                    output = {"solution": output, "explanation": ""}
            # Loud-failure: when the worker stored an error envelope
            # (skill raised, agent caught, returned {_error: True}),
            # show the full error to chat instead of a blank reply.
            if isinstance(output, dict) and output.get("_error"):
                err = output.get("error", "(no error message)")
                failed_at = output.get("failed_at", "?")
                await self._send_long(
                    update,
                    f"❌ /code failed at `{failed_at}`\n\n"
                    f"```\n{str(err)[:3500]}\n```\n\n"
                    f"trace: `{output.get('trace_id', rr.trace_id)}`",
                )
                return
            solved_by = output.get("solved_by", "")
            solution = (output.get("solution") or "").strip()
            # New agentic /code returns ready-to-display markdown in
            # the solution field (Qwen summary + Claude verdict + diff).
            # Legacy paths (qwen, qwen_taught, claude_direct) emit raw
            # code that needs cleaning + fence-wrapping.
            if solved_by in (
                "qwen_agent", "qwen_failed", "qwen_skip_path",
                "decompose_suggested", "qwen_killed", "chain_started",
                "qwen_solo",
            ):
                response = solution[:3800]
            else:
                from skills.code_assist import _clean_solution_text
                clean = _clean_solution_text(solution)
                explanation = (output.get("explanation") or "").strip()
                if explanation:
                    response = (
                        f"{explanation[:300]}\n\n"
                        f"```python\n{clean[:3000]}\n```"
                    )
                else:
                    response = (
                        f"Here's the code:\n\n"
                        f"```python\n{clean[:3000]}\n```"
                    )
            await self._send_long(update, response)
            # Phase 17d -- if the parent task spawned a chain, relay
            # each child completion to chat as it lands. Without this,
            # children complete in the DB but the user sees nothing
            # (children have no Telegram chat_id binding back).
            if solved_by == "chain_started":
                await self._relay_chain_children(
                    update, parent_task_id=rr.task_id,
                )
        elif result and result["status"] == "failed":
            await update.message.reply_text(
                f"Failed: {result.get('error', 'Unknown error')}"
            )
        else:
            await update.message.reply_text(
                "Task timed out. /status for queue state."
            )

    async def handle_qcode(self, update: Update,
                           context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        text = " ".join(context.args) if context.args else ""
        if not text:
            await update.message.reply_text(
                "Usage: /qcode <problem>"
            )
            return
        trace_id = generate_trace_id()
        progress_msg = await update.message.reply_text(f"🔧 Starting...\n{_build_bar(0)}")
        rr = await asyncio.to_thread(route, f"/qcode {text}")
        if rr.status == "error":
            await update.message.reply_text(
                f"Routing error: {rr.message}"
            )
            return
        result = await self._wait_for_task(
            rr.task_id, timeout=config.TELEGRAM_TASK_TIMEOUT,
            progress_message=progress_msg, trace_id=rr.trace_id,
        )
        if result and result["status"] == "completed":
            output = result["result"]
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except Exception:
                    output = {"solution": output, "explanation": ""}
            if isinstance(output, dict) and output.get("_error"):
                err = output.get("error", "(no error message)")
                failed_at = output.get("failed_at", "?")
                await self._send_long(
                    update,
                    f"\u274c /qcode failed at `{failed_at}`\n\n"
                    f"```\n{str(err)[:3500]}\n```\n\n"
                    f"trace: `{output.get('trace_id', rr.trace_id)}`",
                )
                return
            solved_by = output.get("solved_by", "")
            solution = (output.get("solution") or "").strip()
            if solved_by in (
                "qwen_solo", "qwen_agent", "qwen_failed",
            ):
                response = solution[:3800]
            else:
                from skills.code_assist import _clean_solution_text
                clean = _clean_solution_text(solution)
                explanation = (output.get("explanation") or "").strip()
                if explanation:
                    response = (
                        f"{explanation[:300]}\n\n"
                        f"```python\n{clean[:3000]}\n```"
                    )
                else:
                    response = (
                        f"Here's the code:\n\n"
                        f"```python\n{clean[:3000]}\n```"
                    )
            await self._send_long(update, response)
        elif result and result["status"] == "failed":
            await update.message.reply_text(
                f"Failed: {result.get('error', 'Unknown error')}"
            )
        else:
            await update.message.reply_text(
                "Task timed out. /status for queue state."
            )

    async def handle_gwen(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        # Phase 18 fix: read update.message.text directly so multi-line
        # literal recipes survive command parsing. ``context.args`` is
        # whitespace-tokenized and `" ".join(args)` collapses newlines
        # into spaces -- which broke _parse_recipe_steps on pasted
        # 3-line recipes (only 1 step parsed, key=value extractor
        # then greedily grabbed args from later STEPs into the first).
        # Live trigger: 2026-05-07 09:16Z, write_file got a stray
        # `command=` kwarg from STEP 2's run_bash call.
        raw_msg = (update.message.text or "") if update.message else ""
        parts = raw_msg.split(maxsplit=1)
        text = parts[1] if len(parts) > 1 else ""
        if not text:
            await update.message.reply_text(
                "Usage: /gwen <problem>"
            )
            return
        trace_id = generate_trace_id()
        progress_msg = await update.message.reply_text(
            "\U0001f9e0 Starting...\n" + _build_bar(0)
        )
        rr = await asyncio.to_thread(route, f"/gwen {text}")
        if rr.status == "error":
            await update.message.reply_text(
                f"Routing error: {rr.message}"
            )
            return
        result = await self._wait_for_task(
            rr.task_id, timeout=config.TELEGRAM_TASK_TIMEOUT,
            progress_message=progress_msg, trace_id=rr.trace_id,
        )
        if result and result["status"] == "completed":
            output = result["result"]
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except Exception:
                    output = {"solution": output, "explanation": ""}
            if isinstance(output, dict) and output.get("_error"):
                err = output.get("error", "(no error message)")
                failed_at = output.get("failed_at", "?")
                await self._send_long(
                    update,
                    f"\u274c /gwen failed at `{failed_at}`\n\n"
                    f"```\n{str(err)[:3500]}\n```\n\n"
                    f"trace: `{output.get('trace_id', rr.trace_id)}`",
                )
                return
            solved_by = output.get("solved_by", "")
            solution = (output.get("solution") or "").strip()
            if solved_by in ("gwen_ok", "gwen_failed"):
                response = solution[:3800]
            else:
                response = solution[:3800] if solution else "(no output)"
            await self._send_long(update, response)
        elif result and result["status"] == "failed":
            await update.message.reply_text(
                f"Failed: {result.get('error', 'Unknown error')}"
            )
        else:
            await update.message.reply_text(
                "Task timed out. /status for queue state."
            )

    # Per-AI prepend headers added to the brief based on observed
    # failure modes (live testing 2026-05-07). These zero in on each
    # model's actual worst habits before the AI starts the brief proper.
    _AI_PROMPT_HEADERS: ClassVar[dict[str, str]] = {
        "claude": (
            "## CLAUDE-SPECIFIC RULES (read before the rest)\n\n"
            "**FORMAT IS ALREADY DECIDED. DO NOT ASK.** Python source "
            "bundled to a single Windows .exe via PyInstaller "
            "`--onefile --noconsole`, dropped in "
            "`${SENTINEL_DEMOS_DIR}/apps/<name>_Setup.exe`. "
            "Do not ask the user about format, target, or build type.\n\n"
            "Your foundation training will tell you to be helpful by "
            "writing the app code inline as a chat reply. **DO NOT.** "
            "Your sole output is a /gwen recipe -- a single text block "
            "starting with `/gwen STEP 1:` and ending with "
            "`done summary=\"...\"`.\n\n"
            "If you find yourself drafting Python code in chat with "
            "explanation around it, STOP. Put the source INSIDE a "
            "`write_file content=\"...\"` step, not above the recipe.\n\n"
            "Forbidden patterns (live observed):\n"
            "- Preamble paragraph (\"Source uses chr(10)... Recipe below.\")\n"
            "- Markdown code fences (```) wrapping the recipe\n"
            "- Trailing prose after the `done` step\n"
            "- Inline ```python source dumps\n\n"
            "---\n\n"
        ),
        "chatgpt": (
            "## CHATGPT-SPECIFIC RULES (read before the rest)\n\n"
            "**FORMAT IS ALREADY DECIDED. DO NOT ASK.** The format is "
            "Python source bundled to a single Windows .exe via "
            "PyInstaller `--onefile --noconsole`, dropped in "
            "`${SENTINEL_DEMOS_DIR}/apps/<name>_Setup.exe`. "
            "Do not ask \"final build target: windowsexe, pythonsource, "
            "both?\". Do not ask \"what format do you want?\". Just emit "
            "the recipe.\n\n"
            "Live observed failures (2026-05-07):\n\n"
            "1. You will reach for `mkdir -p ~/foo && cd ~/foo && ...` "
            "patterns. **STOP.** Windows cmd.exe has NO `-p` flag. The "
            "&& chain short-circuits on the first failure and every "
            "subsequent step fails. Use `py -3.12 -c \"import pathlib; "
            "pathlib.Path(r'C:/...').mkdir(parents=True, exist_ok=True)\"` "
            "instead.\n\n"
            "2. You will try `npx create-electron-app` or "
            "`npm install react`. **STOP.** These take 3+ minutes and "
            "exceed the 60s run_bash timeout. Stick to Python stdlib + "
            "PyInstaller (pre-installed). NEVER include `npm install` "
            "or heavy `pip install` in a recipe.\n\n"
            "3. You will use `~/Desktop/...` in PyInstaller's --distpath "
            "/--workpath args. **STOP.** Tilde doesn't expand inside "
            "tool args. Use absolute `${SENTINEL_DEMOS_DIR}/...` paths "
            "for any tool argument.\n\n"
            "4. You will assume `code` (VS Code CLI) and `git` are in "
            "PATH. They may not be. Don't depend on them.\n\n"
            "---\n\n"
        ),
        "gemini": (
            "## GEMINI-SPECIFIC RULES (read before the rest)\n\n"
            "**FORMAT IS ALREADY DECIDED. DO NOT ASK.** Python source "
            "bundled to a single Windows .exe via PyInstaller "
            "`--onefile --noconsole`, dropped in "
            "`${SENTINEL_DEMOS_DIR}/apps/<name>_Setup.exe`. "
            "Don't ask the user format/target/etc.\n\n"
            "Live observed failures (2026-05-07):\n\n"
            "1. You will hallucinate base64. Twice now you've emitted "
            "`content_b64=` or `content_b64gz=` values that LOOK like "
            "valid base64 but decode to garbage (or in the b64gz case, "
            "produce bytes with valid zlib magic header but fabricated "
            "deflate stream). **DO NOT use `_b64` or `_b64gz` for "
            "content YOU compute.** Use plain `content=\"...\"` with "
            "`\\n` escapes, even for source up to 2.5KB.\n\n"
            "2. You will put `/gwen ` on EVERY step. **STOP.** /gwen "
            "goes ONLY on STEP 1. STEPs 2..N have no prefix. When the "
            "user's Telegram client collapses newlines, mid-line "
            "/gwen markers smush into prior step values.\n\n"
            "3. You will write `done summary=\"successfully built X\"` "
            "even if you can't verify the build worked. Use cautious "
            "wording: `done summary=\"attempted to build X; verify by "
            "checking apps/X.exe exists\"`.\n\n"
            "---\n\n"
        ),
    }

    async def handle_prompt(self, update: Update,
                             context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send PROMPT_BRIEF.md as a downloadable file. Optional arg
        selects per-AI variant: `/prompt claude`, `/prompt chatgpt`,
        `/prompt gemini` prepend an AI-specific warning section
        targeting that model's observed failure modes. `/prompt` alone
        sends the generic brief.

        Phase 18d-gz polish-8: per-AI variants. Brief grew past 25KB
        and AI-specific quirks (Claude inlines code, ChatGPT uses Unix
        + heavy installs, Gemini hallucinates base64) are easier to
        prevent at the source than diagnose downstream."""
        if not await self._check_auth(update):
            return
        brief_path = config.PERSONA_DIR / "PROMPT_BRIEF.md"
        if not brief_path.exists():
            await update.message.reply_text("❌ PROMPT_BRIEF.md not found.")
            return
        # Parse optional AI arg
        arg = (context.args[0].lower() if context.args else "").strip()
        header = self._AI_PROMPT_HEADERS.get(arg, "")
        try:
            brief_text = brief_path.read_text(encoding="utf-8")
        except Exception as e:
            await update.message.reply_text(f"❌ Could not read brief: {e}")
            return
        full_text = header + brief_text
        suffix = f"_{arg}" if arg in self._AI_PROMPT_HEADERS else ""
        filename = f"sentinel_prompt_brief{suffix}.md"
        if arg and arg not in self._AI_PROMPT_HEADERS:
            caption = (
                f"Unknown AI '{arg}'. Sending generic brief. "
                f"Valid: claude, chatgpt, gemini."
            )
        elif arg:
            caption = (
                f"Tailored for {arg.upper()}: AI-specific failure-mode "
                f"warnings prepended. Drag into the AI's file upload, "
                f"then ask for your /gwen recipe."
            )
        else:
            caption = (
                "Generic brief. For better results use "
                "/prompt claude, /prompt chatgpt, or /prompt gemini "
                "to get a per-AI tailored variant."
            )
        try:
            from io import BytesIO
            buf = BytesIO(full_text.encode("utf-8"))
            buf.name = filename
            await update.message.reply_document(
                document=buf,
                filename=filename,
                caption=caption,
            )
        except Exception as e:
            try:
                await self._send_long(
                    update,
                    f"❌ Could not send as file ({e}). Inline fallback:\n\n"
                    + full_text,
                )
            except Exception as e2:
                await update.message.reply_text(
                    f"❌ Could not send brief: {e2}"
                )

    # Tight Qwen-specific recipe-author prompt. Fits Qwen's 8K context
    # easily (~1.5KB). Format spec only -- no failure-mode lore, no
    # per-AI variants. Qwen's job is "given a small task, output a
    # 4-7-step /gwen recipe in the right shape". Capability ceiling:
    # only works well for simple stdlib-only apps. Complex / multi-file
    # / multi-format jobs require external AI via /prompt + brief.
    GWENASK_SYSTEM = (
        "You are a recipe author. Given a user's app idea, output a "
        "single /gwen recipe that builds it as a Python tkinter app, "
        "bundled to a single Windows .exe via PyInstaller, dropped at "
        "${SENTINEL_DEMOS_DIR}/apps/<Name>_Setup.exe.\n"
        "\n"
        "STRICT OUTPUT RULES:\n"
        "- Reply is ONLY the recipe. NO preamble, NO code fences, "
        "NO explanation in chat.\n"
        "- First chars: `/gwen STEP 1:`. Last chars: `done summary=\"...\"`.\n"
        "- /gwen prefix ONLY on STEP 1. STEPs 2..N have no prefix.\n"
        "- Use plain content=\"...\" with \\n for newlines. NO _b64gz.\n"
        "- Use absolute <your-home>/... paths in PyInstaller args.\n"
        "- STDLIB ONLY (tkinter, csv, sqlite3, re, urllib, json, "
        "pathlib, datetime, time, math, random, os, sys). NO pip install.\n"
        "- Source must be <2500 chars.\n"
        "\n"
        "RECIPE-QUOTING DISCIPLINE (one wrong quote breaks the build):\n"
        "- Every `key=\"value\"` MUST have BOTH the open AND close \".\n"
        "- The closing \" of `content=\"...\"` MUST appear before the "
        "next STEP marker. Put the closing \" on its OWN LINE flush left "
        "(no leading space), then a blank line, then the next STEP.\n"
        "- INSIDE content=\"...\" the Python code MUST use SINGLE QUOTES "
        "for every string literal: text='Start', f'{x}', not text=\"Start\". "
        "An unescaped \" inside content=\"...\" is read as the closing "
        "quote and silently truncates your source mid-line. NO EXCEPTIONS:\n"
        "    WRONG: tk.Label(root, text=f\"Player {p}'s turn\")\n"
        "    RIGHT: tk.Label(root, text=f'Player {p} turn')\n"
        "    WRONG: print(\"hello\")\n"
        "    RIGHT: print('hello')\n"
        "- If you NEED a single-quote inside a single-quoted string, "
        "use the chr(39) trick OR escape: 'isn\\\\'t' (backslash-apos), "
        "but easier is rephrasing to avoid apostrophes (e.g. 'Player 1 turn' "
        "instead of \"Player 1's turn\").\n"
        "- WRONG (missing close \" -> next STEP eaten):\n"
        "    STEP 2: write_file path=\"x.py\" content=\"...code...\n"
        "    root.mainloop()\n"
        "    STEP 3: ...\n"
        "- RIGHT (close \" on its own line):\n"
        "    STEP 2: write_file path=\"x.py\" content=\"...code...\n"
        "    root.mainloop()\n"
        "    \"\n"
        "    STEP 3: ...\n"
        "\n"
        "TKINTER IMPORTS:\n"
        "- ALWAYS `import tkinter as tk` at the top. Use the `tk.` "
        "prefix on every widget: tk.Tk(), tk.Frame, tk.Label, tk.Button, "
        "tk.Text, tk.BOTH, tk.LEFT, tk.END.\n"
        "- For submodules: `from tkinter import messagebox, filedialog, "
        "ttk` -- those are NOT in the tk namespace, must be imported "
        "directly. Then use messagebox.showinfo(...), ttk.Treeview(...).\n"
        "- NEVER `from tkinter import *`. It pollutes namespace AND "
        "breaks examples that use the tk. prefix (you'll mix both and "
        "get NameError on tk.Button when only Button is in scope).\n"
        "\n"
        "TKINTER ESSENTIALS (apps fail without these):\n"
        "1. ANY live-updating UI (timer, clock, animation, progress) "
        "MUST schedule itself with root.after(ms, callback). A button "
        "that 'starts' a timer must call self._tick() once, where "
        "_tick() updates the label AND calls self.root.after(50, "
        "self._tick) to repeat. Without after(), the UI never refreshes.\n"
        "2. Every Button command=self.X needs a method def X(self) on "
        "the same class. Same for label/menu callbacks.\n"
        "3. State that survives ticks lives on self (self.running, "
        "self.elapsed, etc.). Local variables vanish on each callback.\n"
        "4. Stop/pause means self.running=False and the next _tick() "
        "checks the flag and returns WITHOUT re-scheduling.\n"
        "5. Reset zeros state AND updates label text immediately so "
        "the user sees the change.\n"
        "6. NEVER mix .pack() and .grid() inside the same parent widget. "
        "Tk raises TclError. If the app has a grid (3x3 cells, calculator "
        "keypad), put the grid in its own Frame and pack the Frame; any "
        "outside controls (Reset button, status label) go in a SEPARATE "
        "Frame, also packed. Each Frame internally uses one manager.\n"
        "7. INSIDE class methods ALWAYS use self.root (or self.frame, "
        "self.parent), NEVER reference the module-level `root` variable. "
        "Pass the parent in via __init__(self, root) and store as "
        "self.root. Module globals from inside methods is a Python "
        "anti-pattern that breaks subclassing AND testability.\n"
        "FILE I/O ESSENTIALS:\n"
        "- If the task names a SPECIFIC file path (e.g. 'save to "
        "${SENTINEL_DEMOS_DIR}/outputs/notes.txt'), write directly to "
        "that path using `open(path, 'w', encoding='utf-8')`. Do NOT "
        "show a `filedialog` chooser -- the user already told you "
        "where to save. Create parent dirs first with "
        "`pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)`.\n"
        "- Use `filedialog` ONLY when the task explicitly says "
        "'let user pick' or 'open a file' (no fixed path).\n"
        "- ALWAYS use encoding='utf-8' on open() and Path.read_text/"
        "write_text -- Windows defaults to cp1252 which mangles non-ASCII.\n"
        "\n"
        "8. ALWAYS end the module with this exact 4-line block, replacing "
        "<Name> with YOUR class name (must match exactly):\n"
        "    if __name__ == '__main__':\n"
        "        root = tk.Tk()\n"
        "        <Name>(root)\n"
        "        root.mainloop()\n"
        "  CRITICAL: <Name> MUST be identical to the `class <Name>:` you "
        "defined above. If your class is `class Stopwatch:` then write "
        "`Stopwatch(root)`. If `class TicTacToe:` then `TicTacToe(root)`. "
        "Never write `App(root)` unless your class is literally named App.\n"
        "\n"
        "PICK THE RIGHT PATTERN:\n"
        "- TIMER pattern (use ONLY when the app must keep updating UI "
        "without user input -- stopwatch, clock, animation, progress "
        "bar, pomodoro): button handler sets self.running=True and calls "
        "self._tick(); _tick updates state + label, then re-schedules "
        "with self.root.after(ms, self._tick).\n"
        "- EVENT-DRIVEN pattern (use for button-press apps WITHOUT "
        "continuous motion -- counter, calculator, tic-tac-toe, notes, "
        "any app where each click does ONE thing): button handler "
        "updates state directly and calls self.lbl.config(text=...). "
        "NO _tick. NO root.after. NO self.running flag.\n"
        "\n"
        "TIMER MINI-EXAMPLE (notice EVERY widget parent is self.root):\n"
        "  class Timer:\n"
        "      def __init__(self, root):\n"
        "          self.root = root\n"
        "          self.running = False\n"
        "          self.elapsed = 0.0\n"
        "          self.lbl = tk.Label(self.root, text='0.0', font=('Arial',24))\n"
        "          self.lbl.pack()\n"
        "          tk.Button(self.root, text='Start', command=self.start).pack()\n"
        "      def start(self):\n"
        "          if not self.running:\n"
        "              self.running = True\n"
        "              self._tick()\n"
        "      def _tick(self):\n"
        "          if not self.running: return\n"
        "          self.elapsed += 0.05\n"
        "          self.lbl.config(text=f'{self.elapsed:.1f}')\n"
        "          self.root.after(50, self._tick)\n"
        "\n"
        "EVENT-DRIVEN MINI-EXAMPLE (counter):\n"
        "  class Counter:\n"
        "      def __init__(self, root):\n"
        "          self.root = root\n"
        "          self.value = 0\n"
        "          self.lbl = tk.Label(self.root, text='0', font=('Arial',24))\n"
        "          self.lbl.pack()\n"
        "          tk.Button(self.root, text='+', command=self.inc).pack()\n"
        "          tk.Button(self.root, text='-', command=self.dec).pack()\n"
        "      def inc(self):\n"
        "          self.value += 1\n"
        "          self.lbl.config(text=str(self.value))\n"
        "      def dec(self):\n"
        "          self.value -= 1\n"
        "          self.lbl.config(text=str(self.value))\n"
        "\n"
        "GRID MINI-EXAMPLE (3x3 cells, the right way):\n"
        "  class Grid3x3:\n"
        "      def __init__(self, root):\n"
        "          self.root = root\n"
        "          self.frame = tk.Frame(self.root)\n"
        "          self.frame.pack()\n"
        "          self.cells = []\n"
        "          for i in range(3):\n"
        "              row = []\n"
        "              for j in range(3):\n"
        "                  b = tk.Button(self.frame, text='', width=4,\n"
        "                                command=lambda r=i, c=j: self.click(r, c))\n"
        "                  b.grid(row=i, column=j)  # use the loop indices\n"
        "                  row.append(b)\n"
        "              self.cells.append(row)\n"
        "      def click(self, r, c):\n"
        "          self.cells[r][c].config(text='X')\n"
        "\n"
        "RECIPE SHAPE (5 steps):\n"
        "STEP 1: run_bash command=\"py -3.12 -c \\\"import pathlib; "
        "[pathlib.Path(r'${SENTINEL_DEMOS_DIR}/'+d)"
        ".mkdir(parents=True, exist_ok=True) for d in "
        "['apps','sources','build-intermediates/<name>-build']]\\\"\"\n"
        "STEP 2: write_file path=\"${SENTINEL_DEMOS_DIR}"
        "/sources/<Name>.py\" content=\"<source with \\n escapes>\"\n"
        "STEP 3: run_bash command=\"cd "
        "${SENTINEL_DEMOS_DIR}/build-intermediates/<name>-build && py -3.12 -m "
        "PyInstaller --onefile --noconsole --name <Name>_Setup "
        "${SENTINEL_DEMOS_DIR}/sources/<Name>.py\"\n"
        "STEP 4: run_bash command=\"py -3.12 -c \\\"import shutil,os; "
        "shutil.copy(r'${SENTINEL_DEMOS_DIR}/build-"
        "intermediates/<name>-build/dist/<Name>_Setup.exe', "
        "r'${SENTINEL_DEMOS_DIR}/apps/<Name>_Setup.exe'); "
        "print(os.path.getsize(r'${SENTINEL_DEMOS_DIR}"
        "/apps/<Name>_Setup.exe'))\\\"\"\n"
        "STEP 5: done summary=\"<Name>_Setup.exe built\"\n"
        "\n"
        "Pick a PascalCase <Name> from the user's task (e.g. Stopwatch, "
        "Counter, TicTacToe). Write minimal working tkinter source for "
        "the task in STEP 2's content arg, applying TKINTER ESSENTIALS.\n"
    )

    # Substitute ${SENTINEL_DEMOS_DIR} placeholders with the user's actual
    # configured demos path (defaults to ~/Desktop/Sentinel-Demos; override
    # via SENTINEL_DEMOS_DIR env var). Done at class-load so the AI sees
    # the resolved path, not a placeholder. _Template imported at module
    # level (see top of file) -- class-body locals aren't visible inside
    # comprehensions, which create their own scope.
    GWENASK_SYSTEM = _string_Template(GWENASK_SYSTEM).safe_substitute(
        SENTINEL_DEMOS_DIR=config.SENTINEL_DEMOS_DIR_POSIX,
    )
    _AI_PROMPT_HEADERS = {
        _k: _string_Template(_v).safe_substitute(
            SENTINEL_DEMOS_DIR=config.SENTINEL_DEMOS_DIR_POSIX,
        )
        for _k, _v in _AI_PROMPT_HEADERS.items()
    }

    async def handle_gwenask(self, update: Update,
                              context: ContextTypes.DEFAULT_TYPE) -> None:
        """User describes an app idea in plain English. Local Qwen
        Coder writes a /gwen recipe and posts it back. User reviews
        and pastes the recipe into a separate /gwen message to build.

        In-house, no external AI in the loop. Capability ceiling:
        simple stdlib-only apps (Stopwatch, TicTacToe, Counter, simple
        Note utility). Complex / multi-format / dependency-needing
        tasks should use /prompt + external AI instead.

        Phase 18d-gz polish-12.
        """
        if not await self._check_auth(update):
            return
        raw_msg = (update.message.text or "") if update.message else ""
        parts = raw_msg.split(maxsplit=1)
        task = parts[1] if len(parts) > 1 else ""
        if not task.strip():
            await update.message.reply_text(
                "Usage: /gwenask <app idea>\n\n"
                "Examples:\n"
                "  /gwenask build me a stopwatch with start/stop/reset\n"
                "  /gwenask tic-tac-toe app, click cells to play X/O\n"
                "  /gwenask simple counter with + and - buttons\n\n"
                "I'll write a /gwen recipe and post it back. You then "
                "paste the recipe (as a separate message) to build.\n\n"
                "Limit: stdlib-only, source under 1500 chars. For "
                "complex apps use /prompt + external AI."
            )
            return
        progress_msg = await update.message.reply_text(
            "🧠 Asking local Qwen to author a recipe (~10-30s)..."
        )
        # Lazy import to avoid loading skills.code_assist at startup
        from skills.code_assist import _qwen_generate
        from core.telemetry import generate_trace_id
        trace_id = generate_trace_id()
        try:
            recipe = await asyncio.to_thread(
                _qwen_generate,
                self.GWENASK_SYSTEM, task, trace_id, config.WORKER_MODEL,
                180, False, 6144,  # timeout=180s, format_json=False, num_predict=6144
            )
        except Exception as e:
            await progress_msg.edit_text(f"❌ Qwen failed: {type(e).__name__}: {e}")
            return
        recipe = (recipe or "").strip()
        recipe = re.sub(r"^```\w*\n", "", recipe)
        recipe = re.sub(r"\n```\s*$", "", recipe)
        if not re.match(r"^\s*/gwen\s+STEP\s+1:", recipe):
            if re.match(r"^\s*STEP\s+1:", recipe):
                recipe = "/gwen " + recipe.lstrip()
        # Phase 18-fix-1 (2026-05-07): rewrite plain content="..." into
        # content_b64gz="..." server-side so the recipe survives Telegram
        # paste-quote-stripping. Qwen can't hand-compute base64, so we do
        # it for it after generation. Bug surfaced when Qwen-authored
        # Stopwatch recipe had outer double-quotes stripped on user paste,
        # write_file step silently failed, executor falsely reported OK.
        recipe = self._wrap_plain_content_as_b64gz(recipe)
        # Self-correct: one retry if AST/name validation fails.
        validation_err = self._validate_recipe_source(recipe)
        if validation_err is not None:
            await progress_msg.edit_text(
                f"⚠️ Validation failed ({validation_err[:80]}), "
                f"retrying once..."
            )
            retry_user = (
                f"{task}\n\n"
                f"PREVIOUS ATTEMPT FAILED VALIDATION: {validation_err} "
                f"Fix this exact problem and re-emit the COMPLETE recipe."
            )
            try:
                recipe2 = await asyncio.to_thread(
                    _qwen_generate,
                    self.GWENASK_SYSTEM, retry_user, trace_id,
                    config.WORKER_MODEL, 180, False, 6144,
                )
            except Exception as e:
                await progress_msg.edit_text(
                    f"❌ Qwen retry failed: {type(e).__name__}: {e}"
                )
                return
            recipe2 = (recipe2 or "").strip()
            recipe2 = re.sub(r"^```\w*\n", "", recipe2)
            recipe2 = re.sub(r"\n```\s*$", "", recipe2)
            if not re.match(r"^\s*/gwen\s+STEP\s+1:", recipe2):
                if re.match(r"^\s*STEP\s+1:", recipe2):
                    recipe2 = "/gwen " + recipe2.lstrip()
            recipe = self._wrap_plain_content_as_b64gz(recipe2)
        await progress_msg.delete()
        await self._send_long(
            update,
            f"📝 Qwen wrote this recipe ({len(recipe)} chars). "
            f"Review, edit if needed, then paste into a NEW message "
            f"to build:\n\n{recipe}",
        )

    @staticmethod
    def _validate_recipe_source(recipe: str) -> str | None:
        """Decode the b64gz content from a wrapped recipe, run ast.parse
        on the source, and verify class-name vs module-level instantiation
        consistency. Returns None if clean, or a short human-readable
        error string if validation fails. Used by handle_gwenask to drive
        a one-shot self-correction retry against Qwen.
        """
        import ast as _ast
        import base64 as _b64
        import zlib as _zlib
        m = re.search(r'content_b64gz="([A-Za-z0-9+/=]+)"', recipe)
        if m is None:
            return None  # nothing to validate (not a write_file recipe)
        try:
            src = _zlib.decompress(_b64.b64decode(m.group(1))).decode("utf-8")
        except Exception as e:
            return f"content_b64gz decode failed: {type(e).__name__}"
        try:
            tree = _ast.parse(src)
        except SyntaxError as e:
            return (f"source has SyntaxError: {e.msg} at line {e.lineno}. "
                    f"Re-emit the source with proper Python syntax. "
                    f"REMEMBER: SINGLE QUOTES inside content; close every "
                    f"paren/bracket; consistent class name.")
        # Name-consistency: find class names defined and Name(root)
        # calls at module level.
        class_names = {n.name for n in _ast.walk(tree)
                       if isinstance(n, _ast.ClassDef)}
        module_calls: list[str] = []
        for node in tree.body:
            if isinstance(node, _ast.If):
                for inner in node.body:
                    if (isinstance(inner, _ast.Expr)
                            and isinstance(inner.value, _ast.Call)
                            and isinstance(inner.value.func, _ast.Name)):
                        module_calls.append(inner.value.func.id)
            if (isinstance(node, _ast.Expr)
                    and isinstance(node.value, _ast.Call)
                    and isinstance(node.value.func, _ast.Name)):
                module_calls.append(node.value.func.id)
        # Drop ones that are clearly not a class (like `print`, `tk.Tk`)
        suspects = [c for c in module_calls if c[:1].isupper() and c != "Tk"]
        for c in suspects:
            if c not in class_names:
                return (f"module-level calls `{c}(root)` but no `class "
                        f"{c}:` is defined in the source (defined classes: "
                        f"{sorted(class_names)}). Use the SAME name in both "
                        f"places. Re-emit the recipe with class name "
                        f"matching the instantiation.")
        return None

    @staticmethod
    def _wrap_plain_content_as_b64gz(recipe: str) -> str:
        """Find every `write_file ... content="<source>"` step and rewrite
        the content arg as `content_b64gz="<gzip+base64>"`. Paste-mangle-
        immune: base64 alphabet survives Telegram quote-stripping +
        smart-quote conversion + soft-wrap.

        Step-aware: splits the recipe at STEP-boundary markers BEFORE
        running the content regex, so a Qwen-authored content="..."
        missing its closing " cannot scoop up the next step's text.
        Such a step is left untouched (the parser/executor then surface
        the malformed recipe as an abort). Idempotent.
        """
        import base64 as _b64
        import zlib as _zlib
        # Split on STEP boundaries (preserve markers via lookahead).
        # Pattern matches: optional /gwen prefix + STEP + digits + colon.
        chunks = re.split(
            r'(?=^\s*(?:/gwen\s+)?STEP\s+\d+:)',
            recipe, flags=re.MULTILINE,
        )

        def _encode(raw: str) -> str:
            # Best-effort de-escape; harmless if no escape sequences.
            decoded = (raw
                       .replace("\\n", "\n")
                       .replace("\\t", "\t")
                       .replace('\\"', '"')
                       .replace("\\\\", "\\"))
            return _b64.b64encode(
                _zlib.compress(decoded.encode("utf-8"), 9)
            ).decode("ascii")

        # Anchor: find `<prefix> ... content=` then take everything
        # to end-of-chunk. Qwen frequently leaves unescaped " inside
        # content (text="...", f"...", etc.); the strict regex used to
        # match only up to the first inner ", silently truncating
        # the source. Greedy-to-end gets the whole blob.
        anchor = re.compile(
            r'(write_file\s+[^\n]*?\bcontent)\s*=\s*"',
            re.DOTALL,
        )

        def _process_chunk(chunk: str) -> str:
            if "content_b64gz" in chunk:
                return chunk
            m = anchor.search(chunk)
            if m is None:
                return chunk
            prefix = m.group(1)  # "write_file ... content"
            head = chunk[:m.start()]
            tail = chunk[m.end():]  # everything after the opening "
            # Strip trailing " and any blank lines (Qwen sometimes
            # closes properly with " on its own line; we re-emit cleanly).
            payload = tail.rstrip()
            if payload.endswith('"'):
                payload = payload[:-1].rstrip()
            try:
                gzb64 = _encode(payload)
            except Exception:
                return chunk
            return f'{head}{prefix}_b64gz="{gzb64}"\n'

        return "".join(_process_chunk(c) for c in chunks)

    async def handle_encode(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Encode pasted Python source as gzip+base64 ready to drop
        into a /gwen recipe's `content_b64gz="..."` arg.

        Closes the AI-can't-hand-compute-base64 ceiling: AI writes the
        source freely (any size), user pastes it via /encode, gets
        back a recipe-ready b64gz string. Then any AI can produce
        large-source recipes by just providing the source itself and
        a recipe template; user glues the b64gz value in.

        Phase 18d-gz polish-11.
        """
        if not await self._check_auth(update):
            return
        # Read raw message text (preserves multi-line source per
        # Phase 18 fix), strip the leading /encode token.
        raw_msg = (update.message.text or "") if update.message else ""
        parts = raw_msg.split(maxsplit=1)
        source = parts[1] if len(parts) > 1 else ""
        if not source.strip():
            await update.message.reply_text(
                "Usage: /encode <python source>\n\n"
                "Paste your Python source on the next lines. I'll "
                "return a gzip+base64 string you drop into a /gwen "
                "recipe step:\n\n"
                "  STEP N: write_file path=\"...\" content_b64gz=\"<value>\"\n\n"
                "Paste-mangle-immune (no quotes/newlines in b64 alphabet)."
            )
            return
        import base64 as _b64m
        import zlib as _zlib
        try:
            gzb64 = _b64m.b64encode(
                _zlib.compress(source.encode("utf-8"), 9)
            ).decode("ascii")
        except Exception as e:
            await update.message.reply_text(f"❌ encode failed: {e}")
            return
        in_chars = len(source)
        out_chars = len(gzb64)
        ratio = in_chars / out_chars if out_chars else 0
        head = (
            f"✅ encoded {in_chars} chars source → {out_chars} chars "
            f"b64gz ({ratio:.2f}x compression)\n\n"
        )
        # If the b64gz value fits comfortably as one Telegram message
        # (<3500 chars), send inline as ready-to-paste snippet. Else
        # send as a .txt file.
        snippet = f'content_b64gz="{gzb64}"'
        if len(snippet) <= 3500:
            await update.message.reply_text(
                head + "Paste-ready snippet:\n\n" + snippet,
            )
        else:
            try:
                from io import BytesIO
                buf = BytesIO(snippet.encode("utf-8"))
                buf.name = "content_b64gz.txt"
                await update.message.reply_document(
                    document=buf,
                    filename="content_b64gz.txt",
                    caption=(
                        head
                        + f"Snippet too long for inline ({len(snippet)} chars). "
                        f"Download + paste into your write_file step."
                    ),
                )
            except Exception as e:
                await update.message.reply_text(
                    f"{head}❌ Could not send as file: {e}\n"
                    f"First 500 chars: {snippet[:500]}..."
                )

    CLAUDE_PASSTHROUGH_SYSTEM = (
        "You are a senior engineer answering directly. If the user asks "
        "you to teach or explain, walk through it clearly with concrete "
        "examples and reasoning. If they ask a quick question, answer "
        "concisely. Match the depth they asked for. Never refuse a "
        "reasonable coding/teaching request. No preamble like 'Sure!' "
        "or 'I'd be happy to' -- get to the answer."
    )

    async def handle_claude(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        text = " ".join(context.args) if context.args else ""
        if not text:
            await update.message.reply_text(
                "Usage: /claude <your question>"
            )
            return
        trace_id = generate_trace_id()
        log_event(
            trace_id, "INFO", "telegram",
            f"/claude prompt[:200]={text[:200]!r}",
        )
        progress_msg = await update.message.reply_text(
            f"🎓 Asking Claude...\n{_build_bar(0)}"
        )
        result = await self.claude_cli.generate(
            prompt=text,
            system=self.CLAUDE_PASSTHROUGH_SYSTEM,
            timeout=config.TELEGRAM_CLAUDE_TIMEOUT,
            trace_id=trace_id,
        )
        if result.success:
            log_event(
                trace_id, "INFO", "telegram",
                f"/claude response[:300]={result.text[:300]!r}",
            )
            try:
                await progress_msg.edit_text(f"✅ Done\n{_build_bar(100)}")
            except Exception:
                pass
            await self._send_long(update, result.text)
        else:
            log_event(
                trace_id, "WARNING", "telegram",
                f"/claude failed: {result.error}",
            )
            try:
                await progress_msg.edit_text(f"❌ Failed\n{_build_bar(100)}")
            except Exception:
                pass
            await update.message.reply_text(
                f"Claude error: {result.error}"
            )

    async def handle_search(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        text = " ".join(context.args) if context.args else ""
        if not text:
            await update.message.reply_text("Usage: /search <query>")
            return
        await self._route_and_wait(update, f"/search {text}")

    async def handle_jobsearch(self, update: Update,
                               context: ContextTypes.DEFAULT_TYPE) -> None:
        """Phase 12 + extension: three modes.

          /jobsearch                     -- run with profile defaults
          /jobsearch view profile        -- show what profile defaults
                                            would be used; do NOT run
          /jobsearch <query> [--flags]   -- explicit query with flag
                                            overrides over profile defaults

        Top-3 broadcast always fires after a real run."""
        if not await self._check_auth(update):
            return
        args = list(context.args) if context.args else []
        # ---- view profile ----
        if (len(args) >= 2 and args[0].lower() == "view"
                and args[1].lower() == "profile"):
            await self._jobsearch_view_profile(update)
            return
        # ---- build the command string, defaulting from profile ----
        from core import job_profile as _jp
        profile = _jp.load_profile()
        if not args:
            # No-args mode: pull query + location + workplace from profile.
            primary = profile.target_roles.primary
            if not primary:
                await update.message.reply_text(
                    "PROFILE.yml has no target_roles.primary set. "
                    "Run /profile init or pass a query: "
                    "/jobsearch \"Regional Sales Manager\""
                )
                return
            query = primary[0]
            loc = profile.location.primary_city or "Detroit, MI"
            workplace = profile.location.workplace_preference
            text = (
                f"\"{query}\" --location \"{loc}\" "
                f"--workplace {workplace}"
            )
            await update.message.reply_text(
                f"🔎 Using PROFILE defaults: query='{query}' "
                f"location='{loc}' workplace='{workplace}'.\n"
                f"(Use /jobsearch view profile to see all defaults, "
                f"or pass flags to override.)"
            )
        else:
            text = " ".join(args)
        command_string = f"/jobsearch {text}"
        progress_msg = await update.message.reply_text(
            f"🔧 Searching jobs...\n{_build_bar(0)}"
        )
        rr = await asyncio.to_thread(route, command_string)
        if rr.status == "error":
            await update.message.reply_text(
                f"Routing error: {rr.message}"
            )
            return
        result = await self._wait_for_task(
            rr.task_id, timeout=config.TELEGRAM_TASK_TIMEOUT,
            progress_message=progress_msg, trace_id=rr.trace_id,
        )
        if not result:
            await update.message.reply_text(
                "Task timed out. /status for queue state."
            )
            return
        if result.get("status") == "failed":
            await update.message.reply_text(
                f"Failed: {result.get('error', 'Unknown error')}"
            )
            return
        # Brain-summarized result first (the existing path).
        try:
            summary = await self.brain.summarize_result(
                original_request=command_string,
                raw_result=result["result"],
                trace_id=rr.trace_id,
            )
            await self._send_long(update, summary)
        except Exception as e:
            log_event(rr.trace_id, "WARNING", "telegram",
                      f"jobsearch summary failed: {type(e).__name__}: {e}")
        # Phase 12: top-3 broadcast. The job_report skill stashes a
        # pre-formatted Telegram-ready string in result.top_n_telegram
        # so the bot doesn't need to know the scoring schema.
        raw = result.get("result")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = None
        top_msg = ""
        if isinstance(raw, dict):
            top_msg = (raw.get("top_n_telegram") or "").strip()
        if top_msg:
            await self._send_long(update, top_msg)

        # Phase 13 (Batch 5a): adaptive title filter. If the scrape
        # dropped a high fraction to title-filter, ask the brain for
        # 1-3 candidate negative keywords and append them to PROFILE.
        # Best-effort -- any failure here is logged but never bubbles
        # up; the /jobsearch user already got their results.
        try:
            await self._maybe_run_adaptive_filter(
                update, trace_id=rr.trace_id,
            )
        except Exception as e:
            log_event(
                rr.trace_id, "WARNING", "telegram",
                f"adaptive filter post-pass failed: "
                f"{type(e).__name__}: {e}",
            )

    async def _maybe_run_adaptive_filter(
        self, update: Update, trace_id: str,
    ) -> None:
        """Post-/jobsearch hook: read the scrape stats sidecar, decide
        whether to act, ask the brain for new negatives, append them to
        PROFILE.title_filter.negative, notify the user.

        Phase 15b: any persistent writes triggered from this code path
        are background work, not user-driven. The wrapper sets the
        contextvar for the whole hook so any future KB write that
        adaptive_filter eventually performs (or that a brain call here
        triggers as a side-effect) is correctly attributed."""
        from core import adaptive_filter
        from core.write_origin import (
            BACKGROUND, set_current_write_origin,
            reset_current_write_origin,
        )
        token = set_current_write_origin(BACKGROUND)
        try:
            await self._adaptive_filter_body(
                update, trace_id, adaptive_filter,
            )
        finally:
            reset_current_write_origin(token)

    async def _adaptive_filter_body(
        self, update: Update, trace_id: str, adaptive_filter,
    ) -> None:
        stats_path = getattr(
            config, "LAST_SCRAPE_STATS_PATH",
            config.LOG_DIR / "last_scrape_stats.json",
        )
        if not stats_path.exists():
            return
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception as e:
            log_event(
                trace_id, "DEBUG", "telegram",
                f"adaptive filter: stats read failed: "
                f"{type(e).__name__}: {e}",
            )
            return
        dropped = int(stats.get("dropped_title", 0))
        scraped = int(stats.get("scraped_total", 0))
        if not adaptive_filter.should_act(dropped, scraped):
            return
        sample = stats.get("dropped_title_sample") or []
        if not sample:
            return
        # Closure that adapts the brain's OllamaClient.generate to the
        # signature adaptive_filter.extract_candidates expects.
        from core.llm import OllamaClient
        client = OllamaClient()

        def brain_generate(prompt, system, format_json, trace_id):
            return client.generate(
                model=config.BRAIN_MODEL, prompt=prompt, system=system,
                format_json=format_json, trace_id=trace_id,
            )
        from core import job_profile as _jp
        profile = _jp.load_profile(trace_id)
        candidates = await asyncio.to_thread(
            adaptive_filter.extract_candidates,
            sample, profile, brain_generate, trace_id,
        )
        if not candidates:
            return
        added = await asyncio.to_thread(
            adaptive_filter.apply_to_profile, candidates, trace_id,
        )
        if added > 0:
            ratio_pct = int(round(100.0 * dropped / max(1, scraped)))
            await update.message.reply_text(
                f"🧠 Adaptive filter: title pre-filter dropped "
                f"{dropped}/{scraped} ({ratio_pct}%) postings. Added "
                f"{added} new negative keyword{'s' if added != 1 else ''} "
                f"to PROFILE.title_filter.negative: "
                + ", ".join(f"{c!r}" for c in candidates[:added])
                + ". Next /jobsearch will be more selective. Edit via "
                "/profile if needed."
            )

    async def _jobsearch_view_profile(self, update: Update) -> None:
        """Show what /jobsearch (no args) would run with current PROFILE.yml.
        Does NOT execute anything -- view-only."""
        from core import job_profile as _jp
        profile = _jp.load_profile()
        primary = profile.target_roles.primary or []
        archetypes_active = [
            a.name for a in profile.target_roles.archetypes
            if a.fit != "skip"
        ]
        avoid = profile.avoid_companies or []
        positives = profile.title_filter.positive
        negatives = profile.title_filter.negative
        loc = profile.location
        comp = profile.compensation
        lines = [
            "📇 /jobsearch defaults from PROFILE.yml:",
            "",
            f"• Query: {primary[0] if primary else '(none -- set target_roles.primary)'}",
            f"• Other primary roles in rotation (Phase 13): "
            f"{', '.join(primary[1:]) if len(primary) > 1 else '(only one)'}",
            f"• Location: {loc.primary_city}"
            + (f" (zip {loc.primary_zip})" if loc.primary_zip else ""),
            f"• Workplace: {loc.workplace_preference}"
            + (f" (on-site/hybrid must be ≤{loc.onsite_max_miles} mi from "
               f"{loc.primary_zip})"
               if loc.onsite_max_miles and loc.primary_zip else ""),
            f"• Distance: {config.JOBSPY_DEFAULT_DISTANCE} mi",
            f"• Hours back: {config.JOBSPY_DEFAULT_HOURS_OLD} h",
            f"• Results wanted: {config.JOBSPY_DEFAULT_RESULTS}",
            f"• Sites: {', '.join(config.JOBSPY_SITES)}",
            f"• Will not relocate: {not loc.willing_to_relocate}",
            "",
            f"• Active archetypes ({len(archetypes_active)}): "
            f"{', '.join(archetypes_active) or '(none)'}",
            f"• Comp target: {comp.target_range_usd or '(unset)'}",
            f"• Comp floor: {comp.minimum_usd or '(unset)'}",
            "",
            f"• Title-filter positives ({len(positives)}): "
            f"{', '.join(positives[:10])}"
            + (f" ... +{len(positives) - 10}" if len(positives) > 10 else ""),
            f"• Title-filter negatives ({len(negatives)}): "
            f"{', '.join(negatives[:10])}"
            + (f" ... +{len(negatives) - 10}" if len(negatives) > 10 else ""),
            f"• Avoid companies ({len(avoid)}): {', '.join(avoid) or '(none)'}",
            "",
            "To override for a single run, use flags. Examples:",
            "  /jobsearch                              -- run as shown above",
            "  /jobsearch \"Account Executive\"         -- different query, "
            "rest from profile",
            "  /jobsearch --workplace remote --hours 24",
            "  /jobsearch --avoid \"Aerotek,Robert Half\"",
            "",
            "To make a permanent change: /profile set <key.path> <value> "
            "(e.g. /profile set location.workplace_preference hybrid).",
        ]
        await self._send_long(update, "\n".join(lines))

    async def handle_extract(self, update: Update,
                             context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        text = " ".join(context.args) if context.args else ""
        if not text:
            await update.message.reply_text(
                "Usage: /extract <paste job posting text>"
            )
            return
        await self._route_and_wait(update, f"/extract {text}")

    async def handle_status(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        pending = await asyncio.to_thread(
            database.count_tasks_by_status, "pending",
        )
        processing = await asyncio.to_thread(
            database.count_tasks_by_status, "processing",
        )
        completed = await asyncio.to_thread(
            database.count_tasks_by_status, "completed",
        )
        failed = await asyncio.to_thread(
            database.count_tasks_by_status, "failed",
        )
        loaded = await asyncio.to_thread(self.inference.get_loaded_model)
        uptime_s = int(time.monotonic() - self._started_at)
        h, rem = divmod(uptime_s, 3600)
        m, s = divmod(rem, 60)
        await update.message.reply_text(
            f"Sentinel OS Status:\n"
            f"Queue: {pending} pending, {processing} processing\n"
            f"History: {completed} completed, {failed} failed\n"
            f"GPU: {loaded or 'No model loaded'}\n"
            f"Brain: {config.BRAIN_MODEL}\n"
            f"Worker: {config.WORKER_MODEL}\n"
            f"Uptime: {h}h {m}m {s}s"
        )

    async def handle_models(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        models = self.inference.model_registry.list_models()
        await asyncio.to_thread(
            self.inference.model_registry.check_availability,
        )
        lines = ["Registered Models:"]
        for m in models:
            status = "ready" if m.available else "not pulled"
            lines.append(
                f"  {m.name} ({m.backend}) -- "
                f"{m.capability_tier} -- {status}"
            )
        await update.message.reply_text("\n".join(lines))

    async def handle_kb(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE) -> None:
        """Phase 8 + Phase 14a + Phase 15a additions:

          /kb                  -- stats summary (entries + graduation)
          /kb verify <id>      -- run a graduation test on a pattern
          /kb retake <id>      -- clear the needs_reteach flag (after
                                  you've manually re-taught it)
          /kb stale [days]     -- list patterns not verified in N days
          /kb reteach          -- list patterns currently flagged
                                  needs_reteach
          /kb pin <id>         -- protect a pattern from auto-archival
          /kb unpin <id>       -- clear pinned flag (does NOT archive)
          /kb restore <id>     -- bring an archived pattern back
          /kb planning         -- shadow-planning agreement stats
                                  (Claude vs Qwen plan overlap)
          /kb show <id>        -- full record for a pattern: recipes,
                                  agreement breakdown, graduation
                                  stats, lifecycle (Phase 16)
        """
        if not await self._check_auth(update):
            return
        args = list(context.args) if context.args else []

        if not args:
            stats = await asyncio.to_thread(self.kb.stats)
            grad = await asyncio.to_thread(self.kb.graduation_stats)
            origins = await asyncio.to_thread(self.kb.origin_breakdown)
            origin_line = (
                "Origin breakdown: "
                + ", ".join(f"{k}={v}" for k, v in origins.items())
            ) if origins else "Origin breakdown: (no rows)"
            await update.message.reply_text(
                f"📚 Knowledge Base:\n"
                f"Total entries: {stats['total_entries']}\n"
                f"Patterns: {stats['patterns_count']}\n"
                f"Limitations: {stats['limitations_count']}\n"
                f"Avg usage: {stats['avg_usage_count']:.1f}\n"
                f"\n"
                f"🎓 Graduation (Phase 14a):\n"
                f"Verified patterns: {grad['verified_patterns']} of "
                f"{grad['total_patterns']}\n"
                f"Needs reteach: {grad['needs_reteach']}\n"
                f"Solo attempts: {grad['solo_attempts']} "
                f"({grad['solo_passes']} passed, "
                f"rate {grad['solo_pass_rate']:.0%})\n"
                f"\n"
                f"✍️ Provenance (Phase 15b):\n"
                f"{origin_line}"
            )
            return

        sub = args[0].lower()

        if sub == "verify" and len(args) >= 2 and args[1].isdigit():
            pid = int(args[1])
            pattern = await asyncio.to_thread(self.kb.get_pattern, pid)
            if pattern is None:
                await update.message.reply_text(
                    f"No pattern with id={pid}.",
                )
                return
            await update.message.reply_text(
                f"🔧 Running graduation on pattern #{pid}: "
                f"{pattern.problem_summary[:60]}\n"
                f"This calls Qwen + executor (~15-30s)..."
            )
            from core.kb_graduation import graduate_pattern
            from core.llm import INFERENCE_CLIENT
            mname = config.WORKER_MODEL
            mcfg = INFERENCE_CLIENT.model_registry.get(mname)
            mid = mcfg.model_id if mcfg else mname
            grad = await graduate_pattern(
                pattern_id=pid,
                problem=pattern.problem_summary,
                code_context=None,
                kb=self.kb, model_id=mid,
                trace_id=f"SEN-kb-verify-{pid}",
            )
            mark = "✓" if grad["passed"] else "✗"
            warn = (
                "\n⚠️ Pattern is now flagged needs_reteach."
                if grad["needs_reteach"] else ""
            )
            await update.message.reply_text(
                f"{mark} Graduation: {grad['solo_passes']}/"
                f"{grad['solo_attempts']} solo "
                f"({grad['duration_s']}s){warn}"
            )
            return

        if sub == "retake" and len(args) >= 2 and args[1].isdigit():
            pid = int(args[1])
            ok = await asyncio.to_thread(self.kb.clear_needs_reteach, pid)
            if not ok:
                await update.message.reply_text(
                    f"No pattern with id={pid}.",
                )
                return
            await update.message.reply_text(
                f"✓ Pattern #{pid} flag cleared. Future matches will use "
                "it as a few-shot example again."
            )
            return

        if sub == "stale":
            days = 30
            if len(args) >= 2 and args[1].isdigit():
                days = max(1, int(args[1]))
            stale = await asyncio.to_thread(self.kb.list_stale, days)
            if not stale:
                await update.message.reply_text(
                    f"No patterns stale beyond {days} days. ✨",
                )
                return
            lines = [f"📅 Patterns not verified in {days}+ days "
                     f"(top {len(stale)}):"]
            for e in stale[:20]:
                last = e.last_verified_at[:10] if e.last_verified_at else "never"
                lines.append(
                    f"  [{e.id}] {e.problem_summary[:60]} — last: {last}"
                )
            lines.append("\nVerify any: /kb verify <id>")
            await update.message.reply_text("\n".join(lines))
            return

        if sub == "reteach":
            flagged = await asyncio.to_thread(self.kb.list_needs_reteach)
            if not flagged:
                await update.message.reply_text(
                    "No patterns currently flagged needs_reteach. ✨",
                )
                return
            lines = [
                f"⚠️ {len(flagged)} pattern(s) flagged needs_reteach "
                f"(escalating to Claude on next match):"
            ]
            for e in flagged[:20]:
                rate = (
                    f"{e.solo_passes}/{e.solo_attempts}"
                    if e.solo_attempts else "0/0"
                )
                lines.append(
                    f"  [{e.id}] {e.problem_summary[:60]} — solo {rate}"
                )
            lines.append(
                "\nRetry: /kb verify <id>  •  Clear flag: /kb retake <id>"
            )
            await update.message.reply_text("\n".join(lines))
            return

        if sub == "pin" and len(args) >= 2 and args[1].isdigit():
            pid = int(args[1])
            ok = await asyncio.to_thread(self.kb.pin_pattern, pid)
            if not ok:
                await update.message.reply_text(
                    f"No pattern with id={pid}.",
                )
                return
            await update.message.reply_text(
                f"📌 Pattern #{pid} pinned. "
                "Auto-archival will skip it."
            )
            return

        if sub == "unpin" and len(args) >= 2 and args[1].isdigit():
            pid = int(args[1])
            ok = await asyncio.to_thread(self.kb.unpin_pattern, pid)
            if not ok:
                await update.message.reply_text(
                    f"No pattern with id={pid}.",
                )
                return
            await update.message.reply_text(
                f"✓ Pattern #{pid} unpinned."
            )
            return

        if sub == "restore" and len(args) >= 2 and args[1].isdigit():
            pid = int(args[1])
            ok = await asyncio.to_thread(self.kb.restore_pattern, pid)
            if not ok:
                await update.message.reply_text(
                    f"No pattern with id={pid}.",
                )
                return
            await update.message.reply_text(
                f"♻️ Pattern #{pid} restored to active state."
            )
            return

        if sub == "show" and len(args) >= 2 and args[1].isdigit():
            pid = int(args[1])
            entry = await asyncio.to_thread(self.kb.get_pattern, pid)
            if entry is None:
                await update.message.reply_text(
                    f"No pattern with id={pid}.",
                )
                return
            # Compute the per-component agreement breakdown so the
            # user can SEE why the score is what it is, not just a
            # single number. Pure Python, cheap.
            from core.plan_agreement import (
                _files_and_tools, _jaccard, _step_count_proximity,
                W_FILES, W_TOOLS, W_STEPS,
            )
            stored_recipe = entry.solution_pattern or ""
            shadow_recipe = entry.qwen_plan_recipe or ""
            cf, ct = _files_and_tools(stored_recipe)
            qf, qt = _files_and_tools(shadow_recipe)
            file_j = _jaccard(cf, qf) if (cf or qf) else 0.0
            tool_j = _jaccard(ct, qt) if (ct or qt) else 0.0
            step_p = (
                _step_count_proximity(len(ct), len(qt))
                if (ct or qt) else 0.0
            )
            blended = (
                W_FILES * file_j + W_TOOLS * tool_j + W_STEPS * step_p
            )
            # Format the readout. Trim long fields so a single
            # Telegram message fits (4096 char cap; we target ~2500).
            def _trim(s: str | None, n: int) -> str:
                if not s:
                    return "(none)"
                s = s.strip()
                return s if len(s) <= n else s[:n] + "..."
            stored_score = entry.qwen_plan_agreement
            stored_score_s = (
                f"{stored_score:.3f}" if stored_score is not None
                else "(no shadow data)"
            )
            solo_rate = (
                entry.solo_passes / entry.solo_attempts
                if entry.solo_attempts else 0.0
            )
            lines = [
                f"📋 Pattern #{pid}",
                f"category: {entry.category}",
                f"state: {entry.state}  pinned: {entry.pinned}",
                f"created: {entry.created_at[:19]}",
                f"origin: {entry.created_by_origin}",
                f"base_sha: {(entry.base_sha or '(none)')[:12]}",
                "",
                f"📝 problem: {_trim(entry.problem_summary, 200)}",
                "",
                f"🧠 Claude recipe ({len(stored_recipe)} chars):",
                f"{_trim(stored_recipe, 600)}",
                "",
                f"🤖 Qwen shadow recipe ({len(shadow_recipe)} chars):",
                f"{_trim(shadow_recipe, 600)}",
                "",
                f"📐 Agreement (stored): {stored_score_s}",
                f"   live recompute:    {blended:.3f}",
                f"   ├─ file Jaccard ({W_FILES}):  {file_j:.3f}  "
                f"({sorted(cf)[:3]} ∩ {sorted(qf)[:3]})",
                f"   ├─ tool Jaccard ({W_TOOLS}):  {tool_j:.3f}  "
                f"(claude={list(set(ct))[:5]} qwen={list(set(qt))[:5]})",
                f"   └─ step proximity ({W_STEPS}): {step_p:.3f}  "
                f"(claude={len(ct)} qwen={len(qt)})",
                "",
                f"🎓 Graduation: {entry.solo_passes}/"
                f"{entry.solo_attempts} solo "
                f"(rate {solo_rate:.0%})",
                f"   needs_reteach: {entry.needs_reteach}",
                f"   last_verified: "
                f"{(entry.last_verified_at or 'never')[:19]}",
                "",
                f"🏷  tags: {entry.tags or '(none)'}",
            ]
            await self._send_long(update, "\n".join(lines))
            return

        if sub == "planning":
            stats = await asyncio.to_thread(self.kb.planning_stats)
            shadow_n = stats["patterns_with_shadow"]
            tot = stats["patterns_total"]
            if shadow_n == 0:
                await update.message.reply_text(
                    f"📐 Shadow planning (Phase 15c):\n"
                    f"No shadow data yet -- 0 of {tot} patterns "
                    f"have a Qwen plan recorded. The shadow call "
                    f"runs once per /code attempt 1; data will "
                    f"accumulate as you teach."
                )
                return
            mean = stats["mean_agreement"] or 0.0
            lines = [
                "📐 Shadow planning (Phase 15c):",
                f"Patterns with shadow data: {shadow_n} of {tot}",
                f"Mean agreement: {mean:.3f}",
            ]
            # Phase 15d-bugfix: only show percentiles when we have
            # enough samples for them to be meaningful. With n<3 the
            # bucket-quantile formula collapses p25/p50/p75 to a
            # single value and the readout looks like fake precision.
            if shadow_n >= 3:
                p25 = stats["p25"] or 0.0
                p50 = stats["p50"] or 0.0
                p75 = stats["p75"] or 0.0
                lines.append(
                    f"Percentiles: p25={p25:.2f} p50={p50:.2f} "
                    f"p75={p75:.2f}"
                )
            else:
                lines.append(
                    f"Percentiles: (n={shadow_n} -- need ≥3 samples)"
                )
            arch = stats.get("by_archetype") or []
            if arch:
                lines.append("")
                lines.append("By tag (top 10):")
                for row in arch[:10]:
                    ag = row["mean_agreement"]
                    ag_s = f"{ag:.2f}" if ag is not None else "—"
                    tag_label = row["tag"] or "(no tag)"
                    lines.append(
                        f"  {tag_label[:24]:24s}  "
                        f"n={row['n']:<3d} mean={ag_s}"
                    )
            await update.message.reply_text("\n".join(lines))
            return

        await update.message.reply_text(
            "Usage: /kb  •  /kb verify <id>  •  /kb retake <id>  •  "
            "/kb stale [days]  •  /kb reteach  •  /kb pin <id>  •  "
            "/kb unpin <id>  •  /kb restore <id>  •  /kb planning  •  "
            "/kb show <id>"
        )

    # ---------- Phase 10: memory commands ----------

    @staticmethod
    def _parse_remember_arg(text: str) -> tuple[str, str] | None:
        """Parse '/remember key: value' or '/remember key value'.
        Returns (key, value) or None if unparseable.
        """
        if ":" in text:
            key, _, value = text.partition(":")
        else:
            parts = text.strip().split(maxsplit=1)
            if len(parts) < 2:
                return None
            key, value = parts[0], parts[1]
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if not key or not value:
            return None
        return key, value

    async def handle_remember(self, update: Update,
                              context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        text = " ".join(context.args) if context.args else ""
        if not text:
            await update.message.reply_text(
                "Usage: /remember <key>: <value>\n"
                "Example: /remember salary_minimum: 90000",
            )
            return
        parsed = self._parse_remember_arg(text)
        if parsed is None:
            await update.message.reply_text(
                "Couldn't parse. Use: /remember <key>: <value>",
            )
            return
        key, value = parsed
        mem = get_memory()
        await asyncio.to_thread(
            mem.store_fact, key, value, "user_explicit", 1.0,
        )
        await update.message.reply_text(
            f"Stored: {key} = {value} (confidence 1.0)",
        )

    async def handle_forget(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        text = " ".join(context.args) if context.args else ""
        if not text:
            await update.message.reply_text(
                "Usage: /forget <key>\n"
                "Example: /forget salary_minimum",
            )
            return
        key = text.strip().lower().replace(" ", "_")
        mem = get_memory()
        ok = await asyncio.to_thread(mem.delete_fact, key)
        await update.message.reply_text(
            f"Forgot: {key}" if ok else f"No fact with key {key!r}",
        )

    async def handle_recall(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        query = " ".join(context.args) if context.args else ""
        if not query:
            await update.message.reply_text(
                "Usage: /recall <query>\n"
                "Searches both facts (semantic) and episodes "
                "(per-agent activity).",
            )
            return
        mem = get_memory()
        facts = await asyncio.to_thread(mem.search_facts, query, 5)
        episodes = await asyncio.to_thread(mem.search_episodes, query, None, 5)
        lines: list[str] = []
        if facts:
            lines.append("Facts:")
            for f in facts:
                conf_tag = (
                    " (unconfirmed)" if f.confidence < 0.8 else ""
                )
                lines.append(
                    f"- {f.key}: {f.value}{conf_tag}",
                )
        if episodes:
            if lines:
                lines.append("")
            lines.append("Episodes:")
            for e in episodes:
                lines.append(
                    f"- [{e.scope}/{e.event_type}] {e.summary} "
                    f"({e.created_at[:10]})",
                )
        if not lines:
            lines = [f"Nothing found matching {query!r}."]
        await self._send_long(update, "\n".join(lines))

    async def handle_memory(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        mem = get_memory()
        stats = await asyncio.to_thread(mem.stats)
        ep_lines = "\n".join(
            f"  {s['scope']}: {s['count']}"
            for s in stats["episodic_by_scope"]
        ) or "  (none)"
        sm_lines = "\n".join(
            f"  {s['source']}: {s['count']}"
            for s in stats["semantic_by_source"]
        ) or "  (none)"
        await update.message.reply_text(
            f"Memory:\n"
            f"Episodic total: {stats['episodic_total']}\n"
            f"By scope:\n{ep_lines}\n"
            f"Semantic total: {stats['semantic_total']}\n"
            f"By source:\n{sm_lines}\n"
            f"Working: {WORKING_MEMORY.session_count()} active sessions",
        )

    async def handle_jobsearch_or_route(self, update, context, command):
        await self._route_and_wait(update, command)

    async def handle_research(self, update: Update,
                              context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        text = " ".join(context.args) if context.args else ""
        if not text:
            await update.message.reply_text(
                "Usage: /research <topic>\n"
                "Example: /research AI coding-agent benchmarks 2026",
            )
            return
        await self._route_and_wait(update, f"/research {text}")

    async def handle_curate(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        # /curate review -- show currently pending proposals (no propose).
        sub = (context.args[0].lower()
               if context.args else "")
        if sub == "review":
            await self._handle_curate_review(update)
            return
        progress = await update.message.reply_text(
            "🧠 Curating memory (last 24h)...",
        )
        trace_id = generate_trace_id()
        record = await self.curation.propose(trace_id)
        if record.get("_error"):
            try:
                await progress.edit_text(
                    f"❌ Curation failed: {record.get('error')[:300]}",
                )
            except Exception:
                pass
            return
        proposal = record["proposal"]
        token = record["token"]
        if proposal.get("no_changes"):
            # No-op: clear the pending entry so /curate review stays clean.
            self.curation.discard_pending(token)
        body = _format_curation_proposal(record)
        try:
            await progress.edit_text(body)
        except Exception:
            await self._send_long(update, body)

    async def handle_curate_approve(self, update: Update,
                                    context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        token = (
            (context.args[0] if context.args else "").strip().upper()
        )
        if not token:
            await update.message.reply_text(
                "Usage: /curate_approve <TOKEN>",
            )
            return
        result = await asyncio.to_thread(
            self.curation.apply, token, generate_trace_id(),
        )
        if result.get("_error"):
            await update.message.reply_text(
                f"❌ Approve failed: {result.get('error')}",
            )
            return
        bits = []
        if result.get("memory_md_changed"):
            bits.append("MEMORY.md updated")
        if result.get("user_md_changed"):
            bits.append("USER.md updated (curator notes appended)")
        if not bits:
            bits.append("(no file changes -- proposal was no-op)")
        await update.message.reply_text(
            f"✅ Applied [{token}]: {'; '.join(bits)}. "
            f"Brain persona reloaded.",
        )

    async def handle_curate_reject(self, update: Update,
                                   context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        token = (
            (context.args[0] if context.args else "").strip().upper()
        )
        if not token:
            await update.message.reply_text(
                "Usage: /curate_reject <TOKEN>",
            )
            return
        ok = self.curation.discard_pending(token)
        await update.message.reply_text(
            f"🗑️ Rejected [{token}]." if ok
            else f"No pending proposal for token {token!r}.",
        )

    async def _handle_curate_review(self, update: Update) -> None:
        """Re-display every pending curation proposal so the owner can
        audit before approving or rejecting. Pending proposals live in
        CurationFlow._pending and persist only for the bot's lifetime;
        scheduled /internal_curate runs auto-apply and never queue here."""
        tokens = self.curation.list_pending()
        if not tokens:
            await update.message.reply_text(
                "📋 No pending curation proposals.\n"
                "Run /curate to generate one (last 24h of episodes), or "
                "wait for the next scheduled run."
            )
            return
        for tok in tokens:
            record = self.curation.get_pending(tok)
            if record is None:
                continue
            body = _format_curation_proposal(record)
            await self._send_long(update, body)

    # Source paths /commit auto-stages. Excludes logs/, *.db, *.pyc.
    # Phase 16 prep (2026-05-06): added "tools" so migration / stress /
    # operations scripts (e.g. tools/preload_kb.py, tools/sanitize_kb_secrets.py,
    # tools/stress_test_runner.py) get versioned alongside the code that
    # uses them. workspace/stress_test/ remains intentionally OUT of scope
    # -- it's a transient test artifact directory, not source.
    _COMMIT_INCLUDE = (
        "core", "skills", "agents", "tests", "interfaces", "tools",
        "workspace/persona",
        "PHASES.md", "CLAUDE.md", "requirements.txt",
        "main.py", "Modelfile.brain",
    )

    async def _git_run(self, args: list[str]) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(config.PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return (
            proc.returncode or 0,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
        )

    async def handle_commit(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        msg = " ".join(context.args).strip() if context.args else ""
        if not msg:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
            msg = f"manual /commit at {ts}"
        # Stage curated paths (skip logs/, *.db, __pycache__/).
        rc, _, _ = await self._git_run([
            "add", "--", *self._COMMIT_INCLUDE,
        ])
        if rc != 0:
            # Some paths may not exist; that's OK -- git would have
            # warned but kept staging the rest. Continue.
            pass
        # Show what's about to be committed.
        rc, stat_out, _ = await self._git_run([
            "diff", "--cached", "--stat",
        ])
        if not stat_out.strip():
            await update.message.reply_text(
                "Nothing to commit (working tree clean for "
                "source paths -- logs/dbs/pyc are excluded).",
            )
            return
        # Actually commit. Use a fixed identity so the commits are
        # distinguishable from the user's own work on a real workstation.
        rc, _, err = await self._git_run([
            "-c", "user.email=sentinel@local",
            "-c", "user.name=Sentinel",
            "commit", "-q", "-m", msg,
        ])
        if rc != 0:
            log_event(
                generate_trace_id(), "WARNING", "telegram",
                f"/commit failed rc={rc} stderr={err[:200]!r}",
            )
            await update.message.reply_text(
                f"❌ commit failed (rc={rc}):\n```\n{err[:1500]}\n```",
            )
            return
        # Get the new HEAD SHA + short message for confirmation.
        rc2, sha, _ = await self._git_run([
            "rev-parse", "--short", "HEAD",
        ])
        sha = sha.strip() or "(unknown)"
        log_event(
            generate_trace_id(), "INFO", "telegram",
            f"/commit ok sha={sha} msg={msg[:80]!r}",
        )
        body = (
            f"✅ committed {sha}\n"
            f"_msg: {msg}_\n\n"
            f"```\n{stat_out.strip()[:3500]}\n```"
        )
        await self._send_long(update, body)

    async def handle_revert(self, update: Update,
                            context: ContextTypes.DEFAULT_TYPE) -> None:
        """Undo the last git commit, keeping changes staged (soft reset).

        Phase 17d: ``/revert chain`` undoes ALL contiguous chain
        commits at HEAD (commits with messages starting
        ``sentinel-chain:``). Useful after an auto-decompose chain
        ran and you want to roll back the whole batch in one shot.
        """
        if not await self._check_auth(update):
            return
        args = [a.strip() for a in (context.args or [])]
        if args and args[0].lower() == "chain":
            await self._handle_revert_chain(update)
            return
        # Show what is about to be reverted.
        rc, log_out, _ = await self._git_run(["log", "-1", "--oneline"])
        last_commit = log_out.strip() or "(no commits)"
        if not log_out.strip():
            await update.message.reply_text(
                "Nothing to revert — repository has no commits."
            )
            return
        progress = await update.message.reply_text(
            f"\u23ea Reverting: {last_commit}\nRunning git reset --soft HEAD~1 ..."
        )
        rc, out, err = await self._git_run(["reset", "--soft", "HEAD~1"])
        if rc != 0:
            log_event(
                generate_trace_id(), "WARNING", "telegram",
                f"/revert failed rc={rc} stderr={err[:200]!r}",
            )
            try:
                await progress.edit_text(
                    f"\u274c /revert failed (rc={rc}):\n```\n{err[:1500]}\n```"
                )
            except Exception:
                pass
            return
        rc2, new_head, _ = await self._git_run(["log", "-1", "--oneline"])
        new_head = new_head.strip() or "(empty repo)"
        log_event(
            generate_trace_id(), "INFO", "telegram",
            f"/revert ok reverted={last_commit!r} new_head={new_head!r}",
        )
        try:
            await progress.edit_text(
                f"\u2705 Reverted: {last_commit}\n"
                f"HEAD is now: {new_head}\n"
                f"Changes are staged \u2014 use /commit to re-apply."
            )
        except Exception:
            await update.message.reply_text(
                f"\u2705 Reverted: {last_commit}\nHEAD is now: {new_head}"
            )

    async def _handle_revert_chain(self, update: Update) -> None:
        """Phase 17d -- undo all contiguous chain commits at HEAD.

        Walks back from HEAD counting commits whose author email is
        ``chain-child@sentinel.local`` AND whose message starts with
        ``sentinel-chain:``. Stops at the first non-chain commit
        (does NOT cross a manual /commit). Hard-resets HEAD back N
        commits (so working tree is RESTORED to chain children's
        state -- i.e., chain edits are gone from disk too, not just
        unstaged like /revert).
        """
        # Count contiguous chain commits at HEAD.
        rc, log_out, _ = await self._git_run([
            "log", "--format=%H%x09%ae%x09%s",
            "-30",
        ])
        if rc != 0:
            await update.message.reply_text(
                f"\u274c /revert chain: git log failed ({rc})"
            )
            return
        chain_count = 0
        for line in log_out.splitlines():
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            _sha, email, subj = parts
            if (
                email == "chain-child@sentinel.local"
                and subj.startswith("sentinel-chain:")
            ):
                chain_count += 1
            else:
                break
        if chain_count == 0:
            await update.message.reply_text(
                "Nothing to /revert chain \u2014 HEAD is not a chain commit."
            )
            return
        progress = await update.message.reply_text(
            f"\u23ea /revert chain: undoing {chain_count} chain commit(s) "
            f"(git reset --hard HEAD~{chain_count}) ..."
        )
        rc, _, err = await self._git_run([
            "reset", "--hard", f"HEAD~{chain_count}",
        ])
        if rc != 0:
            log_event(
                generate_trace_id(), "WARNING", "telegram",
                f"/revert chain failed rc={rc} stderr={err[:200]!r}",
            )
            try:
                await progress.edit_text(
                    f"\u274c /revert chain failed (rc={rc}):\n"
                    f"```\n{err[:1500]}\n```"
                )
            except Exception:
                pass
            return
        rc2, new_head, _ = await self._git_run(["log", "-1", "--oneline"])
        new_head = new_head.strip() or "(empty)"
        log_event(
            generate_trace_id(), "INFO", "telegram",
            f"/revert chain ok undone={chain_count} new_head={new_head!r}",
        )
        try:
            await progress.edit_text(
                f"\u2705 Reverted {chain_count} chain commit(s).\n"
                f"HEAD is now: {new_head}\n\n"
                f"_Working tree HARD-reset (chain edits gone from disk)._"
            )
        except Exception:
            await update.message.reply_text(
                f"\u2705 Reverted {chain_count} chain commit(s). HEAD: {new_head}"
            )

    async def handle_kill(
        self, update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Phase 17a -- soft-kill the most-recently-claimed processing
        task (typically a /code in mid-retry). Sets `kill_requested=1`
        on the task; the agentic pipeline polls between attempts and
        bails cleanly. Does NOT interrupt mid-Claude-CLI subprocess --
        worst-case latency between /kill and bail = current attempt
        duration (typically 60-120s for Claude pre-teach).

        Idempotent. /kill on no-running-task replies "nothing to kill".
        """
        if not await self._check_auth(update):
            return
        target = await asyncio.to_thread(database.find_kill_target)
        if target is None:
            await update.message.reply_text(
                "🟢 Nothing to kill -- no task is currently processing."
            )
            return
        ok = await asyncio.to_thread(
            database.request_kill, target["task_id"],
        )
        if not ok:
            await update.message.reply_text(
                f"⚠️ Could not kill task `{target['task_id'][:12]}` -- "
                f"it may have just finished. Try /dashboard.",
                parse_mode="Markdown",
            )
            return
        log_event(
            "SEN-system", "WARNING", "telegram",
            f"/kill requested for task_id={target['task_id']} "
            f"command={target['command']}",
        )
        await update.message.reply_text(
            f"🛑 Kill requested for task `{target['task_id'][:12]}` "
            f"({target['command']}).\n\n"
            f"_The pipeline will bail at the next attempt boundary "
            f"(up to ~2 min). Working tree will be reset to a clean "
            f"state. No limitation will be stored._",
            parse_mode="Markdown",
        )

    async def handle_restart(self, update: Update,
                             context: ContextTypes.DEFAULT_TYPE) -> None:
        """Spawn a fresh detached `python main.py`, then trigger graceful
        shutdown of THIS process (worker drain, bot stop, locks released
        by main's shutdown path). The new process inherits no resources
        and recover_stale() on its startup reclaims any briefly-held DB
        locks.

        Phase 12.5 hardening: warns + briefly waits if a task is in-
        flight, to avoid orphaning a GPU lock that a freshly-spawned
        bot would only see released after STALE_LOCK_TIMEOUT (or via
        the new orphan-lock path in recover_stale)."""
        if not await self._check_auth(update):
            return
        # Drain check: if a task is processing, wait briefly for it
        # to complete BEFORE we trigger shutdown. The task naturally
        # releases its GPU lock on completion via execute_task's
        # finally block; if we shutdown mid-task the lock leaks.
        in_flight = await asyncio.to_thread(
            database.count_tasks_by_status, "processing",
        )
        if in_flight > 0:
            await update.message.reply_text(
                f"⏳ {in_flight} task(s) in flight; waiting up to "
                f"{config.RESTART_DRAIN_TIMEOUT_S}s for completion "
                f"before restart..."
            )
            log_event(
                "SEN-system", "INFO", "telegram",
                f"/restart waiting for {in_flight} in-flight task(s) "
                f"to complete (cap {config.RESTART_DRAIN_TIMEOUT_S}s)",
            )
            deadline = time.monotonic() + config.RESTART_DRAIN_TIMEOUT_S
            while time.monotonic() < deadline:
                await asyncio.sleep(2.0)
                in_flight = await asyncio.to_thread(
                    database.count_tasks_by_status, "processing",
                )
                if in_flight == 0:
                    break
            if in_flight > 0:
                await update.message.reply_text(
                    f"⚠️ {in_flight} task(s) still running after "
                    f"{config.RESTART_DRAIN_TIMEOUT_S}s; restarting "
                    f"anyway. Their GPU locks will be auto-released by "
                    f"recover_stale() on the next bot's startup."
                )
                log_event(
                    "SEN-system", "WARNING", "telegram",
                    f"/restart proceeding with {in_flight} in-flight; "
                    f"locks will be orphan-reclaimed on next start",
                )
            else:
                log_event(
                    "SEN-system", "INFO", "telegram",
                    "/restart drained successfully",
                )
        await update.message.reply_text(
            "🔄 Restarting Sentinel... back in ~5 s."
        )
        log_event("SEN-system", "INFO", "telegram",
                  "/restart requested by authorized user")
        import os
        import subprocess
        import sys
        creationflags = 0
        if sys.platform == "win32":
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
            )
        try:
            subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=str(config.PROJECT_ROOT),
                creationflags=creationflags,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            log_event("SEN-system", "ERROR", "telegram",
                      f"/restart spawn failed: {e}")
            await update.message.reply_text(
                f"❌ /restart failed to spawn replacement: {e}"
            )
            return

        async def _shutdown_after_reply_flushes() -> None:
            await asyncio.sleep(0.7)
            from core.worker import get_or_create_shutdown_event
            get_or_create_shutdown_event().set()

        asyncio.create_task(_shutdown_after_reply_flushes())

    async def handle_dashboard(self, update: Update,
                               context: ContextTypes.DEFAULT_TYPE) -> None:
        """Phase 11: chat-readable system snapshot. Mirrors GET /health."""
        if not await self._check_auth(update):
            return
        mon = getattr(self, "health_monitor", None)
        if mon is None:
            await update.message.reply_text(
                "(dashboard unavailable -- health monitor not wired yet)"
            )
            return
        from core.health import render_dashboard
        try:
            snap = await asyncio.to_thread(mon.snapshot)
            body = render_dashboard(snap)
        except Exception as e:
            log_event(generate_trace_id(), "ERROR", "telegram",
                      f"/dashboard failed: {type(e).__name__}: {e}")
            await update.message.reply_text(
                f"❌ /dashboard error: {type(e).__name__}: {e}"
            )
            return
        await self._send_long(update, body)

    # ─────────────────────────────────────────────────────────────────
    # Phase 13 -- /jobs application tracker viewer
    # ─────────────────────────────────────────────────────────────────

    _JOBS_DEFAULT_LIMIT = 20
    _JOBS_VALID_STATES = (
        "evaluated", "applied", "responded", "interview",
        "offer", "rejected", "discarded",
    )

    @staticmethod
    def _jobs_format_row(row: dict) -> str:
        """One-line summary of an application row. Designed to fit in
        a single Telegram line so /jobs lists are scannable."""
        title = (row.get("title") or "(no title)")[:50]
        company = (row.get("company") or "")[:25]
        score = row.get("score")
        score_s = f"{score:.1f}" if isinstance(score, (int, float)) else "?"
        rec = row.get("recommendation") or "-"
        rec_emoji = {
            "apply_now": "🎯",
            "worth_applying": "✅",
            "maybe": "🤔",
            "skip": "⏭️",
        }.get(rec, "•")
        state = row.get("state") or "?"
        return (
            f"[{row.get('id')}] {rec_emoji} {score_s} "
            f"{title} — {company} ({state})"
        )

    @staticmethod
    def _jobs_format_detail(row: dict) -> str:
        """Multi-line detail for a single application drill-in."""
        title = row.get("title") or "(no title)"
        company = row.get("company") or "(unknown company)"
        loc = row.get("location") or "(unknown location)"
        archetype = row.get("archetype") or "?"
        score = row.get("score")
        score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
        rec = row.get("recommendation") or "?"
        state = row.get("state") or "?"
        url = row.get("url") or ""
        first_seen = (row.get("first_seen_at") or "")[:10]
        last_seen = (row.get("last_seen_at") or "")[:10]
        applied_at = (row.get("applied_at") or "")[:10] or "(not applied)"
        # History is JSON in DB; parse defensively.
        history_raw = row.get("history") or "[]"
        try:
            history = json.loads(history_raw)
        except Exception:
            history = []
        if history:
            hist_lines = "\n".join(
                f"  {h.get('ts','')[:10]} {h.get('from','?')} → "
                f"{h.get('to','?')}"
                + (f" — {h.get('note')}" if h.get("note") else "")
                for h in history[-5:]
            )
        else:
            hist_lines = "  (no transitions yet)"
        return (
            f"📋 [{row.get('id')}] {title}\n"
            f"Company: {company}\n"
            f"Location: {loc}\n"
            f"Archetype: {archetype}\n"
            f"Score: {score_s} ({rec})\n"
            f"State: {state}\n"
            f"First seen: {first_seen}  •  Last seen: {last_seen}\n"
            f"Applied: {applied_at}\n"
            f"URL: {url}\n"
            f"History (last 5):\n{hist_lines}\n\n"
            f"Transition: /jobs {row.get('id')} <state>  "
            f"(states: applied, responded, interview, offer, rejected, "
            f"discarded)"
        )

    async def handle_jobs(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE) -> None:
        """Phase 13: browse the applications table from chat.

        Shapes:
          /jobs                       -- 20 most-recent (any state)
          /jobs <state>               -- filter by state
          /jobs <id>                  -- drill into one application
          /jobs <id> <state>          -- transition to <state>
          /jobs help                  -- usage
        """
        if not await self._check_auth(update):
            return
        args = context.args or []

        if args and args[0].lower() == "help":
            await update.message.reply_text(
                "/jobs — list 20 most recent\n"
                "/jobs <state> — filter (evaluated, applied, responded, "
                "interview, offer, rejected, discarded)\n"
                "/jobs <id> — drill into one application\n"
                "/jobs <id> <state> — transition state\n"
                "Examples:\n"
                "  /jobs applied\n"
                "  /jobs 42\n"
                "  /jobs 42 interview"
            )
            return

        # Disambiguate: numeric first arg → drill or transition.
        first = args[0] if args else ""
        if first.isdigit():
            app_id = int(first)
            if len(args) == 1:
                row = await asyncio.to_thread(
                    database.get_application, app_id,
                )
                if row is None:
                    await update.message.reply_text(
                        f"No application with id={app_id}.",
                    )
                    return
                await update.message.reply_text(
                    self._jobs_format_detail(row),
                )
                return
            # /jobs <id> <state> → transition
            target = args[1].lower()
            try:
                from core.database import _normalize_state
                target_norm = _normalize_state(target)
            except (ValueError, ImportError):
                await update.message.reply_text(
                    f"Unknown state {target!r}. Valid: "
                    f"{', '.join(self._JOBS_VALID_STATES)}",
                )
                return
            note = " ".join(args[2:]) if len(args) > 2 else None
            try:
                updated = await asyncio.to_thread(
                    database.transition_application,
                    app_id, target_norm, note,
                )
            except Exception as e:
                log_event(
                    "SEN-system", "WARNING", "telegram",
                    f"/jobs transition failed: {type(e).__name__}: {e}",
                )
                await update.message.reply_text(
                    f"Transition failed: {type(e).__name__}: {e}",
                )
                return
            if updated is None:
                await update.message.reply_text(
                    f"No application with id={app_id}.",
                )
                return
            await update.message.reply_text(
                f"✓ [{app_id}] → {target_norm}\n\n"
                + self._jobs_format_detail(updated),
            )
            return

        # String first arg (or no args): list mode.
        state_filter: str | None = None
        if first:
            try:
                from core.database import _normalize_state
                state_filter = _normalize_state(first)
            except (ValueError, ImportError):
                await update.message.reply_text(
                    f"Unknown state {first!r}. Valid: "
                    f"{', '.join(self._JOBS_VALID_STATES)}\n"
                    "Or pass an id (numeric) to drill in.",
                )
                return

        rows = await asyncio.to_thread(
            database.list_applications,
            state_filter, self._JOBS_DEFAULT_LIMIT,
        )
        if not rows:
            scope = f"state={state_filter}" if state_filter else "all states"
            await update.message.reply_text(
                f"No applications found ({scope}). "
                "Run /jobsearch first to populate.",
            )
            return

        header = (
            f"📋 Applications ({len(rows)} of "
            f"{'state=' + state_filter if state_filter else 'all states'}, "
            f"newest first):\n\n"
        )
        body = "\n".join(self._jobs_format_row(r) for r in rows)
        footer = (
            "\n\nDrill in: /jobs <id>  •  "
            "Transition: /jobs <id> <state>  •  "
            "Help: /jobs help"
        )
        msg = header + body + footer
        # Telegram cap is 4096; trim defensively if needed (each row is
        # ~100 chars × 20 rows ≈ 2000, so should fit easily).
        if len(msg) > 4000:
            msg = msg[:3950] + "\n…(truncated)"
        await update.message.reply_text(msg)

    async def handle_profile(self, update: Update,
                             context: ContextTypes.DEFAULT_TYPE) -> None:
        """Phase 12: candidate profile (workspace/persona/PROFILE.yml).

        Subcommands:
          /profile               -- show current values (alias for show)
          /profile init          -- copy PROFILE.example.yml -> PROFILE.yml
                                    (skips if PROFILE.yml exists)
          /profile show          -- print current parsed profile
          /profile set <key.path> <value>
                                 -- update a single scalar via dotted path,
                                    e.g. /profile set candidate.location "Detroit, MI"
          /profile edit          -- print path + raw text so you can edit on disk
        """
        if not await self._check_auth(update):
            return
        from core import job_profile
        sub = (context.args[0].lower()
               if context.args else "show")
        rest = list(context.args[1:]) if context.args else []
        try:
            if sub == "init":
                await self._profile_init(update)
            elif sub == "show":
                await self._profile_show(update)
            elif sub == "set":
                await self._profile_set(update, rest)
            elif sub == "edit":
                await self._profile_edit(update)
            else:
                await update.message.reply_text(_PROFILE_USAGE)
        except Exception as e:
            log_event(generate_trace_id(), "ERROR", "telegram",
                      f"/profile {sub} failed: {type(e).__name__}: {e}")
            await update.message.reply_text(
                f"❌ /profile {sub} error: {type(e).__name__}: {e}"
            )

    async def _profile_init(self, update: Update) -> None:
        from core import job_profile
        target = job_profile.profile_path()
        example = job_profile.example_path()
        if target.exists():
            await update.message.reply_text(
                f"⚠️ {target} already exists. Use /profile edit to modify on disk, "
                f"or delete it first if you want a clean reset."
            )
            return
        if not example.exists():
            await update.message.reply_text(
                f"❌ template missing: {example}. "
                f"PROFILE.example.yml should ship with the project."
            )
            return
        target.write_text(example.read_text(encoding="utf-8"),
                          encoding="utf-8")
        await update.message.reply_text(
            f"✅ Created {target}\n"
            f"Edit it on disk OR via /profile set <key.path> <value>.\n"
            f"Run /profile show to see what loaded."
        )

    async def _profile_show(self, update: Update) -> None:
        from core import job_profile
        prof = job_profile.load_profile(generate_trace_id())
        path = job_profile.profile_path()
        exists = "exists" if path.exists() else "does NOT exist (defaults shown)"
        archetypes = prof.target_roles.archetypes
        lines = [
            f"📇 PROFILE.yml ({exists}):",
            f"  path: {path}",
            f"  candidate.full_name: {prof.candidate.full_name!r}",
            f"  candidate.location: {prof.candidate.location!r}",
            f"  primary roles: {prof.target_roles.primary or '(none)'}",
            f"  archetypes ({len(archetypes)}):",
        ]
        for a in archetypes:
            lines.append(
                f"    - {a.name} [fit={a.fit}, level={a.level or '-'}]"
            )
        lines.extend([
            f"  title_filter.positive ({len(prof.title_filter.positive)}): "
            f"{prof.title_filter.positive[:8]}{'...' if len(prof.title_filter.positive) > 8 else ''}",
            f"  title_filter.negative ({len(prof.title_filter.negative)}): "
            f"{prof.title_filter.negative[:8]}{'...' if len(prof.title_filter.negative) > 8 else ''}",
            f"  avoid_companies: {prof.avoid_companies or '(none)'}",
            f"  workplace_preference: {prof.location.workplace_preference}",
            f"  primary_city: {prof.location.primary_city!r}",
            f"  comp.target: {prof.compensation.target_range_usd or '(unset)'}",
        ])
        await self._send_long(update, "\n".join(lines))

    async def _profile_set(self, update: Update, args: list[str]) -> None:
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: /profile set <dotted.path> <value>\n"
                "Example: /profile set candidate.location \"Detroit, MI\""
            )
            return
        from core import job_profile
        path = job_profile.profile_path()
        if not path.exists():
            await update.message.reply_text(
                "❌ PROFILE.yml does not exist. Run /profile init first."
            )
            return
        key_path = args[0]
        value = " ".join(args[1:]).strip().strip('"').strip("'")
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            await update.message.reply_text(
                "❌ PROFILE.yml top-level is not a mapping; refuse to edit."
            )
            return
        # Walk the dotted path, creating intermediate dicts as needed.
        parts = key_path.split(".")
        cur = raw
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        # Coerce a few common types from string for ergonomics.
        coerced: object = value
        if value.lower() in ("true", "false"):
            coerced = value.lower() == "true"
        elif value.lstrip("-").isdigit():
            coerced = int(value)
        cur[parts[-1]] = coerced
        path.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        # Re-load to validate -- if it doesn't parse the user's edit broke it.
        prof = job_profile.load_profile(generate_trace_id())
        await update.message.reply_text(
            f"✅ {key_path} = {coerced!r}\n"
            f"PROFILE.yml updated and re-validated. "
            f"({len(prof.target_roles.archetypes)} archetypes loaded)"
        )

    async def _profile_edit(self, update: Update) -> None:
        from core import job_profile
        path = job_profile.profile_path()
        if not path.exists():
            await update.message.reply_text(
                f"PROFILE.yml does not exist at {path}.\n"
                "Run /profile init to create it from the template."
            )
            return
        body = path.read_text(encoding="utf-8")
        await self._send_long(
            update,
            f"📇 {path}\n\n```yaml\n{body[:3500]}\n```\n"
            f"Edit this file on disk, or use /profile set <key> <value>.",
        )

    async def handle_schedule(self, update: Update,
                              context: ContextTypes.DEFAULT_TYPE) -> None:
        """Phase 11: scheduled-job management.

        Subcommands:
          /schedule add "<name>" --cron "<expr>" --command "<cmd>" [--hours HH-HH] [--isolated]
          /schedule add "<name>" --interval 30m --command "<cmd>"
          /schedule add "<name>" --once 2026-05-06T10:00:00Z --command "<cmd>" [--delete-after]
          /schedule list
          /schedule pause <id>
          /schedule resume <id>
          /schedule delete <id>
          /schedule runs <id>
        """
        if not await self._check_auth(update):
            return
        args = list(context.args) if context.args else []
        if not args:
            await update.message.reply_text(_SCHEDULE_USAGE)
            return
        sub = args[0].lower()
        rest = args[1:]
        try:
            if sub == "add":
                await self._schedule_add(update, rest)
            elif sub == "list":
                await self._schedule_list(update)
            elif sub == "pause":
                await self._schedule_set_enabled(update, rest, False)
            elif sub == "resume":
                await self._schedule_set_enabled(update, rest, True)
            elif sub == "delete":
                await self._schedule_delete(update, rest)
            elif sub == "runs":
                await self._schedule_runs(update, rest)
            else:
                await update.message.reply_text(
                    f"Unknown subcommand '{sub}'.\n\n{_SCHEDULE_USAGE}"
                )
        except _ScheduleArgError as e:
            await update.message.reply_text(f"❌ {e}\n\n{_SCHEDULE_USAGE}")
        except Exception as e:
            log_event(generate_trace_id(), "ERROR", "telegram",
                      f"/schedule failed: {type(e).__name__}: {e}")
            await update.message.reply_text(
                f"❌ /schedule error: {type(e).__name__}: {e}"
            )

    async def _schedule_add(self, update: Update, args: list[str]) -> None:
        from core import scheduler as _sched
        parsed = _parse_schedule_add(args)
        next_run = _sched.compute_next_run(
            parsed["schedule_type"], parsed["schedule_value"],
        )
        job_id = database.add_job(
            name=parsed["name"],
            schedule_type=parsed["schedule_type"],
            schedule_value=parsed["schedule_value"],
            command=parsed["command"],
            next_run_at=next_run.isoformat(),
            session_type=parsed["session_type"],
            active_hours_start=parsed["active_hours_start"],
            active_hours_end=parsed["active_hours_end"],
            delete_after_run=parsed["delete_after_run"],
        )
        local = next_run.astimezone(__import__("zoneinfo").ZoneInfo(
            config.SCHEDULER_TIMEZONE,
        ))
        await update.message.reply_text(
            f"✅ Job #{job_id} '{parsed['name']}' added.\n"
            f"Schedule: {parsed['schedule_type']} {parsed['schedule_value']}\n"
            f"Next run: {local.strftime('%Y-%m-%d %H:%M %Z')}"
        )

    async def _schedule_list(self, update: Update) -> None:
        jobs = database.list_jobs()
        if not jobs:
            await update.message.reply_text("(no scheduled jobs)")
            return
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(config.SCHEDULER_TIMEZONE)
        lines = ["Scheduled jobs:"]
        for j in jobs:
            on = "✅" if j["enabled"] else "⏸️"
            nxt = j["next_run_at"]
            if nxt:
                nxt_local = datetime.fromisoformat(nxt).astimezone(tz)
                nxt_disp = nxt_local.strftime("%Y-%m-%d %H:%M %Z")
            else:
                nxt_disp = "(none)"
            last = j["last_status"] or "-"
            lines.append(
                f"  {on} #{j['id']} {j['name']!r} "
                f"[{j['schedule_type']} {j['schedule_value']}] "
                f"next={nxt_disp} last={last}"
            )
        await self._send_long(update, "\n".join(lines))

    async def _schedule_set_enabled(
        self, update: Update, args: list[str], enabled: bool,
    ) -> None:
        if not args or not args[0].isdigit():
            raise _ScheduleArgError("missing job id (integer)")
        job_id = int(args[0])
        if database.get_job(job_id) is None:
            raise _ScheduleArgError(f"no job with id {job_id}")
        database.set_job_enabled(job_id, enabled)
        # When resuming, recompute next_run from now so it doesn't
        # immediately fire on every entry of a long-paused window.
        if enabled:
            from core import scheduler as _sched
            j = database.get_job(job_id)
            try:
                nxt = _sched.compute_next_run(
                    j["schedule_type"], j["schedule_value"],
                )
                database.set_next_run(job_id, nxt)
            except Exception:
                pass
        verb = "resumed" if enabled else "paused"
        await update.message.reply_text(f"✅ Job #{job_id} {verb}")

    async def _schedule_delete(
        self, update: Update, args: list[str],
    ) -> None:
        if not args or not args[0].isdigit():
            raise _ScheduleArgError("missing job id (integer)")
        job_id = int(args[0])
        if database.get_job(job_id) is None:
            raise _ScheduleArgError(f"no job with id {job_id}")
        database.delete_job(job_id)
        await update.message.reply_text(f"✅ Job #{job_id} deleted")

    async def _schedule_runs(
        self, update: Update, args: list[str],
    ) -> None:
        if not args or not args[0].isdigit():
            raise _ScheduleArgError("missing job id (integer)")
        job_id = int(args[0])
        if database.get_job(job_id) is None:
            raise _ScheduleArgError(f"no job with id {job_id}")
        runs = database.last_runs(job_id, limit=10)
        if not runs:
            await update.message.reply_text(
                f"(no runs recorded for job #{job_id})"
            )
            return
        lines = [f"Last {len(runs)} runs of job #{job_id}:"]
        for r in runs:
            mark = {"completed": "✅", "failed": "❌",
                    "skipped": "⏭️", "running": "🔄"}.get(r["status"], "?")
            ended = r["finished_at"] or "(running)"
            tail = (r.get("error") or r.get("result_summary") or "")[:80]
            lines.append(
                f"  {mark} {r['started_at']} -> {ended} | {tail}"
            )
        await self._send_long(update, "\n".join(lines))

    # ---------- free text ----------

    async def handle_message(self, update: Update,
                             context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update):
            return
        text = update.message.text or ""
        trace_id = generate_trace_id()

        # Phase 10: track in working memory + maybe auto-extract.
        session_id = str(update.effective_user.id)
        WORKING_MEMORY.add(session_id, "user", text)
        recent = WORKING_MEMORY.get_recent(session_id)
        if (
            len(recent) >= config.AUTO_EXTRACT_THRESHOLD
            and len(recent) % config.AUTO_EXTRACT_THRESHOLD == 0
        ):
            # Background -- never block the user reply on this.
            asyncio.create_task(
                self._maybe_auto_extract(recent, trace_id),
            )

        brain_result = await self.brain.process(text, trace_id, session_history=recent)

        if brain_result.intent == "chat":
            sent_reply = brain_result.response or "(no response)"
            await self._send_long(update, sent_reply)
            WORKING_MEMORY.add(session_id, "bot", sent_reply)
            try:
                await asyncio.to_thread(
                    get_memory().store_episode,
                    "global", trace_id, "chat",
                    f"user: {text[:120]} \u2192 bot: {sent_reply[:120]}",
                    f"user: {text}\nbot: {sent_reply}",
                    ["telegram", "chat"],
                )
            except Exception as _exc:
                log_event(trace_id, "DEBUG", "telegram",
                          f"episodic persist failed: {type(_exc).__name__}: {_exc}")
            return

        if brain_result.intent == "dispatch":
            progress_msg = await update.message.reply_text(
                f"{brain_result.summary or 'Working...'}\n{_build_bar(0)}"
            )
            cmd = brain_result.command or ""
            args = brain_result.args or ""
            command_string = f"{cmd} {args}".strip()
            rr = await asyncio.to_thread(route, command_string)
            if rr.status == "error":
                await update.message.reply_text(
                    f"Routing error: {rr.message}"
                )
                return
            result = await self._wait_for_task(
                rr.task_id, timeout=config.TELEGRAM_TASK_TIMEOUT,
                progress_message=progress_msg, trace_id=rr.trace_id,
            )
            if result and result["status"] == "completed":
                summary = await self.brain.summarize_result(
                    original_request=text,
                    raw_result=result["result"],
                    trace_id=trace_id,
                )
                await self._send_long(update, summary)
                WORKING_MEMORY.add(session_id, "bot", summary)
                try:
                    await asyncio.to_thread(
                        get_memory().store_episode,
                        "global", trace_id, "dispatch",
                        f"user: {text[:120]} \u2192 bot: {summary[:120]}",
                        f"user: {text}\nbot: {summary}",
                        ["telegram", "dispatch"],
                    )
                except Exception as _exc:
                    log_event(trace_id, "DEBUG", "telegram",
                              f"episodic persist failed: {type(_exc).__name__}: {_exc}")
            elif result and result["status"] == "failed":
                await update.message.reply_text(
                    f"Task failed: {result.get('error', 'Unknown')}"
                )
            else:
                await update.message.reply_text(
                    "Task timed out. /status for queue state."
                )
            return

        await update.message.reply_text(
            "Something went wrong. Try again or use a /command directly."
        )

    async def _maybe_auto_extract(
        self, messages: list[dict], trace_id: str,
    ) -> None:
        """Best-effort auto-extraction of durable facts from the
        recent working memory window. Runs in background so chat
        latency is unaffected. Failures are logged, never raised.

        Phase 15b: every fact written from this path lands in
        semantic_memory with created_by_origin='background_extraction'
        via the contextvar-based provenance system. Caller (asyncio
        task spawned in handle_message) gets its own ContextVar copy
        so the foreground reply path is unaffected."""
        from core.write_origin import (
            BACKGROUND_EXTRACTION, set_current_write_origin,
            reset_current_write_origin,
        )
        token = set_current_write_origin(BACKGROUND_EXTRACTION)
        try:
            mem = get_memory()
            n = await mem.extract_facts_from_conversation(
                messages, trace_id, brain=self.brain,
            )
            if n:
                log_event(
                    trace_id, "INFO", "telegram",
                    f"auto-extracted {n} fact(s) from session",
                )
        except Exception as e:
            log_event(
                trace_id, "WARNING", "telegram",
                f"auto-extract failed: {type(e).__name__}: {e}",
            )
        finally:
            reset_current_write_origin(token)
