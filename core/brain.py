"""BrainRouter: Qwen 3 1.7B intent classifier + result summarizer.

The brain receives free-form user messages from the Telegram interface
and either:
  - dispatches to an agent/command (via the existing router/queue),
  - replies conversationally for chat-style messages.

It also formats raw task results back into human-readable text for
the user.

Design rule: ZERO outbound network calls. Brain runs locally via
Ollama; result summary uses the same model.
"""
import json
from typing import Any

from pydantic import BaseModel

from core import config
from core.llm import INFERENCE_CLIENT
from core.logger import log_event


class BrainResult(BaseModel):
    intent: str                      # "dispatch" | "chat" | "error"
    agent: str | None = None
    command: str | None = None       # The /command to route through
    args: str | None = None
    summary: str | None = None
    response: str | None = None      # Direct chat response
    error: str | None = None
    trace_id: str = ""


SYSTEM_PROMPT = """You are Sentinel, a personal AI assistant running on the user's local PC.
You manage specialized agents that perform tasks. Your job is to understand what the user wants
and either answer directly or dispatch to the right agent.

Available agents and commands:
- job_analyst (command: /extract): extract structured data from job postings
- code_assistant (command: /code): write, fix, or explain code
- web_search (command: /search): search the internet for information

Rules:
- If the user wants a TASK done (search, extract, code, etc), respond with ONLY this JSON:
  {"intent": "dispatch", "agent": "<agent_name>", "command": "<the /command to use>", "args": "<task description>", "summary": "<one sentence explaining what you will do>"}
- If the user is making casual conversation (greeting, math question, chitchat), respond with ONLY this JSON:
  {"intent": "chat", "response": "<your conversational reply>"}
- NEVER include markdown fences, explanation, or preamble. ONLY the JSON object.
/no_think"""

SUMMARIZE_SYSTEM = (
    "You are Sentinel, summarizing task results for the user. "
    "Be concise and conversational. Do not include raw JSON. /no_think"
)


def _strip_think_block(text: str) -> str:
    """Qwen3 sometimes emits a <think>...</think> reasoning block even
    in /no_think mode. Strip it before parsing."""
    if not text:
        return text
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start) + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    text = _strip_think_block(text)
    # Markdown fences (the model may add them despite instructions)
    import re as _re
    fenced = _re.search(
        r"```(?:json)?\s*(.+?)\s*```", text,
        flags=_re.DOTALL | _re.IGNORECASE,
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


class BrainRouter:
    def __init__(self, inference_client=None) -> None:
        self.inference = inference_client or INFERENCE_CLIENT
        self.model = config.BRAIN_MODEL
        # Phase 10: persona files always loaded into context.
        self.persona_files: dict[str, str] = {}
        self.persona_context: str = ""
        self._load_persona_files()

    # ---------- persona ----------

    def _load_persona_files(self) -> None:
        """Read each protected persona file from PERSONA_DIR, cap per
        config.PERSONA_INJECT_MAX_CHARS, and join into a single
        ``self.persona_context`` block. Missing files are skipped
        silently so first-run before seed-write doesn't crash the
        bot.
        """
        files: dict[str, str] = {}
        sections: list[str] = []
        # Stable order for deterministic context.
        order = ["IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md"]
        for name in order:
            path = config.PERSONA_DIR / name
            if not path.exists():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as e:
                log_event(
                    "SEN-system", "WARNING", "brain",
                    f"persona load failed for {name}: "
                    f"{type(e).__name__}: {e}",
                )
                continue
            cap = config.PERSONA_INJECT_MAX_CHARS.get(name, 2000)
            if len(content) > cap:
                content = content[:cap] + "\n... (truncated)"
            files[name] = content
            sections.append(f"=== {name} ===\n{content}")
        self.persona_files = files
        self.persona_context = "\n\n".join(sections)
        log_event(
            "SEN-system", "INFO", "brain",
            f"persona loaded: {sorted(files.keys())} "
            f"total_chars={len(self.persona_context)}",
        )

    def reload_persona(self) -> None:
        """Re-read persona files from disk. Called by the curation
        flow after authorize_update writes new MEMORY.md/USER.md.
        """
        self._load_persona_files()

    def _memory_context(self, query: str, trace_id: str) -> str:
        """Best-effort fetch of episodic context. Returns empty string
        on any failure -- memory is a nice-to-have for the brain, not
        a hard dependency, and we never want a memory hiccup to take
        down a chat.
        """
        try:
            from core.memory import get_memory  # noqa: PLC0415
            mem = get_memory()
            return mem.get_agent_context(
                "global", query, max_chars=1000,
            )
        except Exception as e:
            log_event(
                trace_id, "DEBUG", "brain",
                f"memory context unavailable: "
                f"{type(e).__name__}: {e}",
            )
            return ""

    def _build_system_prompt(self) -> str:
        if not self.persona_context:
            return SYSTEM_PROMPT
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== PERSONA CONTEXT (always-loaded) ===\n"
            f"{self.persona_context}"
        )

    async def process(
        self, user_message: str, trace_id: str,
        session_history: list[dict] | None = None,
    ) -> BrainResult:
        log_event(
            trace_id, "INFO", "brain",
            f"classifying user_message[:80]={user_message[:80]!r}",
        )
        mem_ctx = self._memory_context(user_message, trace_id)
        # In-process working memory: prior turns this session (resets on restart).
        history_lines: list[str] = []
        if session_history:
            for msg in session_history[:-1]:  # exclude the just-added user msg
                role = msg.get("role", "")
                content = msg.get("message", "")[:200]
                if role and content:
                    history_lines.append(f"{role}: {content}")
        history_ctx = "\n".join(history_lines[-10:]) if history_lines else ""
        prompt_parts: list[str] = []
        if history_ctx:
            prompt_parts.append(f"Conversation so far:\n{history_ctx}")
        if mem_ctx:
            prompt_parts.append(f"Recent activity:\n{mem_ctx}")
        prompt_parts.append(f"User: {user_message}")
        prompt = "\n\n".join(prompt_parts)
        system = self._build_system_prompt()
        try:
            raw = await self.inference.generate(
                prompt=prompt,
                model=self.model,
                system=system,
                temperature=config.BRAIN_TEMPERATURE,
                trace_id=trace_id,
            )
        except Exception as e:
            log_event(
                trace_id, "ERROR", "brain",
                f"brain inference failed: {type(e).__name__}: {e}",
            )
            return BrainResult(
                intent="error",
                error=f"brain inference failed: {e}",
                trace_id=trace_id,
            )

        data = _extract_json_object(raw.text)
        if data is None:
            # Retry once with explicit correction prompt
            log_event(
                trace_id, "WARNING", "brain",
                f"brain returned non-JSON; retrying. "
                f"first_attempt[:200]={raw.text[:200]!r}",
            )
            try:
                retry = await self.inference.generate(
                    prompt=(
                        f"Your previous response was not valid JSON. "
                        f"The user said: {user_message!r}. "
                        f"Respond with ONLY a JSON object as instructed."
                    ),
                    model=self.model,
                    system=system,
                    temperature=config.BRAIN_TEMPERATURE,
                    trace_id=trace_id,
                )
                data = _extract_json_object(retry.text)
            except Exception as e:
                log_event(
                    trace_id, "WARNING", "brain",
                    f"brain retry failed: {type(e).__name__}: {e}",
                )

        if data is None:
            # Fall back to chat with the raw text -- never crash on parse
            return BrainResult(
                intent="chat",
                response=_strip_think_block(raw.text)[:2000]
                    or "I'm not sure how to handle that.",
                trace_id=trace_id,
            )

        intent = (data.get("intent") or "").lower()
        if intent == "dispatch":
            command = data.get("command") or ""
            if not command.startswith("/"):
                command = "/" + command if command else ""
            return BrainResult(
                intent="dispatch",
                agent=data.get("agent"),
                command=command,
                args=str(data.get("args", "")),
                summary=data.get("summary", "Working on it..."),
                trace_id=trace_id,
            )
        return BrainResult(
            intent="chat",
            response=str(data.get("response")
                         or _strip_think_block(raw.text)[:2000]
                         or "(no response)"),
            trace_id=trace_id,
        )

    async def summarize_result(
        self, original_request: str, raw_result: str | dict,
        trace_id: str,
    ) -> str:
        if isinstance(raw_result, dict):
            raw_result = json.dumps(raw_result, indent=2)
        prompt = (
            f"The user asked: {original_request!r}\n\n"
            f"Raw result:\n{raw_result[:3000]}\n\n"
            "Answer the user's question DIRECTLY using specific facts "
            "from the raw result. Quote numbers, dates, names, prices, "
            "temperatures, and percentages VERBATIM from the snippets "
            "-- do not paraphrase facts. If a snippet has the answer, "
            "use that snippet's exact wording for the data. Mention "
            "each source by placing its full URL in parentheses after the claim it supports, e.g. (https://example.com/page). "
            "If the result is a list of links and no snippet answers "
            "the question directly, say 'I found these sources but no "
            "direct answer:' followed by 2-3 best links. "
            "3-6 short sentences. No JSON, no preamble, no signoff. "
            "/no_think"
        )
        try:
            result = await self.inference.generate(
                prompt=prompt,
                model=self.model,
                system=SUMMARIZE_SYSTEM,
                temperature=config.BRAIN_SUMMARIZE_TEMPERATURE,
                trace_id=trace_id,
            )
        except Exception as e:
            log_event(
                trace_id, "WARNING", "brain",
                f"summarize failed: {type(e).__name__}: {e}",
            )
            return (
                f"Task completed but I couldn't summarize. "
                f"Raw result:\n{str(raw_result)[:2000]}"
            )
        return _strip_think_block(result.text)[:1200]


BRAIN = BrainRouter()
