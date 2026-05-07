"""LLM access layer.

`OllamaClient` is the low-level Ollama backend. `InferenceClient` is
the unified, model-registry-aware front-end that dispatches to either
Ollama or the local Claude CLI subprocess and handles the fallback
chain when a primary model fails.

Design rule (Phase 7+): Sentinel itself makes ZERO outbound network
calls. Ollama is local; Claude CLI is local subprocess.
"""
import asyncio
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel

from core import config
from core.logger import log_event


class LLMError(Exception):
    """Raised on any failure of the LLM call path: connection refused,
    timeout, HTTP error, or model-not-found."""


class OllamaClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        timeout: int = 5,
    ) -> tuple[int, dict]:
        """Phase 17g: single retry on transient Winsock/connection errors
        (WinError 10035 'would block', WinError 10054 reset). These happen
        when Ollama just loaded a model and the listening socket isn't
        fully ready, OR when Python urllib's socket layer races the OS.
        Live trigger: 2026-05-07 ~02:10Z /qcode hit WinError 10035 on the
        first Ollama call after /restart while the model swap was still
        settling. One retry with a brief sleep handles the transient.
        Permanent connection refused (Ollama not running) still raises
        on first attempt -- the retry loop only fires for *transient*
        connection errors.
        """
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        last_exc: Exception | None = None
        for attempt in range(2):  # 1 initial + 1 retry
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = resp.read().decode("utf-8")
                    return resp.status, json.loads(payload) if payload else {}
            except urllib.error.HTTPError as e:
                # HTTP-level errors aren't transient -- return immediately
                payload = e.read().decode("utf-8", errors="replace")
                try:
                    err_body = json.loads(payload)
                except json.JSONDecodeError:
                    err_body = {"raw": payload}
                return e.code, err_body
            except (urllib.error.URLError, socket.timeout, TimeoutError,
                    ConnectionError, OSError) as e:
                last_exc = e
                # Only retry on transient errors. "Connection refused"
                # is permanent (Ollama not running) -- don't waste time.
                msg = str(e).lower()
                is_transient = (
                    "10035" in msg  # WinError 10035 -- would block
                    or "10054" in msg  # WinError 10054 -- conn reset
                    or "non-blocking" in msg
                    or "temporarily unavailable" in msg
                    or "broken pipe" in msg
                )
                is_permanent = (
                    "refused" in msg
                    or "actively refused" in msg
                )
                if attempt == 0 and is_transient and not is_permanent:
                    # Brief backoff -- let the socket / model settle.
                    time.sleep(0.5)
                    continue
                raise LLMError(self._friendly_network_error(e)) from e
        # Defensive fallback (loop should always return or raise above).
        raise LLMError(self._friendly_network_error(  # type: ignore[arg-type]
            last_exc or RuntimeError("unreachable"),  # type: ignore[arg-type]
        ))

    def _friendly_network_error(self, e: Exception) -> str:
        msg = str(e)
        low = msg.lower()
        if "refused" in low or "actively refused" in low:
            return "Ollama is not running (connection refused)"
        if "timed out" in low or isinstance(e, (socket.timeout, TimeoutError)):
            return "Network timeout reaching Ollama"
        if "name or service not known" in low or "getaddrinfo" in low:
            return f"Cannot resolve Ollama host: {self.base_url}"
        return f"Network error reaching Ollama: {msg}"

    def health_check(self) -> bool:
        try:
            status, _ = self._request("GET", "/api/tags", timeout=3)
            return status == 200
        except LLMError:
            return False

    def is_model_loaded(self, model: str) -> bool:
        try:
            status, body = self._request("GET", "/api/ps", timeout=3)
        except LLMError:
            return False
        if status != 200:
            return False
        for m in body.get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return True
        return False

    def list_loaded_models(self) -> list[str]:
        """Return Ollama's currently-loaded model_ids (typically 0 or 1
        with OLLAMA_MAX_LOADED_MODELS=1)."""
        try:
            status, body = self._request("GET", "/api/ps", timeout=3)
        except LLMError:
            return []
        if status != 200:
            return []
        return [
            m.get("name") or m.get("model")
            for m in body.get("models", [])
            if m.get("name") or m.get("model")
        ]

    def unload_model(self, model: str, trace_id: str = "SEN-system") -> None:
        """Send keep_alive=0 to actively free VRAM. Called after GPU
        tasks so the brain can load without waiting for the timeout."""
        body: dict[str, Any] = {
            "model": model, "prompt": "", "stream": False,
            "keep_alive": 0,
        }
        try:
            self._request("POST", "/api/generate", body=body, timeout=10)
            log_event(
                trace_id, "INFO", "llm",
                f"unload requested for model={model} (keep_alive=0)",
            )
        except LLMError as e:
            log_event(
                trace_id, "WARNING", "llm",
                f"unload failed for model={model}: {e}",
            )

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        format_json: bool = False,
        trace_id: str = "SEN-system",
        num_predict: int | None = None,
    ) -> str:
        if temperature is None:
            temperature = config.LLM_TEMPERATURE
        if timeout is None:
            timeout = config.LLM_TIMEOUT

        opts: dict[str, Any] = {"temperature": temperature}
        if num_predict is not None:
            opts["num_predict"] = num_predict
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": config.MODEL_KEEP_ALIVE,
            "options": opts,
        }
        if system is not None:
            body["system"] = system
        if format_json:
            body["format"] = "json"

        try:
            status, payload = self._request(
                "POST", "/api/generate", body=body, timeout=timeout,
            )
        except LLMError as e:
            if "timeout" in str(e).lower():
                msg = f"Model {model} timed out after {timeout}s"
            else:
                msg = str(e)
            log_event(trace_id, "ERROR", "llm",
                      f"generate failed model={model} error={msg}")
            raise LLMError(msg) from e

        if status == 404 or (
            status >= 400 and "not found" in
            json.dumps(payload).lower()
        ):
            msg = (f"Model {model} not available. "
                   f"Run: ollama pull {model}")
            log_event(trace_id, "ERROR", "llm",
                      f"generate failed model={model} error={msg}")
            raise LLMError(msg)

        if status != 200:
            msg = (f"Ollama returned HTTP {status} for model={model}: "
                   f"{payload}")
            log_event(trace_id, "ERROR", "llm",
                      f"generate failed model={model} error={msg}")
            raise LLMError(msg)

        response_text = payload.get("response", "")
        if not isinstance(response_text, str):
            raise LLMError(
                f"Ollama returned non-string response: {response_text!r}"
            )
        log_event(trace_id, "INFO", "llm",
                  f"generate ok model={model} chars={len(response_text)}")
        return response_text


# ============================================================
# Phase 8 -- unified InferenceClient on top of Ollama + Claude CLI
# ============================================================

class InferenceResult(BaseModel):
    text: str
    model_used: str
    backend: str
    complexity_score: float | None = None
    complexity_tier: str | None = None
    inference_time: float
    fallback_used: bool = False
    fallback_reason: str | None = None


CAPABILITY_RANK = {"basic": 0, "standard": 1, "advanced": 2}


class InferenceClient:
    """Unified inference client. Resolves model name -> backend -> call.

    `generate()` is for explicit model selection (or default).
    `generate_with_complexity()` runs the heuristic complexity classifier
    and fallback chain: try recommended -> escalate tier -> try Claude CLI.
    """

    def __init__(self, model_registry=None) -> None:
        from core.claude_cli import ClaudeCliClient
        from core.model_registry import MODEL_REGISTRY
        self.model_registry = model_registry or MODEL_REGISTRY
        self.ollama = OllamaClient()
        self.claude_cli = ClaudeCliClient()

    def get_loaded_model(self) -> str | None:
        """Return the model_id currently held in Ollama VRAM, or None."""
        loaded = self.ollama.list_loaded_models()
        return loaded[0] if loaded else None

    def unload_ollama_model(self, model_id: str) -> None:
        """Free VRAM by asking Ollama to drop the model immediately."""
        self.ollama.unload_model(model_id)

    async def _call_one(
        self, model_cfg, prompt: str, system: str | None,
        temperature: float | None, max_tokens: int | None,
        trace_id: str,
    ) -> tuple[str, float]:
        """Call exactly one backend. Returns (text, elapsed_seconds).
        Raises on backend failure (LLMError, ClaudeCliError, etc.)."""
        from core.claude_cli import ClaudeCliError
        start = time.monotonic()
        if model_cfg.backend == "ollama":
            text = await asyncio.to_thread(
                self.ollama.generate,
                model_cfg.model_id, prompt, system,
                temperature if temperature is not None
                else model_cfg.default_temperature,
                None,  # use default timeout
                False,  # format_json: caller can wrap
                trace_id,
            )
        elif model_cfg.backend == "claude_cli":
            text = await self.claude_cli.generate(
                prompt=prompt, system=system, trace_id=trace_id,
            )
        else:
            raise LLMError(f"unknown backend: {model_cfg.backend!r}")
        return text, time.monotonic() - start

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace_id: str = "SEN-system",
    ) -> InferenceResult:
        """Single-model generate. `model` is a registry NAME (e.g.,
        'qwen-3b'), not the backend's model_id. Falls back to the
        cheapest standard-tier model if `model` is None."""
        if model is None:
            cfg = self.model_registry.get_cheapest_capable("standard")
            if cfg is None:
                raise LLMError(
                    "no available model meets the 'standard' tier "
                    "threshold; try `ollama pull qwen2.5:3b`"
                )
        else:
            cfg = self.model_registry.get(model)
            if cfg is None:
                raise LLMError(f"unknown model name: {model!r}")
        text, elapsed = await self._call_one(
            cfg, prompt, system, temperature, max_tokens, trace_id,
        )
        return InferenceResult(
            text=text,
            model_used=cfg.name,
            backend=cfg.backend,
            inference_time=elapsed,
        )

    def _build_fallback_chain(
        self, recommended: str,
    ) -> list:
        """Recommended model first, then any other available models at
        equal-or-higher tier (capped at MAX_FALLBACK_ATTEMPTS)."""
        rec = self.model_registry.get(recommended)
        chain = []
        if rec is not None and rec.available:
            chain.append(rec)
        start_rank = (
            CAPABILITY_RANK.get(rec.capability_tier, 1) if rec else 1
        )
        for tier in ("basic", "standard", "advanced"):
            if CAPABILITY_RANK[tier] < start_rank:
                continue
            for m in self.model_registry.get_by_tier(tier):
                if m.available and m not in chain:
                    chain.append(m)
        return chain[: max(1, config.MAX_FALLBACK_ATTEMPTS)]

    async def generate_with_complexity(
        self,
        prompt: str,
        command: str,
        args: dict,
        skill_name: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        trace_id: str = "SEN-system",
    ) -> InferenceResult:
        """Auto-select model based on complexity. On failure, escalate
        through the fallback chain."""
        from core.complexity import classify_complexity
        cls = classify_complexity(command, args, skill_name)
        chain = self._build_fallback_chain(cls.recommended_model)
        if not chain:
            raise LLMError(
                "no models available for the requested tier; check "
                "`ollama list` and `claude --version`"
            )
        log_event(
            trace_id, "INFO", "inference",
            f"auto-route command={command} score={cls.score:.2f} "
            f"tier={cls.tier} chain="
            f"{[m.name for m in chain]}",
        )
        last_reason: str | None = None
        for i, cfg in enumerate(chain):
            try:
                text, elapsed = await self._call_one(
                    cfg, prompt, system, temperature, None, trace_id,
                )
                return InferenceResult(
                    text=text,
                    model_used=cfg.name,
                    backend=cfg.backend,
                    complexity_score=cls.score,
                    complexity_tier=cls.tier,
                    inference_time=elapsed,
                    fallback_used=(i > 0),
                    fallback_reason=last_reason if i > 0 else None,
                )
            except Exception as e:
                last_reason = f"{type(e).__name__}: {str(e)[:200]}"
                log_event(
                    trace_id, "WARNING", "inference",
                    f"model {cfg.name} failed ({last_reason}); "
                    f"trying next in chain",
                )
                continue
        raise LLMError(
            f"all {len(chain)} models in fallback chain failed; "
            f"last_reason={last_reason}"
        )


# Module-level singleton for convenience.
INFERENCE_CLIENT = InferenceClient()
