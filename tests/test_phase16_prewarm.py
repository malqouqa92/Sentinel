"""Phase 16 prewarm fix -- ECC tests for cold-load mitigation.

User-observed symptom (SEN-2dea7bea, 2026-05-06): a /code submitted
17s after bot startup ran for 3 min before manual kill, vs the
documented average of 60-120s. Cold-load on stepfed's first Qwen
step compounded the slowness.

Fix: ``main._prewarm_qwen_coder`` fires a 1-token generate at
startup so Ollama loads weights into VRAM with the configured
keep_alive timer. By the time the user sends /code, qwen-coder is
already resident.

Coverage:
  Function behavior:
    P01 -- _prewarm_qwen_coder is async + accepts trace_id
    P02 -- on success, calls OllamaClient.generate with the coder model
    P03 -- on success, logs an INFO marker that's grep-able from logs
    P04 -- catches LLMError (Ollama unreachable) and continues
    P05 -- catches generic Exception (defensive) and continues
    P06 -- never raises (regardless of failure mode)
    P07 -- prompt is short ("ok") so the warmup itself is cheap
    P08 -- temperature is 0.0 so no sampling overhead
    P09 -- timeout is bounded (~30s) so a hung Ollama doesn't
            block startup forever

  Source-level (main.py wiring):
    P20 -- main.py imports OllamaClient + LLMError
    P21 -- main.py defines _prewarm_qwen_coder
    P22 -- _prewarm_qwen_coder is called via create_task at startup
    P23 -- prewarm_task is cancelled in the shutdown finally block
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

# Import target. main.py is at the project root.
import main as sentinel_main  # noqa: E402
from core.llm import LLMError


# ─────────────────────────────────────────────────────────────────────
# Function-behavior tests
# ─────────────────────────────────────────────────────────────────────


def test_p01_prewarm_is_async_and_takes_trace_id():
    """The prewarm function is async (so it can be `create_task`'d)
    and takes a single positional ``trace_id`` argument."""
    fn = sentinel_main._prewarm_qwen_coder
    assert inspect.iscoroutinefunction(fn), "must be async"
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    assert params == ["trace_id"], f"expected [trace_id], got {params}"


def test_p02_calls_ollama_generate_with_coder_model():
    """On success path, OllamaClient.generate is called with the
    qwen2.5-coder:3b model. This is the assertion that proves
    "the right model gets loaded into VRAM"."""
    captured = {}

    def fake_generate(self, model, prompt, system=None, temperature=None,
                      timeout=None, format_json=False, trace_id="SEN-system"):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["temperature"] = temperature
        captured["timeout"] = timeout
        captured["trace_id"] = trace_id
        return "(ok)"

    with patch("core.llm.OllamaClient.generate", fake_generate):
        asyncio.run(sentinel_main._prewarm_qwen_coder("SEN-test-p02"))

    assert captured["model"] == "qwen2.5-coder:3b"
    assert captured["trace_id"] == "SEN-test-p02"


def test_p03_logs_success_marker(monkeypatch):
    """On success, an INFO log line is emitted that contains 'qwen-coder
    prewarm ok' so log scans / Monitor filters can identify the event.
    """
    events = []

    def fake_log(trace_id, level, component, message):
        events.append((level, component, message))

    def fake_generate(self, model, prompt, *a, **kw):
        return "(ok)"

    with patch("main.log_event", fake_log), \
         patch("core.llm.OllamaClient.generate", fake_generate):
        asyncio.run(sentinel_main._prewarm_qwen_coder("SEN-test-p03"))

    assert any(
        "qwen-coder prewarm ok" in m
        for (_, _, m) in events
    ), f"expected success marker in events: {events}"


def test_p04_catches_llmerror():
    """When Ollama is unreachable (LLMError), prewarm catches and
    logs at INFO -- never raises into the startup sequence."""
    def raises_llmerror(self, *a, **kw):
        raise LLMError("Ollama is not running (connection refused)")

    with patch("core.llm.OllamaClient.generate", raises_llmerror):
        # Must NOT raise.
        asyncio.run(sentinel_main._prewarm_qwen_coder("SEN-test-p04"))


def test_p05_catches_generic_exception():
    """Defense in depth: any other exception (RuntimeError, etc.) is
    also swallowed so a buggy LLM client can't crash startup."""
    def raises_runtime(self, *a, **kw):
        raise RuntimeError("unexpected")

    with patch("core.llm.OllamaClient.generate", raises_runtime):
        asyncio.run(sentinel_main._prewarm_qwen_coder("SEN-test-p05"))


def test_p06_never_raises_regardless_of_inner_error(monkeypatch):
    """Bulk smoke: every failure mode we can construct is silently
    handled. If this test ever fails, startup will be brittle to
    Ollama state."""
    failure_modes = [
        LLMError("connection refused"),
        LLMError("timeout"),
        TimeoutError("hung"),
        RuntimeError("?"),
        ValueError("bad arg"),
        OSError("disk full"),
    ]
    for err in failure_modes:
        def boom(self, *a, _e=err, **kw):
            raise _e
        with patch("core.llm.OllamaClient.generate", boom):
            asyncio.run(sentinel_main._prewarm_qwen_coder("SEN-test-p06"))


def test_p07_prompt_is_short():
    """The prewarm sends a 1-2 token prompt. Sending a long prompt
    would still load the model but waste prefill time. Anything
    longer than 16 chars suggests the warmup got copy-pasted from
    a real prompt."""
    captured = {}

    def fake_generate(self, model, prompt, *a, **kw):
        captured["prompt"] = prompt
        return "(ok)"

    with patch("core.llm.OllamaClient.generate", fake_generate):
        asyncio.run(sentinel_main._prewarm_qwen_coder("SEN-test-p07"))

    assert len(captured["prompt"]) <= 16, (
        f"prewarm prompt should be short, got "
        f"{len(captured['prompt'])} chars: {captured['prompt']!r}"
    )


def test_p08_temperature_zero():
    """Greedy sampling for deterministic, fastest possible warmup."""
    captured = {}

    def fake_generate(self, model, prompt, system=None, temperature=None,
                      timeout=None, format_json=False, trace_id="SEN-system"):
        captured["temperature"] = temperature
        return "(ok)"

    with patch("core.llm.OllamaClient.generate", fake_generate):
        asyncio.run(sentinel_main._prewarm_qwen_coder("SEN-test-p08"))

    assert captured["temperature"] == 0.0


def test_p09_timeout_bounded():
    """Timeout is a small finite number so a hung Ollama doesn't
    block startup forever. The exact value is a tunable; this test
    asserts the bound is tight."""
    captured = {}

    def fake_generate(self, model, prompt, system=None, temperature=None,
                      timeout=None, format_json=False, trace_id="SEN-system"):
        captured["timeout"] = timeout
        return "(ok)"

    with patch("core.llm.OllamaClient.generate", fake_generate):
        asyncio.run(sentinel_main._prewarm_qwen_coder("SEN-test-p09"))

    assert captured["timeout"] is not None
    assert captured["timeout"] <= 60, (
        f"prewarm timeout should be <= 60s, got {captured['timeout']}"
    )


# ─────────────────────────────────────────────────────────────────────
# Source-level wiring tests (don't actually run main())
# ─────────────────────────────────────────────────────────────────────


_MAIN_SRC = (Path(__file__).resolve().parent.parent / "main.py").read_text(
    encoding="utf-8"
)


def test_p20_main_imports_ollamaclient_and_llmerror():
    """The prewarm function references OllamaClient + LLMError; main.py
    must import both at module top so the function body resolves."""
    assert "OllamaClient" in _MAIN_SRC
    assert "LLMError" in _MAIN_SRC


def test_p21_main_defines_prewarm_function():
    assert "async def _prewarm_qwen_coder(" in _MAIN_SRC


def test_p22_prewarm_called_via_create_task_at_startup():
    """The prewarm is fired BACKGROUND (asyncio.create_task) so it
    doesn't block bot startup. If someone changes it to `await`,
    startup would block on Ollama -- unacceptable."""
    assert "asyncio.create_task(_prewarm_qwen_coder(" in _MAIN_SRC, (
        "prewarm must be fired as a background task, not awaited"
    )


def test_p23_prewarm_task_cancelled_at_shutdown():
    """Shutdown should cancel the prewarm task if still in flight.
    Prevents 'task was never awaited' warnings + leaked work."""
    assert "prewarm_task.cancel()" in _MAIN_SRC, (
        "prewarm task must be cancelled in shutdown finally"
    )
