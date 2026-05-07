import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm import (
    InferenceClient, InferenceResult, LLMError,
)
from core.model_registry import MODEL_REGISTRY


@pytest.fixture(autouse=True)
def _all_models_available():
    """Force every model to available for these tests so a missing
    pull doesn't make the chain skip the model under test."""
    for m in MODEL_REGISTRY.list_models():
        m.available = True
    yield


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_j_explicit_qwen_brain():
    client = InferenceClient()
    out = asyncio.run(client.generate(
        prompt="Reply with the single word 'pong' and nothing else.",
        model="qwen3-brain",
        trace_id="SEN-test-J",
    ))
    assert isinstance(out, InferenceResult)
    assert out.model_used == "qwen3-brain"
    assert out.backend == "ollama"
    assert out.text.strip() != ""
    assert out.fallback_used is False


@pytest.mark.requires_claude
@pytest.mark.slow
def test_k_explicit_claude_cli():
    client = InferenceClient()
    out = asyncio.run(client.generate(
        prompt="Reply with the single word 'pong' and nothing else.",
        model="claude-cli",
        trace_id="SEN-test-K-inf",
    ))
    assert out.model_used == "claude-cli"
    assert out.backend == "claude_cli"
    assert "pong" in out.text.lower()


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_l_complexity_picks_simple_tier():
    """For a small input, the classifier picks 'standard' (no basic
    tier exists in the simplified roster). InferenceResult carries
    the score and tier."""
    client = InferenceClient()
    out = asyncio.run(client.generate_with_complexity(
        prompt="Reply with the single word 'ok' and nothing else.",
        command="ping",   # base 0.0 -> tier=basic, recommended falls
                          # through to a standard model since no basic
                          # is available
        args={"text": "tiny"},
        trace_id="SEN-test-L",
    ))
    assert out.complexity_tier in ("basic", "standard")
    assert out.complexity_score is not None
    assert out.text.strip() != ""


@pytest.mark.requires_claude
@pytest.mark.slow
def test_m_complexity_picks_advanced_tier():
    """For a complex problem with 'thread-safe' + 'distributed' etc.,
    classifier picks 'advanced' which routes to claude-cli."""
    client = InferenceClient()
    prompt = (
        "In one short paragraph, describe the safety properties of a "
        "linearizable distributed counter. Just one paragraph."
    )
    out = asyncio.run(client.generate_with_complexity(
        prompt=prompt,
        command="code",
        args={"text": (
            "Implement a linearizable distributed counter that is "
            "thread-safe with atomic eviction and tolerant of "
            "concurrent failure. Optimize for high concurrency."
        )},
        trace_id="SEN-test-M",
    ))
    assert out.complexity_tier == "advanced"
    assert out.model_used == "claude-cli"
    assert out.text.strip() != ""


def test_n_fallback_on_primary_failure(monkeypatch):
    """Force the FIRST model in the chain to raise (whichever it is --
    Phase 9 added sentinel-brain so the recommended pick can vary).
    The InferenceClient must fall through to the next and set
    fallback_used=True."""
    client = InferenceClient()
    real_call = client._call_one
    call_log: list[str] = []

    async def maybe_fail(model_cfg, *args, **kwargs):
        call_log.append(model_cfg.name)
        if len(call_log) == 1:
            raise LLMError(
                f"simulated brain failure for test_n on {model_cfg.name}"
            )
        return await real_call(model_cfg, *args, **kwargs)

    monkeypatch.setattr(client, "_call_one", maybe_fail)

    out = asyncio.run(client.generate_with_complexity(
        prompt="Reply with the word 'ok'.",
        command="extract",   # base 0.4 -> standard tier
        args={"text": "small"},
        trace_id="SEN-test-N",
    ))
    assert out.fallback_used is True
    assert out.fallback_reason is not None
    assert "simulated brain failure" in out.fallback_reason
    assert len(call_log) >= 2
    assert out.model_used != call_log[0], \
        f"fallback should have moved past {call_log[0]}; " \
        f"final={out.model_used} chain={call_log}"
