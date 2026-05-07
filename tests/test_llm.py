import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.llm import LLMError, OllamaClient


@pytest.mark.requires_ollama
def test_a_health_check():
    assert OllamaClient().health_check() is True


@pytest.mark.requires_ollama
def test_b_simple_generate():
    text = OllamaClient().generate(
        model=config.DEFAULT_MODEL, prompt="Say hello in one word.",
        timeout=60,
    )
    assert isinstance(text, str)
    assert text.strip() != ""


@pytest.mark.requires_ollama
def test_c_timeout_handling():
    """Sub-millisecond timeout deterministically fires the timeout path."""
    with pytest.raises(LLMError) as exc_info:
        OllamaClient().generate(
            model=config.DEFAULT_MODEL,
            prompt="Write a long essay about clouds.",
            timeout=1,
        )
    msg = str(exc_info.value).lower()
    assert "timed out" in msg or "timeout" in msg


@pytest.mark.requires_ollama
def test_d_bad_model():
    with pytest.raises(LLMError) as exc_info:
        OllamaClient().generate(
            model="nonexistent-model-xyz-123",
            prompt="hi",
            timeout=10,
        )
    msg = str(exc_info.value).lower()
    assert "not available" in msg or "not found" in msg


def test_health_check_false_when_unreachable():
    """Pure-mock: client pointed at a port nothing is listening on
    returns False from health_check (does not raise)."""
    bogus = OllamaClient(base_url="http://127.0.0.1:1")
    assert bogus.health_check() is False
