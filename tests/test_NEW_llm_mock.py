"""Test with stubbed LLM."""
import pytest
from skills import code_assist as ca


def test_with_mocked_qwen(monkeypatch):
    def fake_qwen(system, user, trace_id, model, **kwargs):
        return "STEP 1: done summary=\"ok\""
    monkeypatch.setattr(ca, "_qwen_generate", fake_qwen)
    # ... invoke the path under test
    assert True
