"""Brain (intent classifier) tests."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.brain import BrainResult, BrainRouter, _extract_json_object


# ---------- pure-function tests (no LLM needed) ----------

def test_strip_think_block_round_trip():
    """Qwen3 sometimes emits <think>...</think>; strip it before parse."""
    raw = '<think>let me think</think>{"intent": "chat", "response": "hi"}'
    parsed = _extract_json_object(raw)
    assert parsed == {"intent": "chat", "response": "hi"}


def test_extract_json_handles_markdown_fences():
    raw = '```json\n{"intent": "chat", "response": "hi"}\n```'
    assert _extract_json_object(raw) == {
        "intent": "chat", "response": "hi",
    }


def test_extract_json_falls_back_to_brace_span():
    raw = "noise before {\"intent\": \"chat\", \"response\": \"hi\"} noise after"
    assert _extract_json_object(raw) == {
        "intent": "chat", "response": "hi",
    }


# ---------- mocked-inference tests for the dispatch logic ----------

class FakeInference:
    """Stand-in inference client. Returns canned responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("FakeInference: no more canned responses")
        text = self.responses.pop(0)
        from types import SimpleNamespace
        return SimpleNamespace(text=text, model_used="sentinel-brain",
                               backend="ollama", inference_time=0.1)


def _run(coro):
    return asyncio.run(coro)


def test_a_chat_intent_for_greeting():
    fake = FakeInference([
        '{"intent": "chat", "response": "Hi! How can I help?"}',
    ])
    brain = BrainRouter(inference_client=fake)
    r = _run(brain.process("hello, how are you?", trace_id="SEN-test-A"))
    assert r.intent == "chat"
    assert r.response and "help" in r.response.lower()


def test_b_dispatch_for_job_search():
    fake = FakeInference([json.dumps({
        "intent": "dispatch", "agent": "job_analyst",
        "command": "/search", "args": "RSM jobs Michigan",
        "summary": "Searching the web for RSM jobs near Michigan.",
    })])
    brain = BrainRouter(inference_client=fake)
    r = _run(brain.process(
        "run the job search for RSM roles near Michigan",
        trace_id="SEN-test-B",
    ))
    assert r.intent == "dispatch"
    assert r.command == "/search"
    assert "michigan" in (r.args or "").lower()


def test_c_chat_intent_for_simple_math():
    fake = FakeInference([
        '{"intent": "chat", "response": "2+2 = 4"}',
    ])
    brain = BrainRouter(inference_client=fake)
    r = _run(brain.process("what's 2+2?", trace_id="SEN-test-C"))
    assert r.intent == "chat"
    assert "4" in r.response


def test_d_dispatch_for_search():
    fake = FakeInference([json.dumps({
        "intent": "dispatch", "agent": "web_search",
        "command": "/search", "args": "python asyncio tutorials",
        "summary": "Searching for python asyncio tutorials.",
    })])
    brain = BrainRouter(inference_client=fake)
    r = _run(brain.process(
        "search for python asyncio tutorials",
        trace_id="SEN-test-D",
    ))
    assert r.intent == "dispatch"
    assert r.command == "/search"


def test_e_dispatch_for_extract():
    fake = FakeInference([json.dumps({
        "intent": "dispatch", "agent": "job_analyst",
        "command": "/extract",
        "args": "Regional Sales Manager at Acme Corp...",
        "summary": "Extracting structured data from the posting.",
    })])
    brain = BrainRouter(inference_client=fake)
    r = _run(brain.process(
        "extract data from this job posting: "
        "Regional Sales Manager at Acme Corp...",
        trace_id="SEN-test-E",
    ))
    assert r.intent == "dispatch"
    assert r.command == "/extract"


def test_f_invalid_json_first_then_valid_retry():
    """Brain emits garbage on first call; retry returns valid JSON."""
    fake = FakeInference([
        "I'm not sure but maybe just chat?",  # garbage
        '{"intent": "chat", "response": "Sorry, can you clarify?"}',
    ])
    brain = BrainRouter(inference_client=fake)
    r = _run(brain.process("hello?", trace_id="SEN-test-F"))
    # After retry parse succeeds -> chat intent
    assert r.intent == "chat"
    assert "clarify" in r.response.lower()
    # Two inference calls happened (initial + retry)
    assert len(fake.calls) == 2


def test_g_dispatch_for_code_problem():
    fake = FakeInference([json.dumps({
        "intent": "dispatch", "agent": "code_assistant",
        "command": "/code",
        "args": "fix my python script that has a bug in the sort function",
        "summary": "Fixing the sort bug.",
    })])
    brain = BrainRouter(inference_client=fake)
    r = _run(brain.process(
        "fix my python script that has a bug in the sort function",
        trace_id="SEN-test-G",
    ))
    assert r.intent == "dispatch"
    assert r.command == "/code"


def test_summarize_falls_through_when_inference_fails():
    """If summarization inference raises, return a graceful fallback
    rather than crashing the bot."""
    class FailingInference:
        async def generate(self, **kwargs):
            raise RuntimeError("simulated brain failure")
    brain = BrainRouter(inference_client=FailingInference())
    out = _run(brain.summarize_result(
        original_request="run a search",
        raw_result={"results": [{"title": "x"}]},
        trace_id="SEN-test-summary-fail",
    ))
    assert "couldn't summarize" in out.lower()
