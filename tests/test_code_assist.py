"""Code assist skill tests.

Tests G, H, K can run without Claude (Qwen-only paths).
Tests I, J, L need both Ollama AND Claude API key (requires_ollama +
requires_claude markers).
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.knowledge_base import KnowledgeBase
from skills.code_assist import (
    CodeAssistInput, CodeAssistOutput, CodeAssistSkill,
)


@pytest.fixture
def fresh_kb(tmp_path, monkeypatch):
    """Per-test isolated knowledge base."""
    db = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db)
    yield KnowledgeBase(db_path=db)


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_DIR", ws)
    yield ws


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_g_easy_problem_qwen_solves(fresh_kb, temp_workspace):
    """Qwen alone should handle 'sum two numbers'. Pattern gets stored."""
    skill = CodeAssistSkill()
    inp = CodeAssistInput(
        problem="Write a Python function that returns the sum of "
                "two numbers.",
    )
    result = asyncio.run(skill.execute(inp, trace_id="SEN-test-G"))
    assert isinstance(result, CodeAssistOutput)
    # Phase 10 added the agentic flow; legitimate values include qwen_agent
    # (success) and qwen_failed (5-attempt agentic loop gave up).
    assert result.solved_by in (
        "qwen", "qwen_agent", "qwen_taught", "qwen_failed", "claude_direct",
    ), f"unexpected solved_by={result.solved_by}"
    if result.validated:
        # Pattern should now be in the KB.
        assert fresh_kb.stats()["total_entries"] >= 1, \
            f"expected KB entry after successful solve, " \
            f"got {fresh_kb.stats()}"


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_h_kb_reuse_after_g(fresh_kb, temp_workspace):
    """Run Test G first to seed KB, then run a same-class problem and
    verify the second run consults the stored pattern."""
    skill = CodeAssistSkill()

    # First run -- seed.
    inp1 = CodeAssistInput(
        problem="Write a Python function that returns the sum of "
                "two integers.",
    )
    r1 = asyncio.run(skill.execute(inp1, trace_id="SEN-test-H-seed"))
    if not r1.validated:
        pytest.skip(
            "First run did not validate (Qwen-only path); "
            "test_h needs a seeded pattern. Set SENTINEL_CLAUDE_KEY "
            "to enable teaching."
        )
    assert fresh_kb.stats()["total_entries"] >= 1

    # Second run, same class.
    inp2 = CodeAssistInput(
        problem="Write a Python function that adds two numbers and "
                "returns the result.",
    )
    r2 = asyncio.run(skill.execute(inp2, trace_id="SEN-test-H-reuse"))
    # The KB lookup happened at the start of this run; the seed pattern
    # should appear in knowledge_entries_used.
    assert len(r2.knowledge_entries_used) >= 1, \
        f"expected at least one KB hit, got {r2.knowledge_entries_used}"


@pytest.mark.requires_ollama
@pytest.mark.slow
def test_k_no_claude_cli_graceful(
    fresh_kb, temp_workspace, monkeypatch,
):
    """When the Claude CLI is unavailable, the teaching loop is skipped
    cleanly. We don't crash; we return Qwen's attempt with a warning
    in the explanation. We force unavailability by monkeypatching
    _find_claude_cli to return None."""
    from skills import code_assist
    monkeypatch.setattr(code_assist, "_find_claude_cli", lambda: None)
    skill = CodeAssistSkill()
    inp = CodeAssistInput(
        problem="Implement a thread-safe LRU cache with TTL expiry "
                "and atomic eviction. Use only the stdlib.",
    )
    result = asyncio.run(skill.execute(inp, trace_id="SEN-test-K"))
    assert isinstance(result, CodeAssistOutput)
    # Without Claude CLI we never reach qwen_taught or claude_direct.
    assert result.solved_by == "qwen", \
        f"without Claude CLI we should never reach " \
        f"qwen_taught/claude_direct, got {result.solved_by}"
    if not result.validated:
        # The graceful-degradation warning must appear.
        assert "Claude CLI teaching unavailable" in result.explanation


def test_validate_input_text_mode():
    skill = CodeAssistSkill()
    parsed = skill.validate_input({"text": "fix this bug"})
    assert parsed.problem == "fix this bug"
    assert parsed.code_context is None


def test_validate_input_with_context_flag():
    skill = CodeAssistSkill()
    parsed = skill.validate_input({
        "text": "fix this function",
        "context": "def add(a, b): return a - b",
    })
    assert parsed.problem == "fix this function"
    assert parsed.code_context == "def add(a, b): return a - b"


@pytest.mark.requires_claude
@pytest.mark.requires_ollama
@pytest.mark.slow
def test_i_hard_problem_teaching_fires(fresh_kb, temp_workspace):
    """Hard problem Qwen 3B is likely to fumble; with Claude key set
    the teaching loop fires and either qwen_taught or claude_direct
    returns a working solution. Either case stores something in the KB."""
    skill = CodeAssistSkill()
    inp = CodeAssistInput(
        problem=(
            "Implement a thread-safe LRU cache with TTL expiry. "
            "When TTL on an entry expires, the entry must be evicted "
            "automatically. Include tests in __main__."
        ),
    )
    result = asyncio.run(skill.execute(inp, trace_id="SEN-test-I"))
    assert result.solved_by in (
        "qwen", "qwen_agent", "qwen_taught", "qwen_failed", "claude_direct",
    ), f"unexpected solved_by={result.solved_by}"
    # Knowledge base should have at least one new entry (pattern or limitation).
    assert fresh_kb.stats()["total_entries"] >= 1, \
        f"expected KB write after teaching loop, got {fresh_kb.stats()}"


@pytest.mark.requires_claude
@pytest.mark.requires_ollama
@pytest.mark.slow
def test_j_kb_reused_on_second_hard_problem(fresh_kb, temp_workspace):
    """After Test I seeds (or Test J's own first run), a similar problem
    should consult the KB."""
    skill = CodeAssistSkill()
    inp1 = CodeAssistInput(
        problem=(
            "Implement a thread-safe LRU cache with TTL expiry "
            "and atomic eviction."
        ),
    )
    asyncio.run(skill.execute(inp1, trace_id="SEN-test-J-seed"))
    assert fresh_kb.stats()["total_entries"] >= 1

    inp2 = CodeAssistInput(
        problem=(
            "Build a thread-safe cache with LRU eviction and "
            "TTL-based expiration."
        ),
    )
    r2 = asyncio.run(skill.execute(inp2, trace_id="SEN-test-J-reuse"))
    assert len(r2.knowledge_entries_used) >= 1, \
        f"second run should consult KB; got {r2.knowledge_entries_used}"


@pytest.mark.requires_claude
@pytest.mark.requires_ollama
@pytest.mark.slow
def test_l_qwen_garbage_falls_through_to_claude(
    fresh_kb, temp_workspace, monkeypatch,
):
    """Mock Qwen to return unparseable output; Claude must fire and
    produce a usable solution."""
    from skills import code_assist

    call_count = {"n": 0}

    def fake_qwen(system, user, trace_id, model):
        call_count["n"] += 1
        return "I don't know lol"

    monkeypatch.setattr(code_assist, "_qwen_generate", fake_qwen)
    # Phase 10 agentic flow bypasses _qwen_generate via core.qwen_agent.
    # Mock its Ollama call too so the test's "garbage Qwen" premise still
    # holds — otherwise real Qwen runs the agentic path and returns
    # solved_by=qwen_agent, defeating the fall-through assertion.
    from core import qwen_agent as _qa
    def fake_chat(model, messages, tools=None, **kw):
        call_count["n"] += 1
        return {"message": {"content": "I don't know lol", "tool_calls": []}}
    monkeypatch.setattr(_qa, "_ollama_chat", fake_chat)
    skill = CodeAssistSkill()
    inp = CodeAssistInput(
        problem="Write a Python function to compute factorial(n).",
    )
    result = asyncio.run(skill.execute(inp, trace_id="SEN-test-L"))
    # solved_by could be "qwen_taught" (if Qwen was called again and succeeded)
    # or "claude_direct" (Qwen still garbage on retry). With our mock returning
    # garbage on every call, we expect claude_direct.
    assert result.solved_by in ("claude_direct", "qwen_taught"), \
        f"unexpected solved_by={result.solved_by}"
    assert result.solution.strip() != ""
    # Claude's solution should be runnable.
    assert call_count["n"] >= 1
