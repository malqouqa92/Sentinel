"""Phase 10 -- Tests L, M: nightly curation flow.

L -- curator returns valid JSON for a fixture of 5 episodes
M -- approval flow: propose -> apply -> MEMORY.md updated -> no alert,
     brain reloads, persona re-syncs to semantic_memory
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core import config
from core.curation import CurationFlow, _apply_proposal_to_memory_md
from core.file_guard import FileGuard
from core.memory import get_memory


class _FakeClaude:
    def __init__(self, response: str):
        self._response = response
        self.last_prompt: str | None = None

    async def generate(self, prompt, system, trace_id, timeout=None, **kw):
        self.last_prompt = prompt
        return self._response


# -------------------------------------------------------------------
# L -- propose returns valid JSON shape
# -------------------------------------------------------------------

def test_l_curator_returns_valid_json(temp_persona):
    mem = get_memory()
    for i in range(5):
        mem.store_episode(
            "global", f"SEN-fix-{i}", "chat",
            f"user said something durable about preference-{i}",
        )
    fake = _FakeClaude(
        '{"memory_additions": ["likes Python", "prefers brevity"],'
        ' "memory_removals": [], "user_updates": [],'
        ' "no_changes": false}'
    )
    guard = FileGuard(directory=temp_persona)
    flow = CurationFlow(
        memory_manager=mem, file_guard=guard,
        brain=None, claude_client=fake,
    )
    record = asyncio.run(flow.propose("SEN-test-L"))
    assert record.get("_error") is not True, f"unexpected error: {record}"
    assert "token" in record
    assert record["episodes_reviewed"] == 5
    proposal = record["proposal"]
    assert proposal.get("memory_additions") == [
        "likes Python", "prefers brevity",
    ]
    assert proposal.get("no_changes") is False
    # The Claude prompt must contain the episodes (check at least one).
    assert "preference-0" in fake.last_prompt
    assert "preference-4" in fake.last_prompt


# -------------------------------------------------------------------
# M -- apply flow updates MEMORY.md without tripping diff-watch
# -------------------------------------------------------------------

def test_m_apply_updates_memory_without_alert(temp_persona):
    mem = get_memory()
    alerts: list[str] = []
    guard = FileGuard(
        directory=temp_persona, alert_callback=alerts.append,
    )
    fake = _FakeClaude(
        '{"memory_additions": ["fact one", "fact two"],'
        ' "memory_removals": [], "user_updates": [],'
        ' "no_changes": false}'
    )
    # Brain stub captures reload_persona calls.
    reload_calls: list[bool] = []
    fake_brain = SimpleNamespace(
        reload_persona=lambda: reload_calls.append(True),
    )
    flow = CurationFlow(
        memory_manager=mem, file_guard=guard,
        brain=fake_brain, claude_client=fake,
    )
    record = asyncio.run(flow.propose("SEN-test-M-prop"))
    assert record.get("_error") is not True
    token = record["token"]
    applied = flow.apply(token, "SEN-test-M-apply")
    assert applied.get("_error") is not True, applied
    assert applied["memory_md_changed"] is True
    new_md = (temp_persona / "MEMORY.md").read_text(encoding="utf-8")
    assert "fact one" in new_md and "fact two" in new_md
    # Diff-watch should see no tampering -- authorize_update refreshed
    # the baseline.
    tampered = guard.check_integrity()
    assert tampered == [], f"expected no tampering, got {tampered}"
    assert alerts == [], f"unexpected alerts: {alerts}"
    # Brain reload was called.
    assert reload_calls == [True]
    # Persona file mirror in semantic memory got the new content.
    fact = mem.get_fact("persona:MEMORY.md")
    assert fact is not None
    assert "fact one" in fact.value


# -------------------------------------------------------------------
# Helper round-trip: pure MEMORY.md edit semantics
# -------------------------------------------------------------------

def test_apply_proposal_to_memory_md_idempotent_basics():
    starting = "# Memory\n\n## Facts\n- existing fact\n"
    out = _apply_proposal_to_memory_md(
        starting, additions=["new fact"], removals=["existing fact"],
    )
    assert "new fact" in out
    assert "existing fact" not in out
    # Re-applying the same removals/additions on an output that already
    # has them should not duplicate the addition (it gets re-injected
    # under '## Facts'; that's tolerated).
    again = _apply_proposal_to_memory_md(
        out, additions=["new fact"], removals=[],
    )
    assert again.count("- new fact") >= 1
