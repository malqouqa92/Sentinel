"""Phase 10 -- Tests A-D: persona files + diff-watch.

Test A -- persona files load on startup; brain context contains all four.
Test B -- external SOUL.md edit triggers tampering detection + alert.
Test C -- authorize_update writes file + refreshes hash; no alert fires.
Test D -- sync_persona_files mirrors USER.md into semantic_memory.
"""
from __future__ import annotations

from core import config
from core.brain import BrainRouter
from core.file_guard import FileGuard
from core.memory import get_memory


class _DummyInference:
    """Avoid touching Ollama -- BrainRouter only needs an attribute."""
    pass


def test_a_persona_files_load_into_brain_context(temp_persona):
    """A: All four persona files load and end up in brain.persona_context."""
    brain = BrainRouter(inference_client=_DummyInference())
    assert set(brain.persona_files.keys()) == {
        "IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md",
    }
    ctx = brain.persona_context
    for name in ("IDENTITY.md", "SOUL.md", "USER.md", "MEMORY.md"):
        assert f"=== {name} ===" in ctx, f"missing section: {name}"
    # And the system prompt (what the brain actually sends to the LLM)
    # includes the persona block.
    sys = brain._build_system_prompt()
    assert "PERSONA CONTEXT" in sys
    assert "IDENTITY.md" in sys


def test_b_external_modification_triggers_alert(temp_persona):
    """B: editing SOUL.md outside authorize_update returns it from
    check_integrity() AND fires the alert callback."""
    alerts: list[str] = []
    guard = FileGuard(
        directory=temp_persona,
        alert_callback=alerts.append,
    )
    # Tamper outside authorize_update.
    (temp_persona / "SOUL.md").write_text(
        "# Soul\n- TAMPERED\n", encoding="utf-8",
    )
    tampered = guard.check_integrity()
    assert tampered == ["SOUL.md"], (
        f"expected ['SOUL.md'], got {tampered}"
    )
    assert len(alerts) == 1, f"expected 1 alert, got {len(alerts)}"
    assert "SOUL.md" in alerts[0]
    assert "NOT" in alerts[0] or "not" in alerts[0]


def test_c_authorize_update_no_alert(temp_persona):
    """C: authorize_update writes file + refreshes hash; subsequent
    check_integrity() finds nothing tampered, no alert fires."""
    alerts: list[str] = []
    guard = FileGuard(
        directory=temp_persona,
        alert_callback=alerts.append,
    )
    new_content = "# Memory\n## Facts\n- approved fact\n"
    guard.authorize_update("MEMORY.md", new_content)
    # File on disk has the new content.
    assert (temp_persona / "MEMORY.md").read_text(
        encoding="utf-8",
    ) == new_content
    # No alert fires from the integrity check.
    tampered = guard.check_integrity()
    assert tampered == [], f"expected no tampering, got {tampered}"
    assert alerts == [], f"expected no alerts, got {alerts}"


def test_d_sync_persona_files_mirrors_into_semantic(temp_persona):
    """D: sync_persona_files puts each file under
    key='persona:<name>' with source='persona_file'."""
    mem = get_memory()
    n = mem.sync_persona_files(persona_dir=temp_persona)
    assert n == 4, f"expected 4 files synced, got {n}"
    user_fact = mem.get_fact("persona:USER.md")
    assert user_fact is not None
    assert user_fact.source == "persona_file"
    assert user_fact.confidence == 1.0
    assert "Tester" in user_fact.value
