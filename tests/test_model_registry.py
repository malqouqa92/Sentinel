import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model_registry import ModelRegistry


REQUIRED_NAMES = {"qwen3-brain", "qwen-coder", "claude-cli"}


def test_a_default_roster_loads():
    """The roster MUST include the 3 core models. Phase 9 added
    sentinel-brain (custom Modelfile) so >= subset is the right check."""
    reg = ModelRegistry()
    names = {m.name for m in reg.list_models()}
    assert names >= REQUIRED_NAMES, f"missing core roster models: {names}"


def test_b_get_by_tier():
    """No basic tier (model swap penalty wipes any speed gain on
    constrained VRAM). Locals are 'standard'; claude-cli is 'advanced'."""
    reg = ModelRegistry()
    basic = reg.get_by_tier("basic")
    standard = reg.get_by_tier("standard")
    advanced = reg.get_by_tier("advanced")
    assert basic == [], "no basic-tier models in simplified roster"
    assert {m.name for m in standard} >= {"qwen3-brain", "qwen-coder"}
    assert {m.name for m in advanced} == {"claude-cli"}


def test_c_get_cheapest_capable_prefers_local():
    """At standard tier, a local Ollama model wins over claude-cli
    because both are cost 0 and faster speed_tier breaks the tie."""
    reg = ModelRegistry()
    for m in reg.list_models():
        m.available = True
    pick = reg.get_cheapest_capable("standard")
    assert pick is not None
    assert pick.backend == "ollama", f"got {pick.name} ({pick.backend})"


@pytest.mark.requires_ollama
def test_d_check_availability_works():
    """check_availability() returns a dict keyed by every roster name,
    and the core required entries are present."""
    reg = ModelRegistry()
    avail = reg.check_availability()
    assert isinstance(avail, dict)
    assert set(avail.keys()) >= REQUIRED_NAMES
    assert avail["qwen3-brain"] is True, \
        "qwen3-brain (qwen3:1.7b) should be pulled (Phase 8)"
    assert "claude-cli" in avail
