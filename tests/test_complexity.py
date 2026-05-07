import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.complexity import ComplexityResult, classify_complexity
from core.knowledge_base import KnowledgeBase
from core.model_registry import MODEL_REGISTRY


@pytest.fixture(autouse=True)
def _all_models_available():
    """Force every model to available=True for these tests so the
    recommendation isn't an artifact of which models happen to be pulled."""
    for m in MODEL_REGISTRY.list_models():
        m.available = True
    yield


def test_e_ping_is_basic():
    r = classify_complexity("/ping", {"text": "hello"})
    assert r.tier == "basic"
    assert r.score < 0.3


def test_f_simple_code_problem_is_standard(tmp_path, monkeypatch):
    # Use an isolated empty KB so a real-DB "limitation" entry can't
    # bump this borderline case to advanced.
    from core import config as _cfg
    monkeypatch.setattr(_cfg, "KNOWLEDGE_DB_PATH", tmp_path / "empty.db")
    r = classify_complexity(
        "code",
        {"text": "reverse a string in python"},
    )
    assert r.tier in ("standard", "basic"), (
        f"got tier={r.tier} score={r.score} reasoning={r.reasoning}"
    )


def test_g_complex_code_problem_is_advanced():
    r = classify_complexity(
        "code",
        {"text": "implement a thread-safe LRU cache with TTL "
                 "and eviction callbacks for distributed workers, "
                 "optimize for high concurrency"},
    )
    # /code 0.5, complex_keywords +0.2, long text near 200+ -> 0.6+
    assert r.tier == "advanced"
    assert r.score >= 0.6


def test_h_kb_limitation_bumps_to_advanced(tmp_path, monkeypatch):
    """An entry with category='limitation' that matches the problem
    keywords forces advanced tier even when score would be lower."""
    db_path = tmp_path / "kb.db"
    monkeypatch.setattr(config, "KNOWLEDGE_DB_PATH", db_path)
    kb = KnowledgeBase(db_path=db_path)
    kb.add_limitation(
        tags=["palindrome", "string"],
        problem_summary="Qwen could not learn: palindrome detection "
                        "with unicode normalization",
        explanation="qwen kept tripping on combining characters",
        trace_id="SEN-test-H-bump",
    )
    # Without the KB limitation, "reverse a string" should NOT be advanced.
    # With a matching limitation in the KB, it's bumped.
    r = classify_complexity(
        "code",
        {"text": "write a palindrome checker that handles unicode"},
    )
    assert r.tier == "advanced", \
        f"expected KB limitation to bump to advanced; got {r.tier}"
    # Reasoning should mention the bump
    assert any("limitation" in line.lower() for line in r.reasoning)


def test_i_auto_routing_disabled_returns_default(monkeypatch):
    monkeypatch.setattr(config, "AUTO_ROUTING_ENABLED", False)
    r = classify_complexity(
        "code",
        {"text": "implement a thread-safe LRU cache with TTL"},
    )
    assert r.tier == "standard"
    # In the 3-model roster, the default-when-routing-disabled is the
    # brain (chat-tuned 3b). The string is a registry name, not model_id.
    assert r.recommended_model in {"qwen3-brain", "qwen-coder"}
    assert any("AUTO_ROUTING_ENABLED is False" in line
               for line in r.reasoning)
