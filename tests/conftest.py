import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config


def pytest_configure(config):  # pytest hook requires this exact parameter name
    config.addinivalue_line(
        "markers",
        "requires_ollama: test needs a running Ollama daemon at "
        "config.OLLAMA_BASE_URL; auto-skipped otherwise",
    )
    config.addinivalue_line(
        "markers",
        "requires_network: test needs internet access (DuckDuckGo etc); "
        "auto-skipped otherwise",
    )
    config.addinivalue_line(
        "markers",
        "requires_claude: test needs SENTINEL_CLAUDE_KEY env var "
        "for the Claude API teaching path; auto-skipped otherwise",
    )
    config.addinivalue_line(
        "markers",
        "slow: long-running integration tests (LLM end-to-end)",
    )


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """Session-scoped health check. Cached so we hit Ollama once per
    test run, not once per test."""
    from core.llm import OllamaClient
    return OllamaClient().health_check()


@pytest.fixture(autouse=True)
def _maybe_skip_if_no_ollama(request, ollama_available):
    if request.node.get_closest_marker("requires_ollama"):
        if not ollama_available:
            pytest.skip(
                "Ollama not reachable at config.OLLAMA_BASE_URL — "
                "start it with `ollama serve` then re-run"
            )


@pytest.fixture(scope="session")
def network_available() -> bool:
    """Session-scoped check for outbound internet via a tiny HEAD request
    to DuckDuckGo. Cached so we hit the network once per pytest run."""
    import httpx
    try:
        r = httpx.head("https://duckduckgo.com", timeout=5.0,
                       follow_redirects=True)
        return r.status_code < 500
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _maybe_skip_if_no_network(request, network_available):
    if request.node.get_closest_marker("requires_network"):
        if not network_available:
            pytest.skip(
                "no outbound internet reachable — requires_network test "
                "needs DuckDuckGo and target sites"
            )


@pytest.fixture(scope="session")
def claude_cli_available() -> bool:
    """Check whether the local Claude Code CLI is reachable. Sentinel's
    teaching loop shells out to it; no internet API calls."""
    from skills.code_assist import _find_claude_cli
    return _find_claude_cli() is not None


@pytest.fixture(autouse=True)
def _maybe_skip_if_no_claude(request, claude_cli_available):
    if request.node.get_closest_marker("requires_claude"):
        if not claude_cli_available:
            pytest.skip(
                "`claude` CLI not found -- the local Claude Code CLI "
                "must be installed and authenticated. Sentinel makes "
                "no API calls; teaching uses the CLI subprocess."
            )


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Per-test isolated SQLite database. Initializes the schema and
    monkeypatches config.DB_PATH so all DB calls in the test see the
    fresh file. Autouse so tests get clean state without ceremony."""
    db_path = tmp_path / "sentinel-test.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    # Import lazily so the monkeypatch lands before init_db runs.
    from core import database
    importlib.reload(database)  # rebind module-level references to new path
    database.init_db()
    yield db_path


@pytest.fixture(autouse=True)
def temp_memory(tmp_path, monkeypatch):
    """Phase 10: per-test isolated memory.db. Resets the singleton so
    each test sees a fresh DB without cross-test bleed. Cheap (FTS5
    init is sub-ms), safe to autouse."""
    mem_path = tmp_path / "memory-test.db"
    monkeypatch.setattr(config, "MEMORY_DB_PATH", mem_path)
    from core import memory as _memory
    _memory.reset_memory_singleton()
    _memory.WORKING_MEMORY.clear()
    yield mem_path
    _memory.reset_memory_singleton()


@pytest.fixture
def temp_persona(tmp_path, monkeypatch):
    """Phase 10: tmp persona directory. NOT autouse -- only the persona
    + file_guard tests opt in. Other tests keep reading the real
    workspace/persona seeds (which is fine, they ignore PERSONA_DIR)."""
    persona = tmp_path / "persona"
    persona.mkdir()
    for name, content in {
        "IDENTITY.md": "# Identity\n- Name: Test\n",
        "SOUL.md": "# Soul\n- Be terse.\n",
        "USER.md": "# User\n- Tester\n",
        "MEMORY.md": "# Memory\n(empty)\n",
    }.items():
        (persona / name).write_text(content, encoding="utf-8")
    monkeypatch.setattr(config, "PERSONA_DIR", persona)
    yield persona


@pytest.fixture(scope="session", autouse=True)
def _initialize_registries():
    """Discover and register all skills + agents once per pytest session.
    Order matters: skills must load first because agents validate their
    pipelines against the skill registry."""
    from core.agent_registry import AGENT_REGISTRY
    from core.registry import SKILL_REGISTRY
    SKILL_REGISTRY.reset()
    SKILL_REGISTRY.discover()
    AGENT_REGISTRY.reset()
    AGENT_REGISTRY.discover()
    yield
