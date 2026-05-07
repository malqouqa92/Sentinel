import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.logger import log_event
from core.telemetry import generate_trace_id

TRACE_ID_PATTERN = re.compile(r"^SEN-[a-f0-9]{8}$")
REQUIRED_FIELDS = {"timestamp", "trace_id", "level", "component", "message"}


def test_trace_id_format():
    ids = [generate_trace_id() for _ in range(10)]
    for tid in ids:
        assert TRACE_ID_PATTERN.fullmatch(tid), f"Bad trace id: {tid!r}"
    assert len(set(ids)) == 10, "Trace ids must be unique"


def test_log_write():
    log_path = config.LOG_DIR / config.LOG_FILE
    log_event("SEN-deadbeef", "INFO", "test", "Phase 1 boot")
    assert log_path.exists(), f"Log file missing: {log_path}"
    last_line = log_path.read_text(encoding="utf-8").splitlines()[-1]
    parsed = json.loads(last_line)
    missing = REQUIRED_FIELDS - parsed.keys()
    assert not missing, f"Missing required fields: {missing}"
    for field in REQUIRED_FIELDS:
        assert parsed[field] != "" and parsed[field] is not None, f"Empty field: {field}"
    assert parsed["trace_id"] == "SEN-deadbeef"
    assert parsed["level"] == "INFO"
    assert parsed["component"] == "test"
    assert parsed["message"] == "Phase 1 boot"


def test_config_integrity():
    assert config.LOG_DIR.exists(), f"LOG_DIR does not exist: {config.LOG_DIR}"
    assert config.VRAM_LIMIT_MB == 4096
    assert config.PROJECT_NAME == "sentinel"
    assert config.WORKSPACE_DIR.exists(), f"WORKSPACE_DIR does not exist: {config.WORKSPACE_DIR}"
    assert config.LOG_FILE == "sentinel.jsonl"
    assert config.DEFAULT_MODEL
    # Subset check: core commands always present; skills register more.
    assert {"/ping", "/help", "/status"}.issubset(config.REGISTERED_COMMANDS)
