import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.router import route


def test_a_standard_happy_path():
    result = route("/ping hello")
    assert result.status == "ok"
    assert result.error_code is None
    assert result.command == "ping"
    assert result.task_id is not None and len(result.task_id) == 32


def test_b_normalization():
    result = route("   /PING   hello world   ")
    assert result.status == "ok"
    assert result.error_code is None
    assert result.command == "ping"
    assert result.route == "/PING"
    assert result.args == {"text": "hello world"}


def test_c_command_not_first():
    result = route("hello /ping")
    assert result.status == "error"
    assert result.error_code == "INVALID_POSITION"


def test_d_empty_args():
    result = route("/ping")
    assert result.status == "ok"
    assert result.error_code is None
    assert result.args == {}
    assert result.task_id is not None


def test_error_results_have_no_task_id():
    assert route("hello /ping").task_id is None
    assert route("").task_id is None
    assert route("/explode boom").task_id is None


def test_e_flag_parsing():
    result = route("/ping --target server1 --verbose")
    assert result.status == "ok"
    assert result.error_code is None
    assert result.args == {"target": "server1", "verbose": True}


def test_e_bonus_case_preservation():
    """Decision #3: flag values preserve case; only command + flag keys are lowercased."""
    result = route("/ping --Target Server1 --VERBOSE")
    assert result.status == "ok"
    assert result.args == {"target": "Server1", "verbose": True}


def test_f_unknown_command():
    result = route("/explode hello")
    assert result.status == "error"
    assert result.error_code == "UNKNOWN_COMMAND"


def test_g_empty_input():
    result = route("")
    assert result.status == "error"
    assert result.error_code == "INVALID_POSITION"
