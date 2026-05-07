import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.skills import SkillError
from skills.code_execute import (
    CodeExecuteInput, CodeExecuteOutput, CodeExecuteSkill,
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_DIR", ws)
    yield ws


def _run(skill, raw):
    return asyncio.run(
        skill.execute(skill.validate_input(raw), trace_id="SEN-test-exec"),
    )


def test_i_simple_print(workspace):
    skill = CodeExecuteSkill()
    out = _run(skill, {"code": 'print("sentinel alive")'})
    assert isinstance(out, CodeExecuteOutput)
    assert out.return_code == 0
    assert "sentinel alive" in out.stdout


def test_j_infinite_loop_times_out(workspace):
    skill = CodeExecuteSkill()
    out = _run(skill, {"code": "while True: pass", "timeout": 3})
    assert out.return_code == -1
    assert "timed out" in out.stderr.lower()


def test_k_explicit_exit_code(workspace):
    skill = CodeExecuteSkill()
    out = _run(skill, {"code": "import sys; sys.exit(1)"})
    assert out.return_code == 1


def test_l_syntax_error(workspace):
    skill = CodeExecuteSkill()
    out = _run(skill, {"code": "def oops(:\n    return 1\n"})
    assert out.return_code != 0
    assert "syntaxerror" in out.stderr.lower()


def test_m_save_as_persists_file_and_executes(workspace):
    skill = CodeExecuteSkill()
    out = _run(skill, {
        "code": "print('via save_as')",
        "save_as": "test_script.py",
    })
    assert out.return_code == 0
    assert "via save_as" in out.stdout
    assert out.saved_path == "test_script.py"
    saved = workspace / "test_script.py"
    assert saved.exists()
    assert "via save_as" in saved.read_text()


def test_n_save_as_traversal_blocked(workspace):
    skill = CodeExecuteSkill()
    with pytest.raises(SkillError) as exc:
        _run(skill, {"code": "print('x')", "save_as": "../escape.py"})
    assert "sandbox" in str(exc.value).lower()


def test_text_mode_treats_text_as_code(workspace):
    skill = CodeExecuteSkill()
    parsed = skill.validate_input({"text": 'print("hello")'})
    assert parsed.code == 'print("hello")'
