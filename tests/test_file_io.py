import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.skills import SkillError
from skills.file_io import FileIOInput, FileIOOutput, FileIOSkill


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Per-test isolated workspace so file ops don't pollute the real one."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_DIR", ws)
    yield ws


def _run(skill, raw):
    return asyncio.run(
        skill.execute(skill.validate_input(raw), trace_id="SEN-test-fio"),
    )


def test_d_write_then_read(workspace):
    skill = FileIOSkill()
    out_w = _run(skill, {
        "action": "write", "path": "test_output/test.txt",
        "content": "hello world",
    })
    assert out_w.success and out_w.bytes_written == len("hello world")
    assert (workspace / "test_output" / "test.txt").exists()

    out_r = _run(skill, {"action": "read", "path": "test_output/test.txt"})
    assert out_r.success
    assert out_r.content == "hello world"


def test_e_relative_traversal_blocked(workspace):
    skill = FileIOSkill()
    with pytest.raises(SkillError) as exc:
        _run(skill, {"action": "read", "path": "../../etc/passwd"})
    assert "sandbox" in str(exc.value).lower()


def test_f_absolute_path_outside_workspace_blocked(workspace):
    skill = FileIOSkill()
    with pytest.raises(SkillError) as exc:
        _run(skill, {"action": "read", "path": "C:/Windows/System32/drivers/etc/hosts"})
    assert "sandbox" in str(exc.value).lower()


def test_g_list_after_write(workspace):
    skill = FileIOSkill()
    _run(skill, {
        "action": "write", "path": "test_output/test.txt",
        "content": "hi",
    })
    out = _run(skill, {"action": "list", "path": "test_output"})
    assert out.success
    assert out.files is not None
    assert "test_output/test.txt" in out.files


def test_h_csv_round_trip_with_preview(workspace):
    skill = FileIOSkill()
    csv_text = (
        "name,age,role\n"
        "Alice,30,Engineer\n"
        "Bob,40,Manager\n"
        "Carol,25,Designer\n"
    )
    _run(skill, {
        "action": "write", "path": "data.csv", "content": csv_text,
    })
    out = _run(skill, {"action": "read", "path": "data.csv"})
    assert out.success
    assert out.content == csv_text
    assert out.csv_preview is not None
    assert len(out.csv_preview) == 3
    assert out.csv_preview[0]["name"] == "Alice"
    assert out.csv_preview[2]["role"] == "Designer"


def test_text_mode_parses_action_path_content(workspace):
    skill = FileIOSkill()
    parsed = skill.validate_input(
        {"text": 'write hello.txt "hello from sentinel"'}
    )
    assert parsed.action == "write"
    assert parsed.path == "hello.txt"
    assert parsed.content == "hello from sentinel"
