from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "dispatch_antigravity.py"
SPEC = importlib.util.spec_from_file_location("dispatch_antigravity", SCRIPT)
assert SPEC and SPEC.loader
dispatch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dispatch
SPEC.loader.exec_module(dispatch)


def state() -> dispatch.WorktreeState:
    return dispatch.WorktreeState(Path("/tmp/repo-wt"), "codex/example", "a" * 40)


def test_validate_ownership_rejects_escape_and_git_paths() -> None:
    for pattern in ("../secret", "/tmp/file", ".git/config"):
        with pytest.raises(dispatch.DispatchError):
            dispatch.validate_ownership([pattern])


def test_prompt_contains_isolation_commit_and_checks() -> None:
    prompt = dispatch.build_prompt(
        "实现调度器",
        state(),
        ["src/example.py", "tests/test_example.py"],
        ["uv run pytest tests/test_example.py"],
    )

    assert "/tmp/repo-wt" in prompt
    assert "未列出的文件只能读取" in prompt
    assert "只创建一个原子提交" in prompt
    assert "提交主题使用中文" in prompt
    assert "uv run pytest tests/test_example.py" in prompt


def test_build_command_uses_added_worktree_and_safe_mode() -> None:
    command = dispatch.build_command(Path("/opt/agy"), state(), "任务", 600, "conversation-1")

    assert command[:3] == ["/opt/agy", "-p", "任务"]
    assert command[command.index("--add-dir") + 1] == "/tmp/repo-wt"
    assert command[command.index("--mode") + 1] == "accept-edits"
    assert "--dangerously-skip-permissions" not in command
    assert command[-2:] == ["--conversation", "conversation-1"]


def test_parse_result_requires_success_json() -> None:
    assert dispatch.parse_result('{"status":"SUCCESS","response":"ok"}')["response"] == "ok"
    with pytest.raises(dispatch.DispatchError):
        dispatch.parse_result("not-json")
    with pytest.raises(dispatch.DispatchError):
        dispatch.parse_result('{"status":"ERROR"}')


def test_path_is_owned_supports_exact_file_and_glob() -> None:
    patterns = ("src/owned.py", "tests/agent/**")

    assert dispatch.path_is_owned("src/owned.py", patterns)
    assert dispatch.path_is_owned("tests/agent/test_cli.py", patterns)
    assert not dispatch.path_is_owned("src/other.py", patterns)
