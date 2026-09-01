#!/usr/bin/env python3
"""在隔离 Git worktree 中调用 Antigravity CLI 完成开发子任务。"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class DispatchError(RuntimeError):
    """调度或交付门禁失败。"""


@dataclass(frozen=True)
class WorktreeState:
    root: Path
    branch: str
    head: str


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise DispatchError(f"Git 命令失败：git {' '.join(args)}\n{detail}")
    return result.stdout.strip()


def inspect_worktree(path: Path) -> WorktreeState:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise DispatchError(f"worktree 不存在：{root}")
    if not (root / ".git").is_file():
        raise DispatchError("为避免两个 Agent 同目录写入，只允许传入外置 linked worktree")

    git_root = Path(run_git(root, "rev-parse", "--show-toplevel")).resolve()
    if git_root != root:
        raise DispatchError(f"必须传入 worktree 根目录，实际根目录为：{git_root}")
    if run_git(root, "status", "--short"):
        raise DispatchError("调度前 worktree 必须干净，请先处理已有修改")

    branch = run_git(root, "branch", "--show-current")
    if not branch:
        raise DispatchError("不允许在 detached HEAD 上调度")
    return WorktreeState(root=root, branch=branch, head=run_git(root, "rev-parse", "HEAD"))


def validate_ownership(patterns: Sequence[str]) -> tuple[str, ...]:
    if not patterns:
        raise DispatchError("至少用一次 --owns 声明 Antigravity 独占修改的文件")

    normalized: list[str] = []
    for raw in patterns:
        value = raw.strip().replace("\\", "/")
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or value.startswith(".git"):
            raise DispatchError(f"非法文件所有权规则：{raw}")
        normalized.append(value)
    return tuple(normalized)


def locate_cli(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        os.environ.get("ANTIGRAVITY_CLI"),
        shutil.which("agy"),
        str(Path.home() / ".local/bin/agy"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return Path(candidate).expanduser().resolve()
    raise DispatchError("未找到 agy；请安装 Antigravity CLI 或设置 ANTIGRAVITY_CLI")


def build_prompt(
    task: str,
    state: WorktreeState,
    ownership: Sequence[str],
    checks: Sequence[str],
) -> str:
    owned = "\n".join(f"- {item}" for item in ownership)
    verification = "\n".join(f"- {item}" for item in checks) or "- 根据改动运行最小相关测试"
    return f"""你是本任务的实现 Agent。严格遵守仓库 AGENTS.md。

任务目标：
{task.strip()}

隔离边界：
- 唯一允许工作的仓库：{state.root}
- 当前分支：{state.branch}
- 所有文件读写和命令都必须把上述目录作为工作目录；不要在 scratch 中创建交付文件。
- 只允许修改以下独占路径：
{owned}
- 未列出的文件只能读取；如发现必须变更公共接口或其他文件，停止并在最终回复中说明，不要越界修改。
- 不得创建、删除或切换 worktree；不得合并、变基、推送或改写历史。
- 不得写入 .stock-analysis/analysis.sqlite3 等生产数据。

验证要求：
{verification}

交付要求：
- 开始前确认 git status --short 为空。
- 完成实现和验证后，只暂存授权路径。
- 创建且只创建一个原子提交；提交主题使用中文并保留 Conventional Commit 类型前缀。
- 最终回复列出修改摘要、验证结果和提交哈希。
"""


def build_command(
    cli: Path,
    state: WorktreeState,
    prompt: str,
    timeout_seconds: int,
    conversation: str | None,
) -> list[str]:
    command = [
        str(cli),
        "-p",
        prompt,
        "--add-dir",
        str(state.root),
        "--mode",
        "accept-edits",
        "--output-format",
        "json",
        "--print-timeout",
        f"{timeout_seconds}s",
    ]
    if conversation:
        command.extend(["--conversation", conversation])
    return command


def parse_result(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise DispatchError("Antigravity 未返回有效 JSON") from exc
    if payload.get("status") != "SUCCESS":
        raise DispatchError(f"Antigravity 执行失败：{payload.get('status', 'UNKNOWN')}")
    return payload


def path_is_owned(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def verify_delivery(before: WorktreeState, ownership: Sequence[str]) -> tuple[str, list[str]]:
    after = inspect_worktree(before.root)
    if after.branch != before.branch:
        raise DispatchError(f"Antigravity 切换了分支：{before.branch} -> {after.branch}")
    if after.head == before.head:
        raise DispatchError("Antigravity 没有创建交付提交")
    count = int(run_git(before.root, "rev-list", "--count", f"{before.head}..{after.head}"))
    if count != 1:
        raise DispatchError(f"要求只创建一个原子提交，实际新增 {count} 个提交")

    changed = run_git(
        before.root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        f"{before.head}..{after.head}",
    ).splitlines()
    unauthorized = [path for path in changed if not path_is_owned(path, ownership)]
    if unauthorized:
        raise DispatchError("提交包含未授权文件：" + ", ".join(unauthorized))
    return after.head, changed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True, help="外置 linked worktree 根目录")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="交给 Antigravity 的任务目标")
    prompt_group.add_argument("--prompt-file", type=Path, help="包含任务目标的 UTF-8 文件")
    parser.add_argument("--owns", action="append", default=[], help="独占路径或 glob，可重复")
    parser.add_argument("--check", action="append", default=[], help="验证命令，可重复")
    parser.add_argument("--conversation", help="继续已有 Antigravity 会话")
    parser.add_argument("--timeout", type=int, default=1800, help="CLI 超时秒数")
    parser.add_argument("--agy", help="agy 可执行文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印调度摘要，不调用 CLI")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        state = inspect_worktree(args.worktree)
        ownership = validate_ownership(args.owns)
        task = args.prompt or args.prompt_file.read_text(encoding="utf-8")
        prompt = build_prompt(task, state, ownership, args.check)
        cli = locate_cli(args.agy)
        command = build_command(cli, state, prompt, args.timeout, args.conversation)

        if args.dry_run:
            print(
                json.dumps({"worktree": str(state.root), "branch": state.branch, "owns": ownership})
            )
            return 0

        result = subprocess.run(
            command,
            cwd=state.root,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout + 30,
        )
        if result.returncode != 0:
            raise DispatchError(result.stderr.strip() or f"agy 退出码：{result.returncode}")
        payload = parse_result(result.stdout)
        commit, changed = verify_delivery(state, ownership)
        print(
            json.dumps(
                {
                    "status": "SUCCESS",
                    "conversation_id": payload.get("conversation_id"),
                    "commit": commit,
                    "changed_files": changed,
                    "response": payload.get("response", ""),
                    "usage": payload.get("usage", {}),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (DispatchError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
