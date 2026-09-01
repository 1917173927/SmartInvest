# Antigravity CLI 双 Agent 协作

本仓库直接调用官方 `agy` CLI，不封装本地 MCP。Codex 负责拆分任务、创建外置 worktree、
声明文件所有权和审查提交；Antigravity 只在该 worktree 中实现一个边界清晰的子任务。

## 一次性准备

安装并登录 Antigravity CLI 后，确认命令可用：

```bash
agy --version
agy -p /model --output-format json
```

调度器按以下顺序寻找 CLI：`--agy` 参数、`ANTIGRAVITY_CLI` 环境变量、`PATH`、
`~/.local/bin/agy`。

## 调度流程

先从主仓库创建外置 worktree：

```bash
git worktree add -b antigravity/<任务名> ../StockAnalysis-wt-<任务名>
```

再声明任务、独占文件和验证命令：

```bash
uv run python scripts/dispatch_antigravity.py \
  --worktree ../StockAnalysis-wt-<任务名> \
  --prompt "实现边界清晰的子任务，并保持现有公共接口兼容" \
  --owns 'src/stock_analysis/example.py' \
  --owns 'tests/test_example.py' \
  --check 'uv run ruff check src/stock_analysis/example.py tests/test_example.py' \
  --check 'uv run pytest tests/test_example.py'
```

含复杂引号或多段说明时，用 `--prompt-file <文件>` 代替 `--prompt`。首次调度前可追加
`--dry-run` 检查 worktree、分支和所有权摘要。

调度器会在调用前后自动执行门禁：

1. 目标必须是干净的 linked worktree，不能是主工作区或 detached HEAD；
2. Antigravity 只能创建一个提交，并且提交后 worktree 必须干净；
3. 提交涉及的所有文件必须匹配 `--owns`；
4. CLI 使用 `accept-edits` 和权限白名单，不使用跳过权限检查的危险参数；
5. 成功时输出会话 ID、提交哈希、文件列表、模型回复和 token 用量的 JSON。

Codex 审查提交后，在集成分支执行 `git cherry-pick <提交哈希>`。如果两个任务需要修改同一文件，
不要并行调度，应按依赖顺序串行完成。

## 权限边界

建议在 `~/.gemini/antigravity-cli/settings.json` 只允许开发所需的 Git、Ruff 和 Pytest 命令，
显式拒绝 `git push`、`git reset`、`git clean`、`git worktree`、网络下载及 `.git/` 写入。
未配置命令在 headless 模式下会被拒绝，避免任务因交互式授权而越权。
