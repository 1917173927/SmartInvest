#!/usr/bin/env python3
"""Install the daily macOS launchd job using the current checkout path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    template = project / "scripts" / "com.stockanalysis.daily.plist.template"
    destination = Path.home() / "Library" / "LaunchAgents" / "com.stockanalysis.daily.plist"
    python = project / ".venv" / "bin" / "python"
    if not python.exists():
        raise SystemExit(
            "缺少 .venv；请先运行 uv sync --extra data --extra forecast --extra charts"
        )
    content = template.read_text(encoding="utf-8")
    content = content.replace("__PROJECT_DIR__", str(project)).replace("__PYTHON__", str(python))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    subprocess.run(["launchctl", "unload", str(destination)], check=False)
    subprocess.run(["launchctl", "load", str(destination)], check=True)
    print(f"已安装每日 18:30 自动任务: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
