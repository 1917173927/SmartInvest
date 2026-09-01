#!/usr/bin/env python3
"""Install the daily macOS launchd job using the current checkout path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    python = project / ".venv" / "bin" / "python"
    if not python.exists():
        raise SystemExit("缺少 .venv；请先运行 uv sync")

    launch_agents = Path.home() / "Library" / "LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            "com.stockanalysis.daily",
            "com.stockanalysis.daily.plist.template",
            "每日 18:30 盘后自动全流程",
        ),
        (
            "com.stockanalysis.morning",
            "com.stockanalysis.morning.plist.template",
            "工作日 09:00 盘前挂单晨报",
        ),
    ]

    for label, template_name, desc in tasks:
        template = project / "scripts" / template_name
        destination = launch_agents / f"{label}.plist"
        content = template.read_text(encoding="utf-8")
        content = content.replace("__PROJECT_DIR__", str(project)).replace(
            "__PYTHON__", str(python)
        )
        destination.write_text(content, encoding="utf-8")
        subprocess.run(["launchctl", "unload", str(destination)], check=False)
        subprocess.run(["launchctl", "load", str(destination)], check=True)
        print(f"已安装 {desc} 定时任务: {destination}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
