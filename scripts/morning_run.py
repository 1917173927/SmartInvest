#!/usr/bin/env python3
"""Unattended entry point for pre-market morning brief (runs at 09:00 AM).

Run with ``uv run python scripts/morning_run.py`` from the project directory.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from stock_analysis.data import AppConfig
from stock_analysis.morning import generate_morning_brief


def main() -> int:
    config = AppConfig.load(Path(__file__).resolve().parents[1])
    log_path = config.cache_dir / "morning.log"
    handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    brief = generate_morning_brief(config, send_notification=True)
    print(f"晨报已生成：{brief.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
