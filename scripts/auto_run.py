#!/usr/bin/env python3
"""Unattended entry point for launchd/cron.

Run with ``uv run python scripts/auto_run.py`` from the project directory.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from stock_analysis.automation import run_automation, summary_json
from stock_analysis.data import AppConfig


def main() -> int:
    config = AppConfig.load(Path(__file__).resolve().parents[1])
    log_path = config.cache_dir / "auto.log"
    handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    summary = run_automation(config)
    print(summary_json(summary))
    return 1 if summary.failed and not summary.succeeded else 0


if __name__ == "__main__":
    raise SystemExit(main())
