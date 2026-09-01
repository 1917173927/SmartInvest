from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stock_analysis.automation import (
    AutomationSummary,
    calibration_symbol_order,
    configured_evidence_paths,
    organize_reports,
    render_summary_markdown,
    run_automation,
)
from stock_analysis.data import AppConfig, Bar, Database, DataQuality


def test_unattended_run_uses_cache_without_network(tmp_path) -> None:
    as_of = date(2025, 1, 1)
    config = AppConfig(
        tmp_path,
        {
            "assets": {"CN:601318": {"name": "中国平安", "role": "core"}},
            "automation": {"use_chronos": False, "use_llm": False},
        },
    )
    dates = pd.bdate_range(end=as_of, periods=100)
    prices = np.linspace(50, 55, len(dates))
    with Database(config.db_path) as database:
        database.upsert_bars(
            [
                Bar(
                    symbol="CN:601318",
                    trade_date=timestamp.date(),
                    open=float(price),
                    high=float(price * 1.01),
                    low=float(price * 0.99),
                    close=float(price),
                    volume=1000,
                    currency="CNY",
                    source="fixture",
                    quality=DataQuality.B,
                )
                for timestamp, price in zip(dates, prices, strict=True)
            ]
        )
    progress = []
    summary = run_automation(config, as_of=as_of, progress_callback=progress.append)
    assert summary.succeeded == ["CN:601318"]
    assert not summary.failed
    assert (
        config.reports_dir / "个股" / "CN-601318" / "2025-01-01-CN-601318-中国平安-all.md"
    ).exists()
    assert (config.reports_dir / "最新摘要.md").exists()
    with Database(config.db_path) as database:
        run = database.connection.execute(
            "SELECT status, summary_json FROM automation_runs WHERE id = ?", (summary.run_id,)
        ).fetchone()
        tasks = database.connection.execute(
            "SELECT task, status FROM automation_tasks WHERE run_id = ? ORDER BY sequence",
            (summary.run_id,),
        ).fetchall()
    assert run["status"] == "completed"
    assert summary.run_id in run["summary_json"]
    assert ("market-sync", "skipped") in [(row["task"], row["status"]) for row in tasks]
    assert progress[-1].stage == "组合与报告"
    assert progress[-1].completed == progress[-1].total == 5
    assert any(item.stage == "同步行情" and item.symbol == "CN:601318" for item in progress)
    assert any(item.stage == "个股分析" and item.symbol == "CN:601318" for item in progress)


def test_organize_reports_moves_known_types_without_overwrite(tmp_path) -> None:
    config = AppConfig(tmp_path, {})
    legacy = config.reports_dir / "2026-08-31-CN-601318-all.md"
    legacy.write_text(
        "---\ntype: automated-stock-analysis\nsymbol: CN:601318\n---\n", encoding="utf-8"
    )
    moved = organize_reports(config)
    assert moved == [config.reports_dir / "个股" / "CN-601318" / legacy.name]
    assert not legacy.exists()


def test_configured_evidence_paths_are_scoped_to_symbol(tmp_path) -> None:
    config = AppConfig(tmp_path, {"automation": {"evidence_dirs": ["07-研究资料"]}})
    target = tmp_path / "07-研究资料" / "CN-601318"
    target.mkdir(parents=True)
    (target / "2026-01-01-report.md").write_text("---\ndate: 2026-01-01\n---\n", encoding="utf-8")
    other = tmp_path / "07-研究资料" / "CN-000933"
    other.mkdir(parents=True)
    (other / "other.md").write_text("other", encoding="utf-8")
    paths = configured_evidence_paths(config, "CN:601318")
    assert [path.name for path in paths] == ["2026-01-01-report.md"]


def test_summary_surfaces_action_score_and_confidence(tmp_path) -> None:
    config = AppConfig(tmp_path, {"assets": {"CN:601318": {"name": "中国平安"}}})
    summary = AutomationSummary(as_of=date(2026, 8, 31), symbols=["CN:601318"])
    summary.highlights = {"CN:601318": {"short": "持有/观察"}}
    summary.decision_scores = {"CN:601318": {"short": {"score": 0.25, "confidence": 0.75}}}
    rendered = render_summary_markdown(config, summary)
    assert "持有/观察<br>评分 +0.25 / 置信 75%" in rendered


def test_calibration_symbol_order_rotates_without_losing_symbols() -> None:
    symbols = ["CN:601398", "CN:601318", "CN:000933"]
    first = calibration_symbol_order(symbols, date(2026, 8, 31))
    second = calibration_symbol_order(symbols, date(2026, 9, 1))
    assert sorted(first) == sorted(symbols)
    assert sorted(second) == sorted(symbols)
    assert first != second
    assert calibration_symbol_order([], date(2026, 8, 31)) == []
