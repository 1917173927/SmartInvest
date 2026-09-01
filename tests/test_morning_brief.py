from __future__ import annotations

from datetime import date

import pandas as pd
from typer.testing import CliRunner

from stock_analysis.cli import app
from stock_analysis.data import AppConfig, Bar, Database, DataQuality, FundamentalRecord
from stock_analysis.morning import generate_morning_brief

runner = CliRunner()


def test_generate_morning_brief(tmp_path) -> None:
    toml_path = tmp_path / "stock-analysis.toml"
    toml_path.write_text(
        """[assets."CN:601318"]
name = "中国平安"
sector = "金融"
role = "core"
valuation_model = "insurer"
fair_pe = 9.0
fair_pb = 1.10
current_shares = 1000

[investor]
capital_is_surplus = true
uses_leverage = false
""",
        encoding="utf-8",
    )
    db_path = tmp_path / ".stock-analysis" / "analysis.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with Database(db_path) as db:
        dates = pd.bdate_range(end=date(2026, 8, 31), periods=30)
        db.upsert_bars(
            [
                Bar(
                    symbol="CN:601318",
                    trade_date=d.date(),
                    open=50.0,
                    high=52.0,
                    low=49.0,
                    close=51.0,
                    volume=10000,
                    currency="CNY",
                    source="fixture",
                    quality=DataQuality.B,
                )
                for d in dates
            ]
        )
        db.upsert_fundamentals(
            [
                FundamentalRecord(
                    symbol="CN:601318",
                    metric="pe",
                    value=6.5,
                    unit="x",
                    as_of=date(2026, 8, 31),
                    source="fixture",
                    quality="B",
                )
            ]
        )

    config = AppConfig.load(tmp_path)
    brief = generate_morning_brief(
        config, total_capital=100000.0, as_of=date(2026, 8, 31), send_notification=False
    )

    assert len(brief.items) == 1
    assert brief.items[0].name == "中国平安"
    assert brief.report_path is not None
    assert brief.report_path.exists()
    content = brief.report_path.read_text(encoding="utf-8")
    assert "SmartInvest 盘前挂单与执行晨报" in content
    assert "优先减仓/退出复核" in content
    assert "建议卖出" in content
    assert "不生成或保留任何买入挂单" in content
    assert "首笔底仓" not in content


def test_cli_morning_command(tmp_path, monkeypatch) -> None:
    toml_path = tmp_path / "stock-analysis.toml"
    toml_path.write_text(
        """[assets."CN:601318"]
name = "中国平安"
sector = "金融"
role = "core"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("STOCK_ANALYSIS_HOME", str(tmp_path))
    db_path = tmp_path / ".stock-analysis" / "analysis.sqlite3"

    with Database(db_path) as db:
        db.upsert_bars(
            [
                Bar(
                    symbol="CN:601318",
                    trade_date=date(2026, 8, 31),
                    open=50.0,
                    high=52.0,
                    low=49.0,
                    close=51.0,
                    volume=10000,
                    currency="CNY",
                    source="fixture",
                    quality=DataQuality.B,
                )
            ]
        )

    result = runner.invoke(app, ["morning", "--no-notify"], env={"COLUMNS": "160"})
    assert result.exit_code == 0
    assert "盘前挂单与执行晨报已生成" in result.output
    assert "中国平安" in result.output
