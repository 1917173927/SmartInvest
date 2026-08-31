from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from stock_analysis.cli import app
from stock_analysis.data import Bar, Database, DataQuality, FundamentalRecord

runner = CliRunner()


def test_cli_add_command_dry_run(tmp_path, monkeypatch) -> None:
    toml_path = tmp_path / "stock-analysis.toml"
    toml_path.write_text('[assets."CN:601318"]\nname = "中国平安"\n', encoding="utf-8")
    monkeypatch.setenv("STOCK_ANALYSIS_HOME", str(tmp_path))

    result = runner.invoke(
        app,
        [
            "add",
            "CN:600519",
            "--name",
            "贵州茅台",
            "--sector",
            "消费",
            "--role",
            "core",
            "--valuation-model",
            "generic",
            "--fair-pe",
            "25.0",
            "--fair-pb",
            "6.0",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "CN:600519" in result.output
    assert "贵州茅台" in result.output
    assert "Dry Run" in result.output


def test_cli_scenario_command(tmp_path, monkeypatch) -> None:
    toml_path = tmp_path / "stock-analysis.toml"
    toml_path.write_text(
        """[assets."CN:601318"]
name = "中国平安"
sector = "金融"
role = "core"
valuation_model = "insurer"
fair_pe = 9.0
fair_pb = 1.10
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
                    open=55.0,
                    high=56.0,
                    low=54.5,
                    close=55.5,
                    volume=10000,
                    currency="CNY",
                    source="fixture",
                    quality=DataQuality.B,
                )
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
                ),
                FundamentalRecord(
                    symbol="CN:601318",
                    metric="pb",
                    value=0.95,
                    unit="x",
                    as_of=date(2026, 8, 31),
                    source="fixture",
                    quality="B",
                ),
            ]
        )

    result = runner.invoke(
        app,
        [
            "scenario",
            "CN:601318",
            "--eps-growth-delta",
            "-0.10",
            "--pe-delta",
            "-1.0",
            "--margin-delta",
            "0.05",
        ],
    )
    assert result.exit_code == 0
    assert "What-If" in result.output
    assert "合理价值区间" in result.output
    assert "中国平安" in result.output


def test_cli_dash_command(tmp_path, monkeypatch) -> None:
    toml_path = tmp_path / "stock-analysis.toml"
    toml_path.write_text(
        """[assets."CN:601318"]
name = "中国平安"
sector = "金融"
role = "core"
valuation_model = "insurer"
fair_pe = 9.0
fair_pb = 1.10
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("STOCK_ANALYSIS_HOME", str(tmp_path))
    db_path = tmp_path / ".stock-analysis" / "analysis.sqlite3"

    dates = pd.bdate_range(end=date(2026, 8, 31), periods=50)
    prices = np.linspace(50, 55, len(dates))
    with Database(db_path) as db:
        db.upsert_bars(
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

    result = runner.invoke(app, ["dash"], env={"COLUMNS": "160"})
    assert result.exit_code == 0
    assert "实时决策看板" in result.output
    assert "中国平" in result.output
