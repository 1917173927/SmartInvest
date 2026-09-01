from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from stock_analysis.automation import AutomationSummary
from stock_analysis.cli import _auto_interactive_enabled, _quote_freshness, app
from stock_analysis.data import Bar, Database, DataQuality, FundamentalRecord, MarketQuote

runner = CliRunner()


def test_quote_freshness_rejects_stale_public_snapshot() -> None:
    fresh, warning = _quote_freshness(
        datetime.now(UTC) - timedelta(minutes=16), max_age=timedelta(minutes=15)
    )

    assert not fresh
    assert warning is not None and "超过 15 分钟" in warning


def test_cli_auto_defaults_to_menu_only_in_interactive_terminal() -> None:
    assert _auto_interactive_enabled(None, has_symbols=False, verbose=False, is_terminal=True)
    assert not _auto_interactive_enabled(None, has_symbols=False, verbose=False, is_terminal=False)
    assert not _auto_interactive_enabled(None, has_symbols=True, verbose=False, is_terminal=True)
    assert not _auto_interactive_enabled(None, has_symbols=False, verbose=True, is_terminal=True)
    assert not _auto_interactive_enabled(False, has_symbols=False, verbose=False, is_terminal=True)
    assert _auto_interactive_enabled(True, has_symbols=False, verbose=False, is_terminal=False)


def test_cli_auto_interactive_selects_symbols_and_quick_mode(tmp_path, monkeypatch) -> None:
    (tmp_path / "stock-analysis.toml").write_text(
        """[assets."CN:601398"]
name = "工商银行"

[assets."CN:601318"]
name = "中国平安"

[assets."CN:000933"]
name = "神火股份"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("STOCK_ANALYSIS_HOME", str(tmp_path))
    captured = {}

    def fake_run(_config, **kwargs):
        captured.update(kwargs)
        selected = kwargs["symbols"]
        return AutomationSummary(
            as_of=date(2026, 9, 1),
            symbols=selected,
            succeeded=list(selected),
        )

    monkeypatch.setattr("stock_analysis.cli.run_automation", fake_run)

    result = runner.invoke(
        app,
        ["auto", "--interactive", "--no-progress"],
        input="2\n1,3\n2\n",
        env={"COLUMNS": "160"},
    )

    assert result.exit_code == 0
    assert captured["symbols"] == ["CN:601398", "CN:000933"]
    assert captured["use_llm"] is False
    assert captured["use_chronos"] is False
    assert "自动分析完成：成功 2" in result.output


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


def test_cli_size_command(tmp_path, monkeypatch) -> None:
    toml_path = tmp_path / "stock-analysis.toml"
    toml_path.write_text(
        """[assets."CN:601318"]
name = "中国平安"
sector = "金融"
role = "core"
valuation_model = "insurer"
fair_pe = 9.0
fair_pb = 1.10
current_shares = 300

[portfolio]
cn_account_assets = 51546.80
cn_account_assets_as_of = "2026-09-01"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("STOCK_ANALYSIS_HOME", str(tmp_path))
    monkeypatch.setattr(
        "stock_analysis.cli.fetch_latest_quote",
        lambda _instrument: (
            MarketQuote(
                symbol="CN:601318",
                price=57.10,
                currency="CNY",
                source="fixture-quote",
            ),
            [],
        ),
    )
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
                    close=55.0,
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
        ["size", "CN:601318", "--target-weight", "0.20"],
        env={"COLUMNS": "160"},
    )
    assert result.exit_code == 0
    assert "实盘阶梯建仓测算" in result.output
    assert "首笔底仓" in result.output
    assert "建议手数" in result.output
    assert "逻辑失效与止损参考线" in result.output
    assert "盘中执行价: 57.10 CNY" in result.output
    assert "账户总资产: 51,546.80 CNY" in result.output
    assert "当前持仓: 300 股" in result.output
    assert "当前动作：不新增买入" in result.output

    manual_result = runner.invoke(
        app,
        ["size", "CN:601318", "--price", "57.10", "--target-weight", "0.20"],
        env={"COLUMNS": "160"},
    )
    assert manual_result.exit_code == 0
    assert "人工输入价格（未由系统验证，仅供复现/测算）" in manual_result.output
    assert "人工输入价格不是系统获取的实时数据" in manual_result.output
    assert "当前动作：暂停下单" in manual_result.output


def test_cli_compare_command(tmp_path, monkeypatch) -> None:
    toml_path = tmp_path / "stock-analysis.toml"
    toml_path.write_text(
        """[assets."CN:601318"]
name = "中国平安"
sector = "金融"
role = "core"

[assets."CN:600519"]
name = "贵州茅台"
sector = "消费"
role = "core"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("STOCK_ANALYSIS_HOME", str(tmp_path))
    db_path = tmp_path / ".stock-analysis" / "analysis.sqlite3"

    with Database(db_path) as db:
        for sym, price in [("CN:601318", 55.0), ("CN:600519", 1500.0)]:
            db.upsert_bars(
                [
                    Bar(
                        symbol=sym,
                        trade_date=date(2026, 8, 31),
                        open=price,
                        high=price * 1.01,
                        low=price * 0.99,
                        close=price,
                        volume=10000,
                        currency="CNY",
                        source="fixture",
                        quality=DataQuality.B,
                    )
                ]
            )

    result = runner.invoke(
        app,
        ["compare", "CN:601318", "CN:600519"],
        env={"COLUMNS": "160"},
    )
    assert result.exit_code == 0
    assert "跨标的多维优选与比对矩阵" in result.output
    assert "中国平安" in result.output
    assert "贵州茅台" in result.output
