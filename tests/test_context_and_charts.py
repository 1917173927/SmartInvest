from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stock_analysis.charts import render_stock_chart
from stock_analysis.context import _news_items
from stock_analysis.data import Database, DataQuality
from stock_analysis.decision import AnalysisPackage, ValuationRange
from stock_analysis.indicators import add_indicators, macro_assessments, macro_exposures
from stock_analysis.research import ResearchResult


def _price_frame(periods: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=periods)
    close = np.linspace(10, 15, periods)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.linspace(1_000, 2_000, periods),
            "currency": "CNY",
            "quality": "B",
        }
    )


def test_indicators_do_not_change_when_future_rows_are_appended() -> None:
    full = _price_frame(140)
    past = full.iloc[:120].copy()
    past_result = add_indicators(past)
    full_result = add_indicators(full).iloc[:120]
    for column in ("ma20", "ma60", "macd", "macd_signal", "rsi14", "atr14"):
        np.testing.assert_allclose(past_result[column], full_result[column], equal_nan=True)


def test_news_requires_a_parseable_publication_date() -> None:
    valid = pd.DataFrame(
        [
            {
                "新闻标题": "公司发布公告",
                "新闻内容": "带日期的事实",
                "发布时间": "2026-08-20 09:00:00",
                "文章来源": "东方财富",
                "新闻链接": "https://example.test/news",
            },
            {"新闻标题": "无日期内容", "新闻内容": "不得入库", "发布时间": "unknown"},
        ]
    )
    items = _news_items(valid, "CN:601318", date(2026, 8, 1), date(2026, 8, 31))
    assert [item["title"] for item in items] == ["公司发布公告"]
    assert items[0]["source_url"] == "https://example.test/news"


def test_macro_storage_and_scoring_are_point_in_time(tmp_path) -> None:
    with Database(tmp_path / "analysis.sqlite3") as database:
        database.upsert_macro_observations(
            [
                {
                    "series": "CSI300",
                    "observation_date": "2026-08-29",
                    "value": 4000,
                    "unit": "level",
                    "source": "fixture",
                },
                {
                    "series": "CSI300",
                    "observation_date": "2026-08-30",
                    "value": 4040,
                    "unit": "level",
                    "source": "fixture",
                },
                {
                    "series": "CSI300",
                    "observation_date": "2026-09-01",
                    "value": 5000,
                    "unit": "level",
                    "source": "fixture",
                },
            ]
        )
        rows, score = macro_assessments(database, date(2026, 8, 31))
        assert len(rows) == 1
        assert rows[0]["value"] == 4040
        assert score > 0


def test_macro_exposures_are_asset_specific() -> None:
    gold = macro_exposures({"sector": "黄金", "valuation_model": "fund"})
    bank = macro_exposures({"sector": "金融", "valuation_model": "bank"})
    agriculture = macro_exposures({"sector": "农业", "valuation_model": "cyclical"})
    assert gold["GOLD"] > bank.get("GOLD", 0)
    assert agriculture["WTI"] < 0 < bank["WTI"]


def test_static_chart_is_generated(tmp_path) -> None:
    frame = _price_frame()
    package = AnalysisPackage(
        symbol="CN:601318",
        name="中国平安",
        as_of=date(2025, 6, 17),
        current_price=float(frame.iloc[-1]["close"]),
        currency="CNY",
        data_quality=DataQuality.B,
        data_warnings=[],
        forecasts=[],
        research=ResearchResult(
            symbol="CN:601318",
            as_of=date(2025, 6, 17),
            status="unavailable",
            summary="fixture",
        ),
        technical=[],
        quality=[],
        valuation=[],
        valuation_range=ValuationRange(available=False, method="fixture"),
        decisions=[],
    )
    output = render_stock_chart(frame, package, tmp_path / "chart.svg")
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<?xml")
