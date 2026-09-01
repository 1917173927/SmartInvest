from __future__ import annotations

import os
import sqlite3
import sys
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from stock_analysis.data import (
    AkShareProvider,
    AppConfig,
    Bar,
    CorporateAction,
    Database,
    Instrument,
    Market,
    MarketQuote,
    YFinanceProvider,
    _normalize_provider_metric,
    coverage_warnings,
    fetch_latest_quote,
    quality_summary,
    sync_symbol,
    total_return_frame,
)


def test_project_dotenv_loads_without_overriding_environment(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "STOCK_ANALYSIS_TEST_KEY=from-file\nSTOCK_ANALYSIS_QUOTED='quoted'\n", encoding="utf-8"
    )
    monkeypatch.setenv("STOCK_ANALYSIS_TEST_KEY", "from-env")
    AppConfig.load(tmp_path)
    assert os.environ["STOCK_ANALYSIS_TEST_KEY"] == "from-env"
    assert os.environ["STOCK_ANALYSIS_QUOTED"] == "quoted"


def test_symbol_normalization() -> None:
    assert Instrument.parse("601318").canonical == "CN:601318"
    assert Instrument.parse("sh.601398").canonical == "CN:601398"
    assert Instrument.parse("0700.HK").canonical == "HK:00700"
    assert Instrument.parse("HK:700").canonical == "HK:00700"
    assert Instrument.parse("AAPL").canonical == "US:AAPL"
    assert Instrument.parse("CNFUND:000217").market is Market.CNFUND


def test_provider_percentages_are_normalized() -> None:
    assert _normalize_provider_metric("dividend_yield", 4.83) == 0.0483
    assert _normalize_provider_metric("dividend_yield", 0.0483) == 0.0483
    assert _normalize_provider_metric("debt_to_equity", 120.0) == 1.2


def test_yfinance_current_snapshot_is_not_backdated(tmp_path, monkeypatch) -> None:
    class FakeTicker:
        info = {"trailingPE": 20.0, "returnOnEquity": 0.15}

    class FakeYFinance:
        @staticmethod
        def Ticker(_symbol):
            return FakeTicker()

    observed_at = datetime(2026, 9, 1, 4, 0, tzinfo=UTC)
    monkeypatch.setitem(sys.modules, "yfinance", FakeYFinance())
    monkeypatch.setattr(YFinanceProvider, "available", staticmethod(lambda: True))
    monkeypatch.setattr("stock_analysis.data.utc_now", lambda: observed_at)

    records = YFinanceProvider().fetch_fundamentals(Instrument.parse("US:AAPL"), date(2020, 1, 1))

    assert records
    assert {record.as_of for record in records} == {observed_at.date()}
    with Database(tmp_path / "analysis.sqlite3") as database:
        database.upsert_fundamentals(records)
        assert database.latest_fundamentals("US:AAPL", date(2020, 1, 1)) == {}
        assert set(database.latest_fundamentals("US:AAPL", observed_at.date())) == {"pe", "roe"}


def test_database_initializes_schema_version_and_busy_timeout(tmp_path) -> None:
    with Database(tmp_path / "analysis.sqlite3") as database:
        timeout = database.connection.execute("PRAGMA busy_timeout").fetchone()[0]
        assert database.schema_version == 1
        assert timeout == 30000


def test_database_upgrades_unversioned_database_without_losing_tables(tmp_path) -> None:
    path = tmp_path / "analysis.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE legacy_marker (value TEXT NOT NULL)")
    connection.execute("INSERT INTO legacy_marker VALUES ('preserved')")
    connection.commit()
    connection.close()

    with Database(path) as database:
        marker = database.connection.execute("SELECT value FROM legacy_marker").fetchone()[0]
        assert database.schema_version == 1
        assert marker == "preserved"


def test_database_rejects_newer_schema_version(tmp_path) -> None:
    path = tmp_path / "analysis.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 999")
    connection.close()

    with pytest.raises(RuntimeError, match="高于当前程序支持"):
        Database(path)


def test_akshare_uses_tencent_when_eastmoney_fails(monkeypatch) -> None:
    class FakeAkShare:
        @staticmethod
        def stock_zh_a_hist(**_kwargs):
            raise RuntimeError("eastmoney unavailable")

        @staticmethod
        def stock_zh_a_hist_tx(**_kwargs):
            return pd.DataFrame(
                [
                    {
                        "date": "2026-01-02",
                        "open": 10,
                        "close": 10.5,
                        "high": 11,
                        "low": 9.5,
                        "amount": 100,
                    }
                ]
            )

    monkeypatch.setitem(sys.modules, "akshare", FakeAkShare())
    monkeypatch.setattr(AkShareProvider, "available", staticmethod(lambda: True))
    bars = AkShareProvider().fetch_bars(
        Instrument.parse("CN:601318"), date(2026, 1, 1), date(2026, 1, 3)
    )
    assert bars[0].source == "akshare-tencent"


def test_akshare_fetches_single_security_quote(monkeypatch) -> None:
    class FakeAkShare:
        @staticmethod
        def stock_bid_ask_em(**_kwargs):
            return pd.DataFrame([{"item": "最新", "value": 57.10}])

    monkeypatch.setitem(sys.modules, "akshare", FakeAkShare())
    monkeypatch.setattr(AkShareProvider, "available", staticmethod(lambda: True))

    quote = AkShareProvider().fetch_quote(Instrument.parse("CN:601318"))

    assert quote.price == 57.10
    assert quote.source == "akshare-eastmoney-quote"


def test_latest_quote_falls_back_to_next_provider() -> None:
    class FailedProvider:
        name = "failed"

        @staticmethod
        def fetch_quote(_instrument):
            raise RuntimeError("unavailable")

    class WorkingProvider:
        name = "working"

        @staticmethod
        def fetch_quote(instrument):
            return MarketQuote(
                symbol=instrument.canonical,
                price=57.10,
                currency="CNY",
                source="working-quote",
            )

    quote, warnings = fetch_latest_quote(
        Instrument.parse("CN:601318"), [FailedProvider(), WorkingProvider()]
    )

    assert quote is not None and quote.price == 57.10
    assert warnings == ["failed 实时报价不可用：RuntimeError；已尝试下一数据源"]


def test_latest_quote_explains_json_provider_failure() -> None:
    class JSONDecodeError(ValueError):
        pass

    class FailedProvider:
        name = "akshare"

        @staticmethod
        def fetch_quote(_instrument):
            raise JSONDecodeError("blocked")

    quote, warnings = fetch_latest_quote(Instrument.parse("CN:601318"), [FailedProvider()])

    assert quote is None
    assert warnings == [
        "akshare 实时报价不可用：接口返回非 JSON，可能被限流或网络拦截；已尝试下一数据源"
    ]


def test_akshare_converts_per_ten_share_corporate_actions(monkeypatch) -> None:
    class FakeAkShare:
        @staticmethod
        def stock_fhps_detail_em(**_kwargs):
            return pd.DataFrame(
                [
                    {
                        "除权除息日": "2005-09-09",
                        "送转股份-送股比例": 0,
                        "送转股份-转股比例": 10,
                        "现金分红-现金分红比例": 15,
                    },
                    {
                        "除权除息日": "2009-06-19",
                        "送转股份-送股比例": 5,
                        "送转股份-转股比例": 0,
                        "现金分红-现金分红比例": 2,
                    },
                ]
            )

    monkeypatch.setitem(sys.modules, "akshare", FakeAkShare())
    monkeypatch.setattr(AkShareProvider, "available", staticmethod(lambda: True))
    actions = AkShareProvider().fetch_actions(
        Instrument.parse("CN:000933"), date(2000, 1, 1), date(2010, 1, 1)
    )
    assert [(item.action_date, item.split_ratio, item.dividend) for item in actions] == [
        (date(2005, 9, 9), 2.0, 1.5),
        (date(2009, 6, 19), 1.5, 0.2),
    ]


def test_raw_prices_and_actions_produce_point_in_time_returns() -> None:
    bars = pd.DataFrame(
        [
            {"trade_date": "2026-01-01", "close": 100.0, "volume": 1},
            {"trade_date": "2026-01-02", "close": 50.0, "volume": 1},
            {"trade_date": "2026-01-03", "close": 49.0, "volume": 1},
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "action_date": "2026-01-02",
                "dividend": 0.0,
                "split_ratio": 2.0,
            },
            {
                "action_date": "2026-01-03",
                "dividend": 1.0,
                "split_ratio": 1.0,
            },
        ]
    )
    result, warnings = total_return_frame(bars, actions)
    assert result.iloc[1]["daily_return"] == 0
    assert result.iloc[2]["daily_return"] == 0
    assert result.iloc[-1]["total_return_index"] == 1
    assert result.iloc[1]["return_anomaly_status"] == "corporate-action-adjusted"
    assert warnings == []


def test_unexplained_large_return_downgrades_data_quality() -> None:
    frame = pd.DataFrame(
        [
            {"trade_date": "2026-01-01", "close": 100, "quality": "B"},
            {"trade_date": "2026-01-02", "close": 50, "quality": "B"},
        ]
    )
    returns, _ = total_return_frame(frame)
    quality, warnings = quality_summary(returns, date(2026, 1, 2))
    assert quality.value == "C"
    assert any("暂停模型动作" in warning for warning in warnings)


def test_database_uses_one_action_source_per_date(tmp_path) -> None:
    with Database(tmp_path / "analysis.sqlite3") as database:
        database.upsert_actions(
            [
                CorporateAction(
                    symbol="CN:000933",
                    action_date=date(2005, 9, 9),
                    dividend=1.5,
                    split_ratio=2.0,
                    source="akshare-corporate-actions",
                ),
                CorporateAction(
                    symbol="CN:000933",
                    action_date=date(2005, 9, 9),
                    dividend=1.5,
                    split_ratio=2.0,
                    source="yfinance",
                ),
            ]
        )
        actions = database.load_actions("CN:000933")
    assert len(actions) == 1
    assert actions.iloc[0]["source"] == "akshare-corporate-actions"


def test_sync_uses_action_fallback_independent_of_price_provider(tmp_path) -> None:
    class PriceProvider:
        name = "price-source"

        def supports(self, _instrument):
            return True

        def fetch_bars(self, instrument, _start, _end):
            return [
                Bar(
                    symbol=instrument.canonical,
                    trade_date=date(2026, 1, 1),
                    open=10,
                    high=10,
                    low=10,
                    close=10,
                    volume=1,
                    currency="CNY",
                    source=self.name,
                )
            ]

        def fetch_actions(self, *_args):
            raise RuntimeError("action endpoint down")

        def fetch_fundamentals(self, *_args):
            return []

    class ActionFallback(PriceProvider):
        name = "action-fallback"

        def fetch_bars(self, *_args):
            raise RuntimeError("must not replace price source")

        def fetch_actions(self, instrument, *_args):
            return [
                CorporateAction(
                    symbol=instrument.canonical,
                    action_date=date(2026, 1, 1),
                    dividend=0.1,
                    split_ratio=1.0,
                    source=self.name,
                )
            ]

    with Database(tmp_path / "analysis.sqlite3") as database:
        result = sync_symbol(
            database,
            "CN:601318",
            start=date(2026, 1, 1),
            end=date(2026, 1, 2),
            providers=[PriceProvider(), ActionFallback()],
        )
        actions = database.load_actions("CN:601318")
    assert result.provider == "price-source"
    assert result.actions == 1
    assert actions.iloc[0]["source"] == "action-fallback"


def test_sync_tries_backup_when_primary_returns_empty(tmp_path) -> None:
    class EmptyProvider:
        name = "empty-primary"

        def supports(self, _instrument):
            return True

        def fetch_bars(self, *_args):
            return []

        def fetch_actions(self, *_args):
            return []

        def fetch_fundamentals(self, *_args):
            return []

    class BackupProvider(EmptyProvider):
        name = "backup"

        def fetch_bars(self, instrument, *_args):
            return [
                Bar(
                    symbol=instrument.canonical,
                    trade_date=date(2026, 1, 2),
                    open=10,
                    high=11,
                    low=9,
                    close=10,
                    volume=100,
                    currency="CNY",
                    source=self.name,
                )
            ]

    with Database(tmp_path / "analysis.sqlite3") as database:
        result = sync_symbol(
            database,
            "CN:601318",
            start=date(2026, 1, 1),
            end=date(2026, 1, 3),
            providers=[EmptyProvider(), BackupProvider()],
        )
        cached = database.load_bars("CN:601318")
    assert result.provider == "backup"
    assert len(cached) == 1
    assert any("未返回行情" in warning for warning in result.warnings)


def test_database_filters_future_bars_and_documents(tmp_path) -> None:
    database = Database(tmp_path / "analysis.sqlite3")
    database.upsert_bars(
        [
            Bar(
                symbol="CN:601318",
                trade_date=date(2026, 1, 2),
                open=10,
                high=11,
                low=9,
                close=10,
                volume=100,
                currency="CNY",
                source="fixture",
            ),
            Bar(
                symbol="CN:601318",
                trade_date=date(2026, 1, 5),
                open=11,
                high=12,
                low=10,
                close=11,
                volume=100,
                currency="CNY",
                source="fixture",
            ),
        ]
    )
    filtered = database.load_bars("CN:601318", date(2026, 1, 2))
    assert len(filtered) == 1
    database.add_document(
        symbol="CN:601318",
        title="已发布报告",
        body="业绩和分红信息",
        source_url="https://example.test/old",
        published_at=date(2026, 1, 2),
    )
    database.add_document(
        symbol="CN:601318",
        title="未来报告",
        body="未来业绩信息",
        source_url="https://example.test/future",
        published_at=date(2026, 2, 2),
    )
    rows = database.search_documents("CN:601318", "业绩", date(2026, 1, 15))
    assert [row["title"] for row in rows] == ["已发布报告"]
    broader = database.search_documents("CN:601318", "不存在 业绩", date(2026, 1, 15))
    assert [row["title"] for row in broader] == ["已发布报告"]
    database.close()


def test_database_selects_preferred_source_per_trade_date(tmp_path) -> None:
    database = Database(tmp_path / "analysis.sqlite3")
    common = dict(
        symbol="CN:601318",
        trade_date=date(2026, 1, 2),
        open=10,
        high=11,
        low=9,
        volume=100,
        currency="CNY",
    )
    database.upsert_bars(
        [
            Bar(**common, close=99, source="yfinance"),
            Bar(**common, close=10, source="akshare"),
        ]
    )
    frame = database.load_bars("CN:601318")
    assert len(frame) == 1
    assert frame.iloc[0]["source"] == "akshare"
    assert frame.iloc[0]["close"] == 10
    database.close()


def test_database_stores_actions_without_rewriting_bars(tmp_path) -> None:
    database = Database(tmp_path / "analysis.sqlite3")
    database.upsert_actions(
        [
            CorporateAction(
                symbol="US:AAPL",
                action_date=date(2020, 8, 31),
                split_ratio=4,
                source="fixture",
            )
        ]
    )
    actions = database.load_actions("US:AAPL")
    assert actions.iloc[0]["split_ratio"] == 4
    database.close()


def test_automation_run_start_is_exclusive(tmp_path) -> None:
    with Database(tmp_path / "analysis.sqlite3") as database:
        database.start_automation_run("run-1", date(2026, 1, 2), ["CN:601318"])
        try:
            database.start_automation_run("run-2", date(2026, 1, 2), ["CN:601318"])
        except RuntimeError as exc:
            assert "run-1" in str(exc)
        else:
            raise AssertionError("并行自动任务应被拒绝")


def test_coverage_does_not_resync_for_old_suspension_gap() -> None:
    recent = pd.bdate_range("2025-09-01", "2026-08-31")
    frame = pd.DataFrame(
        {"trade_date": [pd.Timestamp("2010-01-01"), pd.Timestamp("2010-03-01"), *recent]}
    )
    warnings = coverage_warnings(frame, as_of=date(2026, 8, 31))
    assert not any("日期缺口" in warning for warning in warnings)


def test_stale_automation_run_is_recovered(tmp_path) -> None:
    database = Database(tmp_path / "analysis.sqlite3")
    database.start_automation_run("old-run", date(2026, 1, 1), ["CN:601318"])
    database.connection.execute(
        "UPDATE automation_runs SET started_at = ? WHERE id = 'old-run'",
        ((datetime.now(tz=UTC) - timedelta(hours=1)).isoformat(),),
    )
    database.connection.commit()
    recovered = database.recover_stale_automation_runs(datetime.now(tz=UTC) - timedelta(minutes=20))
    row = database.connection.execute(
        "SELECT status, finished_at FROM automation_runs WHERE id = 'old-run'"
    ).fetchone()
    assert recovered == 1
    assert row["status"] == "interrupted"
    assert row["finished_at"]
    database.close()
