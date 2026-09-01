from __future__ import annotations

from datetime import date

from stock_analysis.data import AppConfig, Database, DataQuality, FundamentalRecord
from stock_analysis.decision import (
    ExitStatus,
    Horizon,
    HorizonDecision,
    MetricAssessment,
    ValuationRange,
    build_decisions,
    compute_exit_plan,
    latest_portfolio_snapshot,
    valuation_range,
)
from stock_analysis.forecast import ForecastBundle, ForecastEstimate, ModelStatus
from stock_analysis.research import ResearchResult


def test_valuation_range_uses_configured_multiples(tmp_path) -> None:
    records = {
        "pe": FundamentalRecord(
            symbol="CN:601318",
            metric="pe",
            value=6,
            as_of=date(2026, 1, 1),
            source="fixture",
        ),
        "pb": FundamentalRecord(
            symbol="CN:601318",
            metric="pb",
            value=1,
            as_of=date(2026, 1, 1),
            source="fixture",
        ),
    }
    result = valuation_range(
        current_price=60,
        currency="CNY",
        records=records,
        profile={"fair_pe": 9, "fair_pb": 1.1, "valuation_model": "insurer"},
    )
    assert result.available
    assert result.buy_high < result.fair_high
    assert "PE/PB" in result.method


def test_portfolio_parser_uses_asset_name_mapping(tmp_path) -> None:
    holdings = tmp_path / "01-持仓"
    holdings.mkdir()
    (holdings / "2026-08-27-持仓快照.md").write_text(
        """---
date: 2026-08-27
---
# 持仓
## A 股明细
| 标的 | 持有/可用 | 成本 | 现价 | 市值 |
|---|---:|---:|---:|---:|
| 中国平安 | 300 / 300 股 | 54 | 55 | 16500 |

## 人民币资产视图
已记录人民币资产暂为 **75,000.00** 元：
| 类别 | 金额（CNY） | 占比 |
|---|---:|---:|
| 现金管理 | 20,000.00 | 26.7% |
""",
        encoding="utf-8",
    )
    config = AppConfig(
        tmp_path,
        {
            "assets": {
                "CN:601318": {
                    "name": "中国平安",
                    "sector": "金融",
                    "role": "core",
                }
            }
        },
    )
    snapshot = latest_portfolio_snapshot(config)
    assert snapshot.total_cny_assets == 75000
    assert snapshot.cash_cny == 20000
    assert snapshot.positions[0].symbol == "CN:601318"
    assert snapshot.positions[0].role == "core"


def _exit_decision(action: str, target: float | None) -> HorizonDecision:
    return HorizonDecision(
        horizon=Horizon.MEDIUM,
        score=-0.5,
        confidence=0.8,
        action=action,
        rationale="fixture",
        target_position=target,
    )


def test_exit_plan_reduces_overweight_a_share_by_board_lot(tmp_path) -> None:
    config = AppConfig(
        tmp_path,
        {
            "risk": {"core_position_limit": 0.35},
            "assets": {"CN:601318": {"name": "中国平安", "role": "core"}},
        },
    )
    plan = compute_exit_plan(
        config=config,
        symbol="CN:601318",
        current_price=50,
        decisions=[_exit_decision("停止加仓；复核减仓", 0.35)],
        current_shares=1000,
        total_assets=100000,
        current_weight=0.50,
    )
    assert plan.status is ExitStatus.REDUCE
    assert plan.sell_shares == 300
    assert plan.target_shares == 700
    assert plan.target_weight == 0.35


def test_exit_plan_uses_negative_signal_reduction_target(tmp_path) -> None:
    config = AppConfig(
        tmp_path,
        {
            "risk": {"core_position_limit": 0.35},
            "assets": {"CN:601318": {"name": "中国平安", "role": "core"}},
        },
    )
    plan = compute_exit_plan(
        config=config,
        symbol="CN:601318",
        current_price=50,
        decisions=[_exit_decision("减仓/回避", 0.10)],
        current_shares=400,
        total_assets=100000,
        current_weight=0.20,
    )
    assert plan.status is ExitStatus.REDUCE
    assert plan.sell_shares == 200
    assert plan.target_shares == 200
    assert plan.target_weight == 0.10


def test_exit_plan_full_exit_can_sell_odd_lot(tmp_path) -> None:
    config = AppConfig(
        tmp_path,
        {"assets": {"CN:601318": {"name": "中国平安", "role": "core"}}},
    )
    plan = compute_exit_plan(
        config=config,
        symbol="CN:601318",
        current_price=50,
        decisions=[_exit_decision("回避/重审退出", 0.0)],
        current_shares=350,
        total_assets=100000,
    )
    assert plan.status is ExitStatus.EXIT
    assert plan.sell_shares == 350
    assert plan.target_shares == 0


def test_database_accepts_official_fundamental_quality(tmp_path) -> None:
    database = Database(tmp_path / "analysis.sqlite3")
    database.upsert_fundamentals(
        [
            FundamentalRecord(
                symbol="US:AAPL",
                metric="assets",
                value=100,
                unit="USD",
                as_of=date(2026, 1, 31),
                period_end=date(2025, 12, 31),
                source="sec-edgar",
                quality="A",
            )
        ]
    )
    assert database.latest_fundamentals("US:AAPL", date(2026, 2, 1))["assets"].value == 100
    database.close()


def test_risk_constraints_override_positive_score(tmp_path) -> None:
    config = AppConfig(tmp_path, {"investor": {"capital_is_surplus": True, "uses_leverage": False}})
    forecast = ForecastBundle(
        symbol="CN:601318",
        as_of=date(2026, 1, 1),
        due_date=date(2026, 2, 1),
        horizon_days=20,
        ensemble=ForecastEstimate(
            model="ensemble",
            horizon_days=20,
            q10=-0.30,
            q50=0.10,
            q90=0.35,
            up_probability=0.70,
            annualized_volatility=0.40,
            potential_drawdown=0.30,
        ),
        components={},
        weights={"random-walk": 1.0},
        status=ModelStatus.DEGRADED,
        data_quality=DataQuality.B,
    )
    metrics = [MetricAssessment(name="x", score=0.8, available=True, value=1, explanation="")]
    research = ResearchResult(
        symbol="CN:601318", as_of=date(2026, 1, 1), status="degraded", summary=""
    )
    value_range = ValuationRange(available=False, currency="CNY", method="fixture")
    decisions = build_decisions(
        config=config,
        forecasts=[forecast],
        research=research,
        technical=metrics,
        quality=metrics,
        valuation=metrics,
        value_range=value_range,
        current_price=55,
        data_quality=DataQuality.B,
        current_weight=0.10,
        role="core",
    )
    assert decisions[0].action == "观察；潜在回撤超预算"

    decisions = build_decisions(
        config=config,
        forecasts=[],
        research=research,
        technical=metrics,
        quality=metrics,
        valuation=metrics,
        value_range=value_range,
        current_price=55,
        data_quality=DataQuality.B,
        current_weight=0.40,
        role="core",
    )
    assert decisions[0].action == "停止加仓；复核减仓"
