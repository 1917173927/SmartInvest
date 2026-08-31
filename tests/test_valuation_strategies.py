from __future__ import annotations

from datetime import date

from stock_analysis.data import FundamentalRecord
from stock_analysis.decision import (
    BankValuationStrategy,
    CyclicalValuationStrategy,
    FundValuationStrategy,
    GenericValuationStrategy,
    InsurerValuationStrategy,
    get_valuation_strategy,
    valuation_assessments,
    valuation_range,
)


def test_valuation_strategies_registry() -> None:
    assert isinstance(get_valuation_strategy("bank"), BankValuationStrategy)
    assert isinstance(get_valuation_strategy("insurer"), InsurerValuationStrategy)
    assert isinstance(get_valuation_strategy("cyclical"), CyclicalValuationStrategy)
    assert isinstance(get_valuation_strategy("fund"), FundValuationStrategy)
    assert isinstance(get_valuation_strategy("generic"), GenericValuationStrategy)
    assert isinstance(get_valuation_strategy("unknown_model"), GenericValuationStrategy)


def test_bank_valuation_multipliers() -> None:
    records = {
        "pe": FundamentalRecord(
            symbol="CN:601398", metric="pe", value=5.0, as_of=date(2026, 1, 1), source="t"
        ),
        "pb": FundamentalRecord(
            symbol="CN:601398", metric="pb", value=0.6, as_of=date(2026, 1, 1), source="t"
        ),
        "dividend_yield": FundamentalRecord(
            symbol="CN:601398",
            metric="dividend_yield",
            value=0.06,
            as_of=date(2026, 1, 1),
            source="t",
        ),
    }
    profile = {"valuation_model": "bank", "fair_pe": 7.0, "fair_pb": 0.75}
    assessments = valuation_assessments(records, profile)
    pe_metric = next(m for m in assessments if m.name == "PE")
    pb_metric = next(m for m in assessments if m.name == "PB")
    assert pe_metric.available
    assert pb_metric.available
    assert pb_metric.score > 0

    v_range = valuation_range(
        current_price=6.0,
        currency="CNY",
        records=records,
        profile=profile,
    )
    assert v_range.available
    assert v_range.buy_low is not None
    assert v_range.buy_high is not None
    assert v_range.fair_low is not None
    assert v_range.fair_high is not None
    assert v_range.buy_high <= v_range.fair_low


def test_fund_valuation_strategy() -> None:
    records = {
        "pe": FundamentalRecord(
            symbol="CNFUND:000217", metric="pe", value=0.0, as_of=date(2026, 1, 1), source="t"
        ),
    }
    profile = {"valuation_model": "fund"}
    assessments = valuation_assessments(records, profile)
    pe_metric = next(m for m in assessments if m.name == "PE")
    assert "公募/ETF" in pe_metric.explanation
