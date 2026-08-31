from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stock_analysis.data import DataQuality, FundamentalRecord
from stock_analysis.decision import quality_assessments, technical_assessments
from stock_analysis.indicators import (
    garman_klass_volatility,
    trend_smoothness_ratio,
    yang_zhang_volatility,
)


def _sample_ohlc_frame(periods: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range(end=date(2026, 8, 31), periods=periods)
    base_price = 100.0
    records = []
    current = base_price
    for d in dates:
        open_p = current * (1.0 + np.random.normal(0, 0.005))
        high_p = max(open_p, current) * (1.0 + abs(np.random.normal(0, 0.01)))
        low_p = min(open_p, current) * (1.0 - abs(np.random.normal(0, 0.01)))
        close_p = (open_p + high_p + low_p) / 3.0
        current = close_p
        records.append(
            {
                "trade_date": d.date(),
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": 10000.0,
                "currency": "CNY",
                "source": "fixture",
                "quality": DataQuality.B,
            }
        )
    return pd.DataFrame(records)


def test_yang_zhang_volatility() -> None:
    frame = _sample_ohlc_frame(120)
    yz = yang_zhang_volatility(frame, window=20)
    assert len(yz) == len(frame)
    assert not yz.isna().any()
    # Volatility should be positive and realistic (e.g. 5% to 80%)
    valid_tail = yz.iloc[-30:]
    assert (valid_tail > 0.0).all()
    assert (valid_tail < 1.5).all()


def test_garman_klass_volatility() -> None:
    frame = _sample_ohlc_frame(120)
    gk = garman_klass_volatility(frame, window=20)
    assert len(gk) == len(frame)
    assert not gk.isna().any()
    valid_tail = gk.iloc[-30:]
    assert (valid_tail > 0.0).all()
    assert (valid_tail < 1.5).all()


def test_trend_smoothness_ratio() -> None:
    frame = _sample_ohlc_frame(120)
    ts = trend_smoothness_ratio(frame, window=60)
    assert len(ts) == len(frame)
    assert not ts.isna().any()


def test_qmj_quality_assessments() -> None:
    records = {
        "roe": FundamentalRecord(
            symbol="CN:601318",
            metric="roe",
            value=0.15,
            unit="ratio",
            as_of=date(2026, 8, 31),
            source="fixture",
            quality="B",
        ),
        "net_income": FundamentalRecord(
            symbol="CN:601318",
            metric="net_income",
            value=100000000.0,
            unit="CNY",
            as_of=date(2026, 8, 31),
            source="fixture",
            quality="B",
        ),
        "operating_cash_flow": FundamentalRecord(
            symbol="CN:601318",
            metric="operating_cash_flow",
            value=120000000.0,
            unit="CNY",
            as_of=date(2026, 8, 31),
            source="fixture",
            quality="B",
        ),
        "assets": FundamentalRecord(
            symbol="CN:601318",
            metric="assets",
            value=1000000000.0,
            unit="CNY",
            as_of=date(2026, 8, 31),
            source="fixture",
            quality="B",
        ),
        "liabilities": FundamentalRecord(
            symbol="CN:601318",
            metric="liabilities",
            value=500000000.0,
            unit="CNY",
            as_of=date(2026, 8, 31),
            source="fixture",
            quality="B",
        ),
        "dividend_yield": FundamentalRecord(
            symbol="CN:601318",
            metric="dividend_yield",
            value=0.045,
            unit="ratio",
            as_of=date(2026, 8, 31),
            source="fixture",
            quality="B",
        ),
    }

    assessments = quality_assessments(records)
    metric_names = [a.name for a in assessments]
    assert "ROE" in metric_names
    assert "现金流/利润" in metric_names
    assert "QMJ复合质量" in metric_names

    qmj = next(a for a in assessments if a.name == "QMJ复合质量")
    assert qmj.score > 0.3  # Healthy company gets strong positive QMJ score


def test_technical_assessments_with_yang_zhang() -> None:
    frame = _sample_ohlc_frame(150)
    tech = technical_assessments(frame)
    names = [m.name for m in tech]
    assert "Yang-Zhang波动率" in names
    assert "特质趋势平滑度" in names
