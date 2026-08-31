from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from stock_analysis.data import AppConfig, Database, DataQuality
from stock_analysis.forecast import (
    Chronos2Forecaster,
    ForecastBundle,
    ForecastEstimate,
    ModelStatus,
    RandomWalkForecaster,
    evaluate_open_receipts,
    forecast_one,
    probability_fan,
    walk_forward_backtest,
)


def price_frame(length: int = 820) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0002, 0.012, length)
    index = np.cumprod(1 + returns)
    dates = pd.bdate_range("2022-01-03", periods=length)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": index,
            "high": index * 1.01,
            "low": index * 0.99,
            "close": index,
            "volume": 1000 + np.arange(length),
            "currency": "CNY",
            "quality": "B",
            "daily_return": np.r_[0.0, index[1:] / index[:-1] - 1],
            "total_return_index": index,
        }
    )


class FixedChronos:
    name = "chronos-2"

    def predict(self, frame, horizon_days, *, event_rows=None):  # noqa: ANN001
        return ForecastEstimate(
            model=self.name,
            horizon_days=horizon_days,
            q10=-0.05,
            q50=0.02,
            q90=0.09,
            up_probability=0.62,
            annualized_volatility=0.2,
            potential_drawdown=0.08,
        )


def config(tmp_path) -> AppConfig:  # noqa: ANN001
    return AppConfig(
        tmp_path,
        {
            "forecast": {
                "initial_chronos_weight": 0.5,
                "minimum_calibration_samples": 100,
                "disable_after_consecutive_losses": 60,
                "minimum_history_days": 756,
            }
        },
    )


def test_random_walk_is_probabilistic_and_ordered() -> None:
    estimate = RandomWalkForecaster(simulations=1000).predict(price_frame(), 20)
    assert estimate.q10 <= estimate.q50 <= estimate.q90
    assert 0 <= estimate.up_probability <= 1
    assert estimate.potential_drawdown >= 0


def test_chronos_features_use_regular_trading_day_index() -> None:
    features = Chronos2Forecaster._feature_frame(price_frame(), event_rows=None)
    assert pd.infer_freq(features["timestamp"]) == "B"
    assert len(features) == len(price_frame())


def test_initial_ensemble_uses_equal_model_weights(tmp_path) -> None:
    database = Database(tmp_path / "analysis.sqlite3")
    bundle = forecast_one(
        symbol="CN:601318",
        as_of=date(2026, 1, 1),
        horizon_days=20,
        frame=price_frame(),
        data_quality=DataQuality.B,
        database=database,
        config=config(tmp_path),
        chronos_forecaster=FixedChronos(),
    )
    assert bundle.weights == {"random-walk": 0.5, "chronos-2": 0.5}
    assert "chronos-2" in bundle.components
    database.close()


def test_probability_fan_preserves_forecast_anchors() -> None:
    estimates = []
    for days, q10, q50, q90 in ((5, -0.05, 0.01, 0.08), (20, -0.12, 0.04, 0.20)):
        estimate = ForecastEstimate(
            model="fixture",
            horizon_days=days,
            q10=q10,
            q50=q50,
            q90=q90,
            up_probability=0.6,
            annualized_volatility=0.2,
            potential_drawdown=-q10,
        )
        estimates.append(
            ForecastBundle(
                symbol="CN:601318",
                as_of=date(2026, 1, 1),
                due_date=date(2026, 2, 1),
                horizon_days=days,
                ensemble=estimate,
                components={"fixture": estimate},
                weights={"fixture": 1.0},
                status=ModelStatus.EXPERIMENTAL,
                data_quality=DataQuality.B,
            )
        )
    fan = probability_fan(estimates)
    for days, q10, q50, q90 in ((5, -0.05, 0.01, 0.08), (20, -0.12, 0.04, 0.20)):
        row = fan.loc[fan["day"] == days].iloc[0]
        np.testing.assert_allclose([row["q10"], row["q50"], row["q90"]], [q10, q50, q90])
    assert (fan[["q10", "q25", "q50", "q75", "q90"]].diff(axis=1).iloc[:, 1:] >= 0).all().all()


def test_evaluate_open_receipt_records_actual_result(tmp_path) -> None:
    database = Database(tmp_path / "analysis.sqlite3")
    dates = [date(2026, 1, 2) + timedelta(days=index) for index in range(10)]
    for index, day in enumerate(dates):
        database.connection.execute(
            "INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "CN:601318",
                day.isoformat(),
                100 + index,
                101 + index,
                99 + index,
                100 + index,
                1000,
                "CNY",
                "fixture",
                "2026-01-20T00:00:00+00:00",
                "B",
            ),
        )
    database.connection.commit()
    component = {
        "model": "random-walk",
        "horizon_days": 5,
        "q10": -0.1,
        "q50": 0,
        "q90": 0.1,
        "up_probability": 0.5,
        "annualized_volatility": 0.2,
        "potential_drawdown": 0.1,
    }
    database.save_receipt(
        {
            "id": "fc-test",
            "symbol": "CN:601318",
            "created_at": "2026-01-02T00:00:00+00:00",
            "as_of": "2026-01-02",
            "horizon_days": 5,
            "due_date": "2026-01-07",
            "model_status": "degraded",
            "forecast": {
                "ensemble": component,
                "components": {"random-walk": component},
                "weights": {"random-walk": 1.0, "chronos-2": 0.0},
            },
            "decision": {"action": "观察"},
            "evidence": [],
        }
    )
    evaluated = evaluate_open_receipts(database)
    assert len(evaluated) == 1
    row = database.receipts(status="evaluated")[0]
    assert json.loads(row["evaluation_json"])["actual_return"] > 0
    database.close()


def test_walk_forward_uses_only_prior_context(tmp_path) -> None:
    database = Database(tmp_path / "analysis.sqlite3")
    summary = walk_forward_backtest(
        symbol="CN:601318",
        frame=price_frame(),
        horizon_days=20,
        database=database,
        config=config(tmp_path),
        max_windows=3,
    )
    assert summary["windows"] == 3
    assert len(database.receipts(status="evaluated")) == 3
    database.close()
