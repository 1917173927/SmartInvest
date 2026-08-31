from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from importlib.util import find_spec
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from stock_analysis.data import (
    AppConfig,
    Database,
    DataQuality,
    next_trading_date,
    total_return_frame,
    utc_now,
)

SHORT_HORIZONS = (5, 10, 20)
MEDIUM_HORIZONS = (60, 120)
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
Z_90 = 1.2815515655446004
Z_75 = 0.6744897501960817


class ModelStatus(StrEnum):
    ACTIVE = "active"
    EXPERIMENTAL = "experimental"
    DEGRADED = "degraded"
    DISABLED = "disabled"


class ForecastEstimate(BaseModel):
    model: str
    horizon_days: int
    q10: float
    q50: float
    q90: float
    up_probability: float
    annualized_volatility: float
    potential_drawdown: float

    @model_validator(mode="after")
    def validate_quantiles(self) -> ForecastEstimate:
        values = sorted((self.q10, self.q50, self.q90))
        self.q10, self.q50, self.q90 = values
        self.up_probability = min(max(float(self.up_probability), 0.0), 1.0)
        self.potential_drawdown = max(float(self.potential_drawdown), 0.0)
        return self


class ForecastBundle(BaseModel):
    symbol: str
    as_of: date
    due_date: date
    horizon_days: int
    ensemble: ForecastEstimate
    components: dict[str, ForecastEstimate]
    weights: dict[str, float]
    status: ModelStatus
    data_quality: DataQuality
    warnings: list[str] = Field(default_factory=list)
    calibration_samples: int = 0
    calibration_target: int = 100


def probability_fan(forecasts: list[ForecastBundle]) -> pd.DataFrame:
    """Interpolate nested probability bands across forecast horizon anchors.

    Only the configured forecast horizons are model outputs. Intermediate days
    are explicitly a visualization interpolation in log-return space.
    """
    by_horizon = {item.horizon_days: item for item in forecasts}
    if not by_horizon:
        return pd.DataFrame(columns=["day", "q10", "q25", "q50", "q75", "q90"])
    anchors: list[dict[str, float]] = [
        {"day": 0.0, "q10": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q90": 0.0}
    ]
    for horizon, bundle in sorted(by_horizon.items()):
        estimate = bundle.ensemble
        q10_log = math.log1p(max(estimate.q10, -0.999999))
        q50_log = math.log1p(max(estimate.q50, -0.999999))
        q90_log = math.log1p(max(estimate.q90, -0.999999))
        sigma = max((q90_log - q10_log) / (2 * Z_90), 1e-8)
        q25 = math.expm1(q50_log - Z_75 * sigma)
        q75 = math.expm1(q50_log + Z_75 * sigma)
        ordered = sorted((estimate.q10, q25, estimate.q50, q75, estimate.q90))
        anchors.append(
            {
                "day": float(horizon),
                "q10": ordered[0],
                "q25": ordered[1],
                "q50": ordered[2],
                "q75": ordered[3],
                "q90": ordered[4],
            }
        )
    anchor_frame = pd.DataFrame(anchors)
    days = np.arange(0, int(anchor_frame["day"].max()) + 1, dtype=float)
    result = {"day": days.astype(int)}
    for field in ("q10", "q25", "q50", "q75", "q90"):
        result[field] = np.interp(days, anchor_frame["day"], anchor_frame[field])
    return pd.DataFrame(result)


class Forecaster(Protocol):
    name: str

    def predict(
        self,
        frame: pd.DataFrame,
        horizon_days: int,
        *,
        event_rows: list[dict[str, Any]] | None = None,
    ) -> ForecastEstimate: ...


def _annualized_volatility(frame: pd.DataFrame) -> float:
    if "daily_return" not in frame or len(frame) < 20:
        return 0.0
    returns = frame["daily_return"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 20:
        return 0.0
    return float(returns.tail(252).std(ddof=1) * math.sqrt(252))


def _probability_positive(q10: float, q50: float, q90: float) -> float:
    points = [(q10, 0.1), (q50, 0.5), (q90, 0.9)]
    if q10 >= 0:
        return 0.95
    if q90 <= 0:
        return 0.05
    for (left_value, left_cdf), (right_value, right_cdf) in zip(points, points[1:], strict=False):
        if left_value <= 0 <= right_value:
            width = right_value - left_value
            cdf_zero = (
                left_cdf
                if width == 0
                else left_cdf + (0 - left_value) / width * (right_cdf - left_cdf)
            )
            return float(1 - cdf_zero)
    return 0.5


class RandomWalkForecaster:
    name = "random-walk"

    def __init__(self, simulations: int = 4_000, seed: int = 20260831):
        self.simulations = simulations
        self.seed = seed

    def predict(
        self,
        frame: pd.DataFrame,
        horizon_days: int,
        *,
        event_rows: list[dict[str, Any]] | None = None,
    ) -> ForecastEstimate:
        if len(frame) < 20:
            raise ValueError("基线预测至少需要 20 个交易日")
        returns = frame["daily_return"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        daily_volatility = float(returns.tail(252).std(ddof=1))
        if not math.isfinite(daily_volatility) or daily_volatility <= 0:
            daily_volatility = 1e-6
        digest = hashlib.sha256(
            f"{self.seed}:{horizon_days}:{len(frame)}:{frame.iloc[-1]['trade_date']}".encode()
        ).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**32)
        rng = np.random.default_rng(seed)
        simulated = rng.normal(
            loc=0.0,
            scale=daily_volatility,
            size=(self.simulations, horizon_days),
        )
        paths = np.cumprod(1 + simulated, axis=1)
        cumulative_returns = paths[:, -1] - 1
        running_peak = np.maximum.accumulate(
            np.column_stack([np.ones(self.simulations), paths]), axis=1
        )
        extended_paths = np.column_stack([np.ones(self.simulations), paths])
        drawdowns = 1 - extended_paths / running_peak
        q10, q50, q90 = np.quantile(cumulative_returns, QUANTILE_LEVELS)
        return ForecastEstimate(
            model=self.name,
            horizon_days=horizon_days,
            q10=float(q10),
            q50=float(q50),
            q90=float(q90),
            up_probability=float(np.mean(cumulative_returns > 0)),
            annualized_volatility=daily_volatility * math.sqrt(252),
            potential_drawdown=float(np.quantile(np.max(drawdowns, axis=1), 0.9)),
        )


class Chronos2Forecaster:
    name = "chronos-2"

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device: str = "cpu",
        context_length: int = 512,
        pipeline: Any | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.context_length = context_length
        self._pipeline = pipeline

    @staticmethod
    def installed() -> bool:
        return find_spec("chronos") is not None and find_spec("torch") is not None

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if not self.installed():
            raise RuntimeError("Chronos 未安装；请运行 uv sync --extra forecast")
        # A cached model should not make an anonymous Hub metadata request on
        # every run.  Temporarily force offline mode for the cache probe, then
        # restore the caller's environment before an optional first download.
        offline_was_set = "HF_HUB_OFFLINE" in os.environ
        offline_previous = os.environ.get("HF_HUB_OFFLINE")
        if not offline_was_set:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            from chronos import Chronos2Pipeline

            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                local_files_only=True,
            )
        except OSError:
            if offline_was_set:
                if str(offline_previous).lower() in {"1", "true", "yes"}:
                    raise
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
            # Respect an explicit user setting; otherwise prefer resumable HTTP
            # over Xet, which can stall on some personal networks.
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
            )
        finally:
            if not offline_was_set:
                os.environ.pop("HF_HUB_OFFLINE", None)
        return self._pipeline

    def predict(
        self,
        frame: pd.DataFrame,
        horizon_days: int,
        *,
        event_rows: list[dict[str, Any]] | None = None,
    ) -> ForecastEstimate:
        if len(frame) < 60:
            raise ValueError("Chronos 预测至少需要 60 个交易日")
        context = self._feature_frame(frame, event_rows).tail(self.context_length)
        pipeline = self._load()
        prediction = pipeline.predict_df(
            context,
            prediction_length=horizon_days,
            quantile_levels=list(QUANTILE_LEVELS),
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        if prediction is None or len(prediction) < horizon_days:
            raise RuntimeError("Chronos 返回的预测长度不足")
        prediction = prediction.sort_values("timestamp")
        final = prediction.iloc[-1]
        last_target = float(context.iloc[-1]["target"])
        if last_target <= 0:
            raise RuntimeError("Chronos 输入目标无效")
        q10 = self._read_quantile(final, 0.1) / last_target - 1
        q50 = self._read_quantile(final, 0.5) / last_target - 1
        q90 = self._read_quantile(final, 0.9) / last_target - 1
        annualized_volatility = _annualized_volatility(frame)
        return ForecastEstimate(
            model=self.name,
            horizon_days=horizon_days,
            q10=q10,
            q50=q50,
            q90=q90,
            up_probability=_probability_positive(q10, q50, q90),
            annualized_volatility=annualized_volatility,
            potential_drawdown=max(0.0, -q10),
        )

    @staticmethod
    def _read_quantile(row: pd.Series, level: float) -> float:
        candidates = [str(level), f"{level:.1f}", level, f"q{int(level * 100)}"]
        for candidate in candidates:
            if candidate in row.index:
                return float(row[candidate])
        if level == 0.5 and "predictions" in row.index:
            return float(row["predictions"])
        raise RuntimeError(f"Chronos 结果缺少 {level:.1f} 分位数")

    @staticmethod
    def _feature_frame(
        frame: pd.DataFrame, event_rows: list[dict[str, Any]] | None
    ) -> pd.DataFrame:
        data = frame.copy().sort_values("trade_date")
        target = data["total_return_index"].astype(float)
        volume = data["volume"].astype(float).replace(0, np.nan)
        volume_z = (volume - volume.rolling(60, min_periods=20).mean()) / volume.rolling(
            60, min_periods=20
        ).std()
        volatility_20 = data["daily_return"].rolling(20, min_periods=10).std() * math.sqrt(252)
        event_score = pd.Series(0.0, index=data.index)
        if event_rows:
            dates = pd.to_datetime(data["trade_date"]).dt.date
            for event in event_rows:
                effective = date.fromisoformat(str(event["effective_from"]))
                expires = date.fromisoformat(str(event["expires_at"]))
                mask = (dates >= effective) & (dates <= expires)
                score = (
                    float(event["direction"])
                    * float(event["strength"])
                    * float(event["confidence"])
                )
                event_score.loc[mask] += score
        return pd.DataFrame(
            {
                "id": "asset",
                # Trading holidays make exchange dates irregular. Chronos needs an
                # inferable frequency, so model time is the ordered trading-day index.
                # Event covariates are aligned to the real dates before this conversion.
                "timestamp": pd.bdate_range("2000-01-03", periods=len(data)),
                "target": target,
                "volume_z": volume_z.fillna(0).clip(-5, 5),
                "realized_volatility_20": volatility_20.fillna(0).clip(0, 5),
                "event_score": event_score.clip(-1, 1),
            }
        )


@dataclass(frozen=True)
class CalibrationWeights:
    chronos: float
    baseline: float
    status: ModelStatus
    samples: int
    reason: str


def pinball_loss(actual: float, prediction: float, quantile: float) -> float:
    error = actual - prediction
    return float(max(quantile * error, (quantile - 1) * error))


def estimate_loss(actual: float, estimate: dict[str, Any]) -> float:
    values = ((0.1, estimate["q10"]), (0.5, estimate["q50"]), (0.9, estimate["q90"]))
    return float(np.mean([pinball_loss(actual, float(value), level) for level, value in values]))


def calibration_weights(
    database: Database,
    symbol: str,
    horizon_days: int,
    config: AppConfig,
) -> CalibrationWeights:
    forecast_config = config.section("forecast")
    initial = float(forecast_config.get("initial_chronos_weight", 0.5))
    minimum_samples = int(forecast_config.get("minimum_calibration_samples", 100))
    losing_streak = int(forecast_config.get("disable_after_consecutive_losses", 60))
    evaluations: list[dict[str, Any]] = []
    for row in database.receipts(symbol=symbol, status="evaluated"):
        if int(row["horizon_days"]) != horizon_days or not row["evaluation_json"]:
            continue
        payload = json.loads(row["evaluation_json"])
        if payload.get("chronos_loss") is not None and payload.get("baseline_loss") is not None:
            evaluations.append(payload)
    samples = len(evaluations)
    if samples >= losing_streak:
        recent = evaluations[-losing_streak:]
        if all(item["chronos_loss"] > item["baseline_loss"] for item in recent):
            return CalibrationWeights(
                chronos=0.0,
                baseline=1.0,
                status=ModelStatus.DISABLED,
                samples=samples,
                reason=f"Chronos 连续 {losing_streak} 次差于基线",
            )
    if samples < minimum_samples:
        return CalibrationWeights(
            chronos=initial,
            baseline=1 - initial,
            status=ModelStatus.EXPERIMENTAL,
            samples=samples,
            reason=f"仅 {samples}/{minimum_samples} 个校准样本",
        )
    chronos_loss = float(np.mean([item["chronos_loss"] for item in evaluations[-250:]]))
    baseline_loss = float(np.mean([item["baseline_loss"] for item in evaluations[-250:]]))
    inverse_chronos = 1 / max(chronos_loss, 1e-9)
    inverse_baseline = 1 / max(baseline_loss, 1e-9)
    raw = inverse_chronos / (inverse_chronos + inverse_baseline)
    chronos_weight = float(np.clip(raw, 0.2, 0.8))
    return CalibrationWeights(
        chronos=chronos_weight,
        baseline=1 - chronos_weight,
        status=ModelStatus.ACTIVE,
        samples=samples,
        reason="按滚动分位数损失定权",
    )


def _blend(
    baseline: ForecastEstimate,
    chronos: ForecastEstimate | None,
    weights: CalibrationWeights,
) -> tuple[ForecastEstimate, dict[str, float]]:
    if chronos is None or weights.chronos <= 0:
        return baseline.model_copy(update={"model": "ensemble-baseline-only"}), {
            "random-walk": 1.0,
            "chronos-2": 0.0,
        }
    baseline_weight = weights.baseline
    chronos_weight = weights.chronos
    payload: dict[str, float | str | int] = {
        "model": "ensemble",
        "horizon_days": baseline.horizon_days,
    }
    for field in (
        "q10",
        "q50",
        "q90",
        "up_probability",
        "annualized_volatility",
        "potential_drawdown",
    ):
        payload[field] = (
            getattr(baseline, field) * baseline_weight + getattr(chronos, field) * chronos_weight
        )
    return ForecastEstimate.model_validate(payload), {
        "random-walk": baseline_weight,
        "chronos-2": chronos_weight,
    }


def forecast_one(
    *,
    symbol: str,
    as_of: date,
    horizon_days: int,
    frame: pd.DataFrame,
    data_quality: DataQuality,
    database: Database,
    config: AppConfig,
    event_rows: list[dict[str, Any]] | None = None,
    use_chronos: bool = True,
    chronos_forecaster: Forecaster | None = None,
) -> ForecastBundle:
    baseline = RandomWalkForecaster().predict(frame, horizon_days, event_rows=event_rows)
    calibration = calibration_weights(database, symbol, horizon_days, config)
    calibration_target = int(config.section("forecast").get("minimum_calibration_samples", 100))
    warnings = [calibration.reason]
    chronos: ForecastEstimate | None = None
    status = calibration.status
    if use_chronos and calibration.status is not ModelStatus.DISABLED:
        forecast_config = config.section("forecast")
        forecaster = chronos_forecaster or Chronos2Forecaster(
            model_id=str(forecast_config.get("model_id", "amazon/chronos-2")),
            device=str(forecast_config.get("device", "cpu")),
            context_length=int(forecast_config.get("context_length", 512)),
        )
        try:
            chronos = forecaster.predict(frame, horizon_days, event_rows=event_rows)
        except Exception as exc:
            warnings.append(f"Chronos 降级: {exc}")
            status = ModelStatus.DEGRADED
    elif not use_chronos:
        warnings.append("本次显式跳过 Chronos")
        status = ModelStatus.DEGRADED
    ensemble, actual_weights = _blend(baseline, chronos, calibration)
    if chronos is None:
        actual_weights = {"random-walk": 1.0, "chronos-2": 0.0}
    components = {baseline.model: baseline}
    if chronos is not None:
        components[chronos.model] = chronos
    return ForecastBundle(
        symbol=symbol,
        as_of=as_of,
        due_date=next_trading_date(frame, as_of, horizon_days),
        horizon_days=horizon_days,
        ensemble=ensemble,
        components=components,
        weights=actual_weights,
        status=status,
        data_quality=data_quality,
        warnings=warnings,
        calibration_samples=calibration.samples,
        calibration_target=calibration_target,
    )


def evaluate_open_receipts(database: Database) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for row in database.receipts(status="open"):
        symbol = str(row["symbol"])
        bars = database.load_bars(symbol)
        actions = database.load_actions(symbol)
        frame, _ = total_return_frame(bars, actions)
        if frame.empty:
            continue
        due_date = date.fromisoformat(row["due_date"])
        latest_date = pd.Timestamp(frame.iloc[-1]["trade_date"]).date()
        if latest_date < due_date:
            continue
        as_of = date.fromisoformat(row["as_of"])
        before = frame[pd.to_datetime(frame["trade_date"]).dt.date <= as_of]
        after = frame[pd.to_datetime(frame["trade_date"]).dt.date >= due_date]
        if before.empty or after.empty:
            continue
        start_index = float(before.iloc[-1]["total_return_index"])
        end_index = float(after.iloc[0]["total_return_index"])
        actual = end_index / start_index - 1
        forecast_payload = json.loads(row["forecast_json"])
        components = forecast_payload.get("components", {})
        evaluation: dict[str, Any] = {
            "actual_return": actual,
            "direction_correct": bool(
                (actual >= 0) == (float(forecast_payload["ensemble"]["q50"]) >= 0)
            ),
            "ensemble_loss": estimate_loss(actual, forecast_payload["ensemble"]),
            "interval_covered": bool(
                float(forecast_payload["ensemble"]["q10"])
                <= actual
                <= float(forecast_payload["ensemble"]["q90"])
            ),
        }
        baseline_payload = components.get("random-walk")
        chronos_payload = components.get("chronos-2")
        evaluation["baseline_loss"] = (
            estimate_loss(actual, baseline_payload) if baseline_payload else None
        )
        evaluation["chronos_loss"] = (
            estimate_loss(actual, chronos_payload) if chronos_payload else None
        )
        receipt = dict(row)
        receipt.update(
            {
                "forecast": forecast_payload,
                "decision": json.loads(row["decision_json"]),
                "evidence": json.loads(row["evidence_json"]),
                "status": "evaluated",
                "realized_return": actual,
                "evaluation": evaluation,
                "evaluated_at": utc_now().isoformat(),
            }
        )
        database.save_receipt(receipt)
        evaluated.append({"id": row["id"], "symbol": symbol, **evaluation})
    return evaluated


def _trading_due_date(frame: pd.DataFrame, cutoff_index: int, horizon_days: int) -> date:
    return pd.Timestamp(frame.iloc[cutoff_index + horizon_days]["trade_date"]).date()


def walk_forward_backtest(
    *,
    symbol: str,
    frame: pd.DataFrame,
    horizon_days: int,
    database: Database,
    config: AppConfig,
    use_chronos: bool = False,
    max_windows: int = 100,
    step: int | None = None,
    chronos_forecaster: Forecaster | None = None,
) -> dict[str, Any]:
    minimum_history = int(config.section("forecast").get("minimum_history_days", 756))
    if len(frame) < minimum_history + horizon_days:
        raise ValueError(f"walk-forward 至少需要 {minimum_history + horizon_days} 个交易日")
    stride = step or max(1, horizon_days)
    cutoffs = list(range(minimum_history - 1, len(frame) - horizon_days, stride))[-max_windows:]
    records: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        context = frame.iloc[: cutoff + 1].copy()
        as_of = pd.Timestamp(context.iloc[-1]["trade_date"]).date()
        actual = (
            float(frame.iloc[cutoff + horizon_days]["total_return_index"])
            / float(context.iloc[-1]["total_return_index"])
            - 1
        )
        baseline = RandomWalkForecaster().predict(context, horizon_days)
        components: dict[str, ForecastEstimate] = {baseline.model: baseline}
        chronos: ForecastEstimate | None = None
        if use_chronos:
            forecaster = chronos_forecaster or Chronos2Forecaster(
                model_id=str(config.section("forecast").get("model_id", "amazon/chronos-2")),
                device=str(config.section("forecast").get("device", "cpu")),
                context_length=int(config.section("forecast").get("context_length", 512)),
            )
            chronos = forecaster.predict(context, horizon_days)
            components[chronos.model] = chronos
        weights = CalibrationWeights(
            chronos=0.5 if chronos else 0.0,
            baseline=0.5 if chronos else 1.0,
            status=ModelStatus.EXPERIMENTAL,
            samples=0,
            reason="walk-forward 固定初始权重",
        )
        ensemble, actual_weights = _blend(baseline, chronos, weights)
        evaluation = {
            "actual_return": actual,
            "forecast_q50": float(ensemble.q50),
            "up_probability": float(ensemble.up_probability),
            "direction_correct": bool((actual >= 0) == (ensemble.q50 >= 0)),
            "interval_covered": bool(ensemble.q10 <= actual <= ensemble.q90),
            "ensemble_loss": estimate_loss(actual, ensemble.model_dump()),
            "baseline_loss": estimate_loss(actual, baseline.model_dump()),
            "chronos_loss": estimate_loss(actual, chronos.model_dump()) if chronos else None,
        }
        receipt_id = (
            "bt-"
            + hashlib.sha256(f"{symbol}:{as_of}:{horizon_days}:{use_chronos}".encode()).hexdigest()[
                :16
            ]
        )
        database.save_receipt(
            {
                "id": receipt_id,
                "symbol": symbol,
                "created_at": datetime.combine(as_of, datetime.min.time()).isoformat(),
                "as_of": as_of.isoformat(),
                "horizon_days": horizon_days,
                "due_date": _trading_due_date(frame, cutoff, horizon_days).isoformat(),
                "model_status": ModelStatus.EXPERIMENTAL.value,
                "forecast": {
                    "ensemble": ensemble.model_dump(mode="json"),
                    "components": {
                        name: estimate.model_dump(mode="json")
                        for name, estimate in components.items()
                    },
                    "weights": actual_weights,
                },
                "decision": {"backtest": True},
                "evidence": [],
                "status": "evaluated",
                "realized_return": actual,
                "evaluation": evaluation,
                "evaluated_at": utc_now().isoformat(),
            }
        )
        records.append(evaluation)
    if not records:
        raise ValueError("没有可用的 walk-forward 窗口")

    # Quantitative Strategy Performance Metrics
    trade_signals = [item for item in records if item["forecast_q50"] > 0]
    trade_returns = [item["actual_return"] for item in trade_signals]
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r < 0]
    win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
    tot_win = sum(wins)
    tot_loss = abs(sum(losses))
    profit_factor = tot_win / tot_loss if tot_loss > 1e-9 else (99.0 if tot_win > 0 else 1.0)
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    expected_value = win_rate * avg_win + (1.0 - win_rate) * avg_loss

    # Strategy Cumulative Equity & Drawdown
    strat_returns = np.array(trade_returns if trade_returns else [0.0])
    equity_curve = np.cumprod(1.0 + strat_returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = float(abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0

    ret_std = float(np.std(strat_returns)) if len(strat_returns) > 1 else 0.0
    ret_mean = float(np.mean(strat_returns))
    periods_per_year = max(1.0, 252.0 / horizon_days)
    sharpe_ratio = (
        float((ret_mean / ret_std) * math.sqrt(periods_per_year)) if ret_std > 1e-9 else 0.0
    )

    return {
        "symbol": symbol,
        "horizon_days": horizon_days,
        "windows": len(records),
        "active_trades": len(trade_signals),
        "chronos_used": use_chronos,
        "direction_accuracy": float(np.mean([item["direction_correct"] for item in records])),
        "interval_coverage": float(np.mean([item["interval_covered"] for item in records])),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expected_return": float(expected_value),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "ensemble_pinball_loss": float(np.mean([item["ensemble_loss"] for item in records])),
        "baseline_pinball_loss": float(np.mean([item["baseline_loss"] for item in records])),
        "chronos_pinball_loss": (
            float(np.mean([item["chronos_loss"] for item in records])) if use_chronos else None
        ),
    }
