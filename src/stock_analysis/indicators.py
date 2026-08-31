from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stock_analysis.data import Database


@dataclass(frozen=True)
class PriceZone:
    kind: str
    low: float
    center: float
    high: float
    strength: float
    touches: int
    last_touch: date
    distance: float


def detect_price_zones(
    frame: pd.DataFrame,
    *,
    as_of: date | None = None,
    lookback_sessions: int = 360,
    pivot_window: int = 3,
    max_per_side: int = 2,
) -> list[PriceZone]:
    """Detect support/resistance bands from point-in-time pivot clusters.

    Bands combine repeated local highs/lows with ATR-scaled clustering,
    relative volume and recency decay. They are descriptive levels, not targets.
    """
    if frame.empty or len(frame) < pivot_window * 2 + 5:
        return []
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date"]).sort_values("trade_date")
    if as_of is not None:
        data = data[data["trade_date"].dt.date <= as_of]
    data = add_indicators(data).tail(lookback_sessions).reset_index(drop=True)
    if len(data) < pivot_window * 2 + 5:
        return []
    current = float(data.iloc[-1]["close"])
    atr = float(data["atr14"].tail(60).median())
    tolerance = max(current * 0.0125, atr * 0.75, 1e-9)
    candidates: list[dict[str, float | date]] = []
    last_index = len(data) - 1
    for index in range(pivot_window, len(data) - pivot_window):
        row = data.iloc[index]
        local = data.iloc[index - pivot_window : index + pivot_window + 1]
        volume_weight = float(np.clip(row["volume_ratio20"], 0.5, 2.0))
        recency_weight = math.exp(-(last_index - index) / 126)
        weight = (0.65 + 0.35 * volume_weight) * (0.45 + 0.55 * recency_weight)
        touched = pd.Timestamp(row["trade_date"]).date()
        if float(row["low"]) <= float(local["low"].min()):
            candidates.append({"price": float(row["low"]), "weight": weight, "date": touched})
        if float(row["high"]) >= float(local["high"].max()):
            candidates.append({"price": float(row["high"]), "weight": weight, "date": touched})
    if not candidates:
        return []
    clusters: list[list[dict[str, float | date]]] = []
    for item in sorted(candidates, key=lambda value: float(value["price"])):
        if not clusters:
            clusters.append([item])
            continue
        previous = clusters[-1]
        center = np.average(
            [float(value["price"]) for value in previous],
            weights=[float(value["weight"]) for value in previous],
        )
        if abs(float(item["price"]) - center) <= tolerance:
            previous.append(item)
        else:
            clusters.append([item])
    zones: list[PriceZone] = []
    for cluster in clusters:
        touches = len(cluster)
        if touches < 2:
            continue
        weights = [float(item["weight"]) for item in cluster]
        center = float(
            np.average([float(item["price"]) for item in cluster], weights=weights)
        )
        raw_strength = sum(weights) * math.log1p(touches)
        strength = float(np.clip(1 - math.exp(-raw_strength / 4), 0, 1))
        half_width = tolerance * min(0.75, 0.35 + touches * 0.04)
        kind = "support" if center <= current else "resistance"
        last_touch = max(item["date"] for item in cluster)
        zones.append(
            PriceZone(
                kind=kind,
                low=max(0.0, center - half_width),
                center=center,
                high=center + half_width,
                strength=strength,
                touches=touches,
                last_touch=last_touch,  # type: ignore[arg-type]
                distance=center / current - 1,
            )
        )
    support = sorted(
        (zone for zone in zones if zone.kind == "support"),
        key=lambda zone: (abs(zone.distance), -zone.strength),
    )[:max_per_side]
    resistance = sorted(
        (zone for zone in zones if zone.kind == "resistance"),
        key=lambda zone: (abs(zone.distance), -zone.strength),
    )[:max_per_side]
    return [*support, *resistance]


def macro_exposures(profile: dict[str, object]) -> dict[str, float]:
    """Return small, explicit macro exposure maps by asset/sector profile."""
    sector = str(profile.get("sector", "")).lower()
    model = str(profile.get("valuation_model", "")).lower()
    if model == "fund" or "黄金" in sector or "gold" in sector:
        return {"GOLD": 0.50, "DXY": 0.25, "US10Y": 0.20, "CSI300": 0.05}
    if model in {"bank", "insurer"} or "金融" in sector:
        return {"CSI300": 0.45, "SHIBOR": -0.20, "US10Y": 0.15, "DXY": 0.10, "WTI": 0.10}
    if "有色" in sector or "金属" in sector:
        return {"GOLD": 0.30, "DXY": 0.20, "US10Y": 0.15, "WTI": 0.10, "CSI300": 0.25}
    if "农业" in sector:
        return {"WTI": -0.25, "DXY": 0.10, "SHIBOR": 0.10, "CSI300": 0.35, "GOLD": 0.20}
    return {"CSI300": 0.35, "DXY": 0.15, "US10Y": 0.15, "SHIBOR": 0.10, "WTI": 0.15, "GOLD": 0.10}


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add trailing technical indicators without using future rows."""
    if frame.empty:
        return frame.copy()
    result = frame.sort_values("trade_date").copy()
    close = result["close"].astype(float)
    high = result.get("high", close).astype(float)
    low = result.get("low", close).astype(float)
    volume = result.get("volume", pd.Series(0.0, index=result.index)).astype(float)
    result["ma20"] = close.rolling(20, min_periods=1).mean()
    result["ma60"] = close.rolling(60, min_periods=1).mean()
    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    result["macd"] = ema12 - ema26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False, min_periods=1).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    losses = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gains / losses.replace(0, np.nan)
    result["rsi14"] = (100 - 100 / (1 + rs)).fillna(50.0)
    previous = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    result["atr14"] = true_range.rolling(14, min_periods=1).mean()
    volume_mean = volume.rolling(20, min_periods=1).mean().replace(0, np.nan)
    result["volume_ratio20"] = (volume / volume_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return result


def technical_values(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame.empty:
        return {}
    enriched = add_indicators(frame)
    row = enriched.iloc[-1]
    keys = (
        "ma20",
        "ma60",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi14",
        "atr14",
        "volume_ratio20",
    )
    return {key: float(row[key]) if pd.notna(row[key]) else None for key in keys}


def macro_assessments(
    database: Database,
    as_of: date,
    exposures: dict[str, float] | None = None,
) -> tuple[list[dict[str, float | str]], float]:
    """Calculate transparent macro momentum scores from point-in-time observations."""
    frame = database.load_macro_observations(as_of=as_of)
    if frame.empty:
        return [], 0.0
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    assessments: list[dict[str, float | str]] = []
    weighted_scores: list[float] = []
    total_weight = 0.0
    for series, group in frame.groupby("series"):
        values = group.sort_values("observation_date")["value"].dropna()
        if len(values) < 2 or values.iloc[-2] == 0:
            continue
        momentum = float(values.iloc[-1] / values.iloc[-2] - 1)
        score = float(np.clip(np.tanh(momentum * 20), -1, 1))
        direction = -1.0 if str(series) in {"DXY", "US10Y", "SHIBOR"} else 1.0
        adjusted = float(np.clip(score * direction, -1, 1))
        exposure = float(
            exposures.get(str(series), 0.0) if exposures is not None else 1.0
        )
        assessments.append(
            {
                "name": str(series),
                "value": float(values.iloc[-1]),
                "score": adjusted,
                "momentum": momentum,
                "exposure": exposure,
            }
        )
        if exposure:
            weighted_scores.append(adjusted * exposure)
            total_weight += abs(exposure)
    score = sum(weighted_scores) / total_weight if total_weight else 0.0
    return assessments, float(np.clip(score, -1, 1))
