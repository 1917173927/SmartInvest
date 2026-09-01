from __future__ import annotations

import hashlib
import math
import re
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from stock_analysis.data import AppConfig, Database, DataQuality, FundamentalRecord, utc_now
from stock_analysis.forecast import ForecastBundle
from stock_analysis.indicators import (
    PriceZone,
    PriceZoneValidation,
    add_indicators,
    detect_price_zones,
    macro_assessments,
    macro_exposures,
    validate_price_zones,
)
from stock_analysis.research import ResearchResult


class Horizon(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    VALUE = "value"


HORIZON_LABELS = {
    Horizon.SHORT: "短线（1–20 个交易日）",
    Horizon.MEDIUM: "中线（1–6 个月）",
    Horizon.LONG: "长线（1–3 年）",
    Horizon.VALUE: "价值（3–10 年）",
}


class MetricAssessment(BaseModel):
    name: str
    value: float | None = None
    score: float
    available: bool
    explanation: str


class ValuationRange(BaseModel):
    available: bool
    fair_low: float | None = None
    fair_high: float | None = None
    buy_low: float | None = None
    buy_high: float | None = None
    currency: str | None = None
    method: str


class HorizonDecision(BaseModel):
    horizon: Horizon
    score: float
    confidence: float
    action: str
    rationale: str
    target_position: float | None = None
    staging: str | None = None
    invalidation_conditions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StagingTier(BaseModel):
    tier_name: str
    target_price: float
    weight_pct: float
    shares: int
    allocated_amount: float
    rationale: str


class StagingPlan(BaseModel):
    available: bool = False
    total_target_weight: float = 0.0
    total_shares: int = 0
    total_capital: float = 0.0
    tiers: list[StagingTier] = Field(default_factory=list)
    invalidation_price: float | None = None
    invalidation_note: str = ""


class ExitStatus(StrEnum):
    UNAVAILABLE = "数据不足"
    HOLD = "无需减仓"
    REVIEW = "暂停买入并复核"
    REDUCE = "减仓"
    EXIT = "退出"


class ExitPlan(BaseModel):
    status: ExitStatus = ExitStatus.UNAVAILABLE
    action: str = "持仓数据不足，无法计算减仓数量"
    current_weight: float | None = None
    target_weight: float | None = None
    current_shares: int | None = None
    target_shares: int | None = None
    sell_shares: int = 0
    reference_price: float | None = None
    estimated_proceeds: float = 0.0
    total_assets: float | None = None
    holding_source: str = "未提供"
    reasons: list[str] = Field(default_factory=list)
    trigger_conditions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnalysisPackage(BaseModel):
    symbol: str
    name: str
    as_of: date
    current_price: float
    currency: str
    data_quality: DataQuality
    data_warnings: list[str]
    forecasts: list[ForecastBundle]
    research: ResearchResult
    technical: list[MetricAssessment]
    quality: list[MetricAssessment]
    valuation: list[MetricAssessment]
    valuation_range: ValuationRange
    decisions: list[HorizonDecision]
    receipt_ids: list[str] = Field(default_factory=list)
    macro: list[MetricAssessment] = Field(default_factory=list)
    macro_score: float = 0.0
    chart_paths: list[str] = Field(default_factory=list)
    price_zones: list[PriceZone] = Field(default_factory=list)
    price_zone_validation: list[PriceZoneValidation] = Field(default_factory=list)
    staging_plan: StagingPlan | None = None
    exit_plan: ExitPlan | None = None


class PortfolioPosition(BaseModel):
    symbol: str | None
    name: str
    quantity: float
    market_value: float
    currency: str = "CNY"
    sector: str = "未分类"
    role: str = "satellite"


class PortfolioSnapshot(BaseModel):
    path: Path
    as_of: date
    total_cny_assets: float | None
    cash_cny: float | None
    positions: list[PortfolioPosition]
    warnings: list[str] = Field(default_factory=list)


def _clip(value: float) -> float:
    return float(np.clip(value, -1, 1))


def _metric(
    name: str,
    value: float | None,
    score: float,
    explanation: str,
) -> MetricAssessment:
    return MetricAssessment(
        name=name,
        value=value,
        score=_clip(score),
        available=value is not None,
        explanation=explanation,
    )


def technical_assessments(frame: pd.DataFrame) -> list[MetricAssessment]:
    if frame.empty:
        return [_metric("趋势", None, 0, "无行情数据")]
    closes = frame["close"].astype(float)
    enriched = add_indicators(frame)
    current = float(closes.iloc[-1])
    results: list[MetricAssessment] = []
    for window in (20, 60):
        if len(closes) < window:
            results.append(_metric(f"相对 MA{window}", None, 0, "样本不足"))
            continue
        average = float(closes.tail(window).mean())
        deviation = current / average - 1
        score = math.tanh(deviation * 8)
        results.append(
            _metric(
                f"相对 MA{window}",
                deviation,
                score,
                f"现价相对 {window} 日均线 {deviation:+.1%}",
            )
        )
    for window in (20, 60):
        if len(closes) <= window:
            continue
        momentum = current / float(closes.iloc[-window - 1]) - 1
        results.append(
            _metric(
                f"{window}日动量",
                momentum,
                math.tanh(momentum * 5),
                f"{window} 日收益 {momentum:+.1%}",
            )
        )
    returns = frame.get("daily_return", closes.pct_change()).astype(float).dropna()
    if len(returns) >= 20:
        volatility = float(returns.tail(120).std(ddof=1) * math.sqrt(252))
        volatility_score = 0.4 - volatility
        results.append(
            _metric(
                "年化波动率",
                volatility,
                volatility_score,
                f"近 120 日年化波动率 {volatility:.1%}",
            )
        )
    if len(closes) >= 120:
        rolling_mean = float(closes.tail(120).mean())
        rolling_std = float(closes.tail(120).std(ddof=1))
        z_score = (current - rolling_mean) / rolling_std if rolling_std else 0.0
        anti_chase = -max(0.0, (z_score - 1.5) / 2)
        results.append(
            _metric(
                "追高约束",
                z_score,
                anti_chase,
                f"现价相对 120 日均值为 {z_score:.2f} 个标准差",
            )
        )
    latest = enriched.iloc[-1]
    macd_hist = float(latest["macd_hist"])
    current = float(closes.iloc[-1])
    macd_score = math.tanh(macd_hist / max(abs(current), 1e-9) * 40)
    results.append(
        _metric(
            "MACD 动能",
            macd_hist / max(abs(current), 1e-9),
            macd_score,
            f"MACD 柱线 {'向上' if macd_hist >= 0 else '向下'}，标准化动能 {macd_score:+.2f}",
        )
    )
    rsi = float(latest["rsi14"])
    rsi_score = float(np.clip((50 - rsi) / 50, -1, 1))
    results.append(
        _metric(
            "RSI14",
            rsi / 100,
            rsi_score,
            f"RSI14 {rsi:.1f}；低位反弹与高位追涨均需结合趋势确认",
        )
    )
    volume_ratio = float(latest["volume_ratio20"])
    results.append(
        _metric(
            "成交量/20日均量",
            volume_ratio,
            float(np.clip((volume_ratio - 1) / 2, -1, 1)),
            f"成交量约为 20 日均量的 {volume_ratio:.2f} 倍",
        )
    )
    if "yz_vol20" in latest and pd.notna(latest["yz_vol20"]):
        yz_vol = float(latest["yz_vol20"])
        yz_score = float(np.clip(math.tanh((0.25 - yz_vol) * 5), -1, 1))
        regime = (
            "低波动扩张期" if yz_vol < 0.18 else "常态波动期" if yz_vol < 0.30 else "高波动风险期"
        )
        results.append(
            _metric(
                "Yang-Zhang波动率",
                yz_vol,
                yz_score,
                f"Yang-Zhang(2000)极小方差无偏年化波动率 {yz_vol:.1%} ({regime})",
            )
        )
    if "trend_smoothness60" in latest and pd.notna(latest["trend_smoothness60"]):
        smoothness = float(latest["trend_smoothness60"])
        smooth_score = float(np.clip(math.tanh(smoothness * 0.7), -1, 1))
        results.append(
            _metric(
                "特质趋势平滑度",
                smoothness,
                smooth_score,
                f"Blitz(2013)残差趋势信噪比 {smoothness:+.2f}；反映低噪音持续性",
            )
        )
    return results


def _value(records: dict[str, FundamentalRecord], key: str) -> float | None:
    item = records.get(key)
    if item is None or not math.isfinite(item.value):
        return None
    return float(item.value)


def quality_assessments(records: dict[str, FundamentalRecord]) -> list[MetricAssessment]:
    """Calculate multi-dimensional Quality-Minus-Junk (QMJ, Asness et al. 2019) scores."""
    results: list[MetricAssessment] = []
    score_prof = 0.0
    roe = _value(records, "roe")
    if roe is not None:
        score_prof = (roe - 0.08) / 0.12
        results.append(_metric("ROE", roe, score_prof, f"ROE {roe:.1%} (盈利能力)"))
    else:
        results.append(_metric("ROE", None, 0, "缺少可核验 ROE"))

    score_cash = 0.0
    net_income = _value(records, "net_income")
    operating_cash_flow = _value(records, "operating_cash_flow")
    if net_income and operating_cash_flow is not None and net_income > 0:
        cash_quality = operating_cash_flow / net_income
        score_cash = (cash_quality - 0.7) / 0.6
        results.append(
            _metric(
                "现金流/利润",
                cash_quality,
                score_cash,
                f"经营现金流约为净利润的 {cash_quality:.2f} 倍 (盈余质量)",
            )
        )
    else:
        results.append(_metric("现金流/利润", None, 0, "缺少同期现金流与净利润"))

    score_safety = 0.0
    assets = _value(records, "assets")
    liabilities = _value(records, "liabilities")
    if assets and liabilities is not None:
        liability_ratio = liabilities / assets
        score_safety = (0.65 - liability_ratio) / 0.35
        results.append(
            _metric(
                "资产负债率",
                liability_ratio,
                score_safety,
                f"资产负债率 {liability_ratio:.1%}；金融企业需结合行业口径解释",
            )
        )
    else:
        debt_to_equity = _value(records, "debt_to_equity")
        if debt_to_equity is not None:
            score_safety = (1.0 - debt_to_equity) / 1.5
            results.append(
                _metric(
                    "债务/权益",
                    debt_to_equity,
                    score_safety,
                    f"债务权益比 {debt_to_equity:.2f}",
                )
            )
        else:
            results.append(_metric("杠杆", None, 0, "缺少可比较杠杆数据"))

    score_payout = 0.0
    dividend = _value(records, "dividend_yield")
    if dividend is not None:
        score_payout = (dividend - 0.02) / 0.04
        results.append(
            _metric(
                "股息率",
                dividend,
                score_payout,
                f"近端股息率 {dividend:.1%}，仍需验证现金来源与持续性",
            )
        )
    else:
        results.append(_metric("股息率", None, 0, "缺少股息率"))

    # Composite QMJ score (Asness, Frazzini, Pedersen, 2019)
    valid_components = sum(
        1
        for val in (
            roe,
            operating_cash_flow,
            liabilities or _value(records, "debt_to_equity"),
            dividend,
        )
        if val is not None
    )
    if valid_components >= 2:
        qmj_score = _clip(
            0.35 * score_prof + 0.25 * score_cash + 0.25 * score_safety + 0.15 * score_payout
        )
        results.append(
            _metric(
                "QMJ复合质量",
                qmj_score,
                qmj_score,
                f"Asness(2019)质量四维度(盈利/现金/安全/回报)复合得分 {qmj_score:+.2f}",
            )
        )

    return results


class ValuationStrategy(Protocol):
    """Protocol for asset/industry specific valuation models."""

    name: str

    def assessments(
        self, records: dict[str, FundamentalRecord], profile: dict[str, Any]
    ) -> list[MetricAssessment]: ...

    def range(
        self,
        *,
        current_price: float,
        currency: str,
        records: dict[str, FundamentalRecord],
        profile: dict[str, Any],
        override_low: float | None = None,
        override_high: float | None = None,
    ) -> ValuationRange: ...


class BaseValuationStrategy:
    name: str = "generic"
    default_fair_pe: float = 15.0
    default_fair_pb: float = 1.5
    pb_multiplier: float = 1.0
    fair_width: float = 0.25
    safety_margin: float = 0.30

    def assessments(
        self, records: dict[str, FundamentalRecord], profile: dict[str, Any]
    ) -> list[MetricAssessment]:
        results: list[MetricAssessment] = []
        pe = _value(records, "pe")
        pb = _value(records, "pb")
        dividend = _value(records, "dividend_yield")
        fair_pe = float(profile.get("fair_pe", self.default_fair_pe))
        fair_pb = float(profile.get("fair_pb", self.default_fair_pb))
        if pe is not None and pe > 0:
            results.append(
                _metric(
                    "PE",
                    pe,
                    (fair_pe / pe - 1) * 1.5,
                    f"PE {pe:.2f} 倍，对照保守参考 {fair_pe:.2f} 倍",
                )
            )
        else:
            results.append(_metric("PE", None, 0, "盈利为负或缺少 PE，不能据此判定便宜"))
        if pb is not None and pb > 0:
            results.append(
                _metric(
                    "PB",
                    pb,
                    (fair_pb / pb - 1) * self.pb_multiplier,
                    f"PB {pb:.2f} 倍，对照保守参考 {fair_pb:.2f} 倍",
                )
            )
        else:
            results.append(_metric("PB", None, 0, "缺少 PB"))
        if dividend is not None:
            results.append(
                _metric(
                    "股息安全垫",
                    dividend,
                    (dividend - 0.025) / 0.035,
                    f"股息率 {dividend:.1%}；不等同于分红可持续",
                )
            )
        else:
            results.append(_metric("股息安全垫", None, 0, "缺少股息率"))
        return results

    def range(
        self,
        *,
        current_price: float,
        currency: str,
        records: dict[str, FundamentalRecord],
        profile: dict[str, Any],
        override_low: float | None = None,
        override_high: float | None = None,
    ) -> ValuationRange:
        if override_low is not None and override_high is not None:
            if override_low <= 0 or override_high < override_low:
                raise ValueError("人工合理价值区间无效")
            margin = 0.25
            return ValuationRange(
                available=True,
                fair_low=override_low,
                fair_high=override_high,
                buy_low=override_low * (1 - margin),
                buy_high=override_low,
                currency=currency,
                method="人工合理价值区间 + 25% 安全边际",
            )
        estimates: list[float] = []
        methods: list[str] = []
        pe = _value(records, "pe")
        pb = _value(records, "pb")
        fair_pe = profile.get("fair_pe", self.default_fair_pe)
        fair_pb = profile.get("fair_pb", self.default_fair_pb)
        if pe and pe > 0 and fair_pe:
            estimates.append(current_price * float(fair_pe) / pe)
            methods.append("PE")
        if pb and pb > 0 and fair_pb:
            estimates.append(current_price * float(fair_pb) / pb)
            methods.append("PB")
        if not estimates:
            return ValuationRange(
                available=False,
                currency=currency,
                method="缺少 EPS/BVPS、PE/PB 或人工价值区间，拒绝伪造买入价",
            )
        central = float(np.median(estimates))
        fair_low = central * (1 - self.fair_width)
        fair_high = central * (1 + self.fair_width)
        return ValuationRange(
            available=True,
            fair_low=fair_low,
            fair_high=fair_high,
            buy_low=fair_low * (1 - self.safety_margin),
            buy_high=fair_low,
            currency=currency,
            method="/".join(methods) + f" 隐含盈利/净资产情景，{self.safety_margin:.0%} 安全边际",
        )


class BankValuationStrategy(BaseValuationStrategy):
    name = "bank"
    default_fair_pe = 6.8
    default_fair_pb = 0.75
    pb_multiplier = 1.4
    fair_width = 0.20
    safety_margin = 0.20


class InsurerValuationStrategy(BaseValuationStrategy):
    name = "insurer"
    default_fair_pe = 10.5
    default_fair_pb = 1.25
    pb_multiplier = 1.4
    fair_width = 0.20
    safety_margin = 0.20


class CyclicalValuationStrategy(BaseValuationStrategy):
    name = "cyclical"
    default_fair_pe = 14.0
    default_fair_pb = 1.80
    pb_multiplier = 1.0
    fair_width = 0.25
    safety_margin = 0.25


class FundValuationStrategy(BaseValuationStrategy):
    name = "fund"
    fair_width = 0.15
    safety_margin = 0.15

    def assessments(
        self, records: dict[str, FundamentalRecord], profile: dict[str, Any]
    ) -> list[MetricAssessment]:
        pe = _value(records, "pe")
        pb = _value(records, "pb")
        results: list[MetricAssessment] = []
        if pe is not None and pe > 0:
            fair_pe = float(profile.get("fair_pe", 15.0))
            score = (fair_pe / pe - 1) * 1.0
            results.append(_metric("PE", pe, score, f"底层指数/资产估算 PE {pe:.2f}"))
        else:
            results.append(_metric("PE", None, 0, "公募/ETF基金主要跟踪底层资产净值"))
        if pb is not None and pb > 0:
            fair_pb = float(profile.get("fair_pb", 1.5))
            score = (fair_pb / pb - 1) * 1.0
            results.append(_metric("PB", pb, score, f"底层指数/资产估算 PB {pb:.2f}"))
        else:
            results.append(_metric("PB", None, 0, "公募/ETF基金主要跟踪底层资产净值"))
        return results


class GenericValuationStrategy(BaseValuationStrategy):
    name = "generic"


VALUATION_STRATEGIES: dict[str, BaseValuationStrategy] = {
    "bank": BankValuationStrategy(),
    "insurer": InsurerValuationStrategy(),
    "cyclical": CyclicalValuationStrategy(),
    "fund": FundValuationStrategy(),
    "generic": GenericValuationStrategy(),
}


def get_valuation_strategy(model_name: str | None) -> BaseValuationStrategy:
    key = str(model_name).lower() if model_name else "generic"
    return VALUATION_STRATEGIES.get(key, VALUATION_STRATEGIES["generic"])


def valuation_assessments(
    records: dict[str, FundamentalRecord], profile: dict[str, Any]
) -> list[MetricAssessment]:
    strategy = get_valuation_strategy(profile.get("valuation_model"))
    return strategy.assessments(records, profile)


def valuation_range(
    *,
    current_price: float,
    currency: str,
    records: dict[str, FundamentalRecord],
    profile: dict[str, Any],
    override_low: float | None = None,
    override_high: float | None = None,
) -> ValuationRange:
    strategy = get_valuation_strategy(profile.get("valuation_model"))
    return strategy.range(
        current_price=current_price,
        currency=currency,
        records=records,
        profile=profile,
        override_low=override_low,
        override_high=override_high,
    )


def _average_score(items: list[MetricAssessment], preferred: set[str] | None = None) -> float:
    candidates = [item for item in items if item.available]
    if preferred:
        preferred_items = [item for item in candidates if item.name in preferred]
        if preferred_items:
            candidates = preferred_items
    return float(np.mean([item.score for item in candidates])) if candidates else 0.0


def _forecast_score(bundle: ForecastBundle | None) -> float:
    if bundle is None:
        return 0.0
    estimate = bundle.ensemble
    horizon_vol = max(estimate.annualized_volatility * math.sqrt(bundle.horizon_days / 252), 0.02)
    return _clip(
        0.55 * math.tanh(estimate.q50 / horizon_vol) + 0.45 * (estimate.up_probability - 0.5) * 2
    )


def _decision_action(
    *,
    score: float,
    confidence: float,
    data_quality: DataQuality,
    valuation: ValuationRange,
    current_price: float,
    quality_score: float,
    current_weight: float | None,
    position_limit: float,
    investor_allowed: bool,
    forecast_drawdown: float | None,
    maximum_drawdown: float,
) -> str:
    if not investor_allowed:
        return "回避（资金属性或杠杆不符合纪律）"
    if quality_score < -0.35:
        return "回避/重审退出"
    if current_weight is not None and current_weight > position_limit:
        return "停止加仓；复核减仓"
    if forecast_drawdown is not None and forecast_drawdown > maximum_drawdown:
        return "观察；潜在回撤超预算"
    if data_quality is DataQuality.C or confidence < 0.35:
        return "暂不判断"
    if score >= 0.35:
        if valuation.available and valuation.buy_high and current_price > valuation.buy_high:
            return "观察；等待安全边际"
        return "分批买入"
    if score >= 0.05:
        return "持有/观察"
    if score > -0.25:
        return "观察；暂停加仓"
    return "减仓/回避"


def build_decisions(
    *,
    config: AppConfig,
    forecasts: list[ForecastBundle],
    research: ResearchResult,
    technical: list[MetricAssessment],
    quality: list[MetricAssessment],
    valuation: list[MetricAssessment],
    value_range: ValuationRange,
    current_price: float,
    data_quality: DataQuality,
    current_weight: float | None = None,
    role: str = "satellite",
    macro_score: float = 0.0,
) -> list[HorizonDecision]:
    forecast_by_days = {item.horizon_days: item for item in forecasts}
    technical_score = _average_score(technical)
    quality_score = _average_score(quality)
    valuation_score = _average_score(valuation)
    llm_score = research.event_score
    available_fundamental = sum(item.available for item in quality + valuation)
    fundamental_confidence = min(1.0, available_fundamental / 6)
    investor = config.section("investor")
    investor_allowed = bool(investor.get("capital_is_surplus", False)) and not bool(
        investor.get("uses_leverage", True)
    )
    risk = config.section("risk")
    maximum_drawdown = float(risk.get("max_portfolio_drawdown", 0.25))
    short_trade_risk = float(risk.get("short_trade_risk", 0.01))
    position_limit = float(
        risk.get("core_position_limit", 0.35)
        if role == "core"
        else risk.get("satellite_position_limit", 0.15)
    )
    safety_score = 0.0
    if value_range.available and value_range.fair_low and value_range.fair_high:
        if current_price <= value_range.fair_low:
            safety_score = 1.0
        elif current_price >= value_range.fair_high:
            safety_score = -1.0
        else:
            midpoint = (value_range.fair_low + value_range.fair_high) / 2
            half_range = max((value_range.fair_high - value_range.fair_low) / 2, 1e-9)
            safety_score = float(np.clip((midpoint - current_price) / half_range, -1, 1))
    decisions: list[HorizonDecision] = []
    for horizon in Horizon:
        if horizon is Horizon.SHORT:
            forecast = forecast_by_days.get(20)
            deterministic_score = technical_score
            score = _clip(
                _forecast_score(forecast) * 0.30
                + technical_score * 0.35
                + macro_score * 0.20
                + llm_score * 0.15
            )
        elif horizon is Horizon.MEDIUM:
            forecast = forecast_by_days.get(120)
            deterministic_score = quality_score * 0.5 + valuation_score * 0.5
            score = _clip(
                deterministic_score * 0.40
                + technical_score * 0.20
                + macro_score * 0.20
                + llm_score * 0.20
            )
        elif horizon is Horizon.LONG:
            forecast = None
            deterministic_score = quality_score * 0.60 + valuation_score * 0.40
            score = _clip(deterministic_score * 0.65 + macro_score * 0.15 + llm_score * 0.20)
        else:
            forecast = None
            deterministic_score = quality_score * 0.40 + valuation_score * 0.60
            score = _clip(deterministic_score * 0.70 + safety_score * 0.20 + llm_score * 0.10)
        calibration_confidence = 0.4
        warnings: list[str] = []
        if forecast:
            calibration_confidence = (
                0.8
                if forecast.status.value == "active"
                else 0.5
                if forecast.status.value == "experimental"
                else 0.25
            )
            warnings.extend(forecast.warnings)
        llm_confidence = 0.75 if research.events else 0.25
        confidence = float(
            np.clip(
                0.35 * fundamental_confidence
                + 0.35 * calibration_confidence
                + 0.20 * llm_confidence
                + (0.10 if data_quality is not DataQuality.C else 0),
                0,
                1,
            )
        )
        action = _decision_action(
            score=score,
            confidence=confidence,
            data_quality=data_quality,
            valuation=value_range,
            current_price=current_price,
            quality_score=quality_score,
            current_weight=current_weight,
            position_limit=position_limit,
            investor_allowed=investor_allowed,
            forecast_drawdown=forecast.ensemble.potential_drawdown if forecast else None,
            maximum_drawdown=maximum_drawdown,
        )
        if research.status != "ready":
            warnings.append("LLM 事件因子不可用或没有通过证据校验")
        if available_fundamental < 3:
            warnings.append("公司质量与估值数据不完整")
        target_position: float | None = None
        staging: str | None = None
        if action == "分批买入":
            initial = 0.10 if role == "core" else 0.05
            risk_position_limit = position_limit
            if horizon is Horizon.SHORT and forecast:
                risk_position_limit = min(
                    position_limit,
                    short_trade_risk / max(forecast.ensemble.potential_drawdown, 0.01),
                )
            if current_weight is not None and current_weight >= risk_position_limit:
                action = "持有；短线风险预算已满"
                target_position = current_weight
            else:
                target_position = min(
                    position_limit,
                    risk_position_limit,
                    (current_weight or 0) + initial,
                )
                staging = "三批 40%/30%/30%；每批前重新检查估值、事件与组合上限"
        elif current_weight is not None:
            if action.startswith("回避"):
                target_position = 0.0
            elif action == "减仓/回避":
                reduction_fraction = float(risk.get("reduction_target_fraction", 0.50))
                target_position = min(position_limit, current_weight * reduction_fraction)
            elif "复核减仓" in action:
                target_position = position_limit
            else:
                target_position = current_weight
        rationale = (
            f"预测={_forecast_score(forecast):+.2f}，确定性规则={deterministic_score:+.2f}，"
            f"宏观={macro_score:+.2f}，新闻证据={llm_score:+.2f}；综合={score:+.2f}。"
        )
        decisions.append(
            HorizonDecision(
                horizon=horizon,
                score=score,
                confidence=confidence,
                action=action,
                rationale=rationale,
                target_position=target_position,
                staging=staging,
                invalidation_conditions=[
                    "盈利与经营现金流持续背离，或分红依赖举债",
                    "竞争优势、治理诚信或资本配置出现可验证恶化",
                    "估值假设所依赖的盈利/净资产基础发生下修",
                    "仓位、行业集中度、现金底线或组合回撤突破硬约束",
                ],
                warnings=list(dict.fromkeys(warnings)),
            )
        )
    return decisions


def compute_staging_plan(
    *,
    current_price: float,
    valuation_range: ValuationRange,
    price_zones: list[PriceZone],
    role: str = "satellite",
    total_capital: float = 100000.0,
    target_position: float | None = None,
    risk_budget: float = 0.02,
    existing_position_value: float = 0.0,
) -> StagingPlan:
    """Compute an actionable 3-tier staging execution grid with precise share counts."""
    if current_price <= 0:
        return StagingPlan()

    target_weight = (
        target_position if target_position is not None else (0.20 if role == "core" else 0.10)
    )
    allocated_total_budget = max(0.0, total_capital * target_weight - existing_position_value)

    vr = valuation_range
    if vr.available and vr.buy_high and current_price > vr.buy_high:
        t1_price = round(vr.buy_high, 2)
        t1_note = f"现价高于买入线，挂单于估值安全边际买入上限 ({t1_price:.2f}) 建立底仓"
    else:
        t1_price = round(current_price, 2)
        t1_note = "现价处于合理/折价击球区，启动首笔跟踪底仓"

    # 支撑位必须严格位于首笔买入价 t1_price 之下，形成递进阶梯
    valid_supports = [z for z in price_zones if z.kind == "support" and z.high < t1_price]
    valid_supports.sort(key=lambda z: z.center, reverse=True)
    nearest_sup = valid_supports[0] if valid_supports else None
    deeper_sup = valid_supports[1] if len(valid_supports) > 1 else None

    if nearest_sup:
        t2_price = round(nearest_sup.high, 2)
        t2_note = f"在近端强支撑区间上沿 ({nearest_sup.low:.2f}–{nearest_sup.high:.2f}) 挂单加仓"
    elif vr.available and vr.fair_low and vr.fair_low < t1_price:
        t2_price = round(vr.fair_low, 2)
        t2_note = f"在合理价值区间下沿 ({vr.fair_low:.2f}) 挂单加仓"
    else:
        t2_price = round(t1_price * 0.96, 2)
        t2_note = f"在首笔买点下方 -4% ({t2_price:.2f}) 挂单加仓"

    if deeper_sup and deeper_sup.high < t2_price:
        t3_price = round(deeper_sup.center, 2)
        t3_note = f"在次级纵深支撑带 ({deeper_sup.low:.2f}–{deeper_sup.high:.2f}) 挂单"
    elif vr.available and vr.buy_low and vr.buy_low < t2_price:
        t3_price = round(vr.buy_low, 2)
        t3_note = f"在深度估值安全边际下沿 ({vr.buy_low:.2f}) 挂单"
    else:
        t3_price = round(t2_price * 0.95, 2)
        t3_note = f"在加仓买点下方 -5% 深度折价处 ({t3_price:.2f}) 挂单"

    if t2_price >= t1_price:
        t2_price = round(t1_price * 0.96, 2)
        t2_note = f"在首笔买点下方 -4% ({t2_price:.2f}) 挂单加仓"
    if t3_price >= t2_price:
        t3_price = round(t2_price * 0.95, 2)
        t3_note = f"在加仓买点下方 -5% 深度折价处 ({t3_price:.2f}) 挂单"

    tier_weights = [0.30, 0.40, 0.30]
    tier_prices = [t1_price, t2_price, t3_price]
    tier_notes = [t1_note, t2_note, t3_note]
    tier_names = [
        "① 首笔底仓 (Initial)",
        "② 回调加仓 (Dip Support)",
        "③ 极限买点 (Value Floor)",
    ]

    tiers: list[StagingTier] = []
    total_shares = 0
    actual_total_capital = 0.0
    remaining_budget = allocated_total_budget

    for index, (name, w_pct, p, note) in enumerate(
        zip(tier_names, tier_weights, tier_prices, tier_notes, strict=True)
    ):
        tier_budget = allocated_total_budget * w_pct
        shares = int(min(tier_budget, remaining_budget) // (p * 100)) * 100
        if index == 0 and shares == 0 and remaining_budget >= p * 100:
            shares = 100
        allocated = shares * p
        remaining_budget = max(0.0, remaining_budget - allocated)
        total_shares += shares
        actual_total_capital += allocated
        tiers.append(
            StagingTier(
                tier_name=name,
                target_price=p,
                weight_pct=w_pct,
                shares=shares,
                allocated_amount=allocated,
                rationale=note,
            )
        )

    lowest_sup_floor = min((z.low for z in valid_supports), default=t3_price * 0.93)
    inval_price = round(min(lowest_sup_floor, t3_price * 0.94), 2)
    inval_note = f"连续两日有效收盘击穿 {inval_price:.2f} 或基本面恶化时止损/失效"

    return StagingPlan(
        available=True,
        total_target_weight=target_weight,
        total_shares=total_shares,
        total_capital=actual_total_capital,
        tiers=tiers,
        invalidation_price=inval_price,
        invalidation_note=inval_note,
    )


def compute_exit_plan(
    *,
    config: AppConfig,
    symbol: str,
    current_price: float,
    decisions: list[HorizonDecision],
    current_shares: int | None,
    total_assets: float | None,
    current_weight: float | None = None,
    invalidation_price: float | None = None,
    target_weight_override: float | None = None,
    holding_source: str = "未提供",
) -> ExitPlan:
    """Build a deterministic reduce/exit plan without placing an order."""
    warnings = ["执行前须在券商端复核实时报价、可卖数量、已挂委托和成交回报"]
    triggers = list(
        dict.fromkeys(
            condition for decision in decisions for condition in decision.invalidation_conditions
        )
    )
    if current_shares is None or total_assets is None or total_assets <= 0:
        warnings.append("请更新最新持仓快照，或配置账户总资产与 current_shares")
        return ExitPlan(
            current_weight=current_weight,
            current_shares=current_shares,
            reference_price=current_price if current_price > 0 else None,
            total_assets=total_assets,
            holding_source=holding_source,
            trigger_conditions=triggers,
            warnings=warnings,
        )
    if current_shares < 0 or current_price <= 0:
        warnings.append("持股数或参考价格无效")
        return ExitPlan(
            current_weight=current_weight,
            current_shares=current_shares,
            reference_price=current_price if current_price > 0 else None,
            total_assets=total_assets,
            holding_source=holding_source,
            trigger_conditions=triggers,
            warnings=warnings,
        )

    calculated_weight = current_shares * current_price / total_assets
    current_weight = calculated_weight if current_weight is None else current_weight
    if current_shares == 0:
        return ExitPlan(
            status=ExitStatus.HOLD,
            action="当前无持仓，无需卖出",
            current_weight=0.0,
            target_weight=0.0,
            current_shares=0,
            target_shares=0,
            reference_price=current_price,
            total_assets=total_assets,
            holding_source=holding_source,
            trigger_conditions=triggers,
            warnings=warnings,
        )

    profile = config.asset(symbol)
    role = str(profile.get("role", "satellite"))
    risk = config.section("risk")
    position_limit = float(
        risk.get("core_position_limit", 0.35)
        if role == "core"
        else risk.get("satellite_position_limit", 0.15)
    )
    full_exit = [item for item in decisions if item.action.startswith("回避")]
    reductions = [item for item in decisions if "减仓" in item.action]
    reasons: list[str] = []
    target_weight = current_weight
    status = ExitStatus.HOLD
    action = "维持现有仓位；未触发减仓或退出条件"

    if full_exit:
        status = ExitStatus.EXIT
        target_weight = 0.0
        action = "退出持仓；取消未成交买单"
        reasons.extend(f"{HORIZON_LABELS[item.horizon]}：{item.action}" for item in full_exit)
    elif target_weight_override is not None and target_weight_override < current_weight:
        status = ExitStatus.REDUCE
        target_weight = max(0.0, target_weight_override)
        action = "减仓至用户指定目标；取消未成交买单"
        reasons.append(f"当前仓位 {current_weight:.2%} 高于指定目标 {target_weight:.2%}")
    elif reductions or current_weight > position_limit:
        status = ExitStatus.REDUCE
        decision_targets = [
            item.target_position
            for item in reductions
            if item.target_position is not None and item.target_position < current_weight
        ]
        target_weight = min(decision_targets or [position_limit])
        action = "减仓至纪律目标；取消未成交买单"
        if current_weight > position_limit:
            reasons.append(
                f"当前仓位 {current_weight:.2%} 超过 {role} 角色上限 {position_limit:.2%}"
            )
        reasons.extend(f"{HORIZON_LABELS[item.horizon]}：{item.action}" for item in reductions)
    elif invalidation_price is not None and current_price < invalidation_price:
        status = ExitStatus.REVIEW
        action = "暂停买入并复核失效条件；单次跌破不自动卖出"
        reasons.append(f"参考价 {current_price:.2f} 已低于失效线 {invalidation_price:.2f}")

    if status in {ExitStatus.HOLD, ExitStatus.REVIEW}:
        return ExitPlan(
            status=status,
            action=action,
            current_weight=current_weight,
            target_weight=current_weight,
            current_shares=current_shares,
            target_shares=current_shares,
            reference_price=current_price,
            total_assets=total_assets,
            holding_source=holding_source,
            reasons=list(dict.fromkeys(reasons)),
            trigger_conditions=triggers,
            warnings=warnings,
        )

    if status is ExitStatus.EXIT or target_weight <= 0:
        target_shares = 0
        sell_shares = current_shares
    else:
        raw_target_shares = max(0, math.floor(total_assets * target_weight / current_price))
        if symbol.startswith("CN:"):
            target_shares = (raw_target_shares // 100) * 100
        else:
            target_shares = raw_target_shares
        target_shares = min(current_shares, target_shares)
        sell_shares = current_shares - target_shares
        if symbol.startswith("CN:") and 0 < sell_shares < current_shares:
            sell_shares = min(current_shares, math.ceil(sell_shares / 100) * 100)
            target_shares = current_shares - sell_shares
    actual_target_weight = target_shares * current_price / total_assets
    return ExitPlan(
        status=status,
        action=action,
        current_weight=current_weight,
        target_weight=actual_target_weight,
        current_shares=current_shares,
        target_shares=target_shares,
        sell_shares=sell_shares,
        reference_price=current_price,
        estimated_proceeds=sell_shares * current_price,
        total_assets=total_assets,
        holding_source=holding_source,
        reasons=list(dict.fromkeys(reasons)),
        trigger_conditions=triggers,
        warnings=warnings,
    )


def analyze_package(
    *,
    config: AppConfig,
    database: Database,
    symbol: str,
    as_of: date,
    frame: pd.DataFrame,
    data_quality: DataQuality,
    data_warnings: list[str],
    forecasts: list[ForecastBundle],
    research: ResearchResult,
    current_weight: float | None = None,
    current_shares: int | None = None,
    total_assets: float | None = None,
    target_weight_override: float | None = None,
    holding_source: str = "未提供",
    fair_value_low: float | None = None,
    fair_value_high: float | None = None,
) -> AnalysisPackage:
    if frame.empty:
        raise ValueError("没有可分析行情")
    price = float(frame.iloc[-1]["close"])
    currency = str(frame.iloc[-1].get("currency", ""))
    profile = config.asset(symbol)
    records = database.latest_fundamentals(symbol, as_of)
    technical = technical_assessments(frame)
    quality = quality_assessments(records)
    valuation = valuation_assessments(records, profile)
    macro_rows, macro_score = macro_assessments(database, as_of, macro_exposures(profile))
    macro = [
        _metric(
            str(item["name"]),
            float(item["value"]),
            float(item["score"]),
            f"最新值 {float(item['value']):.3f}，最近一期变化 "
            f"{float(item['momentum']):+.2%}，观测截至 {item.get('latest_date', '未知')} "
            f"（陈旧 {float(item.get('stale_days', 0)):.0f} 天），"
            f"暴露权重 {float(item['exposure']):+.0%}",
        )
        for item in macro_rows
    ]
    value_range = valuation_range(
        current_price=price,
        currency=currency,
        records=records,
        profile=profile,
        override_low=fair_value_low,
        override_high=fair_value_high,
    )
    chart_config = config.section("charts")
    price_zones = (
        detect_price_zones(
            frame,
            as_of=as_of,
            lookback_sessions=int(chart_config.get("level_lookback_sessions", 360)),
            max_per_side=int(chart_config.get("level_max_per_side", 2)),
        )
        if bool(chart_config.get("include_levels", True))
        else []
    )
    price_zone_validation = (
        validate_price_zones(
            frame,
            as_of=as_of,
            lookback_sessions=int(chart_config.get("level_lookback_sessions", 360)),
            evaluation_horizon=int(chart_config.get("level_validation_horizon", 20)),
            step=int(chart_config.get("level_validation_step", 10)),
            max_windows=int(chart_config.get("level_validation_max_windows", 60)),
        )
        if price_zones and bool(chart_config.get("validate_levels", True))
        else []
    )
    decisions = build_decisions(
        config=config,
        forecasts=forecasts,
        research=research,
        technical=technical,
        quality=quality,
        valuation=valuation,
        value_range=value_range,
        current_price=price,
        data_quality=data_quality,
        current_weight=current_weight,
        role=str(profile.get("role", "satellite")),
        macro_score=macro_score,
    )
    nearest_support = next((zone for zone in price_zones if zone.kind == "support"), None)
    nearest_resistance = next((zone for zone in price_zones if zone.kind == "resistance"), None)
    for decision in decisions:
        if decision.horizon not in {Horizon.SHORT, Horizon.MEDIUM}:
            continue
        if nearest_support:
            decision.invalidation_conditions.append(
                f"连续两日收盘低于 {nearest_support.low:.2f}，当前支撑带视为失效"
            )
        if nearest_resistance:
            decision.invalidation_conditions.append(
                f"突破 {nearest_resistance.high:.2f} 后若无法站稳，仍按压力带处理"
            )
    staging_plan = compute_staging_plan(
        current_price=price,
        valuation_range=value_range,
        price_zones=price_zones,
        role=str(profile.get("role", "satellite")),
        total_capital=total_assets or 100000.0,
        target_position=decisions[0].target_position if decisions else None,
        existing_position_value=(current_shares or 0) * price,
    )
    exit_plan = compute_exit_plan(
        config=config,
        symbol=symbol,
        current_price=price,
        decisions=decisions,
        current_shares=current_shares,
        total_assets=total_assets,
        current_weight=current_weight,
        invalidation_price=staging_plan.invalidation_price,
        target_weight_override=target_weight_override,
        holding_source=holding_source,
    )
    if exit_plan.status in {ExitStatus.REVIEW, ExitStatus.REDUCE, ExitStatus.EXIT}:
        staging_plan.available = False
    return AnalysisPackage(
        symbol=symbol,
        name=str(profile.get("name", symbol)),
        as_of=as_of,
        current_price=price,
        currency=currency,
        data_quality=data_quality,
        data_warnings=data_warnings,
        forecasts=forecasts,
        research=research,
        technical=technical,
        quality=quality,
        valuation=valuation,
        valuation_range=value_range,
        decisions=decisions,
        macro=macro,
        macro_score=macro_score,
        price_zones=price_zones,
        price_zone_validation=price_zone_validation,
        staging_plan=staging_plan,
        exit_plan=exit_plan,
    )


def create_receipts(database: Database, package: AnalysisPackage) -> list[str]:
    decision_by_horizon = {item.horizon: item for item in package.decisions}
    ids: list[str] = []
    for bundle in package.forecasts:
        horizon = Horizon.SHORT if bundle.horizon_days <= 20 else Horizon.MEDIUM
        decision = decision_by_horizon[horizon]
        receipt_id = (
            "fc-"
            + hashlib.sha256(
                f"{package.symbol}:{package.as_of}:{bundle.horizon_days}".encode()
            ).hexdigest()[:16]
        )
        database.save_receipt(
            {
                "id": receipt_id,
                "symbol": package.symbol,
                "created_at": utc_now().isoformat(),
                "as_of": package.as_of.isoformat(),
                "horizon_days": bundle.horizon_days,
                "due_date": bundle.due_date.isoformat(),
                "model_status": bundle.status.value,
                "forecast": {
                    "ensemble": bundle.ensemble.model_dump(mode="json"),
                    "components": {
                        key: value.model_dump(mode="json")
                        for key, value in bundle.components.items()
                    },
                    "weights": bundle.weights,
                    "data_quality": bundle.data_quality.value,
                    "warnings": bundle.warnings,
                },
                "decision": decision.model_dump(mode="json"),
                "evidence": [item.id for item in package.research.evidence],
            }
        )
        ids.append(receipt_id)
    package.receipt_ids = ids
    return ids


def _fmt_percent(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}%}"


def _fmt_number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_analysis_markdown(package: AnalysisPackage) -> str:
    from stock_analysis.renderers import render_analysis_markdown as _render

    return _render(package)


def _parse_number(text: str) -> float:
    cleaned = text.replace(",", "").replace("+", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else 0.0


def latest_portfolio_snapshot(config: AppConfig) -> PortfolioSnapshot:
    candidates = sorted((config.home / "01-持仓").glob("*-持仓快照.md"), reverse=True)
    if not candidates:
        raise FileNotFoundError("01-持仓 中没有持仓快照")
    path = candidates[0]
    text = path.read_text(encoding="utf-8")
    date_match = re.search(r"^date:\s*(\d{4}-\d{2}-\d{2})", text, flags=re.MULTILINE)
    as_of = (
        date.fromisoformat(date_match.group(1))
        if date_match
        else date.fromisoformat(path.name[:10])
    )
    total_match = re.search(r"已记录人民币资产暂为\s*\*\*([\d,.]+)(?:\*\*)?\s*元", text)
    total_assets = _parse_number(total_match.group(1)) if total_match else None
    positions: list[PortfolioPosition] = []
    section_match = re.search(r"## A 股明细\n(.*?)(?=\n## )", text, flags=re.DOTALL)
    if section_match:
        rows = [line for line in section_match.group(1).splitlines() if line.startswith("|")]
        for line in rows[2:]:
            cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
            if len(cells) < 5 or cells[0] in {"合计", ""}:
                continue
            name = cells[0]
            symbol = config.symbol_for_name(name)
            profile = config.asset(symbol) if symbol else {}
            positions.append(
                PortfolioPosition(
                    symbol=symbol,
                    name=name,
                    quantity=_parse_number(cells[1].split("/")[0]),
                    market_value=_parse_number(cells[4]),
                    sector=str(profile.get("sector", "未分类")),
                    role=str(profile.get("role", "satellite")),
                )
            )
    cash = 0.0
    cash_found = False
    view_match = re.search(r"## 人民币资产视图\n(.*?)(?=\n## |\Z)", text, flags=re.DOTALL)
    if view_match:
        for line in view_match.group(1).splitlines():
            if not line.startswith("|"):
                continue
            cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
            if len(cells) >= 2 and ("现金" in cells[0] or "赎回款" in cells[0]):
                cash += _parse_number(cells[1])
                cash_found = True
    warnings: list[str] = []
    if total_assets is None:
        warnings.append("未解析到完整人民币资产合计，仓位只能按已解析资产估算")
    if any(item.symbol is None for item in positions):
        warnings.append("部分持仓名称没有映射到标准证券代码")
    warnings.append("港币和美元资产未折算，不作为完整组合权重")
    return PortfolioSnapshot(
        path=path,
        as_of=as_of,
        total_cny_assets=total_assets,
        cash_cny=cash if cash_found else None,
        positions=positions,
        warnings=warnings,
    )


def position_weight(snapshot: PortfolioSnapshot, symbol: str) -> float | None:
    if not snapshot.total_cny_assets:
        return None
    for item in snapshot.positions:
        if item.symbol == symbol:
            return item.market_value / snapshot.total_cny_assets
    return 0.0


def position_for_symbol(snapshot: PortfolioSnapshot, symbol: str) -> PortfolioPosition | None:
    return next((item for item in snapshot.positions if item.symbol == symbol), None)


def resolve_holding_context(config: AppConfig, symbol: str) -> tuple[int | None, float | None, str]:
    """Resolve holdings from the newest snapshot, then fall back to local config."""
    try:
        snapshot = latest_portfolio_snapshot(config)
        if snapshot.total_cny_assets and symbol.startswith(("CN:", "CNFUND:")):
            position = position_for_symbol(snapshot, symbol)
            if position:
                shares = int(position.quantity)
            elif symbol.startswith("CN:"):
                shares = 0
            else:
                configured_shares = config.asset(symbol).get("current_shares")
                shares = int(configured_shares) if configured_shares is not None else None
            return (
                shares,
                snapshot.total_cny_assets,
                f"持仓快照 {snapshot.path.name}（{snapshot.as_of.isoformat()}）",
            )
    except (FileNotFoundError, OSError, ValueError):
        pass

    profile = config.asset(symbol)
    configured_shares = profile.get("current_shares")
    try:
        shares = int(configured_shares) if configured_shares is not None else None
    except (TypeError, ValueError):
        shares = None
    market = symbol.split(":", 1)[0]
    key = {
        "CN": "cn_account_assets",
        "CNFUND": "cn_account_assets",
        "HK": "hk_account_assets",
        "US": "us_account_assets",
    }.get(market)
    portfolio = config.section("portfolio")
    raw_assets = portfolio.get(key) if key else None
    try:
        total_assets = float(raw_assets) if raw_assets is not None else None
    except (TypeError, ValueError):
        total_assets = None
    as_of = portfolio.get(f"{key}_as_of", "日期未注明") if key else "日期未注明"
    return shares, total_assets, f"配置 current_shares / portfolio.{key}（截至 {as_of}）"


def render_portfolio_markdown(
    snapshot: PortfolioSnapshot, config: AppConfig, database: Database
) -> str:
    from stock_analysis.renderers import render_portfolio_markdown as _render

    return _render(snapshot, config, database)
