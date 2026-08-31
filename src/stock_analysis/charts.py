from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd

from stock_analysis.decision import AnalysisPackage
from stock_analysis.indicators import add_indicators


def available() -> bool:
    return find_spec("matplotlib") is not None


def render_stock_chart(
    frame: pd.DataFrame,
    package: AnalysisPackage,
    output_path: Path,
    *,
    sessions: int = 180,
) -> Path:
    if not available():
        raise RuntimeError("未安装图表依赖；请运行 uv sync --extra charts")
    if frame.empty:
        raise ValueError("没有可绘制行情")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    matplotlib.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    data = add_indicators(frame).tail(sessions).reset_index(drop=True)
    x = np.arange(len(data), dtype=float)
    close = data["close"].astype(float)
    open_ = data["open"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    rising = close >= open_
    colors = np.where(rising, "#d62728", "#1a9850")
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [4.2, 1.2, 1.5, 1.2], "hspace": 0.08},
    )
    price_axis, volume_axis, macd_axis, rsi_axis = axes
    price_axis.vlines(x, low, high, color=colors, linewidth=0.8, alpha=0.9)
    body_bottom = np.minimum(open_, close)
    body_height = np.maximum((close - open_).abs(), close.abs() * 0.001)
    price_axis.bar(x, body_height, bottom=body_bottom, width=0.62, color=colors, alpha=0.85)
    price_axis.plot(x, data["ma20"], color="#f5a623", linewidth=1.2, label="MA20")
    price_axis.plot(x, data["ma60"], color="#4169e1", linewidth=1.2, label="MA60")
    value_range = package.valuation_range
    if value_range.available and value_range.fair_low and value_range.fair_high:
        price_axis.axhspan(
            value_range.fair_low,
            value_range.fair_high,
            color="#4c78a8",
            alpha=0.08,
            label="合理价值区间",
        )
        if value_range.buy_low and value_range.buy_high:
            price_axis.axhspan(
                value_range.buy_low,
                value_range.buy_high,
                color="#2ca02c",
                alpha=0.10,
                label="分批观察区间",
            )
    zone_handles: list[Patch] = []
    for kind, color, label in (
        ("support", "#4c78a8", "支撑带"),
        ("resistance", "#f2a541", "压力带"),
    ):
        selected = [zone for zone in package.price_zones if zone.kind == kind]
        if selected:
            zone_handles.append(Patch(facecolor=color, alpha=0.18, label=label))
        for zone in selected:
            price_axis.axhspan(
                zone.low,
                zone.high,
                color=color,
                alpha=0.06 + zone.strength * 0.10,
            )
            price_axis.axhline(zone.center, color=color, linewidth=0.8, linestyle="--")
            price_axis.text(
                len(data) + 1,
                zone.center,
                f"{label} {zone.low:.2f}–{zone.high:.2f} 强度{zone.strength * 100:.0f}",
                color=color,
                fontsize=7,
                va="center",
            )
    price_axis.set_xlim(-1, len(data) + 17)
    price_axis.set_title(
        f"{package.name}（{package.symbol}） 日线 / "
        f"截止 {package.as_of} / 数据 {package.data_quality.value}"
    )
    price_axis.set_ylabel(package.currency or "Price")
    price_axis.grid(alpha=0.18)
    handles, labels = price_axis.get_legend_handles_labels()
    price_axis.legend(
        handles + zone_handles,
        labels + [item.get_label() for item in zone_handles],
        loc="upper left",
        ncol=3,
        fontsize=8,
    )
    volume_axis.bar(x, data["volume"].astype(float), color=colors, width=0.62, alpha=0.72)
    volume_axis.set_ylabel("Volume")
    volume_axis.grid(alpha=0.15)
    macd_axis.plot(x, data["macd"], color="#1565c0", linewidth=1.0, label="DIF")
    macd_axis.plot(x, data["macd_signal"], color="#ff8f00", linewidth=1.0, label="DEA")
    macd_colors = np.where(data["macd_hist"] >= 0, "#d62728", "#1a9850")
    macd_axis.bar(x, data["macd_hist"], color=macd_colors, width=0.65, alpha=0.65)
    macd_axis.axhline(0, color="#777777", linewidth=0.6)
    macd_axis.set_ylabel("MACD")
    macd_axis.legend(loc="upper left", ncol=2, fontsize=8)
    macd_axis.grid(alpha=0.15)
    rsi_axis.plot(x, data["rsi14"], color="#7b1fa2", linewidth=1.0, label="RSI14")
    rsi_axis.axhspan(70, 100, color="#d62728", alpha=0.06)
    rsi_axis.axhspan(0, 30, color="#1a9850", alpha=0.06)
    rsi_axis.axhline(70, color="#999999", linewidth=0.6, linestyle="--")
    rsi_axis.axhline(30, color="#999999", linewidth=0.6, linestyle="--")
    rsi_axis.set_ylim(0, 100)
    rsi_axis.set_ylabel("RSI")
    rsi_axis.grid(alpha=0.15)
    ticks = np.linspace(0, len(data) - 1, min(7, len(data)), dtype=int)
    labels = [pd.Timestamp(data.iloc[index]["trade_date"]).strftime("%Y-%m-%d") for index in ticks]
    rsi_axis.set_xticks(ticks)
    rsi_axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    legend_handles = [
        Patch(facecolor="#d62728", label="上涨"),
        Patch(facecolor="#1a9850", label="下跌"),
    ]
    volume_axis.legend(handles=legend_handles, loc="upper left", ncol=2, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format=output_path.suffix.lstrip("."), bbox_inches="tight", dpi=150)
    plt.close(figure)
    return output_path


def render_probability_chart(package: AnalysisPackage, output_path: Path) -> Path:
    """Render nested forecast intervals and calibrated directional probability.

    The left axis is cumulative return and the right axis is the model's
    probability of a positive return.  Support/resistance and valuation bands
    are converted from price to return relative to the current price so that
    the chart remains comparable across instruments.  None of these overlays
    is a deterministic price path.
    """
    if not available():
        raise RuntimeError("未安装图表依赖；请运行 uv sync --extra charts")
    if not package.forecasts:
        raise ValueError("没有可绘制概率预测")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from stock_analysis.forecast import ModelStatus, probability_fan

    matplotlib.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fan = probability_fan(package.forecasts)
    if fan.empty:
        raise ValueError("没有可绘制概率区间")
    figure, axis = plt.subplots(figsize=(12, 6.8))
    day = fan["day"].to_numpy(dtype=float)
    q10 = fan["q10"].to_numpy(dtype=float)
    q25 = fan["q25"].to_numpy(dtype=float)
    q50 = fan["q50"].to_numpy(dtype=float)
    q75 = fan["q75"].to_numpy(dtype=float)
    q90 = fan["q90"].to_numpy(dtype=float)
    axis.fill_between(
        day,
        q10,
        q90,
        color="#4c78a8",
        alpha=0.13,
        label="名义80%分位区间（Q10–Q90）",
    )
    axis.fill_between(
        day,
        q25,
        q75,
        color="#4c78a8",
        alpha=0.28,
        label="近似50%区间（Q25–Q75）",
    )
    axis.plot(day, q50, color="#1f4e79", linewidth=2.0, label="中位路径（Q50）")
    axis.plot(day, q10, color="#4c78a8", linewidth=0.7, linestyle=":")
    axis.plot(day, q90, color="#4c78a8", linewidth=0.7, linestyle=":")
    axis.axhline(0, color="#555555", linewidth=0.8)

    # Map price levels to the same return scale as the forecast.  This makes
    # it immediately visible when a forecast band overlaps a nearby level,
    # without presenting the level as a target price.
    current_price = float(package.current_price)
    if current_price > 0:
        for zone in package.price_zones:
            level = zone.center / current_price - 1
            color = "#4c78a8" if zone.kind == "support" else "#f2a541"
            label = "支撑" if zone.kind == "support" else "压力"
            axis.axhline(level, color=color, linewidth=0.9, linestyle="--", alpha=0.8)
            axis.annotate(
                f"{label} {zone.center:.2f}",
                xy=(day[-1], level),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=color,
            )
        value_range = package.valuation_range
        if value_range.available and value_range.fair_low and value_range.fair_high:
            axis.axhspan(
                value_range.fair_low / current_price - 1,
                value_range.fair_high / current_price - 1,
                color="#4169e1",
                alpha=0.06,
                label="合理价值映射区间",
            )
            if value_range.buy_low and value_range.buy_high:
                axis.axhspan(
                    value_range.buy_low / current_price - 1,
                    value_range.buy_high / current_price - 1,
                    color="#2ca02c",
                    alpha=0.08,
                    label="分批观察映射区间",
                )

    # Directional probability is interpolated only for visual continuity; the
    # actual model outputs remain the five annotated horizon anchors.
    probability_axis = axis.twinx()
    probability_days = np.array([0.0, *[float(item.horizon_days) for item in package.forecasts]])
    probability_values = np.array(
        [0.5, *[float(item.ensemble.up_probability) for item in package.forecasts]]
    )
    order = np.argsort(probability_days)
    probability_curve = np.interp(
        day,
        probability_days[order],
        probability_values[order],
    )
    probability_axis.plot(
        day,
        probability_curve,
        color="#7b1fa2",
        linewidth=1.2,
        linestyle="-.",
        label="上涨概率（右轴）",
    )
    probability_axis.scatter(
        [item.horizon_days for item in package.forecasts],
        [item.ensemble.up_probability for item in package.forecasts],
        color="#7b1fa2",
        s=18,
        zorder=6,
    )
    probability_axis.axhline(0.5, color="#7b1fa2", linewidth=0.6, alpha=0.45)
    probability_axis.set_ylim(0, 1)
    probability_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    probability_axis.set_ylabel("上涨概率", color="#7b1fa2")
    probability_axis.tick_params(axis="y", colors="#7b1fa2")
    marker_map = {
        ModelStatus.ACTIVE: ("o", "#1f4e79"),
        ModelStatus.EXPERIMENTAL: ("o", "white"),
        ModelStatus.DEGRADED: ("X", "#f2a541"),
        ModelStatus.DISABLED: ("x", "#777777"),
    }
    for index, bundle in enumerate(sorted(package.forecasts, key=lambda item: item.horizon_days)):
        estimate = bundle.ensemble
        marker, face = marker_map[bundle.status]
        axis.scatter(
            [bundle.horizon_days],
            [estimate.q50],
            marker=marker,
            s=48,
            facecolors=face,
            edgecolors="#1f4e79",
            linewidths=1.1,
            zorder=5,
        )
        vertical = 13 if index % 2 == 0 else -28
        axis.annotate(
            f"{bundle.horizon_days}日  上涨{estimate.up_probability:.0%}\n"
            f"{bundle.status.value} {bundle.calibration_samples}/{bundle.calibration_target}",
            (bundle.horizon_days, estimate.q50),
            xytext=(0, vertical),
            textcoords="offset points",
            ha="center",
            va="bottom" if vertical > 0 else "top",
            fontsize=7,
            color="#333333",
        )
    axis.set_title(
        f"{package.name}（{package.symbol}）未来收益概率区间\n"
        f"截止 {package.as_of} / 数据 {package.data_quality.value} / 非确定价格路径"
    )
    axis.set_xlabel("未来交易日")
    axis.set_ylabel("相对当前价格的累计收益")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.18)
    left_handles, left_labels = axis.get_legend_handles_labels()
    right_handles, right_labels = probability_axis.get_legend_handles_labels()
    axis.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="upper left",
        fontsize=8,
        ncol=2,
    )
    figure.text(
        0.5,
        0.006,
        "仅 5/10/20/60/120 日为模型输出；中间日期按对数收益分位数插值，"
        "上涨概率曲线同样只是锚点之间的可视化插值。\n"
        "带状区域是名义概率边界；实验期限未完成覆盖率校准，不是可实现价格路线或收益保证。",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(right=0.88)
    figure.savefig(output_path, format=output_path.suffix.lstrip("."), bbox_inches="tight", dpi=150)
    plt.close(figure)
    return output_path
