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
    forecast = next((item for item in package.forecasts if item.horizon_days == 20), None)
    if forecast:
        last_x = float(x[-1])
        future_x = np.array([last_x, last_x + forecast.horizon_days])
        current = float(close.iloc[-1])
        lower = np.array([current, current * (1 + forecast.ensemble.q10)])
        upper = np.array([current, current * (1 + forecast.ensemble.q90)])
        median = np.array([current, current * (1 + forecast.ensemble.q50)])
        price_axis.fill_between(
            future_x,
            lower,
            upper,
            color="#9467bd",
            alpha=0.14,
            label="20日 Q10–Q90",
        )
        price_axis.plot(future_x, median, "--", color="#9467bd", linewidth=1.1)
        price_axis.set_xlim(-1, last_x + forecast.horizon_days + 1)
    price_axis.set_title(
        f"{package.name}（{package.symbol}） 日线 / "
        f"截止 {package.as_of} / 数据 {package.data_quality.value}"
    )
    price_axis.set_ylabel(package.currency or "Price")
    price_axis.grid(alpha=0.18)
    price_axis.legend(loc="upper left", ncol=3, fontsize=8)
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
