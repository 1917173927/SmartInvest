from __future__ import annotations

import contextlib
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from stock_analysis.data import (
    AppConfig,
    Database,
    DataQuality,
    Instrument,
    quality_summary,
    total_return_frame,
)
from stock_analysis.decision import (
    AnalysisPackage,
    StagingPlan,
    analyze_package,
    compute_staging_plan,
)
from stock_analysis.files import atomic_write_text
from stock_analysis.indicators import macro_assessments
from stock_analysis.research import run_research

LOGGER = logging.getLogger(__name__)


@dataclass
class MorningItem:
    canonical: str
    name: str
    sector: str
    role: str
    current_price: float
    currency: str
    quality: DataQuality
    valuation_status: str
    dist_to_buy: float | None
    plan: StagingPlan
    package: AnalysisPackage
    priority: str


@dataclass
class MorningBrief:
    as_of: date
    total_capital: float
    items: list[MorningItem] = field(default_factory=list)
    macro_score: float = 0.0
    macro_items: list[dict[str, Any]] = field(default_factory=list)
    report_path: Path | None = None


def generate_morning_brief(
    config: AppConfig,
    *,
    total_capital: float = 100000.0,
    as_of: date | None = None,
    send_notification: bool = True,
) -> MorningBrief:
    """Generate a high-conviction pre-market action briefing for 09:15 order placement."""
    today = as_of or date.today()
    database = Database(config.db_path)
    brief = MorningBrief(as_of=today, total_capital=total_capital)

    # 1. Macro assessments
    macro_items, macro_score = macro_assessments(database, as_of=today)
    brief.macro_items = macro_items
    brief.macro_score = macro_score

    # 2. Iterate through all configured assets
    assets = config.section("assets")
    for raw_symbol, profile in assets.items():
        try:
            canonical = Instrument.parse(raw_symbol).canonical
        except Exception:
            canonical = raw_symbol
        name = str(profile.get("name", canonical))
        sector = str(profile.get("sector", "未分类"))
        role = str(profile.get("role", "satellite"))

        bars = database.load_bars(canonical, today)
        if bars.empty:
            continue
        actions = database.load_actions(canonical)
        frame, _ = total_return_frame(bars, actions)
        if frame.empty:
            continue

        current_price = float(bars.iloc[-1]["close"])
        currency = str(bars.iloc[-1].get("currency", "CNY"))
        quality, _ = quality_summary(bars, today)

        research = run_research(
            database=database,
            config=config,
            symbol=canonical,
            as_of=today,
            use_llm=False,
        )
        pkg = analyze_package(
            config=config,
            database=database,
            symbol=canonical,
            as_of=today,
            frame=frame,
            data_quality=quality,
            data_warnings=[],
            forecasts=[],
            research=research,
        )

        assigned_weight = pkg.decisions[0].target_position if pkg.decisions else None
        plan = compute_staging_plan(
            current_price=current_price,
            valuation_range=pkg.valuation_range,
            price_zones=pkg.price_zones,
            role=role,
            total_capital=total_capital,
            target_position=assigned_weight,
        )

        vr = pkg.valuation_range
        dist_to_buy = (current_price / vr.buy_high - 1) if (vr.available and vr.buy_high) else None

        if vr.available and vr.buy_high and current_price <= vr.buy_high:
            val_status = "🟢 进入安全买入区"
            priority = "🔥 优先建仓"
        elif vr.available and vr.fair_low and current_price <= vr.fair_low:
            val_status = "🟡 合理偏低带"
            priority = "👀 跟踪观察"
        else:
            val_status = "⚪ 高于买入线"
            priority = "✋ 暂缓观望"

        brief.items.append(
            MorningItem(
                canonical=canonical,
                name=name,
                sector=sector,
                role=role,
                current_price=current_price,
                currency=currency,
                quality=quality,
                valuation_status=val_status,
                dist_to_buy=dist_to_buy,
                plan=plan,
                package=pkg,
                priority=priority,
            )
        )

    database.close()

    # Sort items by priority: Priority first, then distance to buy
    def _sort_key(item: MorningItem) -> tuple[int, float]:
        rank = 0 if "优先" in item.priority else 1 if "观察" in item.priority else 2
        dist = item.dist_to_buy if item.dist_to_buy is not None else 999.0
        return (rank, dist)

    brief.items.sort(key=_sort_key)

    # 3. Render Markdown Report
    lines = [
        "---",
        "type: morning-brief",
        f"date: {today.isoformat()}",
        f"capital: {total_capital}",
        "---",
        "",
        f"# 🌅 SmartInvest 盘前挂单与执行晨报（{today.isoformat()}）",
        "",
        "> [!tip] 执行纪律与时点",
        "> - **9:15–9:25（集合竞价前）**：核对挂单价格与建议手数，在券商 App 预埋限价单或条件单。",
        "> - **盘中**：由券商系统自动被动成交，严禁盘中情绪化追涨杀跌。",
        f"> - **基准资金规模**：{total_capital:,.2f} CNY 动态测算。",
        "",
        "## 🎯 今日券商 App 挂单与条件单预埋清单",
        "",
    ]

    for it in brief.items:
        dist_text = f"（距买入线 {it.dist_to_buy:+.1%}）" if it.dist_to_buy is not None else ""
        lines.extend(
            [
                f"### {it.priority}：{it.name} ({it.canonical}) {dist_text}",
                f"- **现价**：{it.current_price:.2f} {it.currency} | "
                f"**角色**：{it.role} | **行业**：{it.sector} | "
                f"**估值状态**：{it.valuation_status}",
                "",
                "| 批次 | 目标价 | 建议手数 | 建议股数 | 占用资金 | 券商下单类型 | 执行逻辑 |",
                "|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for tier in it.plan.tiers:
            lots = tier.shares // 100
            order_type = (
                "集合竞价/开盘限价单"
                if "首笔" in tier.tier_name
                else "回调触达条件单"
                if "强支撑" in tier.tier_name
                else "低位限价埋单"
            )
            lines.append(
                f"| {tier.tier_name} | {tier.target_price:.2f} {it.currency} | "
                f"**{lots} 手** | {tier.shares} 股 | {tier.allocated_amount:,.2f} | "
                f"`{order_type}` | {tier.rationale} |"
            )
        lines.append("")

        if it.plan.invalidation_price:
            lines.extend(
                [
                    "",
                    f"> [!danger] 🛑 券商硬止损预警线："
                    f"**< {it.plan.invalidation_price:.2f} {it.currency}** "
                    f"（{it.plan.invalidation_note}）。",
                    "",
                ]
            )

    # Macro Section
    lines.extend(["## 🌍 隔夜外盘与宏观环境速览", ""])
    if brief.macro_items:
        lines.extend(
            [
                "| 宏观资产 | 最新观测值 | 动量变化 | 宏观环境打分 | 观察日期 |",
                "|---|---:|---:|:---:|---|",
            ]
        )
        for m in brief.macro_items:
            score_symbol = "🟢" if m["score"] > 0.2 else "🔴" if m["score"] < -0.2 else "⚪"
            lines.append(
                f"| {m['name']} | {m['value']:.2f} | {m['momentum']:+.2%} | "
                f"{score_symbol} {m['score']:+.2f} | {m['latest_date']} |"
            )
        lines.append("")
    else:
        lines.extend(["- 暂无最新宏观观测数据更新。", ""])

    # Write report files
    out_dir = config.home / "06-自动分析" / "盘前晨报"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"{today.isoformat()}-盘前挂单晨报.md"
    atomic_write_text(report_file, "\n".join(lines) + "\n")
    brief.report_path = report_file

    latest_file = config.home / "06-自动分析" / "最新盘前挂单晨报.md"
    atomic_write_text(latest_file, "\n".join(lines) + "\n")

    # 4. macOS Native Notification
    if send_notification and platform.system() == "Darwin":
        ready_names = [it.name for it in brief.items if "优先" in it.priority]
        notify_body = (
            f"今日优先建仓: {', '.join(ready_names)}" if ready_names else "今日标的处于正常观察区间"
        )
        sub_title = "请在 9:15 集合竞价前查看预埋单"
        cmd = (
            f'display notification "{notify_body}" '
            f'with title "SmartInvest 盘前挂单晨报已生成" subtitle "{sub_title}"'
        )
        with contextlib.suppress(Exception):
            subprocess.run(["osascript", "-e", cmd], check=False, capture_output=True)

    return brief


def item_action(item: MorningItem) -> str:
    return item.priority
