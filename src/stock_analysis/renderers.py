from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stock_analysis.automation import AutomationSummary
    from stock_analysis.data import AppConfig, Database
    from stock_analysis.decision import (
        AnalysisPackage,
        PortfolioSnapshot,
    )


def _fmt_percent(value: float | None, digits: int = 1) -> str:
    return f"{value * 100:+.{digits}f}%" if value is not None and math.isfinite(value) else "—"


def _fmt_number(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None and math.isfinite(value) else "—"


def render_executive_summary_card(package: AnalysisPackage) -> list[str]:
    """Render a 3-point executive summary card for quick decision-making."""
    # 1. Contradiction & Balance (Core driver vs headwind)
    vr = package.valuation_range
    valuation_part = "估值中性"
    valuation_badge = "⚪"
    if vr.available and vr.buy_high:
        if package.current_price <= vr.buy_high:
            dist_pct = package.current_price / vr.buy_high - 1
            valuation_part = f"处于安全边际买入区间 (距买入上限 {dist_pct:+.1%})"
            valuation_badge = "🟢"
        elif vr.fair_low and package.current_price <= vr.fair_low:
            valuation_part = "处于合理偏低价值区间"
            valuation_badge = "🟡"
        else:
            dist_pct = package.current_price / vr.buy_high - 1
            valuation_part = f"高于保守买入线 (+{dist_pct:.1%}，需等待安全边际)"
            valuation_badge = "⚪"

    tech_notes: list[str] = []
    for m in package.technical:
        if m.name == "RSI14" and m.value is not None:
            if m.value > 0.70:
                tech_notes.append(f"RSI({m.value * 100:.0f})偏热")
            elif m.value < 0.35:
                tech_notes.append(f"RSI({m.value * 100:.0f})超卖")
        elif m.name == "MACD 动能" and m.available and m.score > 0.15:
            tech_notes.append("动能向上")
        elif m.name == "MACD 动能" and m.available and m.score < -0.15:
            tech_notes.append("动能向下")

    # Near-term support/resistance
    sr_notes: list[str] = []
    nearest_res = next((z for z in package.price_zones if z.kind == "resistance"), None)
    nearest_sup = next((z for z in package.price_zones if z.kind == "support"), None)
    if nearest_res:
        sr_notes.append(f"上方阻力 {nearest_res.center:.2f}({nearest_res.distance:+.1%})")
    if nearest_sup:
        sr_notes.append(f"下方支撑 {nearest_sup.center:.2f}({nearest_sup.distance:+.1%})")

    tech_summary = (
        "；".join([*tech_notes, *sr_notes]) if (tech_notes or sr_notes) else "技术形态平稳"
    )

    point1 = (
        f"**核心特征与多空评估**：{valuation_badge} {valuation_part}；{tech_summary}"
        f"（宏观因子分 {package.macro_score:+.2f}）。"
    )

    # 2. Multi-horizon Action Guidance
    short_d = next((d for d in package.decisions if d.horizon == "short"), None)
    medium_d = next((d for d in package.decisions if d.horizon == "medium"), None)
    long_d = next((d for d in package.decisions if d.horizon in ("long", "value")), None)

    actions_summary = []
    if short_d:
        actions_summary.append(f"短线【{short_d.action}】")
    if medium_d:
        actions_summary.append(f"中线【{medium_d.action}】")
    if long_d:
        actions_summary.append(f"长线【{long_d.action}】")

    buy_zone_text = ""
    if vr.available and vr.buy_low and vr.buy_high:
        buy_zone_text = (
            f"；分批买入观察区间为【{vr.buy_low:.2f}–{vr.buy_high:.2f} {package.currency}】"
        )

    target_cap = ""
    if package.decisions and package.decisions[0].target_position:
        target_cap = f"（目标上限 {package.decisions[0].target_position * 100:.0f}%）"

    point2 = f"**多周期建议与操作倾向**：{'，'.join(actions_summary)}{target_cap}{buy_zone_text}。"

    # 3. Defensive Level & Invalidation Conditions
    inval_list: list[str] = []
    for d in package.decisions:
        inval_list.extend(d.invalidation_conditions)
    inval_unique = list(dict.fromkeys(inval_list))
    inval_text = "；".join(inval_unique[:2]) if inval_unique else "盈利能力持续恶化或突破硬约束"

    sup_floor = ""
    if nearest_sup:
        sup_floor = (
            f"关键防守参考【{nearest_sup.low:.2f}–{nearest_sup.high:.2f} {package.currency}】；"
        )
    point3 = f"**关键防守与失效条件**：{sup_floor}当触发【{inval_text}】时模型逻辑失效。"

    return [
        "> [!summary] 执行决策摘要（30秒速读）",
        f"> - {point1}",
        f"> - {point2}",
        f"> - {point3}",
        "",
    ]


def render_analysis_markdown(package: AnalysisPackage) -> str:
    """Render a comprehensive multi-horizon analysis markdown report."""
    from stock_analysis.decision import HORIZON_LABELS
    from stock_analysis.forecast import ModelStatus

    # Determine tags for Obsidian
    actions_set = {d.action for d in package.decisions}
    tags = ["股票/自动分析", f"数据质量/{package.data_quality.value}"]
    if any("买入" in a for a in actions_set):
        tags.append("决策/分批买入")
    elif any("持有" in a for a in actions_set):
        tags.append("决策/持有观察")
    elif any("减仓" in a or "回避" in a for a in actions_set):
        tags.append("决策/回避减仓")

    short_d = next((d for d in package.decisions if d.horizon == "short"), None)
    medium_d = next((d for d in package.decisions if d.horizon == "medium"), None)
    long_d = next((d for d in package.decisions if d.horizon == "long"), None)
    value_d = next((d for d in package.decisions if d.horizon == "value"), None)

    val_status = "neutral"
    vr = package.valuation_range
    if vr.available and vr.buy_high and package.current_price <= vr.buy_high:
        val_status = "discount"
    elif vr.available and vr.fair_high and package.current_price > vr.fair_high:
        val_status = "premium"

    lines = [
        "---",
        "type: automated-stock-analysis",
        f"symbol: {package.symbol}",
        f"name: {package.name}",
        f"date: {package.as_of.isoformat()}",
        f"data_quality: {package.data_quality.value}",
        f"current_price: {package.current_price:.3f}",
        f"currency: {package.currency}",
        f"valuation_status: {val_status}",
    ]
    if short_d:
        lines.append(f"short_action: {short_d.action}")
    if medium_d:
        lines.append(f"medium_action: {medium_d.action}")
    if long_d:
        lines.append(f"long_action: {long_d.action}")
    if value_d:
        lines.append(f"value_action: {value_d.action}")
    lines.append("tags:")
    for tag in tags:
        lines.append(f"  - {tag}")
    vr = package.valuation_range
    if vr.available and vr.buy_low and vr.buy_high:
        lines.append(f"buy_range_low: {vr.buy_low:.2f}")
        lines.append(f"buy_range_high: {vr.buy_high:.2f}")
    lines.extend(
        [
            "status: generated",
            "---",
            "",
            f"# {package.name}（{package.symbol}）多周期分析",
            "",
        ]
    )

    # Executive Summary Card
    lines.extend(render_executive_summary_card(package))

    lines.extend(
        [
            "> [!warning] 决策边界",
            "> 这是概率化研究与纪律检查，不是收益保证或自动交易指令。"
            "C 级数据、低置信度或模型失配时，系统会留白。",
            "",
            "## 结论：多周期冲突卡",
            "",
            "| 周期 | 综合分 | 置信度 | 行动 | 目标仓位 |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for item in package.decisions:
        lines.append(
            f"| {HORIZON_LABELS[item.horizon]} | {item.score:+.2f} | {item.confidence:.0%} | "
            f"{item.action} | {_fmt_percent(item.target_position, 0)} |"
        )
    lines.extend(
        [
            "",
            "## 数据与价格",
            "",
            f"- 分析截止：{package.as_of.isoformat()}",
            f"- 最新价格：{package.current_price:.3f} {package.currency}",
            f"- 数据质量：**{package.data_quality.value}**",
        ]
    )
    for warning in package.data_warnings:
        lines.append(f"- 数据警告：{warning}")
    if package.chart_paths:
        lines.extend(["", "## 技术图表", ""])
        for chart_path in package.chart_paths:
            chart_label = "概率路径" if "概率" in chart_path else "K线与技术指标"
            lines.append(f"![{package.name} {chart_label}](<{chart_path}>)")
    lines.extend(["", "## 支撑与压力", ""])
    if package.price_zones:
        lines.extend(
            [
                "| 类型 | 区间 | 中心 | 距现价 | 强度分 | 触达 | 最近触达 |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for zone in package.price_zones:
            label = "支撑" if zone.kind == "support" else "压力"
            lines.append(
                f"| {label} | {zone.low:.2f}–{zone.high:.2f} | {zone.center:.2f} | "
                f"{zone.distance:+.1%} | {zone.strength * 100:.0f}/100 | {zone.touches} | "
                f"{zone.last_touch.isoformat()} |"
            )
        lines.extend(
            [
                "",
                "> 支撑/压力来自近端局部高低点聚类，并以 ATR、成交量、触达次数和"
                "时间衰减确定区间与强度；强度分不是守住概率，它们仍可能被跳空、重大事件"
                "或趋势加速直接击穿。",
            ]
        )
        if package.price_zone_validation:
            lines.extend(
                [
                    "",
                    "### 样本外检验",
                    "",
                    "| 类型 | 可检验窗口 | 实际触达 | 守住 | 守住率 | 95% 区间 | 状态 |",
                    "|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            for validation in package.price_zone_validation:
                label = "支撑" if validation.kind == "support" else "压力"
                rate = f"{validation.hold_rate:.0%}" if validation.hold_rate is not None else "—"
                interval = (
                    f"{validation.confidence_low:.0%}–{validation.confidence_high:.0%}"
                    if validation.confidence_low is not None
                    and validation.confidence_high is not None
                    else "—"
                )
                status = "可参考" if validation.status == "validated" else "样本不足"
                lines.append(
                    f"| {label} | {validation.windows} | {validation.touched} | "
                    f"{validation.held} | {rate} | {interval} | {status} |"
                )
            lines.extend(
                [
                    "",
                    "> 检验严格使用每个历史截止点之前的数据重新识别区间；只有未来真正触达"
                    "该区间的样本才进入守住率，少于 20 次触达不把百分比当作可靠概率。",
                ]
            )
    else:
        lines.append("- 历史枢轴不足，暂不生成支撑/压力区间。")
    lines.extend(
        [
            "",
            "## 概率预测",
            "",
            "| 期限 | 状态 | 基线/Chronos 权重 | Q10 | 中位数 | Q90 | 上涨概率 | 潜在回撤 |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    degraded_horizons: list[int] = []
    for bundle in package.forecasts:
        weight_text = "/".join(f"{name} {weight:.1%}" for name, weight in bundle.weights.items())
        estimate = bundle.ensemble
        status_text = (
            f"{bundle.status.value} ({bundle.calibration_samples}/{bundle.calibration_target})"
        )
        lines.append(
            f"| {bundle.horizon_days}日 | {status_text} | {weight_text} | "
            f"{estimate.q10:+.1%} | {estimate.q50:+.1%} | {estimate.q90:+.1%} | "
            f"{estimate.up_probability:.0%} | {estimate.potential_drawdown:.1%} |"
        )
        if bundle.status in {ModelStatus.DEGRADED, ModelStatus.DISABLED}:
            degraded_horizons.append(bundle.horizon_days)
    if not package.forecasts:
        lines.append("| — | 长期/价值模式不使用价格路径预测 | — | — | — | — | — | — |")
    if degraded_horizons:
        lines.extend(
            [
                "",
                "> [!warning] "
                + "、".join(f"{days}日" for days in degraded_horizons)
                + "预测当前仅使用随机游走基线；Chronos 未参与，结果仅作保守参考。",
            ]
        )
    lines.extend(["", "## 企业质量与估值", ""])
    lines.append("| 类别 | 指标 | 数值 | 评分 | 说明 |")
    lines.append("|---|---|---:|---:|---|")
    for category, items in (
        ("质量", package.quality),
        ("估值", package.valuation),
        ("市场", package.technical),
        ("宏观", package.macro),
    ):
        for item in items:
            value = _fmt_number(item.value) if item.available else "—"
            lines.append(
                f"| {category} | {item.name} | {value} | {item.score:+.2f} | {item.explanation} |"
            )
    lines.extend(["", "### 合理价值与安全边际", ""])
    if vr.available:
        lines.extend(
            [
                f"- 合理价值区间：{vr.fair_low:.2f}–{vr.fair_high:.2f} {vr.currency}",
                f"- 分批买入观察区间：{vr.buy_low:.2f}–{vr.buy_high:.2f} {vr.currency}",
                f"- 方法：{vr.method}",
            ]
        )
    else:
        lines.append(f"- 估值区间：未提供（{vr.method}）")
    lines.extend(["", "## LLM 证据与事件", ""])
    if package.research.status == "degraded":
        lines.append(
            f"已检索 {len(package.research.evidence)} 条证据；本次显式跳过 LLM，未生成事件因子。"
        )
    else:
        lines.append(
            f"已检索 {len(package.research.evidence)} 条证据；"
            f"生成 {len(package.research.events)} 个有效事件因子。"
        )
    if package.research.events:
        lines.extend(
            [
                "",
                "| 事件类型 | 方向 | 强度 | 置信度 | 生效区间 | 证据 ID | 理由 |",
                "|---|:---:|---:|---:|---|---|---|",
            ]
        )
        for event in package.research.events:
            direction_symbol = "+" if event.direction > 0 else "-" if event.direction < 0 else "0"
            evt_type = (
                event.event_type.value
                if hasattr(event.event_type, "value")
                else str(event.event_type)
            )
            lines.append(
                f"| {evt_type} | {direction_symbol} | {event.strength:.2f} | "
                f"{event.confidence:.0%} | {event.effective_from.isoformat()} 至 "
                f"{event.expires_at.isoformat()} | `{event.evidence_id}` | {event.rationale} |"
            )
    else:
        lines.extend(["", "- 没有通过证据和日期校验的 LLM 事件因子。"])
    if package.research.evidence:
        lines.extend(["", "证据索引："])
        for item in package.research.evidence:
            source_text = f"[{item.title}]({item.source_url})" if item.source_url else item.title
            lines.append(f"- [{item.id}] {source_text}，发布于 {item.published_at.isoformat()}")
    if package.staging_plan and package.staging_plan.available:
        plan = package.staging_plan
        lines.extend(
            [
                "",
                "## 🎯 实盘阶梯挂单执行网格（Staging Execution Grid）",
                "",
                f"> **基准资金规模**：100,000 {package.currency} 测算"
                f"（目标组合占比 **{plan.total_target_weight:.0%}**，"
                f"预计总动用资金 **{plan.total_capital:,.2f} {package.currency}**；"
                "可使用终端 `stock size` 动态调整）。",
                "",
                "| 批次 | 目标价 | 挂单配比 | 建议股数(手数) | 预计占用资金 | 执行逻辑与条件 |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for tier in plan.tiers:
            lots = tier.shares // 100
            lots_text = f"{tier.shares} 股 ({lots}手)" if lots > 0 else f"{tier.shares} 股"
            lines.append(
                f"| {tier.tier_name} | {tier.target_price:.2f} {package.currency} | "
                f"{tier.weight_pct:.0%} | {lots_text} | "
                f"{tier.allocated_amount:,.2f} {package.currency} | {tier.rationale} |"
            )
        if plan.invalidation_price:
            lines.extend(
                [
                    "",
                    f"> [!danger] 硬止损与逻辑失效线："
                    f"**< {plan.invalidation_price:.2f} {package.currency}**"
                    f"（{plan.invalidation_note}）。",
                ]
            )

    lines.extend(["", "## 行动计划与反证", ""])
    for item in package.decisions:
        lines.extend(
            [
                f"### {HORIZON_LABELS[item.horizon]}",
                "",
                f"- 行动：**{item.action}**",
                f"- 依据：{item.rationale}",
            ]
        )
        if item.target_position is not None:
            lines.append(f"- 目标仓位上限：{_fmt_percent(item.target_position, 0)}")
        if item.staging:
            lines.append(f"- 执行方式：{item.staging}")
        if item.invalidation_conditions:
            lines.append(f"- 反证条件：{'；'.join(item.invalidation_conditions)}")
        for warning in item.warnings:
            lines.append(f"- 风险警告：{warning}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_portfolio_markdown(
    snapshot: PortfolioSnapshot, config: AppConfig, database: Database
) -> str:
    """Render the portfolio snapshot compliance check markdown."""
    from collections import defaultdict

    risk = config.section("risk")
    sector_values: dict[str, float] = defaultdict(float)
    for item in snapshot.positions:
        sector_values[item.sector] += item.market_value
    total = snapshot.total_cny_assets or sum(item.market_value for item in snapshot.positions)
    cash_weight = snapshot.cash_cny / total if snapshot.cash_cny is not None and total else None
    lines = [
        "---",
        "type: automated-portfolio-analysis",
        f"date: {date.today().isoformat()}",
        f"snapshot_date: {snapshot.as_of.isoformat()}",
        "status: generated",
        "---",
        "",
        f"# {snapshot.as_of.isoformat()} 持仓自动检查",
        "",
        f"来源：[[{snapshot.path.relative_to(config.home).with_suffix('')}]]",
        "",
        "## 硬约束检查",
        "",
        "| 标的 | 角色 | 市值 | 组合占比 | 上限 | 结论 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in snapshot.positions:
        weight = item.market_value / total if total else 0.0
        limit = float(
            risk.get("core_position_limit", 0.35)
            if item.role == "core"
            else risk.get("satellite_position_limit", 0.15)
        )
        conclusion = "超限，停止加仓并复核" if weight > limit else "合规"
        lines.append(
            f"| {item.name} | {item.role} | {item.market_value:,.2f} CNY | "
            f"{weight:.1%} | {limit:.0%} | {conclusion} |"
        )
    lines.extend(["", "### 行业与现金", ""])
    sector_limit = float(risk.get("sector_limit", 0.60))
    for sector, market_value in sorted(
        sector_values.items(), key=lambda item: item[1], reverse=True
    ):
        weight = market_value / total if total else 0.0
        conclusion = "超限" if weight > sector_limit else "合规"
        lines.append(f"- {sector}：{weight:.1%}（上限 {sector_limit:.0%}，{conclusion}）")
    if cash_weight is not None:
        cash_floor = float(risk.get("cash_floor", 0.05))
        conclusion = "低于底线" if cash_weight < cash_floor else "合规"
        lines.append(f"- 现金类：{cash_weight:.1%}（底线 {cash_floor:.0%}，{conclusion}）")
    lines.extend(["", "## 最近预测回执", ""])
    for item in snapshot.positions:
        if not item.symbol:
            continue
        rows = database.receipts(symbol=item.symbol)[-5:]
        if not rows:
            lines.append(f"- {item.name}：没有预测回执")
            continue
        states: list[str] = []
        for row in rows:
            import json

            decision = json.loads(row["decision_json"])
            states.append(
                f"{row['horizon_days']}日 {decision.get('action', '—')} ({row['status']})"
            )
        lines.append(f"- {item.name}：" + "；".join(states))
    lines.extend(["", "## 数据边界", ""])
    lines.extend(f"- {warning}" for warning in snapshot.warnings)
    lines.extend(
        [
            "- 组合检查执行仓位纪律，不因模型短期看多而放宽单股、行业或现金约束。",
            "- 本报告不包含自动下单。",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_summary_markdown(config: AppConfig, summary: AutomationSummary) -> str:
    """Render the high-level automated summary report."""
    from stock_analysis.data import safe_filename_component

    portfolio_name = summary.portfolio_report.name if summary.portfolio_report else "未生成"
    portfolio_link = f"组合/{portfolio_name}" if summary.portfolio_report else ""
    log_name = f"{summary.as_of.isoformat()}-{summary.run_id}.md"
    lines = [
        "---",
        "type: automated-summary",
        f"date: {summary.as_of.isoformat()}",
        "status: generated",
        "---",
        "",
        f"# 自动分析摘要（{summary.as_of.isoformat()}）",
        "",
        "> 根目录只保留本摘要；详细证据、模型输出和回测结果已按类别归档。",
        "",
        "## 重要结论",
        "",
        "| 标的 | 数据 | 宏观 | 短线 | 中线 | 长线 | 价值 | 详细报告 |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for symbol in summary.symbols:
        safe = symbol.replace(":", "-").replace("/", "-")
        name = safe_filename_component(str(config.asset(symbol).get("name", symbol)))
        actions = summary.highlights.get(symbol, {})
        metrics = summary.decision_scores.get(symbol, {})
        short = _summary_decision(actions, metrics.get("short"), "short")
        medium = _summary_decision(actions, metrics.get("medium"), "medium")
        long = _summary_decision(actions, metrics.get("long"), "long")
        value = _summary_decision(actions, metrics.get("value"), "value")
        report = f"个股/{safe}/{summary.as_of.isoformat()}-{safe}-{name}-all.md"
        lines.append(
            f"| {name}（{symbol}） | {summary.data_quality.get(symbol, '—')} | "
            f"{summary.macro_scores.get(symbol, 0):+.2f} | {short} | {medium} | {long} | {value} | "
            f"[{safe}](<{report}>) |"
        )
    lines.extend(["", f"- 成功：{len(summary.succeeded)} 个；失败：{len(summary.failed)} 个"])
    if summary.failed:
        lines.append(
            "- 失败标的：" + "；".join(f"{key}（{value}）" for key, value in summary.failed.items())
        )
    failed_tasks = [item for item in summary.tasks if item["status"] == "failed"]
    lines.extend(
        [
            f"- 到期回执核对：{summary.evaluated_receipts} 条",
            f"- 自动校准：{sum(len(items) for items in summary.calibrated.values())} 个期限",
            f"- 组合检查：[{portfolio_name}]({portfolio_link})",
            "",
            "## 运行状态",
            "",
            f"- 任务总数：{len(summary.tasks)}；执行 "
            f"{sum(item['status'] == 'executed' for item in summary.tasks)}；"
            f"跳过 {sum(item['status'] == 'skipped' for item in summary.tasks)}；"
            f"失败 {sum(item['status'] == 'failed' for item in summary.tasks)}",
            f"- 完整审计日志：[{summary.run_id}](<运行日志/{log_name}>)",
            "",
            "### 需要关注的失败任务",
            "",
            "",
            "## 使用方式",
            "",
            "- 先读本页的重要结论，再按需打开个股或回测详细报告；任务明细见运行日志。",
            "- 任何动作仍需检查数据质量、回撤预算和失效条件；本系统不会自动下单。",
            "",
        ]
    )
    if failed_tasks:
        insert_at = lines.index("### 需要关注的失败任务") + 1
        lines[insert_at:insert_at] = [
            "",
            "| 标的 | 任务 | 原因 |",
            "|---|---|---|",
            *[f"| {item['symbol']} | {item['task']} | {item['reason']} |" for item in failed_tasks],
        ]
    else:
        insert_at = lines.index("### 需要关注的失败任务") + 1
        lines.insert(insert_at, "- 无失败任务。")
    if summary.calibration_progress:
        marker = "## 运行状态"
        progress_rows = []
        for symbol, horizons in summary.calibration_progress.items():
            active = [days for days, info in horizons.items() if info.get("status") == "active"]
            experimental = [
                days for days, info in horizons.items() if info.get("status") == "experimental"
            ]
            progress_rows.append(
                f"| {symbol} | {', '.join(f'{days}日' for days in active) or '—'} | "
                f"{', '.join(f'{days}日' for days in experimental) or '—'} |"
            )
        progress_lines = [
            "## 校准进度",
            "",
            "| 标的 | 已激活期限 | 实验/待校准期限 |",
            "|---|---|---|",
            *progress_rows,
            "",
        ]
        lines[lines.index(marker) : lines.index(marker)] = progress_lines
    return "\n".join(lines).strip() + "\n"


def _summary_decision(
    actions: dict[str, str], metrics: dict[str, float] | None, horizon: str
) -> str:
    action = actions.get(horizon, "—")
    if not metrics:
        return action
    return f"{action}<br>评分 {metrics['score']:+.2f} / 置信 {metrics['confidence']:.0%}"


def render_task_log(summary: AutomationSummary) -> str:
    """Render verbose task details outside the human-facing summary page."""
    lines = [
        "---",
        "type: automated-run-log",
        f"date: {summary.as_of.isoformat()}",
        f"run_id: {summary.run_id}",
        "---",
        "",
        f"# 自动运行日志（{summary.as_of.isoformat()} / {summary.run_id}）",
        "",
        "| 标的 | 任务 | 状态 | 原因 |",
        "|---|---|---|---|",
        *[
            f"| {item['symbol']} | {item['task']} | {item['status']} | {item['reason']} |"
            for item in summary.tasks
        ],
        "",
        f"- 成功标的：{len(summary.succeeded)}；失败标的：{len(summary.failed)}",
    ]
    if summary.calibration_progress:
        lines.extend(
            [
                "",
                "## 逐期限校准进度",
                "",
                "| 标的 | 期限 | 样本 | 状态 |",
                "|---|---:|---:|---|",
                *[
                    f"| {symbol} | {days}日 | {info.get('samples', 0)}/{info.get('target', 0)} | "
                    f"{info.get('status', 'unknown')} |"
                    for symbol, horizons in summary.calibration_progress.items()
                    for days, info in horizons.items()
                ],
            ]
        )
    return "\n".join(lines).strip() + "\n"
