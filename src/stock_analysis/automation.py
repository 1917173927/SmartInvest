from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from importlib.util import find_spec
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stock_analysis.charts import render_probability_chart, render_stock_chart
from stock_analysis.context import refresh_context
from stock_analysis.data import (
    AppConfig,
    Database,
    Instrument,
    backfill_symbol,
    coverage_warnings,
    quality_summary,
    safe_filename_component,
    sync_actions,
    sync_symbol,
    total_return_frame,
    utc_now,
)
from stock_analysis.decision import (
    analyze_package,
    create_receipts,
    latest_portfolio_snapshot,
    position_weight,
    render_analysis_markdown,
    render_portfolio_markdown,
)
from stock_analysis.forecast import (
    MEDIUM_HORIZONS,
    SHORT_HORIZONS,
    Chronos2Forecaster,
    calibration_weights,
    evaluate_open_receipts,
    forecast_one,
    walk_forward_backtest,
)
from stock_analysis.research import active_event_rows, run_research

LOGGER = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class AutomationSummary:
    as_of: date
    symbols: list[str]
    run_id: str = field(default_factory=lambda: uuid4().hex)
    succeeded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    highlights: dict[str, dict[str, str]] = field(default_factory=dict)
    decision_scores: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    data_quality: dict[str, str] = field(default_factory=dict)
    macro_scores: dict[str, float] = field(default_factory=dict)
    evaluated_receipts: int = 0
    calibrated: dict[str, list[int]] = field(default_factory=dict)
    calibration_progress: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)
    tasks: list[dict[str, str]] = field(default_factory=list)
    portfolio_report: Path | None = None


def _task(
    summary: AutomationSummary,
    *,
    name: str,
    status: str,
    reason: str,
    symbol: str = "system",
) -> None:
    summary.tasks.append({"task": name, "status": status, "reason": reason, "symbol": symbol})


def _market_sync_needed(database: Database, symbol: str, as_of: date) -> tuple[bool, str]:
    frame = database.load_bars(symbol, as_of)
    if frame.empty:
        return True, "没有行情缓存"
    latest = pd.Timestamp(frame.iloc[-1]["trade_date"]).date()
    missing_business_days = len(pd.bdate_range(latest + timedelta(days=1), as_of))
    if missing_business_days > 1:
        return True, f"行情落后约 {missing_business_days} 个工作日"
    issues = coverage_warnings(frame, as_of=as_of)
    gap_issue = next((item for item in issues if "日期缺口" in item), None)
    if gap_issue:
        return True, gap_issue
    if "C" in set(frame.get("quality", pd.Series(["C"])).astype(str)):
        return True, "缓存包含 C 级数据"
    return False, f"最新行情 {latest}"


def _calibration_in_cooldown(
    database: Database, symbol: str, horizon_days: int, cooldown_days: int
) -> bool:
    row = database.connection.execute(
        """
        SELECT MAX(evaluated_at) AS evaluated_at FROM forecast_receipts
        WHERE symbol = ? AND horizon_days = ? AND status = 'evaluated' AND id LIKE 'bt-%'
        """,
        (symbol, horizon_days),
    ).fetchone()
    if not row or not row["evaluated_at"]:
        return False
    evaluated = datetime.fromisoformat(str(row["evaluated_at"]))
    if evaluated.tzinfo is None:
        evaluated = evaluated.replace(tzinfo=UTC)
    return datetime.now(tz=UTC) - evaluated < timedelta(days=cooldown_days)


def configured_symbols(config: AppConfig) -> list[str]:
    """Return configured instruments in stable TOML order, excluding malformed keys."""
    symbols: list[str] = []
    for raw_symbol in config.section("assets"):
        try:
            symbols.append(Instrument.parse(str(raw_symbol)).canonical)
        except ValueError:
            LOGGER.warning("跳过无法识别的配置标的: %s", raw_symbol)
    return symbols


def configured_evidence_paths(config: AppConfig, symbol: str) -> list[Path]:
    """Find dated evidence under per-symbol directories for unattended research."""
    roots = config.section("automation").get("evidence_dirs", [])
    if isinstance(roots, str):
        roots = [roots]
    safe_symbol = symbol.replace(":", "-").replace("/", "-")
    paths: list[Path] = []
    for raw_root in roots:
        root = config.home / str(raw_root) / safe_symbol
        if root.is_dir():
            paths.extend(
                path
                for path in sorted(root.iterdir())
                if path.is_file() and path.suffix.lower() in {".md", ".txt"}
            )
    return paths


def _write_report(
    config: AppConfig, filename: str, content: str, relative_dir: Path | None = None
) -> Path:
    path = (
        config.reports_dir / relative_dir / filename
        if relative_dir
        else config.reports_dir / filename
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def organize_reports(config: AppConfig) -> list[Path]:
    """Move legacy flat reports into stable internal folders without overwriting files."""
    root = config.reports_dir
    moved: list[Path] = []
    for source in sorted(root.glob("*.md")):
        if source.name in {"README.md", "最新摘要.md"}:
            continue
        text = source.read_text(encoding="utf-8")
        report_type = re.search(r"^type:\s*(\S+)", text, flags=re.MULTILINE)
        kind = report_type.group(1) if report_type else "archive"
        if kind == "automated-stock-analysis":
            match = re.search(r"^symbol:\s*(\S+)", text, flags=re.MULTILINE)
            symbol = match.group(1) if match else "未分类"
            relative = Path("个股") / symbol.replace(":", "-").replace("/", "-")
        elif kind == "automated-portfolio-analysis":
            relative = Path("组合")
        elif kind == "forecast-evaluation":
            match = re.search(r"^symbol:\s*(\S+)", text, flags=re.MULTILINE)
            symbol = match.group(1) if match else "未分类"
            relative = Path("回测") / symbol.replace(":", "-").replace("/", "-")
        else:
            relative = Path("归档")
        target = root / relative / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            LOGGER.warning("目标文件已存在，保留旧文件不移动: %s", target)
            continue
        source.replace(target)
        moved.append(target)

    # Add the configured company name to older per-stock report filenames.  The
    # report content already carries the name; this migration keeps the Obsidian
    # file list readable while refusing to overwrite a newer named report.
    stocks_root = root / "个股"
    if stocks_root.is_dir():
        for stock_dir in sorted(path for path in stocks_root.iterdir() if path.is_dir()):
            symbol = stock_dir.name.replace("-", ":", 1)
            name = safe_filename_component(str(config.asset(symbol).get("name", "")))
            if name == "未命名":
                continue
            safe_symbol = stock_dir.name
            for source in sorted(stock_dir.glob("*.md")):
                match = re.match(
                    rf"^(\d{{4}}-\d{{2}}-\d{{2}})-{re.escape(safe_symbol)}-"
                    r"(all|short|medium|long|value)\.md$",
                    source.name,
                )
                if not match:
                    continue
                target = stock_dir / f"{match.group(1)}-{safe_symbol}-{name}-{match.group(2)}.md"
                if target.exists():
                    LOGGER.warning("目标文件已存在，保留旧文件不重命名: %s", target)
                    continue
                source.replace(target)
                moved.append(target)
    return moved


def render_summary_markdown(config: AppConfig, summary: AutomationSummary) -> str:
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
            *[
                f"| {item['symbol']} | {item['task']} | {item['reason']} |"
                for item in failed_tasks
            ],
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
        lines[lines.index(marker):lines.index(marker)] = progress_lines
    return "\n".join(lines)


def _summary_decision(
    actions: dict[str, str], metrics: dict[str, float] | None, horizon: str
) -> str:
    """Keep the root summary actionable without hiding score uncertainty."""
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
    return "\n".join(lines)


def run_automation(
    config: AppConfig,
    *,
    as_of: date | None = None,
    symbols: list[str] | None = None,
    use_llm: bool | None = None,
    use_chronos: bool | None = None,
) -> AutomationSummary:
    """Run the unattended sync → evaluate → analyze → portfolio workflow."""
    automation = config.section("automation")
    analysis_date = as_of or date.today()
    selected = symbols or configured_symbols(config)
    summary = AutomationSummary(as_of=analysis_date, symbols=selected)
    llm_enabled = (
        _env_bool("STOCK_AUTO_USE_LLM", bool(automation.get("use_llm", True)))
        if use_llm is None
        else use_llm
    )
    chronos_enabled = (
        _env_bool("STOCK_AUTO_USE_CHRONOS", bool(automation.get("use_chronos", True)))
        if use_chronos is None
        else use_chronos
    )
    lookback_years = int(automation.get("sync_lookback_years", 5))
    max_runtime_minutes = int(automation.get("max_runtime_minutes", 20))
    deadline = time.monotonic() + max_runtime_minutes * 60
    forecast_config = config.section("forecast")
    chronos_forecaster = (
        Chronos2Forecaster(
            model_id=str(forecast_config.get("model_id", "amazon/chronos-2")),
            device=str(forecast_config.get("device", "cpu")),
            context_length=int(forecast_config.get("context_length", 512)),
        )
        if chronos_enabled
        else None
    )

    with Database(config.db_path) as database:
        recovered = database.recover_stale_automation_runs(
            utc_now() - timedelta(minutes=max_runtime_minutes)
        )
        database.start_automation_run(summary.run_id, analysis_date, selected)
        if recovered:
            _task(
                summary,
                name="stale-run-recovery",
                status="executed",
                reason=f"将 {recovered} 个超时未结束任务标记为 interrupted",
            )
        organize_reports(config)
        dependencies = {
            "akshare": find_spec("akshare") is not None,
            "yfinance": find_spec("yfinance") is not None,
            "chronos": (not chronos_enabled) or Chronos2Forecaster.installed(),
        }
        missing = [name for name, available in dependencies.items() if not available]
        _task(
            summary,
            name="preflight",
            status="failed" if missing else "executed",
            reason="缺少依赖: " + ", ".join(missing) if missing else "核心依赖与数据库可用",
        )
        # Sync all symbols before evaluating receipts. This lets today's
        # calibration affect today's reports instead of waiting for the next run.
        sync_warnings_by_symbol: dict[str, list[str]] = {}
        sync_ready: list[str] = []
        for symbol in selected:
            try:
                sync_warnings: list[str] = []
                if analysis_date >= date.today() - timedelta(days=2):
                    needed, reason = _market_sync_needed(database, symbol, analysis_date)
                    if needed:
                        result = sync_symbol(
                            database,
                            symbol,
                            start=analysis_date - timedelta(days=365 * lookback_years + 2),
                            end=analysis_date,
                        )
                        sync_warnings.extend(result.warnings)
                        _task(
                            summary,
                            name="market-sync",
                            status="executed",
                            reason=f"{reason}；provider={result.provider} bars={result.bars}",
                            symbol=symbol,
                        )
                    else:
                        _task(
                            summary,
                            name="market-sync",
                            status="skipped",
                            reason=reason,
                            symbol=symbol,
                        )
                else:
                    _task(
                        summary,
                        name="market-sync",
                        status="skipped",
                        reason="历史截止日分析使用本地时间点缓存",
                        symbol=symbol,
                    )
                sync_warnings_by_symbol[symbol] = sync_warnings
                sync_ready.append(symbol)
            except Exception as exc:  # one bad provider/symbol must not stop the batch
                summary.failed[symbol] = str(exc)
                _task(
                    summary,
                    name="market-sync",
                    status="failed",
                    reason=str(exc),
                    symbol=symbol,
                )
                LOGGER.exception("自动同步失败: %s", symbol)

        summary.evaluated_receipts = len(evaluate_open_receipts(database))
        _task(
            summary,
            name="receipt-evaluation",
            status="executed",
            reason=f"核对 {summary.evaluated_receipts} 条到期回执",
        )

        # Repair historical ex-rights gaps independently from market-data sync.
        # This path is cheap compared with re-downloading bars and is safe to
        # repeat because corporate_actions uses an idempotent upsert key.
        for symbol in sync_ready:
            bars = database.load_bars(symbol, analysis_date)
            if bars.empty:
                continue
            actions = database.load_actions(symbol, analysis_date)
            frame, return_warnings = total_return_frame(bars, actions)
            if not any("超过 35%" in warning for warning in return_warnings):
                _task(
                    summary,
                    name="action-repair",
                    status="skipped",
                    reason="没有未解释的超大单日变动",
                    symbol=symbol,
                )
                continue
            start = pd.Timestamp(bars.iloc[0]["trade_date"]).date()
            action_count, action_errors = sync_actions(
                database, symbol, start=start, end=analysis_date
            )
            if action_count:
                _task(
                    summary,
                    name="action-repair",
                    status="executed",
                    reason=f"补入 {action_count} 条历史公司行动并重新计算收益",
                    symbol=symbol,
                )
            else:
                _task(
                    summary,
                    name="action-repair",
                    status="failed",
                    reason="; ".join(action_errors) or "备用公司行动源未返回数据",
                    symbol=symbol,
                )

        # Optional unattended bootstrap: only run a walk-forward calibration for
        # a symbol/horizon whose evaluated sample count is still below target.
        # Deterministic backtest receipt IDs make retries idempotent.
        if bool(automation.get("auto_calibrate", True)) and chronos_enabled:
            configured_horizons = automation.get("calibration_horizons", [20])
            if isinstance(configured_horizons, int):
                configured_horizons = [configured_horizons]
            target = int(config.section("forecast").get("minimum_calibration_samples", 100))
            max_windows = int(automation.get("calibration_max_windows", target))
            step = int(automation.get("calibration_step", 1))
            cooldown_days = int(automation.get("calibration_cooldown_days", 7))
            calibration_job_limit = max(1, int(automation.get("calibration_jobs_per_run", 1)))
            calibration_jobs_run = 0
            summary.calibration_progress = {}
            for symbol in sync_ready:
                bars = database.load_bars(symbol, analysis_date)
                actions = database.load_actions(symbol, analysis_date)
                frame, return_warnings = total_return_frame(bars, actions)
                if frame.empty:
                    continue
                for raw_days in configured_horizons:
                    days = int(raw_days)
                    state = calibration_weights(database, symbol, days, config)
                    summary.calibration_progress.setdefault(symbol, {})[str(days)] = {
                        "samples": state.samples,
                        "target": target,
                        "status": state.status.value,
                    }
                    if any("超过 35%" in warning for warning in return_warnings):
                        _task(
                            summary,
                            name=f"calibration-{days}d",
                            status="skipped",
                            reason="存在未解释的超大单日变动，先修复公司行动",
                            symbol=symbol,
                        )
                        continue
                    if state.samples >= target:
                        _task(
                            summary,
                            name=f"calibration-{days}d",
                            status="skipped",
                            reason=f"已有 {state.samples}/{target} 个有效样本",
                            symbol=symbol,
                        )
                        continue
                    if _calibration_in_cooldown(database, symbol, days, cooldown_days):
                        _task(
                            summary,
                            name=f"calibration-{days}d",
                            status="skipped",
                            reason=f"样本 {state.samples}/{target}，处于 {cooldown_days} 天冷却期",
                            symbol=symbol,
                        )
                        continue
                    if calibration_jobs_run >= calibration_job_limit:
                        _task(
                            summary,
                            name=f"calibration-{days}d",
                            status="skipped",
                            reason=f"达到本次校准任务限额 {calibration_job_limit}",
                            symbol=symbol,
                        )
                        continue
                    if time.monotonic() >= deadline:
                        _task(
                            summary,
                            name=f"calibration-{days}d",
                            status="skipped",
                            reason=f"达到单次运行上限 {max_runtime_minutes} 分钟",
                            symbol=symbol,
                        )
                        continue
                    try:
                        calibration_jobs_run += 1
                        minimum_history = int(
                            config.section("forecast").get("minimum_history_days", 756)
                        )
                        if len(frame) < minimum_history + days:
                            backfill = backfill_symbol(
                                database,
                                symbol,
                                start=date(1990, 1, 1),
                                end=analysis_date,
                            )
                            bars = database.load_bars(symbol, analysis_date)
                            actions = database.load_actions(symbol, analysis_date)
                            frame, _ = total_return_frame(bars, actions)
                            _task(
                                summary,
                                name="history-backfill",
                                status="executed",
                                reason=f"校准数据不足，补齐 {backfill.bars} 条历史数据",
                                symbol=symbol,
                            )
                        walk_forward_backtest(
                            symbol=symbol,
                            frame=frame,
                            horizon_days=days,
                            database=database,
                            config=config,
                            use_chronos=True,
                            max_windows=max_windows,
                            step=step,
                            chronos_forecaster=chronos_forecaster,
                        )
                        updated = calibration_weights(database, symbol, days, config)
                        summary.calibration_progress[symbol][str(days)] = {
                            "samples": updated.samples,
                            "target": target,
                            "status": updated.status.value,
                        }
                        summary.calibrated.setdefault(symbol, []).append(days)
                        _task(
                            summary,
                            name=f"calibration-{days}d",
                            status="executed",
                            reason=f"由 {state.samples}/{target} 补齐 walk-forward 样本",
                            symbol=symbol,
                        )
                    except Exception as exc:
                        _task(
                            summary,
                            name=f"calibration-{days}d",
                            status="failed",
                            reason=str(exc),
                            symbol=symbol,
                        )
                        LOGGER.warning("自动校准失败 %s %sd: %s", symbol, days, exc)
        elif bool(automation.get("auto_calibrate", True)):
            _task(
                summary,
                name="calibration",
                status="skipped",
                reason="Chronos 未启用，保留随机游走基线",
            )

        for symbol in sync_ready:
            try:
                sync_warnings = sync_warnings_by_symbol[symbol]
                context = refresh_context(
                    database,
                    config,
                    symbol=symbol,
                    as_of=analysis_date,
                    start=analysis_date - timedelta(days=365 * lookback_years + 2),
                )
                _task(
                    summary,
                    name="context-refresh",
                    status=(
                        "failed"
                        if context.warnings and not context.refreshed
                        else "executed"
                        if context.refreshed
                        else "skipped"
                    ),
                    reason=(
                        f"刷新 {', '.join(context.refreshed) or '无'}；"
                        f"跳过 {', '.join(context.skipped) or '无'}；"
                        f"新闻 {context.news_count} 条，宏观 {context.macro_count} 条"
                    ),
                    symbol=symbol,
                )
                bars = database.load_bars(symbol, analysis_date)
                actions = database.load_actions(symbol, analysis_date)
                frame, return_warnings = total_return_frame(bars, actions)
                if frame.empty:
                    raise RuntimeError("没有可用行情缓存")
                data_quality, quality_warnings = quality_summary(frame, analysis_date)
                research = run_research(
                    database=database,
                    config=config,
                    symbol=symbol,
                    as_of=analysis_date,
                    evidence_paths=configured_evidence_paths(config, symbol),
                    use_llm=llm_enabled,
                )
                event_rows = active_event_rows(database, symbol, analysis_date)
                forecasts = [
                    forecast_one(
                        symbol=symbol,
                        as_of=analysis_date,
                        horizon_days=days,
                        frame=frame,
                        data_quality=data_quality,
                        database=database,
                        config=config,
                        event_rows=event_rows,
                        use_chronos=chronos_enabled,
                        chronos_forecaster=chronos_forecaster,
                    )
                    for days in SHORT_HORIZONS + MEDIUM_HORIZONS
                ]
                try:
                    snapshot = latest_portfolio_snapshot(config)
                    current_weight = position_weight(snapshot, symbol)
                except Exception:
                    current_weight = None
                package = analyze_package(
                    config=config,
                    database=database,
                    symbol=symbol,
                    as_of=analysis_date,
                    frame=frame,
                    data_quality=data_quality,
                    data_warnings=list(
                        dict.fromkeys(
                            sync_warnings
                            + context.warnings
                            + return_warnings
                            + quality_warnings
                        )
                    ),
                    forecasts=forecasts,
                    research=research,
                    current_weight=current_weight,
                )
                safe_symbol = symbol.replace(":", "-").replace("/", "-")
                chart_config = config.section("charts")
                if bool(chart_config.get("enabled", True)):
                    chart_format = str(chart_config.get("format", "svg")).lower()
                    technical_name = (
                        f"{analysis_date.isoformat()}-{safe_symbol}-"
                        f"{safe_filename_component(package.name)}-技术图.{chart_format}"
                    )
                    try:
                        render_stock_chart(
                            frame,
                            package,
                            config.reports_dir / "个股" / safe_symbol / "charts" / technical_name,
                        )
                        package.chart_paths.append(f"charts/{technical_name}")
                        _task(
                            summary,
                            name="chart-render-technical",
                            status="executed",
                            reason=technical_name,
                            symbol=symbol,
                        )
                    except Exception as exc:
                        package.data_warnings.append(f"图表未生成: {exc}")
                        _task(
                            summary,
                            name="chart-render-technical",
                            status="failed",
                            reason=str(exc),
                            symbol=symbol,
                        )
                    if bool(chart_config.get("include_probability_fan", True)) and forecasts:
                        probability_name = (
                            f"{analysis_date.isoformat()}-{safe_symbol}-"
                            f"{safe_filename_component(package.name)}-概率路径图.{chart_format}"
                        )
                        try:
                            render_probability_chart(
                                package,
                                config.reports_dir
                                / "个股"
                                / safe_symbol
                                / "charts"
                                / probability_name,
                            )
                            package.chart_paths.append(f"charts/{probability_name}")
                            _task(
                                summary,
                                name="chart-render-probability",
                                status="executed",
                                reason=probability_name,
                                symbol=symbol,
                            )
                        except Exception as exc:
                            package.data_warnings.append(f"概率图未生成: {exc}")
                            _task(
                                summary,
                                name="chart-render-probability",
                                status="failed",
                                reason=str(exc),
                                symbol=symbol,
                            )
                create_receipts(database, package)
                _write_report(
                    config,
                    f"{analysis_date.isoformat()}-{safe_symbol}-"
                    f"{safe_filename_component(package.name)}-all.md",
                    render_analysis_markdown(package),
                    Path("个股") / safe_symbol,
                )
                summary.succeeded.append(symbol)
                summary.highlights[symbol] = {
                    item.horizon.value: item.action for item in package.decisions
                }
                summary.decision_scores[symbol] = {
                    item.horizon.value: {
                        "score": float(item.score),
                        "confidence": float(item.confidence),
                    }
                    for item in package.decisions
                }
                summary.data_quality[symbol] = package.data_quality.value
                summary.macro_scores[symbol] = package.macro_score
            except Exception as exc:  # one bad provider/symbol must not stop the batch
                summary.failed[symbol] = str(exc)
                LOGGER.exception("自动分析失败: %s", symbol)

        try:
            snapshot = latest_portfolio_snapshot(config)
            summary.portfolio_report = _write_report(
                config,
                f"{analysis_date.isoformat()}-组合检查.md",
                render_portfolio_markdown(snapshot, config, database),
                Path("组合"),
            )
        except FileNotFoundError:
            LOGGER.info("没有持仓快照，跳过组合检查")
        _write_report(
            config,
            f"{analysis_date.isoformat()}-{summary.run_id}.md",
            render_task_log(summary),
            Path("运行日志"),
        )
        _write_report(config, "最新摘要.md", render_summary_markdown(config, summary))
        database.finish_automation_run(
            summary.run_id,
            status="partial" if summary.failed else "completed",
            tasks=summary.tasks,
            summary=summary_json(summary),
        )
    return summary


def summary_json(summary: AutomationSummary) -> str:
    return json.dumps(
        {
            "run_id": summary.run_id,
            "as_of": summary.as_of.isoformat(),
            "symbols": summary.symbols,
            "succeeded": summary.succeeded,
            "failed": summary.failed,
            "highlights": summary.highlights,
            "decision_scores": summary.decision_scores,
            "data_quality": summary.data_quality,
            "macro_scores": summary.macro_scores,
            "evaluated_receipts": summary.evaluated_receipts,
            "calibrated": summary.calibrated,
            "calibration_progress": summary.calibration_progress,
            "tasks": summary.tasks,
            "portfolio_report": str(summary.portfolio_report) if summary.portfolio_report else None,
        },
        ensure_ascii=False,
        indent=2,
    )
