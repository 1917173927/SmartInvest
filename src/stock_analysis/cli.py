from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from stock_analysis.automation import (
    configured_symbols,
    organize_reports,
    run_automation,
    summary_json,
)
from stock_analysis.charts import available as charts_available
from stock_analysis.charts import render_probability_chart, render_stock_chart
from stock_analysis.data import (
    AkShareProvider,
    AppConfig,
    Database,
    DataQuality,
    FundamentalRecord,
    Instrument,
    YFinanceProvider,
    backfill_symbol,
    coverage_warnings,
    quality_summary,
    safe_filename_component,
    sync_symbol,
    total_return_frame,
)
from stock_analysis.decision import (
    HORIZON_LABELS,
    AnalysisPackage,
    Horizon,
    HorizonDecision,
    analyze_package,
    create_receipts,
    get_valuation_strategy,
    latest_portfolio_snapshot,
    position_weight,
    render_analysis_markdown,
    render_portfolio_markdown,
)
from stock_analysis.forecast import (
    MEDIUM_HORIZONS,
    SHORT_HORIZONS,
    Chronos2Forecaster,
    evaluate_open_receipts,
    forecast_one,
    walk_forward_backtest,
)
from stock_analysis.research import active_event_rows, run_research

app = typer.Typer(
    name="stock",
    no_args_is_help=True,
    help="轻量、可校准、证据约束的个人股票分析系统。",
)
console = Console()


def _context() -> tuple[AppConfig, Database]:
    config = AppConfig.load()
    return config, Database(config.db_path)


def _parse_date(value: str | None, fallback: date) -> date:
    try:
        return date.fromisoformat(value) if value else fallback
    except ValueError as exc:
        raise typer.BadParameter("日期格式应为 YYYY-MM-DD") from exc


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


def _load_analysis_frame(
    database: Database, symbol: str, as_of: date
) -> tuple[pd.DataFrame, list[str]]:
    bars = database.load_bars(symbol, as_of)
    actions = database.load_actions(symbol, as_of)
    frame, warnings = total_return_frame(bars, actions)
    return frame, warnings


@app.command()
def doctor() -> None:
    """检查配置、数据库、数据适配器、LLM 和 Chronos 状态。"""
    config, database = _context()
    chronos_installed = Chronos2Forecaster.installed()
    embedding_configured = bool(
        (os.getenv("STOCK_ANALYSIS_EMBEDDING_MODEL") or os.getenv("GEMINI_EMBEDDING_MODEL"))
        and (
            os.getenv("STOCK_ANALYSIS_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
    )
    table = Table(title="StockAnalysis 运行检查")
    table.add_column("项目")
    table.add_column("状态")
    table.add_column("说明")
    checks = [
        ("Python", "OK", ">=3.12 由 pyproject 约束"),
        (
            "配置",
            "OK" if (config.home / "stock-analysis.toml").exists() else "WARN",
            str(config.home),
        ),
        ("SQLite", "OK", str(database.path)),
        (
            "AKShare",
            "OK" if AkShareProvider.available() else "OPTIONAL",
            "A/H/基金主数据源；uv sync --extra data",
        ),
        (
            "yfinance",
            "OK" if YFinanceProvider.available() else "OPTIONAL",
            "港美股备用源；uv sync --extra data",
        ),
        (
            "图表",
            "OK" if charts_available() else "OPTIONAL",
            "K线/MACD/RSI；uv sync --extra charts",
        ),
        (
            "Chronos-2",
            "OK" if chronos_installed else "OPTIONAL",
            (
                "已安装；首次运行下载后本地缓存"
                if chronos_installed
                else "未安装或不可用时自动使用随机游走基线"
            ),
        ),
        (
            "LLM",
            "OK"
            if os.getenv("OPENAI_API_KEY") and os.getenv("STOCK_ANALYSIS_MODEL")
            else "OPTIONAL",
            "缺失时保留证据检索，不生成事件因子",
        ),
        (
            "嵌入模型",
            "OK" if embedding_configured else "OPTIONAL",
            "未配置时使用 SQLite FTS；可与对话模型使用不同服务",
        ),
        (
            "SEC User-Agent",
            "OK" if os.getenv("SEC_USER_AGENT") else "OPTIONAL",
            "美股 SEC 财务事实需要联系人标识",
        ),
        (
            "HF Hub Token",
            "OK" if os.getenv("HF_TOKEN") else "OPTIONAL",
            "模型已缓存时无需 Token；首次下载/高频请求建议配置",
        ),
    ]
    try:
        database.connection.execute("SELECT count(*) FROM documents_fts").fetchone()
        checks.append(("SQLite FTS5", "OK", "全文检索可用"))
    except sqlite3.OperationalError:
        checks.append(("SQLite FTS5", "WARN", "将降级为 LIKE 检索"))
    news_count = database.connection.execute("SELECT count(*) FROM news_items").fetchone()[0]
    macro_count = database.connection.execute("SELECT count(*) FROM macro_observations").fetchone()[
        0
    ]
    run_count = database.connection.execute("SELECT count(*) FROM automation_runs").fetchone()[0]
    checks.extend(
        [
            ("新闻缓存", "OK" if news_count else "WARN", f"{news_count} 条带来源记录"),
            ("宏观缓存", "OK" if macro_count else "WARN", f"{macro_count} 个观测值"),
            ("自动运行审计", "OK" if run_count else "WARN", f"{run_count} 次运行"),
        ]
    )
    # Per-symbol health makes provider outages and stale caches visible without
    # opening a generated report.  This is intentionally read-only.
    for symbol in configured_symbols(config):
        bars = database.load_bars(symbol, date.today())
        if bars.empty:
            checks.append((symbol, "WARN", "没有行情缓存"))
            continue
        actions = database.load_actions(symbol, date.today())
        _, return_warnings = total_return_frame(bars, actions)
        coverage = coverage_warnings(bars, as_of=date.today())
        latest = pd.Timestamp(bars.iloc[-1]["trade_date"]).date()
        quality, quality_warnings = quality_summary(bars, date.today())
        warnings = list(dict.fromkeys(return_warnings + coverage + quality_warnings))
        checks.append(
            (
                symbol,
                "WARN" if warnings or quality.value == "C" else "OK",
                f"{len(bars)} 条行情；最新 {latest}；{len(actions)} 条公司行动"
                + ("；" + "；".join(warnings[:2]) if warnings else ""),
            )
        )
    for name, status, detail in checks:
        table.add_row(name, status, detail)
    console.print(table)
    database.close()


@app.command("sync")
def sync_command(
    symbol: Annotated[str, typer.Argument(help="CN:601318、HK:00700、US:AAPL 等")],
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD；默认五年前")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD；默认今天")] = None,
) -> None:
    """同步指定标的的原始日线、公司行动和可用财务事实。"""
    config, database = _context()
    end_date = _parse_date(end, date.today())
    start_date = _parse_date(start, end_date - timedelta(days=365 * 5 + 2))
    if start_date >= end_date:
        raise typer.BadParameter("开始日期必须早于结束日期")
    try:
        result = sync_symbol(database, symbol, start=start_date, end=end_date)
    except Exception as exc:
        database.close()
        console.print(f"[red]同步失败：{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]{result.symbol}[/green] provider={result.provider} bars={result.bars} "
        f"actions={result.actions} fundamentals={result.fundamentals} "
        f"latest={result.latest_date} quality={result.quality.value}"
    )
    for warning in result.warnings:
        console.print(f"[yellow]- {warning}[/yellow]")
    database.close()


@app.command("backfill")
def backfill_command(
    years: Annotated[int, typer.Option(min=1, max=30, help="非全历史模式的回溯年数")] = 10,
    all_history: Annotated[
        bool,
        typer.Option("--all-history/--recent", help="尝试获取接口支持的最早历史"),
    ] = True,
    symbol: Annotated[str | None, typer.Option(help="可选；不填则同步配置中的全部标的")] = None,
) -> None:
    """通过配置的数据接口批量补齐历史行情、公司行动和财务事实。"""
    config, database = _context()
    selected = [Instrument.parse(symbol).canonical] if symbol else configured_symbols(config)
    end_date = date.today()
    start_date = date(1900, 1, 1) if all_history else end_date - timedelta(days=365 * years + 2)
    results: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    for canonical in selected:
        try:
            result = backfill_symbol(database, canonical, start=start_date, end=end_date)
            results.append(
                {
                    "symbol": canonical,
                    "name": str(config.asset(canonical).get("name", canonical)),
                    "provider": result.provider,
                    "bars": result.bars,
                    "latest": result.latest_date.isoformat() if result.latest_date else None,
                    "quality": result.quality.value,
                    "warnings": result.warnings,
                }
            )
        except Exception as exc:
            failures[canonical] = str(exc)
    database.close()
    console.print_json(
        json.dumps(
            {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "results": results,
                "failed": failures,
            },
            ensure_ascii=False,
        )
    )
    if failures and not results:
        raise typer.Exit(1)


@app.command("auto")
def auto_command(
    use_llm: Annotated[
        bool | None,
        typer.Option("--llm/--no-llm", help="是否启用证据事件抽取；默认读取配置"),
    ] = None,
    use_chronos: Annotated[
        bool | None,
        typer.Option("--chronos/--no-chronos", help="是否启用 Chronos；默认读取配置"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="输出完整任务 JSON；默认只显示重要摘要"),
    ] = False,
) -> None:
    """无交互运行同步、分析、回执核对和组合检查。"""
    config = AppConfig.load()
    try:
        summary = run_automation(config, use_llm=use_llm, use_chronos=use_chronos)
    except Exception as exc:
        console.print(f"[red]自动运行失败：{exc}[/red]")
        raise typer.Exit(1) from exc
    if verbose:
        console.print_json(summary_json(summary))
    else:
        executed = sum(item["status"] == "executed" for item in summary.tasks)
        skipped = sum(item["status"] == "skipped" for item in summary.tasks)
        failed = sum(item["status"] == "failed" for item in summary.tasks)
        console.print(
            f"[green]自动分析完成[/green]：成功 {len(summary.succeeded)}，"
            f"失败 {len(summary.failed)}；任务执行 {executed} / 跳过 {skipped} / 失败 {failed}"
        )
        console.print(f"摘要：{config.reports_dir / '最新摘要.md'}")
        if summary.portfolio_report:
            console.print(f"组合报告：{summary.portfolio_report}")
        if summary.failed:
            console.print("[yellow]失败标的：" + "；".join(summary.failed) + "[/yellow]")
    if summary.failed and not summary.succeeded:
        raise typer.Exit(1)


@app.command("organize")
def organize_command() -> None:
    """整理现有扁平报告到个股、组合、回测和归档目录。"""
    config = AppConfig.load()
    moved = organize_reports(config)
    if moved:
        console.print(f"已整理 {len(moved)} 个报告文件。")
        for path in moved:
            console.print(f"- {path.relative_to(config.home)}")
    else:
        console.print("没有需要整理的扁平报告。")


@app.command("analyze")
def analyze_command(
    symbol: Annotated[str, typer.Argument(help="要分析的标准或常见证券代码")],
    horizon: Annotated[str, typer.Option(help="short/medium/long/value/all")] = "all",
    as_of: Annotated[str | None, typer.Option(help="分析截止日 YYYY-MM-DD")] = None,
    evidence: Annotated[
        list[Path] | None,
        typer.Option("--evidence", help="可重复传入带日期的 Markdown/TXT 证据"),
    ] = None,
    fair_value_low: Annotated[float | None, typer.Option(help="人工保守合理价值下限")] = None,
    fair_value_high: Annotated[float | None, typer.Option(help="人工保守合理价值上限")] = None,
    skip_sync: Annotated[bool, typer.Option(help="不在分析前尝试刷新行情")] = False,
    skip_llm: Annotated[bool, typer.Option(help="只检索证据，不调用 LLM")] = False,
    skip_chronos: Annotated[bool, typer.Option(help="只运行随机游走基线")] = False,
) -> None:
    """生成短线、中线、长线或价值分析及预测回执。"""
    horizon_value = horizon.lower()
    if horizon_value not in {"short", "medium", "long", "value", "all"}:
        raise typer.BadParameter("horizon 必须是 short/medium/long/value/all")
    if (fair_value_low is None) != (fair_value_high is None):
        raise typer.BadParameter("人工合理价值上下限必须同时提供")
    config, database = _context()
    instrument = Instrument.parse(symbol)
    canonical = instrument.canonical
    analysis_date = _parse_date(as_of, date.today())
    sync_warnings: list[str] = []
    if not skip_sync and analysis_date >= date.today() - timedelta(days=2):
        try:
            result = sync_symbol(
                database,
                canonical,
                start=analysis_date - timedelta(days=365 * 5 + 2),
                end=analysis_date,
            )
            sync_warnings.extend(result.warnings)
        except Exception as exc:
            sync_warnings.append(f"刷新失败，尝试缓存: {exc}")
    frame, return_warnings = _load_analysis_frame(database, canonical, analysis_date)
    if frame.empty:
        database.close()
        console.print("[red]没有缓存行情。请先安装 data 依赖并运行 stock sync。[/red]")
        raise typer.Exit(1)
    data_quality, quality_warnings = quality_summary(frame, analysis_date)
    research = run_research(
        database=database,
        config=config,
        symbol=canonical,
        as_of=analysis_date,
        evidence_paths=evidence or [],
        use_llm=not skip_llm,
    )
    event_rows = active_event_rows(database, canonical, analysis_date)
    requested_horizons: tuple[int, ...]
    if horizon_value == "short":
        requested_horizons = SHORT_HORIZONS
    elif horizon_value == "medium":
        requested_horizons = MEDIUM_HORIZONS
    elif horizon_value == "all":
        requested_horizons = SHORT_HORIZONS + MEDIUM_HORIZONS
    else:
        requested_horizons = ()
    forecast_config = config.section("forecast")
    chronos_forecaster = None
    if requested_horizons and not skip_chronos:
        chronos_forecaster = Chronos2Forecaster(
            model_id=str(forecast_config.get("model_id", "amazon/chronos-2")),
            device=str(forecast_config.get("device", "cpu")),
            context_length=int(forecast_config.get("context_length", 512)),
        )
    forecasts = [
        forecast_one(
            symbol=canonical,
            as_of=analysis_date,
            horizon_days=days,
            frame=frame,
            data_quality=data_quality,
            database=database,
            config=config,
            event_rows=event_rows,
            use_chronos=not skip_chronos,
            chronos_forecaster=chronos_forecaster,
        )
        for days in requested_horizons
    ]
    try:
        snapshot = latest_portfolio_snapshot(config)
        current_weight = position_weight(snapshot, canonical)
    except Exception:
        current_weight = None
    package = analyze_package(
        config=config,
        database=database,
        symbol=canonical,
        as_of=analysis_date,
        frame=frame,
        data_quality=data_quality,
        data_warnings=list(dict.fromkeys(sync_warnings + return_warnings + quality_warnings)),
        forecasts=forecasts,
        research=research,
        current_weight=current_weight,
        fair_value_low=fair_value_low,
        fair_value_high=fair_value_high,
    )
    if horizon_value != "all":
        selected = Horizon(horizon_value)
        package.decisions = [item for item in package.decisions if item.horizon is selected]
    safe_symbol = canonical.replace(":", "-").replace("/", "-")
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
            if bool(chart_config.get("include_probability_fan", True)) and forecasts:
                probability_name = (
                    f"{analysis_date.isoformat()}-{safe_symbol}-"
                    f"{safe_filename_component(package.name)}-概率路径图.{chart_format}"
                )
                render_probability_chart(
                    package,
                    config.reports_dir / "个股" / safe_symbol / "charts" / probability_name,
                )
                package.chart_paths.append(f"charts/{probability_name}")
        except Exception as exc:
            package.data_warnings.append(f"图表未生成: {exc}")
    create_receipts(database, package)
    path = _write_report(
        config,
        f"{analysis_date.isoformat()}-{safe_symbol}-"
        f"{safe_filename_component(package.name)}-{horizon_value}.md",
        render_analysis_markdown(package),
        Path("个股") / safe_symbol,
    )
    _print_analysis_summary(package)
    console.print(f"\n[green]报告：{path}[/green]")
    database.close()


def _print_analysis_summary(package: AnalysisPackage) -> None:
    table = Table(title=f"{package.name} {package.symbol} 多周期冲突卡")
    table.add_column("周期")
    table.add_column("评分", justify="right")
    table.add_column("置信度", justify="right")
    table.add_column("行动")
    for item in package.decisions:
        table.add_row(
            HORIZON_LABELS[item.horizon],
            f"{item.score:+.2f}",
            f"{item.confidence:.0%}",
            item.action,
        )
    console.print(table)
    price_text = (
        f"数据 {package.data_quality.value} 级；"
        f"价格 {package.current_price:.3f} {package.currency}；"
    )
    console.print(
        price_text + f"回执 {', '.join(package.receipt_ids) if package.receipt_ids else '无'}"
    )


@app.command("portfolio")
def portfolio_command() -> None:
    """读取最新 Obsidian 持仓快照并执行仓位、行业和现金硬约束检查。"""
    config, database = _context()
    try:
        snapshot = latest_portfolio_snapshot(config)
    except Exception as exc:
        database.close()
        console.print(f"[red]持仓解析失败：{exc}[/red]")
        raise typer.Exit(1) from exc
    content = render_portfolio_markdown(snapshot, config, database)
    path = _write_report(
        config,
        f"{date.today().isoformat()}-组合检查.md",
        content,
        Path("组合"),
    )
    console.print(f"[green]已检查 {len(snapshot.positions)} 个 A 股持仓；报告：{path}[/green]")
    for warning in snapshot.warnings:
        console.print(f"[yellow]- {warning}[/yellow]")
    database.close()


@app.command("evaluate")
def evaluate_command(
    backtest: Annotated[
        str | None,
        typer.Option(help="可选证券代码；提供后执行 walk-forward"),
    ] = None,
    horizon_days: Annotated[int, typer.Option(min=1, max=120)] = 20,
    max_windows: Annotated[int, typer.Option(min=1, max=500)] = 100,
    step: Annotated[
        int | None,
        typer.Option(min=1, help="walk-forward 窗口步长；默认等于期限，设为 1 可快速积累校准样本"),
    ] = None,
    with_chronos: Annotated[
        bool, typer.Option(help="walk-forward 中运行 Chronos；耗时较长")
    ] = False,
) -> None:
    """核对到期预测，或执行至少三年历史的 walk-forward 校准。"""
    config, database = _context()
    if backtest:
        canonical = Instrument.parse(backtest).canonical
        frame, warnings = _load_analysis_frame(database, canonical, date.today())
        if frame.empty:
            database.close()
            console.print("[red]没有可回测行情，请先 stock sync。[/red]")
            raise typer.Exit(1)
        try:
            chronos_forecaster = (
                Chronos2Forecaster(
                    model_id=str(config.section("forecast").get("model_id", "amazon/chronos-2")),
                    device=str(config.section("forecast").get("device", "cpu")),
                    context_length=int(config.section("forecast").get("context_length", 512)),
                )
                if with_chronos
                else None
            )
            summary = walk_forward_backtest(
                symbol=canonical,
                frame=frame,
                horizon_days=horizon_days,
                database=database,
                config=config,
                use_chronos=with_chronos,
                max_windows=max_windows,
                step=step,
                chronos_forecaster=chronos_forecaster,
            )
        except Exception as exc:
            database.close()
            console.print(f"[red]回测失败：{exc}[/red]")
            raise typer.Exit(1) from exc
        lines = [
            "---",
            "type: forecast-evaluation",
            f"date: {date.today().isoformat()}",
            f"symbol: {canonical}",
            f"name: {config.asset(canonical).get('name', canonical)}",
            "---",
            "",
            f"# {config.asset(canonical).get('name', canonical)}（{canonical}）walk-forward 评估",
            "",
            f"- 窗口：{summary['windows']}",
            f"- 期限：{horizon_days} 个交易日",
            f"- Chronos：{'启用' if with_chronos else '未启用'}",
            f"- 方向命中率：{summary['direction_accuracy']:.1%}",
            f"- 80% 区间覆盖率：{summary['interval_coverage']:.1%}",
            f"- 集成分位损失：{summary['ensemble_pinball_loss']:.6f}",
            f"- 基线分位损失：{summary['baseline_pinball_loss']:.6f}",
        ]
        if summary["chronos_pinball_loss"] is not None:
            lines.append(f"- Chronos 分位损失：{summary['chronos_pinball_loss']:.6f}")
        lines.extend(f"- 数据警告：{warning}" for warning in warnings)
        path = _write_report(
            config,
            f"{date.today().isoformat()}-{canonical.replace(':', '-')}-"
            f"{safe_filename_component(str(config.asset(canonical).get('name', canonical)))}-"
            f"回测-{horizon_days}d.md",
            "\n".join(lines) + "\n",
            Path("回测") / canonical.replace(":", "-").replace("/", "-"),
        )
        console.print_json(json.dumps(summary, ensure_ascii=False))
        console.print(f"[green]报告：{path}[/green]")
    else:
        evaluated = evaluate_open_receipts(database)
        if not evaluated:
            console.print("没有已到期且具备行情的开放预测回执。")
        else:
            table = Table(title="到期预测核对")
            for column in ("回执", "标的", "实际收益", "方向", "区间覆盖", "分位损失"):
                table.add_column(column)
            for item in evaluated:
                table.add_row(
                    item["id"],
                    item["symbol"],
                    f"{item['actual_return']:+.1%}",
                    "✓" if item["direction_correct"] else "✗",
                    "✓" if item["interval_covered"] else "✗",
                    f"{item['ensemble_loss']:.5f}",
                )
            console.print(table)
    database.close()


@app.command("calibrate")
def calibrate_command(
    symbol: Annotated[str | None, typer.Option(help="可选；不填则校准配置中的全部标的")] = None,
    horizon_days: Annotated[
        list[int] | None,
        typer.Option(
            "--horizon-days",
            min=1,
            max=120,
            help="预测期限；可重复传入，默认只校准 20 日",
        ),
    ] = None,
    max_windows: Annotated[int, typer.Option(min=1, max=500)] = 100,
    step: Annotated[int, typer.Option(min=1, help="窗口步长，1 会产生最多校准样本")] = 1,
    with_chronos: Annotated[
        bool, typer.Option("--with-chronos/--without-chronos", help="是否比较 Chronos-2")
    ] = True,
) -> None:
    """补齐历史数据并执行 walk-forward 校准；不做参数微调，不使用未来数据。"""
    config, database = _context()
    selected = [Instrument.parse(symbol).canonical] if symbol else configured_symbols(config)
    horizons = tuple(horizon_days or [20])
    end_date = date.today()
    start_date = end_date - timedelta(
        days=365 * int(config.section("automation").get("sync_lookback_years", 5)) + 2
    )
    chronos_forecaster = None
    if with_chronos:
        forecast_config = config.section("forecast")
        chronos_forecaster = Chronos2Forecaster(
            model_id=str(forecast_config.get("model_id", "amazon/chronos-2")),
            device=str(forecast_config.get("device", "cpu")),
            context_length=int(forecast_config.get("context_length", 512)),
        )
    results: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    for canonical in selected:
        try:
            sync_symbol(database, canonical, start=start_date, end=end_date)
            bars = database.load_bars(canonical, end_date)
            actions = database.load_actions(canonical, end_date)
            frame, return_warnings = total_return_frame(bars, actions)
            if frame.empty:
                raise RuntimeError("没有可用行情缓存")
            if return_warnings:
                raise RuntimeError("；".join(return_warnings) + "；请先补齐公司行动")
            for days in horizons:
                summary = walk_forward_backtest(
                    symbol=canonical,
                    frame=frame,
                    horizon_days=days,
                    database=database,
                    config=config,
                    use_chronos=with_chronos,
                    max_windows=max_windows,
                    step=step,
                    chronos_forecaster=chronos_forecaster,
                )
                summary["name"] = str(config.asset(canonical).get("name", canonical))
                results.append(summary)
        except Exception as exc:
            failures[canonical] = str(exc)
    database.close()
    console.print_json(
        json.dumps({"results": results, "failed": failures}, ensure_ascii=False, default=str)
    )
    if failures and not results:
        raise typer.Exit(1)


@app.command("add")
def add_command(
    symbol: Annotated[str, typer.Argument(help="标准证券代码，如 CN:600519、HK:00700、US:NVDA")],
    name: Annotated[str | None, typer.Option(help="资产名称，如 贵州茅台")] = None,
    sector: Annotated[str | None, typer.Option(help="所属行业，如 消费、金融、科技")] = None,
    role: Annotated[
        str, typer.Option(help="组合角色：core (核心) / satellite (卫星)")
    ] = "satellite",
    valuation_model: Annotated[
        str, typer.Option(help="估值模型：generic/bank/insurer/cyclical/fund")
    ] = "generic",
    fair_pe: Annotated[float | None, typer.Option(help="保守合理 PE 参考")] = None,
    fair_pb: Annotated[float | None, typer.Option(help="保守合理 PB 参考")] = None,
    sync: Annotated[
        bool, typer.Option("--sync/--no-sync", help="添加后是否立即同步历史数据")
    ] = True,
    dry_run: Annotated[bool, typer.Option(help="仅预览配置而不写入文件")] = False,
) -> None:
    """向导式添加新标的，校验参数、写入配置并自动同步历史数据。"""
    config, database = _context()
    try:
        instrument = Instrument.parse(symbol)
    except Exception as exc:
        console.print(f"[red]代码格式错误：{exc}[/red]")
        raise typer.Exit(1) from exc

    canonical = instrument.canonical
    display_name = name or canonical
    assigned_sector = sector or "未分类"
    normalized_role = role.lower()
    if normalized_role not in {"core", "satellite"}:
        raise typer.BadParameter("role 必须是 core 或 satellite")
    normalized_model = valuation_model.lower()

    toml_path = config.home / "stock-analysis.toml"
    existing_text = toml_path.read_text(encoding="utf-8") if toml_path.exists() else ""
    if f'[assets."{canonical}"]' in existing_text:
        console.print(f"[yellow]标的 {canonical} 已存在于配置中。[/yellow]")
    else:
        new_block = [
            f'\n[assets."{canonical}"]',
            f'name = "{display_name}"',
            f'sector = "{assigned_sector}"',
            f'role = "{normalized_role}"',
            f'valuation_model = "{normalized_model}"',
        ]
        if fair_pe is not None:
            new_block.append(f"fair_pe = {fair_pe:.1f}")
        if fair_pb is not None:
            new_block.append(f"fair_pb = {fair_pb:.2f}")
        new_block.append("")

        content_to_write = existing_text.rstrip() + "\n" + "\n".join(new_block)
        if dry_run:
            console.print("[cyan][Dry Run] 准备追加如下配置：[/cyan]")
            console.print("\n".join(new_block))
        else:
            toml_path.write_text(content_to_write, encoding="utf-8")
            console.print(
                f"[green]已成功添加 {canonical}（{display_name}）到 {toml_path.name}[/green]"
            )

    table = Table(title=f"新标的配置：{display_name} ({canonical})")
    table.add_column("配置项")
    table.add_column("设定值")
    table.add_row("代码 (Canonical)", canonical)
    table.add_row("名称 (Name)", display_name)
    table.add_row("行业 (Sector)", assigned_sector)
    table.add_row("角色 (Role)", normalized_role)
    table.add_row("估值模型 (Valuation Model)", normalized_model)
    if fair_pe:
        table.add_row("参考 PE", f"{fair_pe:.1f}")
    if fair_pb:
        table.add_row("参考 PB", f"{fair_pb:.2f}")
    console.print(table)

    if sync and not dry_run:
        console.print("\n[cyan]正在同步历史数据...[/cyan]")
        try:
            res = sync_symbol(
                database,
                canonical,
                start=date.today() - timedelta(days=365 * 5 + 2),
                end=date.today(),
            )
            console.print(
                f"[green]同步成功[/green]：获取 {res.bars} 根日线，"
                f"{res.actions} 条公司行动，数据质量 **{res.quality.value}** 级"
            )
        except Exception as exc:
            console.print(f"[yellow]同步出现警告或失败：{exc}[/yellow]")
    database.close()


@app.command("scenario")
def scenario_command(
    symbol: Annotated[str, typer.Argument(help="证券代码，如 CN:601318")],
    eps_growth_delta: Annotated[
        float,
        typer.Option(help="盈利增速/净利润预期变动比例（如 -0.10 表示盈利下调 10%）"),
    ] = 0.0,
    pe_delta: Annotated[
        float,
        typer.Option(help="目标估值 PE 变动（如 -2.0 表示目标 PE 下调 2 倍）"),
    ] = 0.0,
    margin_delta: Annotated[
        float,
        typer.Option(help="安全边际比例变动（如 0.05 表示安全边际从 20% 提高到 25%）"),
    ] = 0.0,
) -> None:
    """What-If 敏感性情景推演：分析盈利变动、估值调整与安全边际对买入底线的影响。"""
    config, database = _context()
    canonical = Instrument.parse(symbol).canonical
    profile = config.asset(canonical)
    name = str(profile.get("name", canonical))
    records = database.latest_fundamentals(canonical, date.today())
    bars = database.load_bars(canonical, date.today())
    if bars.empty:
        database.close()
        console.print("[red]没有可用行情缓存，请先 stock sync。[/red]")
        raise typer.Exit(1)
    current_price = float(bars.iloc[-1]["close"])
    currency = str(bars.iloc[-1]["currency"])

    base_strategy = get_valuation_strategy(profile.get("valuation_model"))
    base_range = base_strategy.range(
        current_price=current_price,
        currency=currency,
        records=records,
        profile=profile,
    )

    scenario_profile = dict(profile)
    base_fair_pe = float(profile.get("fair_pe", base_strategy.default_fair_pe))
    scenario_profile["fair_pe"] = max(1.0, base_fair_pe + pe_delta)

    scenario_records = dict(records)
    if "pe" in scenario_records and scenario_records["pe"].value:
        old_pe = float(scenario_records["pe"].value)
        adj_factor = max(0.01, 1.0 + eps_growth_delta)
        scenario_records["pe"] = FundamentalRecord(
            symbol=canonical,
            metric="pe",
            value=old_pe / adj_factor,
            as_of=date.today(),
            source="scenario",
        )

    scenario_strategy_cls = type(base_strategy)

    class CustomScenarioStrategy(scenario_strategy_cls):
        pass

    custom_strat = CustomScenarioStrategy()
    custom_strat.safety_margin = max(0.05, min(0.60, base_strategy.safety_margin + margin_delta))

    scenario_range = custom_strat.range(
        current_price=current_price,
        currency=currency,
        records=scenario_records,
        profile=scenario_profile,
    )

    table = Table(title=f"What-If 情景推演：{name} ({canonical})")
    table.add_column("维度")
    table.add_column("基准情景 (Baseline)")
    table.add_column("推演情景 (Scenario)")
    table.add_column("变动影响", justify="right")

    table.add_row(
        "现价 (Current Price)",
        f"{current_price:.2f} {currency}",
        f"{current_price:.2f} {currency}",
        "—",
    )
    table.add_row(
        "假设参考 PE",
        f"{base_fair_pe:.1f}x",
        f"{scenario_profile['fair_pe']:.1f}x",
        f"{pe_delta:+.1f}x",
    )
    table.add_row(
        "盈利预期调整",
        "基准 (0%)",
        f"{eps_growth_delta:+.1%}",
        f"{eps_growth_delta:+.1%}",
    )
    table.add_row(
        "安全边际要求",
        f"{base_strategy.safety_margin:.0%}",
        f"{custom_strat.safety_margin:.0%}",
        f"{margin_delta:+.0%}",
    )

    if base_range.available and scenario_range.available:
        b_fair = f"{base_range.fair_low:.2f}–{base_range.fair_high:.2f}"
        s_fair = f"{scenario_range.fair_low:.2f}–{scenario_range.fair_high:.2f}"
        fair_delta = (
            (scenario_range.fair_low / base_range.fair_low - 1) if base_range.fair_low else 0
        )
        table.add_row("合理价值区间", b_fair, s_fair, f"{fair_delta:+.1%}")

        b_buy = f"{base_range.buy_low:.2f}–{base_range.buy_high:.2f}"
        s_buy = f"{scenario_range.buy_low:.2f}–{scenario_range.buy_high:.2f}"
        buy_delta = (
            (scenario_range.buy_high / base_range.buy_high - 1) if base_range.buy_high else 0
        )
        table.add_row("分批买入观察线", b_buy, s_buy, f"{buy_delta:+.1%}")

        status_base = (
            "✓ 处于买入线下方" if current_price <= (base_range.buy_high or 0) else "○ 高于买入线"
        )
        status_scen = (
            "✓ 处于买入线下方"
            if current_price <= (scenario_range.buy_high or 0)
            else "○ 高于买入线"
        )
        table.add_row("当前价格状态", status_base, status_scen, "—")

    console.print(table)
    database.close()


@app.command("dash")
def dash_command() -> None:
    """终端交互式多周期决策与资产配置总览看板。"""
    config, database = _context()
    symbols = configured_symbols(config)
    if not symbols:
        console.print("[yellow]配置中没有配置任何标的。[/yellow]")
        database.close()
        return

    table = Table(title="StockAnalysis 实时决策看板 (Multi-Horizon Dashboard)")
    table.add_column("标的 (Symbol)", style="bold")
    table.add_column("现价 (Price)", justify="right")
    table.add_column("数据", justify="center")
    table.add_column("短线 (1-20d)", justify="center")
    table.add_column("中线 (1-6m)", justify="center")
    table.add_column("长线 (1-3y)", justify="center")
    table.add_column("价值 (3-10y)", justify="center")
    table.add_column("最近预警 / 状态")

    for canonical in symbols:
        name = str(config.asset(canonical).get("name", canonical))
        bars = database.load_bars(canonical, date.today())
        if bars.empty:
            table.add_row(
                f"{name}\n[dim]{canonical}[/dim]",
                "—",
                "无数据",
                "—",
                "—",
                "—",
                "—",
                "[red]未同步行情[/red]",
            )
            continue
        current_price = float(bars.iloc[-1]["close"])
        currency = str(bars.iloc[-1]["currency"])
        quality, _ = quality_summary(bars, date.today())

        frame, _ = _load_analysis_frame(database, canonical, date.today())
        research = run_research(
            database=database,
            config=config,
            symbol=canonical,
            as_of=date.today(),
            use_llm=False,
        )
        package = analyze_package(
            config=config,
            database=database,
            symbol=canonical,
            as_of=date.today(),
            frame=frame,
            data_quality=quality,
            data_warnings=[],
            forecasts=[],
            research=research,
        )

        decisions_by_horizon = {d.horizon.value: d for d in package.decisions}
        short_d = decisions_by_horizon.get("short")
        med_d = decisions_by_horizon.get("medium")
        long_d = decisions_by_horizon.get("long")
        val_d = decisions_by_horizon.get("value")

        def _cell(d: HorizonDecision | None) -> str:
            if not d:
                return "—"
            if "买入" in d.action:
                color = "green"
            elif "持有" in d.action:
                color = "yellow"
            elif "减仓" in d.action or "回避" in d.action:
                color = "red"
            else:
                color = "white"
            return f"[{color}]{d.action}[/{color}]\n[dim]{d.score:+.2f} ({d.confidence:.0%})[/dim]"

        vr = package.valuation_range
        warnings_txt = "正常"
        if quality is DataQuality.C:
            warnings_txt = "[red]数据质量C级[/red]"
        elif vr.available and vr.buy_high and current_price <= vr.buy_high:
            warnings_txt = "[bold green]进入安全买入区[/bold green]"

        table.add_row(
            f"{name}\n[dim]{canonical}[/dim]",
            f"{current_price:.2f} {currency}",
            quality.value,
            _cell(short_d),
            _cell(med_d),
            _cell(long_d),
            _cell(val_d),
            warnings_txt,
        )

    console.print(table)
    database.close()


if __name__ == "__main__":
    app()
