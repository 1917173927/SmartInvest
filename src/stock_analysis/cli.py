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
    fetch_latest_quote,
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
    StagingPlan,
    analyze_package,
    compute_staging_plan,
    create_receipts,
    get_valuation_strategy,
    latest_portfolio_snapshot,
    position_weight,
    render_analysis_markdown,
    render_portfolio_markdown,
)
from stock_analysis.files import atomic_write_text
from stock_analysis.forecast import (
    MEDIUM_HORIZONS,
    SHORT_HORIZONS,
    Chronos2Forecaster,
    evaluate_open_receipts,
    forecast_one,
    walk_forward_backtest,
)
from stock_analysis.morning import generate_morning_brief
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
    return atomic_write_text(path, content)


def _load_analysis_frame(
    database: Database, symbol: str, as_of: date
) -> tuple[pd.DataFrame, list[str]]:
    bars = database.load_bars(symbol, as_of)
    actions = database.load_actions(symbol, as_of)
    frame, warnings = total_return_frame(bars, actions)
    return frame, warnings


def _resolve_size_capital(
    config: AppConfig,
    instrument: Instrument,
    override: float | None,
) -> tuple[float, str]:
    if override is not None:
        if override <= 0:
            raise typer.BadParameter("--capital 必须大于 0")
        return override, "命令行临时覆盖"

    portfolio = config.section("portfolio")
    key = {
        "CN": "cn_account_assets",
        "CNFUND": "cn_account_assets",
        "HK": "hk_account_assets",
        "US": "us_account_assets",
    }[instrument.market.value]
    value = portfolio.get(key)
    try:
        capital = float(value)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(
            f"未配置 portfolio.{key}；请在 stock-analysis.toml 设置真实账户资产，"
            "或显式传入 --capital"
        ) from exc
    if capital <= 0:
        raise typer.BadParameter(f"portfolio.{key} 必须大于 0")
    as_of = portfolio.get(f"{key}_as_of", "日期未注明")
    return capital, f"配置 portfolio.{key}，截至 {as_of}"


def _execution_guidance(
    current_price: float,
    plan: StagingPlan,
    actionable: bool,
    *,
    current_weight: float,
) -> str:
    if not actionable:
        return (
            "[bold red]当前动作：暂停下单。[/bold red] 实时报价不可用；先在券商核对现价，"
            "再用 [bold]--price 券商现价[/bold] 重新运行。"
        )
    if len(plan.tiers) < 3:
        return "[bold red]当前动作：暂停下单。[/bold red] 未生成完整三档计划。"
    if current_weight >= plan.total_target_weight:
        return (
            f"[bold red]当前动作：不新增买入，并取消未成交加仓单。[/bold red] 当前仓位约 "
            f"{current_weight:.2%}，已达到或超过 {plan.total_target_weight:.0%} 目标上限；"
            "先核对持仓数量和账户资产，不能继续按三档计划加仓。"
        )

    first, second, third = plan.tiers[:3]
    invalidation = plan.invalidation_price
    if invalidation and current_price < invalidation:
        return (
            f"[bold red]当前动作：取消未成交买单并复核投资逻辑。[/bold red] 现价已低于失效线 "
            f"{invalidation:.2f}；不要机械补仓。"
        )
    if current_price > first.target_price:
        gap = current_price / first.target_price - 1
        return (
            f"[bold yellow]当前动作：等待，不追涨。[/bold yellow] 现价高于首笔价 "
            f"{gap:.2%}；如需预埋，仅挂不高于 {first.target_price:.2f} 的限价/条件单。"
        )
    if current_price > second.target_price:
        return (
            f"[bold green]当前动作：只处理第一档。[/bold green] 价格已到首笔区；"
            f"限价不高于 {first.target_price:.2f}，本档最多 {first.shares} 股，"
            "第二、三档不要同时市价买入。"
        )
    if current_price > third.target_price:
        return (
            f"[bold yellow]当前动作：核对第一档成交和当前仓位后，再考虑第二档。[/bold yellow] "
            f"第二档限价不高于 {second.target_price:.2f}，本档最多 {second.shares} 股。"
        )
    return (
        "[bold red]当前动作：价格已到第三档深跌区。[/bold red] 先排除基本面或事件性风险；"
        f"仅在前两档已按计划执行且总仓位未超限时，才考虑不高于 {third.target_price:.2f} "
        f"的最多 {third.shares} 股限价单。"
    )


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
            f"- 窗口总数：{summary['windows']}",
            f"- 触发交易信号数：{summary['active_trades']}",
            f"- 期限：{horizon_days} 个交易日",
            f"- Chronos：{'启用' if with_chronos else '未启用'}",
            f"- **模型胜率 (Win Rate)**：**{summary['win_rate']:.1%}**",
            f"- **盈亏比 (Profit Factor)**：**{summary['profit_factor']:.2f}**",
            f"- **平均单笔期望收益 (Expected Return)**：**{summary['expected_return']:+.2%}**",
            f"- **年化夏普比率 (Sharpe Ratio)**：{summary['sharpe_ratio']:.2f}",
            f"- **策略最大回撤 (Max Drawdown)**：{summary['max_drawdown']:.2%}",
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


@app.command("size")
def size_command(
    symbol: Annotated[str, typer.Argument(help="标准证券代码，如 CN:601318")],
    capital: Annotated[
        float | None,
        typer.Option(help="临时覆盖配置中的账户总资产；不传则读取 [portfolio]"),
    ] = None,
    price: Annotated[
        float | None,
        typer.Option(help="券商盘中现价；优先级高于网络报价，用于下单前人工核对"),
    ] = None,
    held_shares: Annotated[
        int | None,
        typer.Option(help="临时覆盖配置中的当前持股数；不传则读取资产的 current_shares"),
    ] = None,
    target_weight: Annotated[
        float | None, typer.Option(help="自定义目标仓位上限（如 0.15 表示 15%）")
    ] = None,
    risk_budget: Annotated[
        float, typer.Option(help="单笔交易最大承受风险比例（如 0.02 表示 2%）")
    ] = 0.02,
) -> None:
    """实盘仓位测算与阶梯挂单生成器：根据账户资产与风险预算，精确计算三档买点股数与止损线。"""
    config, database = _context()
    instrument = Instrument.parse(symbol)
    canonical = instrument.canonical
    profile = config.asset(canonical)
    name = str(profile.get("name", canonical))
    role = str(profile.get("role", "satellite"))
    resolved_capital, capital_source = _resolve_size_capital(config, instrument, capital)
    configured_shares = profile.get("current_shares", 0) if held_shares is None else held_shares
    try:
        current_shares = int(configured_shares)
    except (TypeError, ValueError) as exc:
        database.close()
        raise typer.BadParameter("current_shares/--held-shares 必须是非负整数") from exc
    if current_shares < 0:
        database.close()
        raise typer.BadParameter("current_shares/--held-shares 必须是非负整数")

    bars = database.load_bars(canonical, date.today())
    if bars.empty:
        database.close()
        console.print("[red]没有可用行情缓存，请先运行 stock sync。[/red]")
        raise typer.Exit(1)

    reference_price = float(bars.iloc[-1]["close"])
    reference_date = pd.Timestamp(bars.iloc[-1]["trade_date"]).date()
    currency = str(bars.iloc[-1].get("currency", "CNY"))
    quote_warnings: list[str] = []
    quote_actionable = True
    if price is not None:
        if price <= 0:
            database.close()
            raise typer.BadParameter("--price 必须大于 0")
        current_price = price
        price_source = "券商现价（手工输入）"
    else:
        quote, quote_warnings = fetch_latest_quote(instrument)
        if quote is not None:
            current_price = quote.price
            fetched_at = quote.fetched_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            price_source = f"{quote.source}，获取于 {fetched_at}（可能延迟）"
        else:
            current_price = reference_price
            price_source = f"历史日线缓存，交易日 {reference_date.isoformat()}"
            quote_actionable = False
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

    assigned_weight = (
        target_weight
        if target_weight is not None
        else (package.decisions[0].target_position if package.decisions else None)
    )
    target_weight = (
        assigned_weight if assigned_weight is not None else (0.20 if role == "core" else 0.10)
    )
    current_position_value = current_shares * current_price
    current_weight = current_position_value / resolved_capital

    plan = compute_staging_plan(
        current_price=reference_price,
        valuation_range=package.valuation_range,
        price_zones=package.price_zones,
        role=role,
        total_capital=resolved_capital,
        target_position=assigned_weight,
        risk_budget=risk_budget,
        existing_position_value=current_position_value,
    )

    console.print(f"\n[bold cyan]🎯 实盘阶梯建仓测算：{name} ({canonical})[/bold cyan]")
    allocated_cap = resolved_capital * plan.total_target_weight
    console.print(
        f"盘中执行价: [bold]{current_price:.2f} {currency}[/bold] | "
        f"日线分析基准: [bold]{reference_price:.2f} {currency}[/bold] ({reference_date})"
    )
    console.print(f"报价来源: {price_source}")
    console.print(
        f"账户总资产: [bold]{resolved_capital:,.2f} {currency}[/bold] "
        f"([dim]{capital_source}[/dim]) | "
        f"目标上限: [bold]{plan.total_target_weight:.0%}[/bold] ({allocated_cap:,.2f} {currency})"
    )
    console.print(
        f"当前持仓: [bold]{current_shares} 股[/bold]，按执行价计 "
        f"[bold]{current_position_value:,.2f} {currency}[/bold]，约占账户 "
        f"[bold]{current_weight:.2%}[/bold]；剩余目标额度 "
        f"[bold]{max(0.0, resolved_capital * target_weight - current_position_value):,.2f} "
        f"{currency}[/bold]"
    )
    for warning in quote_warnings:
        console.print(f"[yellow]报价提示：{warning}[/yellow]")

    table = Table(title="阶梯挂单执行计划 (Staging Execution Plan)")
    table.add_column("批次", style="bold")
    table.add_column("挂单/触发价", justify="right")
    table.add_column("配比", justify="center")
    table.add_column("建议股数", justify="right")
    table.add_column("建议手数", justify="right")
    table.add_column("占用资金", justify="right")
    table.add_column("执行逻辑与依据")

    for tier in plan.tiers:
        lots = tier.shares // 100
        table.add_row(
            tier.tier_name,
            f"{tier.target_price:.2f} {currency}",
            f"{tier.weight_pct:.0%}",
            f"{tier.shares} 股",
            f"{lots} 手" if lots > 0 else "不足1手",
            f"{tier.allocated_amount:,.2f} {currency}",
            (
                tier.rationale
                if tier.shares > 0
                else "当前剩余目标额度不足或仓位已达上限；本档不下单"
            ),
        )

    console.print(table)
    console.print(
        _execution_guidance(
            current_price,
            plan,
            quote_actionable,
            current_weight=current_weight,
        )
    )
    console.print(
        "[dim]下单前必须以券商盘口复核价格、可用资金、已成交数量和当前总仓位；"
        "本工具输出是条件计划，不是收益承诺。[/dim]"
    )

    if plan.invalidation_price:
        max_loss = max(
            0.0,
            sum(
                tier.shares * max(0.0, tier.target_price - plan.invalidation_price)
                for tier in plan.tiers
            ),
        )
        loss_pct_of_capital = max_loss / resolved_capital
        console.print(
            f"[bold red]🛑 逻辑失效与止损参考线[/bold red]：< "
            f"[bold]{plan.invalidation_price:.2f} {currency}[/bold]"
        )
        console.print(
            f"预估全单极端最大亏损金额: [bold]{max_loss:,.2f} {currency}[/bold] "
            f"([bold]{loss_pct_of_capital:.2%}[/bold] 账户总资产)"
        )
        console.print(f"失效说明: {plan.invalidation_note}\n")

    database.close()


@app.command("compare")
def compare_command(
    symbols: Annotated[
        list[str],
        typer.Argument(help="待比较的证券代码列表，如 CN:601318 CN:600519 HK:00700"),
    ],
) -> None:
    """跨标的多维横向比对矩阵：对比各标的的估值折扣、多周期信号与建仓优先级。"""
    config, database = _context()
    if not symbols:
        console.print("[yellow]请提供至少一个证券代码。[/yellow]")
        database.close()
        return

    packages: list[AnalysisPackage] = []
    for raw in symbols:
        try:
            canonical = Instrument.parse(raw).canonical
        except Exception:
            canonical = raw
        bars = database.load_bars(canonical, date.today())
        if bars.empty:
            console.print(f"[yellow]标的 {canonical} 没有可用行情缓存，已跳过。[/yellow]")
            continue
        quality, _ = quality_summary(bars, date.today())
        frame, _ = _load_analysis_frame(database, canonical, date.today())
        research = run_research(
            database=database,
            config=config,
            symbol=canonical,
            as_of=date.today(),
            use_llm=False,
        )
        pkg = analyze_package(
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
        packages.append(pkg)

    if not packages:
        console.print("[red]未能成功分析任何标的。[/red]")
        database.close()
        return

    def _priority(p: AnalysisPackage) -> float:
        val_score = 0.0
        if p.valuation_range.available and p.valuation_range.buy_high:
            if p.current_price <= p.valuation_range.buy_high:
                val_score = 1.0 + (p.valuation_range.buy_high / p.current_price - 1)
            elif p.valuation_range.fair_low and p.current_price <= p.valuation_range.fair_low:
                val_score = 0.5
            else:
                val_score = -0.5
        dec_score = sum(d.score for d in p.decisions) / max(1, len(p.decisions))
        return float(val_score * 0.6 + dec_score * 0.4)

    ranked = sorted(packages, key=_priority, reverse=True)

    table = Table(title="SmartInvest 跨标的多维优选与比对矩阵 (Comparison Matrix)")
    table.add_column("排名 / 标的", style="bold")
    table.add_column("现价", justify="right")
    table.add_column("数据", justify="center")
    table.add_column("估值状态", justify="center")
    table.add_column("买入线距离", justify="right")
    table.add_column("短线", justify="center")
    table.add_column("中线", justify="center")
    table.add_column("长线/价值", justify="center")
    table.add_column("优选建议", style="bold cyan")

    for rank, pkg in enumerate(ranked, 1):
        vr = pkg.valuation_range
        dist_str = "—"
        val_str = "中性"
        if vr.available and vr.buy_high:
            dist = pkg.current_price / vr.buy_high - 1
            dist_str = f"{dist:+.1%}"
            if pkg.current_price <= vr.buy_high:
                val_str = "[green]进入买入区[/green]"
            elif vr.fair_low and pkg.current_price <= vr.fair_low:
                val_str = "[yellow]合理偏低[/yellow]"
            else:
                val_str = "[white]高于买入线[/white]"

        decisions_map = {d.horizon.value: d for d in pkg.decisions}
        s_d = decisions_map.get("short")
        m_d = decisions_map.get("medium")
        l_d = decisions_map.get("long") or decisions_map.get("value")

        def _badge(d: HorizonDecision | None) -> str:
            if not d:
                return "—"
            c = "green" if "买入" in d.action else "yellow" if "持有" in d.action else "red"
            return f"[{c}]{d.action}[/{c}]"

        priority_score = _priority(pkg)
        if priority_score >= 0.5:
            rec = "🔥 优先建仓"
        elif priority_score >= 0.0:
            rec = "👀 跟踪观察"
        else:
            rec = "✋ 暂缓观望"

        table.add_row(
            f"#{rank} {pkg.name}\n[dim]{pkg.symbol}[/dim]",
            f"{pkg.current_price:.2f} {pkg.currency}",
            pkg.data_quality.value,
            val_str,
            dist_str,
            _badge(s_d),
            _badge(m_d),
            _badge(l_d),
            rec,
        )

    console.print(table)
    database.close()


@app.command("morning")
def morning_command(
    capital: Annotated[float, typer.Option(help="账户总可用资金（元）")] = 100000.0,
    notify: Annotated[bool, typer.Option(help="是否发送 macOS 桌面弹窗通知")] = True,
) -> None:
    """生成 09:15 集合竞价前券商 App 预埋单/条件单晨报并输出挂单网格。"""
    config = AppConfig.load()
    brief = generate_morning_brief(config, total_capital=capital, send_notification=notify)
    console.print("\n[bold green]🌅 盘前挂单与执行晨报已生成！[/bold green]")
    console.print(f"报告路径: [bold cyan]{brief.report_path}[/bold cyan]")
    console.print(f"基准资金: [bold]{capital:,.2f} CNY[/bold]\n")

    table = Table(title=f"09:15 盘前券商挂单执行表 ({brief.as_of.isoformat()})")
    table.add_column("标的 / 优先级", style="bold")
    table.add_column("现价", justify="right")
    table.add_column("挂单批次")
    table.add_column("挂单价", justify="right")
    table.add_column("建议手数", justify="right")
    table.add_column("建议股数", justify="right")
    table.add_column("预估金额", justify="right")
    table.add_column("券商下单类型", style="cyan")

    for it in brief.items:
        for tier in it.plan.tiers:
            lots = tier.shares // 100
            order_type = (
                "集合竞价限价单"
                if "首笔" in tier.tier_name
                else "回调触达条件单"
                if "强支撑" in tier.tier_name
                else "低位限价埋单"
            )
            table.add_row(
                f"{it.priority}\n{it.name} ({it.canonical})",
                f"{it.current_price:.2f} {it.currency}",
                tier.tier_name,
                f"{tier.target_price:.2f} {it.currency}",
                f"{lots} 手" if lots > 0 else "不足1手",
                f"{tier.shares} 股",
                f"{tier.allocated_amount:,.2f}",
                order_type,
            )

    console.print(table)


if __name__ == "__main__":
    app()
