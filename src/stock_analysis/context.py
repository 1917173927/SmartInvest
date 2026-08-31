from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from stock_analysis.data import AppConfig, Database, DataQuality, Instrument, utc_now

LOGGER = logging.getLogger(__name__)

MACRO_TICKERS = {
    "WTI": "CL=F",
    "GOLD": "GC=F",
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "CSI300": "000300.SS",
}


@dataclass
class ContextRefreshResult:
    symbol: str | None = None
    news_count: int = 0
    macro_count: int = 0
    refreshed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def _fresh_enough(database: Database, table: str, key_column: str, key: str, hours: int) -> bool:
    row = database.connection.execute(
        f"SELECT MAX(fetched_at) AS fetched_at FROM {table} WHERE {key_column} = ?", (key,)
    ).fetchone()
    if not row or not row["fetched_at"]:
        return False
    try:
        fetched = datetime.fromisoformat(str(row["fetched_at"]))
    except ValueError:
        return False
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=UTC)
    return utc_now() - fetched < timedelta(hours=hours)


def _cache_snapshot(
    database: Database,
    *,
    table: str,
    key_column: str,
    key: str,
    date_column: str,
    end: date,
) -> tuple[int, date | None]:
    """Return the usable cache size and latest point-in-time date.

    Table and column names are internal constants supplied by this module; the
    values remain parameterized so a symbol/series cannot alter the query.
    """
    row = database.connection.execute(
        f"SELECT COUNT(*) AS count, MAX({date_column}) AS latest "
        f"FROM {table} WHERE {key_column} = ? AND {date_column} <= ?",
        (key, end.isoformat()),
    ).fetchone()
    if not row or not row["count"] or not row["latest"]:
        return 0, None
    try:
        return int(row["count"]), date.fromisoformat(str(row["latest"])[:10])
    except ValueError:
        return int(row["count"]), None


def _cache_warning(count: int, latest: date | None, end: date) -> str:
    if latest is None:
        return f"使用本地缓存 {count} 条，但缓存日期无法解析"
    stale_days = max(0, (end - latest).days)
    return f"使用本地缓存 {count} 条，最新 {latest.isoformat()}，陈旧 {stale_days} 天"


def _news_items(frame: pd.DataFrame, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    title_column = _column(frame, ("新闻标题", "标题", "title"))
    body_column = _column(frame, ("新闻内容", "内容", "summary", "摘要"))
    date_column = _column(frame, ("发布时间", "发布日期", "日期", "date"))
    url_column = _column(frame, ("新闻链接", "链接", "url", "source_url"))
    source_column = _column(frame, ("文章来源", "来源", "source"))
    if not title_column or not date_column:
        return []
    items: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        parsed = pd.to_datetime(row.get(date_column), errors="coerce")
        if pd.isna(parsed):
            continue
        published = parsed.date()
        if published < start or published > end:
            continue
        title = str(row.get(title_column, "")).strip()
        if not title:
            continue
        summary = str(row.get(body_column, "")).strip() if body_column else ""
        url = str(row.get(url_column, "")).strip() if url_column else ""
        if not url or url.lower() == "nan":
            url = f"https://so.eastmoney.com/news/s?keyword={symbol.split(':')[-1]}"
        source = str(row.get(source_column, "东方财富")) if source_column else "东方财富"
        digest = hashlib.sha256(f"{symbol}\n{published}\n{title}\n{url}".encode()).hexdigest()
        items.append(
            {
                "id": f"news-{digest[:16]}",
                "symbol": symbol,
                "title": title,
                "summary": summary[:10000],
                "source": source or "东方财富",
                "source_url": url,
                "published_at": published.isoformat(),
                "quality": DataQuality.B.value,
                "content_hash": digest,
            }
        )
    return items


def refresh_news(
    database: Database,
    *,
    symbol: str,
    start: date,
    end: date,
    force: bool = False,
    refresh_hours: int = 12,
) -> ContextRefreshResult:
    result = ContextRefreshResult(symbol=symbol)
    if not force and _fresh_enough(database, "news_items", "symbol", symbol, refresh_hours):
        result.skipped.append("news")
        return result
    try:
        import akshare as ak

        frame = ak.stock_news_em(symbol=Instrument.parse(symbol).code)
        items = _news_items(frame, symbol, start, end)
        if not items:
            raise RuntimeError("未返回带日期和标题的可用新闻")
        database.upsert_news(items)
        for item in items:
            database.add_document(
                symbol=symbol,
                title=item["title"],
                body=item["summary"] or item["title"],
                source_url=item["source_url"],
                published_at=date.fromisoformat(item["published_at"]),
            )
        result.news_count = len(items)
        result.refreshed.append("news")
    except Exception as exc:
        warning = f"东方财富新闻刷新失败: {type(exc).__name__}: {exc}"
        count, latest = _cache_snapshot(
            database,
            table="news_items",
            key_column="symbol",
            key=symbol,
            date_column="published_at",
            end=end,
        )
        if count:
            result.skipped.append("news-cache")
            warning += "；" + _cache_warning(count, latest, end)
            result.news_count = count
        result.warnings.append(warning)
    return result


def _macro_items(
    frame: pd.DataFrame, series: str, source: str, start: date, end: date
) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    date_column = _column(frame, ("Date", "日期", "时间", "date"))
    value_column = _column(frame, ("Close", "收盘", "最新价", "value"))
    if not date_column or not value_column:
        return []
    observations: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        parsed = pd.to_datetime(row.get(date_column), errors="coerce")
        try:
            value = float(row.get(value_column))
        except (TypeError, ValueError):
            continue
        if pd.isna(parsed) or not pd.notna(value):
            continue
        observed = parsed.date()
        if start <= observed <= end:
            observations.append(
                {
                    "series": series,
                    "observation_date": observed.isoformat(),
                    "value": value,
                    "unit": "percent" if series == "US10Y" else "level",
                    "source": source,
                    "quality": DataQuality.B.value,
                }
            )
    return observations


def refresh_macro(
    database: Database,
    *,
    series: list[str],
    start: date,
    end: date,
    force: bool = False,
    refresh_hours: int = 24,
) -> ContextRefreshResult:
    result = ContextRefreshResult()
    for name in series:
        normalized = str(name).upper()
        if not force and _fresh_enough(
            database, "macro_observations", "series", normalized, refresh_hours
        ):
            result.skipped.append(normalized)
            continue
        try:
            if normalized == "SHIBOR":
                import akshare as ak

                frame = ak.macro_china_shibor_all()
                frame = frame.rename(columns={"日期": "Date", "1M-定价": "Close"})
                observations = _macro_items(frame, normalized, "akshare-jin10", start, end)
                if not observations:
                    raise RuntimeError("未返回可用 SHIBOR 序列")
                database.upsert_macro_observations(observations)
                result.macro_count += len(observations)
                result.refreshed.append(normalized)
                continue
            ticker = MACRO_TICKERS.get(normalized)
            if not ticker:
                raise ValueError(f"未配置宏观序列: {normalized}")
            import yfinance as yf

            frame = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame = frame.reset_index()
            observations = _macro_items(frame, normalized, "yfinance", start, end)
            if not observations:
                raise RuntimeError("未返回可用序列")
            database.upsert_macro_observations(observations)
            result.macro_count += len(observations)
            result.refreshed.append(normalized)
        except Exception as exc:
            warning = f"{normalized} 刷新失败: {type(exc).__name__}: {exc}"
            count, latest = _cache_snapshot(
                database,
                table="macro_observations",
                key_column="series",
                key=normalized,
                date_column="observation_date",
                end=end,
            )
            if count:
                result.skipped.append(f"{normalized}-cache")
                warning += "；" + _cache_warning(count, latest, end)
            result.warnings.append(warning)
    return result


def refresh_context(
    database: Database,
    config: AppConfig,
    *,
    symbol: str,
    as_of: date,
    start: date,
    force: bool = False,
) -> ContextRefreshResult:
    """Refresh public news and macro context subject to configured TTLs."""
    automation = config.section("automation")
    macro_config = config.section("macro")
    result = refresh_news(
        database,
        symbol=symbol,
        start=max(start, as_of - timedelta(days=365)),
        end=as_of,
        force=force,
        refresh_hours=int(automation.get("news_refresh_hours", 12)),
    )
    if bool(macro_config.get("enabled", True)):
        macro_result = refresh_macro(
            database,
            series=[str(item) for item in macro_config.get("series", list(MACRO_TICKERS))],
            start=max(start, as_of - timedelta(days=365 * 3)),
            end=as_of,
            force=force,
            refresh_hours=int(automation.get("macro_refresh_hours", 24)),
        )
        result.macro_count = macro_result.macro_count
        result.refreshed.extend(macro_result.refreshed)
        result.skipped.extend(macro_result.skipped)
        result.warnings.extend(macro_result.warnings)
    return result


def macro_snapshot(database: Database, *, as_of: date) -> dict[str, float]:
    """Return the latest point-in-time macro values for deterministic scoring."""
    frame = database.load_macro_observations(as_of=as_of)
    if frame.empty:
        return {}
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame = frame.sort_values("observation_date").drop_duplicates("series", keep="last")
    return {str(row["series"]): float(row["value"]) for _, row in frame.iterrows()}
