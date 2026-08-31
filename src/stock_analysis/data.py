from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tomllib
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def resolve_home(start: Path | None = None) -> Path:
    configured = os.getenv("STOCK_ANALYSIS_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "stock-analysis.toml").exists():
            return candidate
    return current


def safe_filename_component(value: str) -> str:
    """Keep report names readable while removing path/control characters."""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", str(value)).strip(" .")
    return cleaned or "未命名"


def _load_dotenv(root: Path) -> None:
    """Load a minimal project .env without adding a runtime dependency."""
    path = root / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class AppConfig:
    home: Path
    values: dict[str, Any]

    @classmethod
    def load(cls, home: Path | None = None) -> AppConfig:
        root = resolve_home(home)
        _load_dotenv(root)
        path = root / "stock-analysis.toml"
        values: dict[str, Any] = {}
        if path.exists():
            with path.open("rb") as handle:
                values = tomllib.load(handle)
        return cls(root, values)

    @property
    def cache_dir(self) -> Path:
        path = self.home / ".stock-analysis"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def db_path(self) -> Path:
        return self.cache_dir / "analysis.sqlite3"

    @property
    def reports_dir(self) -> Path:
        path = self.home / "06-自动分析"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name, {})
        return value if isinstance(value, dict) else {}

    def asset(self, symbol: str) -> dict[str, Any]:
        assets = self.section("assets")
        value = assets.get(symbol, {})
        return value if isinstance(value, dict) else {}

    def symbol_for_name(self, name: str) -> str | None:
        normalized = name.replace(" ", "").lower()
        for symbol, profile in self.section("assets").items():
            if str(profile.get("name", "")).replace(" ", "").lower() == normalized:
                return symbol
        return None


class Market(StrEnum):
    CN = "CN"
    HK = "HK"
    US = "US"
    CNFUND = "CNFUND"


@dataclass(frozen=True)
class Instrument:
    market: Market
    code: str

    @property
    def canonical(self) -> str:
        return f"{self.market.value}:{self.code}"

    @property
    def currency(self) -> str:
        return {
            Market.CN: "CNY",
            Market.CNFUND: "CNY",
            Market.HK: "HKD",
            Market.US: "USD",
        }[self.market]

    @property
    def yahoo_symbol(self) -> str:
        if self.market is Market.HK:
            return f"{int(self.code):04d}.HK"
        if self.market is Market.CN:
            suffix = "SS" if self.code.startswith(("5", "6", "9")) else "SZ"
            return f"{self.code}.{suffix}"
        if self.market is Market.US:
            return self.code
        raise ValueError("开放式基金没有可靠的 Yahoo Finance 代码")

    @classmethod
    def parse(cls, raw: str) -> Instrument:
        value = raw.strip().upper().replace(" ", "")
        if not value:
            raise ValueError("证券代码不能为空")
        if ":" in value:
            market_raw, code = value.split(":", 1)
            try:
                market = Market(market_raw)
            except ValueError as exc:
                raise ValueError(f"不支持的市场: {market_raw}") from exc
            return cls(market, cls._normalize_code(market, code))
        if value.startswith(("SH.", "SZ.")):
            return cls(Market.CN, value.split(".", 1)[1])
        if value.endswith(".HK"):
            return cls(Market.HK, f"{int(value[:-3]):05d}")
        if value.endswith((".SS", ".SZ")):
            return cls(Market.CN, value.split(".", 1)[0])
        if value.isdigit() and len(value) == 6:
            return cls(Market.CN, value)
        if value.isdigit() and len(value) in {4, 5}:
            return cls(Market.HK, f"{int(value):05d}")
        if all(character.isalnum() or character in {".", "-"} for character in value):
            return cls(Market.US, value)
        raise ValueError(f"无法识别证券代码: {raw}")

    @staticmethod
    def _normalize_code(market: Market, code: str) -> str:
        code = code.strip().upper()
        if market in {Market.CN, Market.CNFUND}:
            if not (code.isdigit() and len(code) == 6):
                raise ValueError(f"{market.value} 代码应为 6 位数字")
        elif market is Market.HK:
            if not code.isdigit() or len(code) > 5:
                raise ValueError("港股代码应为数字")
            code = f"{int(code):05d}"
        return code


class DataQuality(StrEnum):
    A = "A"
    B = "B"
    C = "C"


class Bar(BaseModel):
    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    currency: str
    source: str
    fetched_at: datetime = Field(default_factory=utc_now)
    quality: DataQuality = DataQuality.B

    @field_validator("open", "high", "low", "close")
    @classmethod
    def positive_prices(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("价格必须是正数")
        return float(value)

    @field_validator("volume")
    @classmethod
    def nonnegative_volume(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            return 0.0
        return float(value)


class CorporateAction(BaseModel):
    symbol: str
    action_date: date
    dividend: float = 0.0
    split_ratio: float = 1.0
    source: str
    fetched_at: datetime = Field(default_factory=utc_now)


class FundamentalRecord(BaseModel):
    symbol: str
    metric: str
    value: float
    unit: str = "ratio"
    as_of: date
    period_end: date | None = None
    source: str
    source_url: str | None = None
    quality: DataQuality = DataQuality.B
    fetched_at: datetime = Field(default_factory=utc_now)


class SyncResult(BaseModel):
    symbol: str
    provider: str | None = None
    bars: int = 0
    actions: int = 0
    fundamentals: int = 0
    latest_date: date | None = None
    quality: DataQuality = DataQuality.C
    warnings: list[str] = Field(default_factory=list)


class MarketDataProvider(Protocol):
    name: str

    def supports(self, instrument: Instrument) -> bool: ...

    def fetch_bars(self, instrument: Instrument, start: date, end: date) -> list[Bar]: ...

    def fetch_actions(
        self, instrument: Instrument, start: date, end: date
    ) -> list[CorporateAction]: ...

    def fetch_fundamentals(
        self, instrument: Instrument, as_of: date
    ) -> list[FundamentalRecord]: ...


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS bars (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                currency TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                quality TEXT NOT NULL,
                PRIMARY KEY (symbol, trade_date, source)
            );
            CREATE INDEX IF NOT EXISTS bars_symbol_date ON bars(symbol, trade_date);

            CREATE TABLE IF NOT EXISTS corporate_actions (
                symbol TEXT NOT NULL,
                action_date TEXT NOT NULL,
                dividend REAL NOT NULL DEFAULT 0,
                split_ratio REAL NOT NULL DEFAULT 1,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (symbol, action_date, source)
            );

            CREATE TABLE IF NOT EXISTS fundamentals (
                symbol TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                as_of TEXT NOT NULL,
                period_end TEXT,
                source TEXT NOT NULL,
                source_url TEXT,
                quality TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (symbol, metric, as_of, source)
            );
            CREATE INDEX IF NOT EXISTS fundamentals_lookup
                ON fundamentals(symbol, metric, as_of);

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                source_url TEXT NOT NULL,
                published_at TEXT NOT NULL,
                effective_from TEXT NOT NULL,
                expires_at TEXT,
                retrieved_at TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS document_embeddings (
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (document_id, model)
            );

            CREATE TABLE IF NOT EXISTS event_factors (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                event_type TEXT NOT NULL,
                direction INTEGER NOT NULL,
                strength REAL NOT NULL,
                confidence REAL NOT NULL,
                effective_from TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                evidence_id TEXT NOT NULL REFERENCES documents(id),
                rationale TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS event_factor_lookup
                ON event_factors(symbol, effective_from, expires_at);

            CREATE TABLE IF NOT EXISTS forecast_receipts (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                created_at TEXT NOT NULL,
                as_of TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                due_date TEXT NOT NULL,
                model_status TEXT NOT NULL,
                forecast_json TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                realized_return REAL,
                evaluation_json TEXT,
                evaluated_at TEXT
            );
            CREATE INDEX IF NOT EXISTS receipts_lookup
                ON forecast_receipts(symbol, status, due_date);

            CREATE TABLE IF NOT EXISTS sync_runs (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                provider TEXT,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                status TEXT NOT NULL,
                bars INTEGER NOT NULL DEFAULT 0,
                actions INTEGER NOT NULL DEFAULT 0,
                fundamentals INTEGER NOT NULL DEFAULT 0,
                latest_date TEXT,
                quality TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS sync_runs_lookup
                ON sync_runs(symbol, end_date, fetched_at);

            CREATE TABLE IF NOT EXISTS macro_observations (
                series TEXT NOT NULL,
                observation_date TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                quality TEXT NOT NULL,
                PRIMARY KEY (series, observation_date, source)
            );
            CREATE INDEX IF NOT EXISTS macro_lookup
                ON macro_observations(series, observation_date);

            CREATE TABLE IF NOT EXISTS news_items (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL,
                published_at TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                quality TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS news_lookup
                ON news_items(symbol, published_at);

            CREATE TABLE IF NOT EXISTS automation_runs (
                id TEXT PRIMARY KEY,
                as_of TEXT NOT NULL,
                symbols_json TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                summary_json TEXT
            );
            CREATE INDEX IF NOT EXISTS automation_runs_lookup
                ON automation_runs(as_of, started_at);

            CREATE TABLE IF NOT EXISTS automation_tasks (
                run_id TEXT NOT NULL REFERENCES automation_runs(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                task TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (run_id, sequence)
            );
            """
        )
        # Some custom Python distributions omit FTS5; LIKE search remains available.
        with suppress(sqlite3.OperationalError):
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts "
                "USING fts5(document_id UNINDEXED, symbol, title, body, tokenize='unicode61')"
            )
        self.connection.commit()

    def upsert_bars(self, bars: Sequence[Bar]) -> None:
        self.connection.executemany(
            """
            INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, trade_date, source) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume,
                currency=excluded.currency, fetched_at=excluded.fetched_at,
                quality=excluded.quality
            """,
            [
                (
                    item.symbol,
                    item.trade_date.isoformat(),
                    item.open,
                    item.high,
                    item.low,
                    item.close,
                    item.volume,
                    item.currency,
                    item.source,
                    item.fetched_at.isoformat(),
                    item.quality.value,
                )
                for item in bars
            ],
        )
        self.connection.commit()

    def upsert_actions(self, actions: Sequence[CorporateAction]) -> None:
        self.connection.executemany(
            """
            INSERT INTO corporate_actions VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, action_date, source) DO UPDATE SET
                dividend=excluded.dividend, split_ratio=excluded.split_ratio,
                fetched_at=excluded.fetched_at
            """,
            [
                (
                    item.symbol,
                    item.action_date.isoformat(),
                    item.dividend,
                    item.split_ratio,
                    item.source,
                    item.fetched_at.isoformat(),
                )
                for item in actions
            ],
        )
        self.connection.commit()

    def upsert_fundamentals(self, records: Sequence[FundamentalRecord]) -> None:
        self.connection.executemany(
            """
            INSERT INTO fundamentals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, metric, as_of, source) DO UPDATE SET
                value=excluded.value, unit=excluded.unit,
                period_end=excluded.period_end, source_url=excluded.source_url,
                quality=excluded.quality, fetched_at=excluded.fetched_at
            """,
            [
                (
                    item.symbol,
                    item.metric,
                    item.value,
                    item.unit,
                    item.as_of.isoformat(),
                    item.period_end.isoformat() if item.period_end else None,
                    item.source,
                    item.source_url,
                    item.quality.value,
                    item.fetched_at.isoformat(),
                )
                for item in records
            ],
        )
        self.connection.commit()

    def load_bars(self, symbol: str, as_of: date | None = None) -> pd.DataFrame:
        params: list[Any] = [symbol]
        where = "symbol = ?"
        if as_of:
            where += " AND trade_date <= ?"
            params.append(as_of.isoformat())
        rows = self.connection.execute(
            f"SELECT * FROM bars WHERE {where} ORDER BY trade_date, fetched_at DESC", params
        ).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "currency",
                    "source",
                    "fetched_at",
                    "quality",
                ]
            )
        frame = pd.DataFrame([dict(row) for row in rows])
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        quality_rank = {"A": 0, "B": 1, "C": 2}
        source_rank = {
            "akshare": 0,
            "akshare-eastmoney": 0,
            "akshare-tencent": 1,
            "akshare-sina": 2,
            "yfinance": 3,
        }
        frame["_quality_rank"] = frame["quality"].map(quality_rank).fillna(9)
        frame["_source_rank"] = frame["source"].map(source_rank).fillna(5)
        frame = frame.sort_values(
            ["trade_date", "_quality_rank", "_source_rank", "fetched_at"],
            ascending=[True, True, True, False],
        )
        frame = frame.drop_duplicates("trade_date", keep="first").drop(
            columns=["_quality_rank", "_source_rank"]
        )
        return frame.sort_values("trade_date").reset_index(drop=True)

    def load_actions(self, symbol: str, as_of: date | None = None) -> pd.DataFrame:
        params: list[Any] = [symbol]
        where = "symbol = ?"
        if as_of:
            where += " AND action_date <= ?"
            params.append(as_of.isoformat())
        rows = self.connection.execute(
            f"SELECT * FROM corporate_actions WHERE {where} ORDER BY action_date", params
        ).fetchall()
        frame = pd.DataFrame([dict(row) for row in rows])
        if frame.empty:
            return frame
        # A single action can be reported by several providers.  Selecting one
        # authoritative row per date avoids multiplying split ratios or adding
        # the same dividend twice in the total-return calculation.
        source_rank = {
            "akshare-corporate-actions": 0,
            "yfinance": 1,
        }
        frame["_source_rank"] = frame["source"].map(source_rank).fillna(5)
        frame = frame.sort_values(
            ["action_date", "_source_rank", "fetched_at"],
            ascending=[True, True, False],
        )
        return frame.drop_duplicates("action_date", keep="first").drop(
            columns="_source_rank"
        ).reset_index(drop=True)

    def latest_fundamentals(self, symbol: str, as_of: date) -> dict[str, FundamentalRecord]:
        rows = self.connection.execute(
            """
            SELECT * FROM fundamentals
            WHERE symbol = ? AND as_of <= ?
            ORDER BY metric, as_of DESC,
                CASE quality WHEN 'A' THEN 0 WHEN 'B' THEN 1 ELSE 2 END,
                fetched_at DESC
            """,
            (symbol, as_of.isoformat()),
        ).fetchall()
        results: dict[str, FundamentalRecord] = {}
        for row in rows:
            metric = str(row["metric"])
            if metric in results:
                continue
            results[metric] = FundamentalRecord(
                symbol=row["symbol"],
                metric=metric,
                value=row["value"],
                unit=row["unit"],
                as_of=date.fromisoformat(row["as_of"]),
                period_end=date.fromisoformat(row["period_end"]) if row["period_end"] else None,
                source=row["source"],
                source_url=row["source_url"],
                quality=DataQuality(row["quality"]),
                fetched_at=datetime.fromisoformat(row["fetched_at"]),
            )
        return results

    def add_document(
        self,
        *,
        symbol: str,
        title: str,
        body: str,
        source_url: str,
        published_at: date,
        effective_from: date | None = None,
        expires_at: date | None = None,
    ) -> str:
        normalized = "\n".join(line.rstrip() for line in body.strip().splitlines())
        digest = hashlib.sha256(
            f"{symbol}\n{source_url}\n{published_at.isoformat()}\n{normalized}".encode()
        ).hexdigest()
        document_id = f"doc-{digest[:16]}"
        self.connection.execute(
            """
            INSERT OR IGNORE INTO documents
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                symbol,
                title,
                normalized,
                source_url,
                published_at.isoformat(),
                (effective_from or published_at).isoformat(),
                expires_at.isoformat() if expires_at else None,
                utc_now().isoformat(),
                digest,
            ),
        )
        with suppress(sqlite3.OperationalError):
            self.connection.execute(
                "INSERT OR IGNORE INTO documents_fts(document_id, symbol, title, body) "
                "VALUES (?, ?, ?, ?)",
                (document_id, symbol, title, normalized),
            )
        self.connection.commit()
        return document_id

    def get_document(self, document_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()

    def search_documents(
        self, symbol: str, query: str, as_of: date, limit: int = 8
    ) -> list[sqlite3.Row]:
        terms = [term for term in re.split(r"\s+", query.strip()) if term and term != "OR"]
        fts_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        try:
            rows = self.connection.execute(
                """
                SELECT d.* FROM documents_fts f
                JOIN documents d ON d.id = f.document_id
                WHERE documents_fts MATCH ? AND d.symbol = ?
                  AND d.published_at <= ? AND d.effective_from <= ?
                  AND (d.expires_at IS NULL OR d.expires_at >= ?)
                ORDER BY bm25(documents_fts), d.published_at DESC LIMIT ?
                """,
                (
                    fts_query or query,
                    symbol,
                    as_of.isoformat(),
                    as_of.isoformat(),
                    as_of.isoformat(),
                    limit,
                ),
            ).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass
        text_clauses = " OR ".join("title LIKE ? OR body LIKE ?" for _ in terms) or "1 = 1"
        patterns = [value for term in terms for value in (f"%{term}%", f"%{term}%")]
        return self.connection.execute(
            f"""
            SELECT * FROM documents
            WHERE symbol = ? AND published_at <= ? AND effective_from <= ?
              AND (expires_at IS NULL OR expires_at >= ?)
              AND ({text_clauses})
            ORDER BY published_at DESC LIMIT ?
            """,
            (
                symbol,
                as_of.isoformat(),
                as_of.isoformat(),
                as_of.isoformat(),
                *patterns,
                limit,
            ),
        ).fetchall()

    def active_events(self, symbol: str, as_of: date) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT e.*, d.title, d.source_url, d.published_at
            FROM event_factors e JOIN documents d ON d.id = e.evidence_id
            WHERE e.symbol = ? AND e.effective_from <= ? AND e.expires_at >= ?
              AND d.published_at <= ?
            ORDER BY e.confidence * e.strength DESC
            """,
            (symbol, as_of.isoformat(), as_of.isoformat(), as_of.isoformat()),
        ).fetchall()

    def save_event(self, payload: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO event_factors
            (id, symbol, event_type, direction, strength, confidence,
             effective_from, expires_at, evidence_id, rationale, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                payload["symbol"],
                payload["event_type"],
                payload["direction"],
                payload["strength"],
                payload["confidence"],
                payload["effective_from"],
                payload["expires_at"],
                payload["evidence_id"],
                payload["rationale"],
                payload.get("created_at", utc_now().isoformat()),
            ),
        )
        self.connection.commit()

    def save_embedding(self, document_id: str, model: str, vector: list[float]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO document_embeddings VALUES (?, ?, ?, ?)",
            (document_id, model, json.dumps(vector), utc_now().isoformat()),
        )
        self.connection.commit()

    def embeddings(self, symbol: str, model: str, as_of: date) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT d.*, e.vector_json FROM documents d
            JOIN document_embeddings e ON e.document_id = d.id
            WHERE d.symbol = ? AND e.model = ? AND d.published_at <= ?
              AND d.effective_from <= ? AND (d.expires_at IS NULL OR d.expires_at >= ?)
            """,
            (symbol, model, as_of.isoformat(), as_of.isoformat(), as_of.isoformat()),
        ).fetchall()

    def save_receipt(self, payload: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO forecast_receipts
            (id, symbol, created_at, as_of, horizon_days, due_date, model_status,
             forecast_json, decision_json, evidence_json, status,
             realized_return, evaluation_json, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                payload["symbol"],
                payload["created_at"],
                payload["as_of"],
                payload["horizon_days"],
                payload["due_date"],
                payload["model_status"],
                json.dumps(payload["forecast"], ensure_ascii=False),
                json.dumps(payload["decision"], ensure_ascii=False),
                json.dumps(payload.get("evidence", []), ensure_ascii=False),
                payload.get("status", "open"),
                payload.get("realized_return"),
                json.dumps(payload["evaluation"], ensure_ascii=False)
                if payload.get("evaluation") is not None
                else None,
                payload.get("evaluated_at"),
            ),
        )
        self.connection.commit()

    def receipts(
        self, *, symbol: str | None = None, status: str | None = None
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.connection.execute(
            f"SELECT * FROM forecast_receipts {where} ORDER BY as_of", params
        ).fetchall()

    def save_sync_run(self, payload: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO sync_runs
            (id, symbol, provider, start_date, end_date, status, bars, actions,
             fundamentals, latest_date, quality, warnings_json, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                payload["symbol"],
                payload.get("provider"),
                payload["start_date"],
                payload["end_date"],
                payload["status"],
                int(payload.get("bars", 0)),
                int(payload.get("actions", 0)),
                int(payload.get("fundamentals", 0)),
                payload.get("latest_date"),
                payload.get("quality", DataQuality.C.value),
                json.dumps(payload.get("warnings", []), ensure_ascii=False),
                payload.get("fetched_at", utc_now().isoformat()),
            ),
        )
        self.connection.commit()

    def latest_sync_run(self, symbol: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM sync_runs WHERE symbol = ? ORDER BY fetched_at DESC LIMIT 1", (symbol,)
        ).fetchone()

    def start_automation_run(self, run_id: str, as_of: date, symbols: Sequence[str]) -> None:
        self.connection.execute(
            """
            INSERT INTO automation_runs
            (id, as_of, symbols_json, status, started_at, finished_at, summary_json)
            VALUES (?, ?, ?, 'running', ?, NULL, NULL)
            """,
            (
                run_id,
                as_of.isoformat(),
                json.dumps(list(symbols), ensure_ascii=False),
                utc_now().isoformat(),
            ),
        )
        self.connection.commit()

    def recover_stale_automation_runs(self, stale_before: datetime) -> int:
        result = self.connection.execute(
            """
            UPDATE automation_runs
            SET status = 'interrupted', finished_at = ?,
                summary_json = COALESCE(summary_json, ?)
            WHERE status = 'running' AND started_at < ?
            """,
            (
                utc_now().isoformat(),
                json.dumps(
                    {"reason": "任务超过运行时限且未正常结束"}, ensure_ascii=False
                ),
                stale_before.isoformat(),
            ),
        )
        self.connection.commit()
        return int(result.rowcount)

    def finish_automation_run(
        self,
        run_id: str,
        *,
        status: str,
        tasks: Sequence[dict[str, str]],
        summary: str,
    ) -> None:
        now = utc_now().isoformat()
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO automation_tasks
            (run_id, sequence, symbol, task, status, reason, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    index,
                    item.get("symbol", "system"),
                    item["task"],
                    item["status"],
                    item.get("reason", ""),
                    now,
                )
                for index, item in enumerate(tasks, start=1)
            ],
        )
        self.connection.execute(
            """
            UPDATE automation_runs
            SET status = ?, finished_at = ?, summary_json = ?
            WHERE id = ?
            """,
            (status, now, summary, run_id),
        )
        self.connection.commit()

    def upsert_macro_observations(self, observations: Sequence[dict[str, Any]]) -> None:
        self.connection.executemany(
            """
            INSERT INTO macro_observations
            (series, observation_date, value, unit, source, fetched_at, quality)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(series, observation_date, source) DO UPDATE SET
                value=excluded.value, unit=excluded.unit,
                fetched_at=excluded.fetched_at, quality=excluded.quality
            """,
            [
                (
                    item["series"],
                    item["observation_date"],
                    float(item["value"]),
                    item.get("unit", "ratio"),
                    item["source"],
                    item.get("fetched_at", utc_now().isoformat()),
                    item.get("quality", DataQuality.B.value),
                )
                for item in observations
            ],
        )
        self.connection.commit()

    def load_macro_observations(
        self, series: str | None = None, as_of: date | None = None
    ) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[Any] = []
        if series:
            clauses.append("series = ?")
            params.append(series)
        if as_of:
            clauses.append("observation_date <= ?")
            params.append(as_of.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM macro_observations {where} ORDER BY observation_date", params
        ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def upsert_news(self, items: Sequence[dict[str, Any]]) -> None:
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO news_items
            (id, symbol, title, summary, source, source_url, published_at,
             fetched_at, quality, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["id"],
                    item["symbol"],
                    item["title"],
                    item.get("summary", ""),
                    item["source"],
                    item["source_url"],
                    item["published_at"],
                    item.get("fetched_at", utc_now().isoformat()),
                    item.get("quality", DataQuality.B.value),
                    item["content_hash"],
                )
                for item in items
            ],
        )
        self.connection.commit()

    def load_news(
        self, symbol: str, as_of: date | None = None, limit: int = 100
    ) -> pd.DataFrame:
        where = "symbol = ?"
        params: list[Any] = [symbol]
        if as_of:
            where += " AND published_at <= ?"
            params.append(as_of.isoformat())
        params.append(limit)
        rows = self.connection.execute(
            f"SELECT * FROM news_items WHERE {where} ORDER BY published_at DESC LIMIT ?", params
        ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])


def _find_column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for name in names:
        if name.strip().lower() in normalized:
            return normalized[name.strip().lower()]
    return None


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _normalize_provider_metric(metric: str, value: float) -> float:
    """Normalize provider fields that alternate between percent and ratio conventions."""
    if metric == "dividend_yield" and abs(value) > 1:
        return value / 100
    if metric == "debt_to_equity" and abs(value) > 5:
        return value / 100
    return value


class AkShareProvider:
    name = "akshare"

    def supports(self, instrument: Instrument) -> bool:
        return instrument.market in {Market.CN, Market.HK, Market.CNFUND}

    @staticmethod
    def available() -> bool:
        return find_spec("akshare") is not None

    def fetch_bars(self, instrument: Instrument, start: date, end: date) -> list[Bar]:
        if not self.available():
            raise RuntimeError("未安装 AKShare；请运行 uv sync --extra data")
        import akshare as ak

        start_text = start.strftime("%Y%m%d")
        end_text = end.strftime("%Y%m%d")
        source_name = self.name
        if instrument.market is Market.CN:
            try:
                frame = ak.stock_zh_a_hist(
                    symbol=instrument.code,
                    period="daily",
                    start_date=start_text,
                    end_date=end_text,
                    adjust="",
                )
            except Exception:
                # Eastmoney is occasionally blocked by regional proxies. AKShare
                # exposes Tencent and Sina adapters with the same raw OHLCV role.
                try:
                    prefix = "sh" if instrument.code.startswith(("5", "6", "9")) else "sz"
                    frame = ak.stock_zh_a_hist_tx(
                        symbol=prefix + instrument.code,
                        start_date=start_text,
                        end_date=end_text,
                        adjust="",
                    )
                    source_name = "akshare-tencent"
                except Exception:
                    try:
                        frame = ak.stock_zh_a_daily(
                            symbol=prefix + instrument.code,
                            start_date=start_text,
                            end_date=end_text,
                        )
                        source_name = "akshare-sina"
                    except Exception as fallback_error:
                        raise RuntimeError("东财、腾讯和新浪行情接口均不可用") from fallback_error
        elif instrument.market is Market.HK:
            frame = ak.stock_hk_hist(
                symbol=instrument.code,
                period="daily",
                start_date=start_text,
                end_date=end_text,
                adjust="",
            )
        elif instrument.market is Market.CNFUND:
            frame = ak.fund_open_fund_info_em(symbol=instrument.code, indicator="单位净值走势")
            date_column = _find_column(frame, ["净值日期", "日期"])
            if date_column:
                dates = pd.to_datetime(frame[date_column]).dt.date
                frame = frame[(dates >= start) & (dates <= end)].copy()
        else:
            raise ValueError("AKShare 适配器暂不处理该市场")
        if frame is None or frame.empty:
            raise RuntimeError("AKShare 未返回行情")
        return self._frame_to_bars(frame, instrument, source_name=source_name)

    def _frame_to_bars(
        self, frame: pd.DataFrame, instrument: Instrument, *, source_name: str | None = None
    ) -> list[Bar]:
        date_column = _find_column(frame, ["日期", "净值日期", "date"])
        close_column = _find_column(frame, ["收盘", "单位净值", "close"])
        if not date_column or not close_column:
            raise RuntimeError(f"AKShare 返回列不兼容: {list(frame.columns)}")
        open_column = _find_column(frame, ["开盘", "open"]) or close_column
        high_column = _find_column(frame, ["最高", "high"]) or close_column
        low_column = _find_column(frame, ["最低", "low"]) or close_column
        volume_column = _find_column(frame, ["成交量", "volume"])
        results: list[Bar] = []
        for _, row in frame.iterrows():
            trade_date = pd.Timestamp(row[date_column]).date()
            close = _number(row[close_column], -1)
            if close <= 0:
                continue
            results.append(
                Bar(
                    symbol=instrument.canonical,
                    trade_date=trade_date,
                    open=_number(row[open_column], close),
                    high=_number(row[high_column], close),
                    low=_number(row[low_column], close),
                    close=close,
                    volume=_number(row[volume_column]) if volume_column else 0.0,
                    currency=instrument.currency,
                    source=source_name or self.name,
                    quality=DataQuality.B,
                )
            )
        if not results:
            raise RuntimeError("AKShare 行情清洗后为空")
        return results

    def fetch_actions(
        self, instrument: Instrument, start: date, end: date
    ) -> list[CorporateAction]:
        if instrument.market is not Market.CN or not self.available():
            return []
        import akshare as ak

        # Eastmoney's implementation history includes the ex-right date and
        # per-10-share distribution terms.  The second endpoint is deliberately
        # retained as a schema-compatible fallback because public data endpoints
        # occasionally fail independently.
        try:
            frame = ak.stock_fhps_detail_em(symbol=instrument.code)
            date_column = _find_column(frame, ["除权除息日", "除权日"])
            bonus_column = _find_column(frame, ["送转股份-送股比例", "送股"])
            transfer_column = _find_column(frame, ["送转股份-转股比例", "转增"])
            dividend_column = _find_column(frame, ["现金分红-现金分红比例", "派息"])
        except Exception:
            frame = ak.stock_history_dividend_detail(symbol=instrument.code, indicator="分红")
            date_column = _find_column(frame, ["除权除息日", "除权日"])
            bonus_column = _find_column(frame, ["送股"])
            transfer_column = _find_column(frame, ["转增"])
            dividend_column = _find_column(frame, ["派息"])
        if frame is None or frame.empty or not date_column:
            return []
        results: list[CorporateAction] = []
        for _, row in frame.iterrows():
            raw_date = pd.to_datetime(row[date_column], errors="coerce")
            if pd.isna(raw_date):
                continue
            action_date = pd.Timestamp(raw_date).date()
            if action_date < start or action_date > end:
                continue
            bonus = _number(row[bonus_column]) if bonus_column else 0.0
            transfer = _number(row[transfer_column]) if transfer_column else 0.0
            dividend_per_ten = _number(row[dividend_column]) if dividend_column else 0.0
            split_ratio = 1.0 + max(0.0, bonus + transfer) / 10.0
            dividend = max(0.0, dividend_per_ten) / 10.0
            if math.isclose(split_ratio, 1.0) and math.isclose(dividend, 0.0):
                continue
            results.append(
                CorporateAction(
                    symbol=instrument.canonical,
                    action_date=action_date,
                    dividend=dividend,
                    split_ratio=split_ratio,
                    source="akshare-corporate-actions",
                )
            )
        return results

    def fetch_fundamentals(self, instrument: Instrument, as_of: date) -> list[FundamentalRecord]:
        if instrument.market is not Market.CN or not self.available():
            return []
        import akshare as ak

        try:
            frame = ak.stock_a_indicator_lg(symbol=instrument.code)
        except Exception:
            return []
        if frame is None or frame.empty:
            return []
        date_column = _find_column(frame, ["trade_date", "日期"])
        if date_column:
            frame = frame[pd.to_datetime(frame[date_column]).dt.date <= as_of]
        if frame.empty:
            return []
        row = frame.iloc[-1]
        record_date = pd.Timestamp(row[date_column]).date() if date_column else as_of
        aliases = {
            "pe": ["pe_ttm", "pe"],
            "pb": ["pb"],
            "ps": ["ps_ttm", "ps"],
            "dividend_yield": ["dv_ttm", "dv_ratio"],
            "market_cap": ["total_mv", "market_cap"],
        }
        records: list[FundamentalRecord] = []
        for metric, names in aliases.items():
            column = _find_column(frame, names)
            if not column:
                continue
            value = _number(row[column], math.nan)
            if not math.isfinite(value):
                continue
            value = _normalize_provider_metric(metric, value)
            records.append(
                FundamentalRecord(
                    symbol=instrument.canonical,
                    metric=metric,
                    value=value,
                    unit="CNY" if metric == "market_cap" else "ratio",
                    as_of=record_date,
                    source=self.name,
                    quality=DataQuality.B,
                )
            )
        return records


class YFinanceProvider:
    name = "yfinance"

    def supports(self, instrument: Instrument) -> bool:
        return instrument.market in {Market.CN, Market.HK, Market.US}

    @staticmethod
    def available() -> bool:
        return find_spec("yfinance") is not None

    def _history(self, instrument: Instrument, start: date, end: date) -> pd.DataFrame:
        if not self.available():
            raise RuntimeError("未安装 yfinance；请运行 uv sync --extra data")
        import yfinance as yf

        ticker = yf.Ticker(instrument.yahoo_symbol)
        return ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            actions=True,
        )

    def fetch_bars(self, instrument: Instrument, start: date, end: date) -> list[Bar]:
        frame = self._history(instrument, start, end)
        if frame is None or frame.empty:
            raise RuntimeError("yfinance 未返回行情")
        results: list[Bar] = []
        for index, row in frame.iterrows():
            close = _number(row.get("Close"), -1)
            if close <= 0:
                continue
            results.append(
                Bar(
                    symbol=instrument.canonical,
                    trade_date=pd.Timestamp(index).date(),
                    open=_number(row.get("Open"), close),
                    high=_number(row.get("High"), close),
                    low=_number(row.get("Low"), close),
                    close=close,
                    volume=_number(row.get("Volume")),
                    currency=instrument.currency,
                    source=self.name,
                    quality=DataQuality.B,
                )
            )
        if not results:
            raise RuntimeError("yfinance 行情清洗后为空")
        return results

    def fetch_actions(
        self, instrument: Instrument, start: date, end: date
    ) -> list[CorporateAction]:
        frame = self._history(instrument, start, end)
        results: list[CorporateAction] = []
        for index, row in frame.iterrows():
            dividend = _number(row.get("Dividends"))
            split = _number(row.get("Stock Splits"), 0.0)
            if dividend == 0 and split == 0:
                continue
            results.append(
                CorporateAction(
                    symbol=instrument.canonical,
                    action_date=pd.Timestamp(index).date(),
                    dividend=dividend,
                    split_ratio=split if split > 0 else 1.0,
                    source=self.name,
                )
            )
        return results

    def fetch_fundamentals(self, instrument: Instrument, as_of: date) -> list[FundamentalRecord]:
        if not self.available():
            return []
        import yfinance as yf

        try:
            info = yf.Ticker(instrument.yahoo_symbol).info
        except Exception:
            return []
        aliases = {
            "pe": "trailingPE",
            "pb": "priceToBook",
            "dividend_yield": "dividendYield",
            "roe": "returnOnEquity",
            "debt_to_equity": "debtToEquity",
            "market_cap": "marketCap",
            "eps": "trailingEps",
            "book_value_per_share": "bookValue",
        }
        records: list[FundamentalRecord] = []
        for metric, key in aliases.items():
            value = _number(info.get(key), math.nan)
            if not math.isfinite(value):
                continue
            value = _normalize_provider_metric(metric, value)
            records.append(
                FundamentalRecord(
                    symbol=instrument.canonical,
                    metric=metric,
                    value=value,
                    unit=instrument.currency if metric == "market_cap" else "ratio",
                    as_of=as_of,
                    source=self.name,
                    quality=DataQuality.B,
                )
            )
        return records


class SecEdgarProvider:
    name = "sec-edgar"
    tickers_url = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT", "")

    def available(self) -> bool:
        return "@" in self.user_agent or " " in self.user_agent.strip()

    def fetch_fundamentals(self, instrument: Instrument, as_of: date) -> list[FundamentalRecord]:
        if instrument.market is not Market.US or not self.available():
            return []
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        with httpx.Client(timeout=30, headers=headers) as client:
            ticker_payload = client.get(self.tickers_url).raise_for_status().json()
            cik: int | None = None
            for item in ticker_payload.values():
                if str(item.get("ticker", "")).upper() == instrument.code.upper():
                    cik = int(item["cik_str"])
                    break
            if cik is None:
                return []
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
            payload = client.get(url).raise_for_status().json()
        facts = payload.get("facts", {}).get("us-gaap", {})
        concepts = {
            "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],
            "net_income": ["NetIncomeLoss"],
            "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
            "equity": ["StockholdersEquity"],
            "assets": ["Assets"],
            "liabilities": ["Liabilities"],
            "shares_outstanding": ["CommonStockSharesOutstanding"],
        }
        records: list[FundamentalRecord] = []
        for metric, tags in concepts.items():
            candidates: list[dict[str, Any]] = []
            unit = "shares" if metric == "shares_outstanding" else "USD"
            for tag in tags:
                units = facts.get(tag, {}).get("units", {})
                for unit_name, items in units.items():
                    for item in items:
                        filed = item.get("filed")
                        if not filed or date.fromisoformat(filed) > as_of:
                            continue
                        if item.get("form") not in {"10-K", "10-Q", "8-K", "20-F", "40-F", "6-K"}:
                            continue
                        candidate = dict(item)
                        candidate["_unit"] = unit_name
                        candidates.append(candidate)
            if not candidates:
                continue
            latest = max(candidates, key=lambda item: (item.get("filed", ""), item.get("end", "")))
            records.append(
                FundamentalRecord(
                    symbol=instrument.canonical,
                    metric=metric,
                    value=float(latest["val"]),
                    unit=str(latest.get("_unit", unit)),
                    as_of=date.fromisoformat(latest["filed"]),
                    period_end=date.fromisoformat(latest["end"]) if latest.get("end") else None,
                    source=self.name,
                    source_url=url,
                    quality=DataQuality.A,
                )
            )
        return records


def provider_order(instrument: Instrument) -> list[MarketDataProvider]:
    akshare = AkShareProvider()
    yahoo = YFinanceProvider()
    if instrument.market in {Market.CN, Market.CNFUND}:
        return [akshare, yahoo] if instrument.market is Market.CN else [akshare]
    if instrument.market is Market.HK:
        return [akshare, yahoo]
    return [yahoo]


def sync_symbol(
    database: Database,
    raw_symbol: str,
    *,
    start: date,
    end: date,
    providers: Sequence[MarketDataProvider] | None = None,
) -> SyncResult:
    instrument = Instrument.parse(raw_symbol)
    result = SyncResult(symbol=instrument.canonical)
    run_id = "sync-" + hashlib.sha256(
        f"{instrument.canonical}:{start}:{end}:{utc_now().isoformat()}".encode()
    ).hexdigest()[:16]
    selected: MarketDataProvider | None = None
    bars: list[Bar] = []
    for provider in providers or provider_order(instrument):
        if not provider.supports(instrument):
            continue
        try:
            bars = provider.fetch_bars(instrument, start, end)
            selected = provider
            break
        except Exception as exc:
            result.warnings.append(
                f"{provider.name} 请求失败（{type(exc).__name__}）；继续尝试备用数据源"
            )
    if selected is None or not bars:
        result.warnings.append("所有行情提供商均失败；保留本地缓存")
        cached = database.load_bars(instrument.canonical, end)
        if not cached.empty:
            result.bars = len(cached)
            result.latest_date = pd.Timestamp(cached.iloc[-1]["trade_date"]).date()
            result.provider = str(cached.iloc[-1]["source"])
            result.quality = DataQuality.C
            database.save_sync_run(
                {
                    "id": run_id,
                    "symbol": result.symbol,
                    "provider": result.provider,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "status": "cached",
                    "bars": result.bars,
                    "latest_date": result.latest_date.isoformat(),
                    "quality": result.quality.value,
                    "warnings": result.warnings,
                }
            )
            return result
        database.save_sync_run(
            {
                "id": run_id,
                "symbol": result.symbol,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "status": "failed",
                "quality": DataQuality.C.value,
                "warnings": result.warnings,
            }
        )
        raise RuntimeError("无法获取行情且本地没有缓存: " + "; ".join(result.warnings))

    database.upsert_bars(bars)
    result.provider = selected.name
    if result.warnings:
        result.warnings.append(f"已使用数据源 {selected.name} 完成同步")
    result.bars = len(bars)
    result.latest_date = max(item.trade_date for item in bars)
    result.quality = min((item.quality for item in bars), key=lambda item: item.value)

    try:
        actions = selected.fetch_actions(instrument, start, end)
        database.upsert_actions(actions)
        result.actions = len(actions)
    except Exception as exc:
        result.warnings.append(f"公司行动未同步: {exc}")

    fundamental_records: list[FundamentalRecord] = []
    try:
        fundamental_records.extend(selected.fetch_fundamentals(instrument, end))
    except Exception as exc:
        result.warnings.append(f"估值数据未同步: {exc}")
    if instrument.market is Market.US:
        try:
            fundamental_records.extend(SecEdgarProvider().fetch_fundamentals(instrument, end))
        except Exception as exc:
            result.warnings.append(f"SEC 财务事实未同步: {exc}")
    database.upsert_fundamentals(fundamental_records)
    result.fundamentals = len(fundamental_records)

    if result.latest_date and (end - result.latest_date).days > 10:
        result.quality = DataQuality.C
        result.warnings.append("最后交易日距分析日超过 10 天，数据降为 C 级")
    database.save_sync_run(
        {
            "id": run_id,
            "symbol": result.symbol,
            "provider": result.provider,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "status": "success",
            "bars": result.bars,
            "actions": result.actions,
            "fundamentals": result.fundamentals,
            "latest_date": result.latest_date.isoformat() if result.latest_date else None,
            "quality": result.quality.value,
            "warnings": result.warnings,
        }
    )
    return result


def backfill_symbol(
    database: Database,
    raw_symbol: str,
    *,
    start: date,
    end: date,
    chunk_years: int = 10,
    providers: Sequence[MarketDataProvider] | None = None,
) -> SyncResult:
    """Fetch a long history in idempotent chunks to reduce provider timeouts."""
    canonical = Instrument.parse(raw_symbol).canonical
    aggregate = SyncResult(symbol=canonical, quality=DataQuality.B)
    chunk_end = end
    while chunk_end >= start:
        chunk_start = max(
            start,
            (
                pd.Timestamp(chunk_end)
                - pd.DateOffset(years=chunk_years)
                + pd.Timedelta(days=1)
            ).date(),
        )
        try:
            result = sync_symbol(
                database,
                canonical,
                start=chunk_start,
                end=chunk_end,
                providers=providers,
            )
        except RuntimeError as exc:
            existing = database.load_bars(canonical)
            if existing.empty:
                raise
            aggregate.warnings.append(
                f"{chunk_start} 至 {chunk_end} 未取得更早记录，停止向前补齐：{exc}"
            )
            break
        latest_run = database.latest_sync_run(canonical)
        if latest_run and latest_run["status"] == "cached":
            break
        aggregate.provider = result.provider or aggregate.provider
        aggregate.bars += result.bars
        aggregate.actions += result.actions
        aggregate.fundamentals += result.fundamentals
        aggregate.warnings.extend(result.warnings)
        if result.latest_date and (
            aggregate.latest_date is None or result.latest_date > aggregate.latest_date
        ):
            aggregate.latest_date = result.latest_date
        if result.quality is DataQuality.C:
            aggregate.quality = DataQuality.C
        chunk_end = chunk_start - timedelta(days=1)
    aggregate.warnings = list(dict.fromkeys(aggregate.warnings))
    return aggregate


def total_return_frame(
    bars: pd.DataFrame, actions: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    if bars.empty:
        return bars.copy(), ["没有行情数据"]
    frame = bars.sort_values("trade_date").copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.set_index("trade_date", drop=False)
    dividends = pd.Series(0.0, index=frame.index)
    splits = pd.Series(1.0, index=frame.index)
    if actions is not None and not actions.empty:
        action_frame = actions.copy()
        action_frame["action_date"] = pd.to_datetime(action_frame["action_date"])
        action_frame = action_frame.groupby("action_date", as_index=True).agg(
            dividend=("dividend", "sum"), split_ratio=("split_ratio", "prod")
        )
        common = frame.index.intersection(action_frame.index)
        dividends.loc[common] = action_frame.loc[common, "dividend"].astype(float)
        split_values = action_frame.loc[common, "split_ratio"].astype(float).replace(0, 1)
        splits.loc[common] = split_values
    frame["action_dividend"] = dividends
    frame["action_split_ratio"] = splits
    previous = frame["close"].shift(1)
    frame["daily_return"] = ((frame["close"] * splits + dividends) / previous) - 1
    frame.loc[frame.index[0], "daily_return"] = 0.0
    frame["return_anomaly_status"] = "normal"
    action_applied = (splits != 1.0) | (dividends != 0.0)
    frame.loc[action_applied, "return_anomaly_status"] = "corporate-action-adjusted"
    frame["total_return_index"] = (1 + frame["daily_return"].fillna(0)).cumprod()
    warnings: list[str] = []
    unexplained = frame["daily_return"].abs() > 0.35
    if unexplained.any():
        frame.loc[unexplained, "return_anomaly_status"] = "unresolved"
        dates = [timestamp.date().isoformat() for timestamp in frame.index[unexplained][:5]]
        warnings.append("发现超过 35% 的单日变动，可能缺少拆股/分红事件: " + ", ".join(dates))
    return frame.reset_index(drop=True), warnings


def next_trading_date(frame: pd.DataFrame, as_of: date, horizon_days: int) -> date:
    dates = [pd.Timestamp(value).date() for value in frame["trade_date"]]
    future = [value for value in dates if value > as_of]
    if len(future) >= horizon_days:
        return future[horizon_days - 1]
    return as_of + timedelta(days=math.ceil(horizon_days * 7 / 5))


def quality_summary(frame: pd.DataFrame, as_of: date) -> tuple[DataQuality, list[str]]:
    if frame.empty:
        return DataQuality.C, ["无行情数据"]
    warnings: list[str] = []
    latest = pd.Timestamp(frame.iloc[-1]["trade_date"]).date()
    if (as_of - latest).days > 10:
        warnings.append("行情数据已超过 10 天未更新")
        return DataQuality.C, warnings
    qualities = set(frame["quality"].astype(str)) if "quality" in frame else {"C"}
    if "C" in qualities:
        return DataQuality.C, warnings
    if len(frame) < 756:
        warnings.append("少于约三年交易日，模型校准能力有限")
        return DataQuality.C, warnings
    return DataQuality.B, warnings


def coverage_warnings(
    frame: pd.DataFrame,
    *,
    as_of: date,
    expected_start: date | None = None,
    stale_days: int = 5,
    gap_days: int = 14,
    gap_lookback_days: int = 365,
) -> list[str]:
    """Return non-destructive coverage diagnostics for automation decisions."""
    if frame.empty or "trade_date" not in frame:
        return ["没有可检查的行情覆盖"]
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna().dt.date.sort_values()
    if dates.empty:
        return ["行情日期无法解析"]
    warnings: list[str] = []
    latest = dates.iloc[-1]
    if (as_of - latest).days > stale_days:
        warnings.append(f"最后交易日距分析日 {(as_of - latest).days} 天")
    recent_start = as_of - timedelta(days=gap_lookback_days)
    recent_dates = dates[dates >= recent_start]
    gaps = recent_dates.diff().dropna().map(lambda value: value.days)
    if not gaps.empty and int(gaps.max()) > gap_days:
        warnings.append(f"近 {gap_lookback_days} 天行情存在超过 {gap_days} 天的日期缺口")
    if expected_start and dates.iloc[0] > expected_start:
        warnings.append(f"历史起点 {dates.iloc[0]} 晚于期望起点 {expected_start}")
    return warnings


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    if a.shape != b.shape or a.size == 0:
        return -1.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else -1.0
