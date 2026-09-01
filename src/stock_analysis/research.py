from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from stock_analysis.data import AppConfig, Database, cosine_similarity, utc_now


class EventType(StrEnum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    POLICY = "policy"
    MANAGEMENT = "management"
    REGULATORY_LEGAL = "regulatory_legal"
    COMMODITY = "commodity"
    CAPITAL_ALLOCATION = "capital_allocation"


class EventFactor(BaseModel):
    event_type: EventType
    direction: int
    strength: float
    confidence: float
    effective_from: date
    expires_at: date
    evidence_id: str
    rationale: str

    @field_validator("direction")
    @classmethod
    def direction_is_discrete(cls, value: int) -> int:
        if value not in {-1, 0, 1}:
            raise ValueError("direction 只能是 -1、0、1")
        return value

    @field_validator("strength", "confidence")
    @classmethod
    def probability_range(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("强度和置信度必须在 0–1")
        return value

    @model_validator(mode="after")
    def dates_are_ordered(self) -> EventFactor:
        if self.expires_at < self.effective_from:
            raise ValueError("事件失效日早于生效日")
        return self


class Evidence(BaseModel):
    id: str
    title: str
    source_url: str
    published_at: date
    excerpt: str
    score: float | None = None


class ResearchResult(BaseModel):
    symbol: str
    as_of: date
    status: str
    summary: str
    evidence: list[Evidence] = Field(default_factory=list)
    events: list[EventFactor] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def event_score(self) -> float:
        if not self.events:
            return 0.0
        raw = sum(item.direction * item.strength * item.confidence for item in self.events) / max(
            len(self.events), 1
        )
        return float(np.clip(raw, -1, 1))


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    marker = text.find("\n---\n", 4)
    if marker < 0:
        return {}, text
    header = text[4:marker]
    values: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line or line.startswith((" ", "\t")):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values, text[marker + 5 :]


def ingest_evidence_file(
    database: Database,
    *,
    symbol: str,
    path: Path,
    as_of: date,
) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.suffix.lower() not in {".md", ".txt"}:
        raise ValueError("v1 只接收 Markdown 或纯文本证据")
    text = resolved.read_text(encoding="utf-8")
    header, body = _frontmatter(text)
    date_value = header.get("date") or header.get("updated")
    if date_value and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_value):
        published_at = date.fromisoformat(date_value)
    else:
        published_at = datetime.fromtimestamp(resolved.stat().st_mtime).date()
    if published_at > as_of:
        raise ValueError(f"证据发布日期 {published_at} 晚于分析日 {as_of}")
    title_match = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
    title = header.get("title") or (title_match.group(1).strip() if title_match else resolved.stem)
    source_url = header.get("source_url") or header.get("url") or f"local:{resolved}"
    return database.add_document(
        symbol=symbol,
        title=title,
        body=body,
        source_url=source_url,
        published_at=published_at,
    )


class OpenAICompatibleClient:
    def __init__(self, config: AppConfig):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("STOCK_ANALYSIS_MODEL", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "").removeprefix(
            "models/"
        )
        self.embedding_model = (
            os.getenv("STOCK_ANALYSIS_EMBEDDING_MODEL") or self.gemini_embedding_model
        )
        self.embedding_base_url = os.getenv(
            "STOCK_ANALYSIS_EMBEDDING_BASE_URL", self.base_url
        ).rstrip("/")
        self.embedding_api_key = os.getenv("STOCK_ANALYSIS_EMBEDDING_API_KEY", self.api_key)
        research_config = config.section("research")
        self.timeout = float(research_config.get("timeout_seconds", 90))
        self.retry_attempts = max(1, int(research_config.get("retry_attempts", 3)))
        self.retry_backoff_seconds = max(
            0.0, float(research_config.get("retry_backoff_seconds", 0.5))
        )

    @property
    def chat_available(self) -> bool:
        return bool(self.api_key and self.model)

    @property
    def embedding_available(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_embedding_model) or bool(
            self.embedding_api_key and self.embedding_model
        )

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Retry transient transport, rate-limit and server failures only."""
        for attempt in range(self.retry_attempts):
            try:
                response = httpx.post(url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status != 429 and status < 500:
                    raise
                if attempt + 1 >= self.retry_attempts:
                    raise
            except httpx.TransportError:
                if attempt + 1 >= self.retry_attempts:
                    raise
            time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RuntimeError("HTTP 重试流程异常结束")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.embedding_available:
            raise RuntimeError("未配置嵌入模型")
        if self.gemini_api_key and self.gemini_embedding_model:
            response = self._post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.gemini_embedding_model}:batchEmbedContents",
                params={"key": self.gemini_api_key},
                json={
                    "requests": [
                        {
                            "model": f"models/{self.gemini_embedding_model}",
                            "content": {"parts": [{"text": text}]},
                        }
                        for text in texts
                    ]
                },
                timeout=self.timeout,
            )
            return [
                [float(value) for value in item["values"]] for item in response.json()["embeddings"]
            ]
        response = self._post(
            f"{self.embedding_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.embedding_api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.embedding_model, "input": texts},
            timeout=self.timeout,
        )
        payload = response.json()
        ordered = sorted(payload["data"], key=lambda item: int(item["index"]))
        return [[float(value) for value in item["embedding"]] for item in ordered]

    def extract(self, *, symbol: str, as_of: date, evidence: list[Evidence]) -> dict[str, Any]:
        if not self.chat_available:
            raise RuntimeError("未配置 OpenAI-compatible 对话模型")
        evidence_text = "\n\n".join(
            f"[{item.id}] published={item.published_at} source={item.source_url}\n"
            f"title={item.title}\n{item.excerpt}"
            for item in evidence
        )
        system = (
            "你是金融事件抽取器，不预测股价。只能使用用户给出的证据。"
            "不得补充证据中没有的财务数字、日期或未来事件。"
            "输出严格 JSON，不要 Markdown。"
        )
        user = f"""
标的：{symbol}
分析截止日：{as_of.isoformat()}

允许的事件类型：earnings, guidance, policy, management, regulatory_legal,
commodity, capital_allocation。

输出对象：
{{
  "summary": "仅基于证据的简短研究摘要，并主动指出反面证据",
  "events": [
    {{
      "event_type": "允许值之一",
      "direction": -1或0或1,
      "strength": 0到1,
      "confidence": 0到1,
      "effective_from": "YYYY-MM-DD，不得早于证据发布日期",
      "expires_at": "YYYY-MM-DD",
      "evidence_id": "必须是方括号中的证据ID",
      "rationale": "可证伪的理由"
    }}
  ]
}}

没有足够证据时 events 返回空数组。以下是唯一可用证据：
{evidence_text}
""".strip()
        response = self._post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self.timeout,
        )
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_json_object(content)


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM 未返回 JSON 对象") from None
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM 响应顶层必须是对象")
    return value


def index_missing_embeddings(
    database: Database, client: OpenAICompatibleClient, symbol: str
) -> int:
    if not client.embedding_available:
        return 0
    rows = database.connection.execute(
        """
        SELECT d.* FROM documents d
        LEFT JOIN document_embeddings e
          ON e.document_id = d.id AND e.model = ?
        WHERE d.symbol = ? AND e.document_id IS NULL
        ORDER BY d.published_at
        """,
        (client.embedding_model, symbol),
    ).fetchall()
    indexed = 0
    for offset in range(0, len(rows), 16):
        batch = rows[offset : offset + 16]
        vectors = client.embed([f"{row['title']}\n{row['body'][:6000]}" for row in batch])
        for row, vector in zip(batch, vectors, strict=True):
            database.save_embedding(row["id"], client.embedding_model, vector)
            indexed += 1
    return indexed


def retrieve_evidence(
    database: Database,
    client: OpenAICompatibleClient,
    *,
    symbol: str,
    as_of: date,
    limit: int = 8,
) -> list[Evidence]:
    query = "业绩 指引 政策 管理层 监管 分红 回购 商品 风险 反证"
    lexical = database.search_documents(symbol, query, as_of, limit=limit)
    scored: dict[str, tuple[Any, float]] = {row["id"]: (row, 0.0) for row in lexical}
    if client.embedding_available:
        index_missing_embeddings(database, client, symbol)
        query_vector = client.embed([query])[0]
        for row in database.embeddings(symbol, client.embedding_model, as_of):
            score = cosine_similarity(query_vector, json.loads(row["vector_json"]))
            existing = scored.get(row["id"])
            if existing is None or score > existing[1]:
                scored[row["id"]] = (row, score)
    ranked = sorted(
        scored.values(),
        key=lambda item: (item[1], item[0]["published_at"]),
        reverse=True,
    )[:limit]
    return [
        Evidence(
            id=row["id"],
            title=row["title"],
            source_url=row["source_url"],
            published_at=date.fromisoformat(row["published_at"]),
            excerpt=str(row["body"])[:2500],
            score=score if score else None,
        )
        for row, score in ranked
    ]


def _validate_event_against_evidence(
    event: EventFactor,
    *,
    symbol: str,
    as_of: date,
    database: Database,
) -> None:
    row = database.get_document(event.evidence_id)
    if row is None:
        raise ValueError("事件引用了不存在的证据")
    if row["symbol"] != symbol:
        raise ValueError("事件引用了其他标的的证据")
    published_at = date.fromisoformat(row["published_at"])
    if published_at > as_of:
        raise ValueError("事件使用了分析日之后发布的证据")
    if event.effective_from < published_at:
        raise ValueError("事件生效日早于证据发布日期，存在未来信息穿越")
    if event.effective_from > as_of:
        raise ValueError("v1 不允许 LLM 创建未来事件因子")


def run_research(
    *,
    database: Database,
    config: AppConfig,
    symbol: str,
    as_of: date,
    evidence_paths: list[Path] | None = None,
    use_llm: bool = True,
) -> ResearchResult:
    warnings: list[str] = []
    for path in evidence_paths or []:
        try:
            ingest_evidence_file(database, symbol=symbol, path=path, as_of=as_of)
        except Exception as exc:
            warnings.append(f"证据未导入 {path}: {exc}")
    client = OpenAICompatibleClient(config)
    try:
        evidence = retrieve_evidence(database, client, symbol=symbol, as_of=as_of)
    except Exception as exc:
        warnings.append(f"向量检索降级为全文检索: {exc}")
        fallback_client = OpenAICompatibleClient(config)
        fallback_client.embedding_model = ""
        fallback_client.gemini_embedding_model = ""
        fallback_client.gemini_api_key = ""
        evidence = retrieve_evidence(database, fallback_client, symbol=symbol, as_of=as_of)
    if not evidence:
        return ResearchResult(
            symbol=symbol,
            as_of=as_of,
            status="unavailable",
            summary="没有带日期和来源的可用证据；LLM 因子未参与判断。",
            warnings=warnings + ["请通过 stock analyze --evidence 导入 Markdown/TXT 证据"],
        )
    if not use_llm or not client.chat_available:
        message = "本次显式跳过 LLM" if not use_llm else "未配置 LLM API 与模型"
        return ResearchResult(
            symbol=symbol,
            as_of=as_of,
            status="evidence-only",
            summary=f"已检索 {len(evidence)} 条证据；{message}，未生成事件因子。",
            evidence=evidence,
            warnings=warnings + [message],
        )
    try:
        payload = client.extract(symbol=symbol, as_of=as_of, evidence=evidence)
    except Exception as exc:
        return ResearchResult(
            symbol=symbol,
            as_of=as_of,
            status="degraded",
            summary="LLM 调用失败，保留证据但不生成事件因子。",
            evidence=evidence,
            warnings=warnings + [str(exc)],
        )
    events: list[EventFactor] = []
    for raw in payload.get("events", []):
        try:
            event = EventFactor.model_validate(raw)
            _validate_event_against_evidence(event, symbol=symbol, as_of=as_of, database=database)
        except (ValidationError, ValueError, TypeError) as exc:
            warnings.append(f"拒绝无效 LLM 事件: {exc}")
            continue
        event_id = (
            "evt-"
            + hashlib.sha256(
                f"{symbol}:{event.evidence_id}:{event.event_type}:{event.effective_from}".encode()
            ).hexdigest()[:16]
        )
        database.save_event(
            {
                "id": event_id,
                "symbol": symbol,
                **event.model_dump(mode="json"),
                "created_at": utc_now().isoformat(),
            }
        )
        events.append(event)
    return ResearchResult(
        symbol=symbol,
        as_of=as_of,
        status="ready" if events else "evidence-only",
        summary=str(payload.get("summary") or "LLM 未提供摘要"),
        evidence=evidence,
        events=events,
        warnings=warnings,
    )


def active_event_rows(database: Database, symbol: str, as_of: date) -> list[dict[str, Any]]:
    return [dict(row) for row in database.active_events(symbol, as_of)]


def manual_event(
    *,
    event_type: EventType,
    direction: int,
    evidence_id: str,
    effective_from: date,
    duration_days: int = 90,
    strength: float = 0.5,
    confidence: float = 0.7,
    rationale: str = "人工确认事件",
) -> EventFactor:
    return EventFactor(
        event_type=event_type,
        direction=direction,
        strength=strength,
        confidence=confidence,
        effective_from=effective_from,
        expires_at=effective_from + timedelta(days=duration_days),
        evidence_id=evidence_id,
        rationale=rationale,
    )
