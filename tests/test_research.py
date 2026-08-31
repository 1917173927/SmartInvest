from __future__ import annotations

from datetime import date

import pytest

from stock_analysis.data import AppConfig, Database
from stock_analysis.research import (
    EventFactor,
    EventType,
    OpenAICompatibleClient,
    _parse_json_object,
    _validate_event_against_evidence,
)


def test_embedding_service_can_be_separate_from_chat(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "chat-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("STOCK_ANALYSIS_MODEL", "deepseek-chat")
    monkeypatch.setenv("STOCK_ANALYSIS_EMBEDDING_MODEL", "bge-m3")
    monkeypatch.setenv("STOCK_ANALYSIS_EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("STOCK_ANALYSIS_EMBEDDING_API_KEY", "embed-key")
    client = OpenAICompatibleClient(AppConfig(tmp_path, {}))
    assert client.chat_available
    assert client.embedding_available
    assert client.base_url == "https://api.deepseek.com"
    assert client.embedding_base_url == "http://localhost:11434/v1"


def test_gemini_embedding_configuration_is_supported(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    client = OpenAICompatibleClient(AppConfig(tmp_path, {}))
    assert client.embedding_available
    assert client.embedding_model == "gemini-embedding-001"


def test_json_parser_accepts_fenced_payload() -> None:
    assert _parse_json_object('```json\n{"summary":"ok","events":[]}\n```')["summary"] == "ok"


def test_llm_event_cannot_predate_its_evidence(tmp_path) -> None:
    database = Database(tmp_path / "analysis.sqlite3")
    evidence_id = database.add_document(
        symbol="CN:601318",
        title="一季度报告",
        body="公司披露一季度经营情况。",
        source_url="https://example.test/report",
        published_at=date(2026, 4, 30),
    )
    event = EventFactor(
        event_type=EventType.EARNINGS,
        direction=1,
        strength=0.5,
        confidence=0.8,
        effective_from=date(2026, 4, 1),
        expires_at=date(2026, 6, 30),
        evidence_id=evidence_id,
        rationale="盈利改善",
    )
    with pytest.raises(ValueError, match="未来信息穿越"):
        _validate_event_against_evidence(
            event,
            symbol="CN:601318",
            as_of=date(2026, 5, 1),
            database=database,
        )
    database.close()


def test_llm_event_requires_existing_evidence(tmp_path) -> None:
    database = Database(tmp_path / "analysis.sqlite3")
    event = EventFactor(
        event_type=EventType.POLICY,
        direction=1,
        strength=0.5,
        confidence=0.5,
        effective_from=date(2026, 1, 1),
        expires_at=date(2026, 2, 1),
        evidence_id="missing",
        rationale="政策变化",
    )
    with pytest.raises(ValueError, match="不存在"):
        _validate_event_against_evidence(
            event,
            symbol="CN:601318",
            as_of=date(2026, 1, 2),
            database=database,
        )
    database.close()
