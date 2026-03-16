from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LegalDocument(BaseModel):
    source: str = "vbpl.vn"
    source_id: str
    title: str
    doc_code: str | None = None
    issued_date: str | None = None
    effective_date: str | None = None
    status: str | None = None
    raw_html: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class LegalChunk(BaseModel):
    chunk_id: str
    source_id: str
    title: str
    article: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QAPair(BaseModel):
    question: str
    answer: str
    source_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ChatTurn(BaseModel):
    session_id: str
    user_query: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

