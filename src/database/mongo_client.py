from __future__ import annotations

from typing import Any, Iterable

from loguru import logger

from src.database.models import ChatTurn, LegalDocument, QAPair
from src.settings import load_settings

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
except ImportError:  # pragma: no cover
    MongoClient = None  # type: ignore[assignment]
    Collection = Any  # type: ignore[misc,assignment]


class MongoService:
    def __init__(self, mongo_uri: str | None = None, db_name: str | None = None) -> None:
        settings = load_settings()
        storage_cfg = settings.get("storage", {})
        self.mongo_uri = mongo_uri or storage_cfg.get("mongo_uri", "mongodb://localhost:27017")
        self.db_name = db_name or storage_cfg.get("mongo_db", "vn_law_chatbot")
        self._client: MongoClient | None = None  # type: ignore[type-arg]
        self._db = None

    def connect(self) -> None:
        if MongoClient is None:
            raise RuntimeError("pymongo is not installed. Run `uv sync` first.")
        self._client = MongoClient(self.mongo_uri)
        self._db = self._client[self.db_name]
        logger.info("Connected MongoDB: {}", self.db_name)

    @property
    def legal_docs(self) -> Collection:
        self._ensure_connected()
        return self._db["legal_docs"]  # type: ignore[index]

    @property
    def qa_pairs(self) -> Collection:
        self._ensure_connected()
        return self._db["qa_pairs"]  # type: ignore[index]

    @property
    def chat_history(self) -> Collection:
        self._ensure_connected()
        return self._db["chat_history"]  # type: ignore[index]

    def insert_legal_docs(self, docs: Iterable[LegalDocument]) -> int:
        payload = [doc.model_dump() for doc in docs]
        if not payload:
            return 0
        result = self.legal_docs.insert_many(payload, ordered=False)
        return len(result.inserted_ids)

    def insert_qa_pairs(self, pairs: Iterable[QAPair], *, skip_existing: bool = False) -> int:
        payload = [item.model_dump() for item in pairs]
        if not payload:
            return 0

        if skip_existing:
            key_pairs = {
                (item.get("source_url"), item.get("question"))
                for item in payload
                if item.get("question")
            }
            if key_pairs:
                query = [{"source_url": source_url, "question": question} for source_url, question in key_pairs]
                existing_cursor = self.qa_pairs.find(
                    {"$or": query},
                    {"_id": 0, "source_url": 1, "question": 1},
                )
                existing_keys = {(row.get("source_url"), row.get("question")) for row in existing_cursor}
                payload = [
                    item
                    for item in payload
                    if (item.get("source_url"), item.get("question")) not in existing_keys
                ]
                if not payload:
                    return 0

        result = self.qa_pairs.insert_many(payload, ordered=False)
        return len(result.inserted_ids)

    def save_chat_turn(self, turn: ChatTurn) -> None:
        self.chat_history.insert_one(turn.model_dump())

    def stream_legal_docs(self, query: dict[str, Any] | None = None) -> Iterable[dict[str, Any]]:
        query = query or {}
        cursor = self.legal_docs.find(query, no_cursor_timeout=True)
        try:
            for item in cursor:
                item["_id"] = str(item["_id"])
                yield item
        finally:
            cursor.close()

    def _ensure_connected(self) -> None:
        if self._db is None:
            self.connect()
