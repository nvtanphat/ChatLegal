from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.core.model_factory import create_llm
from src.database.mongo_client import MongoService
from src.inference.inference_engine import InferenceEngine
from src.rag.embedder import EmbeddingService
from src.rag.reranker import VietnameseReranker
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import ChromaVectorStore

app = FastAPI(title="VN Law Chatbot API", version="0.1.0")


class ChatRequest(BaseModel):
    session_id: str = "api"
    query: str


class ChatResponse(BaseModel):
    intent: str
    answer: str
    rewritten_query: str | None = None
    citations: list[dict] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_engine() -> InferenceEngine:
    llm = create_llm()
    embedder = EmbeddingService()
    vector_store = ChromaVectorStore()
    retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)
    reranker = VietnameseReranker()
    mongo = None
    try:
        mongo = MongoService()
        mongo.connect()
    except Exception:
        mongo = None
    return InferenceEngine(llm=llm, retriever=retriever, reranker=reranker, mongo_service=mongo)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    output = get_engine().ask(query=payload.query, session_id=payload.session_id)
    return ChatResponse(
        intent=output.intent,
        answer=output.answer,
        rewritten_query=output.rewritten_query,
        citations=output.citations,
    )
