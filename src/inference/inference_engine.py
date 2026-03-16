from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from loguru import logger

from src.core.base_llm import BaseLLM
from src.database.models import ChatTurn
from src.database.mongo_client import MongoService
from src.inference.intent_router import IntentRouter
from src.inference.query_reflector import QueryReflector
from src.inference.response_parser import ResponseParser
from src.prompts.chain import PromptChain
from src.prompts.templates import CHITCHAT_SYSTEM_PROMPT, OUT_OF_SCOPE_MESSAGE
from src.rag.reranker import VietnameseReranker
from src.rag.retriever import HybridRetriever, RetrievedChunk


@dataclass(slots=True)
class InferenceOutput:
    intent: str
    answer: str
    rewritten_query: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PreparedRequest:
    intent: str
    rewritten_query: str | None = None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    error: str | None = None


class InferenceEngine:
    def __init__(
        self,
        llm: BaseLLM,
        retriever: HybridRetriever,
        reranker: VietnameseReranker,
        mongo_service: MongoService | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.router = IntentRouter()
        self.reflector = QueryReflector(llm=llm, use_llm_reflect=None)
        self.prompt_chain = PromptChain()
        self.response_parser = ResponseParser()
        self.mongo_service = mongo_service

    def ask(self, query: str, session_id: str = "default") -> InferenceOutput:
        prepared = self._prepare_request(query)
        if prepared.intent == "out_of_scope":
            output = InferenceOutput(intent=prepared.intent, answer=OUT_OF_SCOPE_MESSAGE)
            self._persist_turn(session_id=session_id, query=query, output=output)
            return output

        if prepared.intent == "chitchat":
            answer = self._safe_generate(
                prompt=query,
                system_prompt=CHITCHAT_SYSTEM_PROMPT,
                fallback_text="Chao ban! Minh san sang ho tro cac cau hoi ve phap luat dan su.",
            )
            output = InferenceOutput(intent=prepared.intent, answer=answer)
            self._persist_turn(session_id=session_id, query=query, output=output)
            return output

        if prepared.error:
            output = InferenceOutput(
                intent=prepared.intent,
                answer=prepared.error,
                rewritten_query=prepared.rewritten_query,
                citations=[],
            )
            self._persist_turn(session_id=session_id, query=query, output=output)
            return output

        rewritten_query = prepared.rewritten_query or query
        answer = self._answer_with_context(rewritten_query, prepared.chunks)
        citations = self.response_parser.make_citations(prepared.chunks)
        output = InferenceOutput(
            intent=prepared.intent,
            answer=answer,
            rewritten_query=rewritten_query,
            citations=citations,
        )
        self._persist_turn(session_id=session_id, query=query, output=output)
        return output

    def _answer_with_context(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "Toi chua tim thay can cu phu hop trong kho du lieu hien tai."
        prompt = self.prompt_chain.build_user_prompt(query=query, chunks=chunks)
        answer = self._safe_generate(
            prompt=prompt,
            system_prompt=self.prompt_chain.system_prompt,
            fallback_text=(
                "He thong LLM dang tam thoi khong phan hoi. "
                "Ban hay kiem tra Ollama (serve + model) va thu lai."
            ),
        )
        return self._finalize_legal_answer(answer, chunks)

    def _safe_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        fallback_text: str = "Xin loi, he thong tam thoi gap loi khi tao cau tra loi.",
    ) -> str:
        try:
            return self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.1)
        except Exception as exc:
            logger.error("LLM generation failed: {}", exc)
            err = str(exc).lower()
            if "requires more system memory" in err or "out of memory" in err:
                return (
                    "Ollama dang thieu RAM cho model hien tai. "
                    "Hay doi sang model nhe hon (vi du `qwen2.5:3b`) roi thu lai."
                )
            return fallback_text

    def ask_stream(self, query: str, session_id: str = "default") -> Iterable[dict[str, Any]]:
        prepared = self._prepare_request(query)
        if prepared.intent == "out_of_scope":
            yield {"type": "answer", "content": OUT_OF_SCOPE_MESSAGE}
            return

        if prepared.intent == "chitchat":
            answer = ""
            for chunk in self.llm.generate_stream(prompt=query, system_prompt=CHITCHAT_SYSTEM_PROMPT, temperature=0.1):
                answer += chunk
                yield {"type": "answer_chunk", "content": chunk}
            return

        if prepared.error:
            yield {"type": "answer", "content": prepared.error}
            return

        rewritten_query = prepared.rewritten_query or query
        citations = self.response_parser.make_citations(prepared.chunks)
        yield {"type": "citations", "content": citations}

        prompt = self.prompt_chain.build_user_prompt(query=rewritten_query, chunks=prepared.chunks)
        full_answer = ""
        try:
            for chunk in self.llm.generate_stream(
                prompt=prompt,
                system_prompt=self.prompt_chain.system_prompt,
                temperature=0.1,
            ):
                full_answer += chunk
                yield {"type": "answer_chunk", "content": chunk}
        except Exception as exc:
            logger.error("LLM streaming failed: {}", exc)
            fallback_answer = (
                self._finalize_legal_answer(full_answer, prepared.chunks)
                if full_answer
                else (
                    "He thong LLM dang tam thoi khong phan hoi. "
                    "Ban hay kiem tra Ollama (serve + model) va thu lai."
                )
            )
            yield {"type": "answer", "content": fallback_answer}
            self._persist_turn(
                session_id=session_id,
                query=query,
                output=InferenceOutput(
                    intent=prepared.intent,
                    answer=fallback_answer,
                    rewritten_query=rewritten_query,
                    citations=citations,
                ),
            )
            return

        final_answer = self._finalize_legal_answer(full_answer, prepared.chunks)
        # Emit finalized answer so UI can override raw streamed text with enforced format.
        yield {"type": "answer", "content": final_answer}

        # Persist at the end
        self._persist_turn(
            session_id=session_id, 
            query=query, 
            output=InferenceOutput(
                intent=prepared.intent, 
                answer=final_answer,
                rewritten_query=rewritten_query,
                citations=citations
            )
        )

    def _build_retrieval_hints(
        self,
        query: str,
    ) -> tuple[str, list[str], str | None, dict[str, Any] | None]:
        rewritten_query = self.reflector.rewrite(query)
        expanded_queries = self.reflector.expand_queries(rewritten_query)
        hyde_answer = self.reflector.build_hyde(rewritten_query)
        metadata_filter = self.reflector.extract_metadata_filter(rewritten_query)
        return rewritten_query, expanded_queries, hyde_answer, metadata_filter

    def _prepare_request(self, query: str) -> PreparedRequest:
        intent_result = self.router.route(query)
        intent = intent_result.intent
        if intent != "legal_query":
            return PreparedRequest(intent=intent)

        rewritten_query, expanded_queries, hyde_answer, metadata_filter = self._build_retrieval_hints(query)
        try:
            retrieved = self._search_retriever(
                rewritten_query,
                expanded_queries=expanded_queries,
                hyde_answer=hyde_answer,
                metadata_filter=metadata_filter,
            )
            reranked = self.reranker.rerank(rewritten_query, retrieved)
            return PreparedRequest(
                intent=intent,
                rewritten_query=rewritten_query,
                chunks=reranked,
            )
        except Exception as exc:
            logger.error("Retrieval pipeline failed: {}", exc)
            return PreparedRequest(
                intent=intent,
                rewritten_query=rewritten_query,
                error=(
                    "He thong truy xuat du lieu dang gap loi. "
                    "Ban hay kiem tra Ollama/Vector DB roi thu lai."
                ),
            )

    def _search_retriever(
        self,
        query: str,
        expanded_queries: list[str],
        hyde_answer: str | None,
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        try:
            return self.retriever.search(
                query,
                extra_queries=expanded_queries,
                hyde_answer=hyde_answer,
                metadata_filter=metadata_filter,
            )
        except TypeError:
            # Keep compatibility with legacy retriever signature `search(query)`.
            return self.retriever.search(query)

    def _finalize_legal_answer(self, raw_answer: str, chunks: list[RetrievedChunk]) -> str:
        cleaned = self.response_parser.clean_answer(raw_answer)
        return self.response_parser.enforce_legal_opening(cleaned, chunks)

    def _persist_turn(self, session_id: str, query: str, output: InferenceOutput) -> None:
        if self.mongo_service is None:
            return
        try:
            self.mongo_service.save_chat_turn(
                ChatTurn(
                    session_id=session_id,
                    user_query=query,
                    answer=output.answer,
                    citations=output.citations,
                )
            )
        except Exception as exc:
            logger.warning("Skip save chat history due to DB error: {}", exc)
