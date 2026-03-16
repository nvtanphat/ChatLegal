from src.inference.inference_engine import InferenceEngine
from src.rag.retriever import RetrievedChunk


class DummyLLM:
    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.1) -> str:
        return "Day la cau tra loi."

    def generate_stream(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.1):
        yield "Day la "
        yield "cau tra loi."

    def healthcheck(self) -> bool:
        return True

    def raw_chat(self, messages, **kwargs):
        return "OK"


class DummyLLMStreamError(DummyLLM):
    def generate_stream(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.1):
        raise TimeoutError("stream timeout")


class DummyRetriever:
    def search(self, query: str):
        return [
            RetrievedChunk(
                chunk_id="c1",
                source_id="91/2015",
                title="BLDS",
                article="Dieu 124",
                text="Noi dung Dieu 124",
                dense_score=1.0,
                lexical_score=1.0,
                final_score=1.0,
            )
        ]


class DummyReranker:
    def rerank(self, query: str, candidates, top_k=None):
        return list(candidates)


def test_inference_engine_legal_query() -> None:
    engine = InferenceEngine(
        llm=DummyLLM(),
        retriever=DummyRetriever(),
        reranker=DummyReranker(),
        mongo_service=None,
    )
    output = engine.ask("Dieu 124 BLDS quy dinh gi?")
    assert output.intent == "legal_query"
    assert output.answer
    assert output.answer.startswith("Theo ")
    assert output.citations


def test_inference_engine_out_of_scope() -> None:
    engine = InferenceEngine(
        llm=DummyLLM(),
        retriever=DummyRetriever(),
        reranker=DummyReranker(),
        mongo_service=None,
    )
    output = engine.ask("Thu tuc khoi kien to tung dan su nhu the nao?")
    assert output.intent == "out_of_scope"


def test_ask_stream_handles_llm_stream_error() -> None:
    engine = InferenceEngine(
        llm=DummyLLMStreamError(),
        retriever=DummyRetriever(),
        reranker=DummyReranker(),
        mongo_service=None,
    )

    events = list(engine.ask_stream("Dieu 124 BLDS quy dinh gi?"))

    assert any(item.get("type") == "citations" for item in events)
    answer_events = [item for item in events if item.get("type") == "answer"]
    assert answer_events
    assert "tam thoi khong phan hoi" in answer_events[-1]["content"]


def test_ask_stream_emits_finalized_answer_after_chunks() -> None:
    engine = InferenceEngine(
        llm=DummyLLM(),
        retriever=DummyRetriever(),
        reranker=DummyReranker(),
        mongo_service=None,
    )

    events = list(engine.ask_stream("Dieu 124 BLDS quy dinh gi?"))

    chunk_events = [item for item in events if item.get("type") == "answer_chunk"]
    assert chunk_events
    answer_events = [item for item in events if item.get("type") == "answer"]
    assert answer_events
    assert answer_events[-1]["content"].startswith("Theo ")
