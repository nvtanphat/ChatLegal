from src.rag.retriever import HybridRetriever


class DummyEmbedder:
    def encode_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class DummyVectorStore:
    def query(self, query_embedding: list[float], n_results: int = 8):
        return [
            {
                "chunk_id": "c1",
                "text": "hop dong mua ban nha dat",
                "metadata": {"source_id": "91/2015", "title": "BLDS", "article": "Dieu 500"},
                "score": 0.9,
            },
            {
                "chunk_id": "c2",
                "text": "thua ke theo phap luat",
                "metadata": {"source_id": "91/2015", "title": "BLDS", "article": "Dieu 650"},
                "score": 0.4,
            },
        ]

    def all_chunks(self):
        return [
            {
                "chunk_id": "c1",
                "text": "hop dong mua ban nha dat",
                "metadata": {"source_id": "91/2015", "title": "BLDS", "article": "Dieu 500"},
            },
            {
                "chunk_id": "c2",
                "text": "thua ke theo phap luat",
                "metadata": {"source_id": "91/2015", "title": "BLDS", "article": "Dieu 650"},
            },
        ]


class DenseOnlyVectorStore:
    def query(self, query_embedding: list[float], n_results: int = 8):
        return [
            {
                "chunk_id": "d1",
                "text": "quyen su dung dat la tai san",
                "metadata": {"source_id": "91/2015", "title": "BLDS", "article": "Dieu 105"},
                "score": 0.8,
            }
        ]


class TraceEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_query(self, query: str) -> list[float]:
        self.calls.append(query)
        return [float(len(self.calls))]


class TraceVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def query(self, query_embedding: list[float], n_results: int = 8, where=None):
        self.calls.append({"embedding": list(query_embedding), "where": where})
        step = int(query_embedding[0])
        if where and where.get("article", {}).get("$eq") == "Dieu 999":
            return []
        return [
            {
                "chunk_id": f"q{step}",
                "text": f"chunk from query {step}",
                "metadata": {"source_id": "91/2015", "title": "BLDS", "article": "Dieu 124"},
                "score": 0.9,
            }
        ]

    def all_chunks(self):
        return []


def test_hybrid_retriever_returns_ranked_chunks() -> None:
    retriever = HybridRetriever(vector_store=DummyVectorStore(), embedder=DummyEmbedder())
    results = retriever.search("hop dong mua ban")

    assert results
    assert results[0].chunk_id == "c1"


def test_hybrid_retriever_works_without_chroma_fields() -> None:
    retriever = HybridRetriever(vector_store=DenseOnlyVectorStore(), embedder=DummyEmbedder())
    results = retriever.search("quyen su dung dat")

    assert results
    assert results[0].chunk_id == "d1"


def test_hybrid_retriever_runs_multi_query_dense_search() -> None:
    embedder = TraceEmbedder()
    store = TraceVectorStore()
    retriever = HybridRetriever(vector_store=store, embedder=embedder)

    results = retriever.search("dieu 124", extra_queries=["bo luat dan su dieu 124"])

    assert results
    assert len(embedder.calls) == 2
    assert embedder.calls[0] == "dieu 124"
    assert embedder.calls[1] == "bo luat dan su dieu 124"
    assert len(store.calls) == 2


def test_hybrid_retriever_fallbacks_when_metadata_filter_too_strict() -> None:
    embedder = TraceEmbedder()
    store = TraceVectorStore()
    retriever = HybridRetriever(vector_store=store, embedder=embedder)

    results = retriever.search(
        "dieu 124",
        metadata_filter={"article": {"$eq": "Dieu 999"}},
    )

    assert results
    assert store.calls[0]["where"] == {"article": {"$eq": "Dieu 999"}}
    assert any(call["where"] is None for call in store.calls)
