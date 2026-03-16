from src.rag.indexer import LegalIndexer


class DummyPreprocessor:
    def clean_text(self, text: str) -> str:
        return text.strip()


class DummyChunk:
    def __init__(
        self,
        chunk_id: str,
        source_id: str,
        title: str,
        article: str | None,
        text: str,
        metadata: dict,
    ) -> None:
        self.chunk_id = chunk_id
        self.source_id = source_id
        self.title = title
        self.article = article
        self.text = text
        self.metadata = metadata


class DummyChunker:
    def chunk_document(self, source_id: str, title: str, text: str) -> list[DummyChunk]:
        return [
            DummyChunk(
                chunk_id=f"{source_id}:1",
                source_id=source_id,
                title=title,
                article="Dieu 124",
                text=text,
                metadata={"strategy": "test"},
            )
        ]


class DummyEmbedder:
    def encode_texts(self, texts, show_progress_bar: bool = False):
        return [[0.1, 0.2] for _ in texts]


class DummyVectorStore:
    def __init__(self) -> None:
        self.last_chunks = []
        self.last_embeddings = []

    def upsert(self, chunks, embeddings):
        self.last_chunks = list(chunks)
        self.last_embeddings = list(embeddings)
        return len(self.last_chunks)


def test_indexer_emits_chunks_callback() -> None:
    vector_store = DummyVectorStore()
    indexer = LegalIndexer(
        preprocessor=DummyPreprocessor(),
        chunker=DummyChunker(),
        embedder=DummyEmbedder(),
        vector_store=vector_store,
    )

    captured: list[dict] = []

    def on_chunks_ready(chunks: list[dict]) -> None:
        captured.extend(chunks)

    inserted = indexer.index_documents(
        docs=[{"source_id": "95942", "title": "BLDS", "text": "Dieu 124. Giao dich vo hieu"}],
        on_chunks_ready=on_chunks_ready,
    )

    assert inserted == 1
    assert len(captured) == 1
    assert captured[0]["chunk_id"] == "95942:1"
    assert captured[0]["article"] == "Dieu 124"
    assert len(vector_store.last_embeddings) == 1


def test_indexer_propagates_url_and_nested_articles_metadata() -> None:
    class NestedChunker:
        def chunk_document(self, source_id: str, title: str, text: str) -> list[DummyChunk]:
            return [
                DummyChunk(
                    chunk_id=f"{source_id}:1",
                    source_id=source_id,
                    title=title,
                    article="Dieu 1",
                    text=text,
                    metadata={"strategy": "test"},
                )
            ]

    vector_store = DummyVectorStore()
    indexer = LegalIndexer(
        preprocessor=DummyPreprocessor(),
        chunker=NestedChunker(),
        embedder=DummyEmbedder(),
        vector_store=vector_store,
    )

    doc_text = (
        "Dieu 1. Quy dinh sua doi\n"
        "Noi dung Dieu 1\n"
        "Dieu 2. Noi dung sua doi thu hai\n"
        "Dieu 3. Noi dung sua doi thu ba"
    )
    inserted = indexer.index_documents(
        docs=[
            {
                "source_id": "11716",
                "title": "Luat so huu tri tue sua doi",
                "text": doc_text,
                "metadata": {"url": "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=11716"},
            }
        ]
    )

    assert inserted == 1
    metadata = vector_store.last_chunks[0]["metadata"]
    assert metadata["url"].endswith("ItemID=11716")
    assert metadata["source_url"].endswith("ItemID=11716")
    assert metadata["nested_articles"] == "2,3"
