from src.inference.response_parser import ResponseParser
from src.rag.retriever import RetrievedChunk


def test_enforce_legal_opening_prefixes_answer() -> None:
    parser = ResponseParser()
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_id="91/2015/QH13",
            title="BLDS",
            article="Dieu 124",
            text="Noi dung Dieu 124",
            dense_score=1.0,
            lexical_score=1.0,
            final_score=1.0,
            metadata={"source_kind": "legal"},
        )
    ]

    output = parser.enforce_legal_opening("Giao dich co the bi vo hieu.", chunks)
    assert output.startswith("Theo 91/2015/QH13, Dieu 124:")


def test_enforce_legal_opening_keeps_existing_theo_prefix() -> None:
    parser = ResponseParser()
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_id="91/2015/QH13",
            title="BLDS",
            article="Dieu 124",
            text="Noi dung Dieu 124",
            dense_score=1.0,
            lexical_score=1.0,
            final_score=1.0,
            metadata={"source_kind": "legal"},
        )
    ]

    output = parser.enforce_legal_opening("Theo Dieu 124, giao dich vo hieu khi ...", chunks)
    assert output == "Theo Dieu 124, giao dich vo hieu khi ..."


def test_make_citations_uses_final_score_and_source_kind() -> None:
    parser = ResponseParser()
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_id="91/2015/QH13",
            title="BLDS",
            article="Dieu 124",
            text="Noi dung Dieu 124",
            dense_score=0.4,
            lexical_score=0.2,
            final_score=0.1234,
            rerank_score=0.0001,
            metadata={"source_kind": "legal"},
        )
    ]

    citations = parser.make_citations(chunks)

    assert len(citations) == 1
    assert citations[0]["score"] == 0.1234
    assert citations[0]["source_kind"] == "legal"
