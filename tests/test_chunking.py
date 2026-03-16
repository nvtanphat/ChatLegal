from src.processing.chunking import LegalChunker


def test_chunk_by_article() -> None:
    text = (
        "Dieu 1. Quy dinh chung\nNoi dung A\n"
        "Dieu 2. Quy dinh rieng\nNoi dung B"
    )
    chunker = LegalChunker(max_chars=2000, overlap_chars=50)
    chunks = chunker.chunk_document(source_id="91-2015", title="BLDS", text=text)

    assert len(chunks) == 2
    assert chunks[0].article == "Dieu 1"
    assert chunks[1].article == "Dieu 2"


def test_window_chunk_fallback() -> None:
    text = "x" * 3000
    chunker = LegalChunker(max_chars=1000, overlap_chars=50)
    chunks = chunker.chunk_document(source_id="doc", title="Doc", text=text)
    assert len(chunks) >= 3


def test_long_article_parts_keep_article_anchor() -> None:
    text = "Dieu 124. " + ("Noi dung rat dai de tach phan. " * 200)
    chunker = LegalChunker(max_chars=400, overlap_chars=80)
    chunks = chunker.chunk_document(source_id="91-2015", title="BLDS", text=text)

    assert len(chunks) > 1
    for item in chunks:
        assert item.article == "Dieu 124"
        assert item.text.strip().lower().startswith("dieu 124")


def test_split_long_chunk_avoids_mid_word_start() -> None:
    text = "Dieu 1. " + ("mot hai ba bon nam sau bay tam chin muoi. " * 120)
    chunker = LegalChunker(max_chars=220, overlap_chars=60)
    chunks = chunker.chunk_document(source_id="doc", title="Doc", text=text)

    # Every split part must start at article anchor instead of truncated fragments.
    assert len(chunks) > 2
    assert all(item.text.lstrip().lower().startswith("dieu 1") for item in chunks)


def test_chunker_deduplicates_repeated_articles_in_same_doc() -> None:
    article_60 = "Dieu 60. Nguyen tac cai tao nha chung cu\nNoi dung A"
    article_169 = "Dieu 169. Mua truoc nha o\nNoi dung B"
    text = f"{article_60}\n{article_169}\n{article_60}\n{article_169}"

    chunker = LegalChunker(max_chars=2000, overlap_chars=100)
    chunks = chunker.chunk_document(source_id="169032", title="Luat Nha o", text=text)

    assert len(chunks) == 2
    assert [item.article for item in chunks] == ["Dieu 60", "Dieu 169"]
