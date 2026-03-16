from src.prompts.chain import PromptChain
from src.rag.retriever import RetrievedChunk


def test_prompt_contains_query_and_context() -> None:
    chain = PromptChain()
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            source_id="91/2015",
            title="BLDS",
            article="Dieu 100",
            text="Noi dung dieu 100",
            dense_score=1.0,
            lexical_score=1.0,
            final_score=1.0,
        )
    ]
    prompt = chain.build_user_prompt(query="Dieu 100 quy dinh gi?", chunks=chunks)
    assert "Dieu 100 quy dinh gi?" in prompt
    assert "Noi dung dieu 100" in prompt
