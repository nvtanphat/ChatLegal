import sys
from pathlib import Path
from loguru import logger

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag.reranker import VietnameseReranker
from src.rag.retriever import RetrievedChunk

def test_real_reranker():
    logger.info("Checking Reranker with ACTUAL model...")
    logger.warning("Note: This might download the model (~400MB) if not already cached.")
    
    # Simple test case: Query about inheritance
    query = "Quy định về thừa kế theo di chúc"
    
    # 2 Candidates: one relevant, one irrelevant
    candidates = [
        RetrievedChunk(
            chunk_id="1",
            source_id="S1",
            title="Thừa kế di chúc",
            article="624",
            text="Di chúc là sự thể hiện ý chí của cá nhân nhằm chuyển tài sản của mình cho người khác sau khi chết.",
            dense_score=0.5, # Low initial scores to see if reranker raises it
            lexical_score=0.5,
            final_score=0.5
        ),
        RetrievedChunk(
            chunk_id="2",
            source_id="S2",
            title="Kết hôn",
            article="8",
            text="Kết hôn là việc nam và nữ xác lập quan hệ vợ chồng với nhau theo quy định của Luật này.",
            dense_score=0.5,
            lexical_score=0.5,
            final_score=0.5
        )
    ]
    
    try:
        # Initialize with default settings (real model)
        reranker = VietnameseReranker(enabled=True)
        logger.info(f"Reranker initialized with model: {reranker.model_name}")
        
        # This will trigger _ensure_model and model loading
        results = reranker.rerank(query, candidates)
        
        logger.info("--- Rerank Results ---")
        for res in results:
            score_str = f"{res.rerank_score:.4f}" if res.rerank_score is not None else "N/A"
            logger.info(f"ID: {res.chunk_id} | Rerank Score: {score_str} | Text: {res.text[:60]}...")
        
        # Validation
        if len(results) > 0 and results[0].chunk_id == "1" and results[0].rerank_score > 0:
            logger.success("SUCCESS: Reranker correctly identified the relevant document!")
        else:
            logger.error("FAILURE: Reranker did not rank the chunks as expected or score is 0.")
            
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        logger.info("Try running `uv sync` or checking your internet connection if the model failed to download.")
        sys.exit(1)

if __name__ == "__main__":
    test_real_reranker()
