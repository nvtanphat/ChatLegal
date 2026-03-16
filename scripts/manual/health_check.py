import sys
from loguru import logger
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.settings import load_settings
from src.rag.embedder import EmbeddingService
from src.rag.vector_store import ChromaVectorStore
from src.rag.retriever import HybridRetriever
from src.rag.reranker import VietnameseReranker
from src.core.model_factory import create_llm
from src.inference.inference_engine import InferenceEngine

def test_full_pipeline():
    load_dotenv()
    logger.info("Starting Full Pipeline Health Check...")
    
    try:
        # 1. Load Settings
        settings = load_settings()
        
        # 2. Setup Components
        embedder = EmbeddingService()
        vector_store = ChromaVectorStore(
            persist_dir=os.getenv("CHROMA_DIR", "data/vectordb/chroma"),
            collection_name="legal_chunks"
        )
        vector_store.connect()
        
        # Check if vector store has data
        count = vector_store.count()
        logger.info(f"Vector Store contains {count} chunks.")
        if count == 0:
            logger.warning("Vector Store is empty! Retrieval will not return anything.")
        
        retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)
        reranker = VietnameseReranker(enabled=True)
        
        llm = create_llm()
        
        # 3. Initialize Engine
        engine = InferenceEngine(
            llm=llm,
            retriever=retriever,
            reranker=reranker
        )
        
        # 4. Run a Test Query
        test_query = "Quy định của pháp luật về thừa kế di sản"
        logger.info(f"Running test query: '{test_query}'")
        
        output = engine.ask(test_query)
        
        logger.success("Pipeline check completed!")
        print("\n--- SYSTEM RESPONSE ---")
        print(f"Intent: {output.intent}")
        print(f"Rewritten Query: {output.rewritten_query}")
        print(f"Answer: {output.answer}")
        print("\nCitations:")
        for c in output.citations:
            print(f"- {c['title']} {c.get('article', '')}: {c['text'][:100]}...")
        print("-----------------------\n")
        
    except Exception as e:
        logger.error(f"Pipeline Health Check FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_full_pipeline()
