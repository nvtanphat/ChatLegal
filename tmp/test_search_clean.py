import sys
import logging
import json
from src.rag.embedder import EmbeddingService
from src.rag.vector_store import ChromaVectorStore

# Disable logging to get clean output
logging.getLogger("src").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

def test_search(query_text):
    try:
        embedder = EmbeddingService()
        vector_store = ChromaVectorStore()
        vector_store.connect()
        query_vector = embedder.encode_query(query_text)
        results = vector_store.query(query_vector, n_results=5)
        
        output = []
        for i, res in enumerate(results, 1):
            meta = res.get("metadata", {})
            output.append({
                "rank": i,
                "score": round(res.get("score", 0), 4),
                "title": meta.get("title"),
                "article": meta.get("article"),
                "text": res.get("text", "")
            })
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Điều 124"
    test_search(q)
