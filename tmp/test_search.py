from src.rag.embedder import EmbeddingService
from src.rag.vector_store import ChromaVectorStore
import json

def test_search(query_text):
    print(f"Searching for: '{query_text}'")
    
    # 1. Initialize services
    embedder = EmbeddingService()
    vector_store = ChromaVectorStore()
    
    # 2. Connect to DB
    vector_store.connect()
    
    # 3. Embed query
    print("Embedding query...")
    query_vector = embedder.encode_query(query_text)
    
    # 4. Search
    print("Executing search...")
    results = vector_store.query(query_vector, n_results=5)
    
    # 5. Output results
    output = []
    for i, res in enumerate(results, 1):
        meta = res.get("metadata", {})
        output.append({
            "rank": i,
            "score": res.get("score"),
            "title": meta.get("title"),
            "article": meta.get("article"),
            "text_snippet": res.get("text")[:200] + "..."
        })
    
    print(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_search("Điều 124")
