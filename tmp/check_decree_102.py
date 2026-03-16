import chromadb
from pathlib import Path

def search_decree_102():
    persist_directory = "data/vectordb/chroma"
    collection_name = "legal_chunks"
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name)
    
    # Query for source_id: 169363 (Decree 102/2024/NĐ-CP)
    results = collection.get(
        where={"source_id": "169363"},
        limit=5
    )
    
    return {
        "count_decree_102": len(results['ids']) if results['ids'] else 0,
        "sample_titles": [m.get('title') for m in results['metadatas']] if results['metadatas'] else []
    }

print(search_decree_102())
