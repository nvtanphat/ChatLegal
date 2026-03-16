import chromadb
from pathlib import Path

def inspect_vector_db():
    persist_directory = "data/vectordb/chroma"
    collection_name = "legal_chunks"
    
    if not Path(persist_directory).exists():
        return f"Error: Directory {persist_directory} does not exist."
        
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        # List all collections
        collections = client.list_collections()
        col_names = [c.name for c in collections]
        
        if collection_name not in col_names:
            return f"Error: Collection '{collection_name}' not found. Available collections: {col_names}"
            
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        
        # Peek at the last few metadata to see if they match Decree 102
        peek = collection.peek(limit=5)
        
        return {
            "status": "OK",
            "collection": collection_name,
            "total_records_in_db": count,
            "sample_metadata": [m for m in peek['metadatas']] if peek['metadatas'] else "No metadata"
        }
    except Exception as e:
        return f"Error connecting to Chroma: {str(e)}"

result = inspect_vector_db()
import json
print(json.dumps(result, indent=2, ensure_ascii=False))
