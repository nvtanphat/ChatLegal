import json
import os

def check_vietnamese_encoding(text):
    # Common double-encoding signatures in Vietnamese
    # Examples: Ã  (à), Ã¡ (á), áº¡ (ạ)
    # However, some of these might be valid in specific contexts if not careful.
    # We look for high-frequency patterns of these characters.
    broken_patterns = ["Ã ", "Ã¡", "áº", "Ãª"]
    count = 0
    for p in broken_patterns:
        count += text.count(p)
    return count > 5 # Threshold for suspected encoding issue

def validate_chunks(file_path):
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found."
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        total_chunks = len(data)
        if total_chunks == 0:
            return "Error: File is empty."
            
        required_fields = {"chunk_id", "source_id", "title", "text", "metadata"}
        
        errors = []
        ids = set()
        stats = {
            "short_text": 0,
            "long_text": 0,
            "missing_article": 0,
            "encoding_suspicion": 0,
        }
        unique_strategies = set()
        unique_sources = set()
        
        for i, chunk in enumerate(data):
            # 1. Structure
            missing = required_fields - set(chunk.keys())
            if missing:
                errors.append(f"Chunk {i} missing: {missing}")
            
            # 2. Duplicate ID
            cid = chunk.get("chunk_id")
            if cid in ids:
                errors.append(f"Duplicate ID: {cid}")
            ids.add(cid)
            
            # 3. Metadata
            meta = chunk.get("metadata", {})
            unique_strategies.add(str(meta.get("strategy")))
            unique_sources.add(str(meta.get("source")))
            
            # 4. Article
            if not chunk.get("article"):
                stats["missing_article"] += 1
            
            # 5. Text Length
            text = str(chunk.get("text", ""))
            tlen = len(text)
            if tlen < 50: stats["short_text"] += 1
            if tlen > 5000: stats["long_text"] += 1
            
            # 6. Encoding
            if check_vietnamese_encoding(text):
                stats["encoding_suspicion"] += 1
                
            if len(errors) > 20: break

        report = [
            "--- DEEP QUALITY REPORT ---",
            f"Total Chunks: {total_chunks}",
            f"Unique Strategies: {unique_strategies}",
            f"Unique Sources: {unique_sources}",
            f"Missing Article Field: {stats['missing_article']} (Preamble/Intro)",
            f"Suspected Encoding Issues: {stats['encoding_suspicion']}",
            f"Short Chunks (<50 chars): {stats['short_text']}",
            f"Long Chunks (>5000 chars): {stats['long_text']}",
            f"Structural Errors: {len(errors)}"
        ]
        
        if errors:
            report.append(f"Errors: {errors[:5]}")
        else:
            report.append("Status: Structure & Integrity OK.")
            
        return "\n".join(report)
        
    except Exception as e:
        return f"Error: {str(e)}"

print(validate_chunks(r'd:\NLP\Project2\data\processed\legal_docs\legal_chunks.json'))
