import sys
import httpx
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def test_stream():
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": "Xin chào, bạn là ai?"}],
        "stream": True
    }
    
    try:
        with httpx.stream("POST", url, json=payload, timeout=30) as response:
            print(f"Status code: {response.status_code}")
            if response.status_code != 200:
                print(f"Error: {response.read()}")
                return
                
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if content := chunk.get("message", {}).get("content"):
                        print(content, end="", flush=True)
                    if chunk.get("done"):
                        print("\nDone!")
                        break
    except Exception as e:
        print(f"\nFailed: {e}")

if __name__ == "__main__":
    test_stream()
