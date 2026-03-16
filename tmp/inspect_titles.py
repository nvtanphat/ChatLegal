import json
with open(r'd:\NLP\Project2\data\processed\legal_docs\legal_chunks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
titles = sorted(list(set(chunk.get('title') for chunk in data)))
print("Titles found:")
for t in titles:
    print(f"- {t}")

# Also check for article "Dieu 124"
dieu_124 = [c for c in data if c.get('article') == "Dieu 124"]
print(f"\nFound {len(dieu_124)} chunks for Dieu 124")
if dieu_124:
    print(f"Sample from: {dieu_124[0].get('title')}")
