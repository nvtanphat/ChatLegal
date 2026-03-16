# VN Law Chatbot

Local-first Vietnamese civil law chatbot using RAG.

## Quick Start

```bash
uv sync
copy .env.example .env
streamlit run app.py
```

## Pipeline

1. Crawl legal docs: `uv run crawl-legal-docs --item-id <VBPL_ID> --to-mongo`
2. Build embeddings: `uv run build-embeddings --from-mongo`
3. Run UI: `uv run streamlit run app.py`

### Crawl Presets

- List presets: `uv run crawl-legal-docs --list-presets`
- Civil focus bundle (BLDS 2015 + Dat dai/Nha o/HN&GĐ/BVQLNTD + NQ/ND huong dan):
  `uv run crawl-legal-docs --preset civil-focus-2024 --output-dir data/raw/legal_docs`

### QA Pipeline (Separated)

```bash
uv run python scripts/crawl_qa_dataset.py \
  --topic quyen-dan-su \
  --topic hon-nhan-gia-dinh \
  --topic bat-dong-san \
  --topic thua-ke \
  --topic thu-tuc-ly-hon \
  --start-page 1 \
  --max-pages 120 \
  --target-count 2500 \
  --timeout 25 \
  --retry-total 2 \
  --retry-connect 2 \
  --retry-backoff 0.4 \
  --show-progress \
  --output-file data/raw/qa_pairs/qa_pairs_raw.json
```

```bash
uv run python scripts/preprocess_qa_dataset.py \
  --input-file data/raw/qa_pairs/qa_pairs_raw.json \
  --legal-docs-dir data/raw/legal_docs \
  --target-count 500 \
  --output-file data/processed/qa_pairs/qa_pairs_processed_vbpl_civil_500.json
```

```bash
uv run build-qa-embeddings \
  --input-file data/processed/qa_pairs/qa_pairs_processed_vbpl_civil_500.json \
  --collection-name qa_collection
```

Notes:
- `crawl_qa_dataset.py`: crawl raw only.
- `preprocess_qa_dataset.py`: filter/process only.
- `build_qa_embeddings.py`: index QA questions into a dedicated Chroma collection.

## Scope

- Civil substantive law only.
- Civil procedure law is out of scope.
- The system output is informational, not official legal advice.
