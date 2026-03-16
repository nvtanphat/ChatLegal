# Chatbot Pháp Luật VN - Kế Hoạch Thực Hiện

> Xây dựng hệ thống chatbot hỏi đáp pháp luật Việt Nam sử dụng RAG. Chuyên về luật dân sự vật chất. Chạy local trên Windows.

---

## 1. Tổng Quan

- **Domain:** Dân sự vật chất (KHÔNG bao gồm tố tụng dân sự)
- **Package Manager:** uv
- **Reranker:** Vietnamese_Reranker (AITeamVN 2025)
- **Nguồn crawl:** vbpl.vn

---

## 2. Danh Sách Văn Bản Pháp Luật

### A. Văn bản trục

| # | Văn bản | Số hiệu |
|---|---------|---------|
| 1 | Bộ luật Dân sự 2015 | 91/2015/QH13 |

### B. Các luật liên quan (dân sự vật chất)

| # | Văn bản | Số hiệu |
|---|---------|---------|
| 2 | Luật Hôn nhân và Gia đình 2014 | 52/2014/QH13 |
| 3 | Luật Đất đai 2024 | 81/2024/QH15 |
| 4 | Luật Sở hữu trí tuệ 2005 (sửa 2022) | 50/2005/QH11 |
| 5 | Luật Bảo vệ quyền lợi người tiêu dùng 2023 | 19/2023/QH15 |

### C. Nghị định hướng dẫn theo cụm bài toán

| Cụm bài toán | Nghị định hướng dẫn |
|--------------|---------------------|
| **Thừa kế** | Nghị định về thừa kế tài sản |
| **Tài sản chung riêng** | Nghị định về sở hữu, tài sản vợ chồng |
| **Giao dịch dân sự vô hiệu** | Nghị định hướng dẫn BLDS về giao dịch |
| **Quyền sử dụng đất** | Nghị định hướng dẫn Luật Đất đai 2024 |
| **Hợp đồng dân sự** | Nghị định về hợp đồng và nghĩa vụ |

---

## 3. Công Nghệ Chính

| Thành phần | Công nghệ | Ghi chú |
|------------|-----------|----------|
| Package Manager | **uv** | Thay thế pip hoàn toàn |
| LLM | Qwen3-8B (Ollama) | Local running |
| Embedding | DEk21_hcmute_embedding | 768 dims, tối ưu pháp luật VN |
| **Reranker** | **Vietnamese_Reranker (AITeamVN 2025)** | Cross-encoder, số 1 VN |
| Vector DB | ChromaDB → Qdrant | Dev → Production |
| Document DB | MongoDB | Lưu docs + QA + history |
| UI | Streamlit | Chat interface |
| Backend | FastAPI | REST API |
| Evaluation | RAGAS | Metrics chuẩn |

---

## 4. Cấu Trúc Thư Mục

```
vn-law-chatbot/
├── config/
│   ├── model_config.yaml
│   └── logging_config.yaml
├── data/
│   ├── raw/
│   │   ├── legal_docs/
│   │   └── qa_pairs/
│   ├── processed/
│   ├── qa_dataset/
│   ├── cache/
│   ├── embeddings/
│   └── vectordb/
├── src/
│   ├── database/
│   │   ├── mongo_client.py
│   │   └── models.py
│   ├── core/
│   │   ├── base_llm.py
│   │   ├── ollama_client.py
│   │   └── model_factory.py
│   ├── prompts/
│   │   ├── templates.py
│   │   └── chain.py
│   ├── rag/
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   ├── vector_store.py
│   │   └── indexer.py
│   ├── processing/
│   │   ├── chunking.py
│   │   ├── tokenizer.py
│   │   └── preprocessor.py
│   ├── inference/
│   │   ├── intent_router.py
│   │   ├── query_reflector.py
│   │   ├── inference_engine.py
│   │   └── response_parser.py
│   └── evaluation/
│       ├── ragas_eval.py
│       └── golden_dataset.py
├── scripts/
│   ├── crawl_legal_docs.py
│   ├── crawl_qa_dataset.py
│   ├── build_embeddings.py
│   └── evaluate.py
├── tests/
│   ├── test_chunking.py
│   ├── test_retriever.py
│   ├── test_inference.py
│   └── test_prompts.py
├── app.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── Dockerfile
└── docker-compose.yml
```

---

## 5. Giai Đoạn Triển Khai

### Giai Đoạn 1: Thiết Lập Môi Trường (Tuần 1)

1. **Khởi tạo dự án với uv**
   ```bash
   uv init vn-law-chatbot
   cd vn-law-chatbot
   mkdir -p config data src scripts tests
   ```

2. **Cài đặt dependencies**
   ```bash
   uv add langchain langchain-community chromadb pymongo underthesea ragas loguru streamlit fastapi uvicorn python-dotenv httpx
   uv add sentence-transformers transformers
   ```

3. **Cấu hình**
   - Tạo `pyproject.toml`
   - Tạo `.env.example`
   - Tạo `config/model_config.yaml`
   - Tạo `config/logging_config.yaml`

4. **Test Ollama**
   - Pull Qwen3-8B
   - Verify API call

---

### Giai Đoạn 2: Data Pipeline - Crawl & Index (Tuần 2-3)

1. **Crawl văn bản pháp luật từ vbpl.vn**
   - Tạo `scripts/crawl_legal_docs.py`
   - Crawl 5 luật + nghị định
   - Lưu vào MongoDB `legal_docs` collection

2. **Crawl QA pairs**
   - Tạo `scripts/crawl_qa_dataset.py`
   - Crawl từ thuvienphapluat.vn/hoi-dap-phap-luat
   - Target: 500+ QA pairs dân sự

3. **Xử lý dữ liệu**
   - Tạo `src/processing/preprocessor.py`
   - Tạo `src/processing/chunking.py` (legal structure)
   - Tạo `src/processing/tokenizer.py`

4. **Embedding & Vector Store**
   - Tạo `src/rag/embedder.py` (DEk21_hcmute_embedding)
   - Tạo `src/rag/vector_store.py` (ChromaDB)
   - Tạo `scripts/build_embeddings.py`

---

### Giai Đoạn 3: RAG Pipeline Core (Tuần 3-4)

1. **Retrieval**
   - Tạo `src/rag/retriever.py` (semantic + BM25 hybrid)
   - Tạo `src/rag/reranker.py` (Vietnamese_Reranker AITeamVN)
   - Tạo `src/rag/indexer.py`

2. **LLM Integration**
   - Tạo `src/core/base_llm.py`
   - Tạo `src/core/ollama_client.py`
   - Tạo `src/core/model_factory.py`

3. **Prompt Engineering**
   - Tạo `src/prompts/templates.py`
   - Tạo `src/prompts/chain.py`

---

### Giai Đoạn 4: Inference Engine & UI (Tuần 4-5)

1. **Inference Pipeline**
   - Tạo `src/inference/intent_router.py`
   - Tạo `src/inference/query_reflector.py`
   - Tạo `src/inference/inference_engine.py`
   - Tạo `src/inference/response_parser.py`

2. **Database Layer**
   - Tạo `src/database/mongo_client.py`
   - Tạo `src/database/models.py`

3. **UI Application**
   - Tạo `app.py` (Streamlit)
   - Tích hợp citation display
   - Thêm conversation memory

4. **Evaluation Setup**
   - Tạo `src/evaluation/ragas_eval.py`
   - Tạo `src/evaluation/golden_dataset.py`
   - Tạo `scripts/evaluate.py`

---

### Giai Đoạn 5: Nâng Cao & Production (Tuần 6-8)

1. **Advanced Features**
   - Cross-reference resolution
   - Law Status Filter
   - Fallback Web Search (Tavily)
   - Semantic Cache (Redis)
   - Job kiểm tra tình trạng hiệu lực

2. **A/B Testing**
   - So sánh Qwen3-8B vs Vistral

3. **Production**
   - FastAPI backend
   - Qdrant
   - Docker + Docker Compose
   - Logging + Monitoring

---

## 6. Thứ Tự Ưu Tiên Triển Khai

```
1. Setup project structure với uv
2. Install Ollama + test models
3. MongoDB setup
4. Crawl văn bản trục BLDS 2015 → MongoDB
5. Crawl các luật liên quan + Nghị định
6. Chunking + Embedding → ChromaDB
7. Retriever → test query
8. Ollama client + prompts
9. Inference engine
10. Streamlit UI
11. Crawl QA dataset chuyên dân sự
12. Evaluation pipeline
13. Advanced features (Vietnamese_Reranker, hybrid search)
14. Production deployment
```

---

## 7. Files Cần Tạo (Top Priority)

- `pyproject.toml` (quản lý bằng uv)
- `.env.example`
- `config/model_config.yaml`
- `src/database/mongo_client.py`
- `src/core/ollama_client.py`
- `src/processing/chunking.py`
- `src/rag/embedder.py`
- `src/rag/retriever.py`
- `src/rag/reranker.py` (Vietnamese_Reranker AITeamVN)
- `src/inference/inference_engine.py`
- `scripts/crawl_legal_docs.py` (vbpl.vn)
- `scripts/build_embeddings.py`
- `app.py`

---

## 8. Verification Plan

### Manual Testing

| Câu hỏi test | Kỳ vọng |
|--------------|---------|
| "Quy định về thừa kế không di chúc?" | Trích dẫn Điều/Khoản từ BLDS 2015 |
| "Tài sản chung của vợ chồng gồm những gì?" | Trích dẫn BLDS 2015 |
| "Hợp đồng mua bán bất động sản cần công chứng?" | Trích dẫn quy định liên quan |
| "Hôm nay thời tiết thế nào?" | Intent routing → Chitchat |
| "Điều 124 BLDS 2015 quy định gì?" | Tìm đúng Điều 124 |

---

> **Lưu ý quan trọng:**
> - KHÔNG bao gồm tố tụng dân sự (BLTTDS 2015)
> - Chatbot KHÔNG phải là tư vấn pháp lý chính thức
> - Cần có disclaimer trong UI
