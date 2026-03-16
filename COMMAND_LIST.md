# Tổng hợp Lệnh Sử dụng Dự án VN Law Chatbot

Tài liệu này tổng hợp toàn bộ các lệnh cần thiết để vận hành hệ thống VN Law Chatbot, từ thiết lập môi trường đến chạy ứng dụng.

## 1. Thiết lập Môi trường (Setup)

```bash
# Cài đặt toàn bộ dependencies (yêu cầu 'uv')
uv sync

# Khởi tạo file môi trường (nếu chưa có)
copy .env.example .env

# Chạy MongoDB local bằng Docker (nếu không dùng Atlas)
docker run -d -p 27017:27017 --name mongodb mongo

# Tải mô hình LLM về local bằng Ollama
ollama pull qwen2.5:3b
```

## 2. Thu thập Dữ liệu (Crawling)

### Cào văn bản luật từ vbpl.vn
```bash
# Ví dụ: Cào Bộ luật Dân sự 2015 (ItemID 91650) vào MongoDB
uv run crawl-legal-docs --item-id 91650 --to-mongo

# Cào từ file URL hoặc ID danh sách
uv run crawl-legal-docs --from-file scripts/item_ids.txt --to-mongo
```

### Cào bộ câu hỏi trả lời (QA)
```bash
# Cào mẫu câu hỏi từ tthc.mca.gov.vn
uv run crawl-qa-dataset --max-pages 2 --target-count 50
```

## 3. Xử lý và Đánh chỉ mục (Indexing)

### Xử lý văn bản luật
```bash
# Chunking và tạo Vector Embeddings cho văn bản luật
uv run build-embeddings --from-mongo
```

### Xử lý bộ QA (Tùy chọn)
```bash
# Preprocess dữ liệu QA đã cào
uv run preprocess-qa-dataset

# Tạo Embeddings cho bộ QA để hỗ trợ Retrieval
uv run build-qa-embeddings
```

## 4. Chạy Ứng dụng (Running)

```bash
# Chạy giao diện Chatbot Streamlit
uv run streamlit run app.py
```

## 5. Kiểm thử và Đánh giá (Testing & Evaluation)

### Chạy các Unit Test
```bash
# Kiểm tra logic hệ thống
uv run pytest

# Kiểm tra một file test cụ thể
uv run pytest tests/test_reranker.py
```

### Đánh giá chất lượng RAG (Ragas)
```bash
# Chạy script đánh giá (Yêu cầu OpenAI API Key hoặc cấu hình LLM phù hợp)
uv run evaluate-rag
```

## 6. Trực quan hóa Dữ liệu (Visualization)

```bash
# Cài đặt thư viện hỗ trợ (đã được thêm vào pyproject.toml)
uv sync

# Chạy Jupyter Notebook để xem phân tích dữ liệu
uv run jupyter notebook notebooks/data_visualization.ipynb
```

## 7. Lệnh Tiện ích khác

```bash
# Kiểm tra định dạng code (Linting)
uv run ruff check .

# Sửa lỗi định dạng tự động
uv run ruff format .
```
