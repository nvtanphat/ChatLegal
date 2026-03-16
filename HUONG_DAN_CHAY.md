# Hướng dẫn Chạy Dự án VN Law Chatbot

Dự án này là hệ thống Chatbot RAG (Retrieval-Augmented Generation) chuyên về luật dân sự Việt Nam, chạy local sử dụng Ollama, MongoDB và ChromaDB.

## 1. Chuẩn bị (Prerequisites)

Trước khi bắt đầu, hãy đảm bảo bạn đã cài đặt:

1.  **[uv](https://github.com/astral-sh/uv)**: Trình quản lý package cực nhanh cho Python.
2.  **[Ollama](https://ollama.com/)**: Để chạy LLM local. Sau khi cài, hãy pull model mặc định:
    ```bash
    ollama pull qwen2.5:7b
    ```
3.  **MongoDB**: Dùng để lưu trữ văn bản luật thô và lịch sử chat. Bạn có thể chạy qua Docker:
    ```bash
    docker run -d -p 27017:27017 --name mongodb mongo
    ```

## 2. Thiết lập dự án

Mở terminal trong thư mục dự án và chạy:

```bash
# Cài đặt môi trường và dependencies
uv sync

# Tạo file cấu hình môi trường
copy .env.example .env
```

**Lưu ý:** Hãy mở file `.env` và kiểm tra cấu hình kết nối tới MongoDB, Ollama và các thông số khác nếu cần.

## 3. Bước 1: Thu thập dữ liệu (Crawl)

Dự án cần có dữ liệu văn bản luật để hoạt động.

### Cào văn bản luật từ VBPL:
Mỗi văn bản trên `vbpl.vn` có một `ItemID`. Ví dụ Bộ luật dân sự 2015 có ID `91650`.
```bash
uv run crawl-legal-docs --item-id 91650 --to-mongo
```

### Cào bộ câu hỏi QA (Tùy chọn):
Dùng để đánh giá hoặc làm giàu dữ liệu:
```bash
uv run crawl-qa-dataset --max-pages 2 --target-count 50
```

## 4. Bước 2: Xây dựng chỉ mục (Indexing)

Sau khi đã có dữ liệu trong MongoDB, chúng ta cần chia nhỏ văn bản (chunking) và chuyển đổi thành vector (embeddings) để lưu vào Vector DB.

```bash
uv run build-embeddings --from-mongo
```

## 5. Bước 3: Kiểm thử hệ thống (Testing)

Trước khi chạy thật, hãy đảm bảo các module hoạt động đúng:

```bash
uv run pytest
```

Hệ thống sẽ chạy các test trong thư mục `tests/` để kiểm tra logic chunking, retrieval và inference.

## 6. Bước 4: Khởi chạy ứng dụng (Running)

Cuối cùng, khởi chạy giao diện chatbot bằng Streamlit:

```bash
uv run streamlit run app.py
```

Sau khi chạy lệnh trên, trình duyệt sẽ tự động mở trang chatbot. Bạn có thể bắt đầu đặt các câu hỏi về luật dân sự (ví dụ: "Quy định về thừa kế theo pháp luật?").

---

### Tóm tắt quy trình:
`Crawl` -> `Build Embeddings` -> `Test` -> `Run App`
