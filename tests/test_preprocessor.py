from src.processing.preprocessor import LegalPreprocessor


def test_preprocessor_removes_common_noise_lines() -> None:
    preprocessor = LegalPreprocessor()
    raw = (
        "Thuộc tính\n"
        "VB liên quan\n"
        "Điều 124. Giao dịch dân sự vô hiệu do giả tạo\n"
        "1. Nội dung chính...\n"
        "Cơ sở dữ liệu quốc gia về VBQPPL\n"
    )

    cleaned = preprocessor.clean_text(raw)

    assert "Thuộc tính" not in cleaned
    assert "VB liên quan" not in cleaned
    assert "Cơ sở dữ liệu quốc gia về VBQPPL" not in cleaned
    assert "Điều 124." in cleaned


def test_preprocessor_dedupes_adjacent_lines() -> None:
    preprocessor = LegalPreprocessor()
    raw = (
        "Điều 1. Quy định chung\n"
        "Nội dung A\n"
        "Nội dung A\n"
        "Nội dung B\n"
    )

    cleaned = preprocessor.clean_text(raw)
    lines = cleaned.splitlines()

    assert lines == ["Điều 1. Quy định chung", "Nội dung A", "Nội dung B"]
