import pytest
from unittest.mock import MagicMock, patch
from src.rag.reranker import VietnameseReranker
from src.rag.retriever import RetrievedChunk

def test_reranker_disabled_returns_original_ranking():
    # Arrange
    reranker = VietnameseReranker(enabled=False)
    candidates = [
        RetrievedChunk("1", "S1", "T1", "A1", "Text 1", 0.8, 0.7, 0.75),
        RetrievedChunk("2", "S2", "T2", "A2", "Text 2", 0.9, 0.8, 0.85),
    ]
    
    # Act
    results = reranker.rerank("query", candidates)
    
    # Assert
    assert len(results) == 2
    # Should be sorted by final_score desc when disabled
    assert results[0].chunk_id == "2"
    assert results[1].chunk_id == "1"

def test_reranker_empty_candidates():
    reranker = VietnameseReranker(enabled=True)
    results = reranker.rerank("query", [])
    assert results == []

@patch("src.rag.reranker.CrossEncoder")
def test_reranker_logic_with_mock_model(mock_cross_encoder):
    # Arrange
    mock_model = MagicMock()
    # Mock predict to return scores for 2 items
    mock_model.predict.return_value = [0.1, 0.9] 
    mock_cross_encoder.return_value = mock_model
    
    reranker = VietnameseReranker(enabled=True)
    candidates = [
        RetrievedChunk("1", "S1", "T1", "A1", "Text 1", 0.5, 0.5, 0.5), # Score 0.1 -> rank 2
        RetrievedChunk("2", "S2", "T2", "A2", "Text 2", 0.5, 0.5, 0.5), # Score 0.9 -> rank 1
    ]
    
    # Act
    results = reranker.rerank("query", candidates)
    
    # Assert
    assert len(results) == 2
    assert results[0].chunk_id == "2"
    assert results[0].rerank_score == 0.9
    assert results[1].chunk_id == "1"
    assert results[1].rerank_score == 0.1

def test_reranker_fallback_on_error():
    reranker = VietnameseReranker(enabled=True)
    # Force error by making _model None and ensuring CrossEncoder is None or mocking _ensure_model
    with patch.object(reranker, "_ensure_model", side_effect=Exception("Model load failed")):
        candidates = [
            RetrievedChunk("1", "S1", "T1", "A1", "Text 1", 0.8, 0.7, 0.75),
            RetrievedChunk("2", "S2", "T2", "A2", "Text 2", 0.9, 0.8, 0.85),
        ]
        
        # Should fallback to final_score
        results = reranker.rerank("query", candidates)
        assert results[0].chunk_id == "2"
        assert results[1].chunk_id == "1"
