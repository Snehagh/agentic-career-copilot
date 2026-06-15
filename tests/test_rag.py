"""Tests for the RAG pipeline (no API keys required)."""
import os
import tempfile
from unittest.mock import MagicMock, patch


def test_ingest_document_returns_chunk_count():
    with patch("app.rag.embedder._client") as mock_client, \
         patch("app.rag.embedder._embed_fn"):
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_client.get_or_create_collection.return_value = mock_col

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Python FastAPI machine learning AWS experience.")
            tmp = f.name

        from app.rag.embedder import ingest_document
        count = ingest_document(tmp, "test_collection")
        os.unlink(tmp)
        assert count >= 1


def test_query_returns_list():
    with patch("app.rag.embedder._client") as mock_client, \
         patch("app.rag.embedder._embed_fn"):
        mock_col = MagicMock()
        mock_col.count.return_value = 2
        mock_col.query.return_value = {"documents": [["chunk1", "chunk2"]]}
        mock_client.get_or_create_collection.return_value = mock_col

        from app.rag.embedder import query
        results = query("Python skills", "resumes")
        assert isinstance(results, list)
        assert results[0] == "chunk1"


def test_query_empty_collection_returns_empty():
    with patch("app.rag.embedder._client") as mock_client, \
         patch("app.rag.embedder._embed_fn"):
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_col

        from app.rag.embedder import query
        results = query("anything", "empty_collection")
        assert results == []
