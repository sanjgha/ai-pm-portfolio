"""Test Voyage reranker functionality."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from exchange_connectivity_hub.retrieval.reranker import rerank_documents


def test_rerank_documents_returns_subset():
    """Should return reranked subset of documents."""
    docs = [Document(page_content=f"Content {i}", metadata={"id": i}) for i in range(20)]

    with (
        patch("exchange_connectivity_hub.retrieval.reranker.VoyageClient") as mock_voyage,
        patch("exchange_connectivity_hub.retrieval.reranker.get_voyage_api_key"),
    ):
        mock_client = MagicMock()
        mock_voyage.return_value = mock_client

        # Mock rerank response - return top 5 indices
        mock_client.rerank.return_value = MagicMock(
            results=[MagicMock(index=i, relevance_score=0.9 - i * 0.1) for i in range(5)]
        )

        reranked = rerank_documents(docs, query="test query", top_n=5)

        # Should return 5 documents
        assert len(reranked) == 5


def test_rerank_documents_preserves_metadata():
    """Reranked docs should preserve original metadata."""
    docs = [
        Document(page_content=f"Content {i}", metadata={"source": f"doc{i}.pdf", "page": i})
        for i in range(10)
    ]

    with (
        patch("exchange_connectivity_hub.retrieval.reranker.VoyageClient") as mock_voyage,
        patch("exchange_connectivity_hub.retrieval.reranker.get_voyage_api_key"),
    ):
        mock_client = MagicMock()
        mock_voyage.return_value = mock_client

        mock_client.rerank.return_value = MagicMock(
            results=[MagicMock(index=0, relevance_score=0.9)]
        )

        reranked = rerank_documents(docs, query="test query", top_n=1)

        assert len(reranked) == 1
        assert reranked[0].metadata["source"] == "doc0.pdf"


def test_rerank_disabled_returns_original():
    """When disabled, should return original docs (up to top_n)."""
    docs = [Document(page_content=f"Content {i}") for i in range(10)]

    reranked = rerank_documents(docs, query="test query", top_n=5, enabled=False)

    # Should return first 5 (no reranking)
    assert len(reranked) == 5


def test_rerank_empty_documents():
    """Empty input should return empty list."""
    reranked = rerank_documents([], query="test query", top_n=5)
    assert reranked == []
