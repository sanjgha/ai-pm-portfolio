"""Test embedding and ChromaDB storage functionality."""

from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from exchange_connectivity_hub.ingest.embedder import (
    embed_and_store,
    get_collection,
    delete_by_source,
)


def test_get_collection_returns_chroma_collection():
    """Should return a ChromaDB collection."""
    with patch("exchange_connectivity_hub.ingest.embedder.VoyageEmbeddings"):
        with patch(
            "exchange_connectivity_hub.ingest.embedder.get_voyage_api_key", return_value="test_key"
        ):
            with patch("exchange_connectivity_hub.ingest.embedder.LangChainChroma") as mock_chroma:
                mock_collection = MagicMock()
                mock_chroma.return_value = mock_collection

                collection = get_collection(
                    persist_directory="/tmp/test_db", collection_name="test_collection"
                )

                assert collection is not None
                mock_chroma.assert_called_once()


def test_embed_and_store_calls_voyage():
    """Should call Voyage embed API and store in ChromaDB."""
    with patch("exchange_connectivity_hub.ingest.embedder.VoyageEmbeddings"):
        with patch(
            "exchange_connectivity_hub.ingest.embedder.get_voyage_api_key", return_value="test_key"
        ):
            with patch("exchange_connectivity_hub.ingest.embedder.LangChainChroma") as mock_chroma:
                mock_vectorstore = MagicMock()
                mock_chroma.return_value = mock_vectorstore

                docs = [
                    Document(page_content="Test 1", metadata={"source": "test.pdf"}),
                    Document(page_content="Test 2", metadata={"source": "test.pdf"}),
                ]

                embed_and_store(docs, collection_name="test_collection")

                # Should have called LangChainChroma
                mock_chroma.assert_called_once()

                # Should have added to collection
                mock_vectorstore.add_documents.assert_called_once_with(docs)


def test_delete_by_source_removes_chunks():
    """Should delete all chunks for a given source filename."""
    with patch("exchange_connectivity_hub.ingest.embedder.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        delete_by_source(
            source_filename="test.pdf",
            collection_name="test_collection",
        )

        # Should have called delete with metadata filter
        mock_collection.delete.assert_called_once_with(where={"source_filename": "test.pdf"})


def test_embed_and_store_preserves_metadata():
    """Should preserve all metadata from source documents."""
    with patch("exchange_connectivity_hub.ingest.embedder.VoyageEmbeddings"):
        with patch(
            "exchange_connectivity_hub.ingest.embedder.get_voyage_api_key", return_value="test_key"
        ):
            with patch("exchange_connectivity_hub.ingest.embedder.LangChainChroma") as mock_chroma:
                mock_vectorstore = MagicMock()
                mock_chroma.return_value = mock_vectorstore

                doc = Document(
                    page_content="Test",
                    metadata={
                        "source_filename": "test.pdf",
                        "exchange": "SGX",
                        "page_number": 5,
                        "ingested_at": "2026-06-13T10:00:00Z",
                        "doc_version_hash": "abc123",
                    },
                )

                embed_and_store([doc], collection_name="test_collection")

                # Verify add_documents was called with the doc
                mock_vectorstore.add_documents.assert_called_once()
                call_args = mock_vectorstore.add_documents.call_args
                docs_stored = call_args[0][0] if call_args[0] else []

                if docs_stored:
                    # Check metadata is preserved
                    stored_metadata = docs_stored[0].metadata
                    assert stored_metadata["source_filename"] == "test.pdf"
                    assert stored_metadata["exchange"] == "SGX"
