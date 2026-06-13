"""Test PDF loader functionality."""

from pathlib import Path
from exchange_connectivity_hub.ingest.loader import load_pdf, extract_text_for_hash


def test_extract_text_for_hash_normalizes_text():
    """Text extraction should normalize whitespace for stable hashing."""
    # Create a test PDF content with irregular spacing
    sample_text = """
    Hello     World

    This  has    irregular    spacing.

    And line breaks.
    """

    # The function should normalize this
    normalized = extract_text_for_hash(sample_text)

    # Should collapse multiple spaces
    assert "Hello     World" not in normalized
    assert "Hello World" in normalized

    # Should trim leading/trailing whitespace per line
    assert normalized.startswith("Hello World")
    assert not normalized.startswith("  ")


def test_extract_text_for_hash_handles_empty():
    """Empty text should return empty string."""
    assert extract_text_for_hash("") == ""
    assert extract_text_for_hash("   ") == ""


def test_load_pdf_returns_documents_with_metadata(tmp_path):
    """Loading a PDF should return LangChain Documents with metadata."""

    # This test requires a real PDF file
    # For unit tests, we'll mock the PyMuPDFLoader
    # Integration tests will use real PDFs
    pass  # Implemented in integration tests


def test_load_pdf_attaches_required_metadata():
    """PDF loader should attach all required metadata fields."""
    from unittest.mock import MagicMock, patch

    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "test.pdf", "page": 0}
    mock_doc.page_content = "Test content"

    with patch("exchange_connectivity_hub.ingest.loader.PyMuPDFLoader") as mock_loader:
        mock_loader.return_value.load.return_value = [mock_doc]

        docs = load_pdf(
            pdf_path=Path("test.pdf"),
            exchange="SGX",
            doc_type="market_model",
        )

        assert len(docs) == 1
        metadata = docs[0].metadata
        assert metadata["source_filename"] == "test.pdf"
        assert metadata["exchange"] == "SGX"
        assert metadata["doc_type"] == "market_model"
        assert metadata["page_number"] == 0
        assert "ingested_at" in metadata
        assert "doc_version_hash" not in metadata  # Added after full text extraction
