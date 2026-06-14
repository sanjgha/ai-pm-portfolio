"""Test document update detection via MD5 hashing."""

import hashlib
from exchange_connectivity_hub.ingest.update_checker import (
    compute_hash,
    check_for_updates,
)


def test_compute_hash_handles_empty_text():
    """Empty text should produce valid hash."""
    hash_value = compute_hash("")
    assert hash_value is not None
    assert len(hash_value) == 32  # MD5 is 32 hex chars


def test_compute_hash_is_deterministic():
    """Same text should produce same hash."""
    text = "Hello, world!"
    hash1 = compute_hash(text)
    hash2 = compute_hash(text)
    assert hash1 == hash2


def test_compute_hash_normalizes_whitespace():
    """Hash should be based on normalized text."""
    # Different spacing, same content
    text1 = "Hello  World"
    text2 = "Hello World"

    # After normalization, should be same
    # (Note: this depends on extract_text_for_hash doing normalization)
    from exchange_connectivity_hub.ingest.loader import extract_text_for_hash

    hash1 = hashlib.md5(extract_text_for_hash(text1).encode()).hexdigest()
    hash2 = hashlib.md5(extract_text_for_hash(text2).encode()).hexdigest()

    assert hash1 == hash2


def test_download_pdf_to_temp(tmp_path):
    """Download PDF to temporary location."""
    # This test requires a real URL - use a test file
    # Integration test only
    pass


def test_check_for_updates_compares_hashes():
    """check_for_updates should detect hash changes."""
    from unittest.mock import MagicMock, patch
    from exchange_connectivity_hub.ingest.doc_registry import DocRegistry

    mock_registry = MagicMock(spec=DocRegistry)
    mock_registry.data = {
        "test.pdf": {
            "source_url": "https://example.com/test.pdf",
            "current_hash": "old_hash",
            "exchange": "SGX",
            "doc_type": "market_model",
        }
    }

    # Mock all external dependencies at their source
    with (
        patch("exchange_connectivity_hub.ingest.update_checker.download_pdf"),
        patch("exchange_connectivity_hub.ingest.loader.load_pdf") as mock_load,
        patch(
            "exchange_connectivity_hub.ingest.loader.extract_full_text_from_docs"
        ) as mock_extract,
        patch("exchange_connectivity_hub.ingest.update_checker.compute_hash") as mock_hash,
    ):
        # Setup mock chain
        mock_load.return_value = []  # Empty docs list
        mock_extract.return_value = "test content"
        mock_hash.return_value = "new_hash"  # Different hash

        changes = check_for_updates(mock_registry)

        # Should detect the change
        assert "test.pdf" in changes
        assert changes["test.pdf"]["old_hash"] == "old_hash"
        assert changes["test.pdf"]["new_hash"] == "new_hash"


def test_check_for_updates_skips_unchanged():
    """check_for_updates should skip unchanged docs."""
    from unittest.mock import MagicMock, patch
    from exchange_connectivity_hub.ingest.doc_registry import DocRegistry

    mock_registry = MagicMock(spec=DocRegistry)
    mock_registry.data = {
        "test.pdf": {
            "source_url": "https://example.com/test.pdf",
            "current_hash": "same_hash",
            "exchange": "SGX",
            "doc_type": "market_model",
        }
    }

    # Mock all external dependencies at their source
    with (
        patch("exchange_connectivity_hub.ingest.update_checker.download_pdf"),
        patch("exchange_connectivity_hub.ingest.loader.load_pdf") as mock_load,
        patch(
            "exchange_connectivity_hub.ingest.loader.extract_full_text_from_docs"
        ) as mock_extract,
        patch("exchange_connectivity_hub.ingest.update_checker.compute_hash") as mock_hash,
    ):
        # Setup mock chain
        mock_load.return_value = []  # Empty docs list
        mock_extract.return_value = "test content"
        mock_hash.return_value = "same_hash"  # Same hash

        changes = check_for_updates(mock_registry)

        # Should not report any changes
        assert len(changes) == 0
