"""Test document registry management."""

from pathlib import Path
from exchange_connectivity_hub.ingest.doc_registry import (
    DocRegistry,
    load_registry,
    save_registry,
)


def test_load_registry_creates_default():
    """Loading non-existent registry should create default."""
    test_path = Path("/tmp/test_registry_empty.json")
    if test_path.exists():
        test_path.unlink()

    registry = load_registry(test_path)
    assert registry == {}
    assert test_path.exists()


def test_save_and_load_registry():
    """Saving and loading should preserve data."""
    test_path = Path("/tmp/test_registry_save.json")
    if test_path.exists():
        test_path.unlink()

    registry = {
        "SGX_doc.pdf": {
            "source_url": "https://sgx.com/doc.pdf",
            "exchange": "SGX",
            "doc_type": "market_model",
            "current_hash": "abc123",
            "version_history": [
                {
                    "hash": "abc123",
                    "ingested_at": "2026-06-13T10:00:00Z",
                    "chunks_count": 100,
                }
            ],
        }
    }

    save_registry(registry, test_path)
    loaded = load_registry(test_path)

    assert loaded == registry
    assert loaded["SGX_doc.pdf"]["exchange"] == "SGX"


def test_doc_registry_add_version():
    """DocRegistry should add new versions correctly."""
    test_path = Path("/tmp/test_registry_add.json")
    if test_path.exists():
        test_path.unlink()

    registry = DocRegistry(test_path)
    registry.register_doc(
        filename="test.pdf",
        source_url="https://test.com/test.pdf",
        exchange="TEST",
        doc_type="test_doc",
        content_hash="hash1",
        chunks_count=50,
    )

    # Verify entry exists
    assert "test.pdf" in registry.data
    assert registry.data["test.pdf"]["current_hash"] == "hash1"
    assert registry.data["test.pdf"]["version_history"][0]["chunks_count"] == 50

    # Add new version
    registry.register_doc(
        filename="test.pdf",
        source_url="https://test.com/test.pdf",
        exchange="TEST",
        doc_type="test_doc",
        content_hash="hash2",
        chunks_count=55,
    )

    # Should have two versions
    assert len(registry.data["test.pdf"]["version_history"]) == 2
    assert registry.data["test.pdf"]["current_hash"] == "hash2"


def test_doc_registry_check_hash():
    """check_hash_changed should detect modifications."""
    test_path = Path("/tmp/test_registry_hash.json")
    if test_path.exists():
        test_path.unlink()

    registry = DocRegistry(test_path)
    registry.register_doc(
        filename="test.pdf",
        source_url="https://test.com/test.pdf",
        exchange="TEST",
        doc_type="test_doc",
        content_hash="original",
        chunks_count=50,
    )

    # Same hash = not changed
    assert registry.check_hash_changed("test.pdf", "original") is False

    # Different hash = changed
    assert registry.check_hash_changed("test.pdf", "new_hash") is True

    # Unknown doc = considered changed (will register)
    assert registry.check_hash_changed("unknown.pdf", "any_hash") is True


def test_doc_registry_get_source_url():
    """getSourceUrl should return URL or None."""
    test_path = Path("/tmp/test_registry_url.json")
    if test_path.exists():
        test_path.unlink()

    registry = DocRegistry(test_path)
    registry.register_doc(
        filename="test.pdf",
        source_url="https://test.com/test.pdf",
        exchange="TEST",
        doc_type="test_doc",
        content_hash="hash1",
        chunks_count=50,
    )

    assert registry.get_source_url("test.pdf") == "https://test.com/test.pdf"
    assert registry.get_source_url("unknown.pdf") is None


def test_doc_registry_get_current_hash():
    """get_current_hash should return hash or None."""
    test_path = Path("/tmp/test_registry_current_hash.json")
    if test_path.exists():
        test_path.unlink()

    registry = DocRegistry(test_path)
    registry.register_doc(
        filename="test.pdf",
        source_url="https://test.com/test.pdf",
        exchange="TEST",
        doc_type="test_doc",
        content_hash="hash1",
        chunks_count=50,
    )

    assert registry.get_current_hash("test.pdf") == "hash1"
    assert registry.get_current_hash("unknown.pdf") is None


def test_doc_registry_get_all_filenames():
    """get_all_filenames should return list of all tracked docs."""
    test_path = Path("/tmp/test_registry_filenames.json")
    if test_path.exists():
        test_path.unlink()

    registry = DocRegistry(test_path)
    registry.register_doc(
        filename="doc1.pdf",
        source_url="https://test.com/doc1.pdf",
        exchange="TEST",
        doc_type="test_doc",
        content_hash="hash1",
        chunks_count=50,
    )
    registry.register_doc(
        filename="doc2.pdf",
        source_url="https://test.com/doc2.pdf",
        exchange="TEST",
        doc_type="test_doc",
        content_hash="hash2",
        chunks_count=60,
    )

    filenames = registry.get_all_filenames()
    assert set(filenames) == {"doc1.pdf", "doc2.pdf"}
    assert len(filenames) == 2
