"""Test config loading."""

import os
import pytest
from exchange_connectivity_hub.config import get_config, get_voyage_api_key


def test_get_config_returns_expected_structure():
    """Config should have all required sections."""
    config = get_config()
    assert "retrieval" in config
    assert "models" in config
    assert "vector_store" in config
    assert "staleness" in config
    assert "api" in config
    assert "ui" in config


def test_retrieval_config_defaults():
    """Retrieval config should have expected defaults."""
    config = get_config()
    assert config["retrieval"]["top_k"] == 20
    assert config["retrieval"]["rerank_top_n"] == 5
    assert config["retrieval"]["rerank_enabled"] is True


def test_model_names_configured():
    """Model names should be set."""
    config = get_config()
    assert config["models"]["embedding"] == "voyage-finance-2"
    assert config["models"]["rerank"] == "rerank-2.5"
    assert config["models"]["llm"] == "claude-sonnet-4-6"


def test_vector_store_config():
    """Vector store config should point to data directory."""
    config = get_config()
    assert "chroma_db" in config["vector_store"]["persist_directory"]
    assert config["vector_store"]["collection_name"] == "exchange_docs"


def test_api_keys_required():
    """API keys should be loaded from environment."""
    # Set a test key
    os.environ["VOYAGE_API_KEY"] = "test-key"
    assert get_voyage_api_key() == "test-key"

    # Test that missing key raises ValueError
    os.environ.pop("VOYAGE_API_KEY", None)
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        get_voyage_api_key()

    # Restore for other tests
    os.environ["VOYAGE_API_KEY"] = "test-key"
