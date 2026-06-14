"""Test document retrieval functionality."""

from unittest.mock import MagicMock, patch

from exchange_connectivity_hub.retrieval.retriever import create_retriever


def test_create_retriever_returns_vectorstore_retriever():
    """Should create a vectorstore-backed retriever."""
    with (
        patch("exchange_connectivity_hub.retrieval.retriever.Chroma") as mock_chroma,
        patch("exchange_connectivity_hub.retrieval.retriever.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.retriever.get_config") as mock_config,
    ):
        mock_config.return_value = {
            "vector_store": {
                "persist_directory": "/tmp/test",
                "collection_name": "test_collection",
            },
            "models": {"embedding": "voyage-3"},
            "retrieval": {"top_k": 5},
        }
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vectorstore

        retriever = create_retriever()

        assert retriever is not None


def test_retriever_with_exchange_filter():
    """Should filter by exchange when specified."""
    with (
        patch("exchange_connectivity_hub.retrieval.retriever.Chroma") as mock_chroma,
        patch("exchange_connectivity_hub.retrieval.retriever.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.retriever.get_config") as mock_config,
    ):
        mock_config.return_value = {
            "vector_store": {
                "persist_directory": "/tmp/test",
                "collection_name": "test_collection",
            },
            "models": {"embedding": "voyage-3"},
            "retrieval": {"top_k": 5},
        }
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vectorstore

        _ = create_retriever(exchange_filter="SGX")

        # Should have created retriever with search_kwargs
        call_kwargs = (
            mock_vectorstore.as_retriever.call_args[1]
            if mock_vectorstore.as_retriever.call_args
            else {}
        )
        assert "search_kwargs" in call_kwargs
        assert call_kwargs["search_kwargs"].get("filter") == {"exchange": "SGX"}


def test_retriever_without_filter():
    """Should not apply filter when exchange_filter is None."""
    with (
        patch("exchange_connectivity_hub.retrieval.retriever.Chroma") as mock_chroma,
        patch("exchange_connectivity_hub.retrieval.retriever.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.retriever.get_config") as mock_config,
    ):
        mock_config.return_value = {
            "vector_store": {
                "persist_directory": "/tmp/test",
                "collection_name": "test_collection",
            },
            "models": {"embedding": "voyage-3"},
            "retrieval": {"top_k": 5},
        }
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vectorstore

        _ = create_retriever(exchange_filter=None)

        # Should not have filter in search_kwargs
        call_kwargs = (
            mock_vectorstore.as_retriever.call_args[1]
            if mock_vectorstore.as_retriever.call_args
            else {}
        )
        filter_value = call_kwargs.get("search_kwargs", {}).get("filter")
        assert filter_value is None or filter_value == {}
