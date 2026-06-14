"""Tests for hybrid BM25 + vector retriever (LIN-135)."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class _StubRetriever(BaseRetriever):
    """Minimal real Runnable stub so EnsembleRetriever's pydantic validator passes."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        return []


MOCK_CONFIG = {
    "vector_store": {
        "persist_directory": "/tmp/test_chroma",
        "collection_name": "test_collection",
    },
    "models": {"embedding": "voyage-finance-2"},
    "retrieval": {"top_k": 5, "bm25_weight": 0.4, "vector_weight": 0.6},
}


@pytest.fixture()
def sample_docs():
    return [
        Document(
            page_content="VCM triggers 5-minute cooling period for ±10% price moves",
            metadata={"source_filename": "HKEX_VCM.pdf", "exchange": "HKSE", "page_number": 1},
        ),
        Document(
            page_content="Board lot sizes target HK$1,000 to HK$2,000 lot value threshold",
            metadata={"source_filename": "HKEX_BoardLot.pdf", "exchange": "HKSE", "page_number": 2},
        ),
        Document(
            page_content="Settlement cycle is T+2 for all equity securities",
            metadata={
                "source_filename": "HKEX_Settlement.pdf",
                "exchange": "HKSE",
                "page_number": 3,
            },
        ),
    ]


def test_create_hybrid_retriever_returns_ensemble(sample_docs):
    """Should return an EnsembleRetriever combining BM25 and vector."""
    from langchain_classic.retrievers import EnsembleRetriever

    with (
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_config") as mock_cfg,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.chromadb") as mock_chroma,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.Chroma") as mock_chroma_lc,
    ):
        mock_cfg.return_value = MOCK_CONFIG

        # ChromaDB raw collection returns doc data
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": [d.page_content for d in sample_docs],
            "metadatas": [d.metadata for d in sample_docs],
            "ids": [f"id_{i}" for i in range(len(sample_docs))],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        # LangChain Chroma vectorstore
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = _StubRetriever()
        mock_chroma_lc.return_value = mock_vs

        from exchange_connectivity_hub.retrieval.hybrid_retriever import create_hybrid_retriever

        retriever = create_hybrid_retriever()

        assert isinstance(retriever, EnsembleRetriever)


def test_hybrid_retriever_exchange_filter_applied(sample_docs):
    """Should pass exchange filter to ChromaDB collection.get()."""
    with (
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_config") as mock_cfg,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.chromadb") as mock_chroma,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.Chroma") as mock_chroma_lc,
    ):
        mock_cfg.return_value = MOCK_CONFIG

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": [sample_docs[0].page_content],
            "metadatas": [sample_docs[0].metadata],
            "ids": ["id_0"],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = _StubRetriever()
        mock_chroma_lc.return_value = mock_vs

        from exchange_connectivity_hub.retrieval.hybrid_retriever import create_hybrid_retriever

        create_hybrid_retriever(exchange_filter="HKSE")

        # Collection.get() should have been called with exchange filter
        call_kwargs = mock_collection.get.call_args[1]
        assert call_kwargs.get("where") == {"exchange": "HKSE"}


def test_hybrid_retriever_no_filter_fetches_all(sample_docs):
    """Without exchange filter, should fetch all documents (no where clause)."""
    with (
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_config") as mock_cfg,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.chromadb") as mock_chroma,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.Chroma") as mock_chroma_lc,
    ):
        mock_cfg.return_value = MOCK_CONFIG

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": [d.page_content for d in sample_docs],
            "metadatas": [d.metadata for d in sample_docs],
            "ids": [f"id_{i}" for i in range(len(sample_docs))],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = _StubRetriever()
        mock_chroma_lc.return_value = mock_vs

        from exchange_connectivity_hub.retrieval.hybrid_retriever import create_hybrid_retriever

        create_hybrid_retriever(exchange_filter=None)

        call_kwargs = mock_collection.get.call_args[1]
        assert call_kwargs.get("where") is None


def test_hybrid_retriever_weights_from_config():
    """Ensemble weights come from config, not hardcoded constants."""
    from langchain_classic.retrievers import EnsembleRetriever

    custom_cfg = {
        "vector_store": {
            "persist_directory": "/tmp/test_chroma",
            "collection_name": "test_collection",
        },
        "models": {"embedding": "voyage-finance-2"},
        "retrieval": {"top_k": 5, "bm25_weight": 0.3, "vector_weight": 0.7},
    }

    with (
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_config") as mock_cfg,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.chromadb") as mock_chroma,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.Chroma") as mock_chroma_lc,
    ):
        mock_cfg.return_value = custom_cfg

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["Some exchange rule content"],
            "metadatas": [{"exchange": "HKSE", "source_filename": "test.pdf", "page_number": 1}],
            "ids": ["id_0"],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = _StubRetriever()
        mock_chroma_lc.return_value = mock_vs

        from exchange_connectivity_hub.retrieval.hybrid_retriever import create_hybrid_retriever

        retriever = create_hybrid_retriever()

        assert isinstance(retriever, EnsembleRetriever)
        assert retriever.weights == [0.3, 0.7]


def test_hybrid_retriever_explicit_weights_override_config():
    """Explicit weight args take precedence over config."""
    from langchain_classic.retrievers import EnsembleRetriever

    with (
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_config") as mock_cfg,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.chromadb") as mock_chroma,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.Chroma") as mock_chroma_lc,
    ):
        mock_cfg.return_value = MOCK_CONFIG  # config says 0.4 / 0.6

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["Some exchange rule content"],
            "metadatas": [{"exchange": "HKSE", "source_filename": "test.pdf", "page_number": 1}],
            "ids": ["id_0"],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = _StubRetriever()
        mock_chroma_lc.return_value = mock_vs

        from exchange_connectivity_hub.retrieval.hybrid_retriever import create_hybrid_retriever

        retriever = create_hybrid_retriever(bm25_weight=0.5, vector_weight=0.5)

        assert isinstance(retriever, EnsembleRetriever)
        assert retriever.weights == [0.5, 0.5]
