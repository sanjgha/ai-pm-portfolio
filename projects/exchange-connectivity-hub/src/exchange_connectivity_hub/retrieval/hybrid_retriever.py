"""Hybrid BM25 + vector retriever using Reciprocal Rank Fusion (LIN-135)."""

from typing import Any

import chromadb
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import VoyageEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from exchange_connectivity_hub.config import get_config, get_voyage_api_key


def create_hybrid_retriever(
    *,
    exchange_filter: str | None = None,
    top_k: int | None = None,
    bm25_weight: float | None = None,
    vector_weight: float | None = None,
) -> EnsembleRetriever:
    """Create a hybrid BM25 + dense vector retriever via Reciprocal Rank Fusion.

    Solves terminology mismatch: BM25 catches exact keyword hits (e.g. "circuit breaker"
    in VCM consultation doc) that pure vector search misses due to semantic drift.

    Args:
        exchange_filter: Optional exchange code to filter both retrievers (e.g. "HKSE")
        top_k: Documents per retriever before fusion (uses config default if None)
        bm25_weight: RRF weight for BM25 (uses config ``retrieval.bm25_weight`` if None)
        vector_weight: RRF weight for vector search (uses config ``retrieval.vector_weight`` if None)

    Returns:
        EnsembleRetriever fusing BM25 and vector results by the configured weights
    """
    config = get_config()

    if top_k is None:
        top_k = config["retrieval"]["top_k"]
    # Config-driven so ranking can be tuned without code changes (LIN-137).
    # Vector-dominant by default (BM25 0.4 / vector 0.6): BM25 rescues keyword
    # mismatches. Explicit args always take precedence.
    if bm25_weight is None:
        bm25_weight = config["retrieval"]["bm25_weight"]
    if vector_weight is None:
        vector_weight = config["retrieval"]["vector_weight"]

    persist_directory = config["vector_store"]["persist_directory"]
    collection_name = config["vector_store"]["collection_name"]
    embedding_model = config["models"]["embedding"]

    # --- BM25 retriever: load docs from ChromaDB and index them ---
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_collection(collection_name)

    get_kwargs: dict[str, Any] = {}
    if exchange_filter:
        get_kwargs["where"] = {"exchange": exchange_filter}

    raw = collection.get(**get_kwargs)

    bm25_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = top_k

    # --- Vector retriever: Chroma + Voyage embeddings ---
    embeddings = VoyageEmbeddings(  # type: ignore[call-arg]
        voyage_api_key=get_voyage_api_key(),
        model=embedding_model,
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    search_kwargs: dict[str, Any] = {"k": top_k}
    if exchange_filter:
        search_kwargs["filter"] = {"exchange": exchange_filter}

    vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )
