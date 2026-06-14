"""Document reranking using Voyage AI rerank-2.5."""

from typing import Any

from voyageai import Client as VoyageClient

from exchange_connectivity_hub.config import get_config, get_voyage_api_key


def rerank_documents(
    docs: list[Any],
    *,
    query: str,
    top_n: int,
    enabled: bool | None = None,
) -> list[Any]:
    """Rerank documents using Voyage AI rerank API.

    Args:
        docs: List of retrieved Documents
        query: Original query string
        top_n: Number of top documents to keep after reranking
        enabled: Whether reranking is enabled (uses config if None)

    Returns:
        Reranked list of Documents (length = top_n)
    """
    if not docs:
        return []

    # Check if reranking is enabled
    if enabled is None:
        config = get_config()
        enabled = config["retrieval"]["rerank_enabled"]

    if not enabled:
        # Return original docs up to top_n
        return docs[:top_n]

    # Get rerank model from config
    config = get_config()
    rerank_model = config["models"]["rerank"]

    # Initialize Voyage client
    client = VoyageClient(api_key=get_voyage_api_key())

    # Extract document contents
    doc_contents = [doc.page_content for doc in docs]

    # Call rerank API
    rerank_results = client.rerank(
        query=query,
        documents=doc_contents,
        model=rerank_model,
        top_k=top_n,
    )

    # Reorder documents based on rerank results
    reranked_docs = []
    for result in rerank_results.results:
        original_index = result.index
        reranked_docs.append(docs[original_index])

    return reranked_docs
