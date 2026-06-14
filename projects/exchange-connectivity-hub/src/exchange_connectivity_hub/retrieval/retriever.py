"""Document retrieval from ChromaDB with optional exchange filter."""

from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import VoyageEmbeddings

from exchange_connectivity_hub.config import get_config, get_voyage_api_key


def create_retriever(
    *,
    exchange_filter: str | None = None,
    top_k: int | None = None,
) -> Any:
    """Create a retriever for querying the vector store.

    Args:
        exchange_filter: Optional exchange code to filter results (e.g., "SGX")
        top_k: Number of documents to retrieve (uses config default if None)

    Returns:
        LangChain retriever instance
    """
    config = get_config()

    if top_k is None:
        top_k = config["retrieval"]["top_k"]

    persist_directory = config["vector_store"]["persist_directory"]
    collection_name = config["vector_store"]["collection_name"]
    embedding_model = config["models"]["embedding"]

    # Initialize embeddings
    embeddings = VoyageEmbeddings(
        voyage_api_key=get_voyage_api_key(),
        model=embedding_model,
    )

    # Connect to ChromaDB
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Build search kwargs
    search_kwargs = {"k": top_k}

    # Add metadata filter if exchange specified
    if exchange_filter:
        search_kwargs["filter"] = {"exchange": exchange_filter}

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs=search_kwargs,
    )

    return retriever
