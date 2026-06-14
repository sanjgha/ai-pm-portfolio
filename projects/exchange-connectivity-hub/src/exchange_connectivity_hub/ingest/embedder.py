"""Embedding and vector storage using Voyage AI and ChromaDB."""

from typing import Any

from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma as LangChainChroma
from langchain_community.embeddings import VoyageEmbeddings
from exchange_connectivity_hub.config import get_config, get_voyage_api_key


def get_collection(*, persist_directory: str, collection_name: str) -> Any:
    """Get or create ChromaDB collection.

    Args:
        persist_directory: Directory for ChromaDB storage
        collection_name: Name of the collection

    Returns:
        ChromaDB collection
    """
    config = get_config()
    model = config["models"]["embedding"]

    # Initialize embeddings
    embeddings = VoyageEmbeddings(
        voyage_api_key=get_voyage_api_key(),
        model=model,
    )

    # Create or get collection
    vectorstore = LangChainChroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    return vectorstore


def embed_and_store(
    docs: list[Any],
    *,
    collection_name: str,
    persist_directory: str | None = None,
) -> None:
    """Embed documents and store in ChromaDB.

    Args:
        docs: List of LangChain Documents with metadata
        collection_name: ChromaDB collection name
        persist_directory: ChromaDB storage directory (uses config if None)
    """
    config = get_config()

    if persist_directory is None:
        persist_directory = config["vector_store"]["persist_directory"]

    model = config["models"]["embedding"]
    embeddings = VoyageEmbeddings(
        voyage_api_key=get_voyage_api_key(),
        model=model,
    )

    # Create vectorstore and add documents
    vectorstore = LangChainChroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Add documents with their metadata
    vectorstore.add_documents(docs)


def delete_by_source(
    *,
    source_filename: str,
    collection_name: str,
    persist_directory: str | None = None,
) -> None:
    """Delete all chunks for a given source filename.

    Args:
        source_filename: Filename to delete (e.g., "SGX_doc.pdf")
        collection_name: ChromaDB collection name
        persist_directory: ChromaDB storage directory (uses config if None)
    """
    config = get_config()

    if persist_directory is None:
        persist_directory = config["vector_store"]["persist_directory"]

    # Connect directly to ChromaDB to delete by metadata
    chroma_client = PersistentClient(path=persist_directory)

    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Delete all documents matching the source filename
    collection.delete(
        where={"source_filename": source_filename},
    )
