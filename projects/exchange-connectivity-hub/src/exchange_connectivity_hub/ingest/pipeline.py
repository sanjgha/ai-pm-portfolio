"""Ingest pipeline orchestrating load → chunk → embed → store."""

from pathlib import Path

from exchange_connectivity_hub.config import get_config
from exchange_connectivity_hub.ingest.chunker import chunk_documents
from exchange_connectivity_hub.ingest.doc_registry import DocRegistry
from exchange_connectivity_hub.ingest.embedder import delete_by_source, embed_and_store
from exchange_connectivity_hub.ingest.loader import extract_full_text_from_docs, load_pdf


def ingest_single_pdf(
    *,
    pdf_path: Path,
    exchange: str,
    doc_type: str,
    source_url: str | None = None,
    force_reingest: bool = False,
) -> dict:
    """Run full ingest pipeline for a single PDF.

    1. Load PDF with metadata
    2. Extract full text for hash
    3. Check if already ingested (unless force)
    4. Chunk documents
    5. Embed and store in ChromaDB
    6. Update doc registry

    Args:
        pdf_path: Path to PDF file
        exchange: Exchange code
        doc_type: Document type
        source_url: Optional source URL for change detection
        force_reingest: Skip hash check and re-ingest

    Returns:
        Dict with ingest results
    """
    config = get_config()
    collection_name = config["vector_store"]["collection_name"]
    registry_path = Path("data/doc_registry.json")

    # Initialize registry
    registry = DocRegistry(registry_path)

    filename = pdf_path.name

    # Load PDF
    docs = load_pdf(pdf_path=pdf_path, exchange=exchange, doc_type=doc_type)

    # Compute hash
    full_text = extract_full_text_from_docs(docs)

    from exchange_connectivity_hub.ingest.update_checker import compute_hash

    content_hash = compute_hash(full_text)

    # Check if already ingested
    if not force_reingest:
        current_hash = registry.get_current_hash(filename)
        if current_hash == content_hash:
            return {
                "status": "skipped",
                "filename": filename,
                "reason": "already ingested (hash unchanged)",
            }

    # Delete existing chunks if re-ingesting
    if force_reingest or registry.get_current_hash(filename):
        delete_by_source(
            source_filename=filename,
            collection_name=collection_name,
        )

    # Attach hash to metadata
    for doc in docs:
        doc.metadata["doc_version_hash"] = content_hash

    # Chunk
    chunks = chunk_documents(docs)

    # Embed and store
    embed_and_store(chunks, collection_name=collection_name)

    # Update registry
    registry.register_doc(
        filename=filename,
        source_url=source_url,
        exchange=exchange,
        doc_type=doc_type,
        content_hash=content_hash,
        chunks_count=len(chunks),
    )

    return {
        "status": "success",
        "filename": filename,
        "chunks_ingested": len(chunks),
        "hash": content_hash,
    }


def ingest_all_from_registry(
    *,
    force_reingest: bool = False,
) -> list[dict]:
    """Ingest all documents registered in doc_registry.json.

    Args:
        force_reingest: Re-ingest all documents regardless of hash

    Returns:
        List of ingest result dicts
    """
    registry_path = Path("data/doc_registry.json")
    registry = DocRegistry(registry_path)

    results = []
    raw_dir = Path("data/raw")

    for filename, entry in registry.data.items():
        pdf_path = raw_dir / filename

        if not pdf_path.exists():
            results.append(
                {
                    "status": "skipped",
                    "filename": filename,
                    "reason": "PDF file not found in data/raw/",
                }
            )
            continue

        result = ingest_single_pdf(
            pdf_path=pdf_path,
            exchange=entry["exchange"],
            doc_type=entry["doc_type"],
            source_url=entry.get("source_url"),
            force_reingest=force_reingest,
        )
        results.append(result)

    return results
