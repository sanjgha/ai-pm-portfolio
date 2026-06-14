"""PDF loading with metadata attachment for exchange documents."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyMuPDFLoader


def extract_text_for_hash(raw_text: str) -> str:
    """Normalize extracted text for stable MD5 hashing.

    Removes cosmetic variations (extra whitespace, line breaks) that
    don't represent content changes.
    """
    if not raw_text:
        return ""

    # Normalize: collapse multiple spaces, strip each line
    lines = raw_text.split("\n")
    normalized_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            # Collapse multiple spaces within line
            collapsed = " ".join(stripped.split())
            normalized_lines.append(collapsed)

    return "\n".join(normalized_lines)


def load_pdf(*, pdf_path: Path, exchange: str, doc_type: str) -> list[Any]:
    """Load a PDF and attach metadata to each page/document.

    Args:
        pdf_path: Path to PDF file
        exchange: Exchange code (SGX, HKSE, TSE, etc.)
        doc_type: Document type (market_model, fix_spec, etc.)

    Returns:
        List of LangChain Documents with enhanced metadata
    """
    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()

    ingested_at = datetime.now(timezone.utc).isoformat()
    filename = pdf_path.name

    for doc in docs:
        # Enhance metadata
        base_metadata = doc.metadata
        doc.metadata = {
            "source_filename": filename,
            "exchange": exchange,
            "doc_type": doc_type,
            "page_number": base_metadata.get("page", 0),
            "ingested_at": ingested_at,
            # doc_version_hash added after full text extraction in pipeline
        }

    return docs


def extract_full_text_from_docs(docs: list[Any]) -> str:
    """Extract and concatenate all page text for hash computation.

    Args:
        docs: List of LangChain Documents from loader

    Returns:
        Normalized full text for MD5 hashing
    """
    all_text = []
    for doc in docs:
        all_text.append(doc.page_content)

    full_text = "\n".join(all_text)
    return extract_text_for_hash(full_text)
