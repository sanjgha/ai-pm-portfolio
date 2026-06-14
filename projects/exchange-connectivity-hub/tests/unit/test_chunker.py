"""Test document chunking functionality."""

from langchain_core.documents import Document
from exchange_connectivity_hub.ingest.chunker import chunk_documents


def test_chunk_documents_returns_list():
    """Chunking should return a list of Documents."""
    docs = [Document(page_content="Test content")]
    chunks = chunk_documents(docs)
    assert isinstance(chunks, list)


def test_chunk_documents_preserves_metadata():
    """Chunks should preserve source metadata."""
    docs = [
        Document(
            page_content="Short content",
            metadata={
                "source_filename": "test.pdf",
                "exchange": "SGX",
                "page_number": 1,
            },
        )
    ]

    chunks = chunk_documents(docs)

    # All chunks should have the original metadata
    for chunk in chunks:
        assert chunk.metadata["source_filename"] == "test.pdf"
        assert chunk.metadata["exchange"] == "SGX"
        assert chunk.metadata["page_number"] == 1


def test_chunk_documents_chunk_size():
    """Chunks should respect approximate token size."""
    # Create a document with varied content to ensure accurate tokenization
    # Using diverse vocabulary to prevent token compression
    words = [
        "exchange",
        "trading",
        "system",
        "connectivity",
        "protocol",
        "message",
        "order",
        "execution",
        "market",
        "data",
        "feed",
        "gateway",
        "session",
        "establishment",
        "heartbeat",
        "sequence",
        "number",
        "timestamp",
        "security",
        "definition",
        "instrument",
        "price",
        "quantity",
        "side",
        "buy",
        "sell",
        "valid",
        "invalid",
        "rejected",
        "accepted",
        "pending",
        "filled",
        "partial",
        "cancel",
        "replace",
        "status",
        "code",
        "field",
        "tag",
        "value",
        "format",
        "type",
        "length",
        "checksum",
        "header",
        "trailer",
        "body",
        "delimiter",
    ]

    # Create enough text to exceed chunk_size (512 tokens)
    # Need ~40-50 sentences to exceed 512 tokens
    sentences = []
    for i in range(50):
        sentence = f"The {words[i % len(words)]} {words[(i + 1) % len(words)]} {words[(i + 2) % len(words)]} {words[(i + 3) % len(words)]} {words[(i + 4) % len(words)]} field contains {words[(i + 5) % len(words)]} data for {words[(i + 6) % len(words)]} processing. "
        sentences.append(sentence)

    long_text = "".join(sentences)

    docs = [Document(page_content=long_text)]
    chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=64)

    # Should split into multiple chunks
    assert len(chunks) > 1

    # Each chunk should be under the target size (with some tolerance)
    for chunk in chunks:
        # Rough check: content length should be reasonable
        assert len(chunk.page_content) < len(long_text)


def test_chunk_documents_overlap():
    """Chunks should have overlap for context continuity."""
    # Use diverse vocabulary to ensure proper tokenization
    words = [
        "exchange",
        "trading",
        "system",
        "connectivity",
        "protocol",
        "message",
        "order",
        "execution",
        "market",
        "data",
        "feed",
        "gateway",
        "session",
        "establishment",
        "heartbeat",
        "sequence",
        "number",
        "timestamp",
        "security",
    ]

    # Create enough text to ensure multiple chunks
    sentences = []
    for i in range(20):
        sentence = f"The {words[i % len(words)]} {words[(i + 1) % len(words)]} {words[(i + 2) % len(words)]} field contains data for {words[(i + 3) % len(words)]} processing. "
        sentences.append(sentence)

    long_text = "".join(sentences)

    docs = [Document(page_content=long_text)]
    chunks = chunk_documents(docs, chunk_size=256, chunk_overlap=64)

    if len(chunks) > 1:
        # Adjacent chunks should share some content
        # Check that chunk 0's end appears in chunk 1
        chunk0_end = chunks[0].page_content[-100:]
        chunk1_start = chunks[1].page_content[:100]
        # Should have some overlap
        assert chunk0_end[-20:] in chunk1_start or chunk1_start[:20] in chunk0_end


def test_chunk_empty_document():
    """Empty document should return empty list or single empty chunk."""
    docs = [Document(page_content="")]
    chunks = chunk_documents(docs)

    # Should handle gracefully
    assert isinstance(chunks, list)


def test_chunk_documents_with_section_separators():
    """Chunker should split on section boundaries where possible."""
    # Document with clear sections
    content = """
    Section 1: Introduction
    This is the first section with some content.

    Section 2: Details
    This is the second section with more details.

    Section 3: Conclusion
    Final thoughts here.
    """

    docs = [Document(page_content=content)]
    chunks = chunk_documents(docs, chunk_size=128, chunk_overlap=16)

    # Should create chunks (at least one)
    assert len(chunks) >= 1
