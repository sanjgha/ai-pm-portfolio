"""Document chunking using LangChain's token-aware splitter."""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: list[Any],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Any]:
    """Split documents into chunks using token-aware splitting.

    Uses tiktoken encoder for accurate token counting, ensuring
    chunks are close to the target token size (not character count).

    Args:
        docs: List of LangChain Documents from loader
        chunk_size: Target chunk size in tokens (default: 512)
        chunk_overlap: Overlap between chunks in tokens (default: 64)

    Returns:
        List of chunked Documents with preserved metadata
    """
    # Create splitter with tiktoken encoder for token-accurate splitting
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",  # GPT-4 tokenizer (standard for LangChain)
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",  # Multiple blank lines (major sections)
            "\n\n",  # Blank lines (subsections)
            "\n",  # Single newlines
            ". ",  # Sentence boundaries
            "! ",  # Exclamation sentences
            "? ",  # Question sentences
            "; ",  # Semicolons
            ", ",  # Commas (last resort)
            " ",  # Spaces (very last resort)
            "",  # Character level (absolute fallback)
        ],
    )

    all_chunks = []
    for doc in docs:
        # Split this document
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    return all_chunks
