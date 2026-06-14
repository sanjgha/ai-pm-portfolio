"""Integration test for RAG chain."""

import os
import pytest
from exchange_connectivity_hub.retrieval.rag_chain import create_rag_chain


def test_rag_chain_returns_answer_with_sources():
    """Chain should return answer with source citations."""
    if not os.getenv("VOYAGE_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("API keys not set")

    chain = create_rag_chain()

    result = chain.invoke(
        {
            "question": "What is the minimum lot size?",
            "exchange_filter": "SGX",
        }
    )

    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
