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


def test_rag_chain_returns_contexts_with_content():
    """Chain should return contexts key containing actual chunk text, not metadata strings."""
    if not os.getenv("VOYAGE_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("API keys not set")

    chain = create_rag_chain()
    result = chain.invoke(
        {
            "question": "What are the trading hours for HKSE morning session?",
            "exchange_filter": "HKSE",
        }
    )

    assert "contexts" in result, "chain must return a 'contexts' key for RAGAS eval"
    assert isinstance(result["contexts"], list)
    assert len(result["contexts"]) > 0, "contexts must be non-empty"
    # Each context must be actual text content, not a metadata string
    for ctx in result["contexts"]:
        assert isinstance(ctx, str)
        assert not ctx.startswith("[Source:"), "contexts must contain chunk text, not metadata"
        assert len(ctx) > 50, "contexts must contain substantive text"
