"""Tests for generic HKEX rerank-query expansion (LIN-137)."""

from exchange_connectivity_hub.retrieval.query_expansion import expand_query


def test_expand_query_appends_synonyms_for_known_term():
    """A query containing a known domain term gets its synonyms appended."""
    out = expand_query("How does the HKSE closing auction work?")
    assert out.startswith("How does the HKSE closing auction work?")
    lowered = out.lower()
    assert "cas" in lowered
    assert "random closing" in lowered


def test_expand_query_is_case_insensitive():
    """Term matching ignores case."""
    out = expand_query("What is the minimum LOT SIZE for HKSE?")
    assert "board lot" in out.lower()


def test_expand_query_no_match_returns_original():
    """A query with no known terms is returned unchanged."""
    q = "What time is lunch?"
    assert expand_query(q) == q


def test_expand_query_does_not_duplicate_original():
    """Original query text is preserved exactly once at the start."""
    q = "tick size rules"
    out = expand_query(q)
    assert out.count(q) == 1
