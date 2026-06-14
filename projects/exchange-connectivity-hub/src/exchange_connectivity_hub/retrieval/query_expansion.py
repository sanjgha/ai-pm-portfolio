"""Generic HKEX domain query expansion for the rerank step (LIN-137).

The reranker scores (query, document) pairs. When a question uses different
surface terminology than the answer chunk (e.g. "closing auction" vs the chunk's
"CAS" / "random closing"), the reranker can under-score a chunk that is in fact
the answer. Appending generic domain synonyms to the rerank query closes that gap.

The term map keys on *general HKEX/markets terminology only* — never on the exact
phrasing of any eval question — so it helps any HKEX query, not just the ones we
are tuning against.
"""

# phrase (lowercased) -> synonyms to append when the phrase appears in the query.
_TERM_EXPANSIONS: dict[str, list[str]] = {
    "closing auction": ["CAS", "random closing", "reference price"],
    "cas": ["closing auction session", "random closing"],
    "lot size": ["board lot", "minimum shares"],
    "board lot": ["lot size", "minimum shares"],
    "tick size": ["spread table", "minimum price increment"],
    "vcm": ["volatility control mechanism"],
    "volatility control": ["VCM"],
    "short selling": ["designated securities", "uptick rule"],
    "settlement": ["clearing cycle"],
}


def expand_query(query: str) -> str:
    """Append generic HKEX domain synonyms for any known terms found in the query.

    Matching is case-insensitive. If no known term is present, the query is
    returned unchanged. The original query text is always preserved verbatim at
    the start so the reranker still sees the user's wording.
    """
    lowered = query.lower()
    additions: list[str] = []
    for term, synonyms in _TERM_EXPANSIONS.items():
        if term in lowered:
            additions.extend(synonyms)

    if not additions:
        return query
    return query + " " + " ".join(additions)
