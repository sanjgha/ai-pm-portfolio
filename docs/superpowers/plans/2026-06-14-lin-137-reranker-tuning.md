# LIN-137 Reranker / Ranking Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift `context_precision` for the HKSE "CAS closing auction" and "minimum lot size" questions by tuning the ranking stage (config-driven ensemble weights, reranker query expansion, sweepable via CLI), without regressing the questions that already pass.

**Architecture:** Three small, config-gated changes plus a measurement loop. (1) Ensemble weights become config-driven (mirrors LIN-136's chunk params). (2) A new `query_expansion` module enriches the rerank query with generic HKEX domain synonyms; the reranker calls it when a config flag is on. (3) `run_eval.py` gains CLI flags that deep-merge overrides into the cached config so each tuning run is one fresh-process command. The reranker reads the expansion flag and `create_hybrid_retriever` reads weights from config, so `rag_chain.py` needs **no** change. The eval then surfaces a per-question CP table, and a one-factor-at-a-time (OFAT) sweep selects the best config.

**Tech Stack:** Python 3.10, LangChain (`EnsembleRetriever`, `BM25Retriever`, Chroma), Voyage `rerank-2.5`, RAGAS 0.2.x, pytest + unittest.mock, ruff.

**Spec:** `docs/superpowers/specs/2026-06-14-lin-137-reranker-tuning-design.md`

**Working directory for all paths below:** `projects/exchange-connectivity-hub/` (the project package, run its own venv/tests — not the root ones).

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `config.yaml` | Add `bm25_weight`, `vector_weight`, `rerank_query_expansion` keys | Modify |
| `src/exchange_connectivity_hub/config.py` | Add `apply_config_overrides()` deep-merge helper | Modify |
| `src/exchange_connectivity_hub/retrieval/hybrid_retriever.py` | Read weights from config; accept explicit overrides | Modify |
| `src/exchange_connectivity_hub/retrieval/query_expansion.py` | Generic HKEX rerank-query expansion (term map + `expand_query`) | Create |
| `src/exchange_connectivity_hub/retrieval/reranker.py` | Call `expand_query` when config flag on | Modify |
| `evals/run_eval.py` | CLI override flags + per-question CP table | Modify |
| `tests/unit/test_hybrid_retriever.py` | Update `MOCK_CONFIG` + weights test; add config-weights test | Modify |
| `tests/unit/test_query_expansion.py` | Tests for expansion | Create |
| `tests/unit/test_reranker.py` | Add expansion wiring tests | Modify |
| `tests/unit/test_config.py` | Add `apply_config_overrides` tests | Modify |
| `tests/unit/test_eval_overrides.py` | Tests for `build_overrides` + per-question CP table | Create |

---

## Task 0: Environment setup & green baseline

**Files:** none (environment only)

- [ ] **Step 1: Install the project package + dev deps in the worktree**

The worktree is a fresh checkout; the project uses its own venv. From the repo root of the worktree:

```bash
cd projects/exchange-connectivity-hub
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 2: Run the unit test suite to confirm a green baseline**

Run: `python -m pytest tests/unit -q`
Expected: all tests PASS (this is the pre-change baseline; new failures later are attributable to our changes).

If any test fails here, STOP and report — do not start implementing on a red baseline.

---

## Task 1: Make ensemble weights config-driven

**Files:**
- Modify: `config.yaml`
- Modify: `src/exchange_connectivity_hub/retrieval/hybrid_retriever.py`
- Test: `tests/unit/test_hybrid_retriever.py`

- [ ] **Step 1: Add config keys**

In `config.yaml`, under `retrieval:`, add three keys (keep existing keys):

```yaml
retrieval:
  top_k: 20              # Number of chunks to retrieve before reranking
  rerank_top_n: 5        # Number of chunks to keep after reranking
  rerank_enabled: true   # Enable Voyage rerank-2.5
  bm25_weight: 0.4       # EnsembleRetriever weight for BM25 (keyword) retriever
  vector_weight: 0.6     # EnsembleRetriever weight for dense vector retriever
  rerank_query_expansion: false  # Enrich rerank query with HKEX domain synonyms
```

- [ ] **Step 2: Update the shared test config and rewrite the existing weights test, and add a config-weights test**

In `tests/unit/test_hybrid_retriever.py`, update `MOCK_CONFIG` so `retrieval` carries the new weight keys:

```python
MOCK_CONFIG = {
    "vector_store": {
        "persist_directory": "/tmp/test_chroma",
        "collection_name": "test_collection",
    },
    "models": {"embedding": "voyage-finance-2"},
    "retrieval": {"top_k": 5, "bm25_weight": 0.4, "vector_weight": 0.6},
}
```

Replace the existing `test_hybrid_retriever_ensemble_weights` (it used an `importlib.reload` hack for the old module constants — no longer needed) with these two tests:

```python
def test_hybrid_retriever_weights_from_config():
    """Ensemble weights come from config, not hardcoded constants."""
    from langchain_classic.retrievers import EnsembleRetriever

    custom_cfg = {
        "vector_store": {
            "persist_directory": "/tmp/test_chroma",
            "collection_name": "test_collection",
        },
        "models": {"embedding": "voyage-finance-2"},
        "retrieval": {"top_k": 5, "bm25_weight": 0.3, "vector_weight": 0.7},
    }

    with (
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_config") as mock_cfg,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.chromadb") as mock_chroma,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.Chroma") as mock_chroma_lc,
    ):
        mock_cfg.return_value = custom_cfg

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["Some exchange rule content"],
            "metadatas": [{"exchange": "HKSE", "source_filename": "test.pdf", "page_number": 1}],
            "ids": ["id_0"],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = _StubRetriever()
        mock_chroma_lc.return_value = mock_vs

        from exchange_connectivity_hub.retrieval.hybrid_retriever import create_hybrid_retriever

        retriever = create_hybrid_retriever()

        assert isinstance(retriever, EnsembleRetriever)
        assert retriever.weights == [0.3, 0.7]


def test_hybrid_retriever_explicit_weights_override_config():
    """Explicit weight args take precedence over config."""
    from langchain_classic.retrievers import EnsembleRetriever

    with (
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_config") as mock_cfg,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.chromadb") as mock_chroma,
        patch("exchange_connectivity_hub.retrieval.hybrid_retriever.Chroma") as mock_chroma_lc,
    ):
        mock_cfg.return_value = MOCK_CONFIG  # config says 0.4 / 0.6

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["Some exchange rule content"],
            "metadatas": [{"exchange": "HKSE", "source_filename": "test.pdf", "page_number": 1}],
            "ids": ["id_0"],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = _StubRetriever()
        mock_chroma_lc.return_value = mock_vs

        from exchange_connectivity_hub.retrieval.hybrid_retriever import create_hybrid_retriever

        retriever = create_hybrid_retriever(bm25_weight=0.5, vector_weight=0.5)

        assert isinstance(retriever, EnsembleRetriever)
        assert retriever.weights == [0.5, 0.5]
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `python -m pytest tests/unit/test_hybrid_retriever.py -q`
Expected: `test_hybrid_retriever_weights_from_config` FAILS (`assert [0.4, 0.6] == [0.3, 0.7]`) and `test_hybrid_retriever_explicit_weights_override_config` FAILS — current code uses the hardcoded constants and ignores both config and args.

- [ ] **Step 4: Implement config-driven weights in `hybrid_retriever.py`**

Remove the module constants `_BM25_WEIGHT` / `_VECTOR_WEIGHT` and the two lines defining them. Update the signature and body:

```python
def create_hybrid_retriever(
    *,
    exchange_filter: str | None = None,
    top_k: int | None = None,
    bm25_weight: float | None = None,
    vector_weight: float | None = None,
) -> EnsembleRetriever:
    """Create a hybrid BM25 + dense vector retriever via Reciprocal Rank Fusion.

    Solves terminology mismatch: BM25 catches exact keyword hits (e.g. "circuit breaker"
    in VCM consultation doc) that pure vector search misses due to semantic drift.

    Args:
        exchange_filter: Optional exchange code to filter both retrievers (e.g. "HKSE")
        top_k: Documents per retriever before fusion (uses config default if None)
        bm25_weight: RRF weight for BM25 (uses config ``retrieval.bm25_weight`` if None)
        vector_weight: RRF weight for vector search (uses config ``retrieval.vector_weight`` if None)

    Returns:
        EnsembleRetriever fusing BM25 and vector results by the configured weights
    """
    config = get_config()

    if top_k is None:
        top_k = config["retrieval"]["top_k"]
    # Config-driven so ranking can be tuned without code changes (LIN-137).
    # Vector-dominant by default (BM25 0.4 / vector 0.6): BM25 rescues keyword
    # mismatches. Explicit args always take precedence.
    if bm25_weight is None:
        bm25_weight = config["retrieval"]["bm25_weight"]
    if vector_weight is None:
        vector_weight = config["retrieval"]["vector_weight"]
```

Then update the final return to use the resolved variables:

```python
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )
```

- [ ] **Step 5: Run the full retriever test file to verify green**

Run: `python -m pytest tests/unit/test_hybrid_retriever.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add config.yaml \
  src/exchange_connectivity_hub/retrieval/hybrid_retriever.py \
  tests/unit/test_hybrid_retriever.py
git commit -m "feat(retrieval): make EnsembleRetriever weights config-driven [LIN-137]"
```

---

## Task 2: Query expansion module

**Files:**
- Create: `src/exchange_connectivity_hub/retrieval/query_expansion.py`
- Test: `tests/unit/test_query_expansion.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_query_expansion.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_query_expansion.py -q`
Expected: FAIL with `ModuleNotFoundError: ... query_expansion`.

- [ ] **Step 3: Implement `query_expansion.py`**

Create `src/exchange_connectivity_hub/retrieval/query_expansion.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/unit/test_query_expansion.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/exchange_connectivity_hub/retrieval/query_expansion.py \
  tests/unit/test_query_expansion.py
git commit -m "feat(retrieval): add generic HKEX query expansion for reranker [LIN-137]"
```

---

## Task 3: Wire query expansion into the reranker

**Files:**
- Modify: `src/exchange_connectivity_hub/retrieval/reranker.py`
- Test: `tests/unit/test_reranker.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_reranker.py`:

```python
def test_rerank_uses_expanded_query_when_enabled():
    """With expansion on, the query sent to Voyage is expanded."""
    docs = [Document(page_content=f"Content {i}") for i in range(5)]

    with (
        patch("exchange_connectivity_hub.retrieval.reranker.VoyageClient") as mock_voyage,
        patch("exchange_connectivity_hub.retrieval.reranker.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.reranker.get_config") as mock_cfg,
    ):
        mock_cfg.return_value = {"models": {"rerank": "rerank-2.5"}}
        mock_client = MagicMock()
        mock_voyage.return_value = mock_client
        mock_client.rerank.return_value = MagicMock(
            results=[MagicMock(index=i) for i in range(5)]
        )

        rerank_documents(
            docs,
            query="How does the HKSE closing auction work?",
            top_n=5,
            enabled=True,
            expand_query=True,
        )

        sent_query = mock_client.rerank.call_args.kwargs["query"]
        assert "closing auction" in sent_query  # original preserved
        assert "CAS" in sent_query  # synonym appended


def test_rerank_uses_original_query_when_expansion_disabled():
    """With expansion off, the original query is sent unchanged."""
    docs = [Document(page_content=f"Content {i}") for i in range(5)]

    with (
        patch("exchange_connectivity_hub.retrieval.reranker.VoyageClient") as mock_voyage,
        patch("exchange_connectivity_hub.retrieval.reranker.get_voyage_api_key"),
        patch("exchange_connectivity_hub.retrieval.reranker.get_config") as mock_cfg,
    ):
        mock_cfg.return_value = {"models": {"rerank": "rerank-2.5"}}
        mock_client = MagicMock()
        mock_voyage.return_value = mock_client
        mock_client.rerank.return_value = MagicMock(
            results=[MagicMock(index=i) for i in range(5)]
        )

        original = "How does the HKSE closing auction work?"
        rerank_documents(docs, query=original, top_n=5, enabled=True, expand_query=False)

        assert mock_client.rerank.call_args.kwargs["query"] == original
```

Note: these tests patch `get_config` in the reranker so the `rerank_model` lookup and the expansion-flag fallback both work without a real config file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_reranker.py -q`
Expected: the two new tests FAIL with `TypeError: rerank_documents() got an unexpected keyword argument 'expand_query'`.

- [ ] **Step 3: Implement expansion wiring in `reranker.py`**

Add the import near the top:

```python
from exchange_connectivity_hub.retrieval.query_expansion import expand_query as _expand_query
```

Update the signature and add resolution + the expansion call. The function becomes:

```python
def rerank_documents(
    docs: list[Any],
    *,
    query: str,
    top_n: int,
    enabled: bool | None = None,
    expand_query: bool | None = None,
) -> list[Any]:
    """Rerank documents using Voyage AI rerank API.

    Args:
        docs: List of retrieved Documents
        query: Original query string
        top_n: Number of top documents to keep after reranking
        enabled: Whether reranking is enabled (uses config if None)
        expand_query: Whether to enrich the rerank query with HKEX domain
            synonyms (uses config ``retrieval.rerank_query_expansion`` if None)

    Returns:
        Reranked list of Documents (length = top_n)
    """
    if not docs:
        return []

    # Check if reranking is enabled
    if enabled is None:
        config = get_config()
        enabled = config["retrieval"]["rerank_enabled"]

    if not enabled:
        # Return original docs up to top_n
        return docs[:top_n]

    # Get rerank model from config
    config = get_config()
    rerank_model = config["models"]["rerank"]

    # Resolve query expansion from config when not explicitly set (LIN-137)
    if expand_query is None:
        expand_query = config["retrieval"]["rerank_query_expansion"]

    rerank_query = _expand_query(query) if expand_query else query

    # Initialize Voyage client
    client = VoyageClient(api_key=get_voyage_api_key())

    # Extract document contents
    doc_contents = [doc.page_content for doc in docs]

    # Call rerank API
    rerank_results = client.rerank(
        query=rerank_query,
        documents=doc_contents,
        model=rerank_model,
        top_k=top_n,
    )

    # Reorder documents based on rerank results
    reranked_docs = []
    for result in rerank_results.results:
        original_index = result.index
        reranked_docs.append(docs[original_index])

    return reranked_docs
```

- [ ] **Step 4: Run the reranker tests to verify green**

Run: `python -m pytest tests/unit/test_reranker.py -q`
Expected: all PASS (existing tests still pass — they pass `enabled=False` or mock the client; the `enabled=True` existing tests do not patch `get_config`, so confirm they still pass; if `test_rerank_documents_returns_subset` now needs the expansion flag, it resolves via the real config where `rerank_query_expansion: false`, so the original query is used and behavior is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/exchange_connectivity_hub/retrieval/reranker.py tests/unit/test_reranker.py
git commit -m "feat(retrieval): apply query expansion in rerank step when enabled [LIN-137]"
```

---

## Task 4: `apply_config_overrides` helper

**Files:**
- Modify: `src/exchange_connectivity_hub/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config.py`:

```python
def test_apply_config_overrides_deep_merges(monkeypatch):
    """Overrides update nested keys without dropping siblings."""
    import exchange_connectivity_hub.config as config_mod

    base = {"retrieval": {"top_k": 20, "rerank_top_n": 5}, "models": {"rerank": "rerank-2.5"}}
    monkeypatch.setattr(config_mod, "_config", base)

    config_mod.apply_config_overrides({"retrieval": {"rerank_top_n": 8}})

    cfg = config_mod.get_config()
    assert cfg["retrieval"]["rerank_top_n"] == 8  # overridden
    assert cfg["retrieval"]["top_k"] == 20  # sibling preserved
    assert cfg["models"]["rerank"] == "rerank-2.5"  # other section preserved


def test_apply_config_overrides_adds_new_key(monkeypatch):
    """Overrides can introduce a new nested key."""
    import exchange_connectivity_hub.config as config_mod

    base = {"retrieval": {"top_k": 20}}
    monkeypatch.setattr(config_mod, "_config", base)

    config_mod.apply_config_overrides({"retrieval": {"bm25_weight": 0.3}})

    assert config_mod.get_config()["retrieval"]["bm25_weight"] == 0.3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_config.py -q`
Expected: FAIL with `AttributeError: module ... has no attribute 'apply_config_overrides'`.

- [ ] **Step 3: Implement the helper in `config.py`**

Add after `get_config()`:

```python
def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Recursively merge ``overrides`` into ``base`` in place."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def apply_config_overrides(overrides: dict[str, Any]) -> None:
    """Deep-merge ``overrides`` into the cached config.

    Used by tuning entry points (e.g. ``run_eval.py`` CLI flags) to adjust
    ranking parameters for a single process run without editing config.yaml.
    Because every module reads the same cached dict via ``get_config()``, the
    merge propagates everywhere downstream.
    """
    config = get_config()
    _deep_merge(config, overrides)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_config.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/exchange_connectivity_hub/config.py tests/unit/test_config.py
git commit -m "feat(config): add apply_config_overrides for tuning runs [LIN-137]"
```

---

## Task 5: `run_eval.py` CLI overrides + per-question CP table

**Files:**
- Modify: `evals/run_eval.py`
- Test: `tests/unit/test_eval_overrides.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_eval_overrides.py`:

```python
"""Tests for run_eval CLI overrides and per-question CP table (LIN-137)."""

import sys
from pathlib import Path
from types import SimpleNamespace


def _import_run_eval():
    sys.path.insert(0, str(Path(__file__).parents[2] / "evals"))
    import run_eval

    return run_eval


def test_build_overrides_maps_set_args_only():
    """Only args the user set appear in the overrides dict."""
    run_eval = _import_run_eval()
    args = SimpleNamespace(
        rerank_top_n=8,
        top_k=None,
        bm25_weight=0.3,
        vector_weight=0.7,
        rerank_query_expansion=True,
    )
    overrides = run_eval.build_overrides(args)
    assert overrides == {
        "retrieval": {
            "rerank_top_n": 8,
            "bm25_weight": 0.3,
            "vector_weight": 0.7,
            "rerank_query_expansion": True,
        }
    }


def test_build_overrides_empty_when_nothing_set():
    """No flags set -> empty overrides (run uses config defaults)."""
    run_eval = _import_run_eval()
    args = SimpleNamespace(
        rerank_top_n=None,
        top_k=None,
        bm25_weight=None,
        vector_weight=None,
        rerank_query_expansion=None,
    )
    assert run_eval.build_overrides(args) == {}


def test_build_per_question_cp_pairs_questions_and_scores():
    """Per-question CP table pairs user_input with context_precision."""
    run_eval = _import_run_eval()
    scores = {
        "user_input": ["Q1", "Q2"],
        "context_precision": [0.2, 0.95],
    }
    table = run_eval.build_per_question_cp(scores)
    assert table == [
        {"question": "Q1", "context_precision": 0.2},
        {"question": "Q2", "context_precision": 0.95},
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_eval_overrides.py -q`
Expected: FAIL with `AttributeError: module 'run_eval' has no attribute 'build_overrides'`.

- [ ] **Step 3: Implement helpers + CLI wiring in `run_eval.py`**

Add the import for the override helper near the other `exchange_connectivity_hub` imports:

```python
from exchange_connectivity_hub.config import (  # noqa: E402
    apply_config_overrides,
    get_anthropic_api_key,
    get_voyage_api_key,
)
```

(Replace the existing `from exchange_connectivity_hub.config import get_anthropic_api_key, get_voyage_api_key` line.)

Add two module-level helpers (place them above `main()`):

```python
def build_overrides(args) -> dict:
    """Build a config-override dict from parsed CLI args (only set values)."""
    retrieval: dict = {}
    if args.rerank_top_n is not None:
        retrieval["rerank_top_n"] = args.rerank_top_n
    if args.top_k is not None:
        retrieval["top_k"] = args.top_k
    if args.bm25_weight is not None:
        retrieval["bm25_weight"] = args.bm25_weight
    if args.vector_weight is not None:
        retrieval["vector_weight"] = args.vector_weight
    if args.rerank_query_expansion is not None:
        retrieval["rerank_query_expansion"] = args.rerank_query_expansion
    return {"retrieval": retrieval} if retrieval else {}


def build_per_question_cp(scores: dict) -> list[dict]:
    """Pair each question with its context_precision score."""
    questions = scores.get("user_input", [])
    cps = scores.get("context_precision", [])
    return [
        {"question": q, "context_precision": cp}
        for q, cp in zip(questions, cps)
    ]
```

In `run_evaluation`, after `scores = result.to_pandas().to_dict(orient="list")`, add the per-question table to the returned `results` dict. Locate the `results = { ... }` dict and add one key:

```python
        "per_question_scores": scores,
        "per_question_cp": build_per_question_cp(scores),
    }
```

In `print_results`, after the `Metrics:` block, add a per-question CP table:

```python
    per_q = results.get("per_question_cp", [])
    if per_q:
        print("\nPer-question context_precision:")
        for row in per_q:
            print(f"  {row['context_precision']:.2f}  {row['question'][:70]}")
```

In `main()`, extend the argument parser and apply overrides before running. Replace the parser/args block with:

```python
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Harness")
    parser.add_argument(
        "--exchanges",
        nargs="+",
        metavar="EXCHANGE",
        help="Limit eval to these exchange codes, e.g. --exchanges HKSE SGX",
    )
    parser.add_argument("--rerank-top-n", type=int, default=None, dest="rerank_top_n")
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--bm25-weight", type=float, default=None, dest="bm25_weight")
    parser.add_argument("--vector-weight", type=float, default=None, dest="vector_weight")
    parser.add_argument(
        "--rerank-query-expansion",
        dest="rerank_query_expansion",
        action="store_true",
        default=None,
        help="Enable HKEX query expansion in the rerank step",
    )
    parser.add_argument(
        "--no-rerank-query-expansion",
        dest="rerank_query_expansion",
        action="store_false",
        help="Disable HKEX query expansion in the rerank step",
    )
    args = parser.parse_args()

    overrides = build_overrides(args)
    if overrides:
        apply_config_overrides(overrides)
        print(f"Config overrides applied: {overrides}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_eval_overrides.py -q`
Expected: all PASS.

- [ ] **Step 5: Run the whole unit suite to confirm no regressions**

Run: `python -m pytest tests/unit -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add evals/run_eval.py tests/unit/test_eval_overrides.py
git commit -m "feat(eval): add ranking CLI overrides and per-question CP table [LIN-137]"
```

---

## Task 6: Re-measure baseline + run the OFAT sweep (measurement, not code)

**Prerequisites (the worktree is a fresh checkout):**
- API keys exported in the shell: `export ANTHROPIC_API_KEY=... VOYAGE_API_KEY=...`
- The ingested ChromaDB must be reachable at the configured `persist_directory`
  (`data/chroma_db`, relative to `projects/exchange-connectivity-hub/`). It exists
  only in the primary checkout, so link it into the worktree once:

```bash
# from projects/exchange-connectivity-hub/ in the worktree
ln -s /home/ubuntu/projects/ai-pm-portfolio/projects/exchange-connectivity-hub/data/chroma_db data/chroma_db
```

(The Chroma store is read-only for eval; linking the primary copy is safe.)

- [ ] **Step 1: Capture the clean 512/64 baseline (HKSE)**

Run (committed defaults — no override flags):

```bash
python evals/run_eval.py --exchanges HKSE
```

Record the printed per-question CP table. Copy the saved JSON to a stable name so later runs don't overwrite it:

```bash
cp evals/results/eval_2026-06-14.json evals/results/lin137_baseline.json
```

Identify the set of **passing** questions (baseline CP ≥ 0.75); these are the regression guard set.

- [ ] **Step 2: Run A — query expansion (the primary lever)**

```bash
python evals/run_eval.py --exchanges HKSE --rerank-query-expansion
cp evals/results/eval_2026-06-14.json evals/results/lin137_A_expand.json
```
Record per-question CP. Expectation: CAS + lot-size rise.

- [ ] **Step 3: Run B — larger rerank window**

```bash
python evals/run_eval.py --exchanges HKSE --rerank-top-n 8
cp evals/results/eval_2026-06-14.json evals/results/lin137_B_topn8.json
```
Record per-question CP. Watch for CP drops on the passing set (extra chunks dilute precision).

- [ ] **Step 4: Run C — ensemble weight shift (control)**

```bash
python evals/run_eval.py --exchanges HKSE --bm25-weight 0.3 --vector-weight 0.7
cp evals/results/eval_2026-06-14.json evals/results/lin137_C_weights.json
```
Record per-question CP. Expectation: roughly flat — record the result either way.

- [ ] **Step 5: Run D — combine the winners**

Combine whichever of A/B/C produced net gains (e.g. expansion on + the best top_n) in one run, and copy to `evals/results/lin137_D_combined.json`. Example if A wins alone and B helped lot-size:

```bash
python evals/run_eval.py --exchanges HKSE --rerank-query-expansion --rerank-top-n 8
cp evals/results/eval_2026-06-14.json evals/results/lin137_D_combined.json
```

- [ ] **Step 6: Build the per-question CP comparison table**

Assemble a markdown table: rows = the 10 HKSE answerable questions, columns = baseline / A / B / C / D, cells = CP. Mark CAS and lot-size, and flag any passing-set cell that dropped by **> 0.05**. Save it to `docs/exchange-connectivity-hub/lin137-cp-sweep.md`.

- [ ] **Step 7: Pick the winning config and set it as the committed default**

Winner = best CAS + lot-size gain with **no > 0.05 drop** on the passing set. Edit `config.yaml` so the committed defaults reflect the winner (e.g. set `rerank_query_expansion: true`, and `rerank_top_n` if B/D won). If only the expansion flag wins, that's the only change.

- [ ] **Step 8: Commit the results table and config default**

```bash
git add config.yaml docs/exchange-connectivity-hub/lin137-cp-sweep.md
git commit -m "feat(retrieval): set tuned ranking defaults from LIN-137 sweep [LIN-137]"
```

(The `evals/results/lin137_*.json` files are gitignored data artifacts — do not commit them unless the project tracks eval results; confirm against `.gitignore`.)

---

## Task 7: Final verification & Definition of Done

**Files:** none (verification only)

- [ ] **Step 1: Full project test suite**

Run: `python -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 2: Lint + format check**

Run: `ruff check src/ tests/ evals/ && ruff format --check src/ tests/ evals/`
Expected: clean (run `ruff format src/ tests/ evals/` to fix if needed).

- [ ] **Step 3: Confirm the Definition of Done against the recorded table**

- [ ] CAS + lot-size CP improved vs the re-measured 512/64 baseline (0.20 / 0.00).
- [ ] No regression > 0.05 on the questions that passed at baseline.
- [ ] Per-question CP table recorded (`docs/exchange-connectivity-hub/lin137-cp-sweep.md`).

If CAS/lot-size did **not** improve with any swept config, STOP and report — the spec's query-expansion hypothesis would be falsified, and the issue needs a rethink (do not force a config that regresses the passing set just to move CAS/lot).

- [ ] **Step 4: Push the branch and open the PR (only when asked)**

Per project workflow, branch into `main` via PR. Do not mark LIN-137 Done until the branch is merged into `main` (see memory: "Verify merged before Linear Done").

---

## Self-Review notes

- **Spec coverage:** config-driven weights (Task 1), query expansion module + wiring (Tasks 2–3), CLI overrides (Tasks 4–5), per-question CP table (Task 5), baseline re-measurement + OFAT sweep + regression rule (Task 6), DoD check (Task 7). All spec sections map to a task.
- **No `rag_chain.py` change:** confirmed — weights resolve inside `create_hybrid_retriever`, expansion resolves inside `rerank_documents`, both from the cached config the CLI overrides mutate.
- **Type/name consistency:** `apply_config_overrides`, `build_overrides`, `build_per_question_cp`, `expand_query`, and the `bm25_weight` / `vector_weight` / `rerank_query_expansion` keys are used identically across all tasks.
```
