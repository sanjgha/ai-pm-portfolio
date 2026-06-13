# Design: Exchange Connectivity Intelligence Hub

**Date:** 2026-06-13
**Status:** Approved — ready for implementation
**Scope:** Full P0 (ingest + RAG chain + API + UI + eval harness + doc versioning)
**Source PRD:** `docs/exchange-connectivity-hub/exchange-connectivity-hub-spec.md`

---

## Problem

Engineers and BAs manually cross-reference hundreds of pages of Asian exchange PDFs (HKSE, SGX, TSE, OSE, Bursa, IDX) to answer connectivity and market microstructure questions. This takes 20–40 minutes per query and risks acting on stale docs. This project delivers a RAG system that answers those questions in under 30 seconds, cites sources, and detects when docs have changed.

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Project location | `projects/exchange-connectivity-hub/` | Monorepo convention over spec's flat-root layout |
| Package layout | `src/exchange_connectivity_hub/` | Monorepo convention over spec's `requirements.txt` flat layout |
| Embeddings | Voyage AI `voyage-finance-2` | Domain-tuned for financial/technical text |
| LLM | Claude `claude-sonnet-4-6` via Anthropic SDK | Strong instruction-following for cite-or-refuse prompts |
| RAG chain | LangChain LCEL | Composable, LangSmith-ready for P1 |
| Vector store | ChromaDB (local persistent) | Zero-infra for MVP; same API surface as pgvector |
| Reranking | Voyage `rerank-2.5` (retrieve 20 → rerank → keep 5), `rerank_enabled` toggle | Lifts Context Precision on near-duplicate exchange text; same Voyage SDK/key; ablation-measurable |
| Chunking | 512 tokens / 64 overlap via `RecursiveCharacterTextSplitter.from_tiktoken_encoder` | Token-accurate (a char-based splitter would yield ~128 tokens); section-aware via recursive separators |
| Change detection | MD5 of normalized extracted text | Deterministic; catches content changes while ignoring cosmetic PDF byte churn (metadata timestamps, re-saves) |
| API | FastAPI | Lightweight, async, auto-docs |
| UI | Streamlit calling FastAPI over HTTP | Keeps UI/backend decoupled |
| Eval framework | RAGAS | Purpose-built for RAG quality measurement |

---

## Project Structure

```
projects/exchange-connectivity-hub/
├── pyproject.toml              # deps: langchain, chromadb, voyageai, anthropic, fastapi, streamlit, ragas
├── Makefile                    # make ci / test / ingest / serve-api / serve-ui / eval / check-updates
├── config.yaml                 # retrieval_top_k=20, rerank_top_n=5, rerank_enabled=true, rerank_model=rerank-2.5, staleness_days=60, model names, chroma_db path
├── .env.example                # VOYAGE_API_KEY, ANTHROPIC_API_KEY
├── src/
│   └── exchange_connectivity_hub/
│       ├── __init__.py
│       ├── ingest/
│       │   ├── loader.py           # PyMuPDF → LangChain Documents + metadata
│       │   ├── chunker.py          # RecursiveCharacterTextSplitter.from_tiktoken_encoder 512/64
│       │   ├── embedder.py         # Voyage embed + ChromaDB upsert
│       │   ├── update_checker.py   # MD5 hash vs doc_registry.json
│       │   └── pipeline.py         # Orchestrates load → chunk → embed
│       ├── retrieval/
│       │   ├── retriever.py        # ChromaDB cosine search + exchange metadata filter
│       │   ├── reranker.py         # Voyage rerank-2.5: reranks retrieval_top_k candidates → rerank_top_n
│       │   └── rag_chain.py        # LCEL: retriever | reranker | prompt | Claude | output parser
│       ├── api/
│       │   └── main.py             # FastAPI POST /query, GET /health
│       └── ui/
│           └── app.py              # Streamlit Q&A interface
├── data/
│   ├── raw/                    # Downloaded PDFs (gitignored if >10MB)
│   ├── chroma_db/              # ChromaDB persistent store (gitignored)
│   └── doc_registry.json       # Committed — version log per tracked doc
├── evals/
│   ├── golden_dataset.json     # 35 Q&A pairs: 30 answerable (10×3 exchanges) + 5 unanswerable (authored by Sanjeev)
│   ├── run_eval.py             # RAGAS eval script
│   └── results/                # gitignored — ephemeral score history
└── tests/
    ├── unit/                   # chunker, update_checker, reranker ordering (mocked Voyage), source-assembly-from-metadata (no API keys needed)
    └── integration/            # live ChromaDB + mocked Voyage/Claude responses
```

---

## Data Flow

### Ingest Path

```
PDF file
  → loader.py        PyMuPDF parses pages; attaches metadata:
                     { source_filename, exchange, doc_type, page_number,
                       ingested_at (ISO datetime), doc_version_hash (MD5 of normalized extracted text) }
  → chunker.py       RecursiveCharacterTextSplitter.from_tiktoken_encoder: 512 tokens, 64 overlap
                     (token-based, not character-based)
  → embedder.py      voyageai SDK → voyage-finance-2 → ChromaDB upsert
                     (persistent store at data/chroma_db/)
  → doc_registry.json  updated: current_hash, ingested_at, chunks_count appended
                        to version_history
```

### Query Path

```
POST /query { question, exchange_filter? }
  → rag_chain.py     Voyage embeds question (same model as ingest — consistent space)
  → retriever.py     ChromaDB cosine top-20 (retrieval_top_k);
                     if exchange_filter set → metadata pre-filter applied before search
  → reranker.py      Voyage rerank-2.5 reranks the 20 candidates against the question;
                     keep top rerank_top_n (5). Skipped when rerank_enabled=false.
                     Adds ~100–300ms (one API call) — within P95<5s budget.
  → prompt template  "Answer from context only. Cite source filename and page number.
                     Say 'I don't have enough information' if context is insufficient.
                     Never hallucinate."
  → Claude claude-sonnet-4-6  via LCEL chain
  → output parser    answer: str parsed from Claude (prose + inline citation markers only).
                     sources: [{ filename, exchange, page_number, ingested_at }] assembled from the
                     reranked chunks' metadata — NOT parsed from LLM text (prevents citation hallucination).
  → staleness check  If any source ingested_at > staleness_days (default 60):
                     prepend ⚠️ warning. Flags re-verification age ("last checked Nd ago"),
                     NOT confirmed obsolescence — the hash compare is the real currency check.
  ← { answer, sources, staleness_warning }
```

### Re-ingestion Path

```
make check-updates  (or auto-trigger after doc change detected)
  → update_checker.py
      For each doc in doc_registry.json:
        download source_url → extract & normalize text → compute MD5
        compare to current_hash
        hash unchanged → log "unchanged", skip
        hash changed   → delete ALL ChromaDB chunks for source_filename
                       → re-run ingest pipeline with new PDF
                       → append to version_history in doc_registry.json
                       → auto-trigger run_eval.py
                       → log alert if Faithfulness < 0.85 or Context Precision < 0.75

  make reingest doc=<filename>  (force re-ingest without URL check, for unstable URLs)
```

---

## API Contract

### `POST /query`

```json
// Request
{ "question": "What is the minimum lot size for SGX equities?", "exchange_filter": "SGX" }

// Response
{
  "answer": "The minimum lot size for SGX equities is 100 shares (Source: SGX_Market_Model_Guide_v4.pdf, p.12).",
  "sources": [
    { "filename": "SGX_Market_Model_Guide_v4.pdf", "exchange": "SGX", "page_number": 12, "ingested_at": "2026-06-13T10:00:00Z" }
  ],
  "staleness_warning": null
}
```

`sources` are derived from the metadata of the chunks actually used (post-rerank top-n), guaranteeing every citation points to a retrieved document.

### `GET /health`

```json
{ "status": "ok", "collection_count": 1842, "version": "0.1.0" }
```

Chain is instantiated once at startup (no per-request cold start on ChromaDB).

---

## Streamlit UI

Calls FastAPI over HTTP — UI does not import the chain directly.

- Free-text question input
- Exchange dropdown: All / SGX / HKSE / TSE (ingested exchanges only — OSE / Bursa / IDX deferred until ingested, P1; dropdown scoped to live data so filters never return empty)
- Answer block with inline citations (filename + page number)
- Amber banner when `staleness_warning` is present — copy reads "last re-verified [date] ([N]d ago)" (re-verification age, not a claim the doc is outdated)
- "Last ingested" timestamp shown per source in citation block

---

## Eval Harness

### Golden Dataset Schema (`evals/golden_dataset.json`)

```json
[
  {
    "question": "What is the minimum lot size for SGX equities?",
    "ground_truth": "100 shares",
    "source_doc": "SGX_Market_Model_Guide_v4.pdf",
    "page": 12,
    "exchange": "SGX"
  },
  {
    "question": "Does the HKSE support iceberg orders on ETDs?",
    "ground_truth": null,
    "source_doc": null,
    "page": null,
    "exchange": null
  }
]
```

- 35 Q&A pairs total = 30 answerable (10 per exchange: SGX, HKSE, TSE) + 5 unanswerable
- Topics: order types, lot sizes, trading hours, tick sizes, FIX tag specifics, error codes
- The 5 unanswerable questions (`ground_truth: null`) test refusal behaviour

### Pass Criteria

| Metric | Threshold |
|---|---|
| Context Precision | ≥ 0.75 |
| Faithfulness | ≥ 0.85 |
| Refusal rate on unanswerable Qs | ≥ 80% (4 of 5) |

> To support a true ≥90% gate, grow the unanswerable set to ≥10 (P1).
>
> **Statistical note:** at n≈30 answerable Qs these are directional gates, not statistical SLAs — the 95% CI on a ~0.75 mean is ≈ ±0.15, so treat sub-5% swings as noise. P1: grow to 100+ and track the trend over time rather than single-run pass/fail.

### Eval Triggers

- Manually before any PR that changes chunking, prompts, or model config
- Automatically after any doc re-ingestion
- Rerank ablation — run eval with `rerank_enabled` on/off to quantify the Context Precision delta and justify the added latency/cost

---

## Doc Registry Schema (`data/doc_registry.json`)

```json
{
  "SGX_Market_Model_Guide_v4.pdf": {
    "source_url": "https://www.sgx.com/.../market-model.pdf",
    "exchange": "SGX",
    "doc_type": "market_model",
    "current_hash": "a3f8c...",
    "version_history": [
      { "hash": "a3f8c...", "ingested_at": "2026-06-13T10:00:00Z", "chunks_count": 328 }
    ]
  }
}
```

`source_url: null` for docs where the exchange URL is unstable — manual re-ingest only.
File is committed to git (config, not secrets).

---

## Makefile Targets

```
make ci              lint + typecheck + unit tests
make test            pytest tests/
make ingest doc=X    ingest a single PDF from data/raw/
make ingest-all      ingest all PDFs registered in doc_registry.json
make check-updates   check all registry URLs for hash changes
make reingest doc=X  force re-ingest (skips hash check)
make serve-api       uvicorn api/main.py on :8000
make serve-ui        streamlit run ui/app.py on :8501
make eval            run RAGAS eval against live chain
```

---

## What Is NOT in This Design

- LangSmith tracing (P1 — deferred)
- Comparative query mode / "vs" detection (P1 — deferred)
- FIX tag lookup secondary collection (P1 — deferred)
- pgvector migration (P2 — deferred)
- Multi-language support (P2 — deferred)
- Streaming responses (P2 — deferred)
- Web UI auth / multi-tenancy (non-goal)

---

## Success Criteria

| Metric | Target |
|---|---|
| RAGAS Faithfulness | ≥ 0.85 |
| RAGAS Context Precision | ≥ 0.75 |
| Refusal rate on unanswerable Qs | ≥ 80% (4 of 5) |
| P95 query latency | < 5 seconds |
| Doc staleness detection | 100% accuracy on test set |
