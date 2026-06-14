# PRD: Exchange Connectivity Intelligence Hub

> **Version:** 0.2 — Monorepo Path Added
> **Author:** Sanjeev Gharde
> **Date:** 2026-06-11
> **Status:** Draft — Pending Engineering Review
> **Portfolio Project:** AI-PM Learning Portfolio (RAG Track)
> **Monorepo:** [github.com/sanjgha/ai-pm-portfolio](https://github.com/sanjgha/ai-pm-portfolio)
> **Project path:** `exchange-connectivity-hub/`

---

## Problem Statement

When connecting to or troubleshooting issues across Asian exchanges (HKSE, SGX, TSE, OSE, Bursa, IDX), engineers and BAs must manually cross-reference hundreds of pages of exchange market operation guides, FIX dialect specs, and trading rules PDFs. A query like "what is the SGX lot size for small-cap equities?" or "does TSE support iceberg orders on ETDs?" can take 20–40 minutes of manual PDF hunting — and there is no guarantee the document consulted is the current version.

This is a high-frequency pain point for teams onboarding new exchange connections, responding to production incidents, and navigating annual rulebook updates from exchanges. The risk is not just inefficiency: acting on stale or misread specs can cause trade rejections, erroneous compliance filings, or connectivity outages.

---

## Goals

1. **Reduce time-to-answer** for exchange connectivity and market microstructure questions from ~30 minutes (manual PDF search) to under 30 seconds.
2. **Improve answer reliability** by grounding every response in a specific, cited source document — eliminating memory-based guesses.
3. **Reduce stale-data risk** by building a hash-based document versioning and change-detection system, plus a re-verification-age warning that flags when a doc hasn't been re-checked recently.
4. **Establish a repeatable eval harness** so retrieval quality and answer faithfulness can be measured before and after any system change (doc update, chunk strategy change, model swap).
5. **Demonstrate RAG + Voyage embeddings + evals** as portfolio-ready, production-pattern AI skills.

---

## Non-Goals

| Non-Goal | Rationale |
|---|---|
| Real-time market data (prices, order books) | Out of scope for v1; requires live data feeds, not document Q&A |
| Automated trade order routing or execution | Compliance and liability risk; human-in-the-loop required |
| Proprietary internal connectivity specs | MVP uses publicly available exchange documentation only |
| Multi-language document support (Japanese, Chinese) | Adds translation complexity; deferred to v2 |
| Web UI with auth / multi-tenancy | Streamlit single-user MVP is sufficient for portfolio; scale later |

---

## User Stories

### Persona A — Exchange Connectivity Engineer

> Responsible for building and maintaining FIX connections to exchanges.

- As a connectivity engineer, I want to ask "what order types does SGX support for equities?" and get a direct answer with the source doc and page number, so I don't have to manually open and search a 200-page PDF.
- As a connectivity engineer, I want every answer to show which version of the rulebook it came from and when it was last ingested, so I know whether to trust it or verify against the exchange website.
- As a connectivity engineer, I want to ask comparative questions like "how does HKSE handle fat-finger limits vs SGX?" and get a side-by-side answer, so I can make informed design decisions during exchange onboarding.

### Persona B — Trading BA / Product Owner (Sanjeev)

> Writes business requirements and acceptance criteria for trading system changes.

- As a trading BA, I want to ask "what is the minimum tick size for SGX futures?" without context-switching to a PDF, so I can write accurate acceptance criteria in my BRD without a 30-minute interruption.
- As a trading BA, I want to receive a staleness warning when an answer is based on a document ingested more than 60 days ago, so I know to verify against the latest exchange circular.
- As a trading BA, I want to run a pre-defined set of eval questions after any doc update, so I can confirm the system's accuracy hasn't regressed.

### Persona C — Incident Responder

> Troubleshooting a live exchange connectivity issue under time pressure.

- As an incident responder, I want to type a free-form description of an error (e.g., "order rejected with reason code 337 on TSE") and get possible causes and relevant rulebook references in under 10 seconds, so I can diagnose issues faster.
- As an incident responder, I want to know if the relevant exchange document has been updated recently (potentially changing the spec I'm relying on), so I can escalate appropriately.

---

## Requirements

### P0 — Must Have (MVP)

#### 1. Document Ingestion Pipeline

**Description:** Ingest PDF exchange documentation for 3+ exchanges (SGX, HKSE, TSE to start) into a Voyage-embedded vector store.

**Acceptance Criteria:**
- [ ] LangChain `PyMuPDFLoader` parses PDFs; each chunk retains metadata: `{source_filename, exchange, doc_type, page_number, ingested_at, doc_version_hash}`
- [ ] Chunking strategy: 512 tokens with 64-token overlap, split on section boundaries where possible
- [ ] `doc_version_hash` is the MD5 of the **normalized extracted text** (not raw PDF bytes — avoids false positives from cosmetic byte churn); used for change detection (see P0.5)
- [ ] Each chunk is embedded using **Voyage AI** (`voyage-finance-2` model) via the `voyageai` Python SDK
- [ ] Embedded chunks stored in **ChromaDB** (local, persistent) with all metadata fields indexed

**Why Voyage over OpenAI embeddings?**
`voyage-finance-2` is a domain-tuned embedding model trained on financial and technical documents. In head-to-head benchmarks on financial corpora, it outperforms `text-embedding-3-small` on retrieval precision for jargon-heavy, structured text like exchange rulebooks and FIX specs — exactly our content type. It also embeds longer passages more coherently, which matters for multi-clause trading rules.

---

#### 2. RAG Query Chain

**Description:** A LangChain retrieval chain that takes a free-text question, retrieves the top-k most relevant chunks, and uses Claude claude-sonnet-4-6 to synthesise a cited answer.

**Acceptance Criteria:**
- [ ] Query is embedded via the same Voyage model used at ingest time (embedding consistency)
- [ ] Top-5 chunks retrieved by cosine similarity from ChromaDB
- [ ] Prompt template instructs LLM to: (a) answer from context only, (b) cite source + page number, (c) say "I don't have enough information" if context is insufficient — never hallucinate
- [ ] Response includes a `sources` block: `[{filename, exchange, page_number, ingested_at}]`
- [ ] If any retrieved chunk has `ingested_at` older than 60 days, response includes a ⚠️ staleness warning
- [ ] Chain exposed via a FastAPI `POST /query` endpoint accepting `{question: str, exchange_filter: str | None}`

---

#### 3. Streamlit UI (Demo Interface)

**Description:** Minimal question-answer interface for demo and portfolio purposes.

**Acceptance Criteria:**
- [ ] Text input for free-form question
- [ ] Optional exchange filter (dropdown: All, SGX, HKSE, TSE, OSE, Bursa, IDX)
- [ ] Answer displayed with inline source citations (filename + page)
- [ ] Staleness warnings rendered prominently (amber banner)
- [ ] "Last ingested" timestamp shown per source in the citation block

---

#### 4. Eval Harness (Golden Dataset + RAGAS)

**Description:** A reproducible evaluation framework to measure and track RAG quality over time.

**Why this matters:** Without evals, you cannot tell whether a chunking change, model swap, or doc update improved or broke retrieval quality. This is the difference between a demo and a production-grade system — and it is the thing most AI tutorials skip. Think of it like pre-trade risk checks: you do not deploy a change to a trading system without running your test suite first. Same principle here.

**Eval dimensions (RAGAS framework):**

```
┌─────────────────────┬──────────────────────────────────────────────────────────┐
│ Metric              │ What it measures                                          │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│ Context Precision   │ Of chunks retrieved, what fraction were actually relevant?│
│ Context Recall      │ Of relevant chunks, what fraction were retrieved?         │
│ Faithfulness        │ Is the answer grounded in the context (no hallucination)? │
│ Answer Relevancy    │ Does the answer actually address the question asked?       │
└─────────────────────┴──────────────────────────────────────────────────────────┘
```

**Golden Dataset Construction:**

- [ ] 35 Q&A pairs total: 30 answerable (10 per exchange, authored by domain expert Sanjeev) + 5 unanswerable
- [ ] Each entry: `{question, ground_truth_answer, relevant_source_doc, relevant_page}`
- [ ] Stored as `evals/golden_dataset.json`
- [ ] Questions must cover: order types, lot sizes, trading hours, tick sizes, FIX tag specifics, error codes
- [ ] Include 5 "unanswerable" questions (no relevant doc in corpus) to test refusal behaviour

**Eval Script (`evals/run_eval.py`):**

- [ ] Loads golden dataset, runs each question through the live RAG chain
- [ ] Computes RAGAS metrics using `ragas` library
- [ ] Outputs a score report to `evals/results/eval_YYYY-MM-DD.json`
- [ ] Prints a pass/fail summary: Context Precision ≥ 0.75, Faithfulness ≥ 0.85 to pass
- [ ] Eval must be runnable in < 5 minutes for the 35-question dataset

**Eval Triggers:**
- Manually before any PR merge that changes chunking, prompts, or model config
- Automatically after any doc re-ingestion (see P0.5)
- On a weekly cron schedule to catch silent model drift

---

#### 5. Document Update & Version Management

**Description:** A system to detect when source exchange documents have changed, re-ingest updated content, and preserve version history so answers remain trustworthy over time.

**Why this is non-trivial:** Exchanges update their market operation guides 2–4 times per year, often without a versioned URL. Without active change detection, the system silently serves answers from stale docs. For a trading environment, this is equivalent to running pre-trade risk checks against an outdated limit table — technically it runs, but it gives you the wrong answer at the worst possible moment.

**Change Detection Flow:**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Scheduled Check (weekly cron or manual trigger)                      │
│                                                                      │
│  1. For each tracked doc URL → download latest PDF                   │
│  2. Compute MD5 of extracted text                                    │
│  3. Compare with stored hash in doc_registry.json                    │
│                                                                      │
│  No change → log "doc unchanged", skip                               │
│  Hash changed → trigger re-ingestion pipeline                        │
└──────────────────────────────────────────────────────────────────────┘
```

**Re-ingestion Strategy (Soft Delete + Replace):**

- [ ] On doc change detected: delete ALL existing chunks in ChromaDB with matching `source_filename`
- [ ] Re-chunk and re-embed the updated PDF with new `ingested_at` and updated `doc_version_hash`
- [ ] Log old hash + new hash + timestamp to `data/doc_registry.json` (append-only version log)
- [ ] Trigger eval harness automatically after re-ingestion; alert if scores drop below threshold
- [ ] **Do NOT simply append** new chunks alongside old — this causes chunk duplication and conflicting answers

**Staleness Flagging (Runtime):**

- [ ] At query time, if any retrieved chunk has `ingested_at` > 60 days ago, include warning:
  `⚠️ Based on [doc name], last re-verified [date] ([N] days ago). We auto-check weekly — confirm against the exchange website for time-critical decisions.`
- [ ] Staleness threshold configurable in `config.yaml` (default: 60 days)
- [ ] Note: this signal measures **re-verification age**, not confirmed obsolescence — detecting an actual new version is the job of hash-based change detection (above); the timer is a backstop for "we haven't re-checked in a while."

**`doc_registry.json` Schema:**

```json
{
  "SGX_Market_Model_Guide_v4.pdf": {
    "source_url": "https://www.sgx.com/.../market-model.pdf",
    "current_hash": "a3f8c...",
    "version_history": [
      { "hash": "b92d1...", "ingested_at": "2025-11-01", "chunks_count": 312 },
      { "hash": "a3f8c...", "ingested_at": "2026-04-15", "chunks_count": 328 }
    ]
  }
}
```

---

### P1 — Nice to Have (v1.1)

- **LangSmith tracing integration** — log every query, retrieved chunks, and LLM call to LangSmith for debugging and eval visualisation. This is the observability layer — equivalent to adding trade audit logs to a system.
- **Exchange filter at retrieval time** — when user specifies an exchange, apply a metadata pre-filter in ChromaDB before cosine search, improving precision.
- **Comparative query mode** — detect "vs" or "compare" in the question and run two parallel retrievals (one per exchange), then prompt LLM to synthesise a comparison table.
- **FIX tag lookup** — a secondary ChromaDB collection seeded with the FIX 4.2/4.4 tag dictionary, enabling queries like "what is FIX tag 38?" without mixing it into the main exchange docs retrieval.

### P2 — Future Considerations

- **pgvector on Railway** — swap ChromaDB for a cloud-hosted pgvector instance; required for any multi-user or API-accessible deployment.
- **Automated exchange website monitoring** — use a lightweight scraper (Firecrawl or custom) to detect when exchange PDF links change, eliminating the need for manual doc URL maintenance.
- **Streaming responses** — FastAPI `StreamingResponse` for long answers; improves perceived latency.
- **Multi-language support** — ingest Japanese-language TSE/OSE specs and Chinese HKSE docs with a translation preprocessing step.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | **Voyage AI** `voyage-finance-2` | Domain-tuned for financial/technical text; higher retrieval precision than general-purpose models on jargon-heavy docs |
| LLM | Anthropic Claude `claude-sonnet-4-6` | Long-context, strong at structured-doc Q&A and instruction-following (cite sources, refuse when unknown) |
| RAG Framework | LangChain | Document loaders, retrieval chains, metadata filtering |
| Vector Store | ChromaDB (local, persistent) | Zero-infra for MVP; same API as pgvector for future migration |
| PDF Parsing | `PyMuPDF` (via LangChain loader) | Preserves page numbers and layout metadata better than PyPDF2 |
| Eval Framework | **RAGAS** | Purpose-built for RAG evaluation; computes Context Precision/Recall, Faithfulness, Answer Relevancy |
| Eval Tracking | JSON files → LangSmith (P1) | Start simple; migrate to LangSmith once tracing is wired |
| Backend API | FastAPI | Lightweight, async, OpenAPI docs auto-generated |
| UI | Streamlit | Fast to build, sufficient for portfolio demo |
| Doc Registry | `doc_registry.json` | Simple, auditable, no DB dependency for v1 |
| Change Detection | MD5 of normalized extracted text | Deterministic; catches content changes while ignoring cosmetic PDF byte churn (metadata timestamps, re-saves) |

---

## Success Metrics

### Leading Indicators (measurable within 2 weeks of launch)

| Metric | Target | Measurement Method |
|---|---|---|
| RAGAS Faithfulness | ≥ 0.85 | `evals/run_eval.py` against golden dataset |
| RAGAS Context Precision | ≥ 0.75 | `evals/run_eval.py` against golden dataset |
| Refusal rate on unanswerable Qs | ≥ 80% (4 of 5) | 5 unanswerable questions in golden dataset |
| P95 query latency | < 5 seconds | FastAPI response time logs |
| Doc staleness detection accuracy | 100% on test set | Manual: modify a test PDF, verify hash change detected |

### Lagging Indicators (measurable over 4–8 weeks)

| Metric | Target |
|---|---|
| Time-to-answer reduction (self-reported) | From ~30 min → < 2 min for known-doc questions |
| Eval score stability after doc re-ingestion | < 5% regression on Faithfulness score |
| Portfolio signal | GitHub project starred / referenced in AI PM job applications |

---

## Open Questions

| Question | Owner | Blocking? |
|---|---|---|
| Does `voyage-finance-2` require an API key? What are the rate limits and cost per 1M tokens? | Sanjeev (setup) | Yes — needed before ingest pipeline can run |
| Should the golden dataset include questions from FIX specs, or only exchange market model docs in v1? | Sanjeev (domain) | No — can start with market model docs and expand |
| What is the right chunk size for exchange docs? 512 tokens is a reasonable default but section-aware splitting (by heading) may improve precision. | Engineering | No — run an ablation test: compare 512-token fixed vs heading-split using eval scores |
| Should `doc_registry.json` be committed to git (version-controlled) or kept out of the repo? | Sanjeev | No — recommend committing it (it is config, not secrets) |
| Are exchange PDF URLs stable enough for direct download, or do they require manual download and commit? | Sanjeev | Yes — affects whether change detection can be automated |

---

## Timeline Considerations

| Phase | Scope | Target Duration |
|---|---|---|
| **Week 1** — Ingest & Retrieval | Ingest 2 exchanges (SGX + HKSE), basic CLI query, Voyage embeddings wired | 5–7 days |
| **Week 2** — Evals + Staleness | Golden dataset (20 Qs), RAGAS eval harness, doc_registry + hash-based change detection | 5–7 days |
| **Week 3** — API + UI | FastAPI endpoint, Streamlit UI, staleness warnings rendered | 3–5 days |
| **Week 4** — Polish + README | Add TSE, run full eval, write portfolio README, record demo GIF | 3–4 days |

**Hard constraint:** No real exchange credentials or proprietary data. All source documents must be publicly available on exchange websites.

---

## GitHub Repo Structure

This project lives inside the `ai-pm-portfolio` monorepo as a flat top-level directory. Each project sits at the same level — no intermediate `projects/` folder. This keeps navigation simple, GitHub URLs short, and the structure immediately readable to anyone browsing the portfolio.

```
ai-pm-portfolio/                               ← github.com/sanjgha/ai-pm-portfolio
│
├── README.md                                  # Portfolio index — links to all projects with skills badges
├── .gitignore                                 # Root: *.env, __pycache__, *.pdf > 10MB, chroma_db/
├── .github/
│   └── workflows/
│       └── eval-on-pr.yml                     # Path-scoped: runs evals only when this project changes
│
├── shared/
│   └── utils/
│       ├── logging.py                         # Structured logger reused across projects
│       └── config_loader.py                   # Reads config.yaml + .env consistently
│
├── exchange-connectivity-hub/                 ← THIS PROJECT
│   │
│   ├── data/
│   │   ├── raw/                               # Downloaded PDFs (gitignored if > 10MB)
│   │   └── doc_registry.json                  # Version log for all tracked documents
│   │
│   ├── ingest/
│   │   ├── loader.py                          # PyMuPDF-based PDF → chunks with metadata
│   │   ├── chunker.py                         # Chunking strategy (fixed-size + section-aware)
│   │   ├── embedder.py                        # Voyage AI embedding + ChromaDB upsert
│   │   ├── update_checker.py                  # MD5 (extracted-text) comparison + re-ingestion trigger
│   │   └── ingest_pipeline.py                 # Orchestrates loader → chunker → embedder
│   │
│   ├── retrieval/
│   │   ├── retriever.py                       # ChromaDB similarity search with metadata filter
│   │   └── rag_chain.py                       # LangChain RAG chain: retrieve → prompt → generate
│   │
│   ├── evals/
│   │   ├── golden_dataset.json                # 35 Q&A pairs (30 answerable + 5 unanswerable)
│   │   ├── run_eval.py                        # RAGAS evaluation script
│   │   └── results/                           # eval_YYYY-MM-DD.json score history (gitignored)
│   │
│   ├── api/
│   │   └── main.py                            # FastAPI app: POST /query, GET /health
│   │
│   ├── ui/
│   │   └── app.py                             # Streamlit Q&A interface
│   │
│   ├── config.yaml                            # Staleness threshold, model names, top-k
│   ├── .env.example                           # VOYAGE_API_KEY, ANTHROPIC_API_KEY
│   ├── requirements.txt                       # Project-scoped dependencies only
│   └── README.md                              # Project README (portfolio-ready)
│
├── fix-message-explainer/                     ← Future project (same flat pattern)
└── trading-news-digest/                       ← Future project
```

**Monorepo conventions:**
- Each project is fully self-contained and runnable in isolation: `cd exchange-connectivity-hub && pip install -r requirements.txt`
- The root `README.md` acts as the portfolio index — one row per project with a description and skills-covered badge (e.g. `RAG · Voyage AI · RAGAS · LangChain`)
- `evals/results/` is gitignored at project level — eval scores are ephemeral; the harness and golden dataset are committed because they represent the intellectual work
- The GitHub Actions workflow is path-scoped to `exchange-connectivity-hub/**` so it only fires on relevant changes, not every monorepo commit

---

## PM-Ready Framing (Portfolio / LinkedIn)

> "Designed and built a production-pattern RAG system that ingests Asian exchange connectivity documentation (SGX, HKSE, TSE) and reduces answer time for trading rulebook queries from ~30 minutes to under 30 seconds. Key differentiators: domain-tuned Voyage AI embeddings for higher retrieval precision on financial text; a RAGAS evaluation harness with a 35-question golden dataset for continuous quality measurement; and a hash-based document versioning system that detects when exchange docs are updated and triggers automatic re-ingestion and eval regression testing. Built in Python using LangChain, ChromaDB, FastAPI, and the Anthropic Claude API."

---

*Next actions:*
1. *Clone or initialise `github.com/sanjgha/ai-pm-portfolio` and create the `exchange-connectivity-hub/` directory at the repo root*
2. *Get Voyage AI API key at [voyageai.com](https://www.voyageai.com) and add to `.env`*
3. *Download SGX Market Model Guide PDF (public, from sgx.com) into `data/raw/`*
4. *Run `ingest_pipeline.py` for the first time and verify chunks land in ChromaDB*
