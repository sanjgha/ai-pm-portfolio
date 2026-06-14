# Exchange Connectivity Hub

Engineers and Business Analysts manually cross-reference hundreds of pages of exchange PDFs (HKSE, SGX, TSE) to answer connectivity questions—taking 20–40 minutes per query with staleness risk. This project delivers a RAG system that answers those questions in seconds, cites sources, and detects when docs have changed.

## Key Features

- **Fast answers:** Sub-30 second responses with inline citations
- **Source attribution:** Every answer cites filename and page number
- **Staleness warnings:** Flags when source docs haven't been re-verified recently
- **Change detection:** MD5-based hashing detects document updates
- **Eval-driven quality:** RAGAS metrics tracked over time

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | Voyage AI voyage-finance-2 |
| Reranking | Voyage AI rerank-2.5 |
| LLM | Claude claude-sonnet-4-6 |
| RAG Framework | LangChain LCEL |
| Vector Store | ChromaDB (local persistent) |
| PDF Parsing | PyMuPDF |
| Eval | RAGAS |
| API | FastAPI |
| UI | Streamlit |

## Project Structure

```
exchange-connectivity-hub/
├── src/exchange_connectivity_hub/
│   ├── ingest/           # PDF ingestion, chunking, and storage
│   ├── retrieval/        # RAG chain, reranking, and retrieval
│   ├── api/              # FastAPI REST endpoints
│   └── ui/               # Streamlit chat interface
├── data/
│   ├── raw/              # Source PDFs (gitignored)
│   └── chroma_db/        # Persistent vector store (gitignored)
├── evals/
│   ├── golden_dataset.json # Eval questions and ground truth
│   └── results/          # RAGAS metric history
├── tests/                # unit/ and integration/ suites
├── config.yaml           # Default configuration
└── Makefile              # Command shortcuts
```

## Setup

### 1. Install dependencies

```bash
make dev-install
```

### 2. Configure API keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env and add:
#   VOYAGE_API_KEY=your-key
#   ANTHROPIC_API_KEY=your-key
```

### 3. Download exchange PDFs

Place exchange technical specification PDFs into `data/raw/`. Supported exchanges include:
- Hong Kong Stock Exchange (HKSE)
- Singapore Exchange (SGX)
- Tokyo Stock Exchange (TSE)

### 4. Ingest documents

```bash
make ingest-all
```

Or ingest a single document:

```bash
make ingest doc=hkse_connectivity_spec.pdf
```

### 5. (Optional) Run evaluation

```bash
make eval
```

## Usage

### Start the FastAPI server

```bash
make serve-api
# Server runs at http://localhost:8000
# Docs available at http://localhost:8000/docs
```

### Start the Streamlit UI

```bash
make serve-ui
# UI runs at http://localhost:8501
```

### Check for document updates

```bash
make check-updates
# Reports which PDFs have changed since ingestion
```

## Eval Metrics

Quality targets tracked with RAGAS:

| Metric | Target |
|--------|--------|
| Faithfulness | >= 0.85 |
| Context Precision | >= 0.75 |
| Refusal Rate (unanswerable) | >= 80% |

Run `make eval` to execute the golden dataset and view results in `evals/results/`.

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make ci` | Run lint + tests |
| `make ingest doc=X` | Ingest single PDF from data/raw/ |
| `make ingest-all` | Ingest all PDFs in data/raw/ |
| `make check-updates` | Check if source PDFs have changed |
| `make reingest doc=X` | Force re-ingest a PDF (skip hash check) |
| `make serve-api` | Start FastAPI server (localhost:8000) |
| `make serve-ui` | Start Streamlit UI (localhost:8501) |
| `make eval` | Run RAGAS evaluation on golden dataset |
| `make test` | Run pytest tests |
| `make test-cov` | Run tests with HTML coverage report |
| `make lint` | Run ruff + mypy checks |
| `make format` | Auto-format code with ruff |

## License

MIT

## Author

Built as part of AI PM learning portfolio
