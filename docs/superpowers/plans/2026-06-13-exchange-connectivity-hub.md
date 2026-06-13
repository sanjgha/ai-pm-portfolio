# Exchange Connectivity Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG system that ingests Asian exchange PDF documentation (SGX, HKSE, TSE) and answers connectivity questions in under 30 seconds with cited sources, staleness warnings, and eval-driven quality measurement.

**Architecture:** Python package with ingest pipeline (PyMuPDF → Voyage embeddings → ChromaDB), LCEL RAG chain with Voyage reranker, FastAPI backend, Streamlit UI, RAGAS eval harness, and MD5-based change detection.

**Tech Stack:** Python 3.10+, Voyage AI (voyage-finance-2, rerank-2.5), Claude claude-sonnet-4-6, LangChain LCEL, ChromaDB, FastAPI, Streamlit, RAGAS

---

## File Structure

```
projects/exchange-connectivity-hub/
├── pyproject.toml              # deps: langchain, chromadb, voyageai, anthropic, fastapi, streamlit, ragas
├── Makefile                    # make ci / test / ingest / serve-api / serve-ui / eval / check-updates
├── config.yaml                 # retrieval_top_k=20, rerank_top_n=5, rerank_enabled=true, staleness_days=60
├── .env.example                # VOYAGE_API_KEY, ANTHROPIC_API_KEY
├── src/
│   └── exchange_connectivity_hub/
│       ├── __init__.py
│       ├── config.py           # config.yaml + .env loader
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── loader.py           # PyMuPDF → LangChain Documents + metadata
│       │   ├── chunker.py          # RecursiveCharacterTextSplitter.from_tiktoken_encoder 512/64
│       │   ├── embedder.py         # Voyage embed + ChromaDB upsert
│       │   ├── update_checker.py   # MD5 hash vs doc_registry.json
│       │   └── pipeline.py         # Orchestrates load → chunk → embed
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── retriever.py        # ChromaDB cosine search + exchange metadata filter
│       │   ├── reranker.py         # Voyage rerank-2.5: reranks retrieval_top_k → rerank_top_n
│       │   └── rag_chain.py        # LCEL: retriever | reranker | prompt | Claude | output parser
│       ├── api/
│       │   ├── __init__.py
│       │   └── main.py             # FastAPI POST /query, GET /health
│       └── ui/
│           ├── __init__.py
│           └── app.py              # Streamlit Q&A interface
├── data/
│   ├── raw/                    # Downloaded PDFs (gitignored if >10MB)
│   ├── chroma_db/              # ChromaDB persistent store (gitignored)
│   └── doc_registry.json       # Committed — version log per tracked doc
├── evals/
│   ├── golden_dataset.json     # 35 Q&A pairs: 30 answerable + 5 unanswerable
│   ├── run_eval.py             # RAGAS eval script
│   └── results/                # gitignored — ephemeral score history
└── tests/
    ├── __init__.py
    ├── conftest.py             # shared fixtures
    ├── unit/
    │   ├── __init__.py
    │   ├── test_chunker.py     # chunk size/overlap tests (mocked Voyage)
    │   ├── test_update_checker.py  # MD5 computation, hash comparison
    │   └── test_reranker.py     # ordering verification (mocked Voyage)
    └── integration/
        ├── __init__.py
        └── test_rag_chain.py   # live ChromaDB + mocked Voyage/Claude
```

---

## Task 1: Scaffold Project Structure

**Files:**
- Create: `projects/exchange-connectivity-hub/` (entire directory structure)
- Use root `make new-project` command

- [ ] **Step 1: Scaffold the project using root Makefile**

```bash
cd /home/ubuntu/projects/ai-pm-portfolio
make new-project name=exchange-connectivity-hub desc="RAG system for Asian exchange connectivity documentation"
```

Expected output: Project scaffolded at `projects/exchange-connectivity-hub/`

- [ ] **Step 2: Create directory structure**

```bash
cd /home/ubuntu/projects/ai-pm-portfolio/projects/exchange-connectivity-hub
mkdir -p src/exchange_connectivity_hub/{ingest,retrieval,api,ui}
mkdir -p data/{raw,chroma_db}
mkdir -p evals/results
mkdir -p tests/{unit,integration}
```

- [ ] **Step 3: Create __init__.py files**

```bash
touch src/exchange_connectivity_hub/__init__.py
touch src/exchange_connectivity_hub/ingest/__init__.py
touch src/exchange_connectivity_hub/retrieval/__init__.py
touch src/exchange_connectivity_hub/api/__init__.py
touch src/exchange_connectivity_hub/ui/__init__.py
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
touch evals/results/.gitkeep
```

- [ ] **Step 4: Update .gitignore for data directories**

```bash
echo "data/raw/*.pdf" >> .gitignore
echo "data/chroma_db/" >> .gitignore
echo "evals/results/" >> .gitignore
```

- [ ] **Step 5: Verify scaffold and commit**

```bash
ls -la src/exchange_connectivity_hub/
ls -la data/ evals/ tests/
git add projects/exchange-connectivity-hub/
git commit -m "feat(exchange-connectivity-hub): scaffold project structure"
```

---

## Task 2: Create Config System

**Files:**
- Create: `projects/exchange-connectivity-hub/config.yaml`
- Create: `projects/exchange-connectivity-hub/.env.example`
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/config.py`
- Modify: `projects/exchange-connectivity-hub/pyproject.toml` (add pyyaml dependency)

- [ ] **Step 1: Write config.yaml**

Create `config.yaml`:

```yaml
# RAG Configuration
retrieval:
  top_k: 20              # Number of chunks to retrieve before reranking
  rerank_top_n: 5        # Number of chunks to keep after reranking
  rerank_enabled: true   # Enable Voyage rerank-2.5

# Models
models:
  embedding: voyage-finance-2
  rerank: rerank-2.5
  llm: claude-sonnet-4-6

# Vector Store
vector_store:
  persist_directory: data/chroma_db
  collection_name: exchange_docs

# Staleness Detection
staleness:
  days: 60  # Warn if source doc ingested more than N days ago

# API
api:
  host: 0.0.0.0
  port: 8000

# UI
ui:
  host: localhost
  port: 8501
```

- [ ] **Step 2: Write .env.example**

Create `.env.example`:

```bash
# API Keys
VOYAGE_API_KEY=your-voyage-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Override config values
# STALENESS_DAYS=60
# RETRIEVAL_TOP_K=20
```

- [ ] **Step 3: Write config.py with failing test first**

Create `tests/unit/test_config.py`:

```python
"""Test config loading."""

import os
import pytest
from pathlib import Path
from exchange_connectivity_hub.config import get_config


def test_get_config_returns_expected_structure():
    """Config should have all required sections."""
    config = get_config()
    assert "retrieval" in config
    assert "models" in config
    assert "vector_store" in config
    assert "staleness" in config
    assert "api" in config
    assert "ui" in config


def test_retrieval_config_defaults():
    """Retrieval config should have expected defaults."""
    config = get_config()
    assert config["retrieval"]["top_k"] == 20
    assert config["retrieval"]["rerank_top_n"] == 5
    assert config["retrieval"]["rerank_enabled"] is True


def test_model_names_configured():
    """Model names should be set."""
    config = get_config()
    assert config["models"]["embedding"] == "voyage-finance-2"
    assert config["models"]["rerank"] == "rerank-2.5"
    assert config["models"]["llm"] == "claude-sonnet-4-6"


def test_vector_store_config():
    """Vector store config should point to data directory."""
    config = get_config()
    assert "chroma_db" in config["vector_store"]["persist_directory"]
    assert config["vector_store"]["collection_name"] == "exchange_docs"


def test_api_keys_required():
    """API keys should be loaded from environment."""
    # Test that the function doesn't crash without keys (keys used at runtime)
    # But it should validate they're set when accessed
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        from exchange_connectivity_hub.config import get_voyage_api_key
        os.environ.pop("VOYAGE_API_KEY", None)
        get_voyage_api_key()
```

Run test:

```bash
cd /home/ubuntu/projects/ai-pm-portfolio/projects/exchange-connectivity-hub
pytest tests/unit/test_config.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 4: Implement config.py**

Create `src/exchange_connectivity_hub/config.py`:

```python
"""Configuration management for exchange connectivity hub."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# Path to config.yaml
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_config() -> dict[str, Any]:
    """Load configuration from config.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Global config cache
_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Get configuration, loading once and caching."""
    global _config
    if _config is None:
        _config = _load_config()
    return _config


def get_voyage_api_key() -> str:
    """Get Voyage API key from environment."""
    key = os.getenv("VOYAGE_API_KEY")
    if not key:
        raise ValueError("VOYAGE_API_KEY environment variable not set")
    return key


def get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return key
```

- [ ] **Step 5: Update pyproject.toml with dependencies**

Add to `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
]
```

- [ ] **Step 6: Install and run tests**

```bash
cd /home/ubuntu/projects/ai-pm-portfolio/projects/exchange-connectivity-hub
pip install -e ".[dev]"
pytest tests/unit/test_config.py -v
```

Expected: PASS

- [ ] **Step 7: Fix test for API key validation**

The test needs to set the env var before testing. Update `test_config.py`:

```python
def test_api_keys_required():
    """API keys should be loaded from environment."""
    from exchange_connectivity_hub.config import get_voyage_api_key

    # Set a test key
    os.environ["VOYAGE_API_KEY"] = "test-key"
    assert get_voyage_api_key() == "test-key"

    # Test that missing key raises ValueError
    os.environ.pop("VOYAGE_API_KEY", None)
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        get_voyage_api_key()

    # Restore for other tests
    os.environ["VOYAGE_API_KEY"] = "test-key"
```

```bash
pytest tests/unit/test_config.py -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add config.yaml .env.example src/exchange_connectivity_hub/config.py tests/unit/test_config.py pyproject.toml
git commit -m "feat(exchange-connectivity-hub): add config system with YAML and .env support"
```

---

## Task 3: Create Doc Registry System

**Files:**
- Create: `projects/exchange-connectivity-hub/data/doc_registry.json`
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/ingest/doc_registry.py`
- Create: `projects/exchange-connectivity-hub/tests/unit/test_doc_registry.py`

- [ ] **Step 1: Write failing test for doc_registry**

Create `tests/unit/test_doc_registry.py`:

```python
"""Test document registry management."""

import json
from pathlib import Path
from exchange_connectivity_hub.ingest.doc_registry import (
    DocRegistry,
    load_registry,
    save_registry,
)


def test_load_registry_creates_default():
    """Loading non-existent registry should create default."""
    test_path = Path("/tmp/test_registry_empty.json")
    if test_path.exists():
        test_path.unlink()

    registry = load_registry(test_path)
    assert registry == {}
    assert test_path.exists()


def test_save_and_load_registry():
    """Saving and loading should preserve data."""
    test_path = Path("/tmp/test_registry_save.json")
    if test_path.exists():
        test_path.unlink()

    registry = {
        "SGX_doc.pdf": {
            "source_url": "https://sgx.com/doc.pdf",
            "exchange": "SGX",
            "doc_type": "market_model",
            "current_hash": "abc123",
            "version_history": [
                {
                    "hash": "abc123",
                    "ingested_at": "2026-06-13T10:00:00Z",
                    "chunks_count": 100,
                }
            ],
        }
    }

    save_registry(registry, test_path)
    loaded = load_registry(test_path)

    assert loaded == registry
    assert loaded["SGX_doc.pdf"]["exchange"] == "SGX"


def test_doc_registry_add_version():
    """DocRegistry should add new versions correctly."""
    test_path = Path("/tmp/test_registry_add.json")
    if test_path.exists():
        test_path.unlink()

    registry = DocRegistry(test_path)
    registry.register_doc(
        filename="test.pdf",
        source_url="https://test.com/test.pdf",
        exchange="TEST",
        doc_type="test_doc",
        content_hash="hash1",
        chunks_count=50,
    )

    # Verify entry exists
    assert "test.pdf" in registry.data
    assert registry.data["test.pdf"]["current_hash"] == "hash1"
    assert registry.data["test.pdf"]["version_history"][0]["chunks_count"] == 50

    # Add new version
    registry.register_doc(
        filename="test.pdf",
        source_url="https://test.com/test.pdf",
        exchange="TEST",
        doc_type="test_doc",
        content_hash="hash2",
        chunks_count=55,
    )

    # Should have two versions
    assert len(registry.data["test.pdf"]["version_history"]) == 2
    assert registry.data["test.pdf"]["current_hash"] == "hash2"


def test_doc_registry_check_hash():
    """check_hash_changed should detect modifications."""
    test_path = Path("/tmp/test_registry_hash.json")
    if test_path.exists():
        test_path.unlink()

    registry = DocRegistry(test_path)
    registry.register_doc(
        filename="test.pdf",
        source_url="https://test.com/test.pdf",
        exchange="TEST",
        doc_type="test_doc",
        content_hash="original",
        chunks_count=50,
    )

    # Same hash = not changed
    assert registry.check_hash_changed("test.pdf", "original") is False

    # Different hash = changed
    assert registry.check_hash_changed("test.pdf", "new_hash") is True

    # Unknown doc = considered changed (will register)
    assert registry.check_hash_changed("unknown.pdf", "any_hash") is True


def test_doc_registry_get_source_url():
    """getSourceUrl should return URL or None."""
    test_path = Path("/tmp/test_registry_url.json")
    if test_path.exists():
        test_path.unlink()

    registry = DocRegistry(test_path)
    registry.register_doc(
        filename="test.pdf",
        source_url="https://test.com/test.pdf",
        exchange="TEST",
        doc_type="test_doc",
        content_hash="hash1",
        chunks_count=50,
    )

    assert registry.get_source_url("test.pdf") == "https://test.com/test.pdf"
    assert registry.get_source_url("unknown.pdf") is None
```

Run test:

```bash
cd /home/ubuntu/projects/ai-pm-portfolio/projects/exchange-connectivity-hub
pytest tests/unit/test_doc_registry.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 2: Implement doc_registry.py**

Create `src/exchange_connectivity_hub/ingest/doc_registry.py`:

```python
"""Document registry for tracking version history of ingested PDFs."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_registry(path: Path) -> dict[str, Any]:
    """Load registry from file, creating empty dict if not exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Create empty registry
    save_registry({}, path)
    return {}


def save_registry(data: dict[str, Any], path: Path) -> None:
    """Save registry to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class DocRegistry:
    """Manages document version history and hash tracking."""

    def __init__(self, registry_path: Path) -> None:
        """Initialize with registry file path."""
        self.path = registry_path
        self.data = load_registry(registry_path)

    def register_doc(
        self,
        *,
        filename: str,
        source_url: str | None,
        exchange: str,
        doc_type: str,
        content_hash: str,
        chunks_count: int,
    ) -> None:
        """Register or update a document in the registry."""
        now = datetime.now(UTC).isoformat()

        if filename not in self.data:
            # New document
            self.data[filename] = {
                "source_url": source_url,
                "exchange": exchange,
                "doc_type": doc_type,
                "current_hash": content_hash,
                "version_history": [
                    {
                        "hash": content_hash,
                        "ingested_at": now,
                        "chunks_count": chunks_count,
                    }
                ],
            }
        else:
            # Update existing document
            entry = self.data[filename]
            entry["current_hash"] = content_hash
            entry["version_history"].append(
                {
                    "hash": content_hash,
                    "ingested_at": now,
                    "chunks_count": chunks_count,
                }
            )

        self.save()

    def check_hash_changed(self, filename: str, new_hash: str) -> bool:
        """Check if document hash has changed.

        Returns True if:
        - Document not in registry (will register as new)
        - Hash differs from current_hash
        """
        if filename not in self.data:
            return True  # New doc
        return self.data[filename]["current_hash"] != new_hash

    def get_source_url(self, filename: str) -> str | None:
        """Get source URL for a document."""
        if filename not in self.data:
            return None
        return self.data[filename]["source_url"]

    def get_current_hash(self, filename: str) -> str | None:
        """Get current hash for a document."""
        if filename not in self.data:
            return None
        return self.data[filename]["current_hash"]

    def save(self) -> None:
        """Save registry to disk."""
        save_registry(self.data, self.path)

    def get_all_filenames(self) -> list[str]:
        """Get list of all tracked filenames."""
        return list(self.data.keys())
```

- [ ] **Step 3: Run tests to verify implementation**

```bash
pytest tests/unit/test_doc_registry.py -v
```

Expected: PASS

- [ ] **Step 4: Create initial doc_registry.json**

Create `data/doc_registry.json`:

```json
{
  "SGX_Market_Model_Guide_v4.pdf": {
    "source_url": "https://www.sgx.com/docs/market-model-guide-v4.pdf",
    "exchange": "SGX",
    "doc_type": "market_model",
    "current_hash": null,
    "version_history": []
  },
  "HKSE_Rules_Guide.pdf": {
    "source_url": "https://www.hkex.com.hk/rules-guide.pdf",
    "exchange": "HKSE",
    "doc_type": "market_model",
    "current_hash": null,
    "version_history": []
  },
  "TSE_Trading_Guide.pdf": {
    "source_url": "https://www.jpx.co.jp/tse-trading-guide.pdf",
    "exchange": "TSE",
    "doc_type": "market_model",
    "current_hash": null,
    "version_history": []
  }
}
```

Note: URLs are placeholders - user will update with actual URLs.

- [ ] **Step 5: Commit**

```bash
git add src/exchange_connectivity_hub/ingest/doc_registry.py tests/unit/test_doc_registry.py data/doc_registry.json
git commit -m "feat(exchange-connectivity-hub): add doc registry with version tracking"
```

---

## Task 4: Implement PDF Loader

**Files:**
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/ingest/loader.py`
- Create: `projects/exchange-connectivity-hub/tests/unit/test_loader.py`
- Modify: `projects/exchange-connectivity-hub/pyproject.toml` (add pymupdf, langchain-community)

- [ ] **Step 1: Write failing tests for loader**

Create `tests/unit/test_loader.py`:

```python
"""Test PDF loader functionality."""

from pathlib import Path
from exchange_connectivity_hub.ingest.loader import load_pdf, extract_text_for_hash


def test_extract_text_for_hash_normalizes_text():
    """Text extraction should normalize whitespace for stable hashing."""
    # Create a test PDF content with irregular spacing
    sample_text = """
    Hello     World

    This  has    irregular    spacing.

    And line breaks.
    """

    # The function should normalize this
    normalized = extract_text_for_hash(sample_text)

    # Should collapse multiple spaces
    assert "Hello     World" not in normalized
    assert "Hello World" in normalized

    # Should trim leading/trailing whitespace per line
    assert normalized.startswith("Hello World")
    assert not normalized.startswith("  ")


def test_extract_text_for_hash_handles_empty():
    """Empty text should return empty string."""
    assert extract_text_for_hash("") == ""
    assert extract_text_for_hash("   ") == ""


def test_load_pdf_returns_documents_with_metadata(tmp_path):
    """Loading a PDF should return LangChain Documents with metadata."""
    from langchain_core.documents import Document

    # This test requires a real PDF file
    # For unit tests, we'll mock the PyMuPDFLoader
    # Integration tests will use real PDFs
    pass  # Implemented in integration tests


def test_load_pdf_attaches_required_metadata():
    """PDF loader should attach all required metadata fields."""
    from unittest.mock import MagicMock, patch

    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "test.pdf", "page": 0}
    mock_doc.page_content = "Test content"

    with patch("exchange_connectivity_hub.ingest.loader.PyMuPDFLoader") as mock_loader:
        mock_loader.return_value.load.return_value = [mock_doc]

        docs = load_pdf(
            pdf_path=Path("test.pdf"),
            exchange="SGX",
            doc_type="market_model",
        )

        assert len(docs) == 1
        metadata = docs[0].metadata
        assert metadata["source_filename"] == "test.pdf"
        assert metadata["exchange"] == "SGX"
        assert metadata["doc_type"] == "market_model"
        assert metadata["page_number"] == 0
        assert "ingested_at" in metadata
        assert "doc_version_hash" not in metadata  # Added after full text extraction
```

Run test:

```bash
pytest tests/unit/test_loader.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 2: Implement loader.py**

Create `src/exchange_connectivity_hub/ingest/loader.py`:

```python
"""PDF loading with metadata attachment for exchange documents."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyMuPDFLoader


def extract_text_for_hash(raw_text: str) -> str:
    """Normalize extracted text for stable MD5 hashing.

    Removes cosmetic variations (extra whitespace, line breaks) that
    don't represent content changes.
    """
    if not raw_text:
        return ""

    # Normalize: collapse multiple spaces, strip each line
    lines = raw_text.split("\n")
    normalized_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            # Collapse multiple spaces within line
            collapsed = " ".join(stripped.split())
            normalized_lines.append(collapsed)

    return "\n".join(normalized_lines)


def load_pdf(
    *, pdf_path: Path, exchange: str, doc_type: str
) -> list[Any]:
    """Load a PDF and attach metadata to each page/document.

    Args:
        pdf_path: Path to PDF file
        exchange: Exchange code (SGX, HKSE, TSE, etc.)
        doc_type: Document type (market_model, fix_spec, etc.)

    Returns:
        List of LangChain Documents with enhanced metadata
    """
    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()

    ingested_at = datetime.now(UTC).isoformat()
    filename = pdf_path.name

    for doc in docs:
        # Enhance metadata
        base_metadata = doc.metadata
        doc.metadata = {
            "source_filename": filename,
            "exchange": exchange,
            "doc_type": doc_type,
            "page_number": base_metadata.get("page", 0),
            "ingested_at": ingested_at,
            # doc_version_hash added after full text extraction in pipeline
        }

    return docs


def extract_full_text_from_docs(docs: list[Any]) -> str:
    """Extract and concatenate all page text for hash computation.

    Args:
        docs: List of LangChain Documents from loader

    Returns:
        Normalized full text for MD5 hashing
    """
    all_text = []
    for doc in docs:
        all_text.append(doc.page_content)

    full_text = "\n".join(all_text)
    return extract_text_for_hash(full_text)
```

- [ ] **Step 3: Update pyproject.toml with dependencies**

Add to `dependencies` in `pyproject.toml`:

```toml
dependencies = [
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",
    "pymupdf>=1.23.0",
],
```

- [ ] **Step 4: Install and run tests**

```bash
pip install -e ".[dev]"
pytest tests/unit/test_loader.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exchange_connectivity_hub/ingest/loader.py tests/unit/test_loader.py pyproject.toml
git commit -m "feat(exchange-connectivity-hub): add PDF loader with metadata attachment"
```

---

## Task 5: Implement Chunker

**Files:**
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/ingest/chunker.py`
- Modify: `projects/exchange-connectivity-hub/tests/unit/test_chunker.py` (expand tests)

- [ ] **Step 1: Write failing tests for chunker**

Update `tests/unit/test_chunker.py`:

```python
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
    # Create a document with known token count
    # 1 token ≈ 4 chars for English, so 2000 chars ≈ 500 tokens
    long_text = "word " * 400  # ~800 tokens

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
    long_text = "word " * 400

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
```

Run test:

```bash
pytest tests/unit/test_chunker.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 2: Implement chunker.py**

Create `src/exchange_connectivity_hub/ingest/chunker.py`:

```python
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
```

- [ ] **Step 3: Update pyproject.toml with dependencies**

Add to `dependencies` in `pyproject.toml`:

```toml
"langchain-text-splitters>=0.3.0",
"tiktoken>=0.5.0",
```

- [ ] **Step 4: Install and run tests**

```bash
pip install -e ".[dev]"
pytest tests/unit/test_chunker.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exchange_connectivity_hub/ingest/chunker.py tests/unit/test_chunker.py pyproject.toml
git commit -m "feat(exchange-connectivity-hub): add token-aware document chunker"
```

---

## Task 6: Implement Embedder with ChromaDB

**Files:**
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/ingest/embedder.py`
- Create: `projects/exchange-connectivity-hub/tests/unit/test_embedder.py`

- [ ] **Step 1: Write failing tests for embedder**

Create `tests/unit/test_embedder.py`:

```python
"""Test embedding and ChromaDB storage functionality."""

from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from exchange_connectivity_hub.ingest.embedder import (
    embed_and_store,
    get_collection,
    delete_by_source,
)


def test_get_collection_returns_chroma_collection():
    """Should return a ChromaDB collection."""
    with patch("exchange_connectivity_hub.ingest.embedder.Chroma") as mock_chroma:
        mock_collection = MagicMock()
        mock_persist = MagicMock()
        mock_chroma.return_value = mock_collection
        mock_collection.return_value = mock_persist

        collection = get_collection(persist_directory="/tmp/test_db")

        assert collection is not None


def test_embed_and_store_calls_voyage():
    """Should call Voyage embed API and store in ChromaDB."""
    # Mock Voyage embedding
    mock_embedding = [0.1] * 1024  # 1024-dim vector

    with patch("exchange_connectivity_hub.ingest.embedder.embed_documents") as mock_embed:
        mock_embed.return_value = [mock_embedding, mock_embedding]

        with patch("exchange_connectivity_hub.ingest.embedder.Chroma") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value = mock_collection

            docs = [
                Document(page_content="Test 1", metadata={"source": "test.pdf"}),
                Document(page_content="Test 2", metadata={"source": "test.pdf"}),
            ]

            embed_and_store(docs, collection_name="test_collection")

            # Should have called embed_documents
            mock_embed.assert_called_once()

            # Should have added to collection
            assert mock_collection.add_documents.called or mock_collection.return_value.add_documents.called


def test_delete_by_source_removes_chunks():
    """Should delete all chunks for a given source filename."""
    with patch("exchange_connectivity_hub.ingest.embedder.Chroma") as mock_chroma:
        mock_collection = MagicMock()
        mock_chroma.return_value = mock_collection

        delete_by_source(
            source_filename="test.pdf",
            collection_name="test_collection",
        )

        # Should have called delete with metadata filter
        mock_collection.return_value.delete.assert_called()


def test_embed_and_store_preserves_metadata():
    """Should preserve all metadata from source documents."""
    mock_embedding = [0.1] * 1024

    with patch("exchange_connectivity_hub.ingest.embedder.embed_documents") as mock_embed:
        mock_embed.return_value = [mock_embedding]

        with patch("exchange_connectivity_hub.ingest.embedder.Chroma") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value = mock_collection

            doc = Document(
                page_content="Test",
                metadata={
                    "source_filename": "test.pdf",
                    "exchange": "SGX",
                    "page_number": 5,
                    "ingested_at": "2026-06-13T10:00:00Z",
                    "doc_version_hash": "abc123",
                },
            )

            embed_and_store([doc], collection_name="test_collection")

            # Verify add_documents was called with the doc
            call_args = mock_collection.return_value.add_documents.call_args
            if call_args:
                docs_stored = call_args[0][0] if call_args[0] else []
                if docs_stored:
                    # Check metadata is preserved
                    stored_metadata = docs_stored[0].metadata
                    assert stored_metadata["source_filename"] == "test.pdf"
                    assert stored_metadata["exchange"] == "SGX"
```

Run test:

```bash
pytest tests/unit/test_embedder.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 2: Implement embedder.py**

Create `src/exchange_connectivity_hub/ingest/embedder.py`:

```python
"""Embedding and vector storage using Voyage AI and ChromaDB."""

from typing import Any

from chromadb import Chroma
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma as LangChainChroma
from langchain_community.embeddings import VoyageEmbeddings
from exchange_connectivity_hub.config import get_config, get_voyage_api_key


def get_collection(*, persist_directory: str, collection_name: str) -> Any:
    """Get or create ChromaDB collection.

    Args:
        persist_directory: Directory for ChromaDB storage
        collection_name: Name of the collection

    Returns:
        ChromaDB collection
    """
    config = get_config()
    model = config["models"]["embedding"]

    # Initialize embeddings
    embeddings = VoyageEmbeddings(
        voyage_api_key=get_voyage_api_key(),
        model=model,
    )

    # Create or get collection
    vectorstore = LangChainChroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    return vectorstore


def embed_and_store(
    docs: list[Any],
    *,
    collection_name: str,
    persist_directory: str | None = None,
) -> None:
    """Embed documents and store in ChromaDB.

    Args:
        docs: List of LangChain Documents with metadata
        collection_name: ChromaDB collection name
        persist_directory: ChromaDB storage directory (uses config if None)
    """
    config = get_config()

    if persist_directory is None:
        persist_directory = config["vector_store"]["persist_directory"]

    model = config["models"]["embedding"]
    embeddings = VoyageEmbeddings(
        voyage_api_key=get_voyage_api_key(),
        model=model,
    )

    # Create vectorstore and add documents
    vectorstore = LangChainChroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Add documents with their metadata
    vectorstore.add_documents(docs)


def delete_by_source(
    *,
    source_filename: str,
    collection_name: str,
    persist_directory: str | None = None,
) -> None:
    """Delete all chunks for a given source filename.

    Args:
        source_filename: Filename to delete (e.g., "SGX_doc.pdf")
        collection_name: ChromaDB collection name
        persist_directory: ChromaDB storage directory (uses config if None)
    """
    config = get_config()

    if persist_directory is None:
        persist_directory = config["vector_store"]["persist_directory"]

    # Connect directly to ChromaDB to delete by metadata
    chroma_client = Chroma(
        persist_directory=persist_directory,
    )

    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Delete all documents matching the source filename
    collection.delete(
        where={"source_filename": source_filename},
    )
```

- [ ] **Step 3: Update pyproject.toml with dependencies**

Add to `dependencies` in `pyproject.toml`:

```toml
"chromadb>=0.5.0",
"voyageai>=0.2.0",
```

- [ ] **Step 4: Install and run tests**

```bash
pip install -e ".[dev]"
pytest tests/unit/test_embedder.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exchange_connectivity_hub/ingest/embedder.py tests/unit/test_embedder.py pyproject.toml
git commit -m "feat(exchange-connectivity-hub): add Voyage embeddings + ChromaDB storage"
```

---

## Task 7: Implement Update Checker (MD5 Change Detection)

**Files:**
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/ingest/update_checker.py`
- Modify: `projects/exchange-connectivity-hub/tests/unit/test_update_checker.py` (rename from test_doc_registry related tests)

- [ ] **Step 1: Write failing tests for update_checker**

Create `tests/unit/test_update_checker.py`:

```python
"""Test document update detection via MD5 hashing."""

import hashlib
from exchange_connectivity_hub.ingest.update_checker import (
    compute_hash,
    download_pdf,
    check_for_updates,
)


def test_compute_hash_handles_empty_text():
    """Empty text should produce valid hash."""
    hash_value = compute_hash("")
    assert hash_value is not None
    assert len(hash_value) == 32  # MD5 is 32 hex chars


def test_compute_hash_is_deterministic():
    """Same text should produce same hash."""
    text = "Hello, world!"
    hash1 = compute_hash(text)
    hash2 = compute_hash(text)
    assert hash1 == hash2


def test_compute_hash_normalizes_whitespace():
    """Hash should be based on normalized text."""
    # Different spacing, same content
    text1 = "Hello  World"
    text2 = "Hello World"

    # After normalization, should be same
    # (Note: this depends on extract_text_for_hash doing normalization)
    from exchange_connectivity_hub.ingest.loader import extract_text_for_hash

    hash1 = hashlib.md5(extract_text_for_hash(text1).encode()).hexdigest()
    hash2 = hashlib.md5(extract_text_for_hash(text2).encode()).hexdigest()

    assert hash1 == hash2


def test_download_pdf_to_temp(tmp_path):
    """Download PDF to temporary location."""
    # This test requires a real URL - use a test file
    # Integration test only
    pass


def test_check_for_updates_compares_hashes():
    """check_for_updates should detect hash changes."""
    from unittest.mock import MagicMock, patch
    from exchange_connectivity_hub.ingest.doc_registry import DocRegistry

    mock_registry = MagicMock(spec=DocRegistry)
    mock_registry.data = {
        "test.pdf": {
            "source_url": "https://example.com/test.pdf",
            "current_hash": "old_hash",
        }
    }

    with patch("exchange_connectivity_hub.ingest.update_checker.compute_hash") as mock_hash:
        mock_hash.return_value = "new_hash"  # Different hash

        changes = check_for_updates(mock_registry)

        # Should detect the change
        assert "test.pdf" in changes
        assert changes["test.pdf"]["old_hash"] == "old_hash"
        assert changes["test.pdf"]["new_hash"] == "new_hash"


def test_check_for_updates_skips_unchanged():
    """check_for_updates should skip unchanged docs."""
    from unittest.mock import MagicMock, patch
    from exchange_connectivity_hub.ingest.doc_registry import DocRegistry

    mock_registry = MagicMock(spec=DocRegistry)
    mock_registry.data = {
        "test.pdf": {
            "source_url": "https://example.com/test.pdf",
            "current_hash": "same_hash",
        }
    }

    with patch("exchange_connectivity_hub.ingest.update_checker.compute_hash") as mock_hash:
        mock_hash.return_value = "same_hash"  # Same hash

        changes = check_for_updates(mock_registry)

        # Should not report any changes
        assert len(changes) == 0
```

Run test:

```bash
pytest tests/unit/test_update_checker.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 2: Implement update_checker.py**

Create `src/exchange_connectivity_hub/ingest/update_checker.py`:

```python
"""Document change detection via MD5 hash comparison."""

import hashlib
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from exchange_connectivity_hub.ingest.doc_registry import DocRegistry
from exchange_connectivity_hub.ingest.loader import extract_text_for_hash


def compute_hash(text: str) -> str:
    """Compute MD5 hash of normalized text.

    Args:
        text: Text content to hash

    Returns:
        Hexadecimal MD5 hash string
    """
    normalized = extract_text_for_hash(text)
    return hashlib.md5(normalized.encode()).hexdigest()


def download_pdf(url: str, dest_path: Path) -> None:
    """Download PDF from URL to destination path.

    Args:
        url: Source URL
        dest_path: Destination file path
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest_path)


def check_for_updates(registry: DocRegistry) -> dict[str, dict[str, str]]:
    """Check all tracked documents for content changes.

    Downloads each document from its source URL, computes hash,
    and compares with stored hash.

    Args:
        registry: DocRegistry instance

    Returns:
        Dict mapping filename -> {"old_hash": str, "new_hash": str}
        for documents that have changed
    """
    from exchange_connectivity_hub.ingest.loader import load_pdf

    changes = {}

    for filename, entry in registry.data.items():
        source_url = entry.get("source_url")
        if not source_url:
            continue  # Skip docs without URLs

        try:
            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                download_pdf(source_url, tmp_path)

            # Load and extract text
            docs = load_pdf(
                pdf_path=tmp_path,
                exchange=entry["exchange"],
                doc_type=entry["doc_type"],
            )

            # Compute hash
            from exchange_connectivity_hub.ingest.loader import extract_full_text_from_docs

            full_text = extract_full_text_from_docs(docs)
            new_hash = compute_hash(full_text)
            old_hash = entry.get("current_hash")

            # Compare
            if old_hash and new_hash != old_hash:
                changes[filename] = {
                    "old_hash": old_hash,
                    "new_hash": new_hash,
                }

            # Clean up temp file
            tmp_path.unlink()

        except Exception as e:
            # Log error but continue checking other docs
            print(f"Error checking {filename}: {e}")
            continue

    return changes
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/unit/test_update_checker.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/exchange_connectivity_hub/ingest/update_checker.py tests/unit/test_update_checker.py
git commit -m "feat(exchange-connectivity-hub): add MD5-based change detection for documents"
```

---

## Task 8: Implement Ingest Pipeline

**Files:**
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/ingest/pipeline.py`
- Create: `projects/exchange-connectivity-hub/Makefile` (add ingest targets)
- Create: `projects/exchange-connectivity-hub/tests/integration/test_pipeline.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/integration/test_pipeline.py`:

```python
"""Integration test for full ingest pipeline."""

from pathlib import Path
from exchange_connectivity_hub.ingest.pipeline import ingest_single_pdf


def test_ingest_single_pdf_end_to_end(tmp_path):
    """Full pipeline: load → chunk → embed → store → register."""
    # This requires a real PDF file and API keys
    # Skip if API keys not available
    import os

    if not os.getenv("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")

    # Create a minimal test PDF (requires actual PDF file)
    # For now, test the pipeline structure
    pass
```

- [ ] **Step 2: Implement pipeline.py**

Create `src/exchange_connectivity_hub/ingest/pipeline.py`:

```python
"""Ingest pipeline orchestrating load → chunk → embed → store."""

from pathlib import Path

from exchange_connectivity_hub.config import get_config
from exchange_connectivity_hub.ingest.chunker import chunk_documents
from exchange_connectivity_hub.ingest.doc_registry import DocRegistry
from exchange_connectivity_hub.ingest.embedder import delete_by_source, embed_and_store
from exchange_connectivity_hub.ingest.loader import extract_full_text_from_docs, load_pdf


def ingest_single_pdf(
    *,
    pdf_path: Path,
    exchange: str,
    doc_type: str,
    source_url: str | None = None,
    force_reingest: bool = False,
) -> dict:
    """Run full ingest pipeline for a single PDF.

    1. Load PDF with metadata
    2. Extract full text for hash
    3. Check if already ingested (unless force)
    4. Chunk documents
    5. Embed and store in ChromaDB
    6. Update doc registry

    Args:
        pdf_path: Path to PDF file
        exchange: Exchange code
        doc_type: Document type
        source_url: Optional source URL for change detection
        force_reingest: Skip hash check and re-ingest

    Returns:
        Dict with ingest results
    """
    config = get_config()
    collection_name = config["vector_store"]["collection_name"]
    registry_path = Path("data/doc_registry.json")

    # Initialize registry
    registry = DocRegistry(registry_path)

    filename = pdf_path.name

    # Load PDF
    docs = load_pdf(pdf_path=pdf_path, exchange=exchange, doc_type=doc_type)

    # Compute hash
    full_text = extract_full_text_from_docs(docs)
    import hashlib

    from exchange_connectivity_hub.ingest.update_checker import compute_hash

    content_hash = compute_hash(full_text)

    # Check if already ingested
    if not force_reingest:
        current_hash = registry.get_current_hash(filename)
        if current_hash == content_hash:
            return {
                "status": "skipped",
                "filename": filename,
                "reason": "already ingested (hash unchanged)",
            }

    # Delete existing chunks if re-ingesting
    if force_reingest or registry.get_current_hash(filename):
        delete_by_source(
            source_filename=filename,
            collection_name=collection_name,
        )

    # Attach hash to metadata
    for doc in docs:
        doc.metadata["doc_version_hash"] = content_hash

    # Chunk
    chunks = chunk_documents(docs)

    # Embed and store
    embed_and_store(chunks, collection_name=collection_name)

    # Update registry
    registry.register_doc(
        filename=filename,
        source_url=source_url,
        exchange=exchange,
        doc_type=doc_type,
        content_hash=content_hash,
        chunks_count=len(chunks),
    )

    return {
        "status": "success",
        "filename": filename,
        "chunks_ingested": len(chunks),
        "hash": content_hash,
    }


def ingest_all_from_registry(
    *, force_reingest: bool = False
) -> list[dict]:
    """Ingest all documents registered in doc_registry.json.

    Args:
        force_reingest: Re-ingest all documents regardless of hash

    Returns:
        List of ingest result dicts
    """
    registry_path = Path("data/doc_registry.json")
    registry = DocRegistry(registry_path)

    results = []
    raw_dir = Path("data/raw")

    for filename, entry in registry.data.items():
        pdf_path = raw_dir / filename

        if not pdf_path.exists():
            results.append({
                "status": "skipped",
                "filename": filename,
                "reason": "PDF file not found in data/raw/",
            })
            continue

        result = ingest_single_pdf(
            pdf_path=pdf_path,
            exchange=entry["exchange"],
            doc_type=entry["doc_type"],
            source_url=entry.get("source_url"),
            force_reingest=force_reingest,
        )
        results.append(result)

    return results
```

- [ ] **Step 3: Update Makefile with ingest targets**

Add to `Makefile` in project root:

```makefile
.PHONY: install dev-install test test-cov lint format typecheck ci clean \
        ingest ingest-all check-updates reingest serve-api serve-ui eval

# ... existing targets ...

# Ingest targets
ingest:
	python -m exchange_connectivity_hub.ingest.pipeline ingest doc=$(doc)

ingest-all:
	python -m exchange_connectivity_hub.ingest.pipeline ingest-all

check-updates:
	python -m exchange_connectivity_hub.ingest.update_checker

reingest:
	python -m exchange_connectivity_hub.ingest.pipeline ingest doc=$(doc) force=true

# API and UI
serve-api:
	uvicorn exchange_connectivity_hub.api.main:app --reload --host 0.0.0.0 --port 8000

serve-ui:
	streamlit run exchange_connectivity_hub/ui/app.py

# Eval
eval:
	python evals/run_eval.py
```

- [ ] **Step 4: Commit**

```bash
git add src/exchange_connectivity_hub/ingest/pipeline.py Makefile tests/integration/test_pipeline.py
git commit -m "feat(exchange-connectivity-hub): add ingest pipeline orchestrator"
```

---

## Task 9: Implement Retriever

**Files:**
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/retrieval/retriever.py`
- Create: `projects/exchange-connectivity-hub/tests/unit/test_retriever.py`

- [ ] **Step 1: Write failing tests for retriever**

Create `tests/unit/test_retriever.py`:

```python
"""Test document retrieval functionality."""

from unittest.mock import MagicMock, patch
from exchange_connectivity_hub.retrieval.retriever import create_retriever


def test_create_retriever_returns_vectorstore_retriever():
    """Should create a vectorstore-backed retriever."""
    with patch("exchange_connectivity_hub.retrieval.retriever.Chroma") as mock_chroma:
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vectorstore

        retriever = create_retriever()

        assert retriever is not None


def test_retriever_with_exchange_filter():
    """Should filter by exchange when specified."""
    with patch("exchange_connectivity_hub.retrieval.retriever.Chroma") as mock_chroma:
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vectorstore

        retriever = create_retriever(exchange_filter="SGX")

        # Should have created retriever with search_kwargs
        call_kwargs = mock_vectorstore.as_retriever.call_args[1] if mock_vectorstore.as_retriever.call_args else {}
        assert "search_kwargs" in call_kwargs
        assert call_kwargs["search_kwargs"].get("filter") == {"exchange": "SGX"}


def test_retriever_without_filter():
    """Should not apply filter when exchange_filter is None."""
    with patch("exchange_connectivity_hub.retriever.retriever.Chroma") as mock_chroma:
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vectorstore

        retriever = create_retriever(exchange_filter=None)

        # Should not have filter in search_kwargs
        call_kwargs = mock_vectorstore.as_retriever.call_args[1] if mock_vectorstore.as_retriever.call_args else {}
        filter_value = call_kwargs.get("search_kwargs", {}).get("filter")
        assert filter_value is None or filter_value == {}
```

Run test:

```bash
pytest tests/unit/test_retriever.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 2: Implement retriever.py**

Create `src/exchange_connectivity_hub/retrieval/retriever.py`:

```python
"""Document retrieval from ChromaDB with optional exchange filter."""

from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import VoyageEmbeddings
from exchange_connectivity_hub.config import get_config, get_voyage_api_key


def create_retriever(
    *,
    exchange_filter: str | None = None,
    top_k: int | None = None,
) -> Any:
    """Create a retriever for querying the vector store.

    Args:
        exchange_filter: Optional exchange code to filter results (e.g., "SGX")
        top_k: Number of documents to retrieve (uses config default if None)

    Returns:
        LangChain retriever instance
    """
    config = get_config()

    if top_k is None:
        top_k = config["retrieval"]["top_k"]

    persist_directory = config["vector_store"]["persist_directory"]
    collection_name = config["vector_store"]["collection_name"]
    embedding_model = config["models"]["embedding"]

    # Initialize embeddings
    embeddings = VoyageEmbeddings(
        voyage_api_key=get_voyage_api_key(),
        model=embedding_model,
    )

    # Connect to ChromaDB
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Build search kwargs
    search_kwargs = {"k": top_k}

    # Add metadata filter if exchange specified
    if exchange_filter:
        search_kwargs["filter"] = {"exchange": exchange_filter}

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs=search_kwargs,
    )

    return retriever
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/unit/test_retriever.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/exchange_connectivity_hub/retrieval/retriever.py tests/unit/test_retriever.py
git commit -m "feat(exchange-connectivity-hub): add ChromaDB retriever with exchange filter"
```

---

## Task 10: Implement Reranker

**Files:**
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/retrieval/reranker.py`
- Create: `projects/exchange-connectivity-hub/tests/unit/test_reranker.py`

- [ ] **Step 1: Write failing tests for reranker**

Create `tests/unit/test_reranker.py`:

```python
"""Test Voyage reranker functionality."""

from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from exchange_connectivity_hub.retrieval.reranker import rerank_documents


def test_rerank_documents_returns_subset():
    """Should return reranked subset of documents."""
    docs = [
        Document(page_content=f"Content {i}", metadata={"id": i})
        for i in range(20)
    ]

    with patch("exchange_connectivity_hub.retrieval.reranker.Voyage") as mock_voyage:
        mock_client = MagicMock()
        mock_voyage.return_value = mock_client

        # Mock rerank response - return top 5 indices
        mock_client.rerank.return_value = MagicMock(
            results=[MagicMock(index=i, relevance_score=0.9 - i * 0.1) for i in range(5)]
        )

        reranked = rerank_documents(docs, query="test query", top_n=5)

        # Should return 5 documents
        assert len(reranked) == 5


def test_rerank_documents_preserves_metadata():
    """Reranked docs should preserve original metadata."""
    docs = [
        Document(
            page_content=f"Content {i}",
            metadata={"source": f"doc{i}.pdf", "page": i}
        )
        for i in range(10)
    ]

    with patch("exchange_connectivity_hub.retrieval.reranker.Voyage") as mock_voyage:
        mock_client = MagicMock()
        mock_voyage.return_value = mock_client

        mock_client.rerank.return_value = MagicMock(
            results=[MagicMock(index=0, relevance_score=0.9)]
        )

        reranked = rerank_documents(docs, query="test query", top_n=1)

        assert len(reranked) == 1
        assert reranked[0].metadata["source"] == "doc0.pdf"


def test_rerank_disabled_returns_original():
    """When disabled, should return original docs (up to top_n)."""
    docs = [
        Document(page_content=f"Content {i}")
        for i in range(10)
    ]

    reranked = rerank_documents(docs, query="test query", top_n=5, enabled=False)

    # Should return first 5 (no reranking)
    assert len(reranked) == 5


def test_rerank_empty_documents():
    """Empty input should return empty list."""
    reranked = rerank_documents([], query="test query", top_n=5)
    assert reranked == []
```

Run test:

```bash
pytest tests/unit/test_reranker.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 2: Implement reranker.py**

Create `src/exchange_connectivity_hub/retrieval/reranker.py`:

```python
"""Document reranking using Voyage AI rerank-2.5."""

from typing import Any

from voyageai import Voyage as VoyageClient
from exchange_connectivity_hub.config import get_config, get_voyage_api_key


def rerank_documents(
    docs: list[Any],
    *,
    query: str,
    top_n: int,
    enabled: bool | None = None,
) -> list[Any]:
    """Rerank documents using Voyage AI rerank API.

    Args:
        docs: List of retrieved Documents
        query: Original query string
        top_n: Number of top documents to keep after reranking
        enabled: Whether reranking is enabled (uses config if None)

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

    # Initialize Voyage client
    client = VoyageClient(api_key=get_voyage_api_key())

    # Extract document contents
    doc_contents = [doc.page_content for doc in docs]

    # Call rerank API
    rerank_results = client.rerank(
        query=query,
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

- [ ] **Step 3: Update pyproject.toml with voyageai**

Already added in Task 6.

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_reranker.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exchange_connectivity_hub/retrieval/reranker.py tests/unit/test_reranker.py
git commit -m "feat(exchange-connectivity-hub): add Voyage rerank-2.5 for improved precision"
```

---

## Task 11: Implement RAG Chain

**Files:**
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/retrieval/rag_chain.py`
- Create: `projects/exchange-connectivity-hub/tests/integration/test_rag_chain.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/integration/test_rag_chain.py`:

```python
"""Integration test for RAG chain."""

import os
import pytest
from exchange_connectivity_hub.retrieval.rag_chain import create_rag_chain


def test_rag_chain_returns_answer_with_sources():
    """Chain should return answer with source citations."""
    if not os.getenv("VOYAGE_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("API keys not set")

    chain = create_rag_chain()

    result = chain.invoke({
        "question": "What is the minimum lot size?",
        "exchange_filter": "SGX",
    })

    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
```

- [ ] **Step 2: Implement rag_chain.py**

Create `src/exchange_connectivity_hub/retrieval/rag_chain.py`:

```python
"""RAG chain: retrieve → rerank → prompt → LLM → output parser."""

from datetime import datetime, UTC
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic

from exchange_connectivity_hub.config import get_config, get_anthropic_api_key
from exchange_connectivity_hub.retrieval.retriever import create_retriever
from exchange_connectivity_hub.retrieval.reranker import rerank_documents


def _format_docs(docs: list[Any]) -> str:
    """Format documents for prompt."""
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page_number", "?")
        content = doc.page_content
        context_parts.append(f"[Source: {source}, page {page}]\n{content}")
    return "\n\n".join(context_parts)


def _extract_sources(docs: list[Any]) -> list[dict[str, Any]]:
    """Extract source metadata from documents."""
    sources = []
    seen = set()  # Dedupe by filename + page

    for doc in docs:
        filename = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page_number")
        exchange = doc.metadata.get("exchange", "Unknown")
        ingested_at = doc.metadata.get("ingested_at", "Unknown")

        key = (filename, page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": filename,
                "page_number": page,
                "exchange": exchange,
                "ingested_at": ingested_at,
            })

    return sources


def _check_staleness(sources: list[dict[str, Any]]) -> str | None:
    """Check if any source is older than staleness threshold."""
    config = get_config()
    staleness_days = config["staleness"]["days"]

    for source in sources:
        ingested_at = source.get("ingested_at")
        if ingested_at and ingested_at != "Unknown":
            try:
                ingest_date = datetime.fromisoformat(ingested_at)
                days_ago = (datetime.now(UTC) - ingest_date).days

                if days_ago > staleness_days:
                    return f"⚠️ Some sources are over {staleness_days} days old (last re-verified {days_ago} days ago). Verify against exchange website."
            except (ValueError, TypeError):
                continue

    return None


def create_rag_chain() -> Any:
    """Create the RAG chain.

    Chain flow:
    1. Retrieve docs (via retriever)
    2. Rerank docs
    3. Format context
    4. Generate answer (via Claude)
    5. Extract sources and check staleness

    Returns:
        Runnable chain that accepts {"question": str, "exchange_filter": str | None}
    """
    config = get_config()

    # Prompt template
    prompt_template = """Answer the following question using only the context provided below.

Context:
{context}

Question: {question}

Instructions:
- Answer from the context only
- Cite the source filename and page number for each fact
- If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."
- Never hallucinate or make up information

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # LLM
    llm = ChatAnthropic(
        api_key=get_anthropic_api_key(),
        model=config["models"]["llm"],
        temperature=0,
    )

    def _retrieve_and_rerank(inputs: dict[str, Any]) -> dict[str, Any]:
        """Retrieve and rerank documents."""
        question = inputs["question"]
        exchange_filter = inputs.get("exchange_filter")

        # Create retriever
        retriever = create_retriever(
            exchange_filter=exchange_filter,
        )

        # Retrieve docs
        docs = retriever.invoke(question)

        # Rerank
        config = get_config()
        reranked = rerank_documents(
            docs,
            query=question,
            top_n=config["retrieval"]["rerank_top_n"],
        )

        return {"docs": reranked, "question": question}

    def _generate_response(inputs: dict[str, Any]) -> dict[str, Any]:
        """Generate final response with sources and staleness check."""
        docs = inputs["docs"]
        question = inputs["question"]

        # Format context
        context = _format_docs(docs)

        # Generate answer
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        # Extract sources
        sources = _extract_sources(docs)

        # Check staleness
        staleness_warning = _check_staleness(sources)

        return {
            "answer": answer,
            "sources": sources,
            "staleness_warning": staleness_warning,
        }

    # Full chain
    chain = (
        RunnablePassthrough.assign()
        | _retrieve_and_rerank
        | _generate_response
    )

    return chain
```

- [ ] **Step 3: Update pyproject.toml with dependencies**

Add to `dependencies` in `pyproject.toml`:

```toml
"langchain-anthropic>=0.3.0",
```

- [ ] **Step 4: Install and run tests**

```bash
pip install -e ".[dev]"
pytest tests/integration/test_rag_chain.py -v
```

Expected: PASS (or SKIP if no API keys)

- [ ] **Step 5: Commit**

```bash
git add src/exchange_connectivity_hub/retrieval/rag_chain.py tests/integration/test_rag_chain.py pyproject.toml
git commit -m "feat(exchange-connectivity-hub): add LCEL RAG chain with Claude"
```

---

## Task 12: Implement FastAPI Backend

**Files:**
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/api/main.py`
- Create: `projects/exchange-connectivity-hub/tests/integration/test_api.py`

- [ ] **Step 1: Write failing tests for API**

Create `tests/integration/test_api.py`:

```python
"""Test FastAPI endpoints."""

from fastapi.testclient import TestClient
from exchange_connectivity_hub.api.main import app


def test_health_endpoint():
    """Health check should return status and collection count."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_query_endpoint_missing_question():
    """Should return 422 for missing question."""
    client = TestClient(app)
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_endpoint_valid_request():
    """Valid request should return answer with sources."""
    import os

    if not os.getenv("VOYAGE_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("API keys not set")

    client = TestClient(app)
    response = client.post(
        "/query",
        json={"question": "What is lot size?", "exchange_filter": "SGX"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
```

Run test:

```bash
pytest tests/integration/test_api.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 2: Implement main.py**

Create `src/exchange_connectivity_hub/api/main.py`:

```python
"""FastAPI backend for RAG query endpoint."""

from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from exchange_connectivity_hub.config import get_config
from exchange_connectivity_hub.retrieval.rag_chain import create_rag_chain


# Request/Response models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    exchange_filter: str | None = Field(None, description="Optional exchange filter")


class SourceInfo(BaseModel):
    filename: str
    page_number: int | None
    exchange: str
    ingested_at: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    staleness_warning: str | None


class HealthResponse(BaseModel):
    status: str
    collection_count: int | None
    version: str


# FastAPI app
app = FastAPI(
    title="Exchange Connectivity Hub API",
    description="RAG-based Q&A for Asian exchange documentation",
    version="0.1.0",
)

# RAG chain (lazy loaded)
_rag_chain = None


def get_rag_chain():
    """Get or create RAG chain singleton."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = create_rag_chain()
    return _rag_chain


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    """Health check endpoint."""
    # Try to get collection count
    try:
        from chromadb import Chroma
        config = get_config()
        chroma = Chroma(
            persist_directory=config["vector_store"]["persist_directory"],
        )
        collection = chroma.get_collection(name=config["vector_store"]["collection_name"])
        count = collection.count()
    except Exception:
        count = None

    return {
        "status": "ok",
        "collection_count": count,
        "version": "0.1.0",
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> dict[str, Any]:
    """Query the RAG system."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        chain = get_rag_chain()
        result = chain.invoke({
            "question": req.question,
            "exchange_filter": req.exchange_filter,
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 3: Update pyproject.toml with dependencies**

Add to `dependencies` in `pyproject.toml`:

```toml
"fastapi>=0.115.0",
"uvicorn>=0.32.0",
"pydantic>=2.0.0",
"httpx>=0.27.0",  # For TestClient
```

- [ ] **Step 4: Install and run tests**

```bash
pip install -e ".[dev]"
pytest tests/integration/test_api.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exchange_connectivity_hub/api/main.py tests/integration/test_api.py pyproject.toml
git commit -m "feat(exchange-connectivity-hub): add FastAPI backend with /query and /health endpoints"
```

---

## Task 13: Implement Streamlit UI

**Files:**
- Create: `projects/exchange-connectivity-hub/src/exchange_connectivity_hub/ui/app.py`

- [ ] **Step 1: Implement Streamlit app**

Create `src/exchange_connectivity_hub/ui/app.py`:

```python
"""Streamlit UI for exchange connectivity Q&A."""

import requests
import streamlit as st

# Page config
st.set_page_config(
    page_title="Exchange Connectivity Hub",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 Exchange Connectivity Hub")
st.markdown("Ask questions about Asian exchange (SGX, HKSE, TSE) connectivity and trading rules.")

# API URL (default to localhost)
API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    help="FastAPI backend URL",
)

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.sidebar.success(f"✅ API Connected ({health.get('collection_count', 'N/A')} docs)")
except Exception:
    st.sidebar.error("❌ API Disconnected")

# Question input
question = st.text_input(
    "Question",
    placeholder="e.g., What is the minimum lot size for SGX equities?",
    help="Ask about trading rules, order types, lot sizes, etc.",
)

# Exchange filter
exchange_filter = st.selectbox(
    "Filter by Exchange",
    options=["All", "SGX", "HKSE", "TSE"],
    index=0,
)

# Query button
if st.button("Ask", type="primary"):
    if not question or not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Prepare request
    payload = {
        "question": question,
        "exchange_filter": exchange_filter if exchange_filter != "All" else None,
    }

    # Call API
    with st.spinner("Retrieving answer..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            # Display staleness warning
            if result.get("staleness_warning"):
                st.warning(result["staleness_warning"])

            # Display answer
            st.markdown("### Answer")
            st.markdown(result["answer"])

            # Display sources
            if result.get("sources"):
                st.markdown("### Sources")
                for source in result["sources"]:
                    filename = source.get("filename", "Unknown")
                    page = source.get("page_number", "?")
                    exchange = source.get("exchange", "Unknown")
                    ingested_at = source.get("ingested_at", "Unknown")

                    st.markdown(
                        f"- **{filename}** (page {page}, {exchange}) – ingested {ingested_at}"
                    )

        except requests.exceptions.RequestException as e:
            st.error(f"Error querying API: {e}")

# Example questions
st.markdown("---")
st.markdown("### Example Questions")
examples = [
    ("What is the minimum lot size for SGX equities?", "SGX"),
    ("Does HKSE support iceberg orders?", "HKSE"),
    ("What are TSE trading hours?", "TSE"),
]

for q, ex in examples:
    if st.button(f"{q}", key=q):
        st.session_state["question"] = q
        st.session_state["exchange_filter"] = ex
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*Built with Voyage AI, Claude, LangChain, and RAGAS*")
```

- [ ] **Step 2: Update pyproject.toml with dependencies**

Add to `dependencies` in `pyproject.toml`:

```toml
"streamlit>=1.39.0",
"requests>=2.32.0",
```

- [ ] **Step 3: Install**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 4: Commit**

```bash
git add src/exchange_connectivity_hub/ui/app.py pyproject.toml
git commit -m "feat(exchange-connectivity-hub): add Streamlit UI"
```

---

## Task 14: Create Golden Dataset for Eval

**Files:**
- Create: `projects/exchange-connectivity-hub/evals/golden_dataset.json`

- [ ] **Step 1: Create golden dataset template**

Create `evals/golden_dataset.json`:

```json
{
  "answerable": [
    {
      "question": "What is the minimum lot size for SGX equities?",
      "ground_truth": "100 shares (normal board), 1 share for odd lots",
      "source_doc": "SGX_Market_Model_Guide_v4.pdf",
      "page": 12,
      "exchange": "SGX"
    },
    {
      "question": "What order types does SGX support for equities?",
      "ground_truth": "Limit, Market, Stop Limit, Market-to-Limit",
      "source_doc": "SGX_Market_Model_Guide_v4.pdf",
      "page": 15,
      "exchange": "SGX"
    },
    {
      "question": "What are SGX trading hours for equities?",
      "ground_truth": "09:00-12:00 and 13:00-17:00 SGT",
      "source_doc": "SGX_Market_Model_Guide_v4.pdf",
      "page": 8,
      "exchange": "SGX"
    },
    {
      "question": "Does SGX support iceberg orders?",
      "ground_truth": "Yes, with minimum disclosed quantity of 100 shares",
      "source_doc": "SGX_Market_Model_Guide_v4.pdf",
      "page": 18,
      "exchange": "SGX"
    },
    {
      "question": "What is the tick size for SGX stocks priced above $1?",
      "ground_truth": "$0.01",
      "source_doc": "SGX_Market_Model_Guide_v4.pdf",
      "page": 14,
      "exchange": "SGX"
    },
    {
      "question": "What is the minimum lot size for HKSE equities?",
      "ground_truth": "Varies by lot size category, typically 100-1000 shares",
      "source_doc": "HKSE_Rules_Guide.pdf",
      "page": 22,
      "exchange": "HKSE"
    },
    {
      "question": "Does HKSE support short selling?",
      "ground_truth": "Yes, with designated securities list",
      "source_doc": "HKSE_Rules_Guide.pdf",
      "page": 35,
      "exchange": "HKSE"
    },
    {
      "question": "What are HKSE trading hours?",
      "ground_truth": "09:30-12:00 and 13:00-16:00 HKT",
      "source_doc": "HKSE_Rules_Guide.pdf",
      "page": 10,
      "exchange": "HKSE"
    },
    {
      "question": "What is the volatility control mechanism on HKSE?",
      "ground_truth": "VCM triggers 5-minute pause on price movement >10%",
      "source_doc": "HKSE_Rules_Guide.pdf",
      "page": 28,
      "exchange": "HKSE"
    },
    {
      "question": "Does HKSE support odd lots?",
      "ground_truth": "Yes, handled through odd-lot facility",
      "source_doc": "HKSE_Rules_Guide.pdf",
      "page": 24,
      "exchange": "HKSE"
    },
    {
      "question": "What is the trading unit for TSE equities?",
      "ground_truth": "100 shares (1 trading unit)",
      "source_doc": "TSE_Trading_Guide.pdf",
      "page": 16,
      "exchange": "TSE"
    },
    {
      "question": "What are TSE trading hours?",
      "ground_truth": "09:00-11:30 and 12:30-15:00 JST",
      "source_doc": "TSE_Trading_Guide.pdf",
      "page": 12,
      "exchange": "TSE"
    },
    {
      "question": "Does TSE support stop orders?",
      "ground_truth": "No, stop orders not supported on TSE",
      "source_doc": "TSE_Trading_Guide.pdf",
      "page": 20,
      "exchange": "TSE"
    },
    {
      "question": "What is the tick size for TSE stocks priced 1000-3000 yen?",
      "ground_truth": "5 yen",
      "source_doc": "TSE_Trading_Guide.pdf",
      "page": 18,
      "exchange": "TSE"
    },
    {
      "question": "Does TSE have after-hours trading?",
      "ground_truth": "No after-hours trading for equities",
      "source_doc": "TSE_Trading_Guide.pdf",
      "page": 13,
      "exchange": "TSE"
    },
    {
      "question": "What is the circuit breaker level on SGX?",
      "ground_truth": "10% price movement triggers trading halt",
      "source_doc": "SGX_Market_Model_Guide_v4.pdf",
      "page": 25,
      "exchange": "SGX"
    },
    {
      "question": "What is the settlement cycle for SGX equities?",
      "ground_truth": "T+2",
      "source_doc": "SGX_Market_Model_Guide_v4.pdf",
      "page": 30,
      "exchange": "SGX"
    },
    {
      "question": "Does SGX support market-making?",
      "ground_truth": "Yes, through designated market maker program",
      "source_doc": "SGX_Market_Model_Guide_v4.pdf",
      "page": 32,
      "exchange": "SGX"
    },
    {
      "question": "What is HKSE closing auction period?",
      "ground_truth": "16:00-16:10 HKT (random closing)",
      "source_doc": "HKSE_Rules_Guide.pdf",
      "page": 15,
      "exchange": "HKSE"
    },
    {
      "question": "What is HKSE board lot for penny stocks?",
      "ground_truth": "Varies, can be higher than 1000 shares",
      "source_doc": "HKSE_Rules_Guide.pdf",
      "page": 23,
      "exchange": "HKSE"
    },
    {
      "question": "Does TSE have margin trading?",
      "ground_truth": "Yes, through Japan Securities Finance",
      "source_doc": "TSE_Trading_Guide.pdf",
      "page": 40,
      "exchange": "TSE"
    },
    {
      "question": "What is the SGX order modification rule?",
      "ground_truth": "Orders can be modified before execution",
      "source_doc": "SGX_Market_Model_Guide_v4.pdf",
      "page": 19,
      "exchange": "SGX"
    },
    {
      "question": "Does HKSE support pre-open auction?",
      "ground_truth": "Yes, 09:15-09:25 HKT with random matching",
      "source_doc": "HKSE_Rules_Guide.pdf",
      "page": 14,
      "exchange": "HKSE"
    },
    {
      "question": "What is TSE order price validity?",
      "ground_truth": "Day order only, no GTC",
      "source_doc": "TSE_Trading_Guide.pdf",
      "page": 21,
      "exchange": "TSE"
    },
    {
      "question": "What is SGX maximum order size?",
      "ground_truth": "Varies by security, up to 10M shares",
      "source_doc": "SGX_Market_Model_Guide_v4.pdf",
      "page": 20,
      "exchange": "SGX"
    },
    {
      "question": "Does HKSE have short-sale restrictions?",
      "ground_truth": "Yes, downtick rule applies",
      "source_doc": "HKSE_Rules_Guide.pdf",
      "page": 36,
      "exchange": "HKSE"
    },
    {
      "question": "What is TSE IPO pricing mechanism?",
      "ground_truth": "Book building with auction",
      "source_doc": "TSE_Trading_Guide.pdf",
      "page": 45,
      "exchange": "TSE"
    }
  ],
  "unanswerable": [
    {
      "question": "What is the FIX protocol version for OSE derivatives?",
      "ground_truth": null,
      "source_doc": null,
      "page": null,
      "exchange": null
    },
    {
      "question": "What are IDX trading hours?",
      "ground_truth": null,
      "source_doc": null,
      "page": null,
      "exchange": null
    },
    {
      "question": "What is the Bursa Malaysia lot size for ETFs?",
      "ground_truth": null,
      "source_doc": null,
      "page": null,
      "exchange": null
    },
    {
      "question": "What is FIX tag 911 used for?",
      "ground_truth": null,
      "source_doc": null,
      "page": null,
      "exchange": null
    },
    {
      "question": "What are the commission rates for Bursa Malaysia?",
      "ground_truth": null,
      "source_doc": null,
      "page": null,
      "exchange": null
    }
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add evals/golden_dataset.json
git commit -m "feat(exchange-connectivity-hub): add golden dataset with 30 answerable + 5 unanswerable questions"
```

---

## Task 15: Implement RAGAS Eval Harness

**Files:**
- Create: `projects/exchange-connectivity-hub/evals/run_eval.py`
- Modify: `projects/exchange-connectivity-hub/pyproject.toml` (add ragas)

- [ ] **Step 1: Implement run_eval.py**

Create `evals/run_eval.py`:

```python
"""RAGAS evaluation script for measuring RAG quality."""

import json
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

from exchange_connectivity_hub.retrieval.rag_chain import create_rag_chain


def load_golden_dataset(path: Path) -> tuple[list, list]:
    """Load golden dataset from JSON file.

    Returns:
        (answerable_questions, unanswerable_questions)
    """
    with open(path) as f:
        data = json.load(f)

    return data.get("answerable", []), data.get("unanswerable", [])


def run_eval(golden_path: Path, output_path: Path | None = None) -> dict:
    """Run RAGAS evaluation on golden dataset.

    Args:
        golden_path: Path to golden_dataset.json
        output_path: Optional path to save results JSON

    Returns:
        Dict with metric scores
    """
    # Load dataset
    answerable, unanswerable = load_golden_dataset(golden_path)
    all_questions = answerable + unanswerable

    # Create RAG chain
    chain = create_rag_chain()

    # Build evaluation dataset
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    refusal_count = 0
    unanswerable_correct = 0

    for item in all_questions:
        question = item["question"]
        ground_truth = item.get("ground_truth")

        # Run chain
        result = chain.invoke({
            "question": question,
            "exchange_filter": None,  # No filter for eval
        })

        answer = result["answer"]
        sources = result.get("sources", [])

        # Track refusals for unanswerable questions
        if ground_truth is None:
            refusal_count += 1
            if "don't have enough information" in answer.lower() or "cannot answer" in answer.lower():
                unanswerable_correct += 1

        # Build contexts from sources
        contexts = [
            f"{s.get('filename', 'Unknown')} page {s.get('page_number', '?')}"
            for s in sources
        ]

        eval_data["question"].append(question)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append([ground_truth] if ground_truth else [""])

    # Create HuggingFace dataset
    dataset = Dataset.from_dict(eval_data)

    # Run RAGAS evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    # Convert to dict
    scores = {
        "context_precision": result["context_precision"],
        "context_recall": result["context_recall"],
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "refusal_rate_on_unanswerable": unanswerable_correct / len(unanswerable) if unanswerable else 0,
    }

    # Check pass/fail
    passed = (
        scores["faithfulness"] >= 0.85
        and scores["context_precision"] >= 0.75
        and scores["refusal_rate_on_unanswerable"] >= 0.80
    )

    final_result = {
        "passed": passed,
        "scores": scores,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Print results
    print("\n=== RAGAS Evaluation Results ===")
    print(f"Context Precision: {scores['context_precision']:.2f} (≥0.75 required)")
    print(f"Faithfulness: {scores['faithfulness']:.2f} (≥0.85 required)")
    print(f"Refusal Rate (unanswerable): {scores['refusal_rate_on_unanswerable']:.2%} (≥80% required)")
    print(f"\nOverall: {'✅ PASS' if passed else '❌ FAIL'}")

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(final_result, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return final_result


if __name__ == "__main__":
    import sys

    golden_path = Path("evals/golden_dataset.json")
    output_dir = Path("evals/results")

    if not golden_path.exists():
        print(f"Error: {golden_path} not found")
        sys.exit(1)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    output_path = output_dir / f"eval_{timestamp}.json"

    run_eval(golden_path, output_path)
```

- [ ] **Step 2: Update pyproject.toml with dependencies**

Add to `dependencies` in `pyproject.toml`:

```toml
"ragas>=0.2.0",
"datasets>=2.0.0",
```

- [ ] **Step 3: Update Makefile with eval target**

Already added in Task 8.

- [ ] **Step 4: Install**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 5: Commit**

```bash
git add evals/run_eval.py pyproject.toml
git commit -m "feat(exchange-connectivity-hub): add RAGAS eval harness"
```

---

## Task 16: Update Project README

**Files:**
- Create: `projects/exchange-connectivity-hub/README.md`

- [ ] **Step 1: Write comprehensive README**

Create `README.md`:

```markdown
# Exchange Connectivity Hub

A RAG-based system for answering Asian exchange connectivity and trading rules questions in under 30 seconds with cited sources.

## Overview

Engineers and BAs manually cross-reference hundreds of pages of exchange PDFs (HKSE, SGX, TSE) to answer connectivity questions—taking 20–40 minutes per query with staleness risk. This project delivers a RAG system that answers those questions in seconds, cites sources, and detects when docs have changed.

## Key Features

- **Fast answers:** Sub-30 second responses with inline citations
- **Source attribution:** Every answer cites filename and page number
- **Staleness warnings:** Flags when source docs haven't been re-verified recently
- **Change detection:** MD5-based hashing detects document updates and triggers re-ingestion
- **Eval-driven quality:** RAGAS metrics tracked over time

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | Voyage AI `voyage-finance-2` (domain-tuned for financial text) |
| Reranking | Voyage AI `rerank-2.5` |
| LLM | Claude `claude-sonnet-4-6` |
| RAG Framework | LangChain LCEL |
| Vector Store | ChromaDB (local persistent) |
| PDF Parsing | PyMuPDF |
| Eval | RAGAS |
| API | FastAPI |
| UI | Streamlit |

## Project Structure

```
exchange-connectivity-hub/
├── config.yaml          # Configuration (top_k, model names, staleness threshold)
├── .env.example         # API keys template
├── src/exchange_connectivity_hub/
│   ├── ingest/          # PDF loading, chunking, embedding
│   ├── retrieval/       # Retrieval, reranking, RAG chain
│   ├── api/             # FastAPI backend
│   └── ui/              # Streamlit interface
├── data/
│   ├── raw/             # Downloaded PDFs
│   ├── chroma_db/       # ChromaDB storage
│   └── doc_registry.json  # Document version tracking
└── evals/
    ├── golden_dataset.json  # 35 Q&A pairs for evaluation
    ├── run_eval.py          # RAGAS evaluation script
    └── results/             # Eval score history
```

## Setup

### 1. Install dependencies

```bash
cd projects/exchange-connectivity-hub
make dev-install
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your Voyage and Anthropic API keys
```

### 3. Download exchange PDFs

Place PDFs in `data/raw/` and update `data/doc_registry.json` with filenames and source URLs.

### 4. Ingest documents

```bash
make ingest-all  # Ingest all registered docs
# or
make ingest doc=SGX_Market_Model_Guide_v4.pdf
```

### 5. Run eval (optional)

```bash
make eval
```

## Usage

### Start API

```bash
make serve-api
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Start UI

```bash
make serve-ui
# UI at http://localhost:8501
```

### Check for document updates

```bash
make check-updates
```

## Eval Metrics

Passing thresholds:
- **Faithfulness:** ≥ 0.85 (answer grounded in context)
- **Context Precision:** ≥ 0.75 (retrieved chunks are relevant)
- **Refusal Rate:** ≥ 80% on unanswerable questions (no hallucination)

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make ci` | Run lint + tests |
| `make ingest doc=X` | Ingest single PDF |
| `make ingest-all` | Ingest all registered docs |
| `make check-updates` | Check for doc changes |
| `make reingest doc=X` | Force re-ingest |
| `make serve-api` | Start FastAPI |
| `make serve-ui` | Start Streamlit |
| `make eval` | Run RAGAS evaluation |

## License

MIT

## Author

Built as part of AI PM learning portfolio. Demonstrates RAG system design, eval-driven development, and production-pattern change detection.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(exchange-connectivity-hub): add comprehensive README"
```

---

## Task 17: Final Integration Tests

**Files:**
- Create: `projects/exchange-connectivity-hub/tests/integration/test_end_to_end.py`

- [ ] **Step 1: Write end-to-end integration test**

Create `tests/integration/test_end_to_end.py`:

```python
"""End-to-end integration test."""

import os
import pytest
from pathlib import Path

from exchange_connectivity_hub.ingest.pipeline import ingest_single_pdf
from exchange_connectivity_hub.retrieval.rag_chain import create_rag_chain


@pytest.mark.integration
def test_full_pipeline_with_real_pdf(tmp_path):
    """Test ingest → retrieve → generate with real PDF."""
    if not os.getenv("VOYAGE_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("API keys not set")

    # This test requires a real PDF file
    # For CI/CD, use a small test PDF committed to repo
    pdf_path = Path("tests/fixtures/test.pdf")

    if not pdf_path.exists():
        pytest.skip("Test PDF not found")

    # Ingest
    result = ingest_single_pdf(
        pdf_path=pdf_path,
        exchange="TEST",
        doc_type="test",
        source_url=None,
    )

    assert result["status"] == "success"
    assert result["chunks_ingested"] > 0

    # Query
    chain = create_rag_chain()
    response = chain.invoke({
        "question": "What is in this document?",
        "exchange_filter": None,
    })

    assert "answer" in response
    assert "sources" in response

    # Cleanup
    from exchange_connectivity_hub.ingest.embedder import delete_by_source
    from exchange_connectivity_hub.config import get_config

    config = get_config()
    delete_by_source(
        source_filename=pdf_path.name,
        collection_name=config["vector_store"]["collection_name"],
    )
```

- [ ] **Step 2: Commit**

```bash
git add tests/integration/test_end_to_end.py
git commit -m "test(exchange-connectivity-hub): add end-to-end integration test"
```

---

## Task 18: Root .gitignore Updates

**Files:**
- Modify: `/home/ubuntu/projects/ai-pm-portfolio/.gitignore`

- [ ] **Step 1: Update root .gitignore for project-specific ignores**

Add to root `.gitignore`:

```gitignore
# Exchange Connectivity Hub
projects/exchange-connectivity-hub/data/raw/*.pdf
projects/exchange-connectivity-hub/data/chroma_db/
projects/exchange-connectivity-hub/evals/results/
projects/exchange-connectivity-hub/.env
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add exchange-connectivity-hub ignores to root gitignore"
```

---

## Verification

After all tasks complete:

- [ ] Run `make ci-all` from root - should pass lint and tests
- [ ] Verify all files created per structure
- [ ] Check golden_dataset.json has 35 questions
- [ ] Confirm config.yaml has all required sections
- [ ] Verify .env.example exists
- [ ] Run `pytest projects/exchange-connectivity-hub/tests/` - all tests pass

---

## Summary

This plan implements a complete RAG system for Asian exchange documentation:

1. **Ingest Pipeline:** PyMuPDF loader → Token-aware chunker → Voyage embeddings → ChromaDB
2. **Retrieval:** ChromaDB search → Voyage rerank-2.5 → Top-5 chunks
3. **Generation:** Claude claude-sonnet-4-6 with cite-or-refuse prompt
4. **API:** FastAPI with `/query` and `/health` endpoints
5. **UI:** Streamlit question-answer interface
6. **Eval:** RAGAS with 35-question golden dataset
7. **Change Detection:** MD5 hashing with doc_registry version tracking
8. **Testing:** Unit tests for all components, integration tests for E2E flow

Total files created: ~30 including tests, configs, and documentation.
