# Execution Quality Copilot (TCA-over-MCP) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a custom **MCP server** that exposes a fund's (synthetic) execution data as five read-only, bounded tools so a PM in Claude Desktop can self-serve Transaction Cost Analysis (TCA) questions in natural language — plus a **tool-use eval harness** that scores tool selection, parameter accuracy, and numeric-answer accuracy.

**Architecture:** A deterministic synthetic generator writes FIX-style ExecutionReports (35=8) into a DuckDB file. Five **pure Python functions** (`get_fills`, `get_benchmarks`, `compute_slippage`, `venue_breakdown`, `top_outliers`) run bounded, parameterised SQL — never raw SQL from the model. A thin **FastMCP** layer wraps each pure function as an `@mcp.tool()` with a rich, prompt-quality description, served over stdio for Claude Desktop. The eval harness reuses the *same* pure functions: it drives a manual Anthropic tool-use loop, intercepts the tool calls, executes them against DuckDB, and scores the model's behaviour.

**Tech Stack:** Python 3.10 · **MCP Python SDK / FastMCP** (`mcp`) · **DuckDB** (in-process OLAP) · Anthropic SDK (`claude-opus-4-8` eval agent) · pytest · Docker.

---

## Context

This is **Portfolio Project #3** (`docs/execution-quality-copilot-spec.md`). The user's instruction for this plan: **where the spec's structure conflicts with the established project structure, the project structure wins.** Two conflicts are resolved that way:

1. **Location.** The spec says `ai-pm-portfolio/execution-quality-copilot/` (repo root). The monorepo convention (CLAUDE.md + the two existing projects `projects/exchange-connectivity-hub/` and `projects/pre-trade-risk-rules-assistant/`) puts every project under `projects/<slug>/` with a `src/` package. → **`projects/execution-quality-copilot/`** with `src/execution_quality_copilot/`.
2. **Layout.** The spec's flat `server/main.py` / `server/tools/` / `db.py` / `data/generate_fills.py` becomes the src-layout equivalents under `src/execution_quality_copilot/` (mapped below). The spec's committed `data/seed.duckdb` becomes a **gitignored, deterministically regenerable** artifact (the monorepo rule: "keep volatile/generated files out of git").

**Decisions locked in:**
- **Scope: MVP only** (spec §"MVP Scope"). The 5 tools, 10k synthetic fills / 50 symbols / 4 brokers / 3 algos, a 15-question tool-use eval, a Dockerfile, and a `tool-design-notes.md`. Stretch goals (real FIX drop-copy, n8n digest, futures, anomaly detection) are an outline at the end, not tasks.
- **Linear: one issue + one branch** — create a single "Execution Quality Copilot MVP" Linear issue; work on `feat/LIN-141-execution-quality-copilot-mvp`. The branch name satisfies the pre-commit Linear-key hook for every per-task commit. Replace `LIN-141` (e.g. `LIN-141`) everywhere below.
- **Eval agent model: `claude-opus-4-8`** (the claude-api skill default; strongest tool-selection). Configurable in `config.yaml`.
- **Seed data is regenerable, not committed.** `make gen-data` rebuilds an identical `data/seed.duckdb` from a fixed RNG seed, so any reviewer reproduces the exact numbers with one command.

**Independent view & recommendation.** The genuinely new skill here is **tool design**, not analytics — so the plan deliberately splits each tool into (a) a *pure function* that takes a DuckDB connection + typed filters and runs whitelisted, parameterised SQL, and (b) a *thin FastMCP wrapper*. Three payoffs: the pure functions are unit-testable with an in-memory DuckDB and **no API key**; the eval harness can call the exact same functions the MCP server exposes (no drift between "what we tested" and "what we shipped"); and the read-only / bounded / enum-whitelist guardrails live in one place. The one cost is a *second* source of tool schemas — FastMCP derives schemas from type hints, while the eval harness hand-writes Anthropic-format `TOOL_DEFS`. For an MVP that is acceptable and explicit; a production system would derive both from one source. I recommend keeping the two in sync by hand and noting it in `tool-design-notes.md` as a real design tension (good interview material).

> **Teacher's note — why this is different from P1 (RAG) and P2 (LangGraph).** In P1 you pushed documents *into* the model's context; in P2 you drove the model through a *fixed graph*. Here the model **chooses** which tool to call — your product surface is the **tool contract** (names, descriptions-as-prompts, parameter schemas, guardrails), exactly like publishing a FIX order-entry spec: the spec determines what the counterparty (here, the LLM) can and cannot do. MCP is that spec for LLMs. The server is "dumb" (it never calls an LLM); the only LLM call in this whole project is inside the *eval harness*, which simulates the Claude Desktop client so you can measure tool-selection quality before release.

---

## File Structure

```
projects/execution-quality-copilot/
├── README.md                                  # T12  setup, Claude Desktop registration, eval metrics, compliance caveat
├── Dockerfile                                 # T11  one-command run of the MCP server
├── .dockerignore                              # T11
├── pyproject.toml                             # T0   deps (mcp, duckdb, anthropic), ruff, mypy, pytest cov target
├── Makefile                                   # T0   ci/test/lint/gen-data/serve/eval
├── config.yaml                                # T0   models, storage paths, generator knobs
├── .env.example                               # T0   ANTHROPIC_API_KEY
├── src/execution_quality_copilot/
│   ├── __init__.py
│   ├── config.py                              # T1   YAML + dotenv, get_anthropic_api_key()
│   ├── models.py                              # T2   Literal enums (Broker/Algo/Venue/Benchmark/Tier/Side) + Fill dataclass
│   ├── db.py                                  # T4   connect(), query() bounded helper, _serialize(), BENCHMARK_COLUMNS, MAX_ROWS
│   ├── datagen/
│   │   ├── __init__.py
│   │   ├── __main__.py                        # T3   `python -m execution_quality_copilot.datagen`
│   │   └── generate.py                        # T3   deterministic FIX ExecutionReport generator → DuckDB
│   └── server/
│       ├── __init__.py
│       ├── main.py                            # T8   FastMCP instance + 5 @mcp.tool() wrappers + mcp.run()
│       └── tools/
│           ├── __init__.py
│           ├── fills.py                       # T5   get_fills, venue_breakdown   (pure functions)
│           ├── benchmarks.py                  # T6   get_benchmarks               (pure function)
│           └── tca.py                         # T7   compute_slippage, top_outliers (pure functions)
├── llm.py  → src/execution_quality_copilot/llm.py   # T9  Anthropic manual tool-use loop (eval client only)
├── evals/
│   ├── __init__.py
│   ├── golden_questions.json                  # T10  15 PM questions + expected tool/params/answer_key
│   ├── run_eval.py                            # T10  TOOL_DEFS + dispatch + scoring + report
│   └── results/.gitkeep                        # T10
├── data/                                      # gitignored: seed.duckdb (regenerable via make gen-data)
└── tests/{unit,integration}/                  # one test file per module
```

**Shared type contract (defined in T2/T4, referenced everywhere):**
- Tool param **enums** (`models.py`): `SIDE = {"BUY","SELL"}`, `BROKERS = ["ALPHA","BRAVO","COBALT","DELTA"]`, `ALGOS = ["VWAP","TWAP","IS"]`, `VENUES = ["XNAS","XNYS","BATS","EDGX","DARK"]`, `TIERS = ["large","mid","small"]`, `BENCHMARKS = ["arrival","vwap","close"]`.
- **`fills` table columns**: `exec_id, order_id, symbol, mkt_cap_tier, side, broker, algo, venue, transact_time, trade_date, last_qty, last_px, arrival_px, interval_vwap, close_px, currency`.
- **`db.BENCHMARK_COLUMNS`**: `{"arrival":"arrival_px", "vwap":"interval_vwap", "close":"close_px"}` — the model picks a benchmark *enum*; we map it to a known column. Tools **never** interpolate model-supplied strings into SQL except via this whitelist.
- **`db.MAX_ROWS = 500`** — hard cap on every result set.
- **Pure tool signature**: `fn(conn, *, <typed kwargs>) -> dict`. Every dict is JSON-serializable (dates → ISO strings, floats rounded) and carries a `caveat` string where a number could be misread.
- **Slippage convention (bps, cost-positive):** per fill, `slip_bps = side_sign * (last_px - bench_px) / bench_px * 10000` with `side_sign = +1` for BUY, `-1` for SELL. Positive = worse than benchmark. Aggregates are **notional-weighted** (`last_qty * last_px`).

**Tool surface (the product):**
```
get_fills(symbol?, broker?, algo?, venue?, side?, start_date?, end_date?, mkt_cap_tier?, limit=100)
get_benchmarks(symbol, date)
compute_slippage(symbol?, broker?, algo?, side?, start_date?, end_date?, benchmark="arrival", mkt_cap_tier?)
venue_breakdown(symbol?, broker?, start_date?, end_date?, benchmark="arrival")
top_outliers(start_date?, end_date?, benchmark="arrival", n=5, mkt_cap_tier?, broker?)
```

---

## Task 0: Scaffold project, Linear issue, branch, config

**Files:**
- Create: whole `projects/execution-quality-copilot/` skeleton
- Modify: `pyproject.toml`, `Makefile`, `.gitignore`
- Create: `config.yaml`, `.env.example`

- [ ] **Step 1: Linear issue + branch (done)**

The Linear issue already exists: **LIN-141 — "Execution Quality Copilot MVP — TCA over MCP"** (project AI PM Portfolio, label Feature, priority Medium). Work happens on the branch `feat/LIN-141-execution-quality-copilot-mvp` (created in the worktree at execution time). The `LIN-141` key in the branch name satisfies the pre-commit Linear-key hook for every per-task commit — no extra action needed in this step.

- [ ] **Step 2: Scaffold the package**

```bash
make new-project name=execution-quality-copilot desc="MCP tool server exposing fund execution data for conversational TCA"
```

- [ ] **Step 3: Replace `pyproject.toml`**

File: `projects/execution-quality-copilot/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "execution-quality-copilot"
version = "0.1.0"
description = "MCP tool server exposing fund execution data for conversational TCA"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.12.0",
    "duckdb>=1.1.0",
    "anthropic>=0.40.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.6.0",
    "mypy>=1.10.0",
    "pre-commit>=3.7.0",
    "detect-secrets>=1.4.0",
]

[tool.setuptools.packages.find]
where = ["src"]
include = ["execution_quality_copilot*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
pythonpath = ["."]
addopts = "-v --cov=execution_quality_copilot --cov-report=term-missing"
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
]

[tool.ruff]
line-length = 100

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

[tool.mypy]
ignore_missing_imports = true
```

- [ ] **Step 4: Replace `Makefile`**

File: `projects/execution-quality-copilot/Makefile`

```make
.PHONY: install dev-install test test-cov lint format typecheck ci clean gen-data serve eval

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=src --cov-report=html

lint:
	ruff check src/ tests/ evals/
	ruff format --check src/ tests/ evals/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/ evals/

typecheck:
	mypy src/ --ignore-missing-imports

ci: lint test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache build/ dist/ *.egg-info htmlcov/ .coverage

gen-data:
	python -m execution_quality_copilot.datagen

serve:
	python -m execution_quality_copilot.server.main

eval:
	python evals/run_eval.py
```

- [ ] **Step 5: Write `config.yaml`**

File: `projects/execution-quality-copilot/config.yaml`

```yaml
models:
  agent: claude-opus-4-8        # eval harness tool-use loop (strongest tool selection)
  max_tokens: 1024

storage:
  db_path: data/seed.duckdb

generator:
  seed: 42
  n_fills: 10000
  n_symbols: 50
  start_date: "2026-05-01"
  end_date: "2026-05-29"        # business days only; inclusive

eval:
  numeric_tolerance: 0.05        # 5% relative tolerance on the headline number
  max_turns: 6                   # max tool-use rounds per question
```

- [ ] **Step 6: Write `.env.example`**

File: `projects/execution-quality-copilot/.env.example`

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

- [ ] **Step 7: Ensure generated data is gitignored**

Append to `projects/execution-quality-copilot/.gitignore` (create the lines if absent):

```
# Generated synthetic dataset (regenerate with `make gen-data`)
data/*.duckdb
data/*.duckdb.wal
# Eval run outputs
evals/results/*.json
```

- [ ] **Step 8: Commit the scaffold**

```bash
cd /home/ubuntu/projects/ai-pm-portfolio/projects/execution-quality-copilot
pip install -e ".[dev]"
git add -A
git commit -m "chore(eqc): scaffold project, deps, config [LIN-141]"
```

---

## Task 1: Config loader

**Files:**
- Create: `src/execution_quality_copilot/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_config.py`

```python
"""Test config loading and the API-key accessor."""

import pytest

from execution_quality_copilot import config


def test_get_config_has_models_and_storage():
    cfg = config.get_config()
    assert cfg["models"]["agent"] == "claude-opus-4-8"
    assert cfg["storage"]["db_path"].endswith("seed.duckdb")
    assert cfg["generator"]["n_fills"] == 10000


def test_get_anthropic_api_key_raises_when_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        config.get_anthropic_api_key()


def test_get_anthropic_api_key_returns_value(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert config.get_anthropic_api_key() == "sk-test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: ... config`.

- [ ] **Step 3: Write minimal implementation**

File: `src/execution_quality_copilot/config.py`

```python
"""Configuration loading: YAML defaults + .env secrets."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Load and cache config.yaml."""
    global _config
    if _config is None:
        with open(CONFIG_PATH) as f:
            _config = yaml.safe_load(f)
    return _config


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key or raise if unset."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return key
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/execution_quality_copilot/config.py tests/unit/test_config.py
git commit -m "feat(eqc): config loader (yaml + dotenv) [LIN-141]"
```

---

## Task 2: Domain enums + Fill record

**Files:**
- Create: `src/execution_quality_copilot/models.py`
- Test: `tests/unit/test_models.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_models.py`

```python
"""Test the domain enums and the Fill dataclass."""

from execution_quality_copilot import models


def test_enum_membership():
    assert models.BROKERS == ["ALPHA", "BRAVO", "COBALT", "DELTA"]
    assert models.ALGOS == ["VWAP", "TWAP", "IS"]
    assert set(models.SIDES) == {"BUY", "SELL"}
    assert models.TIERS == ["large", "mid", "small"]
    assert models.BENCHMARKS == ["arrival", "vwap", "close"]
    assert "DARK" in models.VENUES


def test_fill_to_row_tuple_order_matches_columns():
    from datetime import date, datetime

    fill = models.Fill(
        exec_id="E1",
        order_id="O1",
        symbol="SYM00",
        mkt_cap_tier="large",
        side="BUY",
        broker="ALPHA",
        algo="VWAP",
        venue="XNAS",
        transact_time=datetime(2026, 5, 1, 10, 0, 0),
        trade_date=date(2026, 5, 1),
        last_qty=100,
        last_px=10.5,
        arrival_px=10.4,
        interval_vwap=10.45,
        close_px=10.6,
        currency="USD",
    )
    row = fill.to_row()
    assert row[0] == "E1"
    assert len(row) == len(models.FILL_COLUMNS)
    assert row[models.FILL_COLUMNS.index("last_px")] == 10.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: ... models`.

- [ ] **Step 3: Write minimal implementation**

File: `src/execution_quality_copilot/models.py`

```python
"""Domain enums and the FIX ExecutionReport (35=8) Fill record."""

from dataclasses import dataclass
from datetime import date, datetime

SIDES: list[str] = ["BUY", "SELL"]
BROKERS: list[str] = ["ALPHA", "BRAVO", "COBALT", "DELTA"]
ALGOS: list[str] = ["VWAP", "TWAP", "IS"]
VENUES: list[str] = ["XNAS", "XNYS", "BATS", "EDGX", "DARK"]
TIERS: list[str] = ["large", "mid", "small"]
BENCHMARKS: list[str] = ["arrival", "vwap", "close"]

FILL_COLUMNS: list[str] = [
    "exec_id",
    "order_id",
    "symbol",
    "mkt_cap_tier",
    "side",
    "broker",
    "algo",
    "venue",
    "transact_time",
    "trade_date",
    "last_qty",
    "last_px",
    "arrival_px",
    "interval_vwap",
    "close_px",
    "currency",
]


@dataclass
class Fill:
    """One synthetic FIX ExecutionReport (a single fill)."""

    exec_id: str
    order_id: str
    symbol: str
    mkt_cap_tier: str
    side: str
    broker: str
    algo: str
    venue: str
    transact_time: datetime
    trade_date: date
    last_qty: int
    last_px: float
    arrival_px: float
    interval_vwap: float
    close_px: float
    currency: str

    def to_row(self) -> tuple:
        """Return field values in FILL_COLUMNS order for a DuckDB INSERT."""
        return tuple(getattr(self, c) for c in FILL_COLUMNS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_models.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/execution_quality_copilot/models.py tests/unit/test_models.py
git commit -m "feat(eqc): domain enums + Fill record [LIN-141]"
```

---

## Task 3: Deterministic synthetic data generator

**Files:**
- Create: `src/execution_quality_copilot/datagen/__init__.py` (empty)
- Create: `src/execution_quality_copilot/datagen/generate.py`
- Create: `src/execution_quality_copilot/datagen/__main__.py`
- Test: `tests/unit/test_generate.py`

The generator embeds a **systematic slippage bias** per broker and algo (and an extra penalty for small caps) so the eval questions have stable, sanity-checkable answers: `DELTA` + `IS` + small-cap is reliably the worst.

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_generate.py`

```python
"""Test the deterministic synthetic fill generator."""

from execution_quality_copilot import models
from execution_quality_copilot.datagen import generate


def test_build_fills_is_deterministic():
    a = generate.build_fills(seed=42, n_fills=500, n_symbols=10,
                             start_date="2026-05-01", end_date="2026-05-08")
    b = generate.build_fills(seed=42, n_fills=500, n_symbols=10,
                             start_date="2026-05-01", end_date="2026-05-08")
    assert len(a) == 500
    assert [f.exec_id for f in a] == [f.exec_id for f in b]
    assert a[0].last_px == b[0].last_px


def test_fill_fields_are_valid():
    fills = generate.build_fills(seed=1, n_fills=200, n_symbols=8,
                                 start_date="2026-05-01", end_date="2026-05-08")
    for f in fills:
        assert f.side in models.SIDES
        assert f.broker in models.BROKERS
        assert f.algo in models.ALGOS
        assert f.venue in models.VENUES
        assert f.mkt_cap_tier in models.TIERS
        assert f.last_qty > 0
        assert f.last_px > 0
        assert f.currency == "USD"


def test_write_duckdb_roundtrips_row_count(tmp_path):
    db = tmp_path / "seed.duckdb"
    fills = generate.build_fills(seed=42, n_fills=300, n_symbols=10,
                                 start_date="2026-05-01", end_date="2026-05-08")
    generate.write_duckdb(fills, db)
    import duckdb

    conn = duckdb.connect(str(db), read_only=True)
    count = conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    conn.close()
    assert count == 300


def test_delta_is_smallcap_is_worse_than_alpha_vwap_largecap(tmp_path):
    # The embedded bias must make DELTA+IS+small materially worse than ALPHA+VWAP+large.
    fills = generate.build_fills(seed=42, n_fills=8000, n_symbols=50,
                                 start_date="2026-05-01", end_date="2026-05-29")

    def avg_slip(broker, algo, tier):
        rows = [f for f in fills if f.broker == broker and f.algo == algo and f.mkt_cap_tier == tier]
        vals = []
        for f in rows:
            sign = 1 if f.side == "BUY" else -1
            vals.append(sign * (f.last_px - f.arrival_px) / f.arrival_px * 10000)
        return sum(vals) / len(vals)

    assert avg_slip("DELTA", "IS", "small") > avg_slip("ALPHA", "VWAP", "large")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: ... datagen.generate`.

- [ ] **Step 3: Write minimal implementation**

File: `src/execution_quality_copilot/datagen/__init__.py`

```python
"""Synthetic FIX ExecutionReport generator."""
```

File: `src/execution_quality_copilot/datagen/generate.py`

```python
"""Deterministic synthetic FIX ExecutionReport (35=8) generator → DuckDB.

Embeds a systematic slippage bias per broker/algo plus a small-cap penalty so the
golden eval questions have stable, hand-checkable answers. Run with:

    python -m execution_quality_copilot.datagen
"""

import random
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb

from execution_quality_copilot import models
from execution_quality_copilot.models import Fill

# Systematic slippage (bps) injected by broker and algo; higher = worse execution.
BROKER_BIAS = {"ALPHA": 1.0, "BRAVO": 2.5, "COBALT": 4.0, "DELTA": 6.0}
ALGO_BIAS = {"VWAP": 1.0, "TWAP": 2.0, "IS": 3.5}
TIER_PENALTY = {"large": 0.0, "mid": 1.5, "small": 4.0}


def _tier_for_index(i: int) -> str:
    """Assign a market-cap tier: first 15 large, next 20 mid, rest small."""
    if i < 15:
        return "large"
    if i < 35:
        return "mid"
    return "small"


def _business_days(start: str, end: str) -> list[date]:
    """Inclusive list of weekdays between two ISO dates."""
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    days = []
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def build_fills(
    *,
    seed: int,
    n_fills: int,
    n_symbols: int,
    start_date: str,
    end_date: str,
) -> list[Fill]:
    """Build a deterministic list of synthetic fills for the given parameters."""
    rng = random.Random(seed)
    days = _business_days(start_date, end_date)

    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    tiers = {s: _tier_for_index(i) for i, s in enumerate(symbols)}
    base_px = {s: round(5.0 + (i % 40) * 2.5, 2) for i, s in enumerate(symbols)}
    # Deterministic per (symbol, day) close price, independent of the fill loop.
    close_px = {
        (s, d): round(base_px[s] * (1.0 + ((hash((s, d.isoformat())) % 401) - 200) / 10000.0), 4)
        for s in symbols
        for d in days
    }

    fills: list[Fill] = []
    for n in range(n_fills):
        symbol = rng.choice(symbols)
        tier = tiers[symbol]
        side = rng.choice(models.SIDES)
        broker = rng.choice(models.BROKERS)
        algo = rng.choice(models.ALGOS)
        venue = rng.choice(models.VENUES)
        d = rng.choice(days)

        arrival = round(base_px[symbol] * (1.0 + rng.uniform(-0.004, 0.004)), 4)
        expected_slip = (
            BROKER_BIAS[broker]
            + ALGO_BIAS[algo]
            + TIER_PENALTY[tier]
            + rng.gauss(0.0, 2.0)  # noise; bias still dominates in aggregate
        )
        sign = 1 if side == "BUY" else -1
        last_px = round(arrival * (1.0 + sign * expected_slip / 10000.0), 4)
        interval_vwap = round(arrival * (1.0 + rng.uniform(-0.002, 0.002)), 4)
        qty = rng.choice([100, 200, 500, 1000, 2500, 5000])
        ts = datetime.combine(d, datetime.min.time()) + timedelta(
            hours=9, minutes=rng.randint(30, 390)
        )

        fills.append(
            Fill(
                exec_id=f"E{n:06d}",
                order_id=f"O{n // 3:06d}",
                symbol=symbol,
                mkt_cap_tier=tier,
                side=side,
                broker=broker,
                algo=algo,
                venue=venue,
                transact_time=ts,
                trade_date=d,
                last_qty=qty,
                last_px=last_px,
                arrival_px=arrival,
                interval_vwap=interval_vwap,
                close_px=close_px[(symbol, d)],
                currency="USD",
            )
        )
    return fills


def write_duckdb(fills: list[Fill], db_path: Path) -> None:
    """Create (overwrite) the fills table and bulk-insert the rows."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE fills (
                exec_id VARCHAR PRIMARY KEY,
                order_id VARCHAR,
                symbol VARCHAR,
                mkt_cap_tier VARCHAR,
                side VARCHAR,
                broker VARCHAR,
                algo VARCHAR,
                venue VARCHAR,
                transact_time TIMESTAMP,
                trade_date DATE,
                last_qty INTEGER,
                last_px DOUBLE,
                arrival_px DOUBLE,
                interval_vwap DOUBLE,
                close_px DOUBLE,
                currency VARCHAR
            )
            """
        )
        conn.executemany(
            f"INSERT INTO fills ({', '.join(models.FILL_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in models.FILL_COLUMNS)})",
            [f.to_row() for f in fills],
        )
    finally:
        conn.close()
```

File: `src/execution_quality_copilot/datagen/__main__.py`

```python
"""CLI: regenerate the seed DuckDB from config.yaml (`python -m execution_quality_copilot.datagen`)."""

from pathlib import Path

from execution_quality_copilot.config import get_config
from execution_quality_copilot.datagen.generate import build_fills, write_duckdb

ROOT = Path(__file__).parent.parent.parent.parent


def main() -> None:
    """Build the synthetic dataset and write it to the configured db_path."""
    cfg = get_config()
    g = cfg["generator"]
    fills = build_fills(
        seed=g["seed"],
        n_fills=g["n_fills"],
        n_symbols=g["n_symbols"],
        start_date=g["start_date"],
        end_date=g["end_date"],
    )
    db_path = ROOT / cfg["storage"]["db_path"]
    write_duckdb(fills, db_path)
    print(f"Wrote {len(fills)} fills to {db_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_generate.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Generate the real seed dataset**

Run: `make gen-data`
Expected: `Wrote 10000 fills to .../data/seed.duckdb`.

- [ ] **Step 6: Commit**

```bash
git add src/execution_quality_copilot/datagen/ tests/unit/test_generate.py
git commit -m "feat(eqc): deterministic synthetic FIX ExecutionReport generator [LIN-141]"
```

---

## Task 4: DuckDB connection + bounded query helper

**Files:**
- Create: `src/execution_quality_copilot/db.py`
- Test: `tests/unit/test_db.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_db.py`

```python
"""Test the DuckDB helpers: bounded query, serialization, benchmark whitelist."""

from datetime import date, datetime

import duckdb
import pytest

from execution_quality_copilot import db


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE t (id INTEGER, d DATE, px DOUBLE)")
    c.executemany(
        "INSERT INTO t VALUES (?, ?, ?)",
        [(i, date(2026, 5, 1), 1.23456 + i) for i in range(10)],
    )
    return c


def test_query_returns_dicts(conn):
    rows = db.query(conn, "SELECT id, px FROM t ORDER BY id")
    assert rows[0] == {"id": 0, "px": 1.23456}
    assert len(rows) == 10


def test_query_respects_max_rows(conn):
    rows = db.query(conn, "SELECT id FROM t ORDER BY id", max_rows=3)
    assert len(rows) == 3


def test_query_parameterised(conn):
    rows = db.query(conn, "SELECT id FROM t WHERE id = ?", [4])
    assert rows == [{"id": 4}]


def test_serialize_dates_and_rounds_floats():
    out = db._serialize([{"d": date(2026, 5, 1), "ts": datetime(2026, 5, 1, 9, 30), "px": 1.234567}])
    assert out[0]["d"] == "2026-05-01"
    assert out[0]["ts"] == "2026-05-01T09:30:00"
    assert out[0]["px"] == 1.2346


def test_benchmark_columns_whitelist():
    assert db.BENCHMARK_COLUMNS["arrival"] == "arrival_px"
    assert db.BENCHMARK_COLUMNS["vwap"] == "interval_vwap"
    assert db.BENCHMARK_COLUMNS["close"] == "close_px"
    assert "px; DROP TABLE" not in db.BENCHMARK_COLUMNS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: ... db`.

- [ ] **Step 3: Write minimal implementation**

File: `src/execution_quality_copilot/db.py`

```python
"""DuckDB connection + bounded, parameterised query helpers (the data guardrail layer)."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from execution_quality_copilot.config import get_config

# The model picks a benchmark *enum*; tools map it to a known column via this whitelist.
# Model-supplied strings are NEVER interpolated into SQL except through this dict.
BENCHMARK_COLUMNS: dict[str, str] = {
    "arrival": "arrival_px",
    "vwap": "interval_vwap",
    "close": "close_px",
}

MAX_ROWS = 500  # hard cap on every result set

ROOT = Path(__file__).parent.parent.parent

_conn: duckdb.DuckDBPyConnection | None = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a cached read-only connection to the configured seed DuckDB."""
    global _conn
    if _conn is None:
        db_path = ROOT / get_config()["storage"]["db_path"]
        if not db_path.exists():
            raise FileNotFoundError(
                f"{db_path} not found — run `make gen-data` to build the synthetic dataset"
            )
        _conn = duckdb.connect(str(db_path), read_only=True)
    return _conn


def query(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[Any] | None = None,
    *,
    max_rows: int = MAX_ROWS,
) -> list[dict[str, Any]]:
    """Run a parameterised query and return at most max_rows rows as dicts."""
    cur = conn.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(min(max_rows, MAX_ROWS))
    return [dict(zip(cols, r)) for r in rows]


def _serialize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make rows JSON-safe: dates/datetimes → ISO strings, floats rounded to 4dp."""
    out: list[dict[str, Any]] = []
    for r in rows:
        d: dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, (date, datetime)):
                d[k] = v.isoformat()
            elif isinstance(v, float):
                d[k] = round(v, 4)
            else:
                d[k] = v
        out.append(d)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_db.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/execution_quality_copilot/db.py tests/unit/test_db.py
git commit -m "feat(eqc): duckdb connection + bounded query helper [LIN-141]"
```

---

## Task 5: `fills.py` — get_fills + venue_breakdown

**Files:**
- Create: `src/execution_quality_copilot/server/__init__.py` (empty)
- Create: `src/execution_quality_copilot/server/tools/__init__.py` (empty)
- Create: `src/execution_quality_copilot/server/tools/fills.py`
- Test: `tests/unit/test_tool_fills.py`

A shared in-memory DuckDB fixture is reused by Tasks 5–7. Put it in `tests/conftest.py`.

- [ ] **Step 1: Write the shared fixture + failing test**

File: `tests/conftest.py`

```python
"""Shared pytest fixtures: a small deterministic in-memory fills table."""

import duckdb
import pytest

from execution_quality_copilot.datagen.generate import build_fills
from execution_quality_copilot import models


@pytest.fixture
def fills_conn():
    """In-memory DuckDB seeded with 2000 deterministic fills (no file, no API key)."""
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE fills (
            exec_id VARCHAR, order_id VARCHAR, symbol VARCHAR, mkt_cap_tier VARCHAR,
            side VARCHAR, broker VARCHAR, algo VARCHAR, venue VARCHAR,
            transact_time TIMESTAMP, trade_date DATE, last_qty INTEGER, last_px DOUBLE,
            arrival_px DOUBLE, interval_vwap DOUBLE, close_px DOUBLE, currency VARCHAR
        )
        """
    )
    fills = build_fills(seed=42, n_fills=2000, n_symbols=50,
                        start_date="2026-05-01", end_date="2026-05-29")
    conn.executemany(
        f"INSERT INTO fills ({', '.join(models.FILL_COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in models.FILL_COLUMNS)})",
        [f.to_row() for f in fills],
    )
    return conn
```

File: `tests/unit/test_tool_fills.py`

```python
"""Test get_fills and venue_breakdown pure functions."""

from execution_quality_copilot.server.tools import fills


def test_get_fills_filters_and_caps(fills_conn):
    out = fills.get_fills(fills_conn, broker="DELTA", side="BUY", limit=5)
    assert out["n_returned"] <= 5
    assert out["row_cap"] == 5
    for row in out["fills"]:
        assert row["broker"] == "DELTA"
        assert row["side"] == "BUY"


def test_get_fills_limit_is_clamped(fills_conn):
    out = fills.get_fills(fills_conn, limit=99999)
    assert out["row_cap"] == 500  # MAX_ROWS


def test_get_fills_date_range(fills_conn):
    out = fills.get_fills(fills_conn, start_date="2026-05-01", end_date="2026-05-01", limit=500)
    for row in out["fills"]:
        assert row["trade_date"] == "2026-05-01"


def test_venue_breakdown_groups_by_venue_and_algo(fills_conn):
    out = fills.venue_breakdown(fills_conn, broker="DELTA")
    keys = {(r["venue"], r["algo"]) for r in out["breakdown"]}
    assert ("DARK", "IS") in keys or len(keys) > 0
    for r in out["breakdown"]:
        assert "slippage_bps" in r and "n_fills" in r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tool_fills.py -v`
Expected: FAIL with `ModuleNotFoundError: ... server.tools.fills`.

- [ ] **Step 3: Write minimal implementation**

File: `src/execution_quality_copilot/server/__init__.py`

```python
"""MCP server package."""
```

File: `src/execution_quality_copilot/server/tools/__init__.py`

```python
"""Pure tool functions shared by the MCP server and the eval harness."""
```

File: `src/execution_quality_copilot/server/tools/fills.py`

```python
"""Blotter tools: get_fills (filtered fill query) and venue_breakdown."""

from typing import Any

import duckdb

from execution_quality_copilot.db import BENCHMARK_COLUMNS, MAX_ROWS, _serialize, query


def _eq_clauses(pairs: list[tuple[str, Any]]) -> tuple[list[str], list[Any]]:
    """Build equality WHERE fragments for non-None (column, value) pairs."""
    clauses, params = [], []
    for col, val in pairs:
        if val is not None:
            clauses.append(f"{col} = ?")
            params.append(val)
    return clauses, params


def get_fills(
    conn: duckdb.DuckDBPyConnection,
    *,
    symbol: str | None = None,
    broker: str | None = None,
    algo: str | None = None,
    venue: str | None = None,
    side: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    mkt_cap_tier: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Return a bounded, filtered slice of the fill blotter."""
    limit = max(1, min(int(limit), MAX_ROWS))
    clauses, params = _eq_clauses(
        [
            ("symbol", symbol),
            ("broker", broker),
            ("algo", algo),
            ("venue", venue),
            ("side", side),
            ("mkt_cap_tier", mkt_cap_tier),
        ]
    )
    if start_date:
        clauses.append("trade_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("trade_date <= ?")
        params.append(end_date)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        "SELECT exec_id, trade_date, symbol, mkt_cap_tier, side, broker, algo, venue, "
        f"last_qty, last_px, arrival_px FROM fills{where} ORDER BY trade_date, exec_id LIMIT {limit}"
    )
    rows = query(conn, sql, params, max_rows=limit)
    return {"n_returned": len(rows), "row_cap": limit, "fills": _serialize(rows)}


def venue_breakdown(
    conn: duckdb.DuckDBPyConnection,
    *,
    symbol: str | None = None,
    broker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    benchmark: str = "arrival",
) -> dict[str, Any]:
    """Aggregate fills by venue and algo: count, notional, and avg slippage (bps)."""
    bench = BENCHMARK_COLUMNS.get(benchmark)
    if bench is None:
        raise ValueError(f"unknown benchmark '{benchmark}'; choose {sorted(BENCHMARK_COLUMNS)}")
    clauses, params = _eq_clauses([("symbol", symbol), ("broker", broker)])
    if start_date:
        clauses.append("trade_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("trade_date <= ?")
        params.append(end_date)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sign = "(CASE WHEN side = 'BUY' THEN 1 ELSE -1 END)"
    notional = "(last_qty * last_px)"
    slip = f"{sign} * (last_px - {bench}) / {bench} * 10000"
    sql = (
        f"SELECT venue, algo, COUNT(*) AS n_fills, SUM({notional}) AS notional, "
        f"SUM(({slip}) * {notional}) / NULLIF(SUM({notional}), 0) AS slippage_bps "
        f"FROM fills{where} GROUP BY venue, algo ORDER BY venue, algo"
    )
    rows = query(conn, sql, params, max_rows=MAX_ROWS)
    return {
        "benchmark": benchmark,
        "breakdown": _serialize(rows),
        "caveat": "slippage_bps is notional-weighted vs the chosen benchmark; positive = cost.",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_tool_fills.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/execution_quality_copilot/server/ tests/conftest.py tests/unit/test_tool_fills.py
git commit -m "feat(eqc): get_fills + venue_breakdown tools [LIN-141]"
```

---

## Task 6: `benchmarks.py` — get_benchmarks

**Files:**
- Create: `src/execution_quality_copilot/server/tools/benchmarks.py`
- Test: `tests/unit/test_tool_benchmarks.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_tool_benchmarks.py`

```python
"""Test the get_benchmarks pure function."""

from execution_quality_copilot.server.tools import benchmarks


def test_get_benchmarks_returns_three_reference_prices(fills_conn):
    # Pick a (symbol, date) that exists in the fixture.
    row = fills_conn.execute(
        "SELECT symbol, trade_date FROM fills LIMIT 1"
    ).fetchone()
    symbol, trade_date = row[0], row[1].isoformat()
    out = benchmarks.get_benchmarks(fills_conn, symbol=symbol, date=trade_date)
    assert out["symbol"] == symbol
    assert out["date"] == trade_date
    assert out["n_fills"] >= 1
    assert out["arrival_px"] > 0
    assert out["close_px"] > 0
    assert out["vwap"] > 0


def test_get_benchmarks_empty_when_no_match(fills_conn):
    out = benchmarks.get_benchmarks(fills_conn, symbol="NOPE", date="2026-05-01")
    assert out["n_fills"] == 0
    assert out["close_px"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tool_benchmarks.py -v`
Expected: FAIL with `ModuleNotFoundError: ... server.tools.benchmarks`.

- [ ] **Step 3: Write minimal implementation**

File: `src/execution_quality_copilot/server/tools/benchmarks.py`

```python
"""Benchmark tool: get_benchmarks (arrival / VWAP / close reference prices for a symbol-day)."""

from typing import Any

import duckdb

from execution_quality_copilot.db import query


def get_benchmarks(
    conn: duckdb.DuckDBPyConnection,
    *,
    symbol: str,
    date: str,
) -> dict[str, Any]:
    """Return arrival (avg), VWAP (notional-weighted), and close reference prices for a symbol-day."""
    sql = (
        "SELECT COUNT(*) AS n_fills, "
        "AVG(arrival_px) AS arrival_px, "
        "SUM(interval_vwap * last_qty) / NULLIF(SUM(last_qty), 0) AS vwap, "
        "MAX(close_px) AS close_px "
        "FROM fills WHERE symbol = ? AND trade_date = ?"
    )
    rows = query(conn, sql, [symbol, date], max_rows=1)
    row = rows[0] if rows else {}
    n = int(row.get("n_fills") or 0)
    return {
        "symbol": symbol,
        "date": date,
        "n_fills": n,
        "arrival_px": round(row["arrival_px"], 4) if n and row.get("arrival_px") is not None else None,
        "vwap": round(row["vwap"], 4) if n and row.get("vwap") is not None else None,
        "close_px": round(row["close_px"], 4) if n and row.get("close_px") is not None else None,
        "caveat": "arrival is the average order-arrival price; vwap is notional-weighted; close is the day's close.",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_tool_benchmarks.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/execution_quality_copilot/server/tools/benchmarks.py tests/unit/test_tool_benchmarks.py
git commit -m "feat(eqc): get_benchmarks tool [LIN-141]"
```

---

## Task 7: `tca.py` — compute_slippage + top_outliers

**Files:**
- Create: `src/execution_quality_copilot/server/tools/tca.py`
- Test: `tests/unit/test_tool_tca.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_tool_tca.py`

```python
"""Test compute_slippage and top_outliers pure functions, including the sign convention."""

import pytest

from execution_quality_copilot.server.tools import tca


def test_compute_slippage_shape_and_sign(fills_conn):
    out = tca.compute_slippage(fills_conn, broker="DELTA", benchmark="arrival")
    assert out["benchmark"] == "arrival"
    assert out["n_fills"] > 0
    assert out["total_notional"] > 0
    # DELTA carries the worst injected bias → positive cost vs arrival.
    assert out["slippage_bps"] > 0


def test_compute_slippage_delta_worse_than_alpha(fills_conn):
    delta = tca.compute_slippage(fills_conn, broker="DELTA")["slippage_bps"]
    alpha = tca.compute_slippage(fills_conn, broker="ALPHA")["slippage_bps"]
    assert delta > alpha


def test_compute_slippage_rejects_unknown_benchmark(fills_conn):
    with pytest.raises(ValueError, match="unknown benchmark"):
        tca.compute_slippage(fills_conn, benchmark="midpoint")


def test_top_outliers_sorted_desc_and_capped(fills_conn):
    out = tca.top_outliers(fills_conn, n=5)
    slips = [r["slippage_bps"] for r in out["outliers"]]
    assert slips == sorted(slips, reverse=True)
    assert len(out["outliers"]) == 5
    assert out["worst_slippage_bps"] == slips[0]


def test_top_outliers_n_clamped(fills_conn):
    out = tca.top_outliers(fills_conn, n=9999)
    assert len(out["outliers"]) <= 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tool_tca.py -v`
Expected: FAIL with `ModuleNotFoundError: ... server.tools.tca`.

- [ ] **Step 3: Write minimal implementation**

File: `src/execution_quality_copilot/server/tools/tca.py`

```python
"""TCA tools: compute_slippage (bps vs benchmark) and top_outliers (worst executions)."""

from typing import Any

import duckdb

from execution_quality_copilot.db import BENCHMARK_COLUMNS, MAX_ROWS, _serialize, query
from execution_quality_copilot.server.tools.fills import _eq_clauses

# Per-fill signed slippage in bps; positive = cost (worse than benchmark).
_SIGN = "(CASE WHEN side = 'BUY' THEN 1 ELSE -1 END)"
_NOTIONAL = "(last_qty * last_px)"


def _benchmark_column(benchmark: str) -> str:
    """Map a benchmark enum to its column, or raise for an unknown value."""
    bench = BENCHMARK_COLUMNS.get(benchmark)
    if bench is None:
        raise ValueError(f"unknown benchmark '{benchmark}'; choose {sorted(BENCHMARK_COLUMNS)}")
    return bench


def compute_slippage(
    conn: duckdb.DuckDBPyConnection,
    *,
    symbol: str | None = None,
    broker: str | None = None,
    algo: str | None = None,
    side: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    benchmark: str = "arrival",
    mkt_cap_tier: str | None = None,
) -> dict[str, Any]:
    """Compute notional-weighted slippage (bps) vs the chosen benchmark over a filtered set."""
    bench = _benchmark_column(benchmark)
    clauses, params = _eq_clauses(
        [
            ("symbol", symbol),
            ("broker", broker),
            ("algo", algo),
            ("side", side),
            ("mkt_cap_tier", mkt_cap_tier),
        ]
    )
    if start_date:
        clauses.append("trade_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("trade_date <= ?")
        params.append(end_date)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    slip = f"{_SIGN} * (last_px - {bench}) / {bench} * 10000"
    sql = (
        f"SELECT SUM(({slip}) * {_NOTIONAL}) / NULLIF(SUM({_NOTIONAL}), 0) AS slippage_bps, "
        f"COUNT(*) AS n_fills, SUM({_NOTIONAL}) AS total_notional FROM fills{where}"
    )
    rows = query(conn, sql, params, max_rows=1)
    row = rows[0] if rows else {}
    bps = row.get("slippage_bps")
    used = {
        k: v
        for k, v in {
            "symbol": symbol,
            "broker": broker,
            "algo": algo,
            "side": side,
            "start_date": start_date,
            "end_date": end_date,
            "mkt_cap_tier": mkt_cap_tier,
        }.items()
        if v is not None
    }
    return {
        "benchmark": benchmark,
        "slippage_bps": round(bps, 2) if bps is not None else None,
        "n_fills": int(row.get("n_fills") or 0),
        "total_notional": round(row.get("total_notional") or 0, 2),
        "filters": used,
        "caveat": "bps vs benchmark, notional-weighted; positive = cost (worse than benchmark).",
    }


def top_outliers(
    conn: duckdb.DuckDBPyConnection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    benchmark: str = "arrival",
    n: int = 5,
    mkt_cap_tier: str | None = None,
    broker: str | None = None,
) -> dict[str, Any]:
    """Return the n single fills with the worst slippage (bps) vs the chosen benchmark."""
    bench = _benchmark_column(benchmark)
    n = max(1, min(int(n), 50))
    clauses, params = _eq_clauses([("mkt_cap_tier", mkt_cap_tier), ("broker", broker)])
    if start_date:
        clauses.append("trade_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("trade_date <= ?")
        params.append(end_date)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    slip = f"{_SIGN} * (last_px - {bench}) / {bench} * 10000"
    sql = (
        "SELECT exec_id, trade_date, symbol, mkt_cap_tier, side, broker, algo, venue, "
        f"last_qty, last_px, {bench} AS benchmark_px, {slip} AS slippage_bps "
        f"FROM fills{where} ORDER BY slippage_bps DESC LIMIT {n}"
    )
    rows = _serialize(query(conn, sql, params, max_rows=n))
    return {
        "benchmark": benchmark,
        "n": n,
        "worst_slippage_bps": rows[0]["slippage_bps"] if rows else None,
        "outliers": rows,
        "caveat": "each row is one fill; slippage_bps positive = cost vs benchmark.",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_tool_tca.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/execution_quality_copilot/server/tools/tca.py tests/unit/test_tool_tca.py
git commit -m "feat(eqc): compute_slippage + top_outliers tools [LIN-141]"
```

---

## Task 8: FastMCP server (descriptions-as-prompts)

**Files:**
- Create: `src/execution_quality_copilot/server/main.py`
- Test: `tests/unit/test_server.py`

The tool **descriptions** are the product surface — write them as prompts a PM-facing analyst would need: what the tool does, the units, when to pick which benchmark, and the guardrails. The wrappers are thin: obtain a connection, call the pure function.

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_server.py`

```python
"""Smoke-test that the FastMCP server registers all five tools with descriptions."""

import asyncio

from execution_quality_copilot.server import main


def test_five_tools_registered():
    tools = asyncio.run(main.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "get_fills",
        "get_benchmarks",
        "compute_slippage",
        "venue_breakdown",
        "top_outliers",
    }


def test_tool_descriptions_are_substantial():
    tools = asyncio.run(main.mcp.list_tools())
    for t in tools:
        assert t.description and len(t.description) > 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: ... server.main`.

- [ ] **Step 3: Write minimal implementation**

File: `src/execution_quality_copilot/server/main.py`

```python
"""FastMCP server exposing read-only, bounded TCA tools over stdio for Claude Desktop.

Run locally:   python -m execution_quality_copilot.server.main
The server never calls an LLM — it only answers tool calls from the MCP client.
"""

from typing import Any

from mcp.server.fastmcp import FastMCP

from execution_quality_copilot.db import get_connection
from execution_quality_copilot.server.tools import benchmarks, fills, tca

mcp = FastMCP("execution-quality-copilot")


@mcp.tool()
def get_fills(
    symbol: str | None = None,
    broker: str | None = None,
    algo: str | None = None,
    venue: str | None = None,
    side: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    mkt_cap_tier: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Return raw fills (FIX ExecutionReports) from the blotter, filtered and row-capped.

    Use this to inspect individual executions. Filter by symbol, broker (ALPHA/BRAVO/COBALT/DELTA),
    algo (VWAP/TWAP/IS), venue (XNAS/XNYS/BATS/EDGX/DARK), side (BUY/SELL), market-cap tier
    (large/mid/small), and/or an inclusive ISO trade-date range (YYYY-MM-DD). Returns at most 500
    rows (default 100). For aggregate slippage use compute_slippage, not this tool.
    """
    return fills.get_fills(
        get_connection(),
        symbol=symbol,
        broker=broker,
        algo=algo,
        venue=venue,
        side=side,
        start_date=start_date,
        end_date=end_date,
        mkt_cap_tier=mkt_cap_tier,
        limit=limit,
    )


@mcp.tool()
def get_benchmarks(symbol: str, date: str) -> dict[str, Any]:
    """Return arrival, VWAP, and close reference prices for one symbol on one ISO date (YYYY-MM-DD).

    Use this to look up the benchmark levels a PM would compare fills against. Arrival is the average
    order-arrival price that day; VWAP is notional-weighted; close is the day's closing price.
    """
    return benchmarks.get_benchmarks(get_connection(), symbol=symbol, date=date)


@mcp.tool()
def compute_slippage(
    symbol: str | None = None,
    broker: str | None = None,
    algo: str | None = None,
    side: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    benchmark: str = "arrival",
    mkt_cap_tier: str | None = None,
) -> dict[str, Any]:
    """Compute notional-weighted slippage in basis points vs a benchmark over a filtered fill set.

    `benchmark` is one of: "arrival" (implementation shortfall — the PM's default), "vwap", or "close".
    Positive bps = cost (worse than benchmark). Filter by symbol, broker, algo, side, market-cap tier,
    and/or an inclusive ISO date range. This is the right tool for "how much did X cost me in bps?"
    and broker/algo ranking questions (call once per broker/algo and compare).
    """
    return tca.compute_slippage(
        get_connection(),
        symbol=symbol,
        broker=broker,
        algo=algo,
        side=side,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
        mkt_cap_tier=mkt_cap_tier,
    )


@mcp.tool()
def venue_breakdown(
    symbol: str | None = None,
    broker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    benchmark: str = "arrival",
) -> dict[str, Any]:
    """Break fills down by venue and algo: fill count, notional, and avg slippage (bps) per group.

    Use this to see where (which venue) and how (which algo) execution quality differs. Filter by
    symbol, broker, and/or an inclusive ISO date range. `benchmark` is "arrival" (default), "vwap", or "close".
    """
    return fills.venue_breakdown(
        get_connection(),
        symbol=symbol,
        broker=broker,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
    )


@mcp.tool()
def top_outliers(
    start_date: str | None = None,
    end_date: str | None = None,
    benchmark: str = "arrival",
    n: int = 5,
    mkt_cap_tier: str | None = None,
    broker: str | None = None,
) -> dict[str, Any]:
    """Return the n single worst fills by slippage (bps) vs a benchmark — the worst executions.

    Use this for "show me my worst executions". `n` is capped at 50 (default 5). Optionally scope by
    market-cap tier, broker, and/or an inclusive ISO date range. `benchmark` is "arrival"/"vwap"/"close".
    """
    return tca.top_outliers(
        get_connection(),
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
        n=n,
        mkt_cap_tier=mkt_cap_tier,
        broker=broker,
    )


def main() -> None:
    """Run the MCP server over stdio (the transport Claude Desktop uses)."""
    mcp.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_server.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Manually verify the server boots over stdio**

Run (Ctrl-C to exit after it starts cleanly with no traceback; requires `make gen-data` to have run):

```bash
make serve
```

Expected: process starts and waits on stdio (no error). It will not print much — MCP servers speak JSON-RPC on stdin/stdout.

- [ ] **Step 6: Commit**

```bash
git add src/execution_quality_copilot/server/main.py tests/unit/test_server.py
git commit -m "feat(eqc): fastmcp server with 5 tool wrappers + descriptions [LIN-141]"
```

---

## Task 9: Anthropic tool-use loop (eval client)

**Files:**
- Create: `src/execution_quality_copilot/llm.py`
- Test: `tests/unit/test_llm.py`

This is the only LLM-calling code in the project. It is a **manual agentic loop**: send the question + tool definitions, execute any `tool_use` blocks via a `dispatch` callback, feed `tool_result` back, and stop when the model stops calling tools. It records which tools were called with which inputs (the data the eval scores).

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_llm.py`

```python
"""Test the manual tool-use loop with a fake Anthropic client (no network)."""

from execution_quality_copilot import llm


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_loop_executes_tool_then_returns_answer():
    # Turn 1: model calls compute_slippage. Turn 2: model gives the final answer.
    responses = [
        _Resp("tool_use", [_Block(type="tool_use", id="t1", name="compute_slippage",
                                  input={"broker": "DELTA"})]),
        _Resp("end_turn", [_Block(type="text", text="DELTA cost ~11 bps.\nANSWER: 11.0")]),
    ]
    client = _FakeClient(responses)

    calls = []

    def dispatch(name, args):
        calls.append((name, args))
        return {"slippage_bps": 11.0}

    tools = [{"name": "compute_slippage", "description": "x", "input_schema": {"type": "object"}}]
    text, called = llm.run_tool_loop(client, "claude-opus-4-8", tools, dispatch, "rank brokers")

    assert calls == [("compute_slippage", {"broker": "DELTA"})]
    assert called == [("compute_slippage", {"broker": "DELTA"})]
    assert "ANSWER: 11.0" in text


def test_loop_stops_at_max_turns():
    # Model always calls a tool → loop must bail at max_turns without infinite looping.
    looping = _Resp("tool_use", [_Block(type="tool_use", id="t", name="get_fills", input={})])
    client = _FakeClient([looping] * 10)
    text, called = llm.run_tool_loop(
        client, "claude-opus-4-8", [], lambda n, a: {"ok": True}, "q", max_turns=3
    )
    assert len(called) == 3
    assert text == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: ... llm`.

- [ ] **Step 3: Write minimal implementation**

File: `src/execution_quality_copilot/llm.py`

```python
"""Manual Anthropic tool-use loop used by the eval harness to simulate the MCP client."""

import json
from typing import Any, Callable

import anthropic

from execution_quality_copilot.config import get_anthropic_api_key

SYSTEM = (
    "You are a TCA (Transaction Cost Analysis) analyst for a portfolio manager. "
    "You have read-only tools over the fund's execution data. Use them to answer the question "
    "with specific numbers; do not guess. Brokers: ALPHA/BRAVO/COBALT/DELTA. Algos: VWAP/TWAP/IS. "
    "Benchmarks: arrival/vwap/close. Always finish your final message with a line formatted EXACTLY "
    "as 'ANSWER: <number>' where <number> is the single headline figure (bps, count, or price) with "
    "no commas, units, or extra text."
)

Dispatch = Callable[[str, dict[str, Any]], Any]


def get_client() -> anthropic.Anthropic:
    """Return an Anthropic client authenticated from the environment."""
    return anthropic.Anthropic(api_key=get_anthropic_api_key())


def run_tool_loop(
    client: anthropic.Anthropic,
    model: str,
    tools: list[dict[str, Any]],
    dispatch: Dispatch,
    question: str,
    max_turns: int = 6,
) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """Drive a tool-use loop; return (final_text, [(tool_name, tool_input), ...] in call order)."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    called: list[tuple[str, dict[str, Any]]] = []
    final_text = ""

    for _ in range(max_turns):
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM,
            tools=tools,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            final_text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            break

        messages.append({"role": "assistant", "content": resp.content})
        results: list[dict[str, Any]] = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            args = dict(block.input)
            called.append((block.name, args))
            try:
                out = dispatch(block.name, args)
                content, is_error = json.dumps(out, default=str), False
            except Exception as exc:  # surface tool errors back to the model
                content, is_error = f"error: {exc}", True
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})

    return final_text, called
```

> **Note:** Keep the `Dispatch = Callable[...]` alias defined above `run_tool_loop` (as shown) so it is in scope where the signature uses it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_llm.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/execution_quality_copilot/llm.py tests/unit/test_llm.py
git commit -m "feat(eqc): anthropic manual tool-use loop for evals [LIN-141]"
```

---

## Task 10: Tool-use eval harness

**Files:**
- Create: `evals/__init__.py` (empty)
- Create: `evals/golden_questions.json`
- Create: `evals/run_eval.py`
- Create: `evals/results/.gitkeep`
- Test: `tests/unit/test_eval.py`

The harness scores three axes per question: **tool selection** (was the expected tool called?), **parameter match** (do the expected params appear in some call to that tool?), and **numeric accuracy** (is the model's `ANSWER:` within tolerance of the ground truth — computed by calling the expected tool directly). Questions whose answer is a name/ranking set `answer_key: null` and skip the numeric axis.

- [ ] **Step 1: Write the golden questions**

File: `evals/golden_questions.json`

```json
[
  {"question": "Using arrival price, what was DELTA's overall slippage in basis points across all fills?",
   "expected_tool": "compute_slippage", "expected_params": {"broker": "DELTA", "benchmark": "arrival"}, "answer_key": "slippage_bps"},
  {"question": "What was ALPHA's slippage in bps versus arrival price?",
   "expected_tool": "compute_slippage", "expected_params": {"broker": "ALPHA", "benchmark": "arrival"}, "answer_key": "slippage_bps"},
  {"question": "How costly was the IS algo overall, in bps versus arrival?",
   "expected_tool": "compute_slippage", "expected_params": {"algo": "IS", "benchmark": "arrival"}, "answer_key": "slippage_bps"},
  {"question": "What was VWAP-algo slippage in bps against arrival price?",
   "expected_tool": "compute_slippage", "expected_params": {"algo": "VWAP", "benchmark": "arrival"}, "answer_key": "slippage_bps"},
  {"question": "On small-cap names, what was the overall slippage in bps versus arrival?",
   "expected_tool": "compute_slippage", "expected_params": {"mkt_cap_tier": "small", "benchmark": "arrival"}, "answer_key": "slippage_bps"},
  {"question": "What was large-cap slippage in bps versus arrival price?",
   "expected_tool": "compute_slippage", "expected_params": {"mkt_cap_tier": "large", "benchmark": "arrival"}, "answer_key": "slippage_bps"},
  {"question": "What was COBALT's implementation shortfall in bps (arrival benchmark) for buy orders only?",
   "expected_tool": "compute_slippage", "expected_params": {"broker": "COBALT", "side": "BUY", "benchmark": "arrival"}, "answer_key": "slippage_bps"},
  {"question": "Measured against the closing price, what was BRAVO's slippage in bps?",
   "expected_tool": "compute_slippage", "expected_params": {"broker": "BRAVO", "benchmark": "close"}, "answer_key": "slippage_bps"},
  {"question": "What was the worst single execution's slippage in bps versus arrival price?",
   "expected_tool": "top_outliers", "expected_params": {"benchmark": "arrival"}, "answer_key": "worst_slippage_bps"},
  {"question": "Among small-cap fills only, what was the worst single execution's slippage in bps?",
   "expected_tool": "top_outliers", "expected_params": {"mkt_cap_tier": "small", "benchmark": "arrival"}, "answer_key": "worst_slippage_bps"},
  {"question": "What was the closing price for SYM05 on 2026-05-15?",
   "expected_tool": "get_benchmarks", "expected_params": {"symbol": "SYM05", "date": "2026-05-15"}, "answer_key": "close_px"},
  {"question": "What was the VWAP benchmark for SYM10 on 2026-05-08?",
   "expected_tool": "get_benchmarks", "expected_params": {"symbol": "SYM10", "date": "2026-05-08"}, "answer_key": "vwap"},
  {"question": "How many DELTA buy fills are in the blotter? Report the count.",
   "expected_tool": "get_fills", "expected_params": {"broker": "DELTA", "side": "BUY"}, "answer_key": null},
  {"question": "Show me the venue-and-algo breakdown of execution quality for broker DELTA.",
   "expected_tool": "venue_breakdown", "expected_params": {"broker": "DELTA"}, "answer_key": null},
  {"question": "List my five worst executions for the whole month against arrival price.",
   "expected_tool": "top_outliers", "expected_params": {"benchmark": "arrival", "n": 5}, "answer_key": "worst_slippage_bps"}
]
```

- [ ] **Step 2: Write the failing test for the scoring functions**

File: `tests/unit/test_eval.py`

```python
"""Test the eval scoring functions (no API calls)."""

from evals import run_eval


def test_tool_selection_hit_and_miss():
    called = [("get_fills", {}), ("compute_slippage", {"broker": "DELTA"})]
    assert run_eval.tool_selected("compute_slippage", called) is True
    assert run_eval.tool_selected("top_outliers", called) is False


def test_param_match_requires_subset_on_the_right_tool():
    called = [("compute_slippage", {"broker": "DELTA", "benchmark": "arrival"})]
    assert run_eval.params_matched("compute_slippage", {"broker": "DELTA"}, called) is True
    # Right params but wrong tool → no match.
    assert run_eval.params_matched("top_outliers", {"broker": "DELTA"}, called) is False
    # Missing/incorrect value → no match.
    assert run_eval.params_matched("compute_slippage", {"broker": "ALPHA"}, called) is False


def test_numeric_accuracy_within_tolerance():
    assert run_eval.numeric_ok(10.0, 10.3, tol=0.05) is True
    assert run_eval.numeric_ok(10.0, 12.0, tol=0.05) is False
    assert run_eval.numeric_ok(None, 5.0, tol=0.05) is False


def test_parse_answer_extracts_the_answer_line():
    assert run_eval.parse_answer("Some text.\nANSWER: -3.5") == -3.5
    assert run_eval.parse_answer("ANSWER: 1234.56 bps") == 1234.56
    assert run_eval.parse_answer("no answer line here") is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: ... run_eval`.

- [ ] **Step 4: Write the eval harness**

File: `evals/__init__.py`

```python
"""Tool-use eval harness package."""
```

File: `evals/run_eval.py`

```python
"""Tool-use eval: scores tool selection, parameter accuracy, and numeric-answer accuracy.

Run: python evals/run_eval.py   (needs ANTHROPIC_API_KEY and a built seed DB — `make gen-data`)
"""

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import duckdb  # noqa: E402

from execution_quality_copilot.config import get_config  # noqa: E402
from execution_quality_copilot.llm import get_client, run_tool_loop  # noqa: E402
from execution_quality_copilot.server.tools import benchmarks, fills, tca  # noqa: E402

EVALS_DIR = Path(__file__).parent
THRESHOLDS = {"tool_selection_rate": 0.90, "numeric_accuracy_rate": 0.80}

# Anthropic-format tool definitions (mirror the FastMCP tool signatures in server/main.py).
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "get_fills",
        "description": "Return raw fills filtered by symbol/broker/algo/venue/side/tier/date range; row-capped.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "broker": {"type": "string", "enum": ["ALPHA", "BRAVO", "COBALT", "DELTA"]},
                "algo": {"type": "string", "enum": ["VWAP", "TWAP", "IS"]},
                "venue": {"type": "string", "enum": ["XNAS", "XNYS", "BATS", "EDGX", "DARK"]},
                "side": {"type": "string", "enum": ["BUY", "SELL"]},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "mkt_cap_tier": {"type": "string", "enum": ["large", "mid", "small"]},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "get_benchmarks",
        "description": "Arrival/VWAP/close reference prices for one symbol on one ISO date.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "date": {"type": "string"}},
            "required": ["symbol", "date"],
        },
    },
    {
        "name": "compute_slippage",
        "description": "Notional-weighted slippage (bps) vs arrival/vwap/close over a filtered fill set.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "broker": {"type": "string", "enum": ["ALPHA", "BRAVO", "COBALT", "DELTA"]},
                "algo": {"type": "string", "enum": ["VWAP", "TWAP", "IS"]},
                "side": {"type": "string", "enum": ["BUY", "SELL"]},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "benchmark": {"type": "string", "enum": ["arrival", "vwap", "close"]},
                "mkt_cap_tier": {"type": "string", "enum": ["large", "mid", "small"]},
            },
        },
    },
    {
        "name": "venue_breakdown",
        "description": "Fills grouped by venue and algo: count, notional, avg slippage (bps).",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "broker": {"type": "string", "enum": ["ALPHA", "BRAVO", "COBALT", "DELTA"]},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "benchmark": {"type": "string", "enum": ["arrival", "vwap", "close"]},
            },
        },
    },
    {
        "name": "top_outliers",
        "description": "The n single worst fills by slippage (bps) vs a benchmark.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "benchmark": {"type": "string", "enum": ["arrival", "vwap", "close"]},
                "n": {"type": "integer"},
                "mkt_cap_tier": {"type": "string", "enum": ["large", "mid", "small"]},
                "broker": {"type": "string", "enum": ["ALPHA", "BRAVO", "COBALT", "DELTA"]},
            },
        },
    },
]

_TABLE = {
    "get_fills": fills.get_fills,
    "venue_breakdown": fills.venue_breakdown,
    "get_benchmarks": benchmarks.get_benchmarks,
    "compute_slippage": tca.compute_slippage,
    "top_outliers": tca.top_outliers,
}


def make_dispatch(conn: duckdb.DuckDBPyConnection):
    """Return a dispatch(name, args) that runs a pure tool function against the connection."""

    def dispatch(name: str, args: dict[str, Any]) -> Any:
        return _TABLE[name](conn, **args)

    return dispatch


# ---- scoring ----------------------------------------------------------------

def tool_selected(expected_tool: str, called: list[tuple[str, dict[str, Any]]]) -> bool:
    """True if the expected tool was called at least once."""
    return any(name == expected_tool for name, _ in called)


def params_matched(
    expected_tool: str, expected_params: dict[str, Any], called: list[tuple[str, dict[str, Any]]]
) -> bool:
    """True if some call to the expected tool included all expected params (subset match)."""
    for name, args in called:
        if name == expected_tool and all(args.get(k) == v for k, v in expected_params.items()):
            return True
    return False


def parse_answer(text: str) -> float | None:
    """Extract the number from the 'ANSWER: <number>' line, or None."""
    m = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def numeric_ok(parsed: float | None, ground_truth: float, *, tol: float) -> bool:
    """True if parsed is within relative tolerance of ground_truth."""
    if parsed is None or ground_truth is None:
        return False
    denom = abs(ground_truth) if ground_truth else 1.0
    return abs(parsed - ground_truth) / denom <= tol


# ---- runner -----------------------------------------------------------------

def run_evaluation() -> dict[str, Any]:
    """Run the agent over the golden questions and compute aggregate metrics."""
    cfg = get_config()
    tol = cfg["eval"]["numeric_tolerance"]
    max_turns = cfg["eval"]["max_turns"]
    model = cfg["models"]["agent"]

    db_path = ROOT / cfg["storage"]["db_path"]
    conn = duckdb.connect(str(db_path), read_only=True)
    dispatch = make_dispatch(conn)
    client = get_client()

    cases = json.loads((EVALS_DIR / "golden_questions.json").read_text())
    rows = []
    for case in cases:
        text, called = run_tool_loop(
            client, model, TOOL_DEFS, dispatch, case["question"], max_turns=max_turns
        )
        sel = tool_selected(case["expected_tool"], called)
        par = params_matched(case["expected_tool"], case["expected_params"], called)

        num_ok: bool | None = None
        if case["answer_key"] is not None:
            ground = _TABLE[case["expected_tool"]](conn, **case["expected_params"])[case["answer_key"]]
            num_ok = numeric_ok(parse_answer(text), ground, tol=tol)

        rows.append(
            {
                "question": case["question"],
                "expected_tool": case["expected_tool"],
                "tool_selected": sel,
                "params_matched": par,
                "numeric_ok": num_ok,
                "tools_called": [n for n, _ in called],
            }
        )

    n = len(rows)
    numeric_rows = [r for r in rows if r["numeric_ok"] is not None]
    metrics = {
        "tool_selection_rate": sum(r["tool_selected"] for r in rows) / n,
        "param_match_rate": sum(r["params_matched"] for r in rows) / n,
        "numeric_accuracy_rate": (
            sum(bool(r["numeric_ok"]) for r in numeric_rows) / len(numeric_rows)
            if numeric_rows
            else 0.0
        ),
    }
    passed = (
        metrics["tool_selection_rate"] >= THRESHOLDS["tool_selection_rate"]
        and metrics["numeric_accuracy_rate"] >= THRESHOLDS["numeric_accuracy_rate"]
    )
    conn.close()
    return {"metrics": metrics, "thresholds": THRESHOLDS, "passed": passed, "rows": rows}


def main() -> None:
    """Run the eval, print metrics, save results JSON, exit nonzero on fail."""
    print("Execution Quality Copilot — Tool-Use Eval")
    print("=" * 60)
    results = run_evaluation()
    for k, v in results["metrics"].items():
        print(f"  {k:24s} {v:.2%}")
    print(f"  PASSED: {results['passed']}")

    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"eval_{date.today().isoformat()}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
```

File: `evals/results/.gitkeep` (empty)

- [ ] **Step 5: Run the scoring-function test to verify it passes**

Run: `pytest tests/unit/test_eval.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Run the full eval end-to-end (needs `ANTHROPIC_API_KEY` + `make gen-data`)**

```bash
cp .env.example .env   # then edit .env to add a real ANTHROPIC_API_KEY
make gen-data
make eval
```

Expected: prints `tool_selection_rate`, `param_match_rate`, `numeric_accuracy_rate`, `PASSED: True`, and saves `evals/results/eval_<date>.json`. If `PASSED` is False, inspect the saved `rows` to see which questions chose the wrong tool or missed the number, and tighten the offending tool **description** in `server/main.py` and the mirrored `TOOL_DEFS` description (descriptions are the lever — not the SQL).

- [ ] **Step 7: Commit**

```bash
git add evals/ tests/unit/test_eval.py
git commit -m "feat(eqc): tool-use eval harness (selection + params + numeric) [LIN-141]"
```

---

## Task 11: Docker packaging

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

The image bakes the seed dataset in at build time (running `make gen-data` inside the build) so a reviewer runs the server with one command and no extra setup.

- [ ] **Step 1: Write the `.dockerignore`**

File: `projects/execution-quality-copilot/.dockerignore`

```
.git
.venv
__pycache__
*.pyc
.pytest_cache
.ruff_cache
.mypy_cache
htmlcov
.coverage
data/*.duckdb
evals/results/*.json
.env
```

- [ ] **Step 2: Write the `Dockerfile`**

File: `projects/execution-quality-copilot/Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Copy the rest and build the synthetic dataset into the image.
COPY config.yaml ./
RUN python -m execution_quality_copilot.datagen

# MCP servers speak JSON-RPC over stdio; run the server as the entrypoint.
ENTRYPOINT ["python", "-m", "execution_quality_copilot.server.main"]
```

- [ ] **Step 3: Build and smoke-test the image**

```bash
docker build -t execution-quality-copilot .
# Server starts and waits on stdio; Ctrl-C to exit (no traceback = success).
docker run --rm -i execution-quality-copilot
```

Expected: image builds (the build log shows `Wrote 10000 fills to ...`), and `docker run` starts the server with no error.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat(eqc): dockerfile bakes seed data + runs mcp server [LIN-141]"
```

---

## Task 12: README + tool-design notes

**Files:**
- Create: `README.md`
- Create: `docs/tool-design-notes.md`

- [ ] **Step 1: Write the README**

File: `projects/execution-quality-copilot/README.md`

````markdown
# Execution Quality Copilot — TCA over MCP

A custom **MCP server** that turns a fund's (synthetic) execution records into a conversational
TCA analyst. A PM asks execution-quality questions in natural language in Claude Desktop; Claude
chooses which read-only tool to call, pulls exactly the data it needs, and answers with numbers
and caveats — no quant-team request required.

## Key Features
- **Five read-only tools** — `get_fills`, `get_benchmarks`, `compute_slippage`, `venue_breakdown`, `top_outliers`.
- **Guardrails by design** — tools are read-only, bounded (≤ 500 rows), and never run model-supplied SQL; the model picks *enums*, the server maps them to columns.
- **FIX-native synthetic data** — 10k ExecutionReports (35=8), 50 symbols, 4 brokers, 3 algos, with arrival/VWAP/close benchmarks.
- **Tool-use eval harness** — scores tool selection, parameter accuracy, and numeric-answer accuracy.

## Tech Stack
| Component | Technology |
|---|---|
| MCP server | MCP Python SDK / FastMCP (stdio) |
| Storage / analytics | DuckDB (in-process OLAP) |
| Data | Synthetic FIX ExecutionReport generator (deterministic) |
| Eval client | Anthropic SDK (`claude-opus-4-8`), manual tool-use loop |
| Packaging | Docker |

## Project Structure
```
src/execution_quality_copilot/  config.py, models.py, db.py, datagen/, server/{main.py, tools/}
evals/                          golden_questions.json, run_eval.py
tests/                          unit/ + integration/
```

## Setup
1. `pip install -e ".[dev]"`
2. `cp .env.example .env` and add `ANTHROPIC_API_KEY` (needed only for the eval — the server itself never calls an LLM)
3. `make gen-data` — build the synthetic `data/seed.duckdb`

## Run the MCP server
- Local: `make serve` (speaks JSON-RPC on stdio)
- Docker: `docker build -t execution-quality-copilot . && docker run --rm -i execution-quality-copilot`

## Register in Claude Desktop
Add to `claude_desktop_config.json` (`mcpServers` block), then restart Claude Desktop:
```json
{
  "mcpServers": {
    "execution-quality-copilot": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "execution-quality-copilot"]
    }
  }
}
```
Then ask, e.g.: *"Which broker cost me the most slippage on small-cap names last month?"*

## Eval
`make eval` — runs 15 golden questions and reports tool-selection / param-match / numeric-accuracy.

| Metric | Target |
|---|---|
| Tool-selection rate | ≥ 90% |
| Numeric-answer accuracy | ≥ 80% |

## ⚠️ Compliance Caveat
Demo uses **synthetic fills and mock benchmark data only**. Pointing this at real fills (even your own
PA trades) requires treating the data as confidential: local-only storage, no cloud LLM calls without
anonymisation, and a check against personal-conduct / data-handling policies first.

## License
MIT — built as part of an AI PM learning portfolio.
````

- [ ] **Step 2: Write the tool-design notes**

File: `projects/execution-quality-copilot/docs/tool-design-notes.md`

```markdown
# Tool Design Notes

The product surface of this project is the **tool contract**, not the analytics. These are the
design decisions and the rationale — the material an AI PM would defend in an interview.

## 1. Granularity: five tools, not one
A single `run_sql` tool would be maximally flexible and maximally dangerous. Instead, five
purpose-built tools each answer a class of PM question (inspect / benchmark / cost / where / worst).
This mirrors giving a client DMA with collar checks rather than a raw exchange session: the contract,
not the counterparty's goodwill, enforces safety.

## 2. Read-only, bounded, no raw SQL
- Connections are opened `read_only=True`.
- Every result is capped at 500 rows.
- The model never supplies SQL. Where a column choice is needed (which benchmark), the model picks
  an **enum** (`arrival`/`vwap`/`close`) and the server maps it to a column via a whitelist
  (`BENCHMARK_COLUMNS`). This is the LLM equivalent of pre-trade risk: the unsafe surface simply isn't exposed.

## 3. Descriptions are prompts
Tool descriptions state the unit (bps), the sign convention (positive = cost), when to pick which
benchmark, and which tool to prefer for which question ("for aggregate slippage use compute_slippage,
not get_fills"). Tuning these descriptions — not the SQL — is how eval failures get fixed.

## 4. One source of truth for behaviour, two for schemas
Pure functions in `server/tools/` are called by **both** the FastMCP server and the eval harness, so
there is no drift between what we tested and what we ship. The known tension: FastMCP derives schemas
from type hints, while the eval harness hand-writes Anthropic-format `TOOL_DEFS`. For the MVP these are
kept in sync by hand; a production system would generate both from one definition.

## 5. Agentic evals differ from RAG/structured-output evals
P1 measured retrieval; P2 measured structured-output validity. Here the axis is **did the model call
the right tool with the right arguments** — a behaviour eval, scored separately from the final number.
A model can get the number right by luck through the wrong tool; splitting the axes catches that.
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/tool-design-notes.md
git commit -m "docs(eqc): readme + tool-design notes [LIN-141]"
```

---

## Task 13: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full project CI**

```bash
cd /home/ubuntu/projects/ai-pm-portfolio/projects/execution-quality-copilot
make gen-data
make ci
```

Expected: `ruff check` clean, `ruff format --check` clean, `mypy` clean, all unit tests PASS. (The eval scoring-function tests and all tool/generator tests run without an API key; the live `make eval` is a separate, network-dependent check.)

- [ ] **Step 2: Run mypy explicitly (manual pre-commit stage)**

```bash
pre-commit run mypy --all-files
```

Expected: no type errors.

- [ ] **Step 3: Verify the live eval passes thresholds (needs `ANTHROPIC_API_KEY`)**

```bash
make eval
```

Expected: `tool_selection_rate ≥ 90%`, `numeric_accuracy_rate ≥ 80%`, `PASSED: True`. If not, iterate on tool descriptions (Task 10, Step 6) and re-run.

- [ ] **Step 4: Confirm the checklist**

Verify every box below is ticked, then the task is done:
1. ✅ 5 tools registered and served over stdio (`test_server.py`, `make serve`).
2. ✅ 10k synthetic fills, deterministic, regenerable (`make gen-data`).
3. ✅ All pure tools unit-tested with an in-memory DuckDB, no API key.
4. ✅ Eval harness scores selection + params + numeric and meets thresholds.
5. ✅ Dockerfile builds and runs with one command.
6. ✅ README + tool-design notes written; compliance caveat present.
7. ✅ `make ci` green; mypy clean.

- [ ] **Step 5: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill to open the PR into `main`. After merge, verify the merge landed in `main` before marking the Linear issue Done (per the repo's "verify merged before Linear Done" gate).

---

## Stretch Goals (outline — not part of this plan)

- **Real FIX drop-copy ingestion** — parse `35=8` messages from a log into the same `fills` schema, replacing the generator.
- **n8n nightly digest** — a scheduled workflow that calls `top_outliers` and emails "worst 5 executions yesterday" (also fills the Orchestration tracker gap).
- **Multi-asset** — extend to futures fills with their own benchmark conventions (settlement vs close).
- **Anomaly-flagging tool** — a sixth tool doing statistical outlier detection on the slippage distribution (z-score per broker/algo) rather than a fixed top-n.

---

## Self-Review (completed during planning)

**Spec coverage:** MVP item 1 (synthetic data: 10k/50/4/3 + arrival/VWAP/close) → T2/T3. Item 2 (MCP server, 5 tools, stdio, Claude Desktop) → T5–T8 + README registration block. Item 3 (demo transcript of PM questions) → README usage + `golden_questions.json` examples. Item 4 (eval: 15 questions, tool-selection + answer accuracy) → T10. Item 5 (Dockerfile, one command) → T11. The five named tools all appear with matching signatures across T5–T8, `TOOL_DEFS`, and the FastMCP wrappers.

**Type consistency:** `FILL_COLUMNS` order is used identically in the generator INSERT (T3), the conftest fixture (T5), and `Fill.to_row()` (T2). `BENCHMARK_COLUMNS` / `MAX_ROWS` / `_serialize` / `query` are defined once in `db.py` (T4) and imported by every tool. `run_tool_loop`'s return shape `(text, [(name, args)])` matches how `run_eval.py` consumes it. `answer_key` values (`slippage_bps`, `worst_slippage_bps`, `close_px`, `vwap`, `null`) all correspond to real keys returned by their expected tool.

**Placeholder scan:** every code step contains complete, runnable code; no "TBD"/"add error handling"/"similar to Task N". Tool error handling is concrete (`is_error` in the loop; `ValueError` on bad benchmark). The only deferred item is the documented MVP tension (two schema sources), called out explicitly rather than hidden.
