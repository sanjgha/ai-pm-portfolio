# RuleForge — Pre-Trade Risk Rules Assistant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **On approval:** copy this file to `docs/superpowers/plans/2026-06-14-pre-trade-risk-rules-assistant.md` (the superpowers-canonical location) before executing. The plan file you are reading is the working copy.

**Goal:** Build a LangGraph agent that converts a natural-language pre-trade risk rule into a validated, schema-conformant JSON config — with a generate → validate → self-correct loop that checks its own output before a human sees it, persists approved rules with an audit trail, and reads the rule back in plain English.

**Architecture:** A LangGraph state machine (`parse → generate → validate → {explain | correct→generate | escalate}`). Generation uses **non-strict Anthropic tool use** so the LLM's output can genuinely be *wrong*; a separate deterministic node validates it with Pydantic v2 (the "risk gateway") plus 5 domain lints. Validation failures are fed back to the generator for up to 2 retries, then escalated. Approved rules persist to SQLite + an append-only JSONL audit log.

**Tech Stack:** Python 3.10 · Anthropic SDK (direct, `claude-sonnet-4-6` agent / `claude-haiku-4-5` judge) · LangGraph · Pydantic v2 · FastAPI · SQLite (stdlib) · Streamlit · pytest.

---

## Context

This is **Portfolio Project #2** (`docs/pre-trade-risk-rules-assistant-spec.md`). Where the spec's flat `app/` layout conflicts with the monorepo, **the monorepo wins** (per the user): the project lives at `projects/pre-trade-risk-rules-assistant/` with a `src/` package, mirroring `projects/exchange-connectivity-hub/` (Project 1).

**Decisions locked in (from planning Q&A):**
- **Scope: MVP only** — 3 rule types (`order_notional_limit`, `price_collar`, `restricted_list`), the self-correct loop, 5 lints, a 20-case golden eval, and a Streamlit demo. Stretch goals are an outline at the end, not tasks.
- **Linear: one issue + one branch** — create a single "RuleForge MVP" Linear issue; work on `feat/<KEY>-ruleforge-mvp`. The branch name satisfies the pre-commit hook for every per-task commit. Replace `<KEY>` (e.g. `LIN-140`) everywhere below.

**Independent view & recommendation:** The spec frames structured output as "tool calling," and the Anthropic SDK now also offers `messages.parse()` (guaranteed-valid Pydantic output). I deliberately **avoid** the guaranteed-valid path for the *generator*: if the SDK guarantees schema conformance, the Pydantic validation node can never reject, and the self-correction loop — the whole point of this project — becomes dead code. Using **non-strict tool use** keeps Pydantic as a gateway that genuinely rejects (wrong type, out-of-range, missing field) and lets the loop fire on real failures. The harder, more realistic failures are the **domain lints** (currency≠exchange, insane collar) that no JSON schema can express — those are what make this look like a real risk system.

> **Teacher's note — why this differs from Project 1.** P1 used LangChain (framework-wrapped). Here you call the Anthropic SDK directly (`client.messages.create`) so you see exactly what a framework hides: you build the `tools=[...]` payload, force `tool_choice`, and pull `block.input` out yourself. And P1 was a *linear* RAG chain; this is a *cyclic graph* — the `validate → correct → generate` loop is impossible to express cleanly without a state machine like LangGraph.

---

## File Structure

```
projects/pre-trade-risk-rules-assistant/
├── README.md                       # T15
├── pyproject.toml                  # T0  (deps, ruff, mypy, pytest cov target)
├── Makefile                        # T0  (ci, test, lint, serve-api, serve-ui, eval)
├── config.yaml                     # T0  (models, lints, exchanges, storage, api/ui)
├── .env.example                    # T0  (ANTHROPIC_API_KEY)
├── src/pre_trade_risk_rules_assistant/
│   ├── __init__.py
│   ├── config.py                   # T1  YAML + dotenv, get_anthropic_api_key()
│   ├── schemas/rules.py            # T2  Pydantic rule models + discriminated union
│   ├── lints.py                    # T3  5 domain lints → list[str]
│   ├── store.py                    # T4  SQLite rules store + JSONL audit
│   ├── llm.py                      # T5  Anthropic client + call_tool / call_text
│   ├── state.py                    # T5  RuleForgeState TypedDict
│   ├── nodes/
│   │   ├── parser.py               # T6  parse_intent
│   │   ├── generator.py            # T7  generate_config (tool use)
│   │   ├── validator.py            # T8  validate_config (Pydantic + lints)
│   │   ├── corrector.py            # T9  self_correct + escalate + router
│   │   └── explainer.py            # T10 explain_rule (read-back)
│   ├── graph.py                    # T11 build_graph / run_graph
│   ├── api/main.py                 # T12 FastAPI endpoints
│   └── ui/app.py                   # T13 Streamlit demo
├── evals/
│   ├── golden_rules.json           # T14 20 NL → expected configs
│   ├── run_eval.py                 # T14 schema-pass %, field accuracy, intent fidelity
│   └── results/.gitkeep
├── data/                           # gitignored: rules.db, audit.jsonl
└── tests/{unit,integration}/       # one test file per module
```

**Graph (ASCII):**
```
        START
          │
        parse  ── extract rule_type + intent summary (tool use)
          │
   ┌──> generate ── emit JSON config for rule_type (NON-STRICT tool use)
   │      │
   │   validate ── Pydantic model_validate + run_lints  →  errors[]
   │      │
   │  route_after_validation
   │   ╱   │   ╲
   │ correct│   explain ──► END   (errors == [])
   │ (+1)   │
   └───┘  escalate ──► END         (errors and attempts >= MAX_ATTEMPTS=2)
```

**Shared type contract (defined in T2/T5, referenced everywhere):**
- `RuleType` values: `"order_notional_limit"`, `"price_collar"`, `"restricted_list"`.
- `RuleForgeState` keys: `request, rule_type, intent_summary, draft, errors, attempts, validated_rule, readback, status, escalation_reason`.
- `llm.call_tool(system, user, tool_name, tool_description, input_schema) -> dict` (raw, unvalidated).
- `llm.call_text(system, user, model=None, max_tokens=None) -> str`.
- Nodes import those two functions; **tests monkeypatch them on the node module** (e.g. `nodes.generator.call_tool`).

---

## Task 0: Scaffold project, Linear issue, branch, config

**Files:**
- Create: whole `projects/pre-trade-risk-rules-assistant/` skeleton
- Modify: `projects/pre-trade-risk-rules-assistant/pyproject.toml`, `Makefile`, `.gitignore`
- Create: `config.yaml`, `.env.example`

- [ ] **Step 1: Create the Linear issue + branch**

Create a Linear issue titled **"RuleForge MVP — Pre-Trade Risk Rules Assistant"** (Project Context: AI-PM Portfolio; Classification: priority Medium, label `feature`; Definition of Done = the spec's DoD). Note its key as `<KEY>`. Then from the repo root:

```bash
git checkout -b feat/<KEY>-ruleforge-mvp
```

- [ ] **Step 2: Scaffold the package**

```bash
cd /home/ubuntu/projects/ai-pm-portfolio
make new-project name=pre-trade-risk-rules-assistant desc="LangGraph agent: NL pre-trade risk rules -> validated JSON configs"
```

- [ ] **Step 3: Replace `pyproject.toml`**

File: `projects/pre-trade-risk-rules-assistant/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pre-trade-risk-rules-assistant"
version = "0.1.0"
description = "LangGraph agent converting NL pre-trade risk rules into validated JSON configs"
requires-python = ">=3.10"
dependencies = [
    "anthropic>=0.40.0",
    "langgraph>=0.2.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "streamlit>=1.39.0",
    "httpx>=0.27.0",
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
include = ["pre_trade_risk_rules_assistant*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=pre_trade_risk_rules_assistant --cov-report=term-missing"
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

File: `projects/pre-trade-risk-rules-assistant/Makefile`

```makefile
.PHONY: install dev-install test test-cov lint format typecheck ci clean serve-api serve-ui eval

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

serve-api:
	uvicorn pre_trade_risk_rules_assistant.api.main:app --reload --host 0.0.0.0 --port 8000

serve-ui:
	streamlit run src/pre_trade_risk_rules_assistant/ui/app.py

eval:
	python evals/run_eval.py
```

- [ ] **Step 5: Create `config.yaml`**

File: `projects/pre-trade-risk-rules-assistant/config.yaml`

```yaml
models:
  agent: claude-sonnet-4-6      # generation + read-back
  judge: claude-haiku-4-5       # eval intent-fidelity judge (cost-efficient)
  max_tokens: 2000

agent:
  max_correction_attempts: 2    # spec: max 2 retries, then escalate

exchanges: [SGX, HKEX, ASX, TSE]

exchange_currency:
  SGX: SGD
  HKEX: HKD
  ASX: AUD
  TSE: JPY

lints:
  min_collar_pct: 0.1
  max_collar_pct: 20.0
  min_notional: 1000.0
  max_notional: 1000000000.0    # 1bn sanity ceiling

storage:
  db_path: data/rules.db
  audit_path: data/audit.jsonl

api:
  host: 0.0.0.0
  port: 8000

ui:
  host: localhost
  port: 8501
```

- [ ] **Step 6: Create `.env.example` and gitignore data/**

File: `projects/pre-trade-risk-rules-assistant/.env.example`

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

Append to `projects/pre-trade-risk-rules-assistant/.gitignore` (create the lines if absent):

```
data/
evals/results/*.json
.env
```

Create empty markers:

```bash
cd projects/pre-trade-risk-rules-assistant
mkdir -p data evals/results
: > evals/results/.gitkeep
```

- [ ] **Step 7: Install and verify toolchain**

```bash
cd /home/ubuntu/projects/ai-pm-portfolio/projects/pre-trade-risk-rules-assistant
pip install -e ".[dev]"
```
Expected: installs anthropic, langgraph, pydantic, fastapi, streamlit without error.

- [ ] **Step 8: Commit**

```bash
git add projects/pre-trade-risk-rules-assistant
git commit -m "chore(ruleforge): scaffold project skeleton and config [<KEY>]"
```

> **Teacher's note.** All remaining commands run **inside** `projects/pre-trade-risk-rules-assistant/` (its own venv/deps), per the monorepo's per-project rule. Paths below are relative to that directory unless noted.

---

## Task 1: Config module

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_config.py`

```python
"""Test config loading."""

import os

import pytest

from pre_trade_risk_rules_assistant.config import get_anthropic_api_key, get_config


def test_get_config_has_expected_sections():
    config = get_config()
    for section in ("models", "agent", "exchanges", "exchange_currency", "lints", "storage"):
        assert section in config


def test_model_names_configured():
    config = get_config()
    assert config["models"]["agent"] == "claude-sonnet-4-6"
    assert config["models"]["judge"] == "claude-haiku-4-5"


def test_lint_thresholds():
    config = get_config()
    assert config["lints"]["max_collar_pct"] == 20.0
    assert config["agent"]["max_correction_attempts"] == 2


def test_anthropic_key_required():
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    assert get_anthropic_api_key() == "test-key"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_anthropic_api_key()
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
```

- [ ] **Step 2: Run it — expect failure**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: ... config`.

- [ ] **Step 3: Implement `config.py`**

File: `src/pre_trade_risk_rules_assistant/config.py`

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

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/test_config.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/config.py tests/unit/test_config.py
git commit -m "feat(ruleforge): config loader (yaml + dotenv) [<KEY>]"
```

---

## Task 2: Pydantic rule schemas

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/schemas/__init__.py`, `src/pre_trade_risk_rules_assistant/schemas/rules.py`
- Test: `tests/unit/test_schemas.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_schemas.py`

```python
"""Test Pydantic rule schemas and the discriminated union."""

import pytest
from pydantic import ValidationError

from pre_trade_risk_rules_assistant.schemas.rules import (
    OrderNotionalLimitRule,
    PriceCollarRule,
    RestrictedListRule,
    RuleAdapter,
    RuleType,
)


def _order_dict():
    return {
        "rule_type": "order_notional_limit",
        "description": "Block buy orders over 5m SGD on SGX",
        "scope": {"exchange": "SGX", "symbols": []},
        "side": "buy",
        "action": "block",
        "max_notional": 5000000,
        "currency": "SGD",
    }


def test_adapter_picks_order_notional_subtype():
    rule = RuleAdapter.validate_python(_order_dict())
    assert isinstance(rule, OrderNotionalLimitRule)
    assert rule.rule_type == RuleType.ORDER_NOTIONAL_LIMIT
    assert rule.max_notional == 5000000


def test_negative_notional_rejected():
    bad = _order_dict()
    bad["max_notional"] = -1
    with pytest.raises(ValidationError):
        RuleAdapter.validate_python(bad)


def test_collar_over_100_rejected():
    bad = {
        "rule_type": "price_collar",
        "description": "Collar",
        "scope": {"exchange": "HKEX", "symbols": []},
        "side": "both",
        "action": "warn",
        "collar_pct": 150,
        "reference_price": "last",
    }
    with pytest.raises(ValidationError):
        RuleAdapter.validate_python(bad)


def test_restricted_list_requires_symbols():
    bad = {
        "rule_type": "restricted_list",
        "description": "Restricted",
        "scope": {"exchange": "ASX", "symbols": []},
        "side": "both",
        "action": "block",
        "symbols": [],
    }
    with pytest.raises(ValidationError):
        RuleAdapter.validate_python(bad)


def test_price_collar_and_restricted_subtypes():
    collar = RuleAdapter.validate_python(
        {
            "rule_type": "price_collar",
            "description": "5% collar on SGX",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "both",
            "action": "warn",
            "collar_pct": 5,
            "reference_price": "last",
        }
    )
    assert isinstance(collar, PriceCollarRule)
    restricted = RuleAdapter.validate_python(
        {
            "rule_type": "restricted_list",
            "description": "No DBS",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "both",
            "action": "block",
            "symbols": ["D05"],
        }
    )
    assert isinstance(restricted, RestrictedListRule)


def test_unknown_exchange_rejected():
    bad = _order_dict()
    bad["scope"]["exchange"] = "NASDAQ"
    with pytest.raises(ValidationError):
        RuleAdapter.validate_python(bad)
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/unit/test_schemas.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement schemas**

File: `src/pre_trade_risk_rules_assistant/schemas/__init__.py`

```python
"""Pydantic rule schemas."""
```

File: `src/pre_trade_risk_rules_assistant/schemas/rules.py`

```python
"""Pydantic v2 schemas for pre-trade risk rules (the deterministic risk gateway)."""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class RuleType(str, Enum):
    ORDER_NOTIONAL_LIMIT = "order_notional_limit"
    PRICE_COLLAR = "price_collar"
    RESTRICTED_LIST = "restricted_list"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
    BOTH = "both"


class Action(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    REVIEW = "review"


class Exchange(str, Enum):
    SGX = "SGX"
    HKEX = "HKEX"
    ASX = "ASX"
    TSE = "TSE"


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exchange: Exchange
    symbols: list[str] = Field(default_factory=list)  # empty = all symbols on exchange
    segment: str | None = None  # e.g. "small_cap"


class BaseRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=3)
    scope: Scope
    side: Side = Side.BOTH
    action: Action = Action.BLOCK


class OrderNotionalLimitRule(BaseRule):
    rule_type: Literal[RuleType.ORDER_NOTIONAL_LIMIT]
    max_notional: float = Field(gt=0)
    currency: str = "USD"


class PriceCollarRule(BaseRule):
    rule_type: Literal[RuleType.PRICE_COLLAR]
    collar_pct: float = Field(gt=0, le=100)
    reference_price: Literal["last", "open", "vwap"] = "last"


class RestrictedListRule(BaseRule):
    rule_type: Literal[RuleType.RESTRICTED_LIST]
    symbols: list[str] = Field(min_length=1)  # restricted tickers


Rule = Annotated[
    Union[OrderNotionalLimitRule, PriceCollarRule, RestrictedListRule],
    Field(discriminator="rule_type"),
]

RuleAdapter: TypeAdapter[Rule] = TypeAdapter(Rule)
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/test_schemas.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/schemas tests/unit/test_schemas.py
git commit -m "feat(ruleforge): pydantic rule schemas + discriminated union [<KEY>]"
```

> **Teacher's note — discriminated union.** `RuleAdapter` reads the `rule_type` field and routes the dict to the right model. This is exactly how a real OMS demuxes a config blob. Pydantic's `gt=0`, `le=100`, `min_length` give you free range/cardinality checks that *catch LLM mistakes* — which is the whole reason we don't force the model to be valid.

---

## Task 3: Domain lints

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/lints.py`
- Test: `tests/unit/test_lints.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_lints.py`

```python
"""Test the 5 domain lints (semantic checks beyond schema)."""

from pre_trade_risk_rules_assistant.lints import run_lints
from pre_trade_risk_rules_assistant.schemas.rules import RuleAdapter


def _rule(d):
    return RuleAdapter.validate_python(d)


def test_clean_order_rule_passes():
    rule = _rule(
        {
            "rule_type": "order_notional_limit",
            "description": "Block buy > 5m SGD on SGX",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "buy",
            "action": "block",
            "max_notional": 5000000,
            "currency": "SGD",
        }
    )
    assert run_lints(rule) == []


def test_currency_mismatch_flagged():
    rule = _rule(
        {
            "rule_type": "order_notional_limit",
            "description": "Block buy > 5m on SGX",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "buy",
            "action": "block",
            "max_notional": 5000000,
            "currency": "USD",  # SGX expects SGD
        }
    )
    errs = run_lints(rule)
    assert any("currency" in e.lower() for e in errs)


def test_notional_over_ceiling_flagged():
    rule = _rule(
        {
            "rule_type": "order_notional_limit",
            "description": "huge",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "buy",
            "action": "block",
            "max_notional": 5_000_000_000,  # > 1bn ceiling
            "currency": "SGD",
        }
    )
    assert any("notional" in e.lower() for e in run_lints(rule))


def test_collar_out_of_band_flagged():
    rule = _rule(
        {
            "rule_type": "price_collar",
            "description": "wide collar",
            "scope": {"exchange": "HKEX", "symbols": []},
            "side": "both",
            "action": "warn",
            "collar_pct": 50,  # > 20% band
            "reference_price": "last",
        }
    )
    assert any("collar" in e.lower() for e in run_lints(rule))


def test_malformed_restricted_symbols_flagged():
    rule = _rule(
        {
            "rule_type": "restricted_list",
            "description": "bad tickers",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "both",
            "action": "block",
            "symbols": ["d05", "D05"],  # lowercase + duplicate
        }
    )
    errs = run_lints(rule)
    assert any("symbol" in e.lower() for e in errs)
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/unit/test_lints.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `lints.py`**

File: `src/pre_trade_risk_rules_assistant/lints.py`

```python
"""Domain lint checks: semantic validation beyond what JSON schema can express."""

import re

from pre_trade_risk_rules_assistant.config import get_config
from pre_trade_risk_rules_assistant.schemas.rules import (
    OrderNotionalLimitRule,
    PriceCollarRule,
    RestrictedListRule,
)

_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}$")


def _malformed(symbols: list[str]) -> list[str]:
    """Return the subset of symbols that are NOT well-formed uppercase tickers."""
    return [s for s in symbols if not _TICKER_RE.match(s)]


def run_lints(rule: object) -> list[str]:
    """Run all applicable domain lints; return human-readable error strings ([] = pass)."""
    cfg = get_config()
    lints_cfg = cfg["lints"]
    errors: list[str] = []

    # L5 (all rules): scope.symbols must be well-formed tickers.
    bad_scope = _malformed(getattr(rule, "scope").symbols)
    if bad_scope:
        errors.append(f"scope.symbols not well-formed tickers: {bad_scope}")

    if isinstance(rule, OrderNotionalLimitRule):
        # L2: notional within sane band.
        if rule.max_notional > lints_cfg["max_notional"]:
            errors.append(
                f"max_notional {rule.max_notional} exceeds ceiling {lints_cfg['max_notional']}"
            )
        if rule.max_notional < lints_cfg["min_notional"]:
            errors.append(
                f"max_notional {rule.max_notional} below floor {lints_cfg['min_notional']}"
            )
        # L3: currency must match the exchange's expected currency (units consistency).
        expected = cfg["exchange_currency"].get(rule.scope.exchange.value)
        if expected and rule.currency.upper() != expected:
            errors.append(
                f"currency {rule.currency} inconsistent with {rule.scope.exchange.value} "
                f"(expected {expected})"
            )

    elif isinstance(rule, PriceCollarRule):
        # L1: collar within the sane band.
        if not (lints_cfg["min_collar_pct"] <= rule.collar_pct <= lints_cfg["max_collar_pct"]):
            errors.append(
                f"collar_pct {rule.collar_pct} outside sane band "
                f"[{lints_cfg['min_collar_pct']}, {lints_cfg['max_collar_pct']}]"
            )

    elif isinstance(rule, RestrictedListRule):
        # L4: restricted symbols well-formed and unique.
        bad = _malformed(rule.symbols)
        if bad:
            errors.append(f"restricted symbols not well-formed: {bad}")
        if len(set(rule.symbols)) != len(rule.symbols):
            errors.append(f"restricted symbols contain duplicates: {rule.symbols}")

    return errors
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/test_lints.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/lints.py tests/unit/test_lints.py
git commit -m "feat(ruleforge): 5 domain lint checks [<KEY>]"
```

---

## Task 4: SQLite store + JSONL audit log

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/store.py`
- Test: `tests/unit/test_store.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_store.py`

```python
"""Test the SQLite rules store and JSONL audit log."""

import json

from pre_trade_risk_rules_assistant import store


def _rule_dict():
    return {
        "rule_type": "price_collar",
        "description": "5% collar on SGX",
        "scope": {"exchange": "SGX", "symbols": [], "segment": None},
        "side": "both",
        "action": "warn",
        "collar_pct": 5.0,
        "reference_price": "last",
    }


def test_save_then_get_roundtrip(tmp_path):
    db = tmp_path / "rules.db"
    audit = tmp_path / "audit.jsonl"
    rule_id = store.save_rule(_rule_dict(), db_path=db, audit_path=audit)
    assert rule_id
    fetched = store.get_rule(rule_id, db_path=db)
    assert fetched is not None
    assert fetched["config"]["collar_pct"] == 5.0
    assert fetched["rule_type"] == "price_collar"


def test_get_missing_returns_none(tmp_path):
    db = tmp_path / "rules.db"
    assert store.get_rule("nope", db_path=db) is None


def test_audit_log_appended(tmp_path):
    db = tmp_path / "rules.db"
    audit = tmp_path / "audit.jsonl"
    rule_id = store.save_rule(_rule_dict(), db_path=db, audit_path=audit)
    lines = audit.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "rule_saved"
    assert entry["rule_id"] == rule_id
    assert "timestamp" in entry
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/unit/test_store.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `store.py`**

File: `src/pre_trade_risk_rules_assistant/store.py`

```python
"""Persistence: SQLite rules store + append-only JSONL audit log."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pre_trade_risk_rules_assistant.config import get_config


def _default_paths() -> tuple[Path, Path]:
    cfg = get_config()["storage"]
    return Path(cfg["db_path"]), Path(cfg["audit_path"])


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rules (
            id TEXT PRIMARY KEY,
            rule_type TEXT NOT NULL,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def _append_audit(audit_path: Path, entry: dict[str, Any]) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_rule(
    rule: dict[str, Any],
    db_path: Path | None = None,
    audit_path: Path | None = None,
) -> str:
    """Persist an approved rule and append an audit entry. Returns the new rule id."""
    dflt_db, dflt_audit = _default_paths()
    db_path = db_path or dflt_db
    audit_path = audit_path or dflt_audit

    rule_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO rules (id, rule_type, config_json, created_at) VALUES (?, ?, ?, ?)",
            (rule_id, rule["rule_type"], json.dumps(rule), created_at),
        )
        conn.commit()
    finally:
        conn.close()

    _append_audit(
        audit_path,
        {
            "event": "rule_saved",
            "rule_id": rule_id,
            "rule_type": rule["rule_type"],
            "timestamp": created_at,
        },
    )
    return rule_id


def get_rule(rule_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    """Fetch a stored rule by id, or None."""
    dflt_db, _ = _default_paths()
    db_path = db_path or dflt_db
    if not db_path.exists():
        return None
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, rule_type, config_json, created_at FROM rules WHERE id = ?",
            (rule_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "rule_id": row[0],
        "rule_type": row[1],
        "config": json.loads(row[2]),
        "created_at": row[3],
    }
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/test_store.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/store.py tests/unit/test_store.py
git commit -m "feat(ruleforge): sqlite rules store + jsonl audit log [<KEY>]"
```

---

## Task 5: LLM helpers + graph state

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/llm.py`, `src/pre_trade_risk_rules_assistant/state.py`
- Test: `tests/unit/test_llm.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_llm.py`

```python
"""Test the Anthropic call helpers (with a fake client, no network)."""

from types import SimpleNamespace

from pre_trade_risk_rules_assistant import llm


class _FakeMessages:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        self._last = kwargs
        return SimpleNamespace(content=self._content)


def test_call_tool_returns_tool_input(monkeypatch):
    tool_block = SimpleNamespace(type="tool_use", input={"rule_type": "price_collar"})
    fake = SimpleNamespace(messages=_FakeMessages([tool_block]))
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    out = llm.call_tool(
        system="s", user="u", tool_name="t", tool_description="d", input_schema={"type": "object"}
    )
    assert out == {"rule_type": "price_collar"}


def test_call_tool_raises_without_tool_use(monkeypatch):
    text_block = SimpleNamespace(type="text", text="no tool")
    fake = SimpleNamespace(messages=_FakeMessages([text_block]))
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    try:
        llm.call_tool(system="s", user="u", tool_name="t", tool_description="d", input_schema={})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_call_text_concatenates_text_blocks(monkeypatch):
    blocks = [SimpleNamespace(type="text", text="Hello "), SimpleNamespace(type="text", text="world")]
    fake = SimpleNamespace(messages=_FakeMessages(blocks))
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    assert llm.call_text(system="s", user="u") == "Hello world"
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/unit/test_llm.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `llm.py` and `state.py`**

File: `src/pre_trade_risk_rules_assistant/llm.py`

```python
"""Thin wrappers around the Anthropic SDK: forced single-tool call + plain text call."""

from typing import Any

import anthropic

from pre_trade_risk_rules_assistant.config import get_anthropic_api_key, get_config


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_anthropic_api_key())


def call_tool(
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    """Force the model to emit ONE tool call; return its raw (unvalidated) input dict.

    Non-strict on purpose: the model CAN produce out-of-range / malformed input,
    which the validator node then catches and feeds back into the self-correct loop.
    """
    cfg = get_config()
    client = get_client()
    resp = client.messages.create(
        model=model or cfg["models"]["agent"],
        max_tokens=cfg["models"]["max_tokens"],
        system=system,
        tools=[
            {"name": tool_name, "description": tool_description, "input_schema": input_schema}
        ],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return dict(block.input)
    raise ValueError("Model did not return a tool_use block")


def call_text(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Plain text completion (used for the read-back and the eval judge)."""
    cfg = get_config()
    client = get_client()
    resp = client.messages.create(
        model=model or cfg["models"]["agent"],
        max_tokens=max_tokens or 1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")
```

File: `src/pre_trade_risk_rules_assistant/state.py`

```python
"""LangGraph shared state for the RuleForge agent."""

from typing import Any, TypedDict


class RuleForgeState(TypedDict, total=False):
    request: str               # original NL rule description
    rule_type: str             # set by parser
    intent_summary: str        # set by parser
    draft: dict[str, Any]      # raw config from generator (unvalidated)
    errors: list[str]          # validation + lint errors (set by validator)
    attempts: int              # self-correction attempts so far
    validated_rule: dict[str, Any]  # validated config (set on success)
    readback: str              # plain-English explanation (set by explainer)
    status: str                # "ok" | "escalated"
    escalation_reason: str
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/test_llm.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/llm.py src/pre_trade_risk_rules_assistant/state.py tests/unit/test_llm.py
git commit -m "feat(ruleforge): anthropic call helpers + graph state [<KEY>]"
```

> **Teacher's note — `tool_choice` forcing.** `{"type": "tool", "name": ...}` makes the model emit that tool's input and nothing else — the cleanest way to get JSON out of Claude via the raw SDK. We read `block.input` (already a dict). We do NOT set `strict: True`, so the schema only *guides* the model; enforcement is our job (Task 8).

---

## Task 6: Parser node

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/nodes/__init__.py`, `src/pre_trade_risk_rules_assistant/nodes/parser.py`
- Test: `tests/unit/test_parser_node.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_parser_node.py`

```python
"""Test the intent parser node (LLM mocked)."""

from pre_trade_risk_rules_assistant.nodes import parser


def test_parse_intent_sets_rule_type(monkeypatch):
    def fake_call_tool(**kwargs):
        return {"rule_type": "order_notional_limit", "intent_summary": "cap buy notional"}

    monkeypatch.setattr(parser, "call_tool", fake_call_tool)
    out = parser.parse_intent({"request": "block buy orders over 5m on SGX"})
    assert out["rule_type"] == "order_notional_limit"
    assert out["intent_summary"] == "cap buy notional"
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/unit/test_parser_node.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement parser node**

File: `src/pre_trade_risk_rules_assistant/nodes/__init__.py`

```python
"""LangGraph nodes."""
```

File: `src/pre_trade_risk_rules_assistant/nodes/parser.py`

```python
"""Intent parser node: extract rule_type + a normalized intent summary."""

from typing import Any

from pre_trade_risk_rules_assistant.llm import call_tool
from pre_trade_risk_rules_assistant.schemas.rules import RuleType

_PARSER_SYSTEM = (
    "You are a pre-trade risk analyst. Read a natural-language risk rule and classify it. "
    "Choose exactly one rule_type and write a one-sentence normalized summary of intent."
)

_PARSER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rule_type": {"type": "string", "enum": [t.value for t in RuleType]},
        "intent_summary": {"type": "string"},
    },
    "required": ["rule_type", "intent_summary"],
}


def parse_intent(state: dict[str, Any]) -> dict[str, Any]:
    parsed = call_tool(
        system=_PARSER_SYSTEM,
        user=state["request"],
        tool_name="classify_rule",
        tool_description="Classify the rule type and summarize intent.",
        input_schema=_PARSER_SCHEMA,
    )
    return {
        "rule_type": parsed["rule_type"],
        "intent_summary": parsed.get("intent_summary", ""),
    }
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/test_parser_node.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/nodes/__init__.py src/pre_trade_risk_rules_assistant/nodes/parser.py tests/unit/test_parser_node.py
git commit -m "feat(ruleforge): intent parser node [<KEY>]"
```

---

## Task 7: Generator node

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/nodes/generator.py`
- Test: `tests/unit/test_generator_node.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_generator_node.py`

```python
"""Test the config generator node (LLM mocked)."""

from pre_trade_risk_rules_assistant.nodes import generator


def test_generate_uses_rule_type_schema_and_returns_draft(monkeypatch):
    captured = {}

    def fake_call_tool(**kwargs):
        captured.update(kwargs)
        return {"rule_type": "price_collar", "collar_pct": 5}

    monkeypatch.setattr(generator, "call_tool", fake_call_tool)
    out = generator.generate_config(
        {"request": "5% collar on SGX", "rule_type": "price_collar", "intent_summary": "collar"}
    )
    assert out["draft"] == {"rule_type": "price_collar", "collar_pct": 5}
    # The price_collar schema must drive the tool input_schema.
    assert "collar_pct" in captured["input_schema"]["properties"]


def test_generate_includes_prior_errors_in_prompt(monkeypatch):
    captured = {}

    def fake_call_tool(**kwargs):
        captured.update(kwargs)
        return {"rule_type": "price_collar"}

    monkeypatch.setattr(generator, "call_tool", fake_call_tool)
    generator.generate_config(
        {
            "request": "collar",
            "rule_type": "price_collar",
            "intent_summary": "c",
            "errors": ["collar_pct 50 outside sane band [0.1, 20.0]"],
        }
    )
    assert "outside sane band" in captured["user"]
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/unit/test_generator_node.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement generator node**

File: `src/pre_trade_risk_rules_assistant/nodes/generator.py`

```python
"""Config generator node: emit a JSON rule config conforming to the rule_type's schema."""

from typing import Any

from pre_trade_risk_rules_assistant.llm import call_tool
from pre_trade_risk_rules_assistant.schemas.rules import (
    OrderNotionalLimitRule,
    PriceCollarRule,
    RestrictedListRule,
)

RULE_MODELS = {
    "order_notional_limit": OrderNotionalLimitRule,
    "price_collar": PriceCollarRule,
    "restricted_list": RestrictedListRule,
}

_GENERATOR_SYSTEM = (
    "You are a pre-trade risk engineer. Emit a single rule config that exactly matches the "
    "provided tool schema. Use the correct currency for the exchange (SGX=SGD, HKEX=HKD, "
    "ASX=AUD, TSE=JPY). Keep price collars within a sane band (0.1%-20%). Tickers are "
    "UPPERCASE alphanumeric."
)


def _build_user_prompt(state: dict[str, Any]) -> str:
    parts = [
        f"Original request: {state['request']}",
        f"Intent: {state.get('intent_summary', '')}",
        f"Rule type: {state['rule_type']}",
    ]
    if state.get("errors"):
        parts.append(
            "Your previous attempt FAILED validation with these errors — fix ALL of them:\n"
            + "\n".join(f"- {e}" for e in state["errors"])
        )
    return "\n".join(parts)


def generate_config(state: dict[str, Any]) -> dict[str, Any]:
    model_cls = RULE_MODELS[state["rule_type"]]
    schema = model_cls.model_json_schema()
    draft = call_tool(
        system=_GENERATOR_SYSTEM,
        user=_build_user_prompt(state),
        tool_name="emit_rule",
        tool_description="Emit the rule config as structured JSON.",
        input_schema=schema,
    )
    return {"draft": draft}
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/test_generator_node.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/nodes/generator.py tests/unit/test_generator_node.py
git commit -m "feat(ruleforge): config generator node (tool use) [<KEY>]"
```

> **Teacher's note — self-correction is "just" a prompt.** On a retry, `state["errors"]` is pasted into the prompt with "fix ALL of them." The graph wiring (Task 11) is what loops back here; the node itself is stateless. That separation — deterministic control flow in the graph, LLM work in the nodes — is the core LangGraph idea.

---

## Task 8: Validator node

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/nodes/validator.py`
- Test: `tests/unit/test_validator_node.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_validator_node.py`

```python
"""Test the validator node: Pydantic validation + domain lints."""

from pre_trade_risk_rules_assistant.nodes import validator


def _valid_collar():
    return {
        "rule_type": "price_collar",
        "description": "5% collar on SGX",
        "scope": {"exchange": "SGX", "symbols": []},
        "side": "both",
        "action": "warn",
        "collar_pct": 5,
        "reference_price": "last",
    }


def test_valid_draft_sets_validated_rule_no_errors():
    out = validator.validate_config({"draft": _valid_collar()})
    assert out["errors"] == []
    assert out["validated_rule"]["collar_pct"] == 5


def test_schema_error_populates_errors():
    bad = _valid_collar()
    bad["collar_pct"] = 999  # > 100 -> Pydantic error
    out = validator.validate_config({"draft": bad})
    assert out["errors"]
    assert "validated_rule" not in out


def test_lint_error_populates_errors():
    bad = _valid_collar()
    bad["collar_pct"] = 50  # passes schema (<=100) but fails the 20% lint band
    out = validator.validate_config({"draft": bad})
    assert any("collar" in e.lower() for e in out["errors"])
    assert "validated_rule" not in out
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/unit/test_validator_node.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement validator node**

File: `src/pre_trade_risk_rules_assistant/nodes/validator.py`

```python
"""Validator node: deterministic Pydantic validation + domain lints (the risk gateway)."""

from typing import Any

from pydantic import ValidationError

from pre_trade_risk_rules_assistant.lints import run_lints
from pre_trade_risk_rules_assistant.schemas.rules import RuleAdapter


def validate_config(state: dict[str, Any]) -> dict[str, Any]:
    draft = state["draft"]
    try:
        rule = RuleAdapter.validate_python(draft)
    except ValidationError as exc:
        errors = [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        ]
        return {"errors": errors}

    lint_errors = run_lints(rule)
    if lint_errors:
        return {"errors": lint_errors}

    return {"errors": [], "validated_rule": rule.model_dump(mode="json")}
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/test_validator_node.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/nodes/validator.py tests/unit/test_validator_node.py
git commit -m "feat(ruleforge): validator node (pydantic + lints) [<KEY>]"
```

---

## Task 9: Corrector node, escalate node, router

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/nodes/corrector.py`
- Test: `tests/unit/test_corrector_node.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_corrector_node.py`

```python
"""Test the self-correct node, escalate node, and the post-validation router."""

from pre_trade_risk_rules_assistant.nodes import corrector


def test_self_correct_increments_attempts():
    assert corrector.self_correct({"attempts": 0})["attempts"] == 1
    assert corrector.self_correct({"attempts": 1})["attempts"] == 2
    assert corrector.self_correct({})["attempts"] == 1


def test_escalate_sets_status_and_reason():
    out = corrector.escalate({"errors": ["a", "b"]})
    assert out["status"] == "escalated"
    assert "a" in out["escalation_reason"]


def test_router_no_errors_goes_to_explain():
    assert corrector.route_after_validation({"errors": [], "attempts": 0}) == "explain"


def test_router_with_budget_left_corrects():
    assert corrector.route_after_validation({"errors": ["x"], "attempts": 0}) == "correct"
    assert corrector.route_after_validation({"errors": ["x"], "attempts": 1}) == "correct"


def test_router_exhausted_escalates():
    assert corrector.route_after_validation({"errors": ["x"], "attempts": 2}) == "escalate"
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/unit/test_corrector_node.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement corrector node + router**

File: `src/pre_trade_risk_rules_assistant/nodes/corrector.py`

```python
"""Self-correction control: bump attempts, escalate, and route after validation."""

from typing import Any

from pre_trade_risk_rules_assistant.config import get_config


def self_correct(state: dict[str, Any]) -> dict[str, Any]:
    """Increment the correction counter; errors are already in state for the generator."""
    return {"attempts": state.get("attempts", 0) + 1}


def escalate(state: dict[str, Any]) -> dict[str, Any]:
    """Terminal failure: surface a diagnostic for the human."""
    return {
        "status": "escalated",
        "escalation_reason": "; ".join(state.get("errors", [])),
    }


def route_after_validation(state: dict[str, Any]) -> str:
    """Conditional edge: explain (clean) | correct (retry) | escalate (budget exhausted)."""
    if not state.get("errors"):
        return "explain"
    max_attempts = get_config()["agent"]["max_correction_attempts"]
    if state.get("attempts", 0) >= max_attempts:
        return "escalate"
    return "correct"
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/test_corrector_node.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/nodes/corrector.py tests/unit/test_corrector_node.py
git commit -m "feat(ruleforge): self-correct + escalate + router [<KEY>]"
```

---

## Task 10: Explainer node

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/nodes/explainer.py`
- Test: `tests/unit/test_explainer_node.py`

- [ ] **Step 1: Write the failing test**

File: `tests/unit/test_explainer_node.py`

```python
"""Test the read-back explainer node (LLM mocked)."""

from pre_trade_risk_rules_assistant.nodes import explainer


def test_explain_sets_status_ok_and_readback(monkeypatch):
    monkeypatch.setattr(
        explainer, "call_text", lambda **kwargs: "This rule blocks buy orders over 5m SGD on SGX."
    )
    out = explainer.explain_rule({"validated_rule": {"rule_type": "order_notional_limit"}})
    assert out["status"] == "ok"
    assert "blocks buy orders" in out["readback"]
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/unit/test_explainer_node.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement explainer node**

File: `src/pre_trade_risk_rules_assistant/nodes/explainer.py`

```python
"""Read-back explainer node: plain-English confirmation of the validated rule."""

import json
from typing import Any

from pre_trade_risk_rules_assistant.llm import call_text

_EXPLAINER_SYSTEM = (
    "You are a pre-trade risk analyst. Given a validated rule config as JSON, write a single "
    "plain-English sentence a trading desk BA can confirm. State the action, side, scope, and "
    "threshold. No JSON, no preamble."
)


def explain_rule(state: dict[str, Any]) -> dict[str, Any]:
    readback = call_text(
        system=_EXPLAINER_SYSTEM,
        user=json.dumps(state["validated_rule"]),
    )
    return {"status": "ok", "readback": readback.strip()}
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/unit/test_explainer_node.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/nodes/explainer.py tests/unit/test_explainer_node.py
git commit -m "feat(ruleforge): read-back explainer node [<KEY>]"
```

---

## Task 11: Graph wiring + run_graph

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/graph.py`
- Test: `tests/integration/__init__.py`, `tests/integration/test_graph.py`

- [ ] **Step 1: Write the failing test**

File: `tests/integration/__init__.py` — empty.

File: `tests/integration/test_graph.py`

```python
"""End-to-end graph tests with all LLM nodes mocked (covers the 3 paths)."""

from pre_trade_risk_rules_assistant import graph
from pre_trade_risk_rules_assistant.nodes import explainer, generator, parser


def _mock_parser(monkeypatch, rule_type="price_collar"):
    monkeypatch.setattr(
        parser, "call_tool", lambda **k: {"rule_type": rule_type, "intent_summary": "c"}
    )


def _valid_collar():
    return {
        "rule_type": "price_collar",
        "description": "5% collar on SGX",
        "scope": {"exchange": "SGX", "symbols": []},
        "side": "both",
        "action": "warn",
        "collar_pct": 5,
        "reference_price": "last",
    }


def test_happy_path_first_try(monkeypatch):
    _mock_parser(monkeypatch)
    monkeypatch.setattr(generator, "call_tool", lambda **k: _valid_collar())
    monkeypatch.setattr(explainer, "call_text", lambda **k: "readback")
    final = graph.run_graph("5% collar on SGX")
    assert final["status"] == "ok"
    assert final["attempts"] == 0
    assert final["validated_rule"]["collar_pct"] == 5
    assert final["readback"] == "readback"


def test_self_corrects_then_succeeds(monkeypatch):
    _mock_parser(monkeypatch)
    calls = {"n": 0}

    def flaky(**k):
        calls["n"] += 1
        if calls["n"] == 1:
            bad = _valid_collar()
            bad["collar_pct"] = 50  # fails lint -> triggers correction
            return bad
        return _valid_collar()

    monkeypatch.setattr(generator, "call_tool", flaky)
    monkeypatch.setattr(explainer, "call_text", lambda **k: "readback")
    final = graph.run_graph("collar")
    assert final["status"] == "ok"
    assert final["attempts"] == 1
    assert calls["n"] == 2


def test_escalates_after_two_failures(monkeypatch):
    _mock_parser(monkeypatch)

    def always_bad(**k):
        bad = _valid_collar()
        bad["collar_pct"] = 50
        return bad

    monkeypatch.setattr(generator, "call_tool", always_bad)
    final = graph.run_graph("collar")
    assert final["status"] == "escalated"
    assert final["attempts"] == 2
    assert "collar" in final["escalation_reason"].lower()
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/integration/test_graph.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `graph.py`**

File: `src/pre_trade_risk_rules_assistant/graph.py`

```python
"""LangGraph state machine: parse -> generate -> validate -> {explain | correct | escalate}."""

from langgraph.graph import END, START, StateGraph

from pre_trade_risk_rules_assistant.nodes.corrector import (
    escalate,
    route_after_validation,
    self_correct,
)
from pre_trade_risk_rules_assistant.nodes.explainer import explain_rule
from pre_trade_risk_rules_assistant.nodes.generator import generate_config
from pre_trade_risk_rules_assistant.nodes.parser import parse_intent
from pre_trade_risk_rules_assistant.nodes.validator import validate_config
from pre_trade_risk_rules_assistant.state import RuleForgeState


def build_graph():
    g = StateGraph(RuleForgeState)
    g.add_node("parse", parse_intent)
    g.add_node("generate", generate_config)
    g.add_node("validate", validate_config)
    g.add_node("correct", self_correct)
    g.add_node("explain", explain_rule)
    g.add_node("escalate", escalate)

    g.add_edge(START, "parse")
    g.add_edge("parse", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges(
        "validate",
        route_after_validation,
        {"explain": "explain", "correct": "correct", "escalate": "escalate"},
    )
    g.add_edge("correct", "generate")  # the self-correction cycle
    g.add_edge("explain", END)
    g.add_edge("escalate", END)
    return g.compile()


_APP = None


def run_graph(request: str) -> RuleForgeState:
    """Run the agent on one NL request; return the final state."""
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP.invoke({"request": request, "attempts": 0})
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/integration/test_graph.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/graph.py tests/integration/__init__.py tests/integration/test_graph.py
git commit -m "feat(ruleforge): langgraph wiring + run_graph [<KEY>]"
```

> **Teacher's note — the cycle.** `add_edge("correct", "generate")` is the one line that makes this an *agent* and not a pipeline. `route_after_validation` is the conditional edge — a plain Python function returning a label that LangGraph maps to the next node. The loop is bounded by `attempts >= max_correction_attempts`, so it can never spin forever.

---

## Task 12: FastAPI endpoints

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/api/__init__.py`, `src/pre_trade_risk_rules_assistant/api/main.py`
- Test: `tests/integration/test_api.py`

- [ ] **Step 1: Write the failing test**

File: `tests/integration/test_api.py`

```python
"""Test the FastAPI layer (graph + store mocked)."""

from fastapi.testclient import TestClient

from pre_trade_risk_rules_assistant.api import main


def _client():
    return TestClient(main.app)


def test_draft_ok_persists_and_returns_rule(monkeypatch):
    monkeypatch.setattr(
        main,
        "run_graph",
        lambda req: {
            "status": "ok",
            "attempts": 0,
            "validated_rule": {"rule_type": "price_collar", "collar_pct": 5},
            "readback": "5% collar",
        },
    )
    monkeypatch.setattr(main.store, "save_rule", lambda rule: "rule-123")
    resp = _client().post("/rules/draft", json={"request": "5% collar on SGX"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["rule_id"] == "rule-123"
    assert body["readback"] == "5% collar"


def test_draft_escalated_returns_errors(monkeypatch):
    monkeypatch.setattr(
        main,
        "run_graph",
        lambda req: {
            "status": "escalated",
            "attempts": 2,
            "errors": ["collar bad"],
            "escalation_reason": "collar bad",
        },
    )
    resp = _client().post("/rules/draft", json={"request": "bad"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "escalated"
    assert body["errors"] == ["collar bad"]
    assert "rule_id" not in body


def test_get_rule_found_and_missing(monkeypatch):
    monkeypatch.setattr(
        main.store, "get_rule", lambda rid: {"rule_id": rid, "config": {}} if rid == "x" else None
    )
    client = _client()
    assert client.get("/rules/x").status_code == 200
    assert client.get("/rules/missing").status_code == 404
```

- [ ] **Step 2: Run — expect failure**

Run: `pytest tests/integration/test_api.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `api/main.py`**

File: `src/pre_trade_risk_rules_assistant/api/__init__.py`

```python
"""FastAPI app."""
```

File: `src/pre_trade_risk_rules_assistant/api/main.py`

```python
"""FastAPI endpoints: draft a rule and fetch a stored rule."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pre_trade_risk_rules_assistant import store
from pre_trade_risk_rules_assistant.graph import run_graph

app = FastAPI(title="RuleForge — Pre-Trade Risk Rules Assistant")


class DraftRequest(BaseModel):
    request: str


@app.post("/rules/draft")
def draft_rule(body: DraftRequest) -> dict[str, Any]:
    state = run_graph(body.request)
    if state.get("status") == "ok":
        rule_id = store.save_rule(state["validated_rule"])
        return {
            "status": "ok",
            "rule_id": rule_id,
            "config": state["validated_rule"],
            "readback": state.get("readback", ""),
            "attempts": state.get("attempts", 0),
        }
    return {
        "status": "escalated",
        "errors": state.get("errors", []),
        "escalation_reason": state.get("escalation_reason", ""),
        "attempts": state.get("attempts", 0),
    }


@app.get("/rules/{rule_id}")
def get_rule(rule_id: str) -> dict[str, Any]:
    rule = store.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="rule not found")
    return rule
```

- [ ] **Step 4: Run — expect pass**

Run: `pytest tests/integration/test_api.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/api tests/integration/test_api.py
git commit -m "feat(ruleforge): fastapi draft + get endpoints [<KEY>]"
```

---

## Task 13: Streamlit demo

**Files:**
- Create: `src/pre_trade_risk_rules_assistant/ui/__init__.py`, `src/pre_trade_risk_rules_assistant/ui/app.py`

> Streamlit UI is verified manually (no unit test — UI logic is thin and the agent is already covered). Keep it a thin client over `run_graph` + `store`.

- [ ] **Step 1: Implement the Streamlit app**

File: `src/pre_trade_risk_rules_assistant/ui/__init__.py`

```python
"""Streamlit UI."""
```

File: `src/pre_trade_risk_rules_assistant/ui/app.py`

```python
"""Streamlit demo: type a rule -> see JSON + plain-English read-back -> approve."""

import streamlit as st

from pre_trade_risk_rules_assistant import store
from pre_trade_risk_rules_assistant.graph import run_graph

st.set_page_config(page_title="RuleForge", page_icon="⚙️")
st.title("⚙️ RuleForge — Pre-Trade Risk Rules Assistant")
st.caption("Natural-language risk rule → validated, audit-logged JSON config.")

example = "Block any single buy order over 5 million SGD on SGX small-caps"
request = st.text_area("Describe the rule", value=example, height=100)

if st.button("Draft rule", type="primary"):
    with st.spinner("Running generate → validate → self-correct…"):
        state = run_graph(request)

    if state.get("status") == "ok":
        st.success(f"Validated after {state.get('attempts', 0)} self-correction(s).")
        st.subheader("Plain-English read-back")
        st.info(state.get("readback", ""))
        st.subheader("Generated config")
        st.json(state["validated_rule"])
        if st.button("✅ Approve & persist"):
            rule_id = store.save_rule(state["validated_rule"])
            st.success(f"Saved rule {rule_id} (audit-logged).")
    else:
        st.error("Escalated to human — could not produce a valid rule.")
        st.write("Diagnostic:")
        for err in state.get("errors", []):
            st.write(f"- {err}")
```

- [ ] **Step 2: Smoke-test it launches**

Run (requires a real `ANTHROPIC_API_KEY` in `.env`): `make serve-ui`
Expected: Streamlit serves at `http://localhost:8501`; entering the example rule shows JSON + read-back. (Manual check — stop with Ctrl-C.)

- [ ] **Step 3: Commit**

```bash
git add src/pre_trade_risk_rules_assistant/ui
git commit -m "feat(ruleforge): streamlit demo ui [<KEY>]"
```

---

## Task 14: Golden dataset + eval harness

**Files:**
- Create: `evals/golden_rules.json`, `evals/run_eval.py`
- Test: `tests/unit/test_eval.py` (covers scoring helpers, no network)

- [ ] **Step 1: Create the golden dataset (20 cases)**

File: `evals/golden_rules.json` — author **20** entries. Below are 6 representative entries covering all 3 rule types and the lint edge cases; **extend to 20** following the same shape (vary exchange, side, action, thresholds, and a few deliberately tricky phrasings). `expected` holds only the fields the scorer compares.

```json
[
  {
    "request": "Block any single buy order over 5 million SGD on SGX",
    "expected": {
      "rule_type": "order_notional_limit",
      "scope": {"exchange": "SGX"},
      "side": "buy",
      "action": "block",
      "max_notional": 5000000,
      "currency": "SGD"
    }
  },
  {
    "request": "Warn on sell orders larger than 2,000,000 HKD on HKEX",
    "expected": {
      "rule_type": "order_notional_limit",
      "scope": {"exchange": "HKEX"},
      "side": "sell",
      "action": "warn",
      "max_notional": 2000000,
      "currency": "HKD"
    }
  },
  {
    "request": "Apply a 5% price collar on all SGX orders against the last price",
    "expected": {
      "rule_type": "price_collar",
      "scope": {"exchange": "SGX"},
      "action": "warn",
      "collar_pct": 5,
      "reference_price": "last"
    }
  },
  {
    "request": "Put a 3 percent collar on ASX buy orders versus the opening price",
    "expected": {
      "rule_type": "price_collar",
      "scope": {"exchange": "ASX"},
      "side": "buy",
      "collar_pct": 3,
      "reference_price": "open"
    }
  },
  {
    "request": "Block trading in DBS (D05) and OCBC (O39) on SGX",
    "expected": {
      "rule_type": "restricted_list",
      "scope": {"exchange": "SGX"},
      "action": "block",
      "symbols": ["D05", "O39"]
    }
  },
  {
    "request": "Add CBA to the restricted list for ASX, block both sides",
    "expected": {
      "rule_type": "restricted_list",
      "scope": {"exchange": "ASX"},
      "action": "block",
      "symbols": ["CBA"]
    }
  }
]
```

- [ ] **Step 2: Write the failing test for scoring helpers**

File: `tests/unit/test_eval.py`

```python
"""Test eval scoring helpers (pure, no network)."""

from evals.run_eval import field_accuracy, schema_pass


def test_schema_pass_true_when_validated_rule_present():
    assert schema_pass({"status": "ok", "validated_rule": {"a": 1}}) is True
    assert schema_pass({"status": "escalated"}) is False


def test_field_accuracy_compares_nested_expected_fields():
    actual = {
        "rule_type": "order_notional_limit",
        "scope": {"exchange": "SGX", "symbols": []},
        "side": "buy",
        "max_notional": 5000000,
        "currency": "SGD",
    }
    expected = {
        "rule_type": "order_notional_limit",
        "scope": {"exchange": "SGX"},
        "side": "buy",
        "max_notional": 5000000,
        "currency": "SGD",
    }
    assert field_accuracy(actual, expected) == 1.0

    expected_wrong = dict(expected, currency="USD")
    assert field_accuracy(actual, expected_wrong) < 1.0
```

> The test imports `evals.run_eval`. Ensure `evals/` is importable from the project root in tests by adding an empty `evals/__init__.py`, OR rely on `rootdir` import (pytest adds the project root to `sys.path` via the `tests` rootdir). If the import fails, create `evals/__init__.py` with a one-line docstring.

- [ ] **Step 3: Run — expect failure**

Run: `pytest tests/unit/test_eval.py -v`
Expected: FAIL — `evals.run_eval` not importable.

- [ ] **Step 4: Implement `evals/run_eval.py`**

File: `evals/run_eval.py`

```python
"""Custom eval harness: schema-pass %, field accuracy, first-pass rate, intent fidelity.

Run: python evals/run_eval.py   (needs ANTHROPIC_API_KEY)
"""

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Make the src package importable when run as a script.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pre_trade_risk_rules_assistant.config import get_config  # noqa: E402
from pre_trade_risk_rules_assistant.graph import run_graph  # noqa: E402
from pre_trade_risk_rules_assistant.llm import call_text  # noqa: E402

EVALS_DIR = Path(__file__).parent
THRESHOLDS = {"schema_pass_rate": 0.90, "first_pass_field_accuracy": 0.80}


def schema_pass(state: dict[str, Any]) -> bool:
    return state.get("status") == "ok" and "validated_rule" in state


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def field_accuracy(actual: dict[str, Any], expected: dict[str, Any]) -> float:
    """Fraction of expected (flattened) fields that match the actual config."""
    exp = _flatten(expected)
    act = _flatten(actual)
    if not exp:
        return 1.0
    matches = sum(1 for k, v in exp.items() if act.get(k) == v)
    return matches / len(exp)


def intent_fidelity(request: str, readback: str) -> bool:
    """LLM-judge: does the read-back faithfully restate the request? (haiku)."""
    cfg = get_config()
    verdict = call_text(
        system=(
            "You are a strict QA judge. Answer only YES or NO: does the read-back faithfully "
            "capture the intent of the original rule request (same action, side, scope, threshold)?"
        ),
        user=f"REQUEST: {request}\nREAD-BACK: {readback}",
        model=cfg["models"]["judge"],
        max_tokens=10,
    )
    return verdict.strip().upper().startswith("YES")


def run_evaluation() -> dict[str, Any]:
    cases = json.loads((EVALS_DIR / "golden_rules.json").read_text())
    rows = []
    for case in cases:
        state = run_graph(case["request"])
        passed = schema_pass(state)
        acc = field_accuracy(state.get("validated_rule", {}), case["expected"]) if passed else 0.0
        fidelity = intent_fidelity(case["request"], state.get("readback", "")) if passed else False
        rows.append(
            {
                "request": case["request"],
                "schema_pass": passed,
                "field_accuracy": acc,
                "first_pass": passed and state.get("attempts", 0) == 0,
                "intent_fidelity": fidelity,
                "attempts": state.get("attempts", 0),
                "status": state.get("status"),
            }
        )

    n = len(rows)
    metrics = {
        "schema_pass_rate": sum(r["schema_pass"] for r in rows) / n,
        "first_pass_field_accuracy": (
            sum(r["field_accuracy"] for r in rows if r["first_pass"])
            / max(1, sum(r["first_pass"] for r in rows))
        ),
        "field_accuracy_overall": sum(r["field_accuracy"] for r in rows) / n,
        "intent_fidelity_rate": sum(r["intent_fidelity"] for r in rows) / n,
        "first_pass_rate": sum(r["first_pass"] for r in rows) / n,
    }
    passed_thresholds = (
        metrics["schema_pass_rate"] >= THRESHOLDS["schema_pass_rate"]
        and metrics["first_pass_field_accuracy"] >= THRESHOLDS["first_pass_field_accuracy"]
    )
    return {"metrics": metrics, "thresholds": THRESHOLDS, "passed": passed_thresholds, "rows": rows}


def main() -> None:
    print("RuleForge Eval Harness")
    print("=" * 60)
    results = run_evaluation()
    for k, v in results["metrics"].items():
        print(f"  {k:30s} {v:.2%}")
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

- [ ] **Step 5: Run the scoring-helper test — expect pass**

Run: `pytest tests/unit/test_eval.py -v`
Expected: 2 passed.

- [ ] **Step 6: Run the full eval (needs API key)**

Run: `make eval`
Expected: prints metrics; **schema_pass_rate ≥ 90%** and **first_pass_field_accuracy ≥ 80%** (the spec DoD). Writes `evals/results/eval_2026-06-14.json`. If below threshold, tune the generator system prompt (Task 7) and re-run.

- [ ] **Step 7: Commit**

```bash
git add evals/golden_rules.json evals/run_eval.py tests/unit/test_eval.py
git commit -m "feat(ruleforge): golden dataset + custom eval harness [<KEY>]"
```

> **Teacher's note — why not RAGAS?** P1 measured *retrieval* (faithfulness, context precision). Structured output needs different metrics: exact field match, schema-pass rate, and an LLM-judge for intent. There's no "retrieved context" to score — so you design the eval, you don't reuse one.

---

## Task 15: README + compliance caveat

**Files:**
- Create/replace: `README.md`

- [ ] **Step 1: Write the README**

File: `projects/pre-trade-risk-rules-assistant/README.md`

```markdown
# RuleForge — Pre-Trade Risk Rules Assistant

A LangGraph agent that converts natural-language pre-trade risk rules into
validated, schema-conformant JSON configs — checking its own output before a
human sees it.

## Key Features
- **Self-correcting agent** — generate → validate → retry (max 2) → escalate.
- **Deterministic risk gateway** — Pydantic v2 schemas + 5 domain lints.
- **Round-trip verification** — plain-English read-back of every rule.
- **Audit trail** — approved rules persist to SQLite + an append-only JSONL log.

## Tech Stack
| Component | Technology |
|---|---|
| LLM | Claude `claude-sonnet-4-6` via the Anthropic SDK (direct, tool use) |
| Agent | LangGraph (stateful graph + conditional edges) |
| Validation | Pydantic v2 + domain lints |
| API / UI | FastAPI · Streamlit |
| Storage | SQLite + JSONL audit log |
| Evals | Custom harness (schema-pass %, field accuracy, intent-fidelity judge) |

## Project Structure
\`\`\`
src/pre_trade_risk_rules_assistant/  graph.py, nodes/, schemas/, lints.py, store.py, api/, ui/
evals/                               golden_rules.json, run_eval.py
tests/                               unit/ + integration/
\`\`\`

## Setup
1. \`pip install -e ".[dev]"\`
2. \`cp .env.example .env\` and add \`ANTHROPIC_API_KEY\`

## Usage
- API: \`make serve-api\` → \`POST /rules/draft {"request": "..."}\`, \`GET /rules/{id}\`
- UI: \`make serve-ui\`
- Eval: \`make eval\`

## Eval Metrics (Definition of Done)
| Metric | Target |
|---|---|
| Schema pass rate | ≥ 90% |
| First-pass field accuracy | ≥ 80% |
| Valid after ≤ 2 corrections or escalated | 100% |

## ⚠️ Compliance Caveat
Demo uses **synthetic rules and mock reference data only**. A real deployment
touching production risk configs would require change-management sign-off,
four-eyes approval, and infosec review.

## License
MIT — built as part of an AI PM learning portfolio.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(ruleforge): readme + compliance caveat [<KEY>]"
```

---

## Task 16: Final verification

- [ ] **Step 1: Full CI passes**

Run: `make ci`
Expected: ruff clean, mypy clean, **all unit + integration tests pass**.

- [ ] **Step 2: Coverage check**

Run: `pytest tests/ --cov=pre_trade_risk_rules_assistant --cov-report=term-missing`
Expected: high coverage on `schemas`, `lints`, `store`, `validator`, `corrector`, `graph` (LLM I/O lines in `llm.py`/nodes are exercised via mocks).

- [ ] **Step 3: Confirm Definition of Done against the spec**

Verify each holds and record evidence:
- [ ] 3 rule types implemented (`order_notional_limit`, `price_collar`, `restricted_list`) — Task 2.
- [ ] LangGraph loop end-to-end with max-2 retry + escalation — Task 11 (3-path integration test green).
- [ ] Pydantic schemas + 5 domain lints — Tasks 2, 3.
- [ ] 20-case golden set; eval reports schema-pass-rate + field accuracy — Task 14.
- [ ] Streamlit demo: type rule → JSON + read-back → approve — Task 13.
- [ ] `make eval` shows **schema_pass_rate ≥ 0.90** and **first_pass_field_accuracy ≥ 0.80**.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feat/<KEY>-ruleforge-mvp
gh pr create --title "feat(ruleforge): Pre-Trade Risk Rules Assistant MVP [<KEY>]" \
  --body "Implements RuleForge MVP per docs/pre-trade-risk-rules-assistant-spec.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 5: After merge — close out**

Confirm the branch is merged into `main` before marking the Linear issue **Done** (per the `verify-merged-before-linear-done` memory).

---

## Stretch Goals (future phase — NOT in this plan)

Each is a follow-up Linear issue + branch once the MVP is merged:
1. **ADV % + position-limit rule types** → add a `market_data` tool-calling LangGraph node that queries a mock reference-data table; new schemas + lints.
2. **Rule conflict detection** → a node that diffs the new rule against `store` for duplicates/contradictions before persist.
3. **MCP server wrapper** → expose `draft_rule` / `validate_rule` as MCP tools (bridges to Project 3).
4. **LangSmith tracing** of graph runs.

---

## Self-Review (performed against the spec)

- **Spec coverage:** Problem/solution ✓ (NL→validated JSON via self-correct loop). Architecture nodes (parse/generate/validate/self-correct/explain) ✓ Tasks 6–11. Tech stack: Anthropic SDK direct ✓, LangGraph ✓, Pydantic v2 ✓, FastAPI ✓ T12, SQLite + JSON audit ✓ T4, Streamlit ✓ T13, custom evals ✓ T14. MVP scope (3 rule types, loop, 5 lints, 20-case eval, demo, DoD thresholds) ✓. Stretch goals deferred per the user's "MVP only" choice.
- **Placeholder scan:** Golden dataset gives 6 concrete entries + explicit "extend to 20 following this shape" — the one allowed repeated-data pattern; all code steps contain complete code.
- **Type consistency:** `RuleType` values, `RuleForgeState` keys, and the `call_tool(...) -> dict` / `call_text(...) -> str` signatures are used identically across Tasks 5–14. Router labels `{"explain","correct","escalate"}` match the `add_conditional_edges` mapping in T11. `RuleAdapter` (T2) is the single validation entry point used by T8 and the lint tests.
