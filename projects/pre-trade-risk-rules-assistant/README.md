# RuleForge — Pre-Trade Risk Rules Assistant

A LangGraph agent that converts natural-language pre-trade risk rules into validated, schema-conformant JSON configs — checking its own output before a human sees it.

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
```
src/pre_trade_risk_rules_assistant/  graph.py, nodes/, schemas/, lints.py, store.py, api/, ui/
evals/                               golden_rules.json, run_eval.py
tests/                               unit/ + integration/
```

## Setup
1. `pip install -e ".[dev]"`
2. `cp .env.example .env` and add `ANTHROPIC_API_KEY`

## Usage
- API: `make serve-api` → `POST /rules/draft {"request": "..."}`, `GET /rules/{id}`
- UI: `make serve-ui`
- Eval: `make eval`

## Eval Metrics (Definition of Done)
| Metric | Target |
|---|---|
| Schema pass rate | ≥ 90% |
| First-pass field accuracy | ≥ 80% |
| Valid after ≤ 2 corrections or escalated | 100% |

## ⚠️ Compliance Caveat
Demo uses **synthetic rules and mock reference data only**. A real deployment touching production risk configs would require change-management sign-off, four-eyes approval, and infosec review.

## License
MIT — built as part of an AI PM learning portfolio.
