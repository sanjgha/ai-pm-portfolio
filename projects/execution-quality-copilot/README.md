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
