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
