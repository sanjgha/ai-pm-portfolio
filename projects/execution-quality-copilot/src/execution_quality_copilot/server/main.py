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
