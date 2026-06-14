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
