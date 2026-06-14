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
        "arrival_px": round(row["arrival_px"], 4)
        if n and row.get("arrival_px") is not None
        else None,
        "vwap": round(row["vwap"], 4) if n and row.get("vwap") is not None else None,
        "close_px": round(row["close_px"], 4) if n and row.get("close_px") is not None else None,
        "caveat": "arrival is the average order-arrival price; vwap is notional-weighted; close is the day's close.",
    }
