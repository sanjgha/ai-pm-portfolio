"""TCA tools: compute_slippage (bps vs benchmark) and top_outliers (worst executions)."""

from typing import Any

import duckdb

from execution_quality_copilot.db import BENCHMARK_COLUMNS, _serialize, query
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
