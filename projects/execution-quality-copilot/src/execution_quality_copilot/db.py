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
