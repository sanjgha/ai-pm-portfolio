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
    fills = build_fills(
        seed=42, n_fills=2000, n_symbols=50, start_date="2026-05-01", end_date="2026-05-29"
    )
    conn.executemany(
        f"INSERT INTO fills ({', '.join(models.FILL_COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in models.FILL_COLUMNS)})",
        [f.to_row() for f in fills],
    )
    return conn
