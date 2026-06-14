"""Test the DuckDB helpers: bounded query, serialization, benchmark whitelist."""

from datetime import date, datetime

import duckdb
import pytest

from execution_quality_copilot import db


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE t (id INTEGER, d DATE, px DOUBLE)")
    c.executemany(
        "INSERT INTO t VALUES (?, ?, ?)",
        [(i, date(2026, 5, 1), 1.23456 + i) for i in range(10)],
    )
    return c


def test_query_returns_dicts(conn):
    rows = db.query(conn, "SELECT id, px FROM t ORDER BY id")
    assert rows[0] == {"id": 0, "px": 1.23456}
    assert len(rows) == 10


def test_query_respects_max_rows(conn):
    rows = db.query(conn, "SELECT id FROM t ORDER BY id", max_rows=3)
    assert len(rows) == 3


def test_query_parameterised(conn):
    rows = db.query(conn, "SELECT id FROM t WHERE id = ?", [4])
    assert rows == [{"id": 4}]


def test_serialize_dates_and_rounds_floats():
    out = db._serialize(
        [{"d": date(2026, 5, 1), "ts": datetime(2026, 5, 1, 9, 30), "px": 1.234567}]
    )
    assert out[0]["d"] == "2026-05-01"
    assert out[0]["ts"] == "2026-05-01T09:30:00"
    assert out[0]["px"] == 1.2346


def test_benchmark_columns_whitelist():
    assert db.BENCHMARK_COLUMNS["arrival"] == "arrival_px"
    assert db.BENCHMARK_COLUMNS["vwap"] == "interval_vwap"
    assert db.BENCHMARK_COLUMNS["close"] == "close_px"
    assert "px; DROP TABLE" not in db.BENCHMARK_COLUMNS
