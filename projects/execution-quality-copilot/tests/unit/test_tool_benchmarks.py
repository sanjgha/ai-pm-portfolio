"""Test the get_benchmarks pure function."""

from execution_quality_copilot.server.tools import benchmarks


def test_get_benchmarks_returns_three_reference_prices(fills_conn):
    # Pick a (symbol, date) that exists in the fixture.
    row = fills_conn.execute("SELECT symbol, trade_date FROM fills LIMIT 1").fetchone()
    symbol, trade_date = row[0], row[1].isoformat()
    out = benchmarks.get_benchmarks(fills_conn, symbol=symbol, date=trade_date)
    assert out["symbol"] == symbol
    assert out["date"] == trade_date
    assert out["n_fills"] >= 1
    assert out["arrival_px"] > 0
    assert out["close_px"] > 0
    assert out["vwap"] > 0


def test_get_benchmarks_empty_when_no_match(fills_conn):
    out = benchmarks.get_benchmarks(fills_conn, symbol="NOPE", date="2026-05-01")
    assert out["n_fills"] == 0
    assert out["close_px"] is None
