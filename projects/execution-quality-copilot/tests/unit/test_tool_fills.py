"""Test get_fills and venue_breakdown pure functions."""

from execution_quality_copilot.server.tools import fills


def test_get_fills_filters_and_caps(fills_conn):
    out = fills.get_fills(fills_conn, broker="DELTA", side="BUY", limit=5)
    assert out["n_returned"] <= 5
    assert out["row_cap"] == 5
    for row in out["fills"]:
        assert row["broker"] == "DELTA"
        assert row["side"] == "BUY"


def test_get_fills_limit_is_clamped(fills_conn):
    out = fills.get_fills(fills_conn, limit=99999)
    assert out["row_cap"] == 500  # MAX_ROWS


def test_get_fills_date_range(fills_conn):
    out = fills.get_fills(fills_conn, start_date="2026-05-01", end_date="2026-05-01", limit=500)
    for row in out["fills"]:
        assert row["trade_date"] == "2026-05-01"


def test_venue_breakdown_groups_by_venue_and_algo(fills_conn):
    out = fills.venue_breakdown(fills_conn, broker="DELTA")
    keys = {(r["venue"], r["algo"]) for r in out["breakdown"]}
    assert ("DARK", "IS") in keys or len(keys) > 0
    for r in out["breakdown"]:
        assert "slippage_bps" in r and "n_fills" in r
