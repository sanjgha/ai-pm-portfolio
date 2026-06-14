"""Test the domain enums and the Fill dataclass."""

from execution_quality_copilot import models


def test_enum_membership():
    assert models.BROKERS == ["ALPHA", "BRAVO", "COBALT", "DELTA"]
    assert models.ALGOS == ["VWAP", "TWAP", "IS"]
    assert set(models.SIDES) == {"BUY", "SELL"}
    assert models.TIERS == ["large", "mid", "small"]
    assert models.BENCHMARKS == ["arrival", "vwap", "close"]
    assert "DARK" in models.VENUES


def test_fill_to_row_tuple_order_matches_columns():
    from datetime import date, datetime

    fill = models.Fill(
        exec_id="E1",
        order_id="O1",
        symbol="SYM00",
        mkt_cap_tier="large",
        side="BUY",
        broker="ALPHA",
        algo="VWAP",
        venue="XNAS",
        transact_time=datetime(2026, 5, 1, 10, 0, 0),
        trade_date=date(2026, 5, 1),
        last_qty=100,
        last_px=10.5,
        arrival_px=10.4,
        interval_vwap=10.45,
        close_px=10.6,
        currency="USD",
    )
    row = fill.to_row()
    assert row[0] == "E1"
    assert len(row) == len(models.FILL_COLUMNS)
    assert row[models.FILL_COLUMNS.index("last_px")] == 10.5
