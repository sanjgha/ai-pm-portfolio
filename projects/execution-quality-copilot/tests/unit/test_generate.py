"""Test the deterministic synthetic fill generator."""

from execution_quality_copilot import models
from execution_quality_copilot.datagen import generate


def test_build_fills_is_deterministic():
    a = generate.build_fills(
        seed=42, n_fills=500, n_symbols=10, start_date="2026-05-01", end_date="2026-05-08"
    )
    b = generate.build_fills(
        seed=42, n_fills=500, n_symbols=10, start_date="2026-05-01", end_date="2026-05-08"
    )
    assert len(a) == 500
    assert [f.exec_id for f in a] == [f.exec_id for f in b]
    assert a[0].last_px == b[0].last_px


def test_fill_fields_are_valid():
    fills = generate.build_fills(
        seed=1, n_fills=200, n_symbols=8, start_date="2026-05-01", end_date="2026-05-08"
    )
    for f in fills:
        assert f.side in models.SIDES
        assert f.broker in models.BROKERS
        assert f.algo in models.ALGOS
        assert f.venue in models.VENUES
        assert f.mkt_cap_tier in models.TIERS
        assert f.last_qty > 0
        assert f.last_px > 0
        assert f.currency == "USD"


def test_write_duckdb_roundtrips_row_count(tmp_path):
    db = tmp_path / "seed.duckdb"
    fills = generate.build_fills(
        seed=42, n_fills=300, n_symbols=10, start_date="2026-05-01", end_date="2026-05-08"
    )
    generate.write_duckdb(fills, db)
    import duckdb

    conn = duckdb.connect(str(db), read_only=True)
    count = conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    conn.close()
    assert count == 300


def test_delta_is_smallcap_is_worse_than_alpha_vwap_largecap(tmp_path):
    # The embedded bias must make DELTA+IS+small materially worse than ALPHA+VWAP+large.
    fills = generate.build_fills(
        seed=42, n_fills=8000, n_symbols=50, start_date="2026-05-01", end_date="2026-05-29"
    )

    def avg_slip(broker, algo, tier):
        rows = [
            f for f in fills if f.broker == broker and f.algo == algo and f.mkt_cap_tier == tier
        ]
        vals = []
        for f in rows:
            sign = 1 if f.side == "BUY" else -1
            vals.append(sign * (f.last_px - f.arrival_px) / f.arrival_px * 10000)
        return sum(vals) / len(vals)

    assert avg_slip("DELTA", "IS", "small") > avg_slip("ALPHA", "VWAP", "large")
