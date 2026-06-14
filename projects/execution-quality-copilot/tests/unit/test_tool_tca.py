"""Test compute_slippage and top_outliers pure functions, including the sign convention."""

import pytest

from execution_quality_copilot.server.tools import tca


def test_compute_slippage_shape_and_sign(fills_conn):
    out = tca.compute_slippage(fills_conn, broker="DELTA", benchmark="arrival")
    assert out["benchmark"] == "arrival"
    assert out["n_fills"] > 0
    assert out["total_notional"] > 0
    # DELTA carries the worst injected bias → positive cost vs arrival.
    assert out["slippage_bps"] > 0


def test_compute_slippage_delta_worse_than_alpha(fills_conn):
    delta = tca.compute_slippage(fills_conn, broker="DELTA")["slippage_bps"]
    alpha = tca.compute_slippage(fills_conn, broker="ALPHA")["slippage_bps"]
    assert delta > alpha


def test_compute_slippage_rejects_unknown_benchmark(fills_conn):
    with pytest.raises(ValueError, match="unknown benchmark"):
        tca.compute_slippage(fills_conn, benchmark="midpoint")


def test_top_outliers_sorted_desc_and_capped(fills_conn):
    out = tca.top_outliers(fills_conn, n=5)
    slips = [r["slippage_bps"] for r in out["outliers"]]
    assert slips == sorted(slips, reverse=True)
    assert len(out["outliers"]) == 5
    assert out["worst_slippage_bps"] == slips[0]


def test_top_outliers_n_clamped(fills_conn):
    out = tca.top_outliers(fills_conn, n=9999)
    assert len(out["outliers"]) <= 50
