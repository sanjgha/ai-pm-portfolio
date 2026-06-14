"""Deterministic synthetic FIX ExecutionReport (35=8) generator → DuckDB.

Embeds a systematic slippage bias per broker/algo plus a small-cap penalty so the
golden eval questions have stable, hand-checkable answers. Run with:

    python -m execution_quality_copilot.datagen
"""

import hashlib
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb

from execution_quality_copilot import models
from execution_quality_copilot.models import Fill

# Systematic slippage (bps) injected by broker and algo; higher = worse execution.
BROKER_BIAS = {"ALPHA": 1.0, "BRAVO": 2.5, "COBALT": 4.0, "DELTA": 6.0}
ALGO_BIAS = {"VWAP": 1.0, "TWAP": 2.0, "IS": 3.5}
TIER_PENALTY = {"large": 0.0, "mid": 1.5, "small": 4.0}


def _tier_for_index(i: int) -> str:
    """Assign a market-cap tier: first 15 large, next 20 mid, rest small."""
    if i < 15:
        return "large"
    if i < 35:
        return "mid"
    return "small"


def _business_days(start: str, end: str) -> list[date]:
    """Inclusive list of weekdays between two ISO dates."""
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    days = []
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _stable_hash(symbol: str, d: date) -> int:
    """Process-stable hash for a (symbol, day) pair (Python's built-in hash() is salted)."""
    digest = hashlib.sha256(f"{symbol}|{d.isoformat()}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def build_fills(
    *,
    seed: int,
    n_fills: int,
    n_symbols: int,
    start_date: str,
    end_date: str,
) -> list[Fill]:
    """Build a deterministic list of synthetic fills for the given parameters."""
    rng = random.Random(seed)
    days = _business_days(start_date, end_date)

    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    tiers = {s: _tier_for_index(i) for i, s in enumerate(symbols)}
    base_px = {s: round(5.0 + (i % 40) * 2.5, 2) for i, s in enumerate(symbols)}
    # Deterministic per (symbol, day) close price, independent of the fill loop.
    close_px = {
        (s, d): round(base_px[s] * (1.0 + ((_stable_hash(s, d) % 401) - 200) / 10000.0), 4)
        for s in symbols
        for d in days
    }

    fills: list[Fill] = []
    for n in range(n_fills):
        symbol = rng.choice(symbols)
        tier = tiers[symbol]
        side = rng.choice(models.SIDES)
        broker = rng.choice(models.BROKERS)
        algo = rng.choice(models.ALGOS)
        venue = rng.choice(models.VENUES)
        d = rng.choice(days)

        arrival = round(base_px[symbol] * (1.0 + rng.uniform(-0.004, 0.004)), 4)
        expected_slip = (
            BROKER_BIAS[broker]
            + ALGO_BIAS[algo]
            + TIER_PENALTY[tier]
            + rng.gauss(0.0, 2.0)  # noise; bias still dominates in aggregate
        )
        sign = 1 if side == "BUY" else -1
        last_px = round(arrival * (1.0 + sign * expected_slip / 10000.0), 4)
        interval_vwap = round(arrival * (1.0 + rng.uniform(-0.002, 0.002)), 4)
        qty = rng.choice([100, 200, 500, 1000, 2500, 5000])
        ts = datetime.combine(d, datetime.min.time()) + timedelta(
            hours=9, minutes=rng.randint(30, 390)
        )

        fills.append(
            Fill(
                exec_id=f"E{n:06d}",
                order_id=f"O{n // 3:06d}",
                symbol=symbol,
                mkt_cap_tier=tier,
                side=side,
                broker=broker,
                algo=algo,
                venue=venue,
                transact_time=ts,
                trade_date=d,
                last_qty=qty,
                last_px=last_px,
                arrival_px=arrival,
                interval_vwap=interval_vwap,
                close_px=close_px[(symbol, d)],
                currency="USD",
            )
        )
    return fills


def write_duckdb(fills: list[Fill], db_path: Path) -> None:
    """Create (overwrite) the fills table and bulk-insert the rows."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE fills (
                exec_id VARCHAR PRIMARY KEY,
                order_id VARCHAR,
                symbol VARCHAR,
                mkt_cap_tier VARCHAR,
                side VARCHAR,
                broker VARCHAR,
                algo VARCHAR,
                venue VARCHAR,
                transact_time TIMESTAMP,
                trade_date DATE,
                last_qty INTEGER,
                last_px DOUBLE,
                arrival_px DOUBLE,
                interval_vwap DOUBLE,
                close_px DOUBLE,
                currency VARCHAR
            )
            """
        )
        conn.executemany(
            f"INSERT INTO fills ({', '.join(models.FILL_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in models.FILL_COLUMNS)})",
            [f.to_row() for f in fills],
        )
    finally:
        conn.close()
