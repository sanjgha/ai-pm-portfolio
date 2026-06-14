"""Domain enums and the FIX ExecutionReport (35=8) Fill record."""

from dataclasses import dataclass
from datetime import date, datetime

SIDES: list[str] = ["BUY", "SELL"]
BROKERS: list[str] = ["ALPHA", "BRAVO", "COBALT", "DELTA"]
ALGOS: list[str] = ["VWAP", "TWAP", "IS"]
VENUES: list[str] = ["XNAS", "XNYS", "BATS", "EDGX", "DARK"]
TIERS: list[str] = ["large", "mid", "small"]
BENCHMARKS: list[str] = ["arrival", "vwap", "close"]

FILL_COLUMNS: list[str] = [
    "exec_id",
    "order_id",
    "symbol",
    "mkt_cap_tier",
    "side",
    "broker",
    "algo",
    "venue",
    "transact_time",
    "trade_date",
    "last_qty",
    "last_px",
    "arrival_px",
    "interval_vwap",
    "close_px",
    "currency",
]


@dataclass
class Fill:
    """One synthetic FIX ExecutionReport (a single fill)."""

    exec_id: str
    order_id: str
    symbol: str
    mkt_cap_tier: str
    side: str
    broker: str
    algo: str
    venue: str
    transact_time: datetime
    trade_date: date
    last_qty: int
    last_px: float
    arrival_px: float
    interval_vwap: float
    close_px: float
    currency: str

    def to_row(self) -> tuple:
        """Return field values in FILL_COLUMNS order for a DuckDB INSERT."""
        return tuple(getattr(self, c) for c in FILL_COLUMNS)
