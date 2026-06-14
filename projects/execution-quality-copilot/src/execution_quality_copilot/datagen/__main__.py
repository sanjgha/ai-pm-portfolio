"""CLI: regenerate the seed DuckDB from config.yaml (`python -m execution_quality_copilot.datagen`)."""

from pathlib import Path

from execution_quality_copilot.config import get_config
from execution_quality_copilot.datagen.generate import build_fills, write_duckdb

ROOT = Path(__file__).parent.parent.parent.parent


def main() -> None:
    """Build the synthetic dataset and write it to the configured db_path."""
    cfg = get_config()
    g = cfg["generator"]
    fills = build_fills(
        seed=g["seed"],
        n_fills=g["n_fills"],
        n_symbols=g["n_symbols"],
        start_date=g["start_date"],
        end_date=g["end_date"],
    )
    db_path = ROOT / cfg["storage"]["db_path"]
    write_duckdb(fills, db_path)
    print(f"Wrote {len(fills)} fills to {db_path}")


if __name__ == "__main__":
    main()
