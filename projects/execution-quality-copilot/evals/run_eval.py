"""Tool-use eval: scores tool selection, parameter accuracy, and numeric-answer accuracy.

Run: python evals/run_eval.py   (needs ANTHROPIC_API_KEY and a built seed DB — `make gen-data`)
"""

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import duckdb  # noqa: E402

from execution_quality_copilot.config import get_config  # noqa: E402
from execution_quality_copilot.llm import get_client, run_tool_loop  # noqa: E402
from execution_quality_copilot.server.tools import benchmarks, fills, tca  # noqa: E402

EVALS_DIR = Path(__file__).parent
THRESHOLDS = {"tool_selection_rate": 0.90, "numeric_accuracy_rate": 0.80}

# Anthropic-format tool definitions (mirror the FastMCP tool signatures in server/main.py).
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "get_fills",
        "description": "Return raw fills filtered by symbol/broker/algo/venue/side/tier/date range; row-capped.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "broker": {"type": "string", "enum": ["ALPHA", "BRAVO", "COBALT", "DELTA"]},
                "algo": {"type": "string", "enum": ["VWAP", "TWAP", "IS"]},
                "venue": {"type": "string", "enum": ["XNAS", "XNYS", "BATS", "EDGX", "DARK"]},
                "side": {"type": "string", "enum": ["BUY", "SELL"]},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "mkt_cap_tier": {"type": "string", "enum": ["large", "mid", "small"]},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "get_benchmarks",
        "description": "Arrival/VWAP/close reference prices for one symbol on one ISO date.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "date": {"type": "string"}},
            "required": ["symbol", "date"],
        },
    },
    {
        "name": "compute_slippage",
        "description": "Notional-weighted slippage (bps) vs arrival/vwap/close over a filtered fill set.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "broker": {"type": "string", "enum": ["ALPHA", "BRAVO", "COBALT", "DELTA"]},
                "algo": {"type": "string", "enum": ["VWAP", "TWAP", "IS"]},
                "side": {"type": "string", "enum": ["BUY", "SELL"]},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "benchmark": {"type": "string", "enum": ["arrival", "vwap", "close"]},
                "mkt_cap_tier": {"type": "string", "enum": ["large", "mid", "small"]},
            },
        },
    },
    {
        "name": "venue_breakdown",
        "description": "Fills grouped by venue and algo: count, notional, avg slippage (bps).",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "broker": {"type": "string", "enum": ["ALPHA", "BRAVO", "COBALT", "DELTA"]},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "benchmark": {"type": "string", "enum": ["arrival", "vwap", "close"]},
            },
        },
    },
    {
        "name": "top_outliers",
        "description": "The n single worst fills by slippage (bps) vs a benchmark.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "benchmark": {"type": "string", "enum": ["arrival", "vwap", "close"]},
                "n": {"type": "integer"},
                "mkt_cap_tier": {"type": "string", "enum": ["large", "mid", "small"]},
                "broker": {"type": "string", "enum": ["ALPHA", "BRAVO", "COBALT", "DELTA"]},
            },
        },
    },
]

_TABLE = {
    "get_fills": fills.get_fills,
    "venue_breakdown": fills.venue_breakdown,
    "get_benchmarks": benchmarks.get_benchmarks,
    "compute_slippage": tca.compute_slippage,
    "top_outliers": tca.top_outliers,
}


def make_dispatch(conn: duckdb.DuckDBPyConnection):
    """Return a dispatch(name, args) that runs a pure tool function against the connection."""

    def dispatch(name: str, args: dict[str, Any]) -> Any:
        return _TABLE[name](conn, **args)

    return dispatch


# ---- scoring ----------------------------------------------------------------


def tool_selected(expected_tool: str, called: list[tuple[str, dict[str, Any]]]) -> bool:
    """True if the expected tool was called at least once."""
    return any(name == expected_tool for name, _ in called)


def params_matched(
    expected_tool: str, expected_params: dict[str, Any], called: list[tuple[str, dict[str, Any]]]
) -> bool:
    """True if some call to the expected tool included all expected params (subset match)."""
    for name, args in called:
        if name == expected_tool and all(args.get(k) == v for k, v in expected_params.items()):
            return True
    return False


def parse_answer(text: str) -> float | None:
    """Extract the number from the 'ANSWER: <number>' line, or None."""
    m = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def numeric_ok(parsed: float | None, ground_truth: float, *, tol: float) -> bool:
    """True if parsed is within relative tolerance of ground_truth."""
    if parsed is None or ground_truth is None:
        return False
    denom = abs(ground_truth) if ground_truth else 1.0
    return abs(parsed - ground_truth) / denom <= tol


# ---- runner -----------------------------------------------------------------


def run_evaluation() -> dict[str, Any]:
    """Run the agent over the golden questions and compute aggregate metrics."""
    cfg = get_config()
    tol = cfg["eval"]["numeric_tolerance"]
    max_turns = cfg["eval"]["max_turns"]
    model = cfg["models"]["agent"]

    db_path = ROOT / cfg["storage"]["db_path"]
    conn = duckdb.connect(str(db_path), read_only=True)
    dispatch = make_dispatch(conn)
    client = get_client()

    cases = json.loads((EVALS_DIR / "golden_questions.json").read_text())
    rows = []
    for case in cases:
        text, called = run_tool_loop(
            client, model, TOOL_DEFS, dispatch, case["question"], max_turns=max_turns
        )
        sel = tool_selected(case["expected_tool"], called)
        par = params_matched(case["expected_tool"], case["expected_params"], called)

        num_ok: bool | None = None
        if case["answer_key"] is not None:
            ground = _TABLE[case["expected_tool"]](conn, **case["expected_params"])[
                case["answer_key"]
            ]
            num_ok = numeric_ok(parse_answer(text), ground, tol=tol)

        rows.append(
            {
                "question": case["question"],
                "expected_tool": case["expected_tool"],
                "tool_selected": sel,
                "params_matched": par,
                "numeric_ok": num_ok,
                "tools_called": [n for n, _ in called],
            }
        )

    n = len(rows)
    numeric_rows = [r for r in rows if r["numeric_ok"] is not None]
    metrics = {
        "tool_selection_rate": sum(r["tool_selected"] for r in rows) / n,
        "param_match_rate": sum(r["params_matched"] for r in rows) / n,
        "numeric_accuracy_rate": (
            sum(bool(r["numeric_ok"]) for r in numeric_rows) / len(numeric_rows)
            if numeric_rows
            else 0.0
        ),
    }
    passed = (
        metrics["tool_selection_rate"] >= THRESHOLDS["tool_selection_rate"]
        and metrics["numeric_accuracy_rate"] >= THRESHOLDS["numeric_accuracy_rate"]
    )
    conn.close()
    return {"metrics": metrics, "thresholds": THRESHOLDS, "passed": passed, "rows": rows}


def main() -> None:
    """Run the eval, print metrics, save results JSON, exit nonzero on fail."""
    print("Execution Quality Copilot — Tool-Use Eval")
    print("=" * 60)
    results = run_evaluation()
    for k, v in results["metrics"].items():
        print(f"  {k:24s} {v:.2%}")
    print(f"  PASSED: {results['passed']}")

    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"eval_{date.today().isoformat()}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
