"""Custom eval harness: schema-pass %, field accuracy, first-pass rate, intent fidelity.

Run: python evals/run_eval.py   (needs ANTHROPIC_API_KEY)
"""

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Make the src package importable when run as a script.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pre_trade_risk_rules_assistant.config import get_config  # noqa: E402
from pre_trade_risk_rules_assistant.graph import run_graph  # noqa: E402
from pre_trade_risk_rules_assistant.llm import call_text  # noqa: E402

EVALS_DIR = Path(__file__).parent
THRESHOLDS = {"schema_pass_rate": 0.90, "first_pass_field_accuracy": 0.80}


def schema_pass(state: dict[str, Any]) -> bool:
    """Return True if the run produced a validated rule."""
    return state.get("status") == "ok" and "validated_rule" in state


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dotted keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def field_accuracy(actual: dict[str, Any], expected: dict[str, Any]) -> float:
    """Fraction of expected (flattened) fields that match the actual config."""
    exp = _flatten(expected)
    act = _flatten(actual)
    if not exp:
        return 1.0
    matches = sum(1 for k, v in exp.items() if act.get(k) == v)
    return matches / len(exp)


def intent_fidelity(request: str, readback: str) -> bool:
    """LLM-judge: does the read-back faithfully restate the request? (haiku)."""
    cfg = get_config()
    verdict = call_text(
        system=(
            "You are a strict QA judge. Answer only YES or NO: does the read-back faithfully "
            "capture the intent of the original rule request (same action, side, scope, threshold)?"
        ),
        user=f"REQUEST: {request}\nREAD-BACK: {readback}",
        model=cfg["models"]["judge"],
        max_tokens=10,
    )
    return verdict.strip().upper().startswith("YES")


def run_evaluation() -> dict[str, Any]:
    """Run the agent over the golden set and compute aggregate metrics."""
    cases = json.loads((EVALS_DIR / "golden_rules.json").read_text())
    rows = []
    for case in cases:
        state = run_graph(case["request"])
        passed = schema_pass(state)
        acc = field_accuracy(state.get("validated_rule", {}), case["expected"]) if passed else 0.0
        fidelity = intent_fidelity(case["request"], state.get("readback", "")) if passed else False
        rows.append(
            {
                "request": case["request"],
                "schema_pass": passed,
                "field_accuracy": acc,
                "first_pass": passed and state.get("attempts", 0) == 0,
                "intent_fidelity": fidelity,
                "attempts": state.get("attempts", 0),
                "status": state.get("status"),
            }
        )

    n = len(rows)
    metrics = {
        "schema_pass_rate": sum(r["schema_pass"] for r in rows) / n,
        "first_pass_field_accuracy": (
            sum(r["field_accuracy"] for r in rows if r["first_pass"])
            / max(1, sum(r["first_pass"] for r in rows))
        ),
        "field_accuracy_overall": sum(r["field_accuracy"] for r in rows) / n,
        "intent_fidelity_rate": sum(r["intent_fidelity"] for r in rows) / n,
        "first_pass_rate": sum(r["first_pass"] for r in rows) / n,
    }
    passed_thresholds = (
        metrics["schema_pass_rate"] >= THRESHOLDS["schema_pass_rate"]
        and metrics["first_pass_field_accuracy"] >= THRESHOLDS["first_pass_field_accuracy"]
    )
    return {"metrics": metrics, "thresholds": THRESHOLDS, "passed": passed_thresholds, "rows": rows}


def main() -> None:
    """CLI entry: run the eval, print metrics, save results JSON, exit nonzero on fail."""
    print("RuleForge Eval Harness")
    print("=" * 60)
    results = run_evaluation()
    for k, v in results["metrics"].items():
        print(f"  {k:30s} {v:.2%}")
    print(f"  PASSED: {results['passed']}")

    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"eval_{date.today().isoformat()}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
