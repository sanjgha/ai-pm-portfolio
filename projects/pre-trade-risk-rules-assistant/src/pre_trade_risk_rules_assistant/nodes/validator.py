"""Validator node: deterministic Pydantic validation + domain lints (the risk gateway)."""

from typing import Any

from pydantic import ValidationError

from pre_trade_risk_rules_assistant.lints import run_lints
from pre_trade_risk_rules_assistant.schemas.rules import RuleAdapter


def validate_config(state: dict[str, Any]) -> dict[str, Any]:
    """Validate the draft against the schema then the domain lints."""
    draft = state["draft"]
    try:
        rule = RuleAdapter.validate_python(draft)
    except ValidationError as exc:
        errors = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()]
        return {"errors": errors}

    lint_errors = run_lints(rule)
    if lint_errors:
        return {"errors": lint_errors}

    return {"errors": [], "validated_rule": rule.model_dump(mode="json")}
