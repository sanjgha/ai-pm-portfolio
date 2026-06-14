"""Self-correction control: bump attempts, escalate, and route after validation."""

from typing import Any

from pre_trade_risk_rules_assistant.config import get_config


def self_correct(state: dict[str, Any]) -> dict[str, Any]:
    """Increment the correction counter; errors are already in state for the generator."""
    return {"attempts": state.get("attempts", 0) + 1}


def escalate(state: dict[str, Any]) -> dict[str, Any]:
    """Terminal failure: surface a diagnostic for the human."""
    return {
        "status": "escalated",
        "escalation_reason": "; ".join(state.get("errors", [])),
    }


def route_after_validation(state: dict[str, Any]) -> str:
    """Conditional edge: explain (clean) | correct (retry) | escalate (budget exhausted)."""
    if not state.get("errors"):
        return "explain"
    max_attempts = get_config()["agent"]["max_correction_attempts"]
    if state.get("attempts", 0) >= max_attempts:
        return "escalate"
    return "correct"
