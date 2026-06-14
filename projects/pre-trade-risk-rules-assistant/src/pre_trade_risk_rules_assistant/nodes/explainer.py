"""Read-back explainer node: plain-English confirmation of the validated rule."""

import json
from typing import Any

from pre_trade_risk_rules_assistant.llm import call_text

_EXPLAINER_SYSTEM = (
    "You are a pre-trade risk analyst. Given a validated rule config as JSON, write a single "
    "plain-English sentence a trading desk BA can confirm. State the action, side, scope, and "
    "threshold. No JSON, no preamble."
)


def explain_rule(state: dict[str, Any]) -> dict[str, Any]:
    """Produce a plain-English read-back of the validated rule."""
    readback = call_text(
        system=_EXPLAINER_SYSTEM,
        user=json.dumps(state["validated_rule"]),
    )
    return {"status": "ok", "readback": readback.strip()}
