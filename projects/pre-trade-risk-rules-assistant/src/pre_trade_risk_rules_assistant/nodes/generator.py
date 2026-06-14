"""Config generator node: emit a JSON rule config conforming to the rule_type's schema."""

from typing import Any

from pre_trade_risk_rules_assistant.llm import call_tool
from pre_trade_risk_rules_assistant.schemas.rules import (
    OrderNotionalLimitRule,
    PriceCollarRule,
    RestrictedListRule,
)

RULE_MODELS = {
    "order_notional_limit": OrderNotionalLimitRule,
    "price_collar": PriceCollarRule,
    "restricted_list": RestrictedListRule,
}

_GENERATOR_SYSTEM = (
    "You are a pre-trade risk engineer. Emit a single rule config that exactly matches the "
    "provided tool schema. Use the correct currency for the exchange (SGX=SGD, HKEX=HKD, "
    "ASX=AUD, TSE=JPY). Keep price collars within a sane band (0.1%-20%). Tickers are "
    "UPPERCASE alphanumeric."
)


def _build_user_prompt(state: dict[str, Any]) -> str:
    """Build the generator user prompt, including prior errors on a retry."""
    parts = [
        f"Original request: {state['request']}",
        f"Intent: {state.get('intent_summary', '')}",
        f"Rule type: {state['rule_type']}",
    ]
    if state.get("errors"):
        parts.append(
            "Your previous attempt FAILED validation with these errors — fix ALL of them:\n"
            + "\n".join(f"- {e}" for e in state["errors"])
        )
    return "\n".join(parts)


def generate_config(state: dict[str, Any]) -> dict[str, Any]:
    """Generate a draft rule config for the parsed rule_type via tool use."""
    model_cls = RULE_MODELS[state["rule_type"]]
    schema = model_cls.model_json_schema()
    draft = call_tool(
        system=_GENERATOR_SYSTEM,
        user=_build_user_prompt(state),
        tool_name="emit_rule",
        tool_description="Emit the rule config as structured JSON.",
        input_schema=schema,
    )
    return {"draft": draft}
