"""Intent parser node: extract rule_type + a normalized intent summary."""

from typing import Any

from pre_trade_risk_rules_assistant.llm import call_tool
from pre_trade_risk_rules_assistant.schemas.rules import RuleType

_PARSER_SYSTEM = (
    "You are a pre-trade risk analyst. Read a natural-language risk rule and classify it. "
    "Choose exactly one rule_type and write a one-sentence normalized summary of intent."
)

_PARSER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rule_type": {"type": "string", "enum": [t.value for t in RuleType]},
        "intent_summary": {"type": "string"},
    },
    "required": ["rule_type", "intent_summary"],
}


def parse_intent(state: dict[str, Any]) -> dict[str, Any]:
    """Classify the request into a rule_type and intent summary."""
    parsed = call_tool(
        system=_PARSER_SYSTEM,
        user=state["request"],
        tool_name="classify_rule",
        tool_description="Classify the rule type and summarize intent.",
        input_schema=_PARSER_SCHEMA,
    )
    return {
        "rule_type": parsed["rule_type"],
        "intent_summary": parsed.get("intent_summary", ""),
    }
