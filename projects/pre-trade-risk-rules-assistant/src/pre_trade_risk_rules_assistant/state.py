"""LangGraph shared state for the RuleForge agent."""

from typing import Any, TypedDict


class RuleForgeState(TypedDict, total=False):
    """Typed state threaded through the RuleForge LangGraph nodes."""

    request: str  # original NL rule description
    rule_type: str  # set by parser
    intent_summary: str  # set by parser
    draft: dict[str, Any]  # raw config from generator (unvalidated)
    errors: list[str]  # validation + lint errors (set by validator)
    attempts: int  # self-correction attempts so far
    validated_rule: dict[str, Any]  # validated config (set on success)
    readback: str  # plain-English explanation (set by explainer)
    status: str  # "ok" | "escalated"
    escalation_reason: str
