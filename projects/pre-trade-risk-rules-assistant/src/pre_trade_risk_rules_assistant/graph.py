"""LangGraph state machine: parse -> generate -> validate -> {explain | correct | escalate}."""

from langgraph.graph import END, START, StateGraph

from pre_trade_risk_rules_assistant.nodes.corrector import (
    escalate,
    route_after_validation,
    self_correct,
)
from pre_trade_risk_rules_assistant.nodes.explainer import explain_rule
from pre_trade_risk_rules_assistant.nodes.generator import generate_config
from pre_trade_risk_rules_assistant.nodes.parser import parse_intent
from pre_trade_risk_rules_assistant.nodes.validator import validate_config
from pre_trade_risk_rules_assistant.state import RuleForgeState


def build_graph():
    """Build and compile the RuleForge graph."""
    g = StateGraph(RuleForgeState)
    g.add_node("parse", parse_intent)
    g.add_node("generate", generate_config)
    g.add_node("validate", validate_config)
    g.add_node("correct", self_correct)
    g.add_node("explain", explain_rule)
    g.add_node("escalate", escalate)

    g.add_edge(START, "parse")
    g.add_edge("parse", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges(
        "validate",
        route_after_validation,
        {"explain": "explain", "correct": "correct", "escalate": "escalate"},
    )
    g.add_edge("correct", "generate")  # the self-correction cycle
    g.add_edge("explain", END)
    g.add_edge("escalate", END)
    return g.compile()


_APP = None


def run_graph(request: str) -> RuleForgeState:
    """Run the agent on one NL request; return the final state."""
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP.invoke({"request": request, "attempts": 0})
