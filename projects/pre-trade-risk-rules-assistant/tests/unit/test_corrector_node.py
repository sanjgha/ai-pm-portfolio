"""Test the self-correct node, escalate node, and the post-validation router."""

from pre_trade_risk_rules_assistant.nodes import corrector


def test_self_correct_increments_attempts():
    assert corrector.self_correct({"attempts": 0})["attempts"] == 1
    assert corrector.self_correct({"attempts": 1})["attempts"] == 2
    assert corrector.self_correct({})["attempts"] == 1


def test_escalate_sets_status_and_reason():
    out = corrector.escalate({"errors": ["a", "b"]})
    assert out["status"] == "escalated"
    assert "a" in out["escalation_reason"]


def test_router_no_errors_goes_to_explain():
    assert corrector.route_after_validation({"errors": [], "attempts": 0}) == "explain"


def test_router_with_budget_left_corrects():
    assert corrector.route_after_validation({"errors": ["x"], "attempts": 0}) == "correct"
    assert corrector.route_after_validation({"errors": ["x"], "attempts": 1}) == "correct"


def test_router_exhausted_escalates():
    assert corrector.route_after_validation({"errors": ["x"], "attempts": 2}) == "escalate"
