"""Test the read-back explainer node (LLM mocked)."""

from pre_trade_risk_rules_assistant.nodes import explainer


def test_explain_sets_status_ok_and_readback(monkeypatch):
    monkeypatch.setattr(
        explainer, "call_text", lambda **kwargs: "This rule blocks buy orders over 5m SGD on SGX."
    )
    out = explainer.explain_rule({"validated_rule": {"rule_type": "order_notional_limit"}})
    assert out["status"] == "ok"
    assert "blocks buy orders" in out["readback"]
