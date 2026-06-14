"""Test the intent parser node (LLM mocked)."""

from pre_trade_risk_rules_assistant.nodes import parser


def test_parse_intent_sets_rule_type(monkeypatch):
    def fake_call_tool(**kwargs):
        return {"rule_type": "order_notional_limit", "intent_summary": "cap buy notional"}

    monkeypatch.setattr(parser, "call_tool", fake_call_tool)
    out = parser.parse_intent({"request": "block buy orders over 5m on SGX"})
    assert out["rule_type"] == "order_notional_limit"
    assert out["intent_summary"] == "cap buy notional"
