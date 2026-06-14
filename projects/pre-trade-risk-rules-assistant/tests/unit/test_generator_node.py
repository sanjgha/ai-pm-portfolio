"""Test the config generator node (LLM mocked)."""

from pre_trade_risk_rules_assistant.nodes import generator


def test_generate_uses_rule_type_schema_and_returns_draft(monkeypatch):
    captured = {}

    def fake_call_tool(**kwargs):
        captured.update(kwargs)
        return {"rule_type": "price_collar", "collar_pct": 5}

    monkeypatch.setattr(generator, "call_tool", fake_call_tool)
    out = generator.generate_config(
        {"request": "5% collar on SGX", "rule_type": "price_collar", "intent_summary": "collar"}
    )
    assert out["draft"] == {"rule_type": "price_collar", "collar_pct": 5}
    assert "collar_pct" in captured["input_schema"]["properties"]


def test_generate_includes_prior_errors_in_prompt(monkeypatch):
    captured = {}

    def fake_call_tool(**kwargs):
        captured.update(kwargs)
        return {"rule_type": "price_collar"}

    monkeypatch.setattr(generator, "call_tool", fake_call_tool)
    generator.generate_config(
        {
            "request": "collar",
            "rule_type": "price_collar",
            "intent_summary": "c",
            "errors": ["collar_pct 50 outside sane band [0.1, 20.0]"],
        }
    )
    assert "outside sane band" in captured["user"]
