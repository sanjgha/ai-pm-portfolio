"""End-to-end graph tests with all LLM nodes mocked (covers the 3 paths)."""

from pre_trade_risk_rules_assistant import graph
from pre_trade_risk_rules_assistant.nodes import explainer, generator, parser


def _mock_parser(monkeypatch, rule_type="price_collar"):
    monkeypatch.setattr(
        parser, "call_tool", lambda **k: {"rule_type": rule_type, "intent_summary": "c"}
    )


def _valid_collar():
    return {
        "rule_type": "price_collar",
        "description": "5% collar on SGX",
        "scope": {"exchange": "SGX", "symbols": []},
        "side": "both",
        "action": "warn",
        "collar_pct": 5,
        "reference_price": "last",
    }


def test_happy_path_first_try(monkeypatch):
    _mock_parser(monkeypatch)
    monkeypatch.setattr(generator, "call_tool", lambda **k: _valid_collar())
    monkeypatch.setattr(explainer, "call_text", lambda **k: "readback")
    final = graph.run_graph("5% collar on SGX")
    assert final["status"] == "ok"
    assert final["attempts"] == 0
    assert final["validated_rule"]["collar_pct"] == 5
    assert final["readback"] == "readback"


def test_self_corrects_then_succeeds(monkeypatch):
    _mock_parser(monkeypatch)
    calls = {"n": 0}

    def flaky(**k):
        calls["n"] += 1
        if calls["n"] == 1:
            bad = _valid_collar()
            bad["collar_pct"] = 50  # fails lint -> triggers correction
            return bad
        return _valid_collar()

    monkeypatch.setattr(generator, "call_tool", flaky)
    monkeypatch.setattr(explainer, "call_text", lambda **k: "readback")
    final = graph.run_graph("collar")
    assert final["status"] == "ok"
    assert final["attempts"] == 1
    assert calls["n"] == 2


def test_escalates_after_two_failures(monkeypatch):
    _mock_parser(monkeypatch)

    def always_bad(**k):
        bad = _valid_collar()
        bad["collar_pct"] = 50
        return bad

    monkeypatch.setattr(generator, "call_tool", always_bad)
    final = graph.run_graph("collar")
    assert final["status"] == "escalated"
    assert final["attempts"] == 2
    assert "collar" in final["escalation_reason"].lower()
