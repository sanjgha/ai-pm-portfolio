"""Test the validator node: Pydantic validation + domain lints."""

from pre_trade_risk_rules_assistant.nodes import validator


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


def test_valid_draft_sets_validated_rule_no_errors():
    out = validator.validate_config({"draft": _valid_collar()})
    assert out["errors"] == []
    assert out["validated_rule"]["collar_pct"] == 5


def test_schema_error_populates_errors():
    bad = _valid_collar()
    bad["collar_pct"] = 999
    out = validator.validate_config({"draft": bad})
    assert out["errors"]
    assert "validated_rule" not in out


def test_lint_error_populates_errors():
    bad = _valid_collar()
    bad["collar_pct"] = 50
    out = validator.validate_config({"draft": bad})
    assert any("collar" in e.lower() for e in out["errors"])
    assert "validated_rule" not in out
