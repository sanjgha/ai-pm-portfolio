"""Test the 5 domain lints (semantic checks beyond schema)."""

from pre_trade_risk_rules_assistant.lints import run_lints
from pre_trade_risk_rules_assistant.schemas.rules import RuleAdapter


def _rule(d):
    return RuleAdapter.validate_python(d)


def test_clean_order_rule_passes():
    rule = _rule(
        {
            "rule_type": "order_notional_limit",
            "description": "Block buy > 5m SGD on SGX",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "buy",
            "action": "block",
            "max_notional": 5000000,
            "currency": "SGD",
        }
    )
    assert run_lints(rule) == []


def test_currency_mismatch_flagged():
    rule = _rule(
        {
            "rule_type": "order_notional_limit",
            "description": "Block buy > 5m on SGX",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "buy",
            "action": "block",
            "max_notional": 5000000,
            "currency": "USD",
        }
    )
    errs = run_lints(rule)
    assert any("currency" in e.lower() for e in errs)


def test_notional_over_ceiling_flagged():
    rule = _rule(
        {
            "rule_type": "order_notional_limit",
            "description": "huge",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "buy",
            "action": "block",
            "max_notional": 5_000_000_000,
            "currency": "SGD",
        }
    )
    assert any("notional" in e.lower() for e in run_lints(rule))


def test_collar_out_of_band_flagged():
    rule = _rule(
        {
            "rule_type": "price_collar",
            "description": "wide collar",
            "scope": {"exchange": "HKEX", "symbols": []},
            "side": "both",
            "action": "warn",
            "collar_pct": 50,
            "reference_price": "last",
        }
    )
    assert any("collar" in e.lower() for e in run_lints(rule))


def test_malformed_restricted_symbols_flagged():
    rule = _rule(
        {
            "rule_type": "restricted_list",
            "description": "bad tickers",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "both",
            "action": "block",
            "symbols": ["d05", "D05"],
        }
    )
    errs = run_lints(rule)
    assert any("symbol" in e.lower() for e in errs)
