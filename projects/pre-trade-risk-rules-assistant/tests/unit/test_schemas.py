"""Test Pydantic rule schemas and the discriminated union."""

import pytest
from pydantic import ValidationError

from pre_trade_risk_rules_assistant.schemas.rules import (
    OrderNotionalLimitRule,
    PriceCollarRule,
    RestrictedListRule,
    RuleAdapter,
    RuleType,
)


def _order_dict():
    return {
        "rule_type": "order_notional_limit",
        "description": "Block buy orders over 5m SGD on SGX",
        "scope": {"exchange": "SGX", "symbols": []},
        "side": "buy",
        "action": "block",
        "max_notional": 5000000,
        "currency": "SGD",
    }


def test_adapter_picks_order_notional_subtype():
    rule = RuleAdapter.validate_python(_order_dict())
    assert isinstance(rule, OrderNotionalLimitRule)
    assert rule.rule_type == RuleType.ORDER_NOTIONAL_LIMIT
    assert rule.max_notional == 5000000


def test_negative_notional_rejected():
    bad = _order_dict()
    bad["max_notional"] = -1
    with pytest.raises(ValidationError):
        RuleAdapter.validate_python(bad)


def test_collar_over_100_rejected():
    bad = {
        "rule_type": "price_collar",
        "description": "Collar",
        "scope": {"exchange": "HKEX", "symbols": []},
        "side": "both",
        "action": "warn",
        "collar_pct": 150,
        "reference_price": "last",
    }
    with pytest.raises(ValidationError):
        RuleAdapter.validate_python(bad)


def test_restricted_list_requires_symbols():
    bad = {
        "rule_type": "restricted_list",
        "description": "Restricted",
        "scope": {"exchange": "ASX", "symbols": []},
        "side": "both",
        "action": "block",
        "symbols": [],
    }
    with pytest.raises(ValidationError):
        RuleAdapter.validate_python(bad)


def test_price_collar_and_restricted_subtypes():
    collar = RuleAdapter.validate_python(
        {
            "rule_type": "price_collar",
            "description": "5% collar on SGX",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "both",
            "action": "warn",
            "collar_pct": 5,
            "reference_price": "last",
        }
    )
    assert isinstance(collar, PriceCollarRule)
    restricted = RuleAdapter.validate_python(
        {
            "rule_type": "restricted_list",
            "description": "No DBS",
            "scope": {"exchange": "SGX", "symbols": []},
            "side": "both",
            "action": "block",
            "symbols": ["D05"],
        }
    )
    assert isinstance(restricted, RestrictedListRule)


def test_unknown_exchange_rejected():
    bad = _order_dict()
    bad["scope"]["exchange"] = "NASDAQ"
    with pytest.raises(ValidationError):
        RuleAdapter.validate_python(bad)
