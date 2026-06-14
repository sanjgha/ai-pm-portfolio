"""Test eval scoring helpers (pure, no network)."""

from evals.run_eval import field_accuracy, schema_pass


def test_schema_pass_true_when_validated_rule_present():
    assert schema_pass({"status": "ok", "validated_rule": {"a": 1}}) is True
    assert schema_pass({"status": "escalated"}) is False


def test_field_accuracy_compares_nested_expected_fields():
    actual = {
        "rule_type": "order_notional_limit",
        "scope": {"exchange": "SGX", "symbols": []},
        "side": "buy",
        "max_notional": 5000000,
        "currency": "SGD",
    }
    expected = {
        "rule_type": "order_notional_limit",
        "scope": {"exchange": "SGX"},
        "side": "buy",
        "max_notional": 5000000,
        "currency": "SGD",
    }
    assert field_accuracy(actual, expected) == 1.0

    expected_wrong = dict(expected, currency="USD")
    assert field_accuracy(actual, expected_wrong) < 1.0
