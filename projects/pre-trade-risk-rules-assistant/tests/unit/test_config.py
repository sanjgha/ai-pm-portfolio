"""Test config loading."""

import os

import pytest

from pre_trade_risk_rules_assistant.config import get_anthropic_api_key, get_config


def test_get_config_has_expected_sections():
    config = get_config()
    for section in ("models", "agent", "exchanges", "exchange_currency", "lints", "storage"):
        assert section in config


def test_model_names_configured():
    config = get_config()
    assert config["models"]["agent"] == "claude-sonnet-4-6"
    assert config["models"]["judge"] == "claude-haiku-4-5"


def test_lint_thresholds():
    config = get_config()
    assert config["lints"]["max_collar_pct"] == 20.0
    assert config["agent"]["max_correction_attempts"] == 2


def test_anthropic_key_required():
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    assert get_anthropic_api_key() == "test-key"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_anthropic_api_key()
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
