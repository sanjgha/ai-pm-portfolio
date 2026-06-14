"""Test config loading and the API-key accessor."""

import pytest

from execution_quality_copilot import config


def test_get_config_has_models_and_storage():
    cfg = config.get_config()
    assert cfg["models"]["agent"] == "claude-opus-4-8"
    assert cfg["storage"]["db_path"].endswith("seed.duckdb")
    assert cfg["generator"]["n_fills"] == 10000


def test_get_anthropic_api_key_raises_when_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        config.get_anthropic_api_key()


def test_get_anthropic_api_key_returns_value(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert config.get_anthropic_api_key() == "sk-test"
