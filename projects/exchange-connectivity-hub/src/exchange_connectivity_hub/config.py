"""Configuration management for exchange connectivity hub."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# Path to config.yaml (go up from src/exchange_connectivity_hub/config.py to project root)
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _load_config() -> dict[str, Any]:
    """Load configuration from config.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Global config cache
_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Get configuration, loading once and caching."""
    global _config
    if _config is None:
        _config = _load_config()
    return _config


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Recursively merge ``overrides`` into ``base`` in place."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def apply_config_overrides(overrides: dict[str, Any]) -> None:
    """Deep-merge ``overrides`` into the cached config.

    Used by tuning entry points (e.g. ``run_eval.py`` CLI flags) to adjust
    ranking parameters for a single process run without editing config.yaml.
    Because every module reads the same cached dict via ``get_config()``, the
    merge propagates everywhere downstream.
    """
    config = get_config()
    _deep_merge(config, overrides)


def get_voyage_api_key() -> str:
    """Get Voyage API key from environment."""
    key = os.getenv("VOYAGE_API_KEY")
    if not key:
        raise ValueError("VOYAGE_API_KEY environment variable not set")
    return key


def get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return key
