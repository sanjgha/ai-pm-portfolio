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
