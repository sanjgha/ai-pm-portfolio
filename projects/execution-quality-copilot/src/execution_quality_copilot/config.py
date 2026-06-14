"""Configuration loading: YAML defaults + .env secrets."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Load and cache config.yaml."""
    global _config
    if _config is None:
        with open(CONFIG_PATH) as f:
            _config = yaml.safe_load(f)
    return _config


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key or raise if unset."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return key
