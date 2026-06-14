"""Thin wrappers around the Anthropic SDK: forced single-tool call + plain text call."""

from typing import Any

import anthropic

from pre_trade_risk_rules_assistant.config import get_anthropic_api_key, get_config


def get_client() -> anthropic.Anthropic:
    """Return an Anthropic client authenticated from the environment."""
    return anthropic.Anthropic(api_key=get_anthropic_api_key())


def call_tool(
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    """Force the model to emit ONE tool call; return its raw (unvalidated) input dict.

    Non-strict on purpose: the model CAN produce out-of-range / malformed input,
    which the validator node then catches and feeds back into the self-correct loop.
    """
    cfg = get_config()
    client = get_client()
    resp = client.messages.create(
        model=model or cfg["models"]["agent"],
        max_tokens=cfg["models"]["max_tokens"],
        system=system,
        tools=[{"name": tool_name, "description": tool_description, "input_schema": input_schema}],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return dict(block.input)
    raise ValueError("Model did not return a tool_use block")


def call_text(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Plain text completion (used for the read-back and the eval judge)."""
    cfg = get_config()
    client = get_client()
    resp = client.messages.create(
        model=model or cfg["models"]["agent"],
        max_tokens=max_tokens or 1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")
