"""Manual Anthropic tool-use loop used by the eval harness to simulate the MCP client."""

import json
from typing import Any, Callable

import anthropic

from execution_quality_copilot.config import get_anthropic_api_key

SYSTEM = (
    "You are a TCA (Transaction Cost Analysis) analyst for a portfolio manager. "
    "You have read-only tools over the fund's execution data. Use them to answer the question "
    "with specific numbers; do not guess. Brokers: ALPHA/BRAVO/COBALT/DELTA. Algos: VWAP/TWAP/IS. "
    "Benchmarks: arrival/vwap/close. Always finish your final message with a line formatted EXACTLY "
    "as 'ANSWER: <number>' where <number> is the single headline figure (bps, count, or price) with "
    "no commas, units, or extra text."
)

Dispatch = Callable[[str, dict[str, Any]], Any]


def get_client() -> anthropic.Anthropic:
    """Return an Anthropic client authenticated from the environment."""
    return anthropic.Anthropic(api_key=get_anthropic_api_key())


def run_tool_loop(
    client: anthropic.Anthropic,
    model: str,
    tools: list[dict[str, Any]],
    dispatch: Dispatch,
    question: str,
    max_turns: int = 6,
) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """Drive a tool-use loop; return (final_text, [(tool_name, tool_input), ...] in call order)."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    called: list[tuple[str, dict[str, Any]]] = []
    final_text = ""

    for _ in range(max_turns):
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM,
            tools=tools,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            final_text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            break

        messages.append({"role": "assistant", "content": resp.content})
        results: list[dict[str, Any]] = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            args = dict(block.input)
            called.append((block.name, args))
            try:
                out = dispatch(block.name, args)
                content, is_error = json.dumps(out, default=str), False
            except Exception as exc:  # surface tool errors back to the model
                content, is_error = f"error: {exc}", True
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})

    return final_text, called
