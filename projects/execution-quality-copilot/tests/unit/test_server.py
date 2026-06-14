"""Smoke-test that the FastMCP server registers all five tools with descriptions."""

import asyncio

from execution_quality_copilot.server import main


def test_five_tools_registered():
    tools = asyncio.run(main.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "get_fills",
        "get_benchmarks",
        "compute_slippage",
        "venue_breakdown",
        "top_outliers",
    }


def test_tool_descriptions_are_substantial():
    tools = asyncio.run(main.mcp.list_tools())
    for t in tools:
        assert t.description and len(t.description) > 40
