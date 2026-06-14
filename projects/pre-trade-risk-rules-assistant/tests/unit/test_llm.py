"""Test the Anthropic call helpers (with a fake client, no network)."""

from types import SimpleNamespace

from pre_trade_risk_rules_assistant import llm


class _FakeMessages:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        self._last = kwargs
        return SimpleNamespace(content=self._content)


def test_call_tool_returns_tool_input(monkeypatch):
    tool_block = SimpleNamespace(type="tool_use", input={"rule_type": "price_collar"})
    fake = SimpleNamespace(messages=_FakeMessages([tool_block]))
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    out = llm.call_tool(
        system="s", user="u", tool_name="t", tool_description="d", input_schema={"type": "object"}
    )
    assert out == {"rule_type": "price_collar"}


def test_call_tool_raises_without_tool_use(monkeypatch):
    text_block = SimpleNamespace(type="text", text="no tool")
    fake = SimpleNamespace(messages=_FakeMessages([text_block]))
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    try:
        llm.call_tool(system="s", user="u", tool_name="t", tool_description="d", input_schema={})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_call_text_concatenates_text_blocks(monkeypatch):
    blocks = [
        SimpleNamespace(type="text", text="Hello "),
        SimpleNamespace(type="text", text="world"),
    ]
    fake = SimpleNamespace(messages=_FakeMessages(blocks))
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    assert llm.call_text(system="s", user="u") == "Hello world"
