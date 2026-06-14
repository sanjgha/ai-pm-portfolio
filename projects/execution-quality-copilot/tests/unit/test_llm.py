"""Test the manual tool-use loop with a fake Anthropic client (no network)."""

from execution_quality_copilot import llm


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_loop_executes_tool_then_returns_answer():
    # Turn 1: model calls compute_slippage. Turn 2: model gives the final answer.
    responses = [
        _Resp(
            "tool_use",
            [_Block(type="tool_use", id="t1", name="compute_slippage", input={"broker": "DELTA"})],
        ),
        _Resp("end_turn", [_Block(type="text", text="DELTA cost ~11 bps.\nANSWER: 11.0")]),
    ]
    client = _FakeClient(responses)

    calls = []

    def dispatch(name, args):
        calls.append((name, args))
        return {"slippage_bps": 11.0}

    tools = [{"name": "compute_slippage", "description": "x", "input_schema": {"type": "object"}}]
    text, called = llm.run_tool_loop(client, "claude-opus-4-8", tools, dispatch, "rank brokers")

    assert calls == [("compute_slippage", {"broker": "DELTA"})]
    assert called == [("compute_slippage", {"broker": "DELTA"})]
    assert "ANSWER: 11.0" in text


def test_loop_stops_at_max_turns():
    # Model always calls a tool → loop must bail at max_turns without infinite looping.
    looping = _Resp("tool_use", [_Block(type="tool_use", id="t", name="get_fills", input={})])
    client = _FakeClient([looping] * 10)
    text, called = llm.run_tool_loop(
        client, "claude-opus-4-8", [], lambda n, a: {"ok": True}, "q", max_turns=3
    )
    assert len(called) == 3
    assert text == ""
