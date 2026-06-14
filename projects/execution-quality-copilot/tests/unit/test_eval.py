"""Test the eval scoring functions (no API calls)."""

from evals import run_eval


def test_tool_selection_hit_and_miss():
    called = [("get_fills", {}), ("compute_slippage", {"broker": "DELTA"})]
    assert run_eval.tool_selected("compute_slippage", called) is True
    assert run_eval.tool_selected("top_outliers", called) is False


def test_param_match_requires_subset_on_the_right_tool():
    called = [("compute_slippage", {"broker": "DELTA", "benchmark": "arrival"})]
    assert run_eval.params_matched("compute_slippage", {"broker": "DELTA"}, called) is True
    # Right params but wrong tool → no match.
    assert run_eval.params_matched("top_outliers", {"broker": "DELTA"}, called) is False
    # Missing/incorrect value → no match.
    assert run_eval.params_matched("compute_slippage", {"broker": "ALPHA"}, called) is False


def test_numeric_accuracy_within_tolerance():
    assert run_eval.numeric_ok(10.0, 10.3, tol=0.05) is True
    assert run_eval.numeric_ok(10.0, 12.0, tol=0.05) is False
    assert run_eval.numeric_ok(None, 5.0, tol=0.05) is False


def test_parse_answer_extracts_the_answer_line():
    assert run_eval.parse_answer("Some text.\nANSWER: -3.5") == -3.5
    assert run_eval.parse_answer("ANSWER: 1234.56 bps") == 1234.56
    assert run_eval.parse_answer("no answer line here") is None
