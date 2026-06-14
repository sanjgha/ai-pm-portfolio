"""Tests for run_eval CLI overrides and per-question CP table (LIN-137)."""

import sys
from pathlib import Path
from types import SimpleNamespace


def _import_run_eval():
    sys.path.insert(0, str(Path(__file__).parents[2] / "evals"))
    import run_eval

    return run_eval


def test_build_overrides_maps_set_args_only():
    """Only args the user set appear in the overrides dict."""
    run_eval = _import_run_eval()
    args = SimpleNamespace(
        rerank_top_n=8,
        top_k=None,
        bm25_weight=0.3,
        vector_weight=0.7,
        rerank_query_expansion=True,
    )
    overrides = run_eval.build_overrides(args)
    assert overrides == {
        "retrieval": {
            "rerank_top_n": 8,
            "bm25_weight": 0.3,
            "vector_weight": 0.7,
            "rerank_query_expansion": True,
        }
    }


def test_build_overrides_empty_when_nothing_set():
    """No flags set -> empty overrides (run uses config defaults)."""
    run_eval = _import_run_eval()
    args = SimpleNamespace(
        rerank_top_n=None,
        top_k=None,
        bm25_weight=None,
        vector_weight=None,
        rerank_query_expansion=None,
    )
    assert run_eval.build_overrides(args) == {}


def test_build_per_question_cp_pairs_questions_and_scores():
    """Per-question CP table pairs user_input with context_precision."""
    run_eval = _import_run_eval()
    scores = {
        "user_input": ["Q1", "Q2"],
        "context_precision": [0.2, 0.95],
    }
    table = run_eval.build_per_question_cp(scores)
    assert table == [
        {"question": "Q1", "context_precision": 0.2},
        {"question": "Q2", "context_precision": 0.95},
    ]
