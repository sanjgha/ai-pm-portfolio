"""Unit tests for eval exchange filter logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


GOLDEN_DATASET = {
    "answerable": [
        {
            "question": "What is the HKSE lot size?",
            "exchange": "HKSE",
            "ground_truth": "100 shares",
        },
        {
            "question": "What are HKSE trading hours?",
            "exchange": "HKSE",
            "ground_truth": "9:30-16:00",
        },
        {"question": "What is the SGX lot size?", "exchange": "SGX", "ground_truth": "100 shares"},
        {"question": "What is the TSE lot size?", "exchange": "TSE", "ground_truth": "100 shares"},
    ],
    "unanswerable": [
        {"question": "What is the IDX lot size?"},
    ],
}


@pytest.fixture()
def mock_evals_dir(tmp_path):
    import json

    golden = tmp_path / "golden_dataset.json"
    golden.write_text(json.dumps(GOLDEN_DATASET))
    return tmp_path


def test_load_golden_dataset_returns_all_by_default(mock_evals_dir):
    """Without a filter, all answerable questions are returned."""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[2] / "evals"))
    from run_eval import load_golden_dataset

    answerable, unanswerable = load_golden_dataset(mock_evals_dir / "golden_dataset.json")
    assert len(answerable) == 4
    assert len(unanswerable) == 1


def test_run_evaluation_filters_by_exchange(mock_evals_dir):
    """run_evaluation with exchanges=['HKSE'] processes only HKSE questions."""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[2] / "evals"))

    # Minimal stub: track which questions were processed
    processed = []

    def fake_chain_invoke(inputs):
        processed.append(inputs["question"])
        return {"answer": "test answer", "contexts": ["chunk text"], "sources": []}

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = fake_chain_invoke

    fake_scores = {
        "user_input": ["Q1", "Q2"],
        "faithfulness": [1.0, 1.0],
        "context_precision": [1.0, 1.0],
        "context_recall": [1.0, 1.0],
        "answer_relevancy": [1.0, 1.0],
    }

    import pandas as pd

    mock_ragas_result = MagicMock()
    mock_ragas_result.to_pandas.return_value = pd.DataFrame(fake_scores)

    with (
        patch("run_eval.create_rag_chain", return_value=mock_chain),
        patch("run_eval.evaluate", return_value=mock_ragas_result),
        patch("run_eval.LangchainLLMWrapper"),
        patch("run_eval.LangchainEmbeddingsWrapper"),
        patch("run_eval.ChatAnthropic"),
        patch("run_eval.VoyageEmbeddings"),
        patch("run_eval.get_anthropic_api_key", return_value="test"),
        patch("run_eval.get_voyage_api_key", return_value="test"),
    ):
        from run_eval import run_evaluation

        results = run_evaluation(mock_evals_dir, exchanges=["HKSE"])

    # Only HKSE questions should have been processed
    hkex_questions = [
        q["question"] for q in GOLDEN_DATASET["answerable"] if q["exchange"] == "HKSE"
    ]
    for q in hkex_questions:
        assert q in processed, f"Expected HKSE question to be processed: {q}"

    sgx_questions = [q["question"] for q in GOLDEN_DATASET["answerable"] if q["exchange"] == "SGX"]
    for q in sgx_questions:
        assert q not in processed, f"SGX question should not be processed: {q}"

    assert results["answerable_count"] == 2, "Should report 2 HKSE answerable questions"
